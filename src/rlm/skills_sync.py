"""Mirror the skills root between the harness and a split-execution worker.

The harness keeps the persistent skills root; a worker holds a copy its kernel reads
and imports. Each side scans its tree against what it last exchanged and sends only
what changed, so a package written on either side reaches the other before the next
model call. The receiving side validates every path: a batch never writes outside the
skill folders of its root, and neither side imports what it receives.
"""

from __future__ import annotations

import base64
import logging
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

WORKER_DIR_NAME = "skills"
"""The mirror's directory inside a worker's session directory."""

SYNC_BATCH_BYTES = 4 * 1024 * 1024
"""Raw content bytes in one batch, and the largest file mirrored; base64 keeps a batch
well under the execution wire's frame."""

_TEMP_SUFFIX = ".rlm-sync.tmp"

Manifest = dict[str, tuple[int, int]]
"""``relative path -> (mtime_ns, size)`` of every mirrored file."""


def _skipped(name: str) -> bool:
    return name == "__pycache__" or name.endswith((".pyc", _TEMP_SUFFIX))


def scan(root: Path) -> Manifest:
    """The regular files inside the skill folders under ``root``. Symlinks, bytecode
    and files directly in ``root`` are not mirrored."""
    manifest: Manifest = {}
    if not root.is_dir():
        return manifest
    for directory, dirnames, filenames in os.walk(root):
        base = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if not _skipped(name) and not (base / name).is_symlink()
        ]
        if base == root:
            continue
        for name in filenames:
            path = base / name
            if _skipped(name) or path.is_symlink():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # removed mid-scan; the next scan reports the deletion
            relative = path.relative_to(root).as_posix()
            manifest[relative] = (stat.st_mtime_ns, stat.st_size)
    return manifest


def collect(
    root: Path, seen: Manifest, budget: int = SYNC_BATCH_BYTES
) -> tuple[dict[str, Any] | None, Manifest]:
    """What changed under ``root`` since ``seen``, as one batch of at most ``budget``
    content bytes (None when nothing changed), and the manifest to compare against
    next time. A file that doesn't fit this batch stays out of that manifest, so the
    next call sends it; a file larger than ``budget`` is never sent."""
    current = scan(root)
    after = dict(seen)
    deleted = [relative for relative in seen if relative not in current]
    for relative in deleted:
        del after[relative]
    files: dict[str, str] = {}
    size = 0
    for relative, stat in current.items():
        if seen.get(relative) == stat:
            continue
        if stat[1] > budget:
            logger.warning(
                "rlm: skill file %s exceeds %d bytes and is not mirrored",
                relative,
                budget,
            )
            after[relative] = stat
            continue
        if size + stat[1] > budget:
            continue
        try:
            data = (root / relative).read_bytes()
        except FileNotFoundError:
            continue
        files[relative] = base64.b64encode(data).decode()
        size += len(data)
        after[relative] = stat
    if not files and not deleted:
        return None, after
    return {"files": files, "deleted": deleted}, after


def _target(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or any(part in (".", "..") or _skipped(part) for part in path.parts)
    ):
        raise ValueError(f"invalid skill path {relative!r}")
    target = root.joinpath(*path.parts)
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"skill path {relative!r} crosses a symlink")
    return target


def _prune(root: Path, directory: Path) -> None:
    while directory != root and directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
        directory = directory.parent


def apply(root: Path, batch: dict[str, Any]) -> tuple[Manifest, list[str]]:
    """Write a batch from the other side into ``root``. Returns the written files'
    manifest entries and the removed paths, for the receiver to fold into what it has
    seen so the changes don't echo back."""
    files, deleted = batch.get("files"), batch.get("deleted")
    if not isinstance(files, dict) or not isinstance(deleted, list):
        raise ValueError("invalid skills batch")
    written: Manifest = {}
    for relative, encoded in files.items():
        target = _target(root, relative)
        data = base64.b64decode(encoded, validate=True)
        if len(data) > SYNC_BATCH_BYTES:
            raise ValueError(
                f"skill file {relative!r} exceeds {SYNC_BATCH_BYTES} bytes"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}{_TEMP_SUFFIX}")
        try:
            temp.write_bytes(data)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        stat = target.stat()
        written[relative] = (stat.st_mtime_ns, stat.st_size)
    for relative in deleted:
        target = _target(root, relative)
        target.unlink(missing_ok=True)
        _prune(root, target.parent)
    return written, list(deleted)


def fold(seen: Manifest, written: Manifest, removed: list[str]) -> None:
    """Record a batch applied from the other side as already seen."""
    seen.update(written)
    for relative in removed:
        seen.pop(relative, None)
