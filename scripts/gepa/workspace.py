"""Pinned checkouts of repositories to ask questions about.

    uv run python scripts/gepa/workspace.py --dir scripts/gepa/workspace

prints one ``NAME=PATH`` line per checkout, ready for ``tasks.py --repo``. Sessions run
with a checkout as their working directory and may write into it, so even this
repository is asked about through a pinned copy rather than the live tree.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PINNED = {
    "nano-rlm": (
        "https://github.com/mannuch/nano-rlm",
        "4196d09e1dd9d7c3214f94141ee69358ab4b93fb",
    ),
    "itsdangerous": ("https://github.com/pallets/itsdangerous", "2.2.0"),
    "click": ("https://github.com/pallets/click", "8.1.8"),
    "attrs": ("https://github.com/python-attrs/attrs", "24.2.0"),
}


def checkout(workspace: Path, name: str, url: str, ref: str) -> Path:
    """A shallow checkout of ``ref`` (tag, branch or commit) under ``workspace``."""
    target = workspace / f"{name}@{ref[:12]}"
    if target.exists():
        return target
    target.mkdir(parents=True)
    git = ["git", "-c", "advice.detachedHead=false", "-C", str(target)]
    subprocess.run([*git, "init", "--quiet"], check=True)
    subprocess.run([*git, "fetch", "--quiet", "--depth", "1", url, ref], check=True)
    subprocess.run([*git, "checkout", "--quiet", "FETCH_HEAD"], check=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="scripts/gepa/workspace")
    parser.add_argument("--only", action="append", help="Restrict to these names")
    args = parser.parse_args(argv)
    workspace = Path(args.dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    for name, (url, ref) in PINNED.items():
        if args.only and name not in args.only:
            continue
        print(f"{name}={checkout(workspace, name, url, ref)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
