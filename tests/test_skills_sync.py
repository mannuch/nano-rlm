"""Tests for mirroring the skills root into a split-execution worker and back."""

from __future__ import annotations

import pytest

from rlm.skills_sync import apply, collect, fold, scan


def test_mirror_round_trips_changes_without_echo(tmp_path):
    source, mirror = tmp_path / "source", tmp_path / "mirror"
    package = source / "greet" / "src" / "greet"
    (package / "__pycache__").mkdir(parents=True)
    (package / "__init__.py").write_text("x = 1\n")
    (package / "__pycache__" / "greet.cpython-312.pyc").write_bytes(b"\0")
    (source / "greet" / "SKILL.md").write_text("# greet\n")
    (source / "notes.txt").write_text("not inside a skill folder")
    (source / "linked").symlink_to(tmp_path, target_is_directory=True)

    batch, seen = collect(source, {})
    mirror_seen: dict = {}
    fold(mirror_seen, *apply(mirror, batch))

    assert sorted(scan(mirror)) == ["greet/SKILL.md", "greet/src/greet/__init__.py"]
    assert (mirror / "greet" / "src" / "greet" / "__init__.py").read_text() == "x = 1\n"
    assert collect(source, seen) == (None, seen)
    # What the mirror received is not sent back.
    assert collect(mirror, mirror_seen)[0] is None

    (mirror / "greet" / "SKILL.md").write_text("# greet, edited\n")
    batch, mirror_seen = collect(mirror, mirror_seen)
    fold(seen, *apply(source, batch))
    assert (source / "greet" / "SKILL.md").read_text() == "# greet, edited\n"
    assert collect(source, seen)[0] is None

    (source / "greet" / "src" / "greet" / "__init__.py").unlink()
    batch, seen = collect(source, seen)
    assert batch == {"files": {}, "deleted": ["greet/src/greet/__init__.py"]}
    apply(mirror, batch)
    assert not (mirror / "greet" / "src").exists()
    assert (mirror / "greet" / "SKILL.md").exists()


def test_mirror_batches_by_size_and_never_sends_oversized_files(tmp_path):
    root = tmp_path / "root"
    (root / "skill").mkdir(parents=True)
    for name in ("a", "b", "c"):
        (root / "skill" / name).write_bytes(b"x" * 40)
    (root / "skill" / "big").write_bytes(b"x" * 200)

    batches = []
    seen: dict = {}
    while (result := collect(root, seen, budget=100))[0] is not None:
        batch, seen = result
        batches.append(sorted(batch["files"]))

    assert [len(files) for files in batches] == [2, 1]
    assert sorted(sum(batches, [])) == ["skill/a", "skill/b", "skill/c"]
    assert "skill/big" in seen


@pytest.mark.parametrize(
    "path", ["../escape/x", "/abs/x", "notes.txt", "skill/../../x", "skill/x.pyc"]
)
def test_apply_rejects_paths_outside_skill_folders(tmp_path, path):
    with pytest.raises(ValueError, match="invalid skill path"):
        apply(tmp_path, {"files": {path: ""}, "deleted": []})


def test_apply_rejects_paths_through_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "evil").symlink_to(tmp_path / "outside", target_is_directory=True)
    for batch in (
        {"files": {"evil/x": ""}, "deleted": []},
        {"files": {}, "deleted": ["evil/x"]},
    ):
        with pytest.raises(ValueError, match="crosses a symlink"):
            apply(root, batch)
    assert not (tmp_path / "outside").exists()
