"""Operator pruning of episode records in a global harness store."""

from __future__ import annotations

import json

import pytest

from rlm.cli import main
from rlm.harness import HarnessStore, episode_path
from rlm.harness_ops import (
    PRUNE_TRIGGER,
    parse_duration,
    prune_episodes,
    select_prunable,
)

NOW = 1_800_000_000.0
DAY = 86_400.0


def _store_with_episodes(directory, ages_in_days):
    store = HarnessStore(directory, scope="global")
    for days in ages_in_days:
        started = NOW - days * DAY
        session_id = f"s{days}"
        store.upsert(
            "episode",
            f"task from {days} days ago",
            "Outcome (done): ok",
            id=f"episode-{session_id}",
            path=episode_path(started, session_id),
            metadata={
                "session_dir": f"/tmp/{session_id}",
                "session_id": session_id,
                "stop_reason": "done",
                "prompts": ["task"],
                "turns": 1,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cwd": "/repo",
                "blocks": [],
                "started_at": started,
                "ended_at": started + 60,
            },
            source="engine",
        )
    return store


def test_parse_duration():
    assert parse_duration("30d") == 30 * DAY
    assert parse_duration("12h") == 12 * 3600
    assert parse_duration(" 4w ") == 4 * 7 * DAY
    with pytest.raises(ValueError, match="30d, 12h or 4w"):
        parse_duration("soon")


def test_select_prunable_by_age_and_count(tmp_path):
    store = _store_with_episodes(tmp_path / "g", [1, 10, 40, 100])

    assert select_prunable(store, now=NOW) == []
    old = select_prunable(store, older_than=30 * DAY, now=NOW)
    assert [e.id for e in old] == ["episode-s100", "episode-s40"]
    newest_three = select_prunable(store, keep=3, now=NOW)
    assert [e.id for e in newest_three] == ["episode-s100"]
    both = select_prunable(store, older_than=5 * DAY, keep=1, now=NOW)
    assert [e.id for e in both] == ["episode-s100", "episode-s40", "episode-s10"]
    assert select_prunable(store, keep=10, now=NOW) == []


def test_prune_removes_entries_and_records_the_operation(tmp_path):
    store = _store_with_episodes(tmp_path / "g", [1, 40])
    (old,) = select_prunable(store, older_than=30 * DAY, now=NOW)

    result = prune_episodes(store, [old])

    reloaded = HarnessStore(tmp_path / "g", scope="global").load()
    assert [e.id for e in reloaded.list("episode")] == ["episode-s1"]
    (event,) = reloaded.list_refinements()
    assert event.id == result.id and event.trigger == PRUNE_TRIGGER
    assert event.changes == ["delete episode:episode-s40"]
    (record,) = reloaded.results()
    assert record["trigger"] == PRUNE_TRIGGER
    (edit,) = record["applied_edits"]
    assert edit["applied"] is True
    assert edit["before"]["title"] == "task from 40 days ago"
    assert edit["before"]["metadata"]["session_dir"] == "/tmp/s40"


def test_cli_dry_run_then_apply(tmp_path, capsys, monkeypatch):
    global_dir = tmp_path / "g"
    _store_with_episodes(global_dir, [1, 40])
    monkeypatch.setattr("rlm.harness_ops.time.time", lambda: NOW)

    assert (
        main(["harness", "prune", "--global-dir", str(global_dir), "--keep", "1"]) == 0
    )
    out = capsys.readouterr().out
    assert "Would remove 1 of 2 episode(s)" in out
    assert "episode-s40" in out and "Dry run" in out
    assert HarnessStore(global_dir, scope="global").load().count("episode") == 2

    assert (
        main(
            [
                "harness",
                "prune",
                "--global-dir",
                str(global_dir),
                "--keep",
                "1",
                "--apply",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Removing 1 of 2 episode(s)" in out and "1 episode(s) remain" in out
    remaining = HarnessStore(global_dir, scope="global").load()
    assert [e.id for e in remaining.list("episode")] == ["episode-s1"]
    state = json.loads((global_dir / "harness_state.json").read_text())
    assert state["refinements"][0]["trigger"] == PRUNE_TRIGGER

    with pytest.raises(SystemExit):
        main(["harness", "prune", "--global-dir", str(global_dir)])
    monkeypatch.delenv("RLM_HARNESS_GLOBAL_DIR", raising=False)
    with pytest.raises(SystemExit):
        main(["harness", "prune", "--keep", "1"])
