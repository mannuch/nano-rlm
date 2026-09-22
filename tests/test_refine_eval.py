"""Pure parts of the refinement eval: edit checks, probe scoring, judge grading and
the report. No sessions, no network."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "refine_eval"))

from grade import check_edit, grade_case, probe_score, report, sweep  # noqa: E402
from scenarios import BY_ID, AtMostOne, Changed, Created, NoCreate  # noqa: E402


def _edit(action, kind, id, content="", applied=True):
    return {
        "action": action,
        "kind": kind,
        "id": id,
        "applied": applied,
        "title": id,
        "content": content,
    }


def _ledger(steer, review, edits):
    records = [
        {"type": "user", "content": text, "message_index": 10 * i}
        for i, text in enumerate(steer)
    ]
    records.append({"type": "refinement_review", "reason": "host", **review})
    if edits:
        records.append({"type": "refinement", "result": {"applied_edits": edits}})
    return records


def test_edit_checks_read_applied_edits_and_the_final_store():
    edits = [
        _edit("create", "memory", "counts", "exclude blank lines"),
        _edit("update", "memory", "test-command"),
    ]
    assert check_edit(Created(("memory",), "blank"), edits, [])
    assert not check_edit(Created(("prompt",), "blank"), edits, [])
    assert check_edit(Changed("memory", "test-command"), edits, [])
    assert not check_edit(NoCreate(), edits, [])
    assert check_edit(NoCreate("pytest"), edits, [])
    twice = [{"title": "a", "content": "uv run pytest"}] * 2
    assert not check_edit(AtMostOne("uv run pytest"), [], twice)
    assert check_edit(AtMostOne("uv run pytest"), [], twice[:1])


def test_probe_score_reads_the_last_answer_line():
    assert probe_score("app/store.py", "path\nANSWER: `app/store.py`") == 1.0
    assert probe_score("4", "ANSWER: 7\nANSWER: 4.") == 1.0
    assert probe_score("4", "four") == 0.0


def test_grade_case_scores_gate_focus_and_edits():
    scenario = BY_ID["stale_entry"]
    judge = {
        "gate_decision": True,
        "fired": ["user_correction", "harness_contradicted"],
        "gate": {"user_correction": 0.9, "harness_contradicted": 0.8},
        "focus": {
            "home_user_correction": {
                "choice": "memory",
                "confidence": 0.9,
                "probabilities": {"memory": 0.9, "prompt": 0.1},
            },
            "wrong_0": 0.85,
        },
        "evidence_turns": [10],
        "entries": ["local:test-command"],
        "usage": {"gate": {"input_tokens": 5}},
    }
    edits = [_edit("update", "memory", "test-command", "uv run pytest")]
    records = _ledger(
        scenario.steer,
        {"should_refine": True, "rationale": "fired", "judge": judge},
        edits,
    )
    final = [{"title": "Test command", "content": "uv run pytest"}]

    row = grade_case(scenario, "typesafe", records, final, None, 0.7)

    assert row["decision"] is True and row["expect_refine"] is True
    assert all(row["edit_checks"].values())
    assert row["judge"]["signal_hit"] and row["judge"]["home_hit"] is None
    assert row["judge"]["entry_hit"] == 1 and row["judge"]["lesson_turn_hit"]

    declined = grade_case(
        BY_ID["one_off"],
        "model-gate",
        _ledger(BY_ID["one_off"].steer, {"should_refine": False, "rationale": "x"}, []),
        [],
        None,
        0.7,
    )
    assert declined["decision"] is False and declined["judge"] is None

    rendered = report([row, declined], [0.5, 0.9])
    assert "| typesafe | 1.00 | 1.00 | 1.00 | - |" in rendered
    assert "| model-gate | 1.00 |" in rendered
    assert sweep([row], [0.95]) == [
        "| threshold | call-1 accuracy |",
        "|---|---|",
        "| 0.95 | 0.00 |",
    ]
