"""Pure parts of the refinement eval: edit checks, probe scoring, judge grading and
the report. No sessions, no network."""

from __future__ import annotations

import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "refine_eval"))

from grade import check_edit, grade_case, probe_score, report, sweep  # noqa: E402
from scenarios import BY_ID, AtMostOne, Changed, Created, NoCreate  # noqa: E402


def _edit(action, kind, id, content="", applied=True, scope="local"):
    return {
        "action": action,
        "kind": kind,
        "id": id,
        "applied": applied,
        "title": id,
        "content": content,
        "scope": scope,
    }


def _ledger(steer, outcome, edits, scope="local"):
    records = [
        {"type": "user", "content": text, "message_index": 10 * i}
        for i, text in enumerate(steer)
    ]
    if edits:
        records.append(
            {
                "type": "refinement",
                "trigger": "host",
                "result": {"applied_edits": edits, "scope": scope},
                **outcome,
            }
        )
    else:
        records.append({"type": "refinement_declined", "trigger": "host", **outcome})
    return records


def test_edit_checks_read_applied_edits_and_the_final_store_of_their_scope():
    edits = [
        _edit("create", "memory", "counts", "exclude blank lines"),
        _edit("update", "memory", "test-command"),
        _edit("update", "memory", "answer-style", scope="global"),
    ]
    assert check_edit(Created(("memory",), "blank"), edits, {})
    assert not check_edit(Created(("prompt",), "blank"), edits, {})
    assert not check_edit(Created(("memory",), "blank", scope="global"), edits, {})
    assert check_edit(Changed("memory", "test-command"), edits, {})
    assert check_edit(Changed("memory", "answer-style", scope="global"), edits, {})
    assert not check_edit(Changed("memory", "answer-style"), edits, {})
    assert not check_edit(NoCreate(), edits, {})
    assert check_edit(NoCreate(scope="global"), edits, {})
    assert check_edit(NoCreate("pytest"), edits, {})
    twice = [{"title": "a", "content": "uv run pytest"}] * 2
    assert not check_edit(AtMostOne("uv run pytest"), [], {"local": twice})
    assert check_edit(AtMostOne("uv run pytest"), [], {"local": twice[:1]})
    assert check_edit(AtMostOne("uv run pytest", scope="global"), [], {"local": twice})


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
        {"judge": {"mode": "gate", **judge}},
        edits,
    )
    final = {"local": [{"title": "Test command", "content": "uv run pytest"}]}

    row = grade_case(scenario, "typesafe", records, final, None, 0.7)

    assert row["decision"] is True and row["expect_refine"] is True
    assert all(row["edit_checks"].values())
    assert row["judge"]["signal_hit"] and row["judge"]["home_hit"] is None
    assert row["judge"]["entry_hit"] == 1 and row["judge"]["lesson_turn_hit"]

    declined = grade_case(
        BY_ID["one_off"],
        "force",
        _ledger(
            BY_ID["one_off"].steer, {"reason": "no_edits", "rationale": "noise"}, []
        ),
        {},
        None,
        0.7,
    )
    assert declined["decision"] is False and declined["judge"] is None
    assert declined["declined_by"] == "no_edits"

    failed = {**declined, "declined_by": "failed"}
    rendered = report([row, declined, failed], [0.5, 0.9])
    assert "3 cases, 1 errored or failed." in rendered
    assert "| typesafe | 1.00 | 1.00 | 1.00 | - |" in rendered
    assert "| force | 1.00 |" in rendered
    assert "| force | decline | 0 | 0 | 1 |" in rendered
    assert sweep([row], [0.95]) == [
        "| threshold | call-1 accuracy |",
        "|---|---|",
        "| 0.95 | 0.00 |",
    ]


async def test_run_case_seeds_the_global_store_and_runs_a_global_pass(
    monkeypatch, tmp_path
):
    """The driver seeds each store, triggers a ``global: true`` pass for a global
    scenario and grades the global store's edits."""
    from conftest import DummyClient, DummyMessage
    from run import Settings, run_case

    proposal = {
        "summary": "answers are detailed now",
        "rationale": "the user changed their preference across projects",
        "expected_outcome": "longer answers",
        "edits": [
            {
                "action": "update",
                "kind": "memory",
                "id": "answer-style",
                "title": "Answer style",
                "content": "The user wants detailed answers that explain the reasoning.",
            }
        ],
    }
    client = DummyClient(
        [
            DummyMessage(content="It merges dicts. ANSWER: merge"),
            DummyMessage(content="Understood."),
            DummyMessage(content=json.dumps(proposal)),
        ]
    )

    async def close():
        pass

    client.close = close  # the engine owns clients it makes, so it closes this one
    monkeypatch.setattr("rlm.engine.make_client", lambda provider: client)
    settings = Settings(model="dummy", api_key="k", base_url=None, typesafe_api_key="t")

    row = await run_case(BY_ID["global_stale_entry"], "force", 0, settings, tmp_path)

    assert row["error"] is None
    assert row["scope"] == "global" and row["decision"] is True
    assert row["edits"] == ["update memory:answer-style"]
    assert all(row["edit_checks"].values())
    plan = client.calls[2]["messages"][-1]["content"]
    assert "Requested scope: global" in plan
    assert "[global:answer-style]" in plan
