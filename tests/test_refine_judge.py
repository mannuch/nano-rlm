"""Tests for the TypeSafe refinement judge: evidence building, question selection and
the policy that turns answers into a verdict and planner instructions."""

from __future__ import annotations

import json

from conftest import FakeTypeSafe

from rlm.config import RefineJudgeConfig
from rlm.harness import build_view
from rlm.provenance import runtime_event
from rlm.refine_judge import (
    CONTRADICTED,
    RefineJudge,
    build_evidence,
    decide_focus,
    decide_gate,
    focus_questions,
    focus_state,
    gate_questions,
)

CONFIG = RefineJudgeConfig(api_key="k")


def _view(tmp_path, entries=()):
    view = build_view(tmp_path / "local", global_dir=tmp_path / "global")
    for layer, title, content in entries:
        view.create("memory", title=title, content=content, global_=layer == "global")
    return view


def _evidence(tmp_path, messages, entries=(), **kwargs):
    return build_evidence(
        task="count lines",
        trigger="host",
        messages=list(enumerate(messages)),
        view=_view(tmp_path, entries),
        history=[],
        **kwargs,
    )


def test_build_evidence_labels_turns_counts_errors_and_trims_non_user_turns_first(
    tmp_path,
):
    notice, _ = runtime_event("notice", "inbox has 1 message")
    failure = "Traceback (most recent call last):\nValueError: bad count"
    code = {"code": "print(1)"}
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "no, exclude blank lines from now on"},
        notice,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "ipython", "arguments": json.dumps(code)}}
            ],
        },
        {"role": "tool", "content": failure},
        {"role": "tool", "content": failure + "x" * 400},
    ]
    evidence = _evidence(tmp_path, messages, entries=[("local", "Tests", "uv run")])

    assert [(t["index"], t["role"]) for t in evidence["turns"]] == [
        (1, "user"),
        (2, "runtime"),
        (3, "assistant"),
        (4, "tool"),
        (5, "tool"),
    ]
    assert evidence["turns"][2]["code"] == "print(1)"
    assert evidence["turns"][3]["error"] == "ValueError"
    assert evidence["error_counts"] == {"ValueError": 2}
    assert evidence["harness_entries"] == [
        {"ref": "local:tests", "kind": "memory", "title": "Tests", "content": "uv run"}
    ]

    budget = len(json.dumps(evidence, default=str)) - 300
    trimmed = _evidence(
        tmp_path / "trimmed",
        messages,
        entries=[("local", "Tests", "uv run")],
        max_chars=budget,
    )
    kept = [t["index"] for t in trimmed["turns"]]
    assert 1 in kept and 5 not in kept and len(json.dumps(trimmed)) <= budget


def test_gate_asks_contradiction_only_with_entries_and_fires_at_threshold(tmp_path):
    bare = _evidence(tmp_path / "a", [])
    seeded = _evidence(tmp_path / "b", [], entries=[("local", "Tests", "uv run")])
    assert CONTRADICTED not in gate_questions(bare)
    assert CONTRADICTED in gate_questions(seeded)
    answers = {"user_correction": 0.7, "durable_fact": 0.69, CONTRADICTED: 0.9}
    assert decide_gate(answers, 0.7) == ["user_correction", CONTRADICTED]


def test_focus_questions_follow_what_fired(tmp_path):
    messages = [
        {"role": "user", "content": "no"},
        {"role": "assistant", "content": "ok"},
    ]
    entries = [("local", "Tests", "uv run"), ("global", "Style", "short")]
    evidence = _evidence(tmp_path, messages, entries=entries)

    lesson = focus_questions(
        focus_state(evidence, ["user_correction"]), ["user_correction"]
    )
    assert set(lesson) == {
        "captured_user_correction",
        "home_user_correction",
        "covers_0",
        "turn_0",
        "turn_1",
    }
    assert lesson["home_user_correction"].type == "choice"

    stale = focus_questions(focus_state(evidence, [CONTRADICTED]), [CONTRADICTED])
    assert set(stale) == {"wrong_0", "wrong_1"}


def test_a_global_pass_targets_global_entries_and_future_sessions(tmp_path):
    messages = [{"role": "user", "content": "no"}]
    entries = [("local", "Tests", "uv run"), ("global", "Style", "short")]
    evidence = _evidence(tmp_path, messages, entries=entries, scope="global")

    assert evidence["scope"] == "global"
    assert [e["ref"] for e in evidence["harness_entries"]] == [
        "global:style",
        "local:tests",
    ]
    fact = gate_questions(evidence)["durable_fact"]
    assert "tasks in future sessions" in fact.instructions
    fired = ["durable_fact", CONTRADICTED]
    questions = focus_questions(focus_state(evidence, fired), fired)
    assert {k for k in questions if k.startswith(("covers_", "wrong_"))} == {
        "covers_0",
        "wrong_0",
    }
    assert "tasks in future sessions" in questions["home_durable_fact"].instructions


def test_a_local_pass_overrides_a_contradicted_read_only_entry(tmp_path):
    evidence = _evidence(
        tmp_path,
        [{"role": "user", "content": "we stopped writing short answers"}],
        entries=[("global", "Style", "short answers")],
    )
    decision, rationale, instructions, _ = decide_focus(
        [CONTRADICTED], {"wrong_0": 0.9}, {}, evidence, CONFIG
    )
    assert decision and "contradicted: global:style" in rationale
    assert "Entry global:style is contradicted by the conversation but read-only" in (
        instructions
    )
    assert "create a local entry that overrides it" in instructions


def test_decide_focus_vetoes_names_homes_and_quotes_evidence(tmp_path):
    messages = [
        {"role": "user", "content": "no, exclude blank lines from now on"},
        {"role": "assistant", "content": "ok"},
    ]
    evidence = _evidence(
        tmp_path, messages, entries=[("local", "Counting", "count every line")]
    )
    fired = ["user_correction", "durable_fact"]
    confident = {
        "choice": "memory",
        "confidence": 0.9,
        "probabilities": {"memory": 0.9},
    }
    split = {
        "choice": "prompt",
        "confidence": 0.3,
        "probabilities": {
            "prompt": 0.45,
            "memory": 0.4,
            "skill": 0.1,
            "subagent": 0.05,
        },
    }
    nouls = {
        "captured_durable_fact": 0.95,
        "covers_0": 0.8,
        "turn_0": 0.9,
        "turn_1": 0.1,
    }

    decision, rationale, instructions, turns = decide_focus(
        fired, nouls, {"home_user_correction": confident}, evidence, CONFIG
    )
    assert decision and turns == [0]
    assert "already recorded: durable_fact" in rationale
    assert "a user message corrected" in instructions
    assert "Record it as a memory." in instructions
    assert "project fact" not in instructions
    assert "local:counting" in instructions and "Update them" in instructions
    assert '"no, exclude blank lines from now on"' in instructions

    _, _, split_instructions, _ = decide_focus(
        ["user_correction"], {}, {"home_user_correction": split}, evidence, CONFIG
    )
    assert "Record it as a prompt or a memory." in split_instructions

    captured = {"captured_user_correction": 0.9}
    assert decide_focus(["user_correction"], captured, {}, evidence, CONFIG)[0] is False

    wrong = {"wrong_0": 0.8}
    decision, _, instructions, _ = decide_focus(
        [CONTRADICTED], wrong, {}, evidence, CONFIG
    )
    assert decision and "Entry local:counting is contradicted" in instructions


async def test_review_declines_after_one_call_unless_forced(tmp_path):
    evidence = _evidence(tmp_path, [{"role": "user", "content": "hello"}])

    quiet = FakeTypeSafe([{"durable_fact": 0.4}])
    verdict = await RefineJudge(CONFIG, client=quiet).review(evidence)
    assert len(quiet.calls) == 1 and verdict.should_refine is False
    assert "top durable_fact (0.40)" in verdict.rationale

    forced = await RefineJudge(CONFIG, client=FakeTypeSafe([{}])).review(
        evidence, force=True
    )
    assert forced.should_refine and not forced.gate_decision
    assert forced.instructions is None

    home = {"choice": "memory", "confidence": 0.9, "probabilities": {"memory": 0.9}}
    fired = FakeTypeSafe([{"user_correction": 0.9}, {"home_user_correction": home}])
    verdict = await RefineJudge(CONFIG, client=fired).review(evidence)
    assert len(fired.calls) == 2 and verdict.should_refine
    assert fired.calls[1][0]["gate"]["fired"][0]["id"] == "user_correction"
    assert verdict.usage.keys() == {"gate", "focus"}
    assert "Record it as a memory." in verdict.instructions
