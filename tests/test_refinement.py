"""Tests for continual-harness refinement: proposal parsing, apply/rollback, and the
engine's kernel, host and automatic triggers."""

from __future__ import annotations

import json

import pytest
from conftest import (
    DummyClient,
    DummyMessage,
    DummyToolCall,
    FakeTypeSafe,
    make_runtime_config,
    tool_result,
)

from rlm.config import ExecutionPolicy, HarnessConfig, RefineJudgeConfig
from rlm.engine import RLMEngine
from rlm.refine_judge import RefineJudge
from rlm.harness import HarnessStore, build_view, local_dir
from rlm.history import read_records
from rlm.refinement import (
    BLOCKS_NOTE,
    RefinementRejected,
    apply_proposal,
    baseline_of,
    extract_json_object,
    find_result,
    load_history,
    parse_proposal,
    parse_review,
    review_prompt,
    rollback_proposal,
)
from rlm.staircase import Block

IMPORTABLE = {"websearch", "rlm"}


def _proposal(edits: list[dict], **fields) -> str:
    return json.dumps(
        {
            "summary": "record lessons",
            "rationale": "seen twice",
            "expected_outcome": "fewer retries",
            "edits": edits,
            **fields,
        }
    )


def test_extract_json_object_handles_fences_prose_and_truncation():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Here it is {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(RefinementRejected, match="truncated"):
        extract_json_object('{"edits": [{"action": "create"')
    with pytest.raises(RefinementRejected, match="does not contain"):
        extract_json_object("no json here")
    with pytest.raises(RefinementRejected, match="no edits array"):
        parse_proposal('{"summary": "x"}')
    assert parse_review('{"should_refine": true, "rationale": "r"}') == (
        True,
        "r",
        None,
    )
    assert parse_review('{"should_refine": "yes"}')[0] is False


def test_apply_proposal_validates_each_edit_and_records_snapshots(tmp_path):
    store = HarnessStore(tmp_path / "h")
    store.create("memory", "Old fact", "stale")
    proposal = parse_proposal(
        _proposal(
            [
                {
                    "action": "create",
                    "kind": "memory",
                    "title": "Use uv run",
                    "content": "always",
                },
                {
                    "action": "update",
                    "kind": "memory",
                    "id": "old_fact",
                    "title": "Old fact",
                    "content": "fresh",
                },
                {
                    "action": "create",
                    "kind": "skill",
                    "title": "Search",
                    "content": "x",
                    "reference": {
                        "type": "python",
                        "import": "nope",
                        "callable": "run",
                    },
                    "arguments": {},
                },
                {
                    "action": "create",
                    "kind": "skill",
                    "title": "Real search",
                    "content": "x",
                    "reference": {
                        "type": "python",
                        "import": "websearch",
                        "callable": "run",
                    },
                    "arguments": {"queries": {"type": "array", "required": True}},
                },
                {"action": "delete", "kind": "memory", "id": "missing"},
                {
                    "action": "create",
                    "kind": "prompt",
                    "id": "base_system_prompt",
                    "title": "t",
                    "content": "c",
                },
                {
                    "action": "update",
                    "kind": "memory",
                    "title": "no id",
                    "content": "c",
                },
                {"action": "explode", "kind": "memory", "id": "old_fact"},
            ]
        )
    )
    result = apply_proposal(
        store,
        proposal,
        trigger="test",
        importable_names=IMPORTABLE,
        baseline=baseline_of(store),
    )
    outcomes = [(e.id, e.applied, e.error) for e in result.applied_edits]
    assert outcomes == [
        ("use_uv_run", True, None),
        ("old_fact", True, None),
        ("search", False, "create skill references unknown module 'nope'"),
        ("real_search", True, None),
        ("missing", False, "entry not found"),
        ("base_system_prompt", False, "base system prompt is not editable"),
        ("", False, "update requires id"),
        ("old_fact", False, "unsupported action 'explode'"),
    ]
    assert result.changes == [
        "create memory:use_uv_run",
        "update memory:old_fact",
        "create skill:real_search",
    ]
    update = result.applied_edits[1]
    assert update.before["content"] == "stale" and update.after["version"] == 2
    assert update.after["source"] == "refinement"
    assert store.get("memory", "old_fact").content == "fresh"
    assert [e.id for e in store.list_refinements()] == [result.id]

    history = load_history(store)
    assert len(history) == 1 and history[0].id == result.id
    assert find_result(store, result.id).changes == result.changes

    rollback = apply_proposal(
        store,
        rollback_proposal(history[0]),
        trigger="rollback",
        importable_names=IMPORTABLE,
        rollback_of=result.id,
    )
    assert rollback.changes == [
        "delete skill:real_search",
        "update memory:old_fact",
        "delete memory:use_uv_run",
    ]
    assert store.get("memory", "old_fact").content == "stale"
    assert store.list("skill") == [] and store.get("memory", "use_uv_run") is None
    assert load_history(store)[-1].rollback_of == result.id


def test_apply_proposal_rejects_edits_to_entries_changed_during_planning(tmp_path):
    store = HarnessStore(tmp_path / "h")
    store.create("memory", "Fact", "v1")
    baseline = baseline_of(store)
    HarnessStore(tmp_path / "h").update(
        "memory", "fact", "Fact", "v2"
    )  # another writer
    proposal = parse_proposal(
        _proposal(
            [
                {
                    "action": "update",
                    "kind": "memory",
                    "id": "fact",
                    "title": "Fact",
                    "content": "v3",
                }
            ]
        )
    )
    result = apply_proposal(
        store, proposal, trigger="t", importable_names=set(), baseline=baseline
    )
    assert result.applied_edits[0].error == "entry changed during refinement planning"
    assert store.get("memory", "fact").content == "v2"


def _records(session, kind: str) -> list[dict]:
    return [
        r for r in read_records(session.dir / "messages.jsonl") if r.get("type") == kind
    ]


async def test_kernel_requested_refinement_runs_at_the_turn_boundary(session):
    """``rlm.refine.run()`` schedules a pass; the next model-call boundary plans it with
    a side request, applies the edits, rebuilds the system prompt, and notifies the model."""
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {"code": "print(await rlm.refine.run('note the uv lesson'))"},
                    )
                ]
            ),
            # The refinement request: first reply is unusable, second is a proposal.
            DummyMessage(content="I would rather not."),
            DummyMessage(
                content=_proposal(
                    [
                        {
                            "action": "create",
                            "kind": "memory",
                            "title": "Use uv",
                            "content": "always uv run",
                        }
                    ]
                )
            ),
            DummyMessage(content="done"),
        ]
    )
    config = make_runtime_config(harness=HarnessConfig(max_refinement_attempts=2))
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore

    result = await engine.run("learn")

    assert result.answer == "done"
    assert "'scheduled': True" in tool_result(client)
    plan_calls = client.calls[1:3]
    for call in plan_calls:
        assert call["tool_choice"] == "none"
        assert call["messages"][-1]["role"] == "user"
        assert "Reply with JSON only" in call["messages"][-1]["content"]
        assert "Importable modules:" in call["messages"][-1]["content"]
    ids = [call["extra_headers"]["X-ACP-Model-Request-ID"] for call in client.calls]
    assert len(set(ids)) == 4

    final_call = client.calls[3]["messages"]
    assert "[local:use_uv] Use uv" in final_call[0]["content"]
    assert 'kind="refinement"' in final_call[-1]["content"]
    assert "create memory:use_uv" in final_call[-1]["content"]
    assert final_call[-2]["role"] == "tool"  # the cell's result precedes the notice

    store = HarnessStore(local_dir(session.dir))
    assert store.get("memory", "use_uv").source == "refinement"
    (record,) = _records(session, "refinement")
    assert record["trigger"] == "kernel"
    assert record["result"]["request_ids"] == ids[1:3]
    assert record["result"]["usage"]["prompt_tokens"] == 2
    windows = _records(session, "context_window")
    assert [w["reason"] for w in windows] == ["start", "harness"]
    # The rebuilt window keeps the user/assistant/tool indices and swaps in the new
    # system message (logged after them).
    rebuilt = windows[1]["message_indices"]
    assert rebuilt[1:] == [1, 2, 3] and rebuilt[0] > 3

    edges = engine.execution_snapshot()["semantic_edges"]["edges"]
    typed = {
        (
            ids.index(e["source_request_id"]),
            ids.index(e["target_request_id"]),
            e["type"],
        )
        for e in edges
    }
    assert typed == {
        (0, 1, "refinement_attempt"),
        (0, 2, "refinement_attempt"),
        (2, 3, "refinement"),
        (0, 3, "continuation"),
    }
    assert result.turns == 2
    metrics = engine.execution_snapshot()["metrics"]
    assert metrics["num_refinements"] == 1 and metrics["refinement_edits_applied"] == 1
    assert engine.execution_snapshot()["harness"]["local"]["memory"] == 1


async def test_refinement_failure_is_reported_and_the_run_continues(session):
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall("ipython", {"code": "await rlm.refine.run()"})
                ]
            ),
            DummyMessage(content="not json"),
            DummyMessage(content="done"),
        ]
    )
    config = make_runtime_config(harness=HarnessConfig(max_refinement_attempts=1))
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore

    result = await engine.run("learn")

    assert result.answer == "done"
    notice = client.calls[2]["messages"][-1]["content"]
    assert "Refinement failed: no usable proposal after 1 attempts" in notice
    assert _records(session, "refinement") == []
    assert _records(session, "refinement_declined")[0]["reason"] == "failed"
    assert [w["reason"] for w in _records(session, "context_window")] == ["start"]
    edges = engine.execution_snapshot()["semantic_edges"]["edges"]
    assert sorted(e["type"] for e in edges) == ["continuation", "refinement_attempt"]


async def test_refine_run_is_declined_without_a_global_store(session):
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython", {"code": "print(await rlm.refine.run(global_=True))"}
                    )
                ]
            ),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore
    await engine.run("x")
    assert "'scheduled': False" in tool_result(client)
    assert "no global harness store" in tool_result(client)
    assert len(client.calls) == 2


async def test_host_refinement_and_rollback(session, tmp_path):
    """A host-requested pass runs before the turn; an empty prompt makes it the whole
    turn, and a rollback by id needs no model call."""
    client = DummyClient(
        [
            DummyMessage(
                content=_proposal(
                    [
                        {
                            "action": "create",
                            "kind": "prompt",
                            "title": "Policy",
                            "content": "verify",
                        }
                    ]
                )
            ),
            DummyMessage(content="after"),
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    refined = await engine.prompt(
        "", refine={"instructions": "keep", "global_": False, "rollback_id": None}
    )
    assert refined.turns == 0 and "create prompt:policy" in refined.answer
    assert engine.stop_reason == "refined"
    assert client.calls[0]["messages"][-1]["content"].endswith(
        "<refine_instructions>\nkeep\n</refine_instructions>"
    )
    store = HarnessStore(local_dir(session.dir))
    assert store.get("prompt", "policy") is not None
    result_id = load_history(store)[0].id

    followup = await engine.prompt(
        "now work",
        refine={"instructions": None, "global_": False, "rollback_id": result_id},
    )
    assert followup.answer == "after" and len(client.calls) == 2
    assert store.get("prompt", "policy") is None
    assert load_history(store)[-1].rollback_of == result_id
    assert "[local:policy]" not in client.calls[1]["messages"][0]["content"]
    assert "Rollback refinement" in client.calls[1]["messages"][-2]["content"]

    with pytest.raises(ValueError):
        await RLMEngine(  # type: ignore
            client=DummyClient([]),
            session=__import__("rlm.session", fromlist=["Session"]).Session(
                tmp_path / "s2"
            ),
            runtime_config=make_runtime_config(harness=HarnessConfig(enabled=False)),
        ).prompt(
            "", refine={"instructions": None, "global_": False, "rollback_id": None}
        )


async def test_auto_refine_reviews_on_interval_and_only_refines_when_approved(session):
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})]),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "2"})]),
            # Review after 2 work turns: declined.
            DummyMessage(content='{"should_refine": false, "rationale": "noise"}'),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "3"})]),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "4"})]),
            # Review after 2 more: approved, then the plan.
            DummyMessage(
                content='{"should_refine": true, "rationale": "lesson", "instructions": "record it"}'
            ),
            DummyMessage(
                content=_proposal(
                    [
                        {
                            "action": "create",
                            "kind": "memory",
                            "title": "Lesson",
                            "content": "learned",
                        }
                    ]
                )
            ),
            DummyMessage(content="done"),
        ]
    )
    config = make_runtime_config(
        harness=HarnessConfig(
            auto_refine=True, refine_turn_interval=2, refine_cooldown_seconds=0
        )
    )
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore

    result = await engine.run("go")

    assert result.answer == "done" and result.turns == 5
    review_calls = [client.calls[2], client.calls[5]]
    for call in review_calls:
        assert call["tool_choice"] == "none"
        assert (
            "should run a continual-harness refinement"
            in call["messages"][-1]["content"]
        )
    assert (
        "2 work turns since the last review"
        in client.calls[2]["messages"][-1]["content"]
    )
    plan = client.calls[6]["messages"][-1]["content"]
    assert plan.endswith("<refine_instructions>\nrecord it\n</refine_instructions>")
    reviews = _records(session, "refinement_review")
    assert [(r["reason"], r["should_refine"]) for r in reviews] == [
        ("turn_interval", False),
        ("turn_interval", True),
    ]
    assert _records(session, "refinement")[0]["trigger"] == "auto:turn_interval"
    metrics = engine.execution_snapshot()["metrics"]
    assert metrics["num_auto_refine_reviews"] == 2 and metrics["num_refinements"] == 1
    types = sorted(
        e["type"] for e in engine.execution_snapshot()["semantic_edges"]["edges"]
    )
    assert types.count("refinement_attempt") == 3 and types.count("refinement") == 1


HOME_MEMORY = {"choice": "memory", "confidence": 0.9, "probabilities": {"memory": 0.9}}


def _judged_engine(client, session, answers, *, auto_refine=True, **judge):
    config = make_runtime_config(
        harness=HarnessConfig(
            auto_refine=auto_refine,
            refine_turn_interval=1,
            refine_cooldown_seconds=0,
            refine_judge=RefineJudgeConfig(api_key="k", **judge),
        )
    )
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore
    fake = FakeTypeSafe(answers)
    engine._refine_judge = RefineJudge(config.harness.refine_judge, client=fake)
    return engine, fake


async def test_typesafe_gate_replaces_the_model_review(session):
    """The judge's gate decides without a model review call; an approval hands its
    focus instructions to the planning call, and a decline costs one judge call."""
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})]),
            DummyMessage(
                content=_proposal(
                    [
                        {
                            "action": "create",
                            "kind": "memory",
                            "title": "Lesson",
                            "content": "learned",
                        }
                    ]
                )
            ),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "2"})]),
            DummyMessage(content="done"),
        ]
    )
    engine, fake = _judged_engine(
        client,
        session,
        [{"user_correction": 0.9}, {"home_user_correction": HOME_MEMORY}, {}],
    )

    result = await engine.run("go")

    assert result.answer == "done" and len(client.calls) == 4
    assert not any(
        "should run a continual-harness refinement" in c["messages"][-1]["content"]
        for c in client.calls
    )
    plan = client.calls[1]["messages"][-1]["content"]
    assert "Focus from the refinement review" in plan
    assert "Record it as a memory." in plan
    assert len(fake.calls) == 3
    assert [t["index"] for t in fake.calls[2][0]["turns"]][0] > max(
        t["index"] for t in fake.calls[0][0]["turns"]
    )
    reviews = _records(session, "refinement_review")
    assert [(r["reviewer"], r["should_refine"]) for r in reviews] == [
        ("typesafe", True),
        ("typesafe", False),
    ]
    assert reviews[0]["judge"]["usage"].keys() == {"gate", "focus"}
    assert reviews[1]["judge"]["focus"] is None
    metrics = engine.execution_snapshot()["metrics"]
    assert metrics["num_auto_refine_reviews"] == 2 and metrics["num_refinements"] == 1
    types = [e["type"] for e in engine.execution_snapshot()["semantic_edges"]["edges"]]
    assert types.count("refinement_attempt") == 1 and types.count("refinement") == 1


async def test_shadow_judge_is_logged_beside_the_deciding_model_review(session):
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})]),
            DummyMessage(content='{"should_refine": false, "rationale": "noise"}'),
            DummyMessage(content="done"),
        ]
    )
    engine, _ = _judged_engine(
        client,
        session,
        [{"user_correction": 0.9}, {"home_user_correction": HOME_MEMORY}],
        mode="shadow",
    )

    assert (await engine.run("go")).answer == "done"
    [review] = _records(session, "refinement_review")
    assert review["reviewer"] == "model" and review["should_refine"] is False
    assert review["shadow"]["gate_decision"] is True
    assert "Record it as a memory." in review["shadow"]["instructions"]


async def test_host_review_and_focus_by_the_judge(session):
    """A host review by the judge can decline with no model call; host focus always
    refines, with the judge's instructions ahead of the host's, or the host's alone
    when nothing fired."""
    proposal = DummyMessage(content=_proposal([]))
    client = DummyClient([DummyMessage(content="hi"), proposal, proposal])
    engine, fake = _judged_engine(
        client,
        session,
        [
            {"durable_fact": 0.2},
            {"user_correction": 0.9},
            {"home_user_correction": HOME_MEMORY},
            {},
        ],
        auto_refine=False,
    )
    host = {"instructions": "keep", "global_": False, "rollback_id": None}
    await engine.prompt("hello")

    declined = await engine.prompt("", refine={**host, "review": "typesafe"})
    assert declined.answer.startswith("[refinement declined: no signal reached")
    assert len(client.calls) == 1

    await engine.prompt("", refine={**host, "focus": True})
    plan = client.calls[1]["messages"][-1]["content"]
    assert "Record it as a memory." in plan
    assert plan.endswith("Host instructions: keep\n</refine_instructions>")

    await engine.prompt("", refine={**host, "focus": True})
    assert client.calls[2]["messages"][-1]["content"].endswith(
        "<refine_instructions>\nkeep\n</refine_instructions>"
    )
    assert len(fake.calls) == 4
    reviews = _records(session, "refinement_review")
    assert [(r["reason"], r["reviewer"], r["should_refine"]) for r in reviews] == [
        ("host", "typesafe", False),
        ("host", None, True),
        ("host", None, True),
    ]


async def test_auto_refine_after_compaction_reviews_the_new_blocks(session):
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})]),
            # Every tool result crosses the 1-token threshold: branch summary.
            DummyMessage(content="branch one: the parser fix needed two retries"),
            # Review on the compact trigger, approved, then the plan.
            DummyMessage(
                content='{"should_refine": true, "rationale": "recurring", "instructions": "keep it"}'
            ),
            DummyMessage(
                content=_proposal(
                    [
                        {
                            "action": "create",
                            "kind": "memory",
                            "title": "Parser retries",
                            "content": "two retries were needed",
                        }
                    ]
                )
            ),
            DummyMessage(content="done"),
        ]
    )
    config = make_runtime_config(
        policy=ExecutionPolicy(summarize_at_tokens=1),
        harness=HarnessConfig(auto_refine=True, refine_cooldown_seconds=0),
    )
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore

    result = await engine.run("fix the parser")

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1
    review = client.calls[2]["messages"][-1]["content"]
    assert "trigger:\ncompact" in review
    assert "<compaction_blocks>\n" + BLOCKS_NOTE in review
    assert "[tier 1 | branch 0 |" in review
    assert "the parser fix needed two retries" in review
    plan = client.calls[3]["messages"][-1]["content"]
    assert "<evidence>\n" + BLOCKS_NOTE in plan
    assert plan.index("</evidence>") < plan.index("<refine_instructions>")
    reviews = _records(session, "refinement_review")
    assert [(r["reason"], r["blocks"]) for r in reviews] == [("compact", [[1, 0, 1]])]
    assert _records(session, "refinement")[0]["trigger"] == "auto:compact"
    assert engine._compact_refine_blocks is None


def test_review_prompt_lists_new_blocks_coarse_first(tmp_path):
    view = build_view(local_dir(tmp_path))
    fine = Block(1, (4, 5), (400, 499), (4, 4), (40, 49), "fifth branch", "r5")
    coarse = Block(2, (0, 5), (0, 499), (0, 4), (0, 49), "x" * 700, "r-rollup")
    prompt = review_prompt(
        view, [], trigger="compact", turns_since_review=3, blocks=[fine, coarse]
    )

    assert prompt.index(fine.header()) < prompt.index(coarse.header())
    assert "x" * 600 not in prompt
    assert BLOCKS_NOTE in prompt
    assert "compaction blocks since" in prompt
    assert "<compaction_blocks>" not in review_prompt(
        view, [], trigger="turn_interval", turns_since_review=3
    )


async def test_auto_refine_is_root_only_and_off_by_default(session):
    client = DummyClient(
        [DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})])] * 3
        + [DummyMessage(content="done")]
    )
    config = make_runtime_config(harness=HarnessConfig(refine_turn_interval=1))
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore
    await engine.run("go")
    assert len(client.calls) == 4 and _records(session, "refinement_review") == []


async def test_max_refinements_declines_further_passes(session):
    client = DummyClient(
        [
            DummyMessage(content=_proposal([])),
            DummyMessage(content="done"),
        ]
    )
    config = make_runtime_config(harness=HarnessConfig(max_refinements=1))
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore
    first = await engine.prompt(
        "", refine={"instructions": None, "global_": False, "rollback_id": None}
    )
    assert "No edits applied." in first.answer
    second = await engine.prompt(
        "", refine={"instructions": None, "global_": False, "rollback_id": None}
    )
    assert second.answer == "[refinement declined]"
    assert _records(session, "refinement_declined")[0]["reason"] == "limit"
    assert len(client.calls) == 1


def test_global_scope_targets_the_global_store(tmp_path):
    view = build_view(tmp_path / "l", global_dir=tmp_path / "g")
    proposal = parse_proposal(
        _proposal(
            [{"action": "create", "kind": "memory", "title": "Shared", "content": "x"}]
        )
    )
    result = apply_proposal(view.global_, proposal, trigger="t", importable_names=set())
    assert result.scope == "global"
    assert view.get("memory", "shared").scope == "global"
    assert (tmp_path / "g" / "refinements.jsonl").exists()
    assert not (tmp_path / "l" / "refinements.jsonl").exists()
