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
    refine_prompt,
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


async def test_kernel_requested_pass_declined_by_the_planner_tells_the_model(session):
    """An empty plan changes nothing: no rebuilt window, no ``refinement`` edge; the
    model that asked hears why."""
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall("ipython", {"code": "await rlm.refine.run()"})
                ]
            ),
            DummyMessage(content=_proposal([], rationale="nothing reusable yet")),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    assert (await engine.run("learn")).answer == "done"
    notice = client.calls[2]["messages"][-1]["content"]
    assert 'kind="refinement"' in notice
    assert "Refinement declined: nothing reusable yet" in notice
    [declined] = _records(session, "refinement_declined")
    assert (declined["trigger"], declined["reason"]) == ("kernel", "no_edits")
    assert [w["reason"] for w in _records(session, "context_window")] == ["start"]
    edges = _edges(engine)
    assert edges.count("refinement_attempt") == 1 and "refinement" not in edges
    assert engine.execution_snapshot()["metrics"]["num_refinements"] == 0


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


LESSON = {"action": "create", "kind": "memory", "title": "Lesson", "content": "learned"}


def _edges(engine) -> list[str]:
    return [e["type"] for e in engine.execution_snapshot()["semantic_edges"]["edges"]]


async def test_auto_refine_plans_on_interval_and_an_empty_plan_declines(session):
    """Each automatic pass is one planning call; a plan with no edits declines it
    without touching the store, the context or the next request's edges."""
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})]),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "2"})]),
            # Pass after 2 work turns: nothing worth keeping.
            DummyMessage(content=_proposal([], rationale="noise")),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "3"})]),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "4"})]),
            # Pass after 2 more: a lesson.
            DummyMessage(content=_proposal([LESSON])),
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

    assert result.answer == "done" and result.turns == 5 and len(client.calls) == 7
    for call in (client.calls[2], client.calls[5]):
        plan = call["messages"][-1]["content"]
        assert call["tool_choice"] == "none"
        assert "improve the continual harness" in plan
        assert "<automatic_refinement>" in plan and "auto:turn_interval" in plan
    assert "2 work turns" in client.calls[2]["messages"][-1]["content"]
    [declined] = _records(session, "refinement_declined")
    assert (declined["trigger"], declined["reason"], declined["rationale"]) == (
        "auto:turn_interval",
        "no_edits",
        "noise",
    )
    assert "message" not in declined and len(declined["request_ids"]) == 1
    assert not any(
        "Refinement declined" in str(m.get("content"))
        for m in client.calls[3]["messages"]
    )
    assert _records(session, "refinement")[0]["trigger"] == "auto:turn_interval"
    assert len(load_history(HarnessStore(local_dir(session.dir)))) == 1
    metrics = engine.execution_snapshot()["metrics"]
    assert metrics["num_auto_refine_reviews"] == 2 and metrics["num_refinements"] == 1
    edges = _edges(engine)
    assert edges.count("refinement_attempt") == 2 and edges.count("refinement") == 1


HOME_MEMORY = {"choice": "memory", "confidence": 0.9, "probabilities": {"memory": 0.9}}


def _judged_engine(
    client, session, answers, *, auto_refine=True, global_dir=None, **judge
):
    config = make_runtime_config(
        harness=HarnessConfig(
            auto_refine=auto_refine,
            global_dir=global_dir,
            refine_turn_interval=1,
            refine_cooldown_seconds=0,
            refine_judge=RefineJudgeConfig(api_key="k", **judge),
        )
    )
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore
    fake = FakeTypeSafe(answers)
    engine._refine_judge = RefineJudge(config.harness.refine_judge, client=fake)
    return engine, fake


async def test_typesafe_gate_decides_before_the_planning_call(session):
    """An approving judge hands its focus to the one planning call; a declining one
    costs no model call and leaves no request in the edge graph."""
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})]),
            DummyMessage(content=_proposal([LESSON])),
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
    plan = client.calls[1]["messages"][-1]["content"]
    assert "Focus from the refinement review" in plan
    assert "Record it as a memory." in plan
    assert len(fake.calls) == 3
    assert [t["index"] for t in fake.calls[2][0]["turns"]][0] > max(
        t["index"] for t in fake.calls[0][0]["turns"]
    )
    [applied] = _records(session, "refinement")
    assert applied["judge"]["mode"] == "gate"
    assert applied["judge"]["usage"].keys() == {"gate", "focus"}
    [declined] = _records(session, "refinement_declined")
    assert declined["reason"] == "gate" and declined["judge"]["focus"] is None
    assert "message" not in declined
    metrics = engine.execution_snapshot()["metrics"]
    assert metrics["num_auto_refine_reviews"] == 2 and metrics["num_refinements"] == 1
    edges = _edges(engine)
    assert edges.count("refinement_attempt") == 1 and edges.count("refinement") == 1


async def test_shadow_judge_is_logged_beside_the_deciding_plan(session):
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})]),
            DummyMessage(content=_proposal([], rationale="noise")),
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
    assert (
        "Focus from the refinement review"
        not in (client.calls[1]["messages"][-1]["content"])
    )
    [declined] = _records(session, "refinement_declined")
    assert declined["reason"] == "no_edits"
    assert declined["judge"]["mode"] == "shadow"
    assert declined["judge"]["gate_decision"] is True
    assert "Record it as a memory." in declined["judge"]["instructions"]


async def test_host_review_and_focus_by_the_judge(session):
    """A host review by the judge can decline with no model call; host focus always
    plans, with the judge's instructions ahead of the host's, or the host's alone
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

    declined = await engine.prompt("", refine={**host, "review": True})
    assert declined.answer.startswith("[refinement declined: no signal reached")
    assert len(client.calls) == 1

    focused = await engine.prompt("", refine={**host, "focus": True})
    assert focused.answer == "[refinement declined: seen twice]"
    plan = client.calls[1]["messages"][-1]["content"]
    assert "Record it as a memory." in plan
    assert plan.endswith("Host instructions: keep\n</refine_instructions>")

    await engine.prompt("", refine={**host, "focus": True})
    assert client.calls[2]["messages"][-1]["content"].endswith(
        "<refine_instructions>\nkeep\n</refine_instructions>"
    )
    assert len(fake.calls) == 4
    passes = _records(session, "refinement_declined")
    assert [(r["trigger"], r["reason"], r["judge"]["mode"]) for r in passes] == [
        ("host", "gate", "gate"),
        ("host", "no_edits", "focus"),
        ("host", "no_edits", "focus"),
    ]
    assert not any("message" in r for r in passes)


async def test_host_global_pass_is_judged_against_the_global_store(session, tmp_path):
    """A global pass shows the judge the global store first, with its refinement
    history, and points the planner at the global entry the conversation contradicts."""
    global_dir = tmp_path / "global"
    HarnessStore(global_dir, scope="global").create(
        "memory", "Style", "answers are short"
    )
    client = DummyClient(
        [DummyMessage(content="hi"), DummyMessage(content=_proposal([]))]
    )
    engine, fake = _judged_engine(
        client,
        session,
        [{"harness_contradicted": 0.9}, {"wrong_0": 0.9}],
        auto_refine=False,
        global_dir=str(global_dir),
    )
    await engine.prompt("we want long answers now")

    await engine.prompt(
        "",
        refine={
            "instructions": None,
            "global_": True,
            "rollback_id": None,
            "focus": True,
        },
    )

    state = fake.calls[0][0]
    assert state["scope"] == "global"
    assert state["harness_entries"][0]["ref"] == "global:style"
    assert state["recent_refinements"] == ["No prior refinements."]
    plan = client.calls[1]["messages"][-1]["content"]
    assert "Requested scope: global" in plan
    assert (
        "Entry global:style is contradicted by the conversation: update or delete"
        in (plan)
    )


async def test_auto_refine_after_compaction_plans_on_the_new_blocks(session):
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})]),
            # Every tool result crosses the 1-token threshold: branch summary.
            DummyMessage(content="branch one: the parser fix needed two retries"),
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
    plan = client.calls[2]["messages"][-1]["content"]
    assert "<evidence>\n" + BLOCKS_NOTE in plan
    assert "[tier 1 | branch 0 |" in plan
    assert "the parser fix needed two retries" in plan
    assert "auto:compact" in plan
    assert plan.index("</evidence>") < plan.index("<automatic_refinement>")
    [applied] = _records(session, "refinement")
    assert (applied["trigger"], applied["blocks"]) == ("auto:compact", [[1, 0, 1]])
    assert engine._compact_refine_blocks is None


def test_refine_prompt_lists_blocks_coarse_first_and_notes_automatic_passes(tmp_path):
    view = build_view(local_dir(tmp_path))
    fine = Block(1, (4, 5), (400, 499), (4, 4), (40, 49), "fifth branch", "r5")
    coarse = Block(2, (0, 5), (0, 499), (0, 4), (0, 49), "x" * 700, "r-rollup")
    args = dict(scope="local", instructions=None, importable_names=["rlm"])
    prompt = refine_prompt(
        view, [], evidence=[fine, coarse], auto_note="automatic", **args
    )

    assert prompt.index(fine.header()) < prompt.index(coarse.header())
    assert "x" * 600 not in prompt
    assert BLOCKS_NOTE in prompt
    assert "<automatic_refinement>\nautomatic\n" in prompt
    plain = refine_prompt(view, [], **args)
    assert "<evidence>" not in plain and "<automatic_refinement>" not in plain


async def test_auto_refine_is_root_only_and_off_by_default(session):
    client = DummyClient(
        [DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "1"})])] * 3
        + [DummyMessage(content="done")]
    )
    config = make_runtime_config(harness=HarnessConfig(refine_turn_interval=1))
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore
    await engine.run("go")
    assert len(client.calls) == 4
    assert _records(session, "refinement") == []
    assert _records(session, "refinement_declined") == []


async def test_max_refinements_declines_further_passes(session):
    client = DummyClient([DummyMessage(content=_proposal([LESSON]))])
    config = make_runtime_config(harness=HarnessConfig(max_refinements=1))
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore
    host = {"instructions": None, "global_": False, "rollback_id": None}
    first = await engine.prompt("", refine=host)
    assert "create memory:lesson" in first.answer
    second = await engine.prompt("", refine=host)
    assert second.answer == (
        "[refinement declined: this agent reached max_refinements=1]"
    )
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
