from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import httpx
from openai import BadRequestError
import pytest

from conftest import (
    DummyChoice,
    DummyClient,
    DummyMessage,
    DummyResponse,
    DummyToolCall,
    DummyUsage,
)
from rlm.compaction import (
    ROLLUP_PROMPT,
    STAIRCASE_FRAMING,
    CompactionFailed,
    is_context_overflow,
)
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.history import history
from rlm.session import Session
from rlm.staircase import Block, Staircase, select_tail
from rlm.supervisor import SessionTreeSupervisor


def _response(
    message: DummyMessage,
    *,
    prompt_tokens: int = 1,
    completion_tokens: int = 1,
    finish_reason: str = "stop",
) -> DummyResponse:
    return DummyResponse(
        choices=[DummyChoice(message=message, finish_reason=finish_reason)],
        usage=DummyUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _overflow() -> BadRequestError:
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://interceptor/v1/chat/completions"),
    )
    return BadRequestError(
        "This model's maximum context length is 4096 tokens.",
        response=response,
        body={
            "error": {"message": "This model's maximum context length is 4096 tokens."}
        },
    )


def test_overflow_detection_is_status_gated():
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://interceptor/v1/chat/completions"),
    )
    error = BadRequestError(
        "maximum context length is 32,768 tokens",
        response=response,
        body={"error": {"message": "maximum context length is 32,768 tokens"}},
    )

    assert is_context_overflow(error)


class _ScriptedClient(DummyClient):
    def __init__(
        self,
        actions: list[DummyResponse | BaseException],
        *,
        max_model_len: int | None = None,
    ):
        super().__init__([])
        self.actions = list(actions)
        self.max_model_len = max_model_len
        self.base_url = f"http://scripted-{id(self)}"

    @property
    def models(self):
        outer = self

        class _Models:
            async def list(self):
                extra = (
                    {"max_model_len": outer.max_model_len}
                    if outer.max_model_len is not None
                    else {}
                )
                card = SimpleNamespace(id="test-model", model_extra=extra)
                return SimpleNamespace(data=[card])

        return _Models()

    async def create(self, **kwargs: Any) -> DummyResponse:
        self.calls.append(deepcopy(kwargs))
        if not self.actions:
            raise AssertionError("script exhausted")
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


def _config(
    *,
    max_depth: int = 0,
    summarize_at_tokens: int | None = None,
    compaction: bool = True,
    max_compaction_attempts: int = 5,
    compaction_fanout: int = 5,
    compaction_tail_tokens: int = 12_000,
):
    return RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(base_url=None, api_key="test-key"),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(
            max_depth=max_depth,
            max_concurrent_subagents=max(4, max_depth),
            compaction=compaction,
            summarize_at_tokens=summarize_at_tokens,
            max_compaction_attempts=max_compaction_attempts,
            compaction_fanout=compaction_fanout,
            compaction_tail_tokens=compaction_tail_tokens,
        ),
    )


def _block(tier: int, start: int, end: int) -> Block:
    return Block(
        tier=tier,
        branches=(start, end),
        messages=(start * 10, end * 10 - 1),
        windows=(start, end - 1),
        turns=(start * 4, end * 4 - 1),
        summary=f"t{tier}:{start}-{end}",
        request_id=f"req-{tier}-{start}",
    )


def test_staircase_layout_decomposes_branches_in_base_fanout():
    staircase = Staircase(2)
    layouts = []
    for branch in range(9):
        staircase.add_branch(_block(1, branch, branch + 1))
        while pending := staircase.unsealed():
            for tier, start, end in pending:
                staircase.seal(_block(tier, start, end))
        layouts.append([block.key for block in staircase.segments()])

    assert layouts == [
        [(1, 0, 1)],
        [(2, 0, 2)],
        [(2, 0, 2), (1, 2, 3)],
        [(3, 0, 4)],
        [(3, 0, 4), (1, 4, 5)],
        [(3, 0, 4), (2, 4, 6)],
        [(3, 0, 4), (2, 4, 6), (1, 6, 7)],
        [(4, 0, 8)],
        [(4, 0, 8), (1, 8, 9)],
    ]
    top = staircase.blocks[(4, 0, 8)]
    assert staircase.children(4, 0, 8) == [
        staircase.blocks[(3, 0, 4)],
        staircase.blocks[(3, 4, 8)],
    ]
    assert top.header() == (
        "[tier 4 | branches 0-7 | windows 0-7 | messages 0-79 | turns 0-31]"
    )
    assert staircase.blocks[(1, 8, 9)].header() == (
        "[tier 1 | branch 8 | window 8 | messages 80-89 | turns 32-35]"
    )


def test_staircase_shows_children_of_a_missing_rollup():
    staircase = Staircase(3)
    for branch in range(4):
        staircase.add_branch(_block(1, branch, branch + 1))

    assert staircase.unsealed() == [(2, 0, 3)]
    assert [block.key for block in staircase.segments()] == [
        (1, 0, 1),
        (1, 1, 2),
        (1, 2, 3),
        (1, 3, 4),
    ]
    staircase.seal(_block(2, 0, 3))
    assert staircase.unsealed() == []
    assert [block.key for block in staircase.segments()] == [(2, 0, 3), (1, 3, 4)]
    assert staircase.render().startswith(
        "[tier 2 | branches 0-2 | windows 0-2 | messages 0-29 | turns 0-11]\nt2:0-3\n\n"
        "[tier 1 | branch 3"
    )


async def test_compaction_attempt_limit_is_configurable(session):
    client = _ScriptedClient(
        [
            _response(DummyMessage(tool_calls=[DummyToolCall("ipython", {})])),
            _response(DummyMessage(tool_calls=[DummyToolCall("ipython", {})])),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(max_compaction_attempts=2),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    session.replace_context(messages, reason="start")

    try:
        with pytest.raises(CompactionFailed, match="after 2 attempts"):
            await engine._compact_branch(messages, turn=0)
    finally:
        engine.close()

    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "finish_reason", ["length", "content_filter", "tool_calls", None]
)
@pytest.mark.parametrize("recovers", [False, True])
async def test_compaction_requires_normal_termination(session, finish_reason, recovers):
    rejected = _response(
        DummyMessage(content="unfinished summary"), finish_reason=finish_reason
    )
    client = _ScriptedClient(
        [
            rejected,
            _response(DummyMessage(content="complete summary"))
            if recovers
            else rejected,
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(max_compaction_attempts=2, compaction_tail_tokens=0),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "progress"},
    ]
    session.replace_context(messages, reason="start")
    original = deepcopy(session.messages)
    try:
        if recovers:
            await engine._compact_branch(messages, turn=0)
            assert len(session.messages) == 2
            assert "complete summary" in session.messages[1]["content"]
            assert "unfinished summary" not in session.messages[1]["content"]
            assert engine._metrics.num_compactions == 1
        else:
            with pytest.raises(CompactionFailed, match="after 2 attempts"):
                await engine._compact_branch(messages, turn=0)
            assert session.messages == original
            assert engine._metrics.num_compactions == 0
        assert len(client.calls) == 2
        assert client.calls[0]["messages"] == client.calls[1]["messages"]
    finally:
        engine.close()


@pytest.mark.parametrize("content", [None, "", " \n\t"])
async def test_compaction_retries_reasoning_without_final_content(session, content):
    message = DummyMessage(content=content)
    message.reasoning_content = "unfinished reasoning"
    client = _ScriptedClient(
        [
            _response(message),
            _response(DummyMessage(content="complete summary")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(max_compaction_attempts=2),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    session.replace_context(messages, reason="start")
    try:
        await engine._compact_branch(messages, turn=0)
        assert "complete summary" in session.messages[1]["content"]
        assert "unfinished reasoning" not in session.messages[1]["content"]
        assert len(client.calls) == 2
        assert engine._metrics.num_compactions == 1
    finally:
        engine.close()


async def test_tool_result_overflow_compacts_and_retries(session):
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(
                    tool_calls=[
                        DummyToolCall("ipython", {"code": "print('x' * 40000)"})
                    ]
                )
            ),
            _overflow(),
            _overflow(),
            _response(DummyMessage(content="summary")),
            _response(
                DummyMessage(
                    tool_calls=[
                        DummyToolCall(
                            "ipython",
                            {
                                "code": (
                                    "from rlm import history\n"
                                    "h = await history()\n"
                                    "print(h.user_messages()[0]['content'])\n"
                                    "print(len(next(r['message']['content'] for r in h.events if r['type'] == 'tool_result')))\n"
                                    "print(h.windows[0].messages[1]['content'])"
                                )
                            },
                        )
                    ]
                )
            ),
            _response(DummyMessage(content="done")),
        ],
        max_model_len=32_768,
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
    )

    try:
        result = await engine.run("produce a large tool result")
    finally:
        engine.close()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1
    assert client.calls[3]["tool_choice"] == "none"
    retry_messages = client.calls[4]["messages"]
    assert len(retry_messages) == 2
    assert retry_messages[1]["content"].startswith(
        '<runtime_event kind="compaction">\n' + STAIRCASE_FRAMING
    )
    assert str(session.dir / "messages.jsonl") in retry_messages[1]["content"]
    records = [
        json.loads(line)
        for line in (session.dir / "messages.jsonl").read_text().splitlines()
    ]
    tool_records = [entry for entry in records if entry["type"] == "tool_result"]
    assert tool_records[0]["message"] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "x" * 40000 + "\n",
    }
    assistant = next(entry for entry in records if entry["type"] == "assistant")
    assert (
        assistant["message"]["tool_calls"][0]["id"]
        == tool_records[0]["message"]["tool_call_id"]
    )
    assert "40001" in tool_records[1]["content"]
    assert "produce a large tool result" in tool_records[1]["content"]
    assert (
        next(entry for entry in records if entry["type"] == "system")["message"]["role"]
        == "system"
    )
    assert not any(entry["type"].startswith("checkpoint_") for entry in records)
    ledger = await history(session.dir)
    assert ledger.windows[0].messages == client.calls[1]["messages"]
    assert ledger.windows[1].messages[:2] == retry_messages
    assert ledger.windows[1].messages == session.messages
    assert ledger.windows[0].message_indices[0] == ledger.windows[1].message_indices[0]


async def test_overflow_recovers_without_discovered_threshold(session):
    """Reactive compaction works when the provider advertises no context window."""
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(
                    tool_calls=[DummyToolCall("ipython", {"code": "print('x' * 4000)"})]
                )
            ),
            _overflow(),
            _overflow(),
            _response(DummyMessage(content="summary")),
            _response(DummyMessage(content="done")),
        ],
        max_model_len=None,
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
    )

    try:
        result = await engine.run("produce a large tool result")
    finally:
        engine.close()

    assert engine.summarize_at_tokens is None
    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1


async def test_context_overflow_propagates_when_compaction_is_disabled(session):
    client = _ScriptedClient([_overflow()])
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(compaction=False),
    )

    try:
        with pytest.raises(BadRequestError):
            await engine.run("overflow without compaction")
    finally:
        engine.close()

    assert engine._metrics.num_compactions == 0


async def test_decode_context_limit_uses_discovered_threshold(session):
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(
                    tool_calls=[DummyToolCall("ipython", {"code": "print('ready')"})]
                )
            ),
            _response(
                DummyMessage(content="partial decode"),
                prompt_tokens=95,
                completion_tokens=5,
                finish_reason="length",
            ),
            _response(DummyMessage(content="summary")),
            _response(DummyMessage(content="done")),
        ],
        max_model_len=112,
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
    )

    try:
        result = await engine.run("fill the remaining context")
    finally:
        engine.close()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1
    checkpoint_messages = client.calls[2]["messages"]
    assert all(
        message.get("content") != "partial decode" for message in checkpoint_messages
    )


async def test_subagent_recovers_from_context_overflow(tmp_path):
    clients: list[_ScriptedClient] = []
    engines: list[RLMEngine] = []

    def engine_factory(**kwargs: Any) -> RLMEngine:
        client = _ScriptedClient(
            [
                _response(
                    DummyMessage(
                        tool_calls=[DummyToolCall("ipython", {"code": "print('hi')"})]
                    )
                ),
                _overflow(),
                _response(DummyMessage(content="summary")),
                _response(DummyMessage(content="child done")),
            ],
            max_model_len=32_768,
        )
        engine = RLMEngine(client=client, **kwargs)  # type: ignore[arg-type]
        clients.append(client)
        engines.append(engine)
        return engine

    config = _config(max_depth=1)
    root = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=root,
        runtime_config=config,
        cwd=str(tmp_path),
        engine_factory=engine_factory,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    try:
        child = supervisor._spawn(
            supervisor._caller(endpoint.capability, scope),
            scope,
            "recover in the child",
            None,
            False,
        )
        await child.done.wait()
        result = child.result
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        root.close()

    assert result.answer == "child done"
    assert engines[0].depth == 1
    assert engines[0]._metrics.num_compactions == 1
    assert len(clients[0].calls) == 4


async def test_checkpoint_fallback_preserves_kernel_warning_without_large_output(
    session,
):
    client = _ScriptedClient(
        [_overflow(), _response(DummyMessage(content="recovered summary"))]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        runtime_config=_config(max_compaction_attempts=2),
    )
    session.replace_context(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "oversized output" * 5000},
        ],
        reason="start",
    )
    engine._last_good = 2
    warning = "Supervisor: IPython restarted; variables were lost. Do not replay the interrupted cell."
    engine._repl = SimpleNamespace(take_recovery_notices=lambda: [warning])
    engine._deliver_kernel_notices()
    engine._repl = None
    try:
        await engine._compact_branch(session.messages, turn=0)
        assert len(client.calls) == 2
        fallback = client.calls[1]["messages"]
        assert warning in fallback[-1]["content"]
        assert all("oversized output" not in m.get("content", "") for m in fallback)
        assert engine._metrics.num_compactions == 1
    finally:
        engine.close()


def _work(prompt_tokens: int = 200) -> DummyResponse:
    return _response(
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "print(1)"})]),
        prompt_tokens=prompt_tokens,
    )


async def test_staircase_rolls_up_branch_summaries(session):
    client = _ScriptedClient(
        [
            _work(),
            _response(DummyMessage(content="branch one")),
            _work(),
            _response(DummyMessage(content="branch two")),
            _response(DummyMessage(content="branches one and two")),
            _work(),
            _response(DummyMessage(content="branch three")),
            _response(DummyMessage(content="done")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(summarize_at_tokens=100, compaction_fanout=2),
    )
    try:
        result = await engine.run("task")
    finally:
        engine.close()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 3
    assert engine._own_turns == 4

    rollup_call = client.calls[4]
    assert rollup_call["tool_choice"] == "none"
    assert rollup_call["messages"][0]["role"] == "system"
    assert rollup_call["messages"][1]["content"].startswith(ROLLUP_PROMPT)
    assert "[tier 1 | branch 0" in rollup_call["messages"][1]["content"]
    assert "branch two" in rollup_call["messages"][1]["content"]

    staircase = session.messages[1]["content"]
    assert staircase.startswith(
        '<runtime_event kind="compaction">\n' + STAIRCASE_FRAMING
    )
    assert staircase.index("[tier 2 | branches 0-1") < staircase.index(
        "[tier 1 | branch 2"
    )
    assert "branches one and two" in staircase
    assert "branch three" in staircase
    assert "branch one\n" not in staircase
    assert "h.blocks" in staircase

    ledger = await history(session.dir)
    blocks = ledger.blocks
    # A rollup is sealed, and logged, before the compaction that triggered it.
    assert [(b["tier"], *b["branches"]) for b in blocks] == [
        (1, 0, 1),
        (2, 0, 2),
        (1, 1, 2),
        (1, 2, 3),
    ]
    first, rollup, second, third = blocks
    assert first["messages"][0] == 1
    assert first["windows"] == [0, 0]
    assert first["turns"] == [0, 0]
    # The compaction message sits between consecutive branches.
    assert second["messages"][0] == first["messages"][1] + 2
    assert third["messages"][0] == second["messages"][1] + 2
    assert third["turns"] == [2, 2]
    assert rollup["messages"] == [first["messages"][0], second["messages"][1]]
    assert rollup["windows"] == [0, 1]
    assert rollup["summary"] == "branches one and two"
    for block in blocks:
        assert block["request_id"]
    rollup_record = next(e for e in ledger.events if e["type"] == "rollup")
    assert rollup_record["usage"] == {"prompt_tokens": 1, "completion_tokens": 1}
    compactions = [e for e in ledger.events if e["type"] == "compaction"]
    assert [c["rollups_sealed"] for c in compactions] == [[], [[2, 0, 2]], []]

    edges = engine._semantic_edges.snapshot()["edges"]
    compaction_edges = [e for e in edges if e["type"] == "compaction"]
    assert len(compaction_edges) == 3
    assert compaction_edges[1]["source_request_id"] == second["request_id"]
    rollup_edges = [e for e in edges if e["target_request_id"] == rollup["request_id"]]
    assert [e["type"] for e in rollup_edges] == ["compaction_attempt"]
    assert (
        rollup_edges[0]["source_request_id"] == compaction_edges[0]["target_request_id"]
    )


async def test_failed_rollup_keeps_finer_blocks(session):
    client = _ScriptedClient(
        [
            _work(),
            _response(DummyMessage(content="branch one")),
            _work(),
            _response(DummyMessage(content="branch two")),
            _response(DummyMessage(tool_calls=[DummyToolCall("ipython", {})])),
            _response(DummyMessage(tool_calls=[DummyToolCall("ipython", {})])),
            _response(DummyMessage(content="done")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(
            summarize_at_tokens=100, compaction_fanout=2, max_compaction_attempts=2
        ),
    )
    try:
        result = await engine.run("task")
    finally:
        engine.close()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 2
    staircase = session.messages[1]["content"]
    assert staircase.index("[tier 1 | branch 0") < staircase.index("[tier 1 | branch 1")
    assert "branch one" in staircase and "branch two" in staircase
    assert engine._staircase.unsealed() == [(2, 0, 2)]
    ledger = await history(session.dir)
    assert [b["tier"] for b in ledger.blocks] == [1, 1]
    assert ledger.events[-1]["type"] != "rollup"


async def test_rollup_calls_share_one_budget_per_compaction(session):
    refused = _response(DummyMessage(tool_calls=[DummyToolCall("ipython", {})]))
    client = _ScriptedClient(
        [
            _work(),
            _response(DummyMessage(content="branch one")),
            _work(),
            _response(DummyMessage(content="branch two")),
            refused,
            refused,
            _work(),
            _response(DummyMessage(content="branch three")),
            refused,
            refused,
            _work(),
            _response(DummyMessage(content="branch four")),
            _response(DummyMessage(content="branches three and four")),
            refused,
            _response(DummyMessage(content="done")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(
            summarize_at_tokens=100, compaction_fanout=2, max_compaction_attempts=2
        ),
    )
    try:
        result = await engine.run("task")
    finally:
        engine.close()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 4
    # The fresh range is rolled up before the one that keeps failing, and the
    # second call of the budget goes to the retry.
    assert [block.key for block in engine._staircase.segments()] == [
        (1, 0, 1),
        (1, 1, 2),
        (2, 2, 4),
    ]
    assert engine._staircase.unsealed() == [(2, 0, 2)]
    rollup_calls = [
        call
        for call in client.calls
        if call["messages"][-1]["content"].startswith(ROLLUP_PROMPT)
    ]
    assert len(rollup_calls) == 6
    ledger = await history(session.dir)
    compactions = [e for e in ledger.events if e["type"] == "compaction"]
    assert [c["rollups_sealed"] for c in compactions] == [[], [], [], [[2, 2, 4]]]


async def test_rollback_restores_staircase(session):
    client = _ScriptedClient(
        [
            _work(),
            _response(DummyMessage(content="branch one")),
            _response(DummyMessage(content="done")),
            _work(),
            _response(DummyMessage(content="branch two")),
            RuntimeError("boom"),
            _work(),
            _response(DummyMessage(content="branch two again")),
            _response(DummyMessage(content="done again")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(summarize_at_tokens=100, compaction_fanout=5),
    )
    try:
        await engine.prompt("task")
        assert engine._staircase.branch_count == 1
        branch_start = (engine._branch_first_index, engine._branch_first_window)
        with pytest.raises(RuntimeError, match="boom"):
            await engine.prompt("more")
        assert engine._staircase.branch_count == 1
        assert (engine._branch_first_index, engine._branch_first_window) == branch_start
        result = await engine.prompt("more again")
    finally:
        await engine.aclose()

    assert result.answer == "done again"
    blocks = [block.key for block in engine._staircase.segments()]
    assert blocks == [(1, 0, 1), (1, 1, 2)]
    assert engine._staircase.blocks[(1, 1, 2)].summary == "branch two again"
    ledger = await history(session.dir)
    assert [(b["tier"], *b["branches"], b["summary"]) for b in ledger.blocks] == [
        (1, 0, 1, "branch one"),
        (1, 1, 2, "branch two again"),
    ]
    assert sum(1 for e in ledger.events if e["type"] == "compaction") == 3


def test_select_tail_fits_budget_and_opens_with_a_call():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a" * 40},
        {"role": "tool", "content": "r" * 40},
        {"role": "assistant", "content": "b" * 40},
        {"role": "tool", "content": "q" * 40},
    ]
    indices = [0, 1, 2, 3, 4, 5]

    assert select_tail(messages, indices, 1, 10_000) == 2
    assert select_tail(messages, indices, 1, 40) == 4
    assert select_tail(messages, indices, 1, 0) == 6
    # The branch's first message is never part of the tail.
    assert select_tail(messages, indices, 4, 10_000) == 6
    assert select_tail(messages, indices, 3, 10_000) == 4


@pytest.mark.parametrize("tail_tokens", [12_000, 0])
async def test_compaction_keeps_recent_messages_verbatim(session, tail_tokens):
    client = _ScriptedClient(
        [
            _work(prompt_tokens=50_000),
            _response(DummyMessage(content="branch one")),
            _work(prompt_tokens=50_000),
            _response(DummyMessage(content="branch two")),
            _response(DummyMessage(content="done")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(
            summarize_at_tokens=40_000, compaction_tail_tokens=tail_tokens
        ),
    )
    try:
        result = await engine.run("task")
    finally:
        engine.close()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 2
    ledger = await history(session.dir)
    seeds = [
        e["message_indices"] for e in ledger.events if e["type"] == "context_window"
    ]
    compactions = [e for e in ledger.events if e["type"] == "compaction"]
    first, second = ledger.blocks
    # Ledger indices: 0 system, 1 task, 2 call, 3 result, 4 staircase, 5 call,
    # 6 result, 7 staircase.
    assert seeds[0] == []
    if tail_tokens:
        # The assistant call and its result keep their indices in the new window.
        assert seeds[1] == [0, 4, 2, 3]
        assert [m["role"] for m in ledger.windows[1].messages[:4]] == [
            "system",
            "user",
            "assistant",
            "tool",
        ]
        assert compactions[0]["tail_message_indices"] == [2, 3]
        assert compactions[0]["dropped_chars"] == len("task")
        assert first["messages"] == [1, 1]
        assert second["messages"] == [2, 4]
        assert seeds[2] == [0, 7, 5, 6]
        assert (
            "remain in context verbatim" in client.calls[1]["messages"][-1]["content"]
        )
    else:
        assert seeds[1] == [0, 4]
        assert compactions[0]["tail_message_indices"] == []
        assert first["messages"] == [1, 3]
        assert second["messages"] == [5, 6]
        assert seeds[2] == [0, 7]
        assert (
            "remain in context verbatim"
            not in client.calls[1]["messages"][-1]["content"]
        )
