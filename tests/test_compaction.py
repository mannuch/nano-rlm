from __future__ import annotations

import itertools
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
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
    CUT_REPLY_NOTE,
    PINNED_PROMPT_NOTE,
    ROLLUP_PROMPT,
    STAIRCASE_FRAMING,
    TOOL_OUTPUT_STUB,
    CompactionFailed,
    CompactionFormatViolation,
    is_context_overflow,
    omission_marker,
    prompt_pointer_note,
    trim_summary_input,
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


# Discovered context windows are cached per base URL for the whole process, and an
# id() can be reused once its object is freed, so each client gets a fresh number.
_client_numbers = itertools.count()


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
        self.base_url = f"http://scripted-{next(_client_numbers)}"

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
    compaction_prompt_tokens: int = 4_000,
    prompt_overrides: dict[str, str] | None = None,
    summary_budget_tokens: int = 24_576,
):
    return RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(base_url=None, api_key="test-key"),
        invocation=InvocationContext(),
        prompt_overrides=prompt_overrides or {},
        policy=ExecutionPolicy(
            max_depth=max_depth,
            max_concurrent_subagents=max(4, max_depth),
            compaction=compaction,
            summarize_at_tokens=summarize_at_tokens,
            max_compaction_attempts=max_compaction_attempts,
            compaction_fanout=compaction_fanout,
            compaction_tail_tokens=compaction_tail_tokens,
            compaction_prompt_tokens=compaction_prompt_tokens,
            summary_budget_tokens=summary_budget_tokens,
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


@pytest.mark.parametrize("attempts", [1, 2])
async def test_compaction_attempt_limit_is_configurable(session, attempts):
    client = _ScriptedClient(
        [
            _response(DummyMessage(tool_calls=[DummyToolCall("ipython", {})])),
            _response(DummyMessage(tool_calls=[DummyToolCall("ipython", {})])),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(max_compaction_attempts=attempts),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    session.replace_context(messages, reason="start")

    try:
        with pytest.raises(
            CompactionFormatViolation, match=f"after {attempts} attempts"
        ):
            await engine._compact_branch(messages, turn=0)
    finally:
        engine.close()

    assert len(client.calls) == attempts


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
@pytest.mark.parametrize("recovers", [False, True])
@pytest.mark.parametrize("threshold", [None, 8, 32])
async def test_compaction_requires_normal_termination(
    session, finish_reason, recovers, threshold
):
    rejected = _response(
        DummyMessage(content="unfinished summary"),
        finish_reason=finish_reason,
        prompt_tokens=10,
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
        {"role": "user", "content": "middle" * 100},
        {"role": "assistant", "content": "progress"},
    ]
    session.replace_context(messages, reason="start")
    original = deepcopy(session.messages)
    engine._last_good = 2
    engine.summarize_at_tokens = threshold
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
        if finish_reason == "length" and threshold in (None, 8):
            assert client.calls[1]["messages"] == [
                *original[:2],
                omission_marker(1),
                original[-1],
                client.calls[0]["messages"][-1],
            ]
        else:
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


@pytest.mark.parametrize("split", [False, True])
async def test_tool_result_overflow_compacts_and_retries(session, tmp_path, split):
    """Under split execution the kernel reads the worker's synced copy of the history,
    and the compaction notice names that copy."""
    config = _config()
    if split:
        config = config.model_copy(
            update={
                "execution_command": (sys.executable, "-m", "rlm.execution"),
                "builtin_tools": ("ipython",),
            }
        )
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
        runtime_config=config,
        cwd=str(tmp_path),
    )

    try:
        result = await engine.run("produce a large tool result")
    finally:
        await engine.aclose()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1
    assert client.calls[3]["tool_choice"] == "none"
    retry_messages = client.calls[4]["messages"]
    assert len(retry_messages) == 3
    # The task prompt is re-attached ahead of the staircase, keeping its index.
    assert retry_messages[1] == client.calls[1]["messages"][1]
    assert retry_messages[2]["content"].startswith(
        '<runtime_event kind="compaction">\n' + STAIRCASE_FRAMING
    )
    named = re.search(r"available in (\S+)\. ", retry_messages[2]["content"])
    assert named is not None
    assert (Path(named.group(1)) != session.dir / "messages.jsonl") is split
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
    assert ledger.windows[1].messages[:3] == retry_messages
    assert ledger.windows[1].messages == session.messages
    assert (
        ledger.windows[1].message_indices[:2] == ledger.windows[0].message_indices[:2]
    )
    compaction = next(entry for entry in records if entry["type"] == "compaction")
    assert compaction["pinned_prompt_index"] == 1
    assert compaction["summary_message_index"] == ledger.windows[1].message_indices[2]


async def test_overflow_recovers_without_discovered_threshold(session):
    """Reactive compaction works when the provider advertises no context window, and
    the summary request uses the configured checkpoint prompt."""
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
        runtime_config=_config(
            prompt_overrides={"checkpoint": "CUSTOM CHECKPOINT INSTRUCTIONS"}
        ),
    )

    try:
        result = await engine.run("produce a large tool result")
    finally:
        engine.close()

    assert engine.summarize_at_tokens is None
    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1
    summary_request = client.calls[3]["messages"][-1]["content"]
    assert summary_request.startswith("CUSTOM CHECKPOINT INSTRUCTIONS")
    assert engine.execution_snapshot()["limits"]["prompt_overrides"] == ["checkpoint"]


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


async def test_window_cut_reply_is_kept_and_compacted(session):
    """A reply that runs into the context window stays in the conversation, without the
    tool call it was cut off in, and the summary request carries it."""
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(
                    tool_calls=[DummyToolCall("ipython", {"code": "print('ready')"})]
                ),
                prompt_tokens=1_000,
                completion_tokens=50,
            ),
            _response(
                DummyMessage(
                    content="partial decode",
                    tool_calls=[
                        DummyToolCall("ipython", {"code": "print(1)"}, id="cut")
                    ],
                ),
                prompt_tokens=1_200,
                completion_tokens=2_800,
                finish_reason="length",
            ),
            _response(DummyMessage(content="summary")),
            _response(DummyMessage(content="done")),
        ],
        max_model_len=4_000,
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(summary_budget_tokens=1_000, max_compaction_attempts=1),
    )

    try:
        result = await engine.run("fill the remaining context")
    finally:
        engine.close()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1
    assert len(client.calls) == 4
    checkpoint = client.calls[2]["messages"]
    assert CUT_REPLY_NOTE in checkpoint[-1]["content"]
    assert checkpoint[-2] == {"role": "assistant", "content": "partial decode"}
    assert omission_marker(2) in checkpoint
    resumed = client.calls[3]["messages"]
    # No verbatim tail: the window ends on the staircase, never on the cut reply.
    assert [message["role"] for message in resumed] == ["system", "user", "user"]
    assert STAIRCASE_FRAMING in resumed[-1]["content"]
    records = [
        json.loads(line)
        for line in (session.dir / "messages.jsonl").read_text().splitlines()
    ]
    assert not any(
        entry["type"] == "tool_result" and entry["turn"] == 1 for entry in records
    )
    assert {"role": "assistant", "content": "partial decode"} in [
        entry["message"] for entry in records if entry["type"] == "assistant"
    ]


@pytest.mark.parametrize("prompt_tokens", [100, 1_500])
async def test_length_cut_below_the_window_ends_the_run(session, prompt_tokens):
    """A reply stopped by a completion cap is the answer, even past the threshold."""
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(content="capped answer"),
                prompt_tokens=prompt_tokens,
                completion_tokens=1_000,
                finish_reason="length",
            ),
        ],
        max_model_len=4_000,
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(summary_budget_tokens=1_000),
    )

    try:
        result = await engine.run("answer at length")
    finally:
        engine.close()

    assert engine.summarize_at_tokens == 2_000
    assert result.answer == "capped answer"
    assert engine._metrics.stop_reason == "done"
    assert engine._metrics.num_compactions == 0
    assert len(client.calls) == 1


def _exchanges(count: int, output_chars: int) -> list[dict]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for i in range(count):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "r" * 1000,
                    "tool_calls": [
                        {
                            "id": str(i),
                            "type": "function",
                            "function": {"name": "ipython", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": str(i), "content": "o" * output_chars},
            ]
        )
    return messages


@pytest.mark.parametrize("context_tokens", [1_000, 9_800])
async def test_summary_request_keeps_its_budget(session, context_tokens):
    """A summary request that would leave the summary less than its budget is trimmed
    before the first attempt, so a single attempt suffices."""
    client = _ScriptedClient([_response(DummyMessage(content="summary"))])
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(summary_budget_tokens=2_000, max_compaction_attempts=1),
    )
    engine.context_window = 10_000
    messages = _exchanges(4, 4_000)
    session.replace_context(messages, reason="start")
    original = deepcopy(messages)
    try:
        await engine._compact_branch(messages, turn=0, context_tokens=context_tokens)
    finally:
        engine.close()

    sent = client.calls[0]["messages"][:-1]
    assert engine._metrics.num_compactions == 1
    assert messages == original
    if context_tokens == 1_000:
        assert sent == original
    else:
        assert sent[3]["content"] == TOOL_OUTPUT_STUB
        assert sent[-2:] == original[-2:]
        assert len(sent) == len(original)


async def test_overflow_compacts_with_a_trimmed_request(session):
    client = _ScriptedClient(
        [
            _overflow(),
            _response(DummyMessage(content="summary")),
            _response(DummyMessage(content="done")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(
            summary_budget_tokens=2_000,
            max_compaction_attempts=1,
            compaction_tail_tokens=0,
        ),
    )
    engine.context_window = 10_000
    messages = _exchanges(4, 4_000)
    session.replace_context(messages, reason="start")
    try:
        response, _ = await engine._complete(session.messages, turn=0)
    finally:
        engine.close()

    assert response.choices[0].message.content == "done"
    assert client.calls[1]["messages"][3]["content"] == TOOL_OUTPUT_STUB
    assert len(client.calls[2]["messages"]) == 2


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


async def test_compaction_preserves_skills_announced_after_the_prompt(
    session, tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    monkeypatch.setattr("rlm.engine.SKILLS_ROOT", root)
    write_skill = (
        "from pathlib import Path\n"
        f"skill = Path({str(root)!r}) / 'greeter'\n"
        "skill.mkdir(parents=True)\n"
        "(skill / 'SKILL.md').write_text("
        "'---\\nname: greeter\\ndescription: Greet people by name\\n---\\n')\n"
    )
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(
                    tool_calls=[DummyToolCall("ipython", {"code": write_skill})]
                )
            ),
            _work(prompt_tokens=50_000),
            _response(DummyMessage(content="branch summary")),
            _response(DummyMessage(content="done")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(summarize_at_tokens=40_000),
    )
    try:
        result = await engine.run("task")
    finally:
        engine.close()

    line = f"- greeter: Greet people by name (file: {root / 'greeter' / 'SKILL.md'})"
    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1
    assert "- greeter:" not in client.calls[0]["messages"][0]["content"]
    assert line in client.calls[1]["messages"][-1]["content"]
    checkpoint = client.calls[2]["messages"][-1]["content"]
    assert (
        "Preserve these skills, added after the system prompt was built" in checkpoint
    )
    assert line in checkpoint
    # Announced once: the compacted window keeps the original notice in its verbatim
    # tail and gets no second one.
    final = [str(m.get("content")) for m in client.calls[3]["messages"]]
    assert sum('<runtime_event kind="skills">' in c for c in final) == 1


async def test_failed_turn_rolls_back_a_skill_notice_and_resends_it(
    session, tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    monkeypatch.setattr("rlm.engine.SKILLS_ROOT", root)
    write_skill = (
        "from pathlib import Path\n"
        f"skill = Path({str(root)!r}) / 'greeter'\n"
        "skill.mkdir(parents=True)\n"
        "(skill / 'SKILL.md').write_text('# greeter\\n')\n"
    )
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(
                    tool_calls=[DummyToolCall("ipython", {"code": write_skill})]
                )
            ),
            RuntimeError("model failed"),
            _response(DummyMessage(content="done")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
    )
    try:
        with pytest.raises(RuntimeError, match="model failed"):
            await engine.prompt("write a skill")
        result = await engine.prompt("again")
    finally:
        await engine.aclose()

    def notices(call: dict) -> list[str]:
        return [
            str(m.get("content"))
            for m in call["messages"]
            if '<runtime_event kind="skills">' in str(m.get("content"))
        ]

    line = f"- greeter: (file: {root / 'greeter' / 'SKILL.md'})"
    assert result.answer == "done"
    # The failed turn's notice is rolled back with its context, so the next turn
    # announces the skill again, once.
    assert [line in n for n in notices(client.calls[1])] == [True]
    assert all(
        "write a skill" not in str(m.get("content"))
        for m in client.calls[2]["messages"]
    )
    assert [line in n for n in notices(client.calls[2])] == [True]


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

    staircase = session.messages[2]["content"]
    assert staircase.startswith(
        '<runtime_event kind="compaction">\n' + STAIRCASE_FRAMING
    )
    assert staircase.index("[tier 2 | branches 0-1") < staircase.index(
        "[tier 1 | branch 2"
    )
    assert "branches one and two" in staircase
    assert "branch three" in staircase
    assert "branch one\n" not in staircase
    assert "hist.blocks" in staircase

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
    assert (
        ledger.expand(rollup)
        == ledger.messages[first["messages"][0] : second["messages"][1] + 1]
    )
    assert ledger.expand(0) == ledger.expand(first)
    assert ledger.expand(-1)[0] == ledger.messages[third["messages"][0]]
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
    staircase = session.messages[2]["content"]
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
        pinned = engine._pinned_prompt
        assert pinned is not None and pinned[1]["content"] == "task"
        with pytest.raises(RuntimeError, match="boom"):
            await engine.prompt("more")
        assert engine._staircase.branch_count == 1
        assert (engine._branch_first_index, engine._branch_first_window) == branch_start
        assert engine._pinned_prompt == pinned
        result = await engine.prompt("more again")
    finally:
        await engine.aclose()

    assert result.answer == "done again"
    assert engine._pinned_prompt[1]["content"] == "more again"
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
        # The task prompt, the assistant call and its result keep their indices in
        # the new window.
        assert seeds[1] == [0, 1, 4, 2, 3]
        assert [m["role"] for m in ledger.windows[1].messages[:5]] == [
            "system",
            "user",
            "user",
            "assistant",
            "tool",
        ]
        assert ledger.windows[1].messages[1] == ledger.windows[0].messages[1]
        assert compactions[0]["tail_message_indices"] == [2, 3]
        # The pinned prompt and the tail both stay, so nothing was dropped.
        assert compactions[0]["dropped_chars"] == 0
        assert first["messages"] == [1, 1]
        assert second["messages"] == [2, 4]
        assert seeds[2] == [0, 1, 7, 5, 6]
        assert (
            "remain in context verbatim" in client.calls[1]["messages"][-1]["content"]
        )
        assert "do not restate it" in client.calls[1]["messages"][-1]["content"]
    else:
        assert seeds[1] == [0, 1, 4]
        assert compactions[0]["tail_message_indices"] == []
        assert first["messages"] == [1, 3]
        assert second["messages"] == [5, 6]
        assert seeds[2] == [0, 1, 7]
        assert (
            "remain in context verbatim"
            not in client.calls[1]["messages"][-1]["content"]
        )


async def test_oversized_prompt_is_referenced_instead_of_pinned(session):
    client = _ScriptedClient(
        [
            _work(prompt_tokens=50_000),
            _response(DummyMessage(content="branch one")),
            _response(DummyMessage(content="done")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(
            summarize_at_tokens=40_000,
            compaction_tail_tokens=0,
            compaction_prompt_tokens=8,
        ),
    )
    try:
        result = await engine.run("a task far longer than eight tokens allow to pin")
    finally:
        engine.close()

    assert result.answer == "done"
    ledger = await history(session.dir)
    seeds = [
        e["message_indices"] for e in ledger.events if e["type"] == "context_window"
    ]
    assert seeds[1] == [0, 4]
    staircase = ledger.windows[1].messages[1]["content"]
    assert prompt_pointer_note(1) in staircase
    assert PINNED_PROMPT_NOTE not in staircase
    compaction = next(e for e in ledger.events if e["type"] == "compaction")
    assert compaction["pinned_prompt_index"] is None
    # The prompt left the window along with the call and its result.
    assert compaction["dropped_chars"] > len(
        "a task far longer than eight tokens allow to pin"
    )
    assert "do not restate it" not in client.calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("prompt_tokens", [129229, 129614])
async def test_context_full_summary_preserves_recent_exchange(session, prompt_tokens):
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(content="partial"),
                finish_reason="length",
                prompt_tokens=prompt_tokens,
            ),
            _response(DummyMessage(content="complete summary")),
        ]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        runtime_config=_config(max_compaction_attempts=2),
    )
    engine._active_tool_schemas = [
        {
            "type": "function",
            "function": {"name": "ipython", "parameters": {"type": "object"}},
        }
    ]
    engine.summarize_at_tokens = 114688
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task"},
        {"role": "user", "content": "old context " * 20000},
        {
            "role": "assistant",
            "content": "latest edit",
            "tool_calls": [
                {
                    "id": "edit",
                    "type": "function",
                    "function": {"name": "ipython", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "edit",
            "content": "file changed successfully",
        },
    ]
    session.replace_context(messages, reason="start")
    original = deepcopy(messages)
    try:
        await engine._compact_branch(messages, turn=0)
        retry = client.calls[1]["messages"]
        assert retry[:2] == original[:2]
        assert retry[2] == omission_marker(1)
        assert retry[3:5] == original[-2:]
        assert messages == original
        assert len(json.dumps(retry)) < len(json.dumps(client.calls[0]["messages"])) / 2
        assert client.calls[1]["tool_choice"] == "none"
    finally:
        engine.close()


def test_summary_trim_spends_outputs_then_reasoning_then_exchanges():
    messages = _exchanges(6, 1000)
    original = deepcopy(messages)
    tokens = len(json.dumps(messages)) // 4

    def assert_paired(trimmed: list[dict]) -> None:
        for i, message in enumerate(trimmed):
            if message["role"] == "tool":
                call = trimmed[i - 1]["tool_calls"][0]
                assert call["id"] == message["tool_call_id"]

    light = trim_summary_input(messages, context_tokens=tokens, remove_tokens=200)
    assert [m["content"] == TOOL_OUTPUT_STUB for m in light[3::2]] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert all(m.get("reasoning_content") for m in light[2::2])

    medium = trim_summary_input(messages, context_tokens=tokens, remove_tokens=1_260)
    assert all(m["content"] == TOOL_OUTPUT_STUB for m in medium[3:-2:2])
    assert [bool(m.get("reasoning_content")) for m in medium[2:-2:2]] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert all(m["tool_calls"] for m in medium[2::2])
    assert medium[-2:] == original[-2:]

    heavy = trim_summary_input(messages, context_tokens=tokens, remove_tokens=tokens)
    assert heavy[:3] == [*original[:2], omission_marker(10)]
    assert heavy[-1] == original[-1]
    assert heavy[-2]["tool_calls"] == original[-2]["tool_calls"]
    assert heavy[-2]["reasoning_content"].startswith("r")
    assert heavy[-2]["reasoning_content"].endswith("r")
    assert len(heavy[-2]["reasoning_content"]) < len(original[-2]["reasoning_content"])
    again = trim_summary_input(medium, context_tokens=tokens, remove_tokens=tokens)
    assert again == heavy
    for trimmed in (light, medium, heavy):
        assert_paired(trimmed)
    assert messages == original


def test_oversized_newest_reply_keeps_its_ends():
    reasoning = "H" * 20_000 + "M" * 200_000 + "T" * 70_000
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "reasoning_content": reasoning},
    ]
    trimmed = trim_summary_input(messages, context_tokens=80_000, remove_tokens=40_000)
    kept = trimmed[-1]["reasoning_content"]
    assert kept.startswith("H" * 16_384 + "\n[... ")
    assert kept.endswith("]\n" + "T" * 65_536)
    assert "M" not in kept
    again = trim_summary_input(trimmed, context_tokens=20_000, remove_tokens=10_000)
    assert len(again[-1]["reasoning_content"]) < len(kept)
    assert again[-1]["reasoning_content"].startswith("H")
    assert again[-1]["reasoning_content"].endswith("T")


@pytest.mark.parametrize("summary_budget", [4_096, 24_576])
@pytest.mark.parametrize("overflow", [False, True])
async def test_small_window_cut_reply_survives_summary_retries(
    session, summary_budget, overflow
):
    client = _ScriptedClient(
        [
            _overflow()
            if overflow
            else _response(
                DummyMessage(content="unfinished summary"),
                finish_reason="length",
                prompt_tokens=16_000,
                completion_tokens=384,
            ),
            _response(DummyMessage(content="complete summary")),
        ]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        runtime_config=_config(
            summary_budget_tokens=summary_budget, max_compaction_attempts=2
        ),
    )
    engine.context_window = 16_384
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "H" * 15_000 + "T" * 15_000,
            "reasoning_content": "R" * 15_000 + "S" * 15_000,
        },
    ]
    original = deepcopy(messages)
    session.replace_context(messages, reason="start")
    engine._last_good = 2
    try:
        assert engine._summary_budget == min(summary_budget, 8_192)
        await engine._compact_branch(
            messages, turn=0, context_tokens=16_384, cut_reply=True
        )
        for call in client.calls:
            checkpoint = call["messages"]
            assert checkpoint[:2] == original[:2]
            assert CUT_REPLY_NOTE in checkpoint[-1]["content"]
            reply = checkpoint[-2]
            assert reply["role"] == "assistant"
            assert reply["content"].startswith("H")
            assert reply["content"].endswith("T")
            assert reply["reasoning_content"].startswith("R")
            assert reply["reasoning_content"].endswith("S")
        first, retry = (call["messages"][:-1] for call in client.calls)
        assert (
            len(json.dumps(retry)) < len(json.dumps(first)) < len(json.dumps(original))
        )
        assert messages == original
        assert engine._metrics.num_compactions == 1
    finally:
        engine.close()


@pytest.mark.parametrize("overflow", [False, True])
async def test_unshrinkable_cut_reply_never_uses_an_older_snapshot(session, overflow):
    client = _ScriptedClient(
        [
            _overflow()
            if overflow
            else _response(
                DummyMessage(content="unfinished summary"),
                finish_reason="length",
                prompt_tokens=16_000,
                completion_tokens=384,
            ),
        ]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        runtime_config=_config(max_compaction_attempts=2),
    )
    engine.context_window = 16_384
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task" * 16_384},
        {"role": "assistant", "content": "partial reply"},
    ]
    session.replace_context(messages, reason="start")
    engine._last_good = 2
    try:
        with pytest.raises(CompactionFailed, match="cannot be shortened"):
            await engine._compact_branch(
                messages, turn=0, context_tokens=16_384, cut_reply=True
            )
        assert len(client.calls) == 1
        assert session.messages == messages
        assert engine._metrics.num_compactions == 0
    finally:
        engine.close()


async def test_unshrinkable_summary_input_fails_fast(session):
    """A retained beginning and end that alone overflow are not resent unchanged."""
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(content="partial"),
                finish_reason="length",
                prompt_tokens=200_000,
            ),
            _response(DummyMessage(content="never requested")),
        ]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        runtime_config=_config(max_compaction_attempts=3),
    )
    engine.summarize_at_tokens = 114688
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "x" * 400_000},
        {"role": "assistant", "content": "partial"},
    ]
    session.replace_context(messages, reason="start")
    engine._last_good = len(messages)
    try:
        with pytest.raises(CompactionFailed, match="cannot be shortened"):
            await engine._compact_branch(messages, turn=0)
        assert len(client.calls) == 1
    finally:
        engine.close()


async def test_hollowed_input_falls_back_to_last_good_snapshot(session):
    """A longer-by-count snapshot that fit when it was live is tried before failing."""
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(content="partial"),
                finish_reason="length",
                prompt_tokens=200_000,
            ),
            _response(
                DummyMessage(content="partial"),
                finish_reason="length",
                prompt_tokens=200_000,
            ),
            _response(DummyMessage(content="complete summary")),
        ]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        runtime_config=_config(max_compaction_attempts=4),
    )
    engine.summarize_at_tokens = 114688
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task" * 100_000},
        {"role": "assistant", "content": "step one"},
        {"role": "assistant", "content": "step two"},
        {"role": "assistant", "content": "partial"},
    ]
    session.replace_context(messages, reason="start")
    engine._last_good = 4
    try:
        await engine._compact_branch(messages, turn=0)
        hollowed = client.calls[1]["messages"]
        assert hollowed[2] == omission_marker(2)
        assert client.calls[2]["messages"][:-1] == messages[:4]
        assert len(client.calls) == 3
    finally:
        engine.close()
