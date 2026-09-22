"""One nano-rlm session per (candidate, task): run it, score it, digest its ledger.

The digest is what the reflection LM later reads, so it records what each prompt text
was responsible for: the cells the agent ran, every compaction block and rollup, cells
repeated after a compaction, and the outcome of each ``api`` question's ledger check.
"""

from __future__ import annotations

import asyncio
import json
import re
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlm.config import (
    ExecutionPolicy,
    HarnessConfig,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.history import History
from rlm.session import Session

from tasks import Question, Task, extract_answer, score_answer

TOKENS_PER_PENALTY_POINT = 100_000
"""Score loses 0.05 per this many total tokens, so ties break toward cheaper sessions."""

SNIPPET = 240


@dataclass
class RolloutSettings:
    model: str
    api_key: str
    base_url: str | None = None
    summarize_at_tokens: int = 10_000
    compaction_tail_tokens: int = 1_000
    compaction_fanout: int = 2
    max_depth: int = 1
    delegation_prompt: bool = True
    """Append the runtime's delegation guidance; it is fixed text, not a candidate."""
    max_total_tokens: int = 300_000
    exec_timeout: int = 60
    timeout_s: float = 900.0


@dataclass
class Cell:
    index: int
    """Ledger message index of the assistant message that issued the call."""
    code: str
    output: str
    error: bool
    after_compaction: bool


@dataclass
class Compaction:
    message_index: int
    """Index of the staircase message in the ledger."""
    header: str
    summary: str
    tail_messages: int
    pinned: bool
    rollups: list[dict[str, Any]] = field(default_factory=list)
    """Each sealed rollup: its header, text, and the child summaries it merged."""
    staircase_message: str = ""


@dataclass
class QuestionResult:
    kind: str
    check: str | None
    text: str
    expected: Any
    answer: str | None
    score: float
    check_ok: bool | None = None
    check_note: str = ""
    after_compaction: bool = False
    turns: int = 0


@dataclass
class Rollout:
    task_id: str
    session_dir: str
    score: float = 0.0
    correctness: float = 0.0
    results: list[QuestionResult] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    turns: int = 0
    error: str | None = None
    cells: list[Cell] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
    duplicate_cells: list[tuple[int, int]] = field(default_factory=list)
    """(earlier index, later index) pairs of identical cells across a compaction."""
    spawns: list[tuple[int, str]] = field(default_factory=list)
    """(ledger message index the spawn followed, child task prompt)."""
    final_answers: list[str] = field(default_factory=list)

    @property
    def compacted(self) -> bool:
        return bool(self.compactions)

    @property
    def rolled_up(self) -> bool:
        return any(c.rollups for c in self.compactions)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _config(candidate: dict[str, str], settings: RolloutSettings) -> RuntimeConfig:
    return RuntimeConfig(
        model=settings.model,
        provider=ProviderConfig(base_url=settings.base_url, api_key=settings.api_key),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(
            max_depth=settings.max_depth,
            delegation_prompt=settings.delegation_prompt,
            max_total_tokens=settings.max_total_tokens,
            exec_timeout=settings.exec_timeout,
            summarize_at_tokens=settings.summarize_at_tokens,
            compaction_tail_tokens=settings.compaction_tail_tokens,
            compaction_fanout=settings.compaction_fanout,
        ),
        harness=HarnessConfig(auto_refine=False),
        prompt_overrides=candidate,
    )


async def run_rollout(
    candidate: dict[str, str],
    task: Task,
    settings: RolloutSettings,
    session_root: Path,
) -> Rollout:
    """Run every question of ``task`` in one session; never raises."""
    session_dir = session_root / task.id / uuid.uuid4().hex[:8]
    rollout = Rollout(task_id=task.id, session_dir=str(session_dir))
    answers: list[tuple[Question, str | None, int]] = []
    engine: RLMEngine | None = None
    try:
        session = Session(session_dir=session_dir)
        engine = RLMEngine(
            cwd=task.cwd, session=session, runtime_config=_config(candidate, settings)
        )
        for question in task.questions:
            turns_before = engine._turn
            result = await asyncio.wait_for(
                engine.prompt(question.prompt), timeout=settings.timeout_s
            )
            answers.append((question, result.answer, turns_before))
            if engine.stop_reason not in (None, "done"):
                break
        rollout.prompt_tokens = engine._total_usage.prompt_tokens
        rollout.completion_tokens = engine._total_usage.completion_tokens
        rollout.turns = engine._turn
    except BaseException as error:  # noqa: BLE001 - GEPA requires a scored result per task
        rollout.error = (
            f"{type(error).__name__}: {error}\n{traceback.format_exc()[-1500:]}"
        )
    finally:
        if engine is not None:
            try:
                await asyncio.wait_for(engine.aclose(), timeout=120)
            except BaseException as error:  # noqa: BLE001
                rollout.error = (rollout.error or "") + f"\nclose failed: {error!r}"
    if (session_dir / "messages.jsonl").exists():
        try:
            _digest(rollout, session_dir)
            rollout.results = _score(task, answers, rollout, session_dir)
        except Exception as error:  # noqa: BLE001 - keep the answers' scores on a digest bug
            rollout.error = (rollout.error or "") + f"\ndigest failed: {error!r}"
            rollout.results = [
                QuestionResult(q.kind, q.check, q.text, q.answer, a, score_answer(q, a))
                for q, a, _ in answers
            ]
    else:
        rollout.results = [
            QuestionResult(q.kind, q.check, q.text, q.answer, None, 0.0)
            for q in task.questions
        ]
    scored = [r.score for r in rollout.results]
    rollout.correctness = sum(scored) / len(task.questions) if task.questions else 0.0
    penalty = (
        0.05
        * (rollout.prompt_tokens + rollout.completion_tokens)
        / TOKENS_PER_PENALTY_POINT
    )
    rollout.score = round(max(0.0, rollout.correctness - penalty), 4)
    rollout.final_answers = [answer or "" for _, answer, _ in answers]
    return rollout


# --- ledger digest -----------------------------------------------------------------


def _normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code).strip()


def _digest(rollout: Rollout, session_dir: Path) -> None:
    hist = History(session_dir)
    events = hist.events
    calls_by_id: dict[str, tuple[int, str]] = {}
    last_index = -1
    for event in events:
        kind = event["type"]
        if "message_index" in event:
            last_index = event["message_index"]
        if kind == "assistant":
            for call in event["message"].get("tool_calls") or []:
                if call["function"]["name"] == "ipython":
                    try:
                        code = json.loads(call["function"]["arguments"]).get("code", "")
                    except json.JSONDecodeError:
                        code = call["function"]["arguments"]
                    calls_by_id[call["id"]] = (event["message_index"], code)
        elif kind == "tool_result":
            call_id = event["message"].get("tool_call_id")
            if call_id in calls_by_id:
                index, code = calls_by_id.pop(call_id)
                content = event.get("content") or ""
                rollout.cells.append(
                    Cell(
                        index=index,
                        code=code,
                        output=content[:SNIPPET],
                        error="Traceback (most recent call last)" in content
                        or "Error" in content[:200],
                        after_compaction=False,
                    )
                )
        elif kind == "sub_spawn":
            rollout.spawns.append((last_index, event.get("prompt", "")[:SNIPPET]))
    for index, code in calls_by_id.values():
        rollout.cells.append(Cell(index, code, "(no result)", True, False))
    rollout.cells.sort(key=lambda c: c.index)

    rollups = [e for e in events if e["type"] == "rollup"]
    first_compaction = None
    for event in events:
        if event["type"] != "compaction":
            continue
        block = event["block"]
        staircase = (
            hist.messages[event["summary_message_index"]]["content"]
            if event.get("summary_message_index") is not None
            else ""
        )
        compaction = Compaction(
            message_index=event.get("summary_message_index") or 0,
            header=_header(block),
            summary=block["summary"],
            tail_messages=len(event.get("tail_message_indices") or []),
            pinned=event.get("pinned_prompt_index") is not None,
            staircase_message=staircase,
        )
        for key in event.get("rollups_sealed") or []:
            for rb in rollups:
                if [rb["tier"], *rb["branches"]] == list(key):
                    children = [
                        b
                        for b in hist.blocks
                        if b["tier"] == rb["tier"] - 1
                        and rb["branches"][0] <= b["branches"][0]
                        and b["branches"][1] <= rb["branches"][1]
                    ]
                    compaction.rollups.append(
                        {
                            "header": _header(rb),
                            "text": rb["summary"],
                            "children": [
                                f"{_header(b)}\n{b['summary']}" for b in children
                            ],
                        }
                    )
        rollout.compactions.append(compaction)
        if first_compaction is None:
            first_compaction = compaction.message_index

    if first_compaction is not None:
        before: dict[str, int] = {}
        for cell in rollout.cells:
            key = _normalize_code(cell.code)
            if cell.index < first_compaction:
                before.setdefault(key, cell.index)
            else:
                cell.after_compaction = True
                if key in before:
                    rollout.duplicate_cells.append((before[key], cell.index))


def _header(block: dict[str, Any]) -> str:
    branches = block["branches"]
    branch = (
        f"branch {branches[0]}"
        if branches[1] - branches[0] == 1
        else f"branches {branches[0]}-{branches[1] - 1}"
    )
    return f"[tier {block['tier']} | {branch} | messages {block['messages'][0]}-{block['messages'][1]}]"


# --- scoring, including ledger checks for api questions ----------------------------


def _score(
    task: Task,
    answers: list[tuple[Question, str | None, int]],
    rollout: Rollout,
    session_dir: Path,
) -> list[QuestionResult]:
    hist = History(session_dir)
    user_indices = [
        i
        for i, m in enumerate(hist.messages)
        if m.get("role") == "user"
        and not str(m.get("content", "")).startswith("<runtime_event")
    ]
    first_compaction = (
        rollout.compactions[0].message_index if rollout.compactions else None
    )
    results = []
    for position, question in enumerate(task.questions):
        answered = answers[position] if position < len(answers) else None
        answer_text = answered[1] if answered else None
        asked_at = (
            user_indices[position]
            if position < len(user_indices)
            else len(hist.messages)
        )
        result = QuestionResult(
            kind=question.kind,
            check=question.check,
            text=question.text,
            expected=question.answer,
            answer=extract_answer(answer_text or ""),
            score=0.0,
            after_compaction=first_compaction is not None
            and asked_at > first_compaction,
        )
        if answered is None:
            result.check_note = "session ended before this question"
            results.append(result)
            continue
        if question.check is not None:
            expected = _expected_for_api(question, task, hist, asked_at)
            question = Question(**{**asdict(question), "answer": expected})
            result.expected = expected
            ok, note = CHECKS[question.check](
                question, rollout, hist, session_dir, asked_at
            )
            result.check_ok, result.check_note = ok, note
            result.expected = question.answer
            result.score = score_answer(question, answer_text) if ok else 0.0
        else:
            result.score = score_answer(question, answer_text)
        results.append(result)
    return results


def _expected_for_api(
    question: Question, task: Task, hist: History, asked_at: int
) -> Any:
    if question.check == "history_cells":
        return sum(
            1
            for m in hist.messages[:asked_at]
            if m.get("role") == "assistant"
            for call in m.get("tool_calls") or []
            if call["function"]["name"] == "ipython"
        )
    if question.check == "history_expand":
        return " ".join(task.questions[0].text.split()[:6])
    return question.answer


def _cells_after(rollout: Rollout, asked_at: int) -> list[Cell]:
    return [c for c in rollout.cells if c.index > asked_at]


def _check_shell_exit_code(question, rollout, hist, session_dir, asked_at):
    cells = _cells_after(rollout, asked_at)
    marker = f"sys.exit({question.params['code']})"
    used = [c for c in cells if "rlm.shell.run" in c.code and marker in c.code]
    if not used:
        other = [c for c in cells if marker in c.code]
        how = (
            "via "
            + (
                "subprocess/os.system"
                if any("subprocess" in c.code or "os.system" in c.code for c in other)
                else "an unknown path"
            )
            if other
            else "never ran the command"
        )
        return False, f"no `rlm.shell.run` cell ran the command ({how})"
    return True, "ran the command through rlm.shell.run"


def _check_delegate_count(question, rollout, hist, session_dir, asked_at):
    spawned = [index for index, _ in rollout.spawns if index > asked_at]
    if not spawned:
        return (
            False,
            "no child agent was spawned (no `rlm.agent.spawn` cell / sub_spawn record after the question)",
        )
    return True, f"spawned {len(spawned)} child agent(s)"


def _check_harness_memory(question, rollout, hist, session_dir, asked_at):
    state = session_dir / "harness" / "harness_state.json"
    if not state.exists():
        return (
            False,
            "no local harness state was written (h.create_memory never called)",
        )
    entries = json.loads(state.read_text()).get("entries", {}).get("memory", {})
    titles = [e.get("title", "") for e in entries.values()]
    if question.params["title"] not in titles:
        return (
            False,
            f"memory titled {question.params['title']!r} not found; local memories: {titles}",
        )
    question.answer = len(entries)
    return True, f"memory recorded; local memory count {len(entries)}"


def _check_history_cells(question, rollout, hist, session_dir, asked_at):
    if not any("history" in c.code for c in _cells_after(rollout, asked_at)):
        return False, "no cell used the history API after the question"
    return True, "used the history API"


def _check_history_expand(question, rollout, hist, session_dir, asked_at):
    cells = _cells_after(rollout, asked_at)
    if not any("history" in c.code for c in cells):
        return False, "no cell used the history API after the question"
    return True, "used the history API"


CHECKS = {
    "shell_exit_code": _check_shell_exit_code,
    "delegate_count": _check_delegate_count,
    "harness_memory": _check_harness_memory,
    "history_cells": _check_history_cells,
    "history_expand": _check_history_expand,
}
