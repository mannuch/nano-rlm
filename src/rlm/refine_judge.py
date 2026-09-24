"""TypeSafe System One judge for continual-harness refinement reviews.

A review is two ``system_one`` calls over a compact evidence state built in code (the
whole conversation does not fit Jev's 32k-token state budget):

1. gate: one yes/no signal per kind of lesson a refinement should act on, plus one
   per harness entry the pass could fix: is it contradicted by the conversation? (An
   entry of the store the pass writes is updated or deleted; in a local pass, a
   read-only global or ancestor entry is overridden locally.) Anything at or above
   the threshold fires; nothing firing declines the review.
2. focus: asked only about the lessons that fired, stated as premises in the state.
   Per lesson: is it already recorded (a veto) and which harness kind should hold it.
   Per entry of the store the pass writes: does it cover a lesson. Per turn: is it
   direct evidence for a lesson.

Questions are worded for the pass's scope: a local pass serves later tasks in this
session, a global pass future sessions.

Code turns the answers into a verdict and deterministic instructions for the planning
call, which the task model still makes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, NoulCriteria, Question

from rlm.config import RefineJudgeConfig
from rlm.harness import HarnessScope, HarnessView, compact
from rlm.provenance import AGENT_INPUT, RUNTIME_EVENT
from rlm.refinement import BLOCK_SUMMARY_CHARS, RefinementResult, history_for_prompt
from rlm.staircase import Block

EVIDENCE_CHARS = 60_000
"""Budget for the rendered evidence; about 15k tokens, well inside Jev's 32k."""
TASK_CHARS = 2_000
TURN_TEXT_CHARS = 600
TURN_CODE_CHARS = 400
ENTRY_CONTENT_CHARS = 240
MAX_ENTRIES = 30
MAX_ENTRY_QUESTIONS = 20
MAX_TURN_QUESTIONS = 40
EVIDENCE_QUOTES = 3
QUOTE_CHARS = 160

_EXCEPTION_RE = re.compile(
    r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))\b", re.MULTILINE
)

LESSON_SIGNALS = (
    "repeated_failure",
    "reusable_tactic",
    "delegation_role",
    "durable_fact",
    "user_correction",
)
CONTRADICTED = "harness_contradicted"
"""Fired when any entry's ``wrong_<k>`` gate question fires."""
GATE_SIGNALS = (*LESSON_SIGNALS, CONTRADICTED)

HORIZONS = {
    "local": "later work in this session",
    "global": "tasks in future sessions",
}
"""Who a pass's edits serve, by the scope of the store it writes: ``{horizon}`` in the
question and lesson texts below."""
HORIZON_NOTES = {
    "local": " Later work in this session is the rest of the current task, which may go "
    "on after older turns are summarized away, and any later tasks.",
    "global": "",
}
"""``{horizon_note}`` in the gate questions: a local pass may serve a session of one
task, where what helps is what the rest of that task needs again."""

LESSONS = {
    "repeated_failure": "the same error or failed approach happened more than once",
    "reusable_tactic": "a workspace-specific technique that worked and will be needed "
    "again in {horizon}",
    "delegation_role": "the same kind of subtask was delegated to child agents repeatedly",
    "durable_fact": "a project fact or user preference that will be needed again in "
    "{horizon}",
    "user_correction": "a user message corrected the assistant or redirected its work",
}

_GATE_QUESTIONS: dict[str, tuple[str, str, str]] = {
    "repeated_failure": (
        "Do the messages in `turns` show the same error, or the same failed approach, "
        "happening more than once? `error_counts` counts each exception name raised in "
        "the tool outputs of `turns`.",
        "The same failure recurs: the same exception, wrong command or dead end appears "
        "in two or more separate attempts.",
        "Each failure happened once, the failures are unrelated, or nothing failed.",
    ),
    "reusable_tactic": (
        "Do the messages in `turns` show a technique, command or procedure specific to "
        "this workspace or task that worked and that will be needed again in "
        "{horizon}?{horizon_note}",
        "The assistant had to discover how to get something done here: a required flag, "
        "an entry point, a workaround, or a sequence of steps that the obvious approach "
        "missed.",
        "Only general methods a capable assistant already uses on any codebase, such as "
        "reading files, searching with grep or parsing code with ast; nothing that "
        "worked generalizes beyond one answer; or it served only a step that is already "
        "finished, and {horizon} would not use it again.",
    ),
    "delegation_role": (
        "Do the messages in `turns` show the assistant delegating the same kind of "
        "subtask to child agents (for example by calling rlm.agent.spawn) more than "
        "once?",
        "Two or more child agents were given the same role or kind of subtask.",
        "No delegation, a single delegation, or delegations of unrelated subtasks.",
    ),
    "durable_fact": (
        "Do the messages in `turns` establish a fact about the project or workspace, or "
        "a preference of the user, that will be needed again in {horizon}?"
        "{horizon_note}",
        "The user stated a convention, preference or constraint, or the conversation "
        "uncovered a fact that is easy to get wrong, such as a required flag or a "
        "non-standard location.",
        "Only answers to the questions asked and facts that are quick to look up again, "
        "such as what a file contains or where something is defined; or facts that "
        "served only a step that is already finished, and {horizon} would not use them "
        "again.",
    ),
    "user_correction": (
        'Does a message in `turns` with role "user" correct the assistant or tell it to '
        "work differently from now on?",
        "The user says the assistant was wrong, rejects its approach, or states a rule "
        "it should follow.",
        "User messages only ask questions or give new tasks.",
    ),
}
"""Lesson signal -> ``(instructions, true, false)``: the Noul's question, then the
``NoulCriteria`` descriptions of its yes and no outcomes. ``{horizon}`` and
``{horizon_note}`` are filled from ``HORIZONS`` and ``HORIZON_NOTES``."""

HOME_KINDS = {
    "memory": "A memory: a durable fact, decision, failure, preference or outcome.",
    "prompt": "A prompt note: a narrow behavioural rule for how the assistant works.",
    "skill": "A skill: a repeatable procedure exposed as an existing importable "
    "Python module.",
    "subagent": "A subagent spec: a reusable delegation role for child agents.",
}


@dataclass
class JudgeVerdict:
    """The judge's answer to one review.

    ``gate_decision`` is what the judge decides on its own; ``should_refine`` is the
    decision it returns, which is always True for a pass the judge does not gate.
    """

    should_refine: bool
    gate_decision: bool
    rationale: str
    instructions: str | None
    gate: dict[str, float]
    fired: list[str]
    focus: dict[str, Any] | None = None
    evidence_turns: list[int] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    """Refs of the evidence's harness entries: ``covers_<k>``/``wrong_<k>`` index it."""
    usage: dict[str, dict[str, int | None]] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """The ledger form, stored under ``judge`` on the pass's ledger record."""
        return {
            "gate_decision": self.gate_decision,
            "rationale": self.rationale,
            "instructions": self.instructions,
            "gate": self.gate,
            "fired": self.fired,
            "focus": self.focus,
            "evidence_turns": self.evidence_turns,
            "entries": self.entries,
            "usage": self.usage,
        }


# --- evidence ----------------------------------------------------------------


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def _role(message: dict) -> str:
    role = message.get("role", "")
    if role == "user":
        content = _text(message.get("content", "")).lstrip()
        if content.startswith(f"<{RUNTIME_EVENT}"):
            return "runtime"
        if content.startswith(f"<{AGENT_INPUT}"):
            return "parent"
    return role


def _code(message: dict) -> str | None:
    parts = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or ""
        try:
            code = json.loads(arguments).get("code")
        except (json.JSONDecodeError, AttributeError):
            code = None
        parts.append(
            code if isinstance(code, str) else f"{function.get('name')}({arguments})"
        )
    return "\n".join(parts) or None


def _exceptions(text: str) -> list[str]:
    return _EXCEPTION_RE.findall(text)


def _turn(index: int, message: dict) -> dict[str, Any]:
    role = _role(message)
    text = _text(message.get("content") or "")
    turn: dict[str, Any] = {
        "index": index,
        "role": role,
        "text": compact(text, TURN_TEXT_CHARS),
    }
    if code := _code(message):
        turn["code"] = compact(code, TURN_CODE_CHARS)
    if role == "tool" and (raised := _exceptions(text)):
        turn["error"] = raised[-1]
    return turn


def _size(value: Any) -> int:
    return len(json.dumps(value, default=str))


def build_evidence(
    *,
    task: str,
    trigger: str,
    messages: list[tuple[int, dict]],
    view: HarnessView,
    history: list[RefinementResult],
    scope: HarnessScope = "local",
    blocks: list[Block] | None = None,
    max_chars: int = EVIDENCE_CHARS,
) -> dict[str, Any]:
    """The judge's state: the conversation since the last review as short turns, the
    compaction blocks under review, and the harness entries a refinement could touch.

    ``scope`` is the store the pass writes and ``history`` that store's refinements.
    Entries of that store come first, so the entry cap drops read-only ones first.
    Turns are trimmed newest first to fit ``max_chars``; user turns are kept ahead of
    the rest because corrections are a primary signal.
    """
    turns = [_turn(i, m) for i, m in messages if m.get("role") != "system"]
    errors = Counter(t["error"] for t in turns if "error" in t)
    entries = [
        {
            "ref": f"{layer}:{entry.id}",
            "kind": entry.kind,
            "title": entry.title,
            "content": compact(entry.content, ENTRY_CONTENT_CHARS),
        }
        for layer, entry in sorted(view.entries(), key=lambda item: item[0] != scope)
        if entry.kind != "episode"
    ][:MAX_ENTRIES]
    evidence: dict[str, Any] = {
        "task": compact(task, TASK_CHARS),
        "trigger": trigger,
        "scope": scope,
        "turns": [],
        "error_counts": dict(errors),
        "harness_entries": entries,
        "recent_refinements": history_for_prompt(history).splitlines(),
    }
    if blocks:
        evidence["compaction_blocks"] = [
            {"block": b.header(), "summary": compact(b.summary, BLOCK_SUMMARY_CHARS)}
            for b in blocks
        ]
    budget = max_chars - _size(evidence)
    kept: set[int] = set()
    for turn in [
        *(t for t in reversed(turns) if t["role"] == "user"),
        *(t for t in reversed(turns) if t["role"] != "user"),
    ]:
        cost = _size(turn) + 2
        if cost <= budget:
            kept.add(turn["index"])
            budget -= cost
    evidence["turns"] = [t for t in turns if t["index"] in kept]
    return evidence


# --- questions ---------------------------------------------------------------


def _noul(instructions: str, true: str, false: str) -> Noul:
    return Noul(
        instructions=instructions, criteria=NoulCriteria(true=true, false=false)
    )


def gate_questions(evidence: dict[str, Any]) -> dict[str, Question]:
    """Call 1: one Noul per gate signal. The contradiction check is asked only when
    there are entries to contradict."""
    scope = evidence["scope"]
    questions: dict[str, Question] = {
        signal: _noul(
            *(
                text.format(horizon=HORIZONS[scope], horizon_note=HORIZON_NOTES[scope])
                for text in _GATE_QUESTIONS[signal]
            )
        )
        for signal in LESSON_SIGNALS
    }
    for k, _ in _contradictable(evidence):
        questions[f"wrong_{k}"] = _noul(
            f"Do the messages in `turns` contradict `harness_entries[{k}]` or show "
            "that it is outdated?",
            "The conversation shows this entry's fact, rule or procedure no longer "
            "holds.",
            "The conversation is consistent with this entry or unrelated to it.",
        )
    return questions


def decide_gate(answers: dict[str, float], threshold: float) -> list[str]:
    """The fired signals, in ``GATE_SIGNALS`` order."""
    fired = [s for s in LESSON_SIGNALS if answers.get(s, 0.0) >= threshold]
    if any(p >= threshold for k, p in answers.items() if k.startswith("wrong_")):
        fired.append(CONTRADICTED)
    return fired


def focus_state(evidence: dict[str, Any], fired: list[str]) -> dict[str, Any]:
    """Call 2's state: the evidence plus the fired lessons stated as premises. The
    gate's probabilities stay out: the premises are what call 2 reasons from."""
    return {
        **evidence,
        "gate": {
            "fired": [
                {"id": s, "lesson": lesson(s, evidence["scope"])}
                for s in fired
                if s in LESSONS
            ],
            "harness_contradicted": CONTRADICTED in fired,
        },
    }


def lesson(signal: str, scope: HarnessScope) -> str:
    return LESSONS[signal].format(horizon=HORIZONS[scope])


def _editable(evidence: dict[str, Any]) -> list[tuple[int, dict]]:
    """Entries of the store the pass writes (entry layers are named like scopes)."""
    return [
        (k, entry)
        for k, entry in enumerate(evidence["harness_entries"])
        if entry["ref"].startswith(f"{evidence['scope']}:")
    ][:MAX_ENTRY_QUESTIONS]


def _overridable(evidence: dict[str, Any]) -> list[tuple[int, dict]]:
    """Read-only entries a local pass can override with a local entry when they are
    contradicted: ancestor and global ones. A global pass has none."""
    if evidence["scope"] != "local":
        return []
    room = MAX_ENTRY_QUESTIONS - len(_editable(evidence))
    return [
        (k, entry)
        for k, entry in enumerate(evidence["harness_entries"])
        if not entry["ref"].startswith("local:")
    ][: max(0, room)]


def _contradictable(evidence: dict[str, Any]) -> list[tuple[int, dict]]:
    """The entries whose contradiction the gate asks about: those the pass can edit,
    then those a local pass can override."""
    return [*_editable(evidence), *_overridable(evidence)]


def _turn_slots(evidence: dict[str, Any]) -> list[tuple[int, dict]]:
    turns = list(enumerate(evidence["turns"]))
    return turns[-MAX_TURN_QUESTIONS:]


def focus_questions(state: dict[str, Any], fired: list[str]) -> dict[str, Question]:
    """Call 2, built only for what fired. Question ids are code-side keys:
    ``captured_<signal>``, ``home_<signal>``, ``covers_<k>`` (index into
    ``harness_entries``) and ``turn_<j>`` (index into ``turns``). The gate's
    ``wrong_<k>`` questions use the same entry index."""
    lessons = [s for s in fired if s in LESSONS]
    questions: dict[str, Question] = {}
    for i, signal in enumerate(lessons):
        subject = f"the lesson in `turns` described by `gate.fired[{i}].lesson`"
        questions[f"captured_{signal}"] = _noul(
            f"Consider {subject}. Is that specific lesson already recorded in an entry "
            "of `harness_entries` or in `recent_refinements`?",
            "An existing entry or past refinement already states the same fact, rule "
            "or procedure.",
            "No existing entry states it; recording it would add something new.",
        )
        questions[f"home_{signal}"] = Choice(
            instructions=f"Where should {subject} be recorded so it helps in "
            f"{HORIZONS[state['scope']]}? Pick the smallest component that fits.",
            criteria=HOME_KINDS,
        )
    for k, _ in _editable(state) if lessons else []:
        questions[f"covers_{k}"] = _noul(
            f"Is `harness_entries[{k}]` about the same topic as a lesson listed in "
            "`gate.fired`, as it appears in `turns`?",
            "The entry addresses the same fact, rule, procedure or role, so the "
            "lesson belongs in it.",
            "The entry is about something else.",
        )
    if lessons:
        for j, _ in _turn_slots(state):
            questions[f"turn_{j}"] = _noul(
                f"Is `turns[{j}]` direct evidence for a lesson listed in `gate.fired`?",
                "This message itself shows the failure, correction, fact, tactic or "
                "delegation.",
                "This message is background or unrelated to the lessons.",
            )
    return questions


def _quote(turn: dict[str, Any]) -> str:
    return compact(turn.get("text") or turn.get("code") or "", QUOTE_CHARS)


def decide_focus(
    fired: list[str],
    nouls: dict[str, float],
    choices: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    config: RefineJudgeConfig,
) -> tuple[bool, str, str | None, list[int]]:
    """``(gate_decision, rationale, instructions, evidence_turn_indices)`` from both
    calls' Noul answers (``nouls`` holds the gate's ``wrong_<k>`` and call 2's).

    A lesson recorded already is dropped; with no lesson left and no contradicted
    entry the judge declines. Instructions name each lesson's home kind (one kind when
    the choice is confident, else the top two), the entries of the target store that
    cover a lesson or are contradicted, the read-only entries a local pass should
    override, and quotes of the strongest evidence turns.
    """
    entries = evidence["harness_entries"]
    lessons = [s for s in fired if s in LESSONS]
    kept = [
        s for s in lessons if nouls.get(f"captured_{s}", 0.0) < config.veto_threshold
    ]
    captured = [s for s in lessons if s not in kept]
    scope = evidence["scope"]

    def flagged(prefix: str, slots: list[tuple[int, dict]]) -> list[str]:
        return [
            entries[k]["ref"]
            for k, _ in slots
            if nouls.get(f"{prefix}_{k}", 0.0) >= config.threshold
        ]

    covers = flagged("covers", _editable(evidence))
    wrong = flagged("wrong", _editable(evidence))
    overrides = flagged("wrong", _overridable(evidence))
    ranked = sorted(
        (
            (p, j)
            for j, _ in _turn_slots(evidence)
            if (p := nouls.get(f"turn_{j}", 0.0)) >= config.threshold
        ),
        reverse=True,
    )[:EVIDENCE_QUOTES]
    turns = [evidence["turns"][j] for _, j in sorted(ranked, key=lambda x: x[1])]

    rationale_parts = [f"fired: {', '.join(fired)}"]
    if captured:
        rationale_parts.append(f"already recorded: {', '.join(captured)}")
    if wrong or overrides:
        rationale_parts.append(f"contradicted: {', '.join([*wrong, *overrides])}")
    decision = bool(kept or wrong or overrides)
    if not decision:
        return False, "; ".join(rationale_parts), None, []

    lines = []
    for signal in kept:
        answer = choices.get(f"home_{signal}")
        if answer is None:
            home = "the smallest fitting component"
        elif answer["confidence"] >= config.home_confidence:
            home = f"a {answer['choice']}"
        else:
            top = sorted(answer["probabilities"], key=answer["probabilities"].get)
            home = f"a {top[-1]} or a {top[-2]}"
        lines.append(f"- Lesson: {lesson(signal, scope)}. Record it as {home}.")
    if covers and kept:
        lines.append(
            f"- Existing entries on the same topic: {', '.join(covers)}. Update them "
            "rather than creating duplicates."
        )
    for ref in wrong:
        lines.append(
            f"- Entry {ref} is contradicted by the conversation: update or delete it."
        )
    for ref in overrides:
        lines.append(
            f"- Entry {ref} is contradicted by the conversation but read-only here: "
            "create a local entry that overrides it."
        )
    if turns:
        quoted = "; ".join(f'"{_quote(t)}"' for t in turns)
        lines.append(f"- Evidence (quoted from the conversation): {quoted}")
    lines.append(
        "Keep edits small; propose other kinds only if the conversation clearly "
        "justifies them."
    )
    instructions = "Focus from the refinement review:\n" + "\n".join(lines)
    return True, "; ".join(rationale_parts), instructions, [t["index"] for t in turns]


# --- client --------------------------------------------------------------------


def _usage(response: Any) -> dict[str, int | None]:
    return {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


class RefineJudge:
    """Runs the two-call review against TypeSafe. ``force=True`` is for passes the
    judge does not gate (focus, shadow): call 1 only selects the lessons for call 2
    and cannot decline."""

    def __init__(
        self, config: RefineJudgeConfig, client: AsyncTypeSafeClient | None = None
    ):
        self.config = config
        self._client = client or AsyncTypeSafeClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def review(
        self, evidence: dict[str, Any], *, force: bool = False
    ) -> JudgeVerdict:
        config = self.config
        response = await self._client.system_one(
            evidence, gate_questions(evidence), model=config.model
        )
        gate = {k: a.noul for k, a in response.nouls.items()}
        usage = {"gate": _usage(response)}
        fired = decide_gate(gate, config.threshold)
        if not fired:
            top = max(gate, key=gate.get)
            return JudgeVerdict(
                should_refine=force,
                gate_decision=False,
                rationale=f"no signal reached {config.threshold}: top {top} "
                f"({gate[top]:.2f})",
                instructions=None,
                gate=gate,
                fired=[],
                entries=[e["ref"] for e in evidence["harness_entries"]],
                usage=usage,
            )

        state = focus_state(evidence, fired)
        questions = focus_questions(state, fired)
        nouls: dict[str, float] = {}
        choices: dict[str, dict[str, Any]] = {}
        # A contradicted entry with no lesson has nothing left to ask.
        if questions:
            response = await self._client.system_one(
                state, questions, model=config.model
            )
            usage["focus"] = _usage(response)
            nouls = {k: a.noul for k, a in response.nouls.items()}
            choices = {
                k: {
                    "choice": a.choice,
                    "confidence": a.confidence,
                    "probabilities": dict(a.probabilities),
                }
                for k, a in response.choices.items()
            }
        decision, rationale, instructions, turns = decide_focus(
            fired, {**gate, **nouls}, choices, evidence, config
        )
        return JudgeVerdict(
            should_refine=force or decision,
            gate_decision=decision,
            rationale=rationale,
            instructions=instructions,
            gate=gate,
            fired=fired,
            focus={**nouls, **choices},
            evidence_turns=turns,
            entries=[e["ref"] for e in evidence["harness_entries"]],
            usage=usage,
        )
