"""Which prompt texts the optimizer may rewrite, and what each rewrite must keep.

A rewrite that drops a required token, or that mentions what exists only in the
optimization sessions (scoring, checks, the answer format), is rejected before any rollout
is spent, with feedback naming the problem, so the reflection LM learns to keep the API
facts the runtime guide exists to convey and to state practices that generalize.
"""

from __future__ import annotations

import re

from rlm.prompt import DEFAULT_PROMPTS, REQUIRED_PROMPT_MARKERS

DEFAULT_COMPONENTS = (
    "task",
    "repl_doctrine",
    "delegation_doctrine",
    "checkpoint",
    "rollup",
    "staircase_framing",
)
"""``history`` stays opt-in: the reflection model's provider has answered every reflection
on it with ``finish_reason: content_filter`` and no text."""

REFERENCE_COMPONENTS = (
    "runtime_reference",
    "delegation_reference",
    "harness_api",
)

REQUIRED_TOKENS: dict[str, tuple[str, ...]] = {
    "runtime_reference": (
        "rlm.shell.run",
        "yield_after",
        "job.result()",
        "job.cancel()",
        "<runtime_event",
        "rlm.inbox",
        "rlm.watch",
    ),
    "delegation_reference": (
        "rlm.agent.spawn",
        "rlm.agent.list()",
        "rlm.agent.get(",
        "child.result()",
        "child.info()",
        "persistent=True",
        "child.send(",
        "child.steer(",
        "child.wait(",
        "child.cancel()",
        "rlm.watch.agent(",
    ),
    "history": (
        "from rlm import history",
        "hist.messages",
        "hist.windows",
        "hist.user_messages()",
        "hist.blocks",
        "hist.expand(",
        "history(session_dir=",
    ),
    "harness_api": (
        "rlm.harness.harness()",
        "h.overview()",
        "h.search(",
        "h.list(",
        "h.get(",
        "h.create_memory(",
        "h.create_prompt_note(",
        "h.create_skill(",
        "h.create_subagent(",
        "help(h)",
        "global_=True",
        "rlm.refine.run(",
    ),
    "checkpoint": ("Do not call tools",),
    "rollup": ("Do not call tools",),
    "staircase_framing": ("oldest first",),
}

LEAK_PATTERNS: dict[str, str] = {
    r"ANSWER:": "the benchmark's answer-line format",
    r"\bscor(?:e|es|ed|ing)\b": "scoring",
    r"\bgrad(?:er|ers|ed|ing)\b": "grading",
    r"\bruntime checks?\b": "the runtime checks",
    r"\bQuestion \d": "question numbering",
}
"""What only the optimization sessions have: a rewrite that mentions them has learned
the benchmark instead of a working practice, and would ship that to real users."""

MAX_GROWTH = 2.0
MIN_ALLOWED_CHARS = 1_200
"""A rewrite may grow to twice the seed text or to this floor, whichever is larger."""
STATED_LIMIT_FRACTION = 0.85
"""The reflection prompt states a limit this far below the enforced one, since models
overshoot a character count they are asked to respect."""

ROLES: dict[str, str] = {
    "task": (
        "the opening line of the system prompt: one or two sentences saying what kind of "
        "agent this is and how it works; the runtime guide that follows documents every API"
    ),
    "repl_doctrine": (
        "one paragraph inserted into the 'Runtime and ownership' section right after the "
        "sentence that introduces the persistent IPython REPL; it states how to work with "
        "the REPL (orchestrate in Python, probe first, bind results to variables, keep cell "
        "output small); the API reference around it is fixed and must not be repeated"
    ),
    "delegation_doctrine": (
        "the opening paragraph of the 'Delegation' section: when to delegate to a child "
        "agent versus work inline; the spawn/handle API that follows is fixed"
    ),
    "runtime_reference": (
        "the 'Runtime and ownership' section: the API reference for the REPL, supervisor, "
        "shell jobs, inbox and subscriptions; it contains the line `<repl_doctrine>` where "
        "the doctrine paragraph is inserted"
    ),
    "delegation_reference": (
        "the 'Delegation' section: the API reference for spawning and controlling child "
        "agents; it contains the line `<delegation_doctrine>` where the doctrine is inserted"
    ),
    "history": "the 'Conversation history' section: how to read the session ledger",
    "harness_api": "the continual-harness API paragraph inside the harness block",
    "checkpoint": (
        "the instruction for the side call that summarizes the branch since the last "
        "compaction; the runtime appends notes about the pinned request and the verbatim tail"
    ),
    "rollup": (
        "the instruction for the side call that merges several consecutive block summaries "
        "into one; the runtime appends the blocks after it"
    ),
    "staircase_framing": (
        "the sentence(s) that open the compaction message the agent sees after its context "
        "was compacted, before the block summaries"
    ),
    "review": "the JSON-only side-call prompt deciding whether a refinement should run",
    "refine": "the planning prompt for a harness refinement",
}


def max_chars(name: str, seed: dict[str, str]) -> int:
    return int(max(MAX_GROWTH * len(seed.get(name, "")), MIN_ALLOWED_CHARS))


def required_tokens(name: str) -> tuple[str, ...]:
    return (*REQUIRED_TOKENS.get(name, ()), *REQUIRED_PROMPT_MARKERS.get(name, ()))


def reflection_template(name: str, seed: dict[str, str]) -> str:
    """The reflection prompt for one component; GEPA fills ``<curr_param>`` and
    ``<side_info>``."""
    tokens = required_tokens(name)
    keep = (
        "It must keep these exact substrings: "
        + ", ".join(f"`{t}`" for t in tokens)
        + ". "
        if tokens
        else ""
    )
    return (
        f"An agent runs with the following text as the `{name}` part of its prompts. "
        f"This text is {ROLES.get(name, 'one prompt component')}.\n\n"
        "```\n<curr_param>\n```\n\n"
        "Below are sessions the agent ran with this text: the questions it was asked, the "
        "cells it ran and answers it gave, and feedback with the scores and runtime checks.\n\n"
        "```\n<side_info>\n```\n\n"
        f"Write an improved `{name}` text that makes sessions like these go better. "
        f"It must be at most {int(max_chars(name, seed) * STATED_LIMIT_FRACTION)} characters "
        f"(the current text is {len(seed.get(name, ''))}); longer texts are rejected "
        f"without being tried. {keep}"
        "The text ships to real users whose tasks look nothing like these sessions, and "
        "the sessions' scoring, runtime checks, answer-line format and question numbering "
        "exist only here. So never mention scores, grading, checks, `ANSWER:` lines, "
        "numbered questions, or an output shape for answers; name the general working "
        "practice that would have produced the better behavior. Rewrites that mention "
        "them are rejected without being tried. Do not describe the specific repositories "
        "or questions above. Do not add API details that the surrounding prompt already "
        "documents. Provide only the new text, inside a single ``` block."
    )


def validate_candidate(
    candidate: dict[str, str], seed: dict[str, str]
) -> dict[str, str]:
    """Problems per component (empty when the candidate is acceptable)."""
    problems: dict[str, str] = {}
    for name, text in candidate.items():
        if name not in DEFAULT_PROMPTS:
            problems[name] = f"unknown component {name!r}"
            continue
        if not text.strip():
            problems[name] = "the rewritten text is empty"
            continue
        if len(text) > max_chars(name, seed):
            problems[name] = (
                f"the rewritten text is {len(text)} characters, more than the "
                f"{max_chars(name, seed)} allowed for {name}; keep it shorter"
            )
            continue
        missing = [token for token in required_tokens(name) if token not in text]
        if missing:
            problems[name] = (
                "the rewrite dropped text the runtime depends on: "
                + ", ".join(f"`{token}`" for token in missing)
                + "; keep these exact spellings"
            )
            continue
        leaked = [
            what
            for pattern, what in LEAK_PATTERNS.items()
            if re.search(pattern, text) and not re.search(pattern, seed.get(name, ""))
        ]
        if leaked:
            problems[name] = (
                "the rewrite refers to the optimization sessions ("
                + ", ".join(leaked)
                + "); state a working practice that holds for any task instead"
            )
    return problems
