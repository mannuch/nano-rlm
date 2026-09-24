"""Labeled refinement scenarios: a fixture workspace, optional seeded harness entries,
steering prompts that walk a live session into a situation, and what a refinement
triggered afterwards should decide, focus on and write.

Each positive family has a nearby negative, so the gate is scored in both directions.
Steering is phrased the way a user or a tool would, not as a statement of the lesson.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ANSWER_FORMAT = "End your reply with a line `ANSWER: <value>`."


Scope = str
"""A harness store: ``local`` (the session's own) or ``global`` (shared across
sessions). Every seed and edit check names the store it belongs to."""


@dataclass(frozen=True)
class Seed:
    kind: str
    id: str
    title: str
    content: str
    scope: Scope = "local"


@dataclass(frozen=True)
class Created:
    """An applied create in ``scope`` of one of ``kinds`` whose title or content
    matches."""

    kinds: tuple[str, ...]
    pattern: str
    scope: Scope = "local"


@dataclass(frozen=True)
class Changed:
    """An applied update or delete of the entry ``kind:id`` in ``scope``."""

    kind: str
    id: str
    scope: Scope = "local"


@dataclass(frozen=True)
class NoCreate:
    """No applied create in ``scope`` matches ``pattern`` (any create when empty)."""

    pattern: str = ""
    scope: Scope = "local"


@dataclass(frozen=True)
class AtMostOne:
    """At most one entry in the final ``scope`` store matches ``pattern``: no
    duplicate."""

    pattern: str
    scope: Scope = "local"


EditCheck = Created | Changed | NoCreate | AtMostOne


@dataclass(frozen=True)
class Probe:
    prompt: str
    answer: str


@dataclass
class Scenario:
    id: str
    family: str
    files: dict[str, str]
    steer: list[str]
    expect_refine: bool
    lesson_steer: int | None = None
    """Index into ``steer`` of the prompt that carries the lesson, when a prompt does
    (a lesson in tool output has none)."""
    lesson: str | None = None
    """Pattern an entry recording the lesson matches. The agent may record it itself
    during the steering turns; the pass should then decline."""
    expect_signals: set[str] = field(default_factory=set)
    """At least one of these gate signals should fire."""
    expect_home: set[str] = field(default_factory=set)
    expect_entries: dict[str, str] = field(default_factory=dict)
    """Entry ref -> ``covers`` or ``wrong``: the focus call should flag it so."""
    expect_captured: set[str] = field(default_factory=set)
    """Lessons the focus call should veto as already recorded."""
    seed: list[Seed] = field(default_factory=list)
    scope: Scope = "local"
    """The store the triggered refinement writes: a ``global`` pass is a host
    refinement with ``global: true``."""
    edit_checks: list[EditCheck] = field(default_factory=list)
    probe: Probe | None = None

    def build(self, workspace: Path) -> None:
        for rel, text in self.files.items():
            path = workspace / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


NOTES = "alpha\n\nbeta\ngamma\n\n\ndelta\nepsilon\n\nzeta\n"
TODO = "buy milk\n\n\nfix bike\n\ncall mom\nwrite report\n\n"

CONFIG_PY = """import json


def load_config(path):
    with open(path) as f:
        return json.load(f)


def merge(base, extra):
    return {**base, **extra}
"""
STORE_PY = """def save_config(path, data):
    with open(path, "w") as f:
        f.write(repr(data))
"""

REPORT_PY = """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("section")
parser.add_argument("--root")
args = parser.parse_args()
if not args.root:
    raise RuntimeError("report.py could not locate its data directory")
counts = json.loads((Path(args.root) / "counts.json").read_text())
print(f"{args.section}: {counts[args.section]}")
"""
COUNTS = '{"summary": 41, "errors": 7, "warnings": 12}\n'

FETCH_PY = """from pathlib import Path

marker = Path(".fetched")
if not marker.exists():
    marker.write_text("1")
    raise TimeoutError("upstream did not answer within 5s")
print("value: 314")
"""

MODULE_A = '"""Parse CSV rows into dicts keyed by header."""\n\ndef parse(rows):\n    header, *body = rows\n    return [dict(zip(header, r)) for r in body]\n'
MODULE_B = '"""Retry a callable with exponential backoff."""\n\nimport time\n\ndef retry(fn, attempts=3):\n    for i in range(attempts):\n        try:\n            return fn()\n        except Exception:\n            time.sleep(2 ** i)\n    return fn()\n'

PROJECT = {
    "pyproject.toml": """[project]
name = "app"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
""",
    "src/app/__init__.py": "",
    "src/app/config.py": CONFIG_PY,
    "src/app/testing.py": '"""Pytest plugin: shared fixtures for the app tests."""\n',
    "tests/conftest.py": 'pytest_plugins = ["app.testing"]\n',
    "tests/test_config.py": """from app.config import merge


def test_merge_prefers_extra():
    assert merge({"a": 1}, {"a": 2}) == {"a": 2}
""",
}
"""A small but complete project, so a question about running its tests is answerable
inside the workspace instead of sending the agent looking elsewhere."""

REPORT_V2_PY = """import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "var" / "data" / "counts.json"
section = sys.argv[1]
print(f"{section}: {json.loads(DATA.read_text())[section]}")
"""
REPORT_V2 = {
    "tools/report.py": REPORT_V2_PY,
    "var/data/counts.json": COUNTS,
    "README.md": "Report tooling. The counts are regenerated nightly.\n",
}

BLANK_LINES = r"blank|empty line|non-empty"


def _non_blank(text: str) -> str:
    return str(sum(1 for line in text.splitlines() if line.strip()))


SCENARIOS = [
    Scenario(
        id="user_correction",
        family="user correction",
        files={"notes.txt": NOTES, "todo.txt": TODO},
        steer=[
            f"How many lines does notes.txt have? {ANSWER_FORMAT}",
            "That's not how we count here: in this project a line count never "
            "includes blank lines. Recount notes.txt. " + ANSWER_FORMAT,
        ],
        lesson_steer=1,
        lesson=BLANK_LINES,
        expect_refine=True,
        expect_signals={"user_correction", "durable_fact"},
        expect_home={"memory", "prompt"},
        edit_checks=[Created(("memory", "prompt"), BLANK_LINES)],
        probe=Probe(
            f"How many lines does todo.txt have? {ANSWER_FORMAT}", _non_blank(TODO)
        ),
    ),
    Scenario(
        id="already_captured",
        family="already captured (-)",
        files={"notes.txt": NOTES},
        seed=[
            Seed(
                "memory",
                "line-counts",
                "Line counts",
                "In this project line counts exclude blank lines.",
            )
        ],
        steer=[
            f"How many lines does notes.txt have? {ANSWER_FORMAT}",
            "Right, and as always here, blank lines don't count. Thanks.",
        ],
        lesson_steer=1,
        expect_refine=False,
        expect_captured={"user_correction", "durable_fact"},
        edit_checks=[AtMostOne(BLANK_LINES)],
    ),
    Scenario(
        id="durable_fact",
        family="durable fact",
        files={"src/app/config.py": CONFIG_PY, "src/app/store.py": STORE_PY},
        steer=[
            "Quick context before questions: in this repository we always write paths "
            "relative to the `src/` directory, e.g. `app/x.py`, never `src/app/x.py`. "
            f"Which file defines `load_config`? {ANSWER_FORMAT}",
        ],
        lesson_steer=0,
        lesson=r"relative",
        expect_refine=True,
        expect_signals={"durable_fact", "user_correction"},
        expect_home={"memory", "prompt"},
        edit_checks=[Created(("memory", "prompt"), r"src/|relative")],
        probe=Probe(
            f"Which file defines `save_config`? {ANSWER_FORMAT}", "app/store.py"
        ),
    ),
    Scenario(
        id="one_off",
        family="one-off noise (-)",
        files={"src/app/config.py": CONFIG_PY, "src/app/store.py": STORE_PY},
        steer=[
            f"How many functions does src/app/config.py define? {ANSWER_FORMAT}",
            f"Which file under src/ imports `json`? {ANSWER_FORMAT}",
        ],
        expect_refine=False,
        edit_checks=[NoCreate()],
    ),
    Scenario(
        id="repeated_failure",
        family="repeated failure",
        files={"tools/report.py": REPORT_PY, "data/counts.json": COUNTS},
        steer=[
            f"Run `python tools/report.py summary` and tell me the number. {ANSWER_FORMAT}",
            f"Now run `python tools/report.py errors` and tell me the number. {ANSWER_FORMAT}",
        ],
        lesson=r"--root",
        expect_refine=True,
        expect_signals={"repeated_failure", "reusable_tactic", "durable_fact"},
        expect_home={"memory", "prompt", "skill"},
        edit_checks=[Created(("memory", "prompt", "skill"), r"--root")],
        probe=Probe(
            f"Run `python tools/report.py warnings` and tell me the number. {ANSWER_FORMAT}",
            "12",
        ),
    ),
    Scenario(
        id="transient_failure",
        family="transient failure (-)",
        files={"tools/fetch.py": FETCH_PY},
        steer=[
            f"Run `python tools/fetch.py` and tell me the value it prints. {ANSWER_FORMAT}",
        ],
        expect_refine=False,
        edit_checks=[NoCreate(r"timeout|retry|fetch")],
    ),
    Scenario(
        id="delegation_role",
        family="delegation role",
        files={"lib/a.py": MODULE_A, "lib/b.py": MODULE_B},
        steer=[
            "Spawn a child agent to read lib/a.py and summarize what it does in one "
            "sentence, then give me the child's summary.",
            "Do the same for lib/b.py: a child agent reads it and summarizes it in one "
            "sentence.",
        ],
        lesson_steer=1,
        lesson=r"summar",
        expect_refine=True,
        expect_signals={"delegation_role"},
        expect_home={"subagent"},
        edit_checks=[Created(("subagent",), r"summar")],
    ),
    Scenario(
        id="stale_entry",
        family="stale entry",
        files=PROJECT,
        seed=[
            Seed(
                "memory",
                "test-command",
                "Test command",
                "Run the test suite with `python -m pytest`.",
            )
        ],
        steer=[
            "What command should I use to run the tests here?",
            "No, `python -m pytest` breaks in this repo because of the plugin setup. "
            "The command is `uv run pytest -p no:cacheprovider`.",
        ],
        lesson_steer=1,
        lesson=r"uv run pytest",
        expect_refine=True,
        expect_signals={"harness_contradicted", "user_correction"},
        expect_entries={"local:test-command": "wrong"},
        edit_checks=[
            Changed("memory", "test-command"),
            AtMostOne(r"uv run pytest"),
        ],
    ),
    Scenario(
        id="global_stale_entry",
        family="global stale entry (global pass)",
        files={"src/app/config.py": CONFIG_PY},
        seed=[
            Seed(
                "memory",
                "answer-style",
                "Answer style",
                "The user wants answers of one or two sentences, never more.",
                scope="global",
            )
        ],
        steer=[
            f"What does `merge` in src/app/config.py do? {ANSWER_FORMAT}",
            "That's too terse. Across all my projects, not just this one, I now want "
            "detailed answers that explain the reasoning behind them.",
        ],
        lesson_steer=1,
        lesson=r"detail|reasoning",
        scope="global",
        expect_refine=True,
        expect_signals={"harness_contradicted", "user_correction", "durable_fact"},
        expect_entries={"global:answer-style": "wrong"},
        edit_checks=[
            Changed("memory", "answer-style", scope="global"),
            AtMostOne(r"sentence|detail|reasoning|terse", scope="global"),
        ],
    ),
    Scenario(
        id="global_override",
        family="contradicted global entry (local pass)",
        files=PROJECT,
        seed=[
            Seed(
                "memory",
                "test-command",
                "Test command",
                "Run the test suite with `python -m pytest`.",
                scope="global",
            )
        ],
        steer=[
            "What command should I use to run the tests here?",
            "Not in this repo: `python -m pytest` breaks here because of the plugin "
            "setup. Use `uv run pytest -p no:cacheprovider` in this project.",
        ],
        lesson_steer=1,
        lesson=r"uv run pytest",
        expect_refine=True,
        expect_signals={"harness_contradicted", "user_correction"},
        expect_entries={"global:test-command": "wrong"},
        edit_checks=[
            Created(("memory", "prompt"), r"uv run pytest"),
            NoCreate(scope="global"),
        ],
    ),
    Scenario(
        id="session_fact_global",
        family="session-only fact (global pass, -)",
        files={"notes.txt": NOTES},
        steer=[
            "For this debugging session only, the staging server is on port 6543. "
            f"How many lines does notes.txt have? {ANSWER_FORMAT}",
        ],
        scope="global",
        expect_refine=False,
        edit_checks=[NoCreate(scope="global")],
    ),
    Scenario(
        id="stale_path",
        family="stale entry seen in tool output",
        files=REPORT_V2,
        seed=[
            Seed(
                "memory",
                "report-data",
                "Report data",
                "The report counts live in data/counts.json; read that file for counts.",
            )
        ],
        steer=[
            f"How many errors are in the report counts? {ANSWER_FORMAT}",
            f"And how many warnings? {ANSWER_FORMAT}",
        ],
        lesson=r"var/data",
        expect_refine=True,
        expect_signals={"harness_contradicted", "repeated_failure", "durable_fact"},
        expect_entries={"local:report-data": "wrong"},
        edit_checks=[
            Changed("memory", "report-data"),
            AtMostOne(r"var/data|data/counts"),
        ],
    ),
    Scenario(
        id="consistent_entry",
        family="consistent entry (-)",
        files=REPORT_V2,
        seed=[
            Seed(
                "memory",
                "report-data",
                "Report data",
                "The report counts live in var/data/counts.json; read that file for "
                "counts.",
            )
        ],
        steer=[
            f"How many errors are in the report counts? {ANSWER_FORMAT}",
            f"And how many warnings? {ANSWER_FORMAT}",
        ],
        expect_refine=False,
        expect_captured={"durable_fact", "reusable_tactic"},
        edit_checks=[NoCreate(r"counts|var/data"), AtMostOne(r"var/data")],
    ),
]

BY_ID = {s.id: s for s in SCENARIOS}
assert len(BY_ID) == len(SCENARIOS), "scenario ids must be unique"


def matches(pattern: str, *texts: str | None) -> bool:
    return any(re.search(pattern, t or "", re.IGNORECASE) for t in texts)
