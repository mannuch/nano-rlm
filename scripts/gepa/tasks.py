"""Repo-grounded question sessions with programmatically computed answers.

A task is one session of questions about a checked-out repository. Repo questions have
answers computed here from the AST or the filesystem; ``api`` questions ask the agent to
use a documented runtime surface and are additionally checked against the session ledger
by ``rollout.py``. Everything in this module is pure so it can be tested without rollouts.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

QuestionKind = Literal[
    "count_defs",
    "param_default",
    "callers",
    "test_containing",
    "line_count",
    "importers",
    "decorator_users",
    "api",
]

ANSWER_FORMAT = (
    "End your final answer with a line `ANSWER: <value>` where <value> is a number, a "
    "name or path, or a comma-separated list, and nothing else."
)


@dataclass
class Question:
    kind: str
    text: str
    answer: Any
    path: str | None = None
    check: str | None = None
    """For ``api`` questions: the ledger check ``rollout.py`` applies."""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return f"{self.text}\n\n{ANSWER_FORMAT}"


@dataclass
class Task:
    id: str
    repo: str
    cwd: str
    questions: list[Question]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Task:
        return cls(
            id=data["id"],
            repo=data["repo"],
            cwd=data["cwd"],
            questions=[Question(**q) for q in data["questions"]],
        )


# --- repo analysis ---------------------------------------------------------


@dataclass
class _Module:
    path: str
    tree: ast.AST
    source: str


def _modules(repo: Path) -> list[_Module]:
    modules = []
    for file in sorted(repo.rglob("*.py")):
        rel = file.relative_to(repo).as_posix()
        if any(
            part.startswith(".") or part in {"build", "dist", "node_modules"}
            for part in file.parts
        ):
            continue
        try:
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (UnicodeDecodeError, SyntaxError):
            continue
        modules.append(_Module(rel, tree, source))
    return modules


def _top_level_defs(tree: ast.AST) -> list[ast.AST]:
    return [
        node
        for node in getattr(tree, "body", [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _functions(
    tree: ast.AST,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function and method with a qualified name (``Class.method``)."""
    found = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.append((f"{node.name}.{item.name}", item))
    return found


def _literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (int, str, bool, float, type(None))
    ):
        return repr(node.value)
    return None


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                names.add(sub.func.id)
            elif isinstance(sub.func, ast.Attribute):
                names.add(sub.func.attr)
    return names


def _imported_modules(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


# --- question generators ---------------------------------------------------


def _q_count_defs(modules: list[_Module], rng: random.Random) -> Question | None:
    candidates = [m for m in modules if len(_top_level_defs(m.tree)) >= 3]
    if not candidates:
        return None
    m = rng.choice(candidates)
    return Question(
        kind="count_defs",
        text=(
            f"How many module-level definitions does `{m.path}` contain, counting "
            "functions, async functions and classes together as one total? Count only "
            "definitions at module level, not methods or nested ones."
        ),
        answer=len(_top_level_defs(m.tree)),
        path=m.path,
    )


def _q_param_default(modules: list[_Module], rng: random.Random) -> Question | None:
    options = []
    for m in modules:
        for qualname, fn in _functions(m.tree):
            args = fn.args
            positional = args.posonlyargs + args.args
            for arg, default in zip(
                positional[len(positional) - len(args.defaults) :], args.defaults
            ):
                if (value := _literal(default)) is not None:
                    options.append((m, qualname, arg.arg, value))
            for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                if default is not None and (value := _literal(default)) is not None:
                    options.append((m, qualname, arg.arg, value))
    if not options:
        return None
    m, qualname, param, value = rng.choice(options)
    return Question(
        kind="param_default",
        text=(
            f"What is the default value of the parameter `{param}` of `{qualname}` in "
            f"`{m.path}`? Give the literal exactly as written in the source."
        ),
        answer=value,
        path=m.path,
    )


def _q_callers(modules: list[_Module], rng: random.Random) -> Question | None:
    defined = {}
    for m in modules:
        for qualname, fn in _functions(m.tree):
            defined.setdefault(fn.name, []).append(qualname)
    options = []
    for name, qualnames in defined.items():
        if len(qualnames) != 1 or len(name) < 4 or name.startswith("_"):
            continue
        callers = sorted(
            {
                f"{m.path}::{qualname}"
                for m in modules
                for qualname, fn in _functions(m.tree)
                if name in _called_names(fn) and qualname != qualnames[0]
            }
        )
        if 2 <= len(callers) <= 6:
            options.append((name, callers))
    if not options:
        return None
    name, callers = rng.choice(options)
    return Question(
        kind="callers",
        text=(
            f"Which functions or methods in this repository call `{name}(...)`? Answer "
            "with a comma-separated list of `path::QualifiedName` entries (methods as "
            "`Class.method`), excluding the definition itself."
        ),
        answer=callers,
    )


def _q_test_containing(modules: list[_Module], rng: random.Random) -> Question | None:
    tests = {}
    for m in modules:
        if not m.path.startswith("tests/") and "/tests/" not in m.path:
            continue
        for qualname, fn in _functions(m.tree):
            if fn.name.startswith("test_"):
                tests.setdefault(fn.name, set()).add(m.path)
    unique = [
        (name, next(iter(paths))) for name, paths in tests.items() if len(paths) == 1
    ]
    if not unique:
        return None
    name, path = rng.choice(unique)
    return Question(
        kind="test_containing",
        text=f"Which file defines the test function `{name}`? Answer with its path relative to the repository root.",
        answer=path,
        path=path,
    )


def _q_line_count(modules: list[_Module], rng: random.Random) -> Question | None:
    candidates = [m for m in modules if 40 <= m.source.count("\n") <= 2000]
    if not candidates:
        return None
    m = rng.choice(candidates)
    return Question(
        kind="line_count",
        text=f"How many lines does `{m.path}` have, as `wc -l` would count them?",
        answer=m.source.count("\n"),
        path=m.path,
    )


def _q_importers(modules: list[_Module], rng: random.Random) -> Question | None:
    counts: dict[str, list[str]] = {}
    for m in modules:
        for name in _imported_modules(m.tree):
            counts.setdefault(name, []).append(m.path)
    options = [
        (name, sorted(paths))
        for name, paths in counts.items()
        if 2 <= len(paths) <= 8 and "." in name
    ]
    if not options:
        return None
    name, paths = rng.choice(options)
    return Question(
        kind="importers",
        text=(
            f"Which Python files in this repository import the module `{name}` (either "
            f"`import {name}` or `from {name} import ...`)? Answer with a comma-separated "
            "list of paths relative to the repository root."
        ),
        answer=paths,
    )


def _q_decorator_users(modules: list[_Module], rng: random.Random) -> Question | None:
    options = []
    for m in modules:
        users: dict[str, list[str]] = {}
        for qualname, fn in _functions(m.tree):
            for dec in fn.decorator_list:
                if (name := _decorator_name(dec)) is not None:
                    users.setdefault(name, []).append(qualname)
        for name, qualnames in users.items():
            if 2 <= len(set(qualnames)) <= 8:
                options.append((m, name, sorted(set(qualnames))))
    if not options:
        return None
    m, name, qualnames = rng.choice(options)
    return Question(
        kind="decorator_users",
        text=(
            f"Which functions or methods in `{m.path}` are decorated with `@{name}` (with "
            "or without arguments)? Answer with a comma-separated list of qualified names "
            "(methods as `Class.method`)."
        ),
        answer=qualnames,
        path=m.path,
    )


REPO_GENERATORS = [
    _q_count_defs,
    _q_param_default,
    _q_callers,
    _q_test_containing,
    _q_line_count,
    _q_importers,
    _q_decorator_users,
]


def _q_shell_exit_code(rng: random.Random, repo: Path) -> Question:
    code = rng.randint(1, 9)
    command = f"python3 -c 'import sys; sys.exit({code})'"
    return Question(
        kind="api",
        check="shell_exit_code",
        text=(
            f"Run the shell command `{command}` through the supervisor's shell API "
            "(`rlm.shell.run`) and report the exit code it returned."
        ),
        answer=code,
        params={"command": command, "code": code},
    )


def _q_delegate_count(rng: random.Random, repo: Path) -> Question:
    dirs = sorted(
        {
            p.parent.relative_to(repo).as_posix()
            for p in repo.rglob("*.py")
            if not any(part.startswith(".") for part in p.parts)
        }
    )
    dirs = [d for d in dirs if d and 3 <= len(list((repo / d).glob("*.py"))) <= 60] or [
        "."
    ]
    directory = rng.choice(dirs)
    count = len(list((repo / directory).glob("*.py")))
    return Question(
        kind="api",
        check="delegate_count",
        text=(
            f"Delegate to a child agent (`rlm.agent.spawn`) the task of counting the `.py` "
            f"files directly inside `{directory}/` (not recursively), wait for its result, "
            "and report the number it returned."
        ),
        answer=count,
        params={"directory": directory},
    )


def _q_harness_memory(rng: random.Random, repo: Path) -> Question:
    token = rng.randint(1000, 9999)
    title = f"Probe {token}"
    return Question(
        kind="api",
        check="harness_memory",
        text=(
            f"Record a memory in the continual harness titled `{title}` whose content is "
            "the repository's top-level directory listing, then report how many memory "
            "entries the local harness holds afterwards."
        ),
        answer=1,
        params={"title": title},
    )


def _q_history_cells(rng: random.Random, repo: Path) -> Question:
    return Question(
        kind="api",
        check="history_cells",
        text=(
            "Using the conversation history API, count how many `ipython` tool calls you "
            "made in this session before this question, and report the number."
        ),
        answer=None,
    )


def _q_history_expand(rng: random.Random, repo: Path) -> Question:
    return Question(
        kind="api",
        check="history_expand",
        text=(
            "Using the conversation history API, retrieve the exact text of the first "
            "question asked in this session and report its first six words."
        ),
        answer=None,
    )


API_GENERATORS = {
    "shell_exit_code": _q_shell_exit_code,
    "delegate_count": _q_delegate_count,
    "harness_memory": _q_harness_memory,
    "history_cells": _q_history_cells,
    "history_expand": _q_history_expand,
}


def make_tasks(
    repo: Path,
    *,
    name: str,
    count: int,
    seed: int = 0,
    repo_questions: int = 4,
    api_questions: int = 2,
) -> list[Task]:
    """``count`` sessions over ``repo``: ``repo_questions`` distinct-file questions, then
    ``api_questions`` runtime-surface questions, the last of which needs history."""
    rng = random.Random(f"{name}:{seed}")
    modules = _modules(repo)
    tasks = []
    for index in range(count):
        questions: list[Question] = []
        used_paths: set[str] = set()
        generators = REPO_GENERATORS[:]
        rng.shuffle(generators)
        for generator in generators * 3:
            if len(questions) >= repo_questions:
                break
            question = generator(modules, rng)
            if question is None or (question.path and question.path in used_paths):
                continue
            if question.path:
                used_paths.add(question.path)
            questions.append(question)
        api_kinds = [
            "shell_exit_code",
            "delegate_count",
            "harness_memory",
            "history_cells",
        ]
        rng.shuffle(api_kinds)
        for kind in api_kinds[: max(0, api_questions - 1)]:
            questions.append(API_GENERATORS[kind](rng, repo))
        if api_questions:
            questions.append(_q_history_expand(rng, repo))
        tasks.append(
            Task(
                id=f"{name}-{index:03d}", repo=name, cwd=str(repo), questions=questions
            )
        )
    return tasks


# --- scoring ----------------------------------------------------------------

_ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.MULTILINE)


def extract_answer(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text or "")
    return matches[-1] if matches else None


def _normalize(value: Any) -> str:
    text = str(value).strip().rstrip(".").strip("`'\"").strip().rstrip(".")
    if text.lower() in {"none", "null"}:
        return "None"
    if text.lower() in {"true", "false"}:
        return text.capitalize()
    return text


def _normalize_list(value: Any) -> set[str]:
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[,\n]", str(value))
    return {_normalize(item) for item in items if _normalize(item)}


def score_answer(question: Question, answer_text: str | None) -> float:
    """1.0 for an exact scalar match, set-F1 for list answers, 0 without an answer."""
    extracted = extract_answer(answer_text or "")
    if extracted is None or question.answer is None:
        return 0.0
    if isinstance(question.answer, list):
        expected = _normalize_list(question.answer)
        got = _normalize_list(extracted)
        if not got:
            return 0.0
        hits = len(expected & got)
        if hits == 0:
            return 0.0
        precision = hits / len(got)
        recall = hits / len(expected)
        return round(2 * precision * recall / (precision + recall), 3)
    expected = _normalize(question.answer)
    got = _normalize(extracted)
    if got == expected:
        return 1.0
    try:
        return 1.0 if float(got) == float(expected) else 0.0
    except ValueError:
        return 0.0


# --- CLI ------------------------------------------------------------------------


def write_tasks(path: Path, tasks: list[Task]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_json()) + "\n")


def read_tasks(path: Path) -> list[Task]:
    with open(path, encoding="utf-8") as handle:
        return [Task.from_json(json.loads(line)) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate question sessions over repositories."
    )
    parser.add_argument("--repo", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", required=True, help="Output tasks.jsonl")
    parser.add_argument("--per-repo", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo-questions", type=int, default=4)
    parser.add_argument("--api-questions", type=int, default=2)
    args = parser.parse_args(argv)
    tasks = []
    for spec in args.repo:
        name, _, path = spec.partition("=")
        tasks.extend(
            make_tasks(
                Path(path).resolve(),
                name=name,
                count=args.per_repo,
                seed=args.seed,
                repo_questions=args.repo_questions,
                api_questions=args.api_questions,
            )
        )
    write_tasks(Path(args.out), tasks)
    print(f"wrote {len(tasks)} tasks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
