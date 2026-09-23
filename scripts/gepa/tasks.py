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
    "subclasses",
    "longest_function",
    "followup",
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


def numbered_prompt(position: int, question: Question) -> str:
    """The prompt as sent: labelled so later questions can refer back by number."""
    return f"Question {position + 1}: {question.prompt}"


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


_caller_options: dict[int, list[tuple[str, list[str]]]] = {}


def _callers_index(modules: list[_Module]) -> list[tuple[str, list[str]]]:
    """(name, callers) for every uniquely defined function with 2-6 callers; the
    called-name sets are computed once per module list."""
    key = id(modules)
    if key in _caller_options:
        return _caller_options[key]
    defined: dict[str, list[str]] = {}
    calls: list[tuple[str, str, set[str]]] = []
    for m in modules:
        for qualname, fn in _functions(m.tree):
            defined.setdefault(fn.name, []).append(qualname)
            calls.append((m.path, qualname, _called_names(fn)))
    options = []
    for name, qualnames in defined.items():
        if len(qualnames) != 1 or len(name) < 4 or name.startswith("_"):
            continue
        callers = sorted(
            f"{path}::{qualname}"
            for path, qualname, called in calls
            if name in called and qualname != qualnames[0]
        )
        if 2 <= len(callers) <= 6:
            options.append((name, callers))
    _caller_options[key] = options
    return options


def _q_callers(modules: list[_Module], rng: random.Random) -> Question | None:
    options = _callers_index(modules)
    if not options:
        return None
    name, callers = rng.choice(options)
    return Question(
        kind="callers",
        text=(
            f"Which module-level functions and class methods in this repository call "
            f"`{name}(...)`? Answer with a comma-separated list of `path::QualifiedName` "
            "entries, where a qualified name is a module-level function (`function`) or a "
            "method of a module-level class (`Class.method`). A call inside a nested "
            "function counts for the module-level function or method that encloses it; "
            "never report a nested function's own name. Exclude the definition itself."
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
            f"Which module-level functions and methods of module-level classes in "
            f"`{m.path}` are decorated with `@{name}` (with or without arguments)? Ignore "
            "definitions nested inside functions or inside nested classes. Answer with a "
            "comma-separated list of qualified names (`function` or `Class.method`)."
        ),
        answer=qualnames,
        path=m.path,
    )


def _base_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _q_subclasses(modules: list[_Module], rng: random.Random) -> Question | None:
    defined: dict[str, int] = {}
    subclasses: dict[str, list[str]] = {}
    for m in modules:
        for node in getattr(m.tree, "body", []):
            if not isinstance(node, ast.ClassDef):
                continue
            defined[node.name] = defined.get(node.name, 0) + 1
            for base in node.bases:
                if (name := _base_name(base)) is not None:
                    subclasses.setdefault(name, []).append(f"{m.path}::{node.name}")
    options = [
        (name, sorted(set(entries)))
        for name, entries in subclasses.items()
        if defined.get(name) == 1 and 2 <= len(set(entries)) <= 8
    ]
    if not options:
        return None
    name, entries = rng.choice(options)
    return Question(
        kind="subclasses",
        text=(
            f"Which module-level classes in this repository list `{name}` directly among "
            f"their base classes, written either as `{name}` or as an attribute ending in "
            f"`.{name}`? Only direct bases count, not indirect inheritance, and classes "
            "nested inside functions or other classes do not count. Answer with a "
            "comma-separated list of `path::ClassName` entries."
        ),
        answer=entries,
    )


def _q_longest_function(modules: list[_Module], rng: random.Random) -> Question | None:
    by_dir: dict[str, list[_Module]] = {}
    for m in modules:
        by_dir.setdefault(m.path.rpartition("/")[0], []).append(m)
    options = []
    for directory, members in by_dir.items():
        if len(members) < 3:
            continue
        spans = sorted(
            (
                (fn.end_lineno or fn.lineno) - fn.lineno + 1,
                f"{m.path}::{qualname}",
            )
            for m in members
            for qualname, fn in _functions(m.tree)
        )
        if len(spans) >= 2 and spans[-1][0] > spans[-2][0]:
            options.append((directory, spans[-1][1]))
    if not options:
        return None
    directory, answer = rng.choice(options)
    where = f"`{directory}/`" if directory else "the repository root"
    return Question(
        kind="longest_function",
        text=(
            "Which module-level function or method of a module-level class, among the "
            f"`.py` files directly inside {where} (not in subdirectories), spans the most "
            "lines, counting from its `def` line through its last line with decorators "
            "excluded? Answer as `path::QualifiedName`, where the qualified name is "
            "`function` or `Class.method`."
        ),
        answer=answer,
    )


REPO_GENERATORS = [
    _q_count_defs,
    _q_param_default,
    _q_callers,
    _q_test_containing,
    _q_line_count,
    _q_importers,
    _q_decorator_users,
    _q_subclasses,
    _q_longest_function,
]

_SUBJECT_FILE = {
    "count_defs": "the file Question {n} asked about",
    "param_default": "the file Question {n} asked about",
    "decorator_users": "the file Question {n} asked about",
    "line_count": "the file Question {n} asked about",
    "test_containing": "the file that correctly answers Question {n}",
}


def _q_followup(
    earlier: list[Question], modules: list[_Module], rng: random.Random
) -> Question | None:
    """A question about the subject of an earlier one, referring to it only by number,
    so answering it after a compaction needs the earlier work carried forward."""
    by_path = {m.path: m for m in modules}
    options = []
    for index, question in enumerate(earlier):
        n = index + 1
        if question.kind in _SUBJECT_FILE and question.path in by_path:
            m = by_path[question.path]
            subject = _SUBJECT_FILE[question.kind].format(n=n)
            if question.kind != "line_count":
                options.append(
                    (
                        f"How many lines does {subject} have, as `wc -l` would count them?",
                        m.source.count("\n"),
                    )
                )
            if question.kind != "count_defs":
                options.append(
                    (
                        f"How many module-level definitions does {subject} contain, "
                        "counting functions, async functions and classes together as one "
                        "total? Count only definitions at module level.",
                        len(_top_level_defs(m.tree)),
                    )
                )
        elif question.kind in ("callers", "importers", "subclasses") and isinstance(
            question.answer, list
        ):
            paths = sorted({entry.split("::")[0] for entry in question.answer})
            sized = sorted(
                (by_path[p].source.count("\n"), p) for p in paths if p in by_path
            )
            if len(sized) >= 2 and sized[-1][0] > sized[-2][0]:
                options.append(
                    (
                        f"Among the distinct files that appear in the correct answer to "
                        f"Question {n}, which one has the most lines, as `wc -l` would "
                        "count them? Answer with its path.",
                        sized[-1][1],
                    )
                )
    if not options:
        return None
    text, answer = rng.choice(options)
    return Question(kind="followup", text=text, answer=answer)


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
            "Using the conversation history API, count the `ipython` tool calls recorded "
            "in the ledger before the user message that contains this question, and "
            "report that number. Do not count the cells you run to answer this."
        ),
        answer=None,
    )


def _q_history_expand(rng: random.Random, repo: Path) -> Question:
    return Question(
        kind="api",
        check="history_expand",
        text=(
            "Using the conversation history API, retrieve the exact text of Question 1 "
            "of this session and report the first six words of the question itself (after "
            "its `Question 1:` label), separated by single spaces, exactly as written."
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
    repo_questions: int = 6,
    followups: int = 2,
    api_questions: int = 2,
) -> list[Task]:
    """``count`` sessions over ``repo``: ``repo_questions`` distinct-file questions,
    ``followups`` questions about earlier ones' subjects, then ``api_questions``
    runtime-surface questions, the last of which needs history after a compaction."""
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
        earlier = questions[: max(1, len(questions) // 2)]
        for _ in range(followups):
            question = _q_followup(earlier, modules, rng)
            if question is not None and question.text not in {
                q.text for q in questions
            }:
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
    parser.add_argument(
        "--out",
        default="scripts/gepa/tasks/tasks.jsonl",
        help="Output file (its directory is created)",
    )
    parser.add_argument("--per-repo", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo-questions", type=int, default=6)
    parser.add_argument("--followups", type=int, default=2)
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
                followups=args.followups,
                api_questions=args.api_questions,
            )
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_tasks(out, tasks)
    print(f"wrote {len(tasks)} tasks to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
