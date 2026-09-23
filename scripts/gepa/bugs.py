"""Bug-fix sessions: injected bugs in a pinned repository, verified by its own tests.

A mutant is one small edit to a library function (a flipped comparison, ``and``/``or``,
``+``/``-``, an integer off by one, a flipped boolean, a dropped ``not``) that makes
1-25 tests fail across at most three test files, with no collection errors. A session
works in a private copy of the checkout and introduces its mutants one at a time, each
in a different source file, right before the prompt that reports its failing tests,
like bug reports arriving in turn; it ends by asking which file an earlier fix changed.
A fix scores the share of its target tests that pass, or zero when another test in those
files fails (earlier bugs' tests that are still failing aside) or a test file was
edited.

    uv run python scripts/gepa/bugs.py --repo itsdangerous=<path> --repo click=<path> \\
        --per-repo 10 --out scripts/gepa/tasks/bugs.jsonl

Each repository's tests run in their own virtualenv (``<workspace>/.venvs/<name>``,
built from ``requirements/tests.txt``) with ``PYTHONPATH=src``, so the library under test
is always the session's copy.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import os
import queue
import random
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tasks import API_GENERATORS, Question, Task, write_tasks

MAX_FAILURES = 25
MAX_TEST_FILES = 3
TEST_TIMEOUT_S = 120
"""Per test run while scoring a fix; screening uses a timeout scaled to the baseline."""


@dataclass
class Mutation:
    path: str
    line: int
    """1-based line of the edit."""
    start: int
    end: int
    """UTF-8 byte offsets of the replaced segment within the line."""
    original: str
    mutated: str
    operator: str


@dataclass
class Mutant:
    mutation: Mutation
    test_files: list[str]
    targets: list[str]
    """Test ids (``classname::name``) that fail with the mutation applied."""


# --- mutation sites ----------------------------------------------------------------

_COMPARE_SWAPS = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}


def _mutations_of(node: ast.AST) -> list[tuple[str, ast.AST]]:
    """(operator name, mutated copy) for each way ``node`` can be mutated."""
    found: list[tuple[str, ast.AST]] = []
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        swap = _COMPARE_SWAPS.get(type(node.ops[0]))
        if swap is not None:
            mutated = copy.deepcopy(node)
            mutated.ops = [swap()]
            found.append(("compare", mutated))
    elif isinstance(node, ast.BoolOp):
        mutated = copy.deepcopy(node)
        mutated.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        found.append(("boolean", mutated))
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        if not any(
            isinstance(side, ast.Constant) and isinstance(side.value, str)
            for side in (node.left, node.right)
        ):
            mutated = copy.deepcopy(node)
            mutated.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
            found.append(("arithmetic", mutated))
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            found.append(("boolean_constant", ast.Constant(not node.value)))
        elif isinstance(node.value, int) and 0 <= node.value <= 10:
            found.append(("off_by_one", ast.Constant(node.value + 1)))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        found.append(("drop_not", copy.deepcopy(node.operand)))
    return found


def mutation_sites(repo: Path) -> list[Mutation]:
    """Every single-line mutation inside a function body under ``src/``."""
    sites = []
    for file in sorted((repo / "src").rglob("*.py")):
        rel = file.relative_to(repo).as_posix()
        source = file.read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)
        tree = ast.parse(source)
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for statement in function.body:
                for node in ast.walk(statement):
                    line_no = getattr(node, "lineno", None)
                    if line_no is None or line_no != node.end_lineno:
                        continue
                    line = lines[node.lineno - 1].encode("utf-8")
                    original = line[node.col_offset : node.end_col_offset].decode()
                    for operator, mutated in _mutations_of(node):
                        text = ast.unparse(mutated)
                        if text != original:
                            sites.append(
                                Mutation(
                                    rel,
                                    node.lineno,
                                    node.col_offset,
                                    node.end_col_offset,
                                    original,
                                    text,
                                    operator,
                                )
                            )
    unique = {(m.path, m.line, m.start, m.mutated): m for m in sites}
    return list(unique.values())


def apply_mutation(root: Path, mutation: Mutation) -> None:
    file = root / mutation.path
    lines = file.read_text(encoding="utf-8").splitlines(keepends=True)
    line = lines[mutation.line - 1].encode("utf-8")
    segment = line[mutation.start : mutation.end].decode()
    if segment != mutation.original:
        raise ValueError(
            f"{mutation.path}:{mutation.line}: expected {mutation.original!r}, found {segment!r}"
        )
    lines[mutation.line - 1] = (
        line[: mutation.start] + mutation.mutated.encode() + line[mutation.end :]
    ).decode()
    file.write_text("".join(lines), encoding="utf-8")


# --- test environment and runs ------------------------------------------------------


def test_python(repo: Path, name: str, workspace: Path, python: str = "3.12") -> Path:
    """The interpreter of this repository's test virtualenv, created on first use."""
    venv = workspace / ".venvs" / name
    interpreter = venv / "bin" / "python"
    if interpreter.exists():
        return interpreter
    subprocess.run(["uv", "venv", "--quiet", "--python", python, str(venv)], check=True)
    requirements = repo / "requirements" / "tests.txt"
    install = ["-r", str(requirements)] if requirements.exists() else ["pytest"]
    subprocess.run(
        ["uv", "pip", "install", "--quiet", "--python", str(interpreter), *install],
        check=True,
    )
    return interpreter


def test_command(python: Path | str, test_files: list[str]) -> str:
    return f"PYTHONPATH=src {python} -m pytest -q {' '.join(test_files)}"


def run_tests(
    root: Path,
    python: Path | str,
    test_files: list[str] | None = None,
    timeout: float = TEST_TIMEOUT_S,
) -> dict[str, str]:
    """Outcome per test id (``passed``/``failed``/``skipped``, or ``error`` for a module
    that failed to collect, as ``::<module>``); empty when pytest did not run or timed
    out."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.xml"
        try:
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    f"--junitxml={report}",
                    *(test_files or []),
                ],
                cwd=root,
                env={
                    **{
                        k: v
                        for k, v in os.environ.items()
                        if not k.startswith("PYTHON")
                    },
                    "PYTHONPATH": "src",
                    # Bytecode caches are keyed on whole-second mtimes and size, which a
                    # same-size edit within a second does not change.
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {}
        if not report.exists():
            return {}
        outcomes = {}
        for case in ET.parse(report).iter("testcase"):
            test_id = f"{case.get('classname')}::{case.get('name')}"
            if not case.get("classname"):
                outcomes[test_id] = "error"
            elif case.find("failure") is not None or case.find("error") is not None:
                outcomes[test_id] = "failed"
            elif case.find("skipped") is not None:
                outcomes[test_id] = "skipped"
            else:
                outcomes[test_id] = "passed"
        return outcomes


def _test_file(root: Path, test_id: str) -> str:
    """The file defining a test id: the longest dotted prefix of its JUnit classname
    that is a module under ``root`` (the rest names test classes)."""
    parts = test_id.split("::")[0].split(".")
    for end in range(len(parts), 0, -1):
        candidate = "/".join(parts[:end]) + ".py"
        if (root / candidate).is_file():
            return candidate
    raise ValueError(f"no file under {root} defines {test_id}")


def _copy(repo: Path, target: Path) -> Path:
    shutil.copytree(
        repo, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
    )
    return target


# --- generation ---------------------------------------------------------------------


def _failures(
    root: Path, python: Path, timeout: float, files: list[str] | None = None
) -> list[str] | None:
    """Failing test ids, or None when the run errored, timed out or collected nothing."""
    outcomes = run_tests(root, python, files, timeout=timeout)
    if not outcomes or "error" in outcomes.values():
        return None
    return sorted(t for t, v in outcomes.items() if v == "failed")


def _screen(
    root: Path, python: Path, mutation: Mutation, timeout: float
) -> Mutant | None:
    """Apply ``mutation`` to ``root``, run the suite, restore the file; the mutant if its
    failures are 1-``MAX_FAILURES`` tests in at most ``MAX_TEST_FILES`` files. Targets
    come from re-running just those files, as scoring does, so a test that only fails
    after others in the full suite is not one."""
    file = root / mutation.path
    original = file.read_text(encoding="utf-8")
    try:
        apply_mutation(root, mutation)
        ast.parse(file.read_text(encoding="utf-8"))
        failed = _failures(root, python, timeout)
        if not failed or len(failed) > MAX_FAILURES:
            return None
        files = sorted({_test_file(root, t) for t in failed})
        if len(files) > MAX_TEST_FILES:
            return None
        targets = _failures(root, python, timeout, files)
    except SyntaxError:
        return None
    finally:
        file.write_text(original, encoding="utf-8")
    if not targets:
        return None
    return Mutant(mutation, sorted({_test_file(root, t) for t in targets}), targets)


def find_mutants(
    repo: Path,
    python: Path,
    rng: random.Random,
    want: int,
    attempts: int,
    workers: int = 8,
) -> list[Mutant]:
    """The first ``want`` valid mutants in a seeded order of mutation sites, screened in
    parallel, one private copy per worker. A suite run that takes ten times the
    unmodified suite's time (a mutant that loops) counts as invalid."""
    sites = mutation_sites(repo)
    rng.shuffle(sites)
    sites = sites[:attempts]
    with tempfile.TemporaryDirectory() as tmp:
        roots = [_copy(repo, Path(tmp) / f"worker-{i}") for i in range(workers)]
        started = time.monotonic()
        baseline = run_tests(roots[0], python)
        timeout = max(15.0, 10 * (time.monotonic() - started))
        if not baseline or any(v == "failed" for v in baseline.values()):
            raise RuntimeError(f"{repo}: the unmodified test suite does not pass")
        free: queue.Queue[Path] = queue.Queue()
        for root in roots:
            free.put(root)
        mutants: list[Mutant] = []

        def screen(mutation: Mutation) -> Mutant | None:
            root = free.get()
            try:
                return _screen(root, python, mutation, timeout)
            finally:
                free.put(root)

        with ThreadPoolExecutor(workers) as pool:
            for start in range(0, len(sites), workers * 4):
                batch = sites[start : start + workers * 4]
                mutants.extend(m for m in pool.map(screen, batch) if m is not None)
                if len(mutants) >= want:
                    break
    return mutants[:want]


def _fix_prompt(mutant: Mutant, python: Path) -> str:
    files = ", ".join(f"`{f}`" for f in mutant.test_files)
    return (
        f"Some tests in {files} fail. The bug is in the library source under `src/`, "
        "not in the tests. Find it and fix it so every test in "
        f"{'that file' if len(mutant.test_files) == 1 else 'those files'} passes, "
        "without editing any test file. Run the tests with:\n\n"
        f"    {test_command(python, mutant.test_files)}\n\n"
        "Reply with one line describing the fix."
    )


def make_bug_tasks(
    repo: Path,
    *,
    name: str,
    count: int,
    python: Path,
    seed: int = 0,
    bugs_per_task: int = 3,
    attempts: int = 400,
) -> list[Task]:
    """``count`` sessions of ``bugs_per_task`` fixes in different source files, a
    follow-up about an earlier fix, and the history question. Mutants are spread over
    the sessions before any is reused."""
    rng = random.Random(f"{name}:{seed}:bugs")
    pool = find_mutants(
        repo, python, rng, want=count * bugs_per_task, attempts=attempts
    )
    if len({m.mutation.path for m in pool}) < bugs_per_task:
        raise RuntimeError(
            f"{repo}: valid mutants span fewer than {bugs_per_task} source files"
        )
    uses = [0] * len(pool)
    tasks: list[Task] = []
    for index in range(count):
        order = sorted(range(len(pool)), key=lambda i: (uses[i], rng.random()))
        chosen: list[int] = []
        for i in order:
            if all(pool[j].mutation.path != pool[i].mutation.path for j in chosen):
                chosen.append(i)
            if len(chosen) == bugs_per_task:
                break
        for i in chosen:
            uses[i] += 1
        mutants = [pool[i] for i in chosen]
        questions = [
            Question(
                kind="bugfix",
                text=_fix_prompt(m, python),
                answer=None,
                path=m.mutation.path,
                params={
                    "mutation": asdict(m.mutation),
                    "test_files": m.test_files,
                    "targets": m.targets,
                },
            )
            for m in mutants
        ]
        asked = rng.randrange(len(mutants) - 1) if len(mutants) > 1 else 0
        questions.append(
            Question(
                kind="followup",
                text=(
                    f"Which library source file did your fix for Question {asked + 1} "
                    "change? Answer with its path relative to the repository root."
                ),
                answer=mutants[asked].mutation.path,
            )
        )
        questions.append(API_GENERATORS["history_expand"](rng, repo))
        tasks.append(
            Task(
                id=f"{name}-bugs-{index:03d}",
                repo=name,
                cwd=str(repo),
                questions=questions,
                setup={"python": str(python)},
            )
        )
    return tasks


# --- rollouts -----------------------------------------------------------------------


def prepare_workdir(task: Task, session_dir: Path) -> Path:
    """A private copy of the checkout, without ``.git`` or bytecode caches."""
    return _copy(Path(task.cwd), session_dir / "repo")


def introduce_bug(question: Question, root: Path) -> None:
    """Apply a bugfix question's mutation; raises ``ValueError`` when an earlier edit
    changed the line it targets."""
    apply_mutation(root, Mutation(**question.params["mutation"]))


def snapshot(root: Path) -> dict[str, str]:
    """Contents of every Python file under ``src/`` and ``tests/``."""
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace")
        for folder in ("src", "tests")
        for p in sorted((root / folder).rglob("*.py"))
    }


def _digest(files: dict[str, str]) -> str:
    return hashlib.sha256(
        "".join(f"{k}\0{v}\0" for k, v in sorted(files.items())).encode()
    ).hexdigest()


def patch(before: dict[str, str], after: dict[str, str]) -> str:
    chunks = []
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path, ""), after.get(path, "")
        if old != new:
            chunks.extend(
                difflib.unified_diff(
                    old.splitlines(),
                    new.splitlines(),
                    f"a/{path}",
                    f"b/{path}",
                    lineterm="",
                    n=1,
                )
            )
    return "\n".join(chunks)


def check_fix(
    question: Question,
    root: Path,
    python: str,
    before: dict[str, str],
    unresolved: set[str],
) -> dict[str, Any]:
    """Score one fix right after its prompt: the share of target tests that pass, zero on
    a regression in the same test files or an edited test file. ``unresolved`` holds
    earlier bugs' target tests that were still failing, which are not regressions."""
    after = snapshot(root)
    tests_before = {k: v for k, v in before.items() if k.startswith("tests/")}
    tests_after = {k: v for k, v in after.items() if k.startswith("tests/")}
    targets = question.params["targets"]
    result: dict[str, Any] = {
        "patch": patch(before, after),
        "snapshot": after,
        "failing": set(targets),
    }
    if _digest(tests_before) != _digest(tests_after):
        return {**result, "score": 0.0, "note": "a test file was edited"}
    for cache in list(root.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    outcomes = run_tests(root, python, question.params["test_files"])
    if not outcomes:
        return {
            **result,
            "score": 0.0,
            "note": "the tests did not run or timed out",
        }
    passed = [t for t in targets if outcomes.get(t) == "passed"]
    result["failing"] = set(targets) - set(passed)
    regressions = sorted(
        t
        for t, v in outcomes.items()
        if v in ("failed", "error") and t not in targets and t not in unresolved
    )
    if regressions:
        return {
            **result,
            "score": 0.0,
            "note": f"{len(passed)}/{len(targets)} target tests pass but other tests in "
            f"those files now fail: {', '.join(regressions[:5])}",
        }
    return {
        **result,
        "score": round(len(passed) / len(targets), 3),
        "note": f"{len(passed)}/{len(targets)} target tests pass",
    }


# --- CLI ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", default="scripts/gepa/tasks/bugs.jsonl")
    parser.add_argument("--workspace", default="scripts/gepa/workspace")
    parser.add_argument("--per-repo", type=int, default=10)
    parser.add_argument("--bugs-per-task", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=400)
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    tasks: list[Task] = []
    for spec in args.repo:
        name, _, path = spec.partition("=")
        repo = Path(path).resolve()
        python = test_python(repo, name, workspace, args.python)
        made = make_bug_tasks(
            repo,
            name=name,
            count=args.per_repo,
            python=python,
            seed=args.seed,
            bugs_per_task=args.bugs_per_task,
            attempts=args.attempts,
        )
        print(f"{name}: {len(made)} sessions", flush=True)
        tasks.extend(made)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_tasks(out, tasks)
    print(f"wrote {len(tasks)} tasks to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
