"""Pure parts of the GEPA prompt-optimization scripts: task generation, answer scoring,
ledger checks for api questions, and the candidate guard. No rollouts, no GEPA import."""

from __future__ import annotations

import json
import random
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gepa"))

from components import validate_candidate  # noqa: E402
from rollout import CHECKS, Cell, Rollout, _quoted_words  # noqa: E402
from tasks import (  # noqa: E402
    Question,
    _modules,
    _q_callers,
    _q_count_defs,
    _q_decorator_users,
    _q_importers,
    _q_line_count,
    _q_param_default,
    _q_test_containing,
    make_tasks,
    read_tasks,
    score_answer,
    write_tasks,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        textwrap.dedent(
            """
            import json
            import os.path

            def helper(x, retries=3):
                return x

            class Thing:
                @property
                def size(self):
                    return helper(1)

                @property
                def name(self):
                    return "t"

            def use_a():
                return helper(2)
            """
        ).lstrip()
    )
    (tmp_path / "pkg" / "b.py").write_text(
        textwrap.dedent(
            """
            import os.path
            from pkg.a import helper

            def use_b(*, flag=None):
                return helper(3)
            """
        ).lstrip()
    )
    (tmp_path / "tests" / "test_a.py").write_text(
        "def test_helper():\n    assert True\n"
    )
    return tmp_path


def test_generators_compute_answers_from_the_ast(repo: Path):
    modules = _modules(repo)
    rng = random.Random(0)

    count = _q_count_defs(modules, rng)
    assert count.path == "pkg/a.py" and count.answer == 3

    callers = _q_callers(modules, rng)
    assert callers.answer == [
        "pkg/a.py::Thing.size",
        "pkg/a.py::use_a",
        "pkg/b.py::use_b",
    ]

    test = _q_test_containing(modules, rng)
    assert (test.answer, "test_helper" in test.text) == ("tests/test_a.py", True)

    importers = _q_importers(modules, rng)
    assert importers.answer == ["pkg/a.py", "pkg/b.py"] and "os.path" in importers.text

    decorated = _q_decorator_users(modules, rng)
    assert decorated.answer == ["Thing.name", "Thing.size"]

    defaults = {
        (q.text.split("`")[1], q.answer)
        for q in (_q_param_default(modules, random.Random(i)) for i in range(20))
    }
    assert defaults == {("retries", "3"), ("flag", "None")}

    lines = _q_line_count([m for m in modules if m.path == "pkg/a.py"] * 1, rng)
    assert lines is None  # below the 40-line floor


def test_make_tasks_orders_api_questions_last_and_round_trips(
    repo: Path, tmp_path: Path
):
    tasks = make_tasks(
        repo, name="fix", count=2, seed=3, repo_questions=2, api_questions=2
    )
    assert [t.id for t in tasks] == ["fix-000", "fix-001"]
    for task in tasks:
        kinds = [q.kind for q in task.questions]
        assert kinds[:2] != ["api", "api"] and kinds[-2:] == ["api", "api"]
        assert task.questions[-1].check == "history_expand"
        assert all("ANSWER:" in q.prompt for q in task.questions)
    path = tmp_path / "tasks.jsonl"
    write_tasks(path, tasks)
    assert [t.to_json() for t in read_tasks(path)] == [t.to_json() for t in tasks]


def test_score_answer_scalars_and_lists():
    scalar = Question("count_defs", "", 7)
    assert score_answer(scalar, "reasoning...\nANSWER: 7") == 1.0
    assert score_answer(scalar, "ANSWER: `7`.") == 1.0
    assert score_answer(scalar, "ANSWER: seven") == 0.0
    assert score_answer(scalar, "no answer line") == 0.0
    assert score_answer(Question("param_default", "", "None"), "ANSWER: null") == 1.0
    listed = Question("callers", "", ["a::f", "b::g", "c::h"])
    assert score_answer(listed, "ANSWER: c::h, a::f, b::g") == 1.0
    assert score_answer(listed, "ANSWER: a::f, zz") == 0.4
    assert score_answer(listed, "ANSWER: none of them") == 0.0
    assert score_answer(Question("api", "", None), "ANSWER: 1") == 0.0


def _rollout(cells: list[tuple[int, str]], spawns=()) -> Rollout:
    rollout = Rollout(task_id="t", session_dir="")
    rollout.cells = [Cell(i, code, "", False, False) for i, code in cells]
    rollout.spawns = list(spawns)
    return rollout


def test_ledger_checks_for_api_questions(tmp_path: Path):
    shell = Question(
        "api", "", 4, check="shell_exit_code", params={"command": "x", "code": 4}
    )
    ok, _ = CHECKS["shell_exit_code"](
        shell,
        _rollout(
            [(5, "job = await rlm.shell.run(\"python3 -c 'import sys; sys.exit(4)'\")")]
        ),
        None,
        tmp_path,
        3,
    )
    assert ok
    ok, note = CHECKS["shell_exit_code"](
        shell,
        _rollout(
            [
                (
                    5,
                    "import subprocess; subprocess.run(['python3','-c','import sys; sys.exit(4)'])",
                )
            ]
        ),
        None,
        tmp_path,
        3,
    )
    assert not ok and "subprocess" in note
    ok, _ = CHECKS["shell_exit_code"](
        shell, _rollout([(1, "await rlm.shell.run('sys.exit(4)')")]), None, tmp_path, 3
    )
    assert not ok  # ran before the question was asked

    delegate = Question("api", "", 2, check="delegate_count")
    assert CHECKS["delegate_count"](
        delegate, _rollout([], spawns=[(9, "count")]), None, tmp_path, 3
    )[0]
    assert not CHECKS["delegate_count"](
        delegate, _rollout([], spawns=[(2, "count")]), None, tmp_path, 3
    )[0]

    memory = Question("api", "", 1, check="harness_memory", params={"title": "Probe 1"})
    assert not CHECKS["harness_memory"](memory, _rollout([]), None, tmp_path, 3)[0]
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "harness_state.json").write_text(
        json.dumps(
            {
                "entries": {
                    "memory": {"a": {"title": "Probe 1"}, "b": {"title": "Other"}}
                }
            }
        )
    )
    ok, _ = CHECKS["harness_memory"](memory, _rollout([]), None, tmp_path, 3)
    assert ok and memory.answer == 2

    history = Question("api", "", None, check="history_cells")
    assert CHECKS["history_cells"](
        history, _rollout([(4, "hist = await history()")]), None, tmp_path, 3
    )[0]
    assert not CHECKS["history_cells"](
        history, _rollout([(4, "print(1)")]), None, tmp_path, 3
    )[0]


def test_history_expand_ignores_quoting_punctuation(tmp_path: Path):
    question = Question(
        "api", "", "Which functions or methods in", check="history_expand"
    )
    rollout = _rollout([(4, "hist = await history(); hist.expand(0)")])
    ok, _ = CHECKS["history_expand"](question, rollout, None, tmp_path, 3)
    assert ok
    assert _quoted_words("Which, functions, or, methods, in") == _quoted_words(
        "`Which` functions or methods in."
    )
    assert _quoted_words("Which functions in") != _quoted_words(
        "Which functions or methods in"
    )


def test_candidate_guard_names_dropped_tokens():
    seed = {
        "history": "use `hist.blocks` and `hist.expand(` and `from rlm import history` "
        * 3,
        "task": "Do the task.",
    }
    seed["history"] += (
        " hist.messages hist.windows hist.user_messages() history(session_dir="
    )
    assert validate_candidate(seed, seed) == {}
    problems = validate_candidate(
        {**seed, "history": seed["history"].replace("hist.expand(", "hist.open(")}, seed
    )
    assert list(problems) == ["history"] and "`hist.expand(`" in problems["history"]
    assert "too long" not in problems["history"]
    problems = validate_candidate({**seed, "task": "x" * 1_300}, seed)
    assert "more than the 1200 allowed" in problems["task"]
    assert validate_candidate({**seed, "task": "x" * 1_000}, seed) == {}
    assert (
        validate_candidate({**seed, "task": "  "}, seed)["task"]
        == "the rewritten text is empty"
    )
    assert "unknown component" in validate_candidate({"bogus": "x"}, seed)["bogus"]
