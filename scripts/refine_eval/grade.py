"""Grade refinement-eval cases from their session ledger and final harness state, and
aggregate the rows into a report. Pure functions: no model or network calls."""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean
from typing import Any

from scenarios import (
    AtMostOne,
    Changed,
    Created,
    EditCheck,
    NoCreate,
    Scenario,
    matches,
)

REFINING_ARMS = ("force", "force+focus", "typesafe")
_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)")


def applied_edits(records: list[dict]) -> list[dict]:
    """Every applied edit of every refinement in the ledger, with the scope of the
    store it was applied to."""
    return [
        {**edit, "scope": record["result"]["scope"]}
        for record in records
        if record.get("type") == "refinement"
        for edit in record["result"]["applied_edits"]
        if edit["applied"]
    ]


def rejected_edits(records: list[dict]) -> int:
    return sum(
        1
        for record in records
        if record.get("type") == "refinement"
        for edit in record["result"]["applied_edits"]
        if not edit["applied"]
    )


def check_edit(
    check: EditCheck, edits: list[dict], final: dict[str, list[dict]]
) -> bool:
    """``final`` maps each store's scope to its entries after the run, as dicts."""
    edits = [e for e in edits if e["scope"] == check.scope]
    if isinstance(check, Created):
        return any(
            e["action"] == "create"
            and e["kind"] in check.kinds
            and matches(check.pattern, e.get("title"), e.get("content"))
            for e in edits
        )
    if isinstance(check, Changed):
        return any(
            e["action"] in ("update", "delete")
            and e["kind"] == check.kind
            and e["id"] == check.id
            for e in edits
        )
    if isinstance(check, NoCreate):
        return not any(
            e["action"] == "create"
            and (
                not check.pattern
                or matches(check.pattern, e.get("title"), e.get("content"))
            )
            for e in edits
        )
    if isinstance(check, AtMostOne):
        return (
            sum(
                1
                for e in final.get(check.scope, [])
                if matches(check.pattern, e.get("title"), e.get("content"))
            )
            <= 1
        )
    raise TypeError(f"unknown edit check {check!r}")


def _normalize(value: str) -> str:
    return value.strip().strip("`'\".").strip().lower()


def probe_score(expected: str, answer: str | None) -> float:
    found = _ANSWER_RE.findall(answer or "")
    return float(bool(found) and _normalize(found[-1]) == _normalize(expected))


def _top(choice: dict[str, Any]) -> str:
    return max(choice["probabilities"], key=choice["probabilities"].get)


def visible_scopes(scenario: Scenario) -> tuple[str, ...]:
    """The stores a session sees for a scenario's pass: a local session also sees
    the global store."""
    return ("local", "global") if scenario.scope == "local" else ("global",)


def lesson_entries(
    scenario: Scenario, stores: dict[str, list[dict]], *, agent_only: bool = False
) -> list[dict]:
    """The entries in visible stores that record the scenario's lesson;
    ``agent_only`` leaves out entries the eval seeded."""
    if scenario.lesson is None:
        return []
    return [
        entry
        for scope in visible_scopes(scenario)
        for entry in stores.get(scope, [])
        if matches(scenario.lesson, entry.get("title"), entry.get("content"))
        and not (agent_only and entry.get("source") == "eval")
    ]


def after_agent(
    scenario: Scenario, before: dict[str, list[dict]], edits: list[dict], applied: bool
) -> str:
    """What a pass did about a lesson the agent had already recorded."""
    if not applied:
        return "declined"
    held = {
        (scope, entry["kind"])
        for scope in visible_scopes(scenario)
        for entry in before.get(scope, [])
        if entry.get("source") != "eval"
        and matches(scenario.lesson, entry.get("title"), entry.get("content"))
    }
    created = {
        (edit["scope"], edit["kind"])
        for edit in edits
        if edit["action"] == "create"
        and matches(scenario.lesson, edit.get("title"), edit.get("content"))
    }
    if created & held:
        return "duplicated"
    if {scope for scope, _ in created} - {scope for scope, _ in held}:
        return "overrode"
    return "applied"


def lesson_in(
    scenario: Scenario, stores: dict[str, list[dict]], *, agent_only: bool = False
) -> bool:
    return bool(lesson_entries(scenario, stores, agent_only=agent_only))


def judge_scores(
    scenario: Scenario,
    judge: dict[str, Any],
    lesson_index: int | None,
    threshold: float,
    captured: set[str],
) -> dict[str, Any]:
    """How well the judge's two calls matched the scenario's labels. A score is None
    when the scenario says nothing about it or the call that answers it did not run.
    ``captured`` is the lessons the veto should drop."""
    fired = judge["fired"]
    focus = judge["focus"] or {}
    homes = [focus[f"home_{s}"] for s in fired if f"home_{s}" in focus]
    entries = judge.get("entries", [])
    scores: dict[str, Any] = {
        "fired": fired,
        "gate_decision": judge["gate_decision"],
        "no_focus": judge["focus"] is None,
        "signal_hit": bool(set(fired) & scenario.expect_signals)
        if scenario.expect_signals
        else None,
        "home_hit": mean(_top(h) in scenario.expect_home for h in homes)
        if scenario.expect_home and homes
        else None,
        "entry_hit": None,
        "captured_hit": None,
        "lesson_turn_hit": None,
    }
    if scenario.expect_entries:
        # wrong_<k> is asked in the gate call, covers_<k> in the focus call.
        answers = {**judge["gate"], **focus}
        scores["entry_hit"] = mean(
            ref in entries
            and answers.get(f"{label}_{entries.index(ref)}", 0.0) >= threshold
            for ref, label in scenario.expect_entries.items()
        )
    vetoable = [s for s in fired if s in captured]
    if vetoable:
        scores["captured_hit"] = mean(
            focus.get(f"captured_{s}", 0.0) >= threshold for s in vetoable
        )
    if lesson_index is not None and judge["gate_decision"]:
        scores["lesson_turn_hit"] = lesson_index in judge["evidence_turns"]
    return scores


def grade_case(
    scenario: Scenario,
    arm: str,
    records: list[dict],
    before: dict[str, list[dict]],
    final: dict[str, list[dict]],
    probe_answer: str | None,
    threshold: float,
) -> dict[str, Any]:
    """One results row. ``records`` is the session ledger; ``before`` and ``final``
    are each store's entries, by scope, just before the pass and after the run. A pass is applied (a ``refinement`` record) or not (a
    ``refinement_declined`` record, whose ``reason`` says who declined).

    When the agent recorded the lesson itself before the pass (in any store the
    session sees), declining and folding more into the agent's entry are both fine,
    so the case has no expected decision (``expect_refine`` None, the scenario's own
    label kept as ``label_refine``). ``after_agent`` says what the pass did instead:
    ``declined``; ``duplicated`` (it created an entry of the same kind, in the same
    store, recording the lesson the agent's entry records); ``overrode`` (it recorded
    the lesson in the other store, as a local pass does when the agent wrote it to
    the global store); or ``applied``. Checks that credit the pass are skipped, and
    the lesson's signals are the ones the veto should drop. "lesson recorded" checks the outcome
    whoever wrote it, in any store the session sees."""
    passes = [
        r
        for r in records
        if r.get("type") in ("refinement", "refinement_declined")
        and r.get("trigger") == "host"
    ]
    last = passes[-1] if passes else None
    lesson_index = None
    if scenario.lesson_steer is not None:
        text = scenario.steer[scenario.lesson_steer]
        lesson_index = next(
            (
                r["message_index"]
                for r in records
                if r.get("type") == "user" and r.get("content") == text
            ),
            None,
        )
    edits = applied_edits(records)
    first = last is not None and lesson_in(scenario, before, agent_only=True)
    checks = {
        repr(check): check_edit(check, edits, final)
        for check in scenario.edit_checks
        if not (first and isinstance(check, (Created, Changed)))
    }
    if scenario.lesson is not None:
        checks["lesson recorded"] = lesson_in(scenario, final)
    row: dict[str, Any] = {
        "scenario": scenario.id,
        "family": scenario.family,
        "scope": scenario.scope,
        "arm": arm,
        "label_refine": scenario.expect_refine,
        "agent_recorded_first": first,
        "expect_refine": None if first else scenario.expect_refine,
        "after_agent": None,
        "decision": None,
        "declined_by": last.get("reason") if last else None,
        "rationale": None
        if last is None
        else last["result"]["rationale"]
        if last["type"] == "refinement"
        else last["rationale"],
        "edits": [f"{e['action']} {e['kind']}:{e['id']}" for e in edits],
        "rejected_edits": rejected_edits(records),
        "edit_checks": checks if arm != "none" else {},
        "probe": probe_score(scenario.probe.answer, probe_answer)
        if scenario.probe is not None
        else None,
        "judge": None,
        "judge_error": (last.get("judge") or {}).get("error") if last else None,
    }
    if arm in REFINING_ARMS:
        row["decision"] = bool(last and last["type"] == "refinement")
        if first:
            row["after_agent"] = after_agent(scenario, before, edits, row["decision"])
    judge = last and last.get("judge")
    if judge and "error" not in judge:
        captured = (
            scenario.expect_signals if first else set()
        ) | scenario.expect_captured
        row["judge"] = judge_scores(scenario, judge, lesson_index, threshold, captured)
        row["judge"]["gate"] = judge["gate"]
        row["judge"]["usage"] = judge["usage"]
    return row


# --- report --------------------------------------------------------------------


def _rate(values: list[Any]) -> str:
    known = [float(v) for v in values if v is not None]
    return f"{mean(known):.2f} (n={len(known)})" if known else "-"


def gate_table(rows: list[dict]) -> list[str]:
    lines = [
        "| arm | accuracy | precision | recall | negatives declined |",
        "|---|---|---|---|---|",
    ]
    for arm in REFINING_ARMS:
        cases = [r for r in rows if r["arm"] == arm and r["expect_refine"] is not None]
        if not cases:
            continue
        tp = sum(r["decision"] and r["expect_refine"] for r in cases)
        fp = sum(r["decision"] and not r["expect_refine"] for r in cases)
        fn = sum(not r["decision"] and r["expect_refine"] for r in cases)
        negatives = [not r["decision"] for r in cases if not r["expect_refine"]]
        accuracy = mean(r["decision"] == r["expect_refine"] for r in cases)
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        lines.append(
            f"| {arm} | {accuracy:.2f} | {precision:.2f} | {recall:.2f} | "
            f"{_rate(negatives)} |"
        )
    return lines


def outcome_table(rows: list[dict]) -> list[str]:
    """Per arm: how passes ended, split by who declined (the judge's gate or the
    planner proposing no edits), for positive and negative scenarios."""
    lines = [
        "| arm | label | applied | declined by judge | declined by planner |",
        "|---|---|---|---|---|",
    ]
    for arm in REFINING_ARMS:
        for label in (True, False):
            cases = [r for r in rows if r["arm"] == arm and r["expect_refine"] is label]
            if not cases:
                continue
            counts = [
                sum(r["decision"] for r in cases),
                sum(r["declined_by"] == "gate" for r in cases),
                sum(r["declined_by"] == "no_edits" for r in cases),
            ]
            lines.append(
                f"| {arm} | {'refine' if label else 'decline'} | "
                + " | ".join(str(c) for c in counts)
                + " |"
            )
    return lines


def agent_first_table(rows: list[dict]) -> list[str]:
    """Per arm, for cases where the agent recorded the lesson itself before the pass:
    whether the pass declined, applied edits without adding a copy of the lesson, or
    duplicated it (the only real failure here). ``overrode`` records the lesson in the
    other store, as a local pass does when the agent put it in the global store."""
    lines = [
        "| arm | cases | declined | applied | overrode in the other store | duplicated |",
        "|---|---|---|---|---|---|",
    ]
    for arm in REFINING_ARMS:
        outcomes = [
            r.get("after_agent")
            for r in rows
            if r["arm"] == arm and r.get("after_agent")
        ]
        if outcomes:
            counts = [
                outcomes.count(o)
                for o in ("declined", "applied", "overrode", "duplicated")
            ]
            lines.append(
                f"| {arm} | {len(outcomes)} | " + " | ".join(map(str, counts)) + " |"
            )
    return lines


def family_table(rows: list[dict]) -> list[str]:
    arms = sorted({r["arm"] for r in rows})
    lines = [
        "| family | " + " | ".join(f"{a} checks / probe" for a in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    by_family: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_family[row["family"]][row["arm"]].append(row)
    for family, by_arm in sorted(by_family.items()):
        cells = []
        for arm in arms:
            cases = by_arm.get(arm, [])
            checks = [v for r in cases for v in r["edit_checks"].values()]
            cells.append(f"{_rate(checks)} / {_rate([r['probe'] for r in cases])}")
        lines.append(f"| {family} | " + " | ".join(cells) + " |")
    return lines


def judge_table(rows: list[dict]) -> list[str]:
    lines = [
        "| arm | signal recall | home | entries | veto | lesson turn | no focus on positives |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in sorted({r["arm"] for r in rows if r["judge"]}):
        judged = [r["judge"] for r in rows if r["arm"] == arm and r["judge"]]
        positives = [
            r["judge"]["no_focus"]
            for r in rows
            if r["arm"] == arm and r["judge"] and r["expect_refine"]
        ]
        lines.append(
            f"| {arm} | {_rate([j['signal_hit'] for j in judged])} | "
            f"{_rate([j['home_hit'] for j in judged])} | "
            f"{_rate([j['entry_hit'] for j in judged])} | "
            f"{_rate([j['captured_hit'] for j in judged])} | "
            f"{_rate([j['lesson_turn_hit'] for j in judged])} | {_rate(positives)} |"
        )
    return lines


def sweep(rows: list[dict], thresholds: list[float]) -> list[str]:
    """Gate-call accuracy of the ``typesafe`` arm at other thresholds, recomputed from
    logged probabilities. Only call 1 can be replayed: call 2 was asked about the
    lessons that fired at the threshold the run used."""
    cases = [
        r
        for r in rows
        if r["arm"] == "typesafe" and r["judge"] and r["expect_refine"] is not None
    ]
    lines = ["| threshold | call-1 accuracy |", "|---|---|"]
    for t in thresholds:
        if not cases:
            break
        accuracy = mean(
            any(p >= t for p in r["judge"]["gate"].values()) == r["expect_refine"]
            for r in cases
        )
        lines.append(f"| {t:.2f} | {accuracy:.2f} |")
    return lines


def report(rows: list[dict], thresholds: list[float]) -> str:
    """Markdown summary; errored cases and failed passes are counted but left out of
    every table."""
    ok = [r for r in rows if not r.get("error") and r.get("declined_by") != "failed"]
    sections = [
        ("Decision vs label", gate_table(ok)),
        ("Pass outcomes", outcome_table(ok)),
        ("Agent recorded first", agent_first_table(ok)),
        ("Edit checks / probe score by family", family_table(ok)),
        ("Judge focus", judge_table(ok)),
        ("Call-1 threshold sweep (typesafe arm)", sweep(ok, thresholds)),
    ]
    body = "\n\n".join(
        f"## {title}\n\n" + "\n".join(lines) for title, lines in sections
    )
    first = sum(bool(r.get("agent_recorded_first")) for r in ok)
    leaked = sum(bool(r.get("touched_repo")) for r in rows)
    changed = sum(bool(r.get("venv_changed")) for r in rows)
    return (
        f"# Refinement eval\n\n{len(rows)} cases, {len(rows) - len(ok)} errored or "
        f"failed. In {first}, the agent recorded the lesson itself before the pass; "
        "they have no expected decision and are reported under Agent recorded first. "
        f"{leaked} reached into this repository; in {changed}, the agent changed "
        "the installed packages of its own case.\n\n"
        f"{body}\n"
    )
