"""Operator maintenance of a harness store, run outside any session.

The engine only ever inserts episodes and nothing inside a session may remove one, so
trimming a shared global store is an explicit command: select by age or count, show
the plan, and apply under the store lock with an audit record.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict

from rlm.harness import HarnessEntry, HarnessStore, RefinementEvent
from rlm.refinement import AppliedEdit, RefinementResult

_DURATION = re.compile(r"^(\d+)([smhdw])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86_400, "w": 604_800}
PRUNE_TRIGGER = "operator:prune"


def parse_duration(text: str) -> float:
    """Seconds for ``30d``, ``12h``, ``4w``, ``90m`` or ``45s``."""
    match = _DURATION.match(text.strip())
    if match is None:
        raise ValueError(f"expected a duration like 30d, 12h or 4w, got {text!r}")
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def episode_started(entry: HarnessEntry) -> float:
    """When the episode's session started; the entry's write time for older records."""
    started = entry.metadata.get("started_at")
    if isinstance(started, (int, float)):
        return float(started)
    return time.mktime(time.strptime(entry.created_at[:19], "%Y-%m-%dT%H:%M:%S"))


def select_prunable(
    store: HarnessStore,
    *,
    older_than: float | None = None,
    keep: int | None = None,
    now: float | None = None,
) -> list[HarnessEntry]:
    """Episodes to remove, oldest first: those that started more than ``older_than``
    seconds ago, and everything beyond the ``keep`` newest. With neither selector
    nothing is selected."""
    if older_than is None and keep is None:
        return []
    now = time.time() if now is None else now
    episodes = sorted(store.list("episode"), key=episode_started)
    selected: dict[str, HarnessEntry] = {}
    if older_than is not None:
        for entry in episodes:
            if now - episode_started(entry) > older_than:
                selected[entry.id] = entry
    if keep is not None and len(episodes) > keep:
        for entry in episodes[: len(episodes) - keep]:
            selected[entry.id] = entry
    return sorted(selected.values(), key=episode_started)


def prune_episodes(
    store: HarnessStore, entries: list[HarnessEntry]
) -> RefinementResult:
    """Delete ``entries`` from ``store`` under its lock and record the prune both as a
    refinement event in the state file and as a full result (with each entry's
    snapshot) in ``refinements.jsonl``, so it is inspectable and reversible by hand."""
    result = RefinementResult(
        id=uuid.uuid4().hex,
        trigger=PRUNE_TRIGGER,
        scope=store.scope,
        summary=f"pruned {len(entries)} episode(s)",
        rationale="operator command",
        expected_outcome="older sessions no longer appear in the harness",
        applied_edits=[],
    )
    with store.transaction() as writer:
        records = writer.entries["episode"]
        for entry in entries:
            before = records.pop(entry.id, None)
            result.applied_edits.append(
                AppliedEdit(
                    action="delete",
                    kind="episode",
                    id=entry.id,
                    applied=before is not None,
                    error=None if before is not None else "entry not found",
                    before=before.model_dump() if before is not None else None,
                )
            )
        writer.refinements.append(
            RefinementEvent(
                id=result.id,
                trigger=PRUNE_TRIGGER,
                changes=[
                    f"delete episode:{e.id}" for e in result.applied_edits if e.applied
                ],
            )
        )
    store.append_result(asdict(result))
    return result
