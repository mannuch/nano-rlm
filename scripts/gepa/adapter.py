"""The GEPA adapter: nano-rlm sessions as rollouts, ledgers as reflective evidence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch

from components import validate_candidate
from rollout import Rollout, RolloutSettings, run_rollout
from tasks import Task

RECORD_CHARS = 2_400
"""Cap per reflective record; GEPA concatenates a whole minibatch into one prompt."""

COMPACTION_COMPONENTS = {"checkpoint", "rollup", "staircase_framing"}


def _clip(text: str, limit: int = RECORD_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 15] + " ...[truncated]"


class NanoRlmAdapter:
    propose_new_texts = None
    """GEPA's default reflective proposer writes the new texts."""

    def __init__(
        self,
        settings: RolloutSettings,
        session_root: Path,
        *,
        seed: dict[str, str],
        concurrency: int = 6,
    ):
        self.settings = settings
        self.session_root = session_root
        self.seed = seed
        self.concurrency = concurrency

    # -- evaluation -------------------------------------------------------------

    def evaluate(
        self, batch: list[Task], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        return self.batch_evaluate([(candidate, batch)], capture_traces=capture_traces)[
            0
        ]

    def batch_evaluate(
        self,
        items: list[tuple[dict[str, str], list[Task]]],
        *,
        capture_traces: bool = True,
    ) -> list[EvaluationBatch]:
        """Run every (candidate, task) pair concurrently under one semaphore."""
        jobs: list[tuple[int, int, dict[str, str], Task]] = []
        results: list[list[Rollout | None]] = []
        for item_index, (candidate, batch) in enumerate(items):
            results.append([None] * len(batch))
            problems = validate_candidate(candidate, self.seed)
            for task_index, task in enumerate(batch):
                if problems:
                    results[item_index][task_index] = Rollout(
                        task_id=task.id,
                        session_dir="",
                        error="candidate rejected before running: "
                        + "; ".join(f"{k}: {v}" for k, v in problems.items()),
                    )
                else:
                    jobs.append((item_index, task_index, candidate, task))
        if jobs:
            rollouts = asyncio.run(self._run_all(jobs))
            for (item_index, task_index, _, _), rollout in zip(jobs, rollouts):
                results[item_index][task_index] = rollout
        batches = []
        for item_rollouts in results:
            rollouts = [r for r in item_rollouts if r is not None]
            batches.append(
                EvaluationBatch(
                    outputs=[r.final_answers for r in rollouts],
                    scores=[r.score for r in rollouts],
                    trajectories=rollouts,
                )
            )
        return batches

    async def _run_all(
        self, jobs: list[tuple[int, int, dict[str, str], Task]]
    ) -> list[Rollout]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(candidate: dict[str, str], task: Task) -> Rollout:
            async with semaphore:
                rollout = await run_rollout(
                    candidate, task, self.settings, self.session_root
                )
                self._log(rollout)
                return rollout

        return await asyncio.gather(*(one(c, t) for _, _, c, t in jobs))

    def _log(self, rollout: Rollout) -> None:
        with open(
            self.session_root / "rollouts.jsonl", "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(rollout.to_json()) + "\n")

    # -- reflection ---------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        dataset: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            records = []
            for rollout in eval_batch.trajectories or []:
                record = self._record(component, rollout)
                if record is not None:
                    records.append(record)
            if records:
                dataset[component] = records
        return dataset

    def _record(self, component: str, rollout: Rollout) -> dict[str, Any] | None:
        if rollout.error and rollout.error.startswith("candidate rejected"):
            return {
                "Inputs": "(no rollout was run)",
                "Generated Outputs": "(none)",
                "Feedback": rollout.error,
            }
        if component in COMPACTION_COMPONENTS:
            return self._compaction_record(component, rollout)
        return self._session_record(component, rollout)

    def _session_record(self, component: str, rollout: Rollout) -> dict[str, Any]:
        questions = "\n".join(
            f"{i + 1}. {r.text}" for i, r in enumerate(rollout.results)
        )
        cells = "\n".join(
            f"- cell@{c.index}{' (after compaction)' if c.after_compaction else ''}"
            f"{' ERROR' if c.error else ''}: {_clip(c.code, 200)!r} -> {_clip(c.output, 160)!r}"
            for c in rollout.cells
        )
        answers = "\n".join(
            f"{i + 1}. {_clip(a, 300)}" for i, a in enumerate(rollout.final_answers)
        )
        feedback = [
            f"Session score {rollout.score:.2f} (correctness {rollout.correctness:.2f}); "
            f"{rollout.turns} model turns, {rollout.prompt_tokens + rollout.completion_tokens} tokens, "
            f"{sum(c.error for c in rollout.cells)} cells raised errors, "
            f"{len(rollout.compactions)} compactions, {len(rollout.spawns)} child agents spawned."
        ]
        for i, r in enumerate(rollout.results):
            line = f"Q{i + 1} [{r.kind}] score {r.score:.2f}: expected {r.expected!r}, got {r.answer!r}"
            if r.check is not None:
                line += f"; runtime check ({r.check}) {'passed' if r.check_ok else 'FAILED'}: {r.check_note}"
            feedback.append(line)
        if rollout.duplicate_cells:
            feedback.append(
                f"{len(rollout.duplicate_cells)} cell(s) after a compaction repeated a cell run before it "
                "(work redone because the summary did not carry the result forward)."
            )
        if rollout.error:
            feedback.append(f"Runtime error: {_clip(rollout.error, 400)}")
        if component in ("delegation_doctrine", "delegation_reference"):
            feedback.append(
                "Children spawned: "
                + ("; ".join(_clip(p, 160) for _, p in rollout.spawns) or "none")
            )
        return {
            "Inputs": _clip(questions),
            "Generated Outputs": _clip(
                f"Cells run:\n{cells}\n\nFinal answers:\n{answers}"
            ),
            "Feedback": _clip("\n".join(feedback)),
        }

    def _compaction_record(
        self, component: str, rollout: Rollout
    ) -> dict[str, Any] | None:
        if not rollout.compactions:
            return None
        if component == "rollup" and not rollout.rolled_up:
            return None
        post = [r for r in rollout.results if r.after_compaction]
        post_line = (
            f"{sum(r.score for r in post):.1f}/{len(post)} questions answered correctly after compaction"
            if post
            else "no questions were asked after the compaction"
        )
        dup_line = (
            f"{len(rollout.duplicate_cells)} cell(s) after compaction repeated earlier cells"
            if rollout.duplicate_cells
            else "no earlier cells were repeated after compaction"
        )
        first = rollout.compactions[0]
        if component == "checkpoint":
            branch_cells = [c for c in rollout.cells if c.index < first.message_index]
            branch = "\n".join(
                f"- {_clip(c.code, 160)!r} -> {_clip(c.output, 120)!r}"
                for c in branch_cells
            )
            answered = [
                f"Q{i + 1}: {r.answer!r}"
                for i, r in enumerate(rollout.results)
                if not r.after_compaction and r.answer is not None
            ]
            carried = [
                a for a in answered if a.split(": ", 1)[1].strip("'\"") in first.summary
            ]
            return {
                "Inputs": _clip(
                    f"Branch that was summarized ({len(branch_cells)} cells; "
                    f"answers so far: {', '.join(answered) or 'none'}):\n{branch}"
                ),
                "Generated Outputs": _clip(f"{first.header}\n{first.summary}"),
                "Feedback": _clip(
                    f"{post_line}; {dup_line}. Summary is {len(first.summary)} characters; "
                    f"{len(carried)}/{len(answered)} earlier answers appear verbatim in it. "
                    f"Tail kept verbatim: {first.tail_messages} messages; current prompt pinned: {first.pinned}. "
                    f"Session score {rollout.score:.2f}."
                ),
            }
        if component == "rollup":
            sealed = next(c for c in rollout.compactions if c.rollups)
            rollup = sealed.rollups[0]
            children = "\n\n".join(rollup["children"])
            return {
                "Inputs": _clip(f"Child summaries merged:\n{children}"),
                "Generated Outputs": _clip(f"{rollup['header']}\n{rollup['text']}"),
                "Feedback": _clip(
                    f"Rollup is {len(rollup['text'])} characters versus "
                    f"{max(len(c) for c in rollup['children'])} for the longest child. "
                    f"{post_line}; {dup_line}. Session score {rollout.score:.2f}."
                ),
            }
        after = [c for c in rollout.cells if c.after_compaction][:4]
        return {
            "Inputs": _clip(
                f"Compaction message as the agent saw it:\n{first.staircase_message}"
            ),
            "Generated Outputs": _clip(
                "First cells after compaction:\n"
                + "\n".join(f"- {_clip(c.code, 200)!r}" for c in after)
            ),
            "Feedback": _clip(
                f"{post_line}; {dup_line}. Session score {rollout.score:.2f}."
            ),
        }


class ExercisedComponentSelector:
    """Round-robin over the candidate's components, skipping any the minibatch did not
    exercise (compaction texts when nothing compacted, rollup when none sealed)."""

    def __init__(self) -> None:
        self._next: dict[int, int] = {}

    def __call__(self, state, trajectories, subsample_scores, candidate_idx, candidate):
        names = list(candidate)
        rollouts = [t for t in trajectories if isinstance(t, Rollout)]
        exercised = {
            name
            for name in names
            if name not in COMPACTION_COMPONENTS
            or (name == "rollup" and any(r.rolled_up for r in rollouts))
            or (name != "rollup" and any(r.compacted for r in rollouts))
        }
        start = self._next.get(candidate_idx, 0)
        for offset in range(len(names)):
            index = (start + offset) % len(names)
            if names[index] in exercised:
                self._next[candidate_idx] = (index + 1) % len(names)
                return [names[index]]
        self._next[candidate_idx] = (start + 1) % len(names)
        return [names[start % len(names)]]
