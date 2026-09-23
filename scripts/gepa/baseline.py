"""Measure the current prompts on a task file before spending an optimization run on it.

    uv run python scripts/gepa/baseline.py --tasks scripts/gepa/tasks/bugs.jsonl \\
        --run-dir scripts/gepa/runs/pilot --model deepseek/deepseek-v4.1-flash

Runs every task once with no prompt overrides and prints what decides whether the task
set is worth optimizing against: the score per question kind (a set the seed nearly aces
cannot separate candidates), tokens, turns and compactions per rollout, and errors.
Rollouts are appended to ``<run-dir>/rollouts.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout import Rollout, RolloutSettings, run_rollout  # noqa: E402
from tasks import read_tasks  # noqa: E402


def summarize(rollouts: list[Rollout]) -> str:
    kinds: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        for result in rollout.results:
            if not result.skipped:
                kinds[result.check or result.kind].append(result.score)
    n = len(rollouts)
    lines = [
        f"{n} rollouts; mean correctness "
        f"{statistics.mean(r.correctness for r in rollouts):.3f}, "
        f"mean score {statistics.mean(r.score for r in rollouts):.3f}",
        f"per rollout: {statistics.mean(r.prompt_tokens for r in rollouts) / 1e3:.0f}k input / "
        f"{statistics.mean(r.completion_tokens for r in rollouts) / 1e3:.1f}k output tokens, "
        f"{statistics.mean(r.turns for r in rollouts):.1f} turns, "
        f"{statistics.mean(len(r.compactions) for r in rollouts):.2f} compactions, "
        f"{sum(r.rolled_up for r in rollouts)}/{n} sealed a rollup",
        f"errors: {sum(bool(r.error) for r in rollouts)}",
        "",
        "score by question kind:",
    ]
    for kind, scores in sorted(
        kinds.items(), key=lambda item: statistics.mean(item[1])
    ):
        solved = sum(score == 1 for score in scores)
        lines.append(
            f"  {kind:18} mean {statistics.mean(scores):.3f}, fully solved {solved}/{len(scores)}"
        )
    notes = [
        result.check_note
        for rollout in rollouts
        for result in rollout.results
        if result.kind == "bugfix" and result.score < 1
    ]
    if notes:
        lines += ["", "unsolved fixes:"] + [f"  {note[:160]}" for note in notes]
    return "\n".join(lines)


async def _run(tasks, settings: RolloutSettings, run_dir: Path, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)
    log = run_dir / "rollouts.jsonl"

    async def one(task) -> Rollout:
        async with semaphore:
            rollout = await run_rollout({}, task, settings, run_dir / "sessions")
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(rollout.to_json()) + "\n")
            print(
                f"{task.id}: correctness {rollout.correctness:.2f}, "
                f"{rollout.prompt_tokens // 1000}k in, {len(rollout.compactions)} compactions"
                + (f", error {rollout.error.splitlines()[0]}" if rollout.error else ""),
                flush=True,
            )
            return rollout

    return await asyncio.gather(*(one(task) for task in tasks))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("RLM_BASE_URL"))
    parser.add_argument("--limit", type=int, help="Only the first N tasks")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--summarize-at", type=int, default=9_000)
    parser.add_argument("--tail", type=int, default=1_000)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    api_key = os.environ.get("RLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("set RLM_API_KEY or OPENAI_API_KEY")
    tasks = read_tasks(Path(args.tasks))[: args.limit]
    run_dir = Path(args.run_dir)
    (run_dir / "sessions").mkdir(parents=True, exist_ok=True)
    settings = RolloutSettings(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        summarize_at_tokens=args.summarize_at,
        compaction_tail_tokens=args.tail,
        max_depth=args.max_depth,
        timeout_s=args.timeout,
    )
    rollouts = asyncio.run(_run(tasks, settings, run_dir, args.concurrency))
    report = summarize(rollouts)
    (run_dir / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
