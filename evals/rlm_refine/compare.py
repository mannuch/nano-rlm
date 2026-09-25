"""Compare vf-eval runs of the same taskset, paired by task.

    python compare.py off=outputs/<run>/traces.jsonl planner=... gate=...

Rollouts whose trace carries an error (provider failures, for example) are left out;
tasks are paired over those with a scored rollout in every arm. Differences are per-task
mean rewards, with 95% intervals bootstrapped over tasks.
"""

import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict

REFINE_METRICS = (
    "num_compactions",
    "num_auto_refine_reviews",
    "num_refinements",
    "num_refinements_declined_gate",
    "num_refinements_declined_no_edits",
    "num_refinements_failed",
    "num_judge_reviews",
    "num_judge_errors",
    "judge_input_tokens",
)


def load(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        episode = json.loads(line)
        trace = episode["traces"][0]
        errors = trace.get("errors") or []
        rows.append(
            {
                "task": episode["task"]["data"]["name"],
                "error": errors[0].get("type") if errors else None,
                "reward": (trace.get("rewards") or {}).get("solved", {}).get("score"),
                "stop": trace.get("stop_condition"),
                "tokens": trace.get("num_total_tokens") or 0,
                "metrics": trace.get("metrics") or {},
            }
        )
    return rows


def bootstrap(diffs: list[float], n: int = 10_000) -> tuple[float, float, float]:
    rng = random.Random(0)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


def main(args: list[str]) -> None:
    arms = {name: load(path) for name, path in (arg.split("=", 1) for arg in args)}
    scored = {
        name: [r for r in rows if not r["error"] and r["reward"] is not None]
        for name, rows in arms.items()
    }

    print("| arm | rollouts | scored | errors | stops | tokens/rollout |")
    print("|---|---|---|---|---|---|")
    for name, rows in arms.items():
        errors = Counter(r["error"] for r in rows if r["error"])
        stops = Counter(r["stop"] for r in rows)
        tokens = st.mean(r["tokens"] for r in scored[name]) if scored[name] else 0
        print(
            f"| {name} | {len(rows)} | {len(scored[name])} | {dict(errors) or '-'} "
            f"| {dict(stops)} | {tokens:,.0f} |"
        )

    per_task = {name: defaultdict(list) for name in arms}
    for name, rows in scored.items():
        for r in rows:
            per_task[name][r["task"]].append(r["reward"])
    tasks = sorted(set.intersection(*(set(t) for t in per_task.values())))
    means = {name: {t: st.mean(per_task[name][t]) for t in tasks} for name in arms}
    print(f"\nPaired over {len(tasks)} tasks scored in every arm.\n")
    print("| arm | mean reward |")
    print("|---|---|")
    for name in arms:
        print(f"| {name} | {st.mean(means[name].values()):.3f} |")

    print("\n| comparison | difference | 95% interval | tasks better / worse / same |")
    print("|---|---|---|---|")
    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diffs = [means[b][t] - means[a][t] for t in tasks]
            mean, low, high = bootstrap(diffs)
            better = sum(d > 0 for d in diffs)
            worse = sum(d < 0 for d in diffs)
            print(
                f"| {b} − {a} | {mean:+.3f} | {low:+.3f} to {high:+.3f} "
                f"| {better} / {worse} / {len(diffs) - better - worse} |"
            )

    print("\n| arm | " + " | ".join(REFINE_METRICS) + " |")
    print("|---|" + "---|" * len(REFINE_METRICS))
    for name, rows in scored.items():
        totals = [sum(r["metrics"].get(k) or 0 for r in rows) for k in REFINE_METRICS]
        print(f"| {name} | " + " | ".join(f"{int(v):,}" for v in totals) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
