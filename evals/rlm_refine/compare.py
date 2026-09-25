"""Compare vf-eval runs of the same taskset, paired by task.

    python compare.py off=outputs/<run>/traces.jsonl planner=... gate=...
    python compare.py --metric=required_tests_passed off=... planner=... gate=...
    python compare.py --metric=cost off=... planner=... gate=...

Rollouts whose trace carries an error (provider failures, for example) are left out;
tasks are paired over those with a scored rollout in every arm. Differences are per-task
means of the reward, of a numeric trace metric with `--metric`, or of the rollout's cost
in dollars with `--metric=cost`, with 95% intervals bootstrapped over tasks.

Cost is what the provider billed for the task model's calls (each call's `usage.cost`,
cached reads at their discount), plus the TypeSafe judge's input tokens at its list
price. Verifiers' `num_total_tokens` counts each distinct input token once, so it leaves
out the repeated, cached re-reads of the conversation that make up most of a rollout's
bill.
"""

import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict

TYPESAFE_USD_PER_INPUT_TOKEN = 0.042 / 1e6
"""TypeSafe's list price, $0.042 per 1M input tokens; output is free."""

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


def load(path: str, metric: str) -> list[dict]:
    rows = []
    for line in open(path):
        episode = json.loads(line)
        trace = episode["traces"][0]
        errors = trace.get("errors") or []
        metrics = trace.get("metrics") or {}
        reward = (trace.get("rewards") or {}).get("solved", {}).get("score")
        usage = [call["usage"] for call in trace.get("calls") or [] if call.get("usage")]
        model_cost = sum(u.get("cost") or 0 for u in usage)
        judge_cost = (metrics.get("judge_input_tokens") or 0) * TYPESAFE_USD_PER_INPUT_TOKEN
        values = {"reward": reward, "cost": model_cost + judge_cost}
        rows.append(
            {
                "task": episode["task"]["data"]["name"],
                "error": errors[0].get("type") if errors else None,
                "value": values[metric] if metric in values else metrics.get(metric),
                "stop": trace.get("stop_condition"),
                "model_cost": model_cost,
                "judge_cost": judge_cost,
                "input": sum(u.get("prompt_tokens", 0) + u.get("cached_input_tokens", 0) for u in usage),
                "cached": sum(u.get("cached_input_tokens", 0) for u in usage),
                "metrics": metrics,
            }
        )
    return rows


def bootstrap(diffs: list[float], n: int = 10_000) -> tuple[float, float, float]:
    rng = random.Random(0)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


def main(args: list[str]) -> None:
    metric = "reward"
    paths = {}
    for arg in args:
        key, value = arg.split("=", 1)
        if key == "--metric":
            metric = value
        else:
            paths[key] = value
    arms = {name: load(path, metric) for name, path in paths.items()}
    scored = {
        name: [r for r in rows if not r["error"] and r["value"] is not None]
        for name, rows in arms.items()
    }

    print(
        "| arm | rollouts | scored | errors | stops | $/rollout | model $ | judge $ "
        "| input tokens/rollout | cached |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, rows in arms.items():
        errors = Counter(r["error"] for r in rows if r["error"])
        stops = Counter(r["stop"] for r in rows)
        ok = scored[name] or [{"model_cost": 0, "judge_cost": 0, "input": 0, "cached": 0}]
        model = st.mean(r["model_cost"] for r in ok)
        judge = st.mean(r["judge_cost"] for r in ok)
        cached = sum(r["cached"] for r in ok) / max(1, sum(r["input"] for r in ok))
        print(
            f"| {name} | {len(rows)} | {len(scored[name])} | {dict(errors) or '-'} "
            f"| {dict(stops)} | {model + judge:.4f} | {model:.4f} | {judge:.4f} "
            f"| {st.mean(r['input'] for r in ok):,.0f} | {cached:.0%} |"
        )

    per_task = {name: defaultdict(list) for name in arms}
    for name, rows in scored.items():
        for r in rows:
            per_task[name][r["task"]].append(r["value"])
    tasks = sorted(set.intersection(*(set(t) for t in per_task.values())))
    means = {name: {t: st.mean(per_task[name][t]) for t in tasks} for name in arms}
    print(f"\nPaired over {len(tasks)} tasks scored in every arm.\n")
    print(f"| arm | mean {metric} |")
    print("|---|---|")
    for name in arms:
        print(f"| {name} | {st.mean(means[name].values()):.4f} |")

    print("\n| comparison | difference | 95% interval | tasks higher / lower / same |")
    print("|---|---|---|---|")
    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diffs = [means[b][t] - means[a][t] for t in tasks]
            mean, low, high = bootstrap(diffs)
            higher = sum(d > 0 for d in diffs)
            lower = sum(d < 0 for d in diffs)
            print(
                f"| {b} − {a} | {mean:+.4f} | {low:+.4f} to {high:+.4f} "
                f"| {higher} / {lower} / {len(diffs) - higher - lower} |"
            )

    print("\n| arm | " + " | ".join(REFINE_METRICS) + " |")
    print("|---|" + "---|" * len(REFINE_METRICS))
    for name, rows in scored.items():
        totals = [sum(r["metrics"].get(k) or 0 for r in rows) for k in REFINE_METRICS]
        print(f"| {name} | " + " | ".join(f"{int(v):,}" for v in totals) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
