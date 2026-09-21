"""Optimize nano-rlm's prompt texts with GEPA against repo-grounded question sessions.

    uv run --group gepa python scripts/gepa/optimize.py --tasks tasks.jsonl \\
        --run-dir scripts/gepa/runs/first --model openai/gpt-4.1-mini \\
        --reflection-model anthropic/claude-sonnet-4.5 --max-metric-calls 300

Re-running with the same ``--run-dir`` resumes from ``gepa_state.bin``. The result is a
``prompt_overrides`` object (``best_prompt_overrides.json``) plus ``report.md``; landing
a winner in ``rlm.prompt`` is a reviewed change, never automatic.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gepa  # noqa: E402
from openai import OpenAI  # noqa: E402

from adapter import ExercisedComponentSelector, NanoRlmAdapter  # noqa: E402
from components import FIRST_RUN_COMPONENTS, reflection_template  # noqa: E402
from rlm.prompt import DEFAULT_PROMPTS  # noqa: E402
from rollout import RolloutSettings  # noqa: E402
from tasks import read_tasks  # noqa: E402


class ReflectionLM:
    """A GEPA ``LanguageModel``: one chat completion per reflection prompt."""

    def __init__(self, model: str, api_key: str, base_url: str | None):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(self, prompt):
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content or ""


def write_report(result, seed: dict[str, str], valset, run_dir: Path) -> None:
    best = result.best_candidate
    lines = [
        "# GEPA run report",
        "",
        f"Candidates: {len(result.candidates)}; metric calls: {result.total_metric_calls}; "
        f"best candidate #{result.best_idx} with validation score "
        f"{result.val_aggregate_scores[result.best_idx]:.3f} "
        f"(seed {result.val_aggregate_scores[0]:.3f}).",
        "",
        "## Per-task validation scores (seed -> best)",
        "",
    ]
    seed_scores = result.val_subscores[0]
    best_scores = result.val_subscores[result.best_idx]
    for key in sorted(seed_scores, key=str):
        task_id = valset[key].id if isinstance(key, int) and key < len(valset) else key
        lines.append(
            f"- {task_id}: {seed_scores[key]:.2f} -> {best_scores.get(key, float('nan')):.2f}"
        )
    lines += ["", "## Component diffs (seed -> best)", ""]
    for name, text in best.items():
        if text == seed[name]:
            lines.append(f"### {name}\n\nunchanged\n")
            continue
        diff = difflib.unified_diff(
            seed[name].splitlines(), text.splitlines(), "seed", "best", lineterm="", n=1
        )
        lines.append(f"### {name}\n\n```diff\n" + "\n".join(diff) + "\n```\n")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", required=True, help="tasks.jsonl from tasks.py")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.33,
        help="Share of tasks held out for validation (split by task id)",
    )
    parser.add_argument(
        "--model", default=os.environ.get("RLM_MODEL", "openai/gpt-4.1-mini")
    )
    parser.add_argument("--base-url", default=os.environ.get("RLM_BASE_URL"))
    parser.add_argument("--reflection-model", required=True)
    parser.add_argument(
        "--components",
        default=",".join(FIRST_RUN_COMPONENTS),
        help="Comma-separated registry names to optimize",
    )
    parser.add_argument("--max-metric-calls", type=int, default=300)
    parser.add_argument("--minibatch", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--summarize-at", type=int, default=7_000)
    parser.add_argument("--tail", type=int, default=1_000)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="Seconds per question prompt"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    api_key = os.environ.get("RLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("set RLM_API_KEY or OPENAI_API_KEY")
    components = [name.strip() for name in args.components.split(",") if name.strip()]
    unknown = [name for name in components if name not in DEFAULT_PROMPTS]
    if unknown:
        parser.error(
            f"unknown components {unknown}; registry: {sorted(DEFAULT_PROMPTS)}"
        )
    seed = {name: DEFAULT_PROMPTS[name] for name in components}

    tasks = read_tasks(Path(args.tasks))
    split = max(1, int(len(tasks) * args.val_fraction))
    ordered = sorted(tasks, key=lambda t: t.id)
    valset = ordered[:: max(1, len(ordered) // split)][:split]
    val_ids = {t.id for t in valset}
    trainset = [t for t in ordered if t.id not in val_ids]
    if not trainset:
        parser.error("no training tasks left after the validation split")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    session_root = run_dir / "sessions"
    session_root.mkdir(exist_ok=True)
    settings = RolloutSettings(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        summarize_at_tokens=args.summarize_at,
        compaction_tail_tokens=args.tail,
        max_depth=args.max_depth,
        timeout_s=args.timeout,
    )
    adapter = NanoRlmAdapter(
        settings, session_root, seed=seed, concurrency=args.concurrency
    )
    print(
        f"train {len(trainset)} / val {len(valset)} tasks; components {components}; run_dir {run_dir}"
    )

    result = gepa.optimize(
        seed_candidate=seed,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=ReflectionLM(args.reflection_model, api_key, args.base_url),
        module_selector=ExercisedComponentSelector(),
        reflection_prompt_template={
            name: reflection_template(name, seed) for name in components
        },
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        reflection_minibatch_size=args.minibatch,
        use_merge=True,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(run_dir),
        seed=args.seed,
        raise_on_exception=False,
        display_progress_bar=False,
    )

    best = result.best_candidate
    overrides = {name: text for name, text in best.items() if text != seed[name]}
    (run_dir / "best_prompt_overrides.json").write_text(
        json.dumps(overrides, indent=2), encoding="utf-8"
    )
    write_report(result, seed, valset, run_dir)
    print(
        f"best candidate #{result.best_idx}: val {result.val_aggregate_scores[result.best_idx]:.3f} "
        f"(seed {result.val_aggregate_scores[0]:.3f}); {len(overrides)} component(s) changed; "
        f"see {run_dir / 'report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
