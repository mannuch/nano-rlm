"""Optimize nano-rlm's prompt texts with GEPA against repo-grounded question sessions.

    uv run --group gepa python scripts/gepa/optimize.py --tasks tasks.jsonl \\
        --run-dir scripts/gepa/runs/first --model deepseek/deepseek-v4.1-flash \\
        --reflection-model openai/gpt-6-astra --max-metric-calls 300

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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gepa  # noqa: E402
from openai import OpenAI  # noqa: E402

from adapter import ExercisedComponentSelector, NanoRlmAdapter  # noqa: E402
from components import DEFAULT_COMPONENTS, reflection_template  # noqa: E402
from rlm.prompt import DEFAULT_PROMPTS  # noqa: E402
from rollout import RolloutSettings  # noqa: E402
from tasks import read_tasks  # noqa: E402


class ReflectionLM:
    """A GEPA ``LanguageModel``: one chat completion per reflection prompt, with the
    provider-reported token usage accumulated across the run and every exchange appended
    to ``log_path``."""

    def __init__(self, model: str, api_key: str, base_url: str | None, log_path: Path):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.log_path = log_path
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def __call__(self, prompt):
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        self.calls += 1
        usage = response.usage
        if usage is not None:
            self.tokens_in += usage.prompt_tokens or 0
            self.tokens_out += usage.completion_tokens or 0
        choice = response.choices[0]
        content = choice.message.content or ""
        with open(self.log_path, "a", encoding="utf-8") as handle:
            record = {
                "time": time.time(),
                "model": self.model,
                "finish_reason": choice.finish_reason,
                "refusal": getattr(choice.message, "refusal", None),
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "messages": messages,
                "response": content,
            }
            handle.write(json.dumps(record) + "\n")
        return content

    def summary(self) -> str:
        return (
            f"{self.calls} reflection call(s) on {self.model}: "
            f"{self.tokens_in} input / {self.tokens_out} output tokens"
        )


def write_report(
    result, seed: dict[str, str], valset, run_dir: Path, reflection: str
) -> None:
    best = result.best_candidate
    lines = [
        "# GEPA run report",
        "",
        f"Candidates: {len(result.candidates)}; metric calls: {result.total_metric_calls}; "
        f"best candidate #{result.best_idx} with validation score "
        f"{result.val_aggregate_scores[result.best_idx]:.3f} "
        f"(seed {result.val_aggregate_scores[0]:.3f}).",
        "",
        f"Reflection usage this invocation: {reflection}.",
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("RLM_BASE_URL"))
    parser.add_argument("--reflection-model", required=True)
    parser.add_argument(
        "--components",
        default=",".join(DEFAULT_COMPONENTS),
        help="Comma-separated registry names to optimize",
    )
    parser.add_argument("--max-metric-calls", type=int, default=300)
    parser.add_argument("--minibatch", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--summarize-at", type=int, default=10_000)
    parser.add_argument("--tail", type=int, default=1_000)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument(
        "--delegation-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run sessions with policy.delegation_prompt (the guidance itself is not optimized)",
    )
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
        delegation_prompt=args.delegation_prompt,
        timeout_s=args.timeout,
    )
    adapter = NanoRlmAdapter(
        settings, session_root, seed=seed, concurrency=args.concurrency
    )
    print(
        f"train {len(trainset)} / val {len(valset)} tasks; components {components}; run_dir {run_dir}"
    )

    reflection_lm = ReflectionLM(
        args.reflection_model, api_key, args.base_url, run_dir / "reflection_log.jsonl"
    )
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
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
    write_report(result, seed, valset, run_dir, reflection_lm.summary())
    print(
        f"best candidate #{result.best_idx}: val {result.val_aggregate_scores[result.best_idx]:.3f} "
        f"(seed {result.val_aggregate_scores[0]:.3f}); {len(overrides)} component(s) changed; "
        f"see {run_dir / 'report.md'}"
    )
    print(reflection_lm.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
