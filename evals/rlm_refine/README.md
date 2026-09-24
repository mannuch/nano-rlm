# rlm-refine

This runs nano-rlm with automatic refinement on, as a
[verifiers v1](https://docs.primeintellect.ai/verifiers/v1/overview) harness, on
[OpenThoughts TBLite](https://github.com/PrimeIntellect-ai/prime-envs/tree/main/environments/terminal/openthoughts_tblite).
TBLite has 100 terminal tasks, each in its own prebuilt container and scored by the task's
hidden pass/fail verifier.

The first experiment is a **shadow run**. The planner (the task model) decides every
automatic refinement pass. The TypeSafe judge reviews the same evidence, but its verdict is
only recorded. The run measures:

- how often each wants to refine over long, real trajectories
- where they disagree
- what the judge costs

## What's in the package

The package exports a harness and a taskset.

- **`RlmRefineHarness`** installs nano-rlm from a git commit (`repo`, `version`) and runs it
  over ACP.
  - It sends two tables to nano-rlm unchanged: `policy` becomes runtime-v1 `policy`, and
    `harness` becomes runtime-v1 `harness` (`auto_refine`, `refine_judge`, …). nano-rlm
    validates both when the session starts.
  - It records nano-rlm's session metrics on the trace.
  - It stores the session's refinement passes, applied or declined, with any judge
    verdict, in `trace.info["rlm_refinements"]`.
- **`RlmRefineTaskset`** is TBLite with one change: any network-restricted task also
  allowlists `api.typesafe.ai`, because the judge is called from inside the sandbox.
  - A runtime allowlist can't do this. Verifiers intersects it with the task's own policy,
    so an offline task stays offline.
  - At the pinned prime-envs commit every task is public, so this changes nothing yet.
  - The run log reports how many tasks were widened.

This is its own uv project, not a dependency group of nano-rlm. The verifiers commit it
needs pins `mcp==2.0.0`, which conflicts with nano-rlm's `mcp<2`. On the host, nano-rlm
isn't installed at all; it's installed only inside each sandbox.

## Setup

From this directory:

```bash
uv sync --python 3.12
export PRIME_API_KEY=...        # task model, through Prime inference (verifiers' default client)
export TYPESAFE_API_KEY=...     # the judge, from https://console.typesafe.ai/
```

`forward_env = ["TYPESAFE_API_KEY"]` in the configs hands the key to the harness. The
harness puts it into the judge's config at launch, so it never appears in the saved config.
A judge config without the key fails in the harness's `setup`.

The configs pin `version` to a nano-rlm commit. The sandbox fetches that commit from
`repo` (your fork by default), so **it has to be pushed**. Bump it when nano-rlm changes;
installs are cached per commit.

## Run

```bash
uv run vf-eval @ configs/smoke.toml --dry-run   # resolves the taskset, harness and tables
uv run vf-eval @ configs/smoke.toml             # 3 tasks
uv run vf-eval @ configs/shadow.toml            # all 100 tasks
```

Both configs use the `prime` runtime: TBLite's images are Prime platform images that local
Docker can't pull.

- **Model:** `model` picks the task model; override it with `-m <id>`.
- **Refinement cadence:** `refine_turn_interval` (8) sets how often an automatic pass runs.
  Passes also run after each compaction, which `policy.summarize_at_tokens` (60k) makes
  happen within a task.
- **Scope:** only local harness stores are used. There's no `global_dir`, so nothing
  carries across tasks.

Output goes to `outputs/<taskset>--<model>--<harness>/<uuid>/`, and `traces.jsonl` holds
one trace per task. `--resume <output-dir>` re-runs missing or errored rollouts.

## Smoke checks

- Rollouts finish and have a reward.
- The traces' metrics include `num_auto_refine_reviews` > 0 on longer tasks, and
  `num_judge_errors` = 0.
- `info.rlm_refinements` holds `auto:` passes, each with a `judge` object carrying
  `mode: "shadow"`, `gate_decision` and per-call `usage`. TypeSafe input tokens stay well
  under 32k per call.

## Reading the results

All counts are per trace; sum or average them across `traces.jsonl`.

| metric | meaning |
|---|---|
| `num_auto_refine_reviews` | automatic passes that reached a decision |
| `num_refinements` | passes that applied edits |
| `num_refinements_declined_no_edits` | passes where the planner proposed no edits |
| `num_judge_reviews` | passes the judge reviewed |
| `num_judge_errors` | judge calls that failed; the planner still decided |
| `judge_input_tokens` | TypeSafe input tokens |
| `num_judge_would_decline_applied` | the planner edited, but the judge would have declined |
| `num_judge_would_refine_declined` | the planner declined, but the judge would have refined |

Agreement is 1 − (the two disagreement counts ÷ `num_judge_reviews`).

To see why they disagree, read the passes in `info.rlm_refinements`. Each holds:
- `trigger`, e.g. `auto:turn_interval` or `auto:compact`
- the planner's `rationale`, and its edits under `result` when it applied them
- the judge's `gate` probabilities, `fired` signals, focus `instructions` and
  `evidence_turns`

The pass rate (the reward) is task performance with the planner deciding. Comparing it
against a run with auto-refinement off or with `refine_judge.mode = "gate"` is the next
experiment.
