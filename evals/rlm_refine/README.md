# rlm-refine

This runs nano-rlm with automatic refinement on, as a
[verifiers v1](https://docs.primeintellect.ai/verifiers/v1/overview) harness, on two
tasksets from prime-envs:

- [OpenThoughts TBLite](https://github.com/PrimeIntellect-ai/prime-envs/tree/main/environments/terminal/openthoughts_tblite):
  100 terminal tasks, each in its own prebuilt container and scored by the task's hidden
  pass/fail verifier.
- [SWE-bench Pro V2](https://github.com/PrimeIntellect-ai/prime-envs/tree/main/environments/swe/swebench_pro):
  harder, long-context repository work. See "SWE-bench Pro" below.

The first experiment is a **shadow run**. The planner (the task model) decides every
automatic refinement pass. The TypeSafe judge reviews the same evidence, but its verdict is
only recorded. The run measures:

- how often each wants to refine over long, real trajectories
- where they disagree
- what the judge costs

## What's in the package

The `rlm_refine` module exports the harness and the TBLite taskset. `rlm_refine_swebench_pro`
exports the SWE-bench Pro taskset.

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
export PRIME_API_KEY=...        # or run `prime login` once (see below)
export TYPESAFE_API_KEY=...     # the judge, from https://console.typesafe.ai/
```

The Prime key covers the sandboxes, the tunnel from the sandboxes back to your machine,
and the task model, which runs on Prime Inference (verifiers' default client). Instead of
exporting it, you can run `prime login` once. All three read `api_key` from
`~/.prime/config.json` when `PRIME_API_KEY` is unset, and an exported `PRIME_API_KEY` takes
precedence. There's no such fallback for `TYPESAFE_API_KEY`, so always export it.

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
  Passes also run after each compaction, which `policy.summarize_at_tokens` (25k) makes
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

## SWE-bench Pro

`rlm-refine-swebench-pro` is prime-envs' SWE-bench Pro V2 taskset with one change: every
task also allowlists `api.typesafe.ai`.

- **Why it's needed:** the V2 protocol runs the agent offline, apart from hosts a task
  allowlists, and the judge calls TypeSafe from inside the sandbox. Everything else, PyPI
  included, stays blocked, as the benchmark intends. nano-rlm's install runs during setup,
  which stays online.
- **Grading is unchanged:** the agent's changes to the repository are captured as a diff
  and replayed in a fresh box, where the upstream verifier runs. The module also exports
  prime-envs' `HarborEnv`, which does that.
- **Configs:** the `swe_*.toml` configs run the 51-task HARD-51 subset (`subset = "hard51"`):
  - `configs/swe_smoke.toml`: 3 tasks, the judge gating
  - `configs/swe_off.toml`, `swe_planner.toml`, `swe_gate.toml`: the three arms, 2 rollouts
    per task

```bash
uv run vf-eval @ configs/swe_smoke.toml
uv run vf-eval @ configs/swe_off.toml
uv run vf-eval @ configs/swe_planner.toml
uv run vf-eval @ configs/swe_gate.toml
```

Few HARD-51 tasks may pass outright. The taskset also records `required_tests_passed`, the
fraction of the task's required tests that pass. Compare it too, with
`python compare.py --metric=required_tests_passed ...`.

The traces record the captured diff in `info.model_patch`. Check in the smoke run that it
holds only the agent's changes to the repository: nano-rlm's own files live under `/tmp`.

## Reward comparison

Three configs run the same tasks and differ only in nano-rlm's `harness` table:

| config | refinement |
|---|---|
| `configs/ab_off.toml` | auto-refinement off (the baseline) |
| `configs/ab_planner.toml` | auto-refinement on; the planner (task model) decides every pass |
| `configs/ab_gate.toml` | auto-refinement on; the TypeSafe judge gates every pass first |

```bash
uv run vf-eval @ configs/ab_off.toml
uv run vf-eval @ configs/ab_planner.toml
uv run vf-eval @ configs/ab_gate.toml
```

Each config runs 100 tasks with 3 rollouts per task. Compare pass rates per task across the
arms (paired by task), not only in aggregate. Tasks differ far more from one another than
the arms are likely to. `compare.py` does this:

```bash
uv run python compare.py off=outputs/<off run>/traces.jsonl \
  planner=outputs/<planner run>/traces.jsonl gate=outputs/<gate run>/traces.jsonl
```

It leaves out rollouts that errored, such as provider failures, and pairs the tasks scored in
every arm. It reports each pairwise difference in mean reward, with a 95% interval
bootstrapped over tasks, then the refinement and judge totals per arm.

Every config uses `deepseek/deepseek-v4.1-flash`. Don't use
`deepseek/deepseek-v4-flash` here. On Prime Inference it ignores `tool_choice="none"`,
which nano-rlm's compaction and refinement calls rely on. In the first shadow run, 184
of 610 of those calls came back as tool calls, and 9 of 100 rollouts stopped with
`compaction_failed`.


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
