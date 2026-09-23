# Running the refinement eval

Everything runs from the repository root. The eval checks whether harness refinement
works: whether a pass refines when it should, targets the right thing and writes the
right edits. See "Refinement" and "TypeSafe review judge" in the README for the
machinery under test.

Each case is one live session:

1. It runs in its own fixture workspace, with fresh local and global harness stores.
2. Seeded entries are written into those stores.
3. The scenario's steering prompts run as ordinary turns.
4. The arm's host refinement is triggered (`ai.prime.rlm/refine-v1`, empty prompt).
5. A probe question is optionally asked.
6. The case is graded from the session ledger and the final state of both stores.

## 1. Sync

```bash
git pull
uv sync
```

`typesafe-sdk` is a regular dependency, so there is no extra group.

## 2. Check the pieces offline

These tests need no credentials. They cover the judge's evidence, questions and policy,
the grading, and one global-pass case run through the driver against a scripted client.

```bash
uv run pytest tests/test_refine_judge.py tests/test_refine_eval.py tests/test_refinement.py
```

## 3. Credentials

```bash
export RLM_API_KEY=...                          # task model (or OPENAI_API_KEY)
export RLM_BASE_URL=https://openrouter.ai/api/v1
export TYPESAFE_API_KEY=...                     # judge, from https://console.typesafe.ai/
```

The script refuses to start without both keys, even for arms that never call the judge.

## 4. Smoke run

Start with one positive and one negative scenario, one repeat, and the planner-only and
judge arms: 4 cases, a few minutes.

```bash
ulimit -n 4096
uv run python scripts/refine_eval/run.py \
  --model deepseek/deepseek-v4.1-flash \
  --run-dir scripts/refine_eval/runs/smoke \
  --scenarios user_correction,one_off --arms force,typesafe
```

Each case prints one line as it finishes (`user_correction typesafe #0: decision=True`).
Before going further, check:

- No case printed `error`.
- `results.jsonl` rows for the `typesafe` arm have a `judge` object with `gate`
  probabilities. `judge_error` should be null.
- In a session ledger, the `refinement` or `refinement_declined` record carries
  `judge.usage` per call, with input tokens well under 32k.

## 5. Full run

Every scenario, every arm, 3 repeats: 11 scenarios × 3 refining arms, plus the `none`
control for the 3 scenarios with a probe, is 36 cases per repeat and 108 in total. Run
it in `tmux` or under `nohup`:

```bash
ulimit -n 4096
nohup uv run python scripts/refine_eval/run.py \
  --model deepseek/deepseek-v4.1-flash \
  --run-dir scripts/refine_eval/runs/first \
  --repeats 3 --concurrency 4 \
  > scripts/refine_eval/runs/first.log 2>&1 < /dev/null &
disown
```

Options:

| flag | default | meaning |
|---|---|---|
| `--scenarios` | all | comma-separated ids (see below) |
| `--arms` | all | `none`, `force`, `force+focus`, `typesafe` |
| `--repeats` | 1 | trajectories are stochastic even when steered; use 3 or more for numbers you act on |
| `--concurrency` | 4 | cases in flight; each is an IPython kernel, plus children in `delegation_role` |
| `--threshold` | 0.7 | the judge's gate/flag threshold (`veto_threshold` 0.8 and `home_confidence` 0.6 stay at their defaults) |
| `--timeout` | 600 | seconds per prompt |
| `--base-url` | `$RLM_BASE_URL` | task-model endpoint |

Time and cost have not been measured yet; record them from the first full run here.
Each case costs:

- **Task model:** one call or more per steering prompt, one planning call, and one or
  more for the probe.
- **TypeSafe** (`force+focus` and `typesafe` arms only): one or two calls per pass.

If the log shows 429 retries, lower `--concurrency`.

`--run-dir` must be new or empty. There is no resume: a stopped run keeps the rows it
finished in `results.jsonl`, and `--report-only` (step 7) reports on them. Re-run the
missing scenarios into a new directory.

## 6. Monitor

```bash
tail -f scripts/refine_eval/runs/first.log            # one line per finished case, plus tracebacks
wc -l scripts/refine_eval/runs/first/results.jsonl    # cases finished
```

## 7. Results

`scripts/refine_eval/runs/first/report.md` is rebuilt at the end of a run. To rebuild
it from a partial or edited `results.jsonl`:

```bash
uv run python scripts/refine_eval/run.py --run-dir scripts/refine_eval/runs/first --report-only
```

Errored cases and failed passes are counted in the header and left out of every table.

| section | what it answers |
|---|---|
| Decision vs label | Per refining arm: accuracy, precision and recall of "applied edits" against the scenario's label, and the share of negative scenarios declined. |
| Pass outcomes | Per arm and label: how many passes applied edits, were declined by the judge's gate, or were declined by the planner proposing no edits. |
| Edit checks / probe score by family | Per scenario family and arm: the pass rate of the scenario's edit checks, and the probe score (compare against `none`). |
| Judge focus | For arms where the judge ran: whether an expected signal fired, whether the lesson's home kind matched, whether expected entries were flagged, whether an already-recorded lesson was vetoed, whether the lesson's turn was picked as evidence, and how often a positive scenario got no focus at all. |
| Call-1 threshold sweep | The `typesafe` arm's gate accuracy at thresholds 0.3–0.9, replayed from logged probabilities. Only call 1 can be replayed; other `veto_threshold` or `home_confidence` values need a new run. |

What to look for:

- **Does the planner decline on its own?** In `force`, compare the "declined by planner"
  count on negatives with `typesafe`. This is the open question from merging the review
  into the plan.
- **Does the focus help?** Compare `force+focus` with `force` on edit checks: the right
  kind, the stale entry updated rather than duplicated, and read-only entries overridden
  locally.
- **Does scope work?** Look at the three scope families: a stale global entry fixed by a
  global pass, a contradicted global entry overridden by a local pass, and a
  session-only fact declined by a global pass.
- **Does refinement help later turns?** Compare probe scores against `none`.
- **What threshold should the gate use?** Pick it from the sweep and the declined
  counts, not from 0.7.

To look at one case, find its row in `results.jsonl` (`session_dir` names the case
`<scenario>.<arm>.<repeat>`). Then read:

- `sessions/<case>/messages.jsonl`: the ledger. The last `refinement` or
  `refinement_declined` record with `trigger: "host"` is the pass under test. Its `judge`
  holds the verdict, the gate and focus probabilities, the chosen evidence turns and
  the planner instructions it wrote.
- `sessions/<case>/harness/harness_state.json` and
  `global/<case>/harness_state.json`: the final local and global stores.
- `workspaces/<case>/`: the fixture files the session ran in.

## Scenarios

| id | pass | label | what it tests |
|---|---|---|---|
| `user_correction` | local | refine | a user corrects how lines are counted; a probe checks the rule is applied |
| `already_captured` | local | decline | the same correction when a memory already records it |
| `durable_fact` | local | refine | a stated path convention; a probe checks it is used |
| `one_off` | local | decline | two ordinary questions |
| `repeated_failure` | local | refine | a script fails the same way twice without `--root`; a probe runs it again |
| `transient_failure` | local | decline | one timeout that succeeds on retry |
| `delegation_role` | local | refine | two child agents spawned for the same kind of subtask |
| `stale_entry` | local | refine | a seeded local memory the user contradicts |
| `global_stale_entry` | global | refine | a seeded global memory the user contradicts across projects |
| `global_override` | local | refine | a seeded global memory contradicted for this repo only; expects a local override |
| `session_fact_global` | global | decline | a fact the user limits to this session |

To add one, append a `Scenario` in `scenarios.py` and give it:

- fixture files and steering prompts
- the label and expected signals, home kinds and entries
- edit checks, with a scope
- optionally, a probe

Word the steering the way a user or a tool would, not as a statement of the lesson, and
pair each positive with a nearby negative.

## Cleanup

`scripts/refine_eval/runs/` is gitignored. Each case keeps its session, workspace and
global store under the run directory; delete the directory once the run is reviewed.
