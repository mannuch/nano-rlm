# Running a prompt optimization

Everything runs from the repository root on `main`. See the "Prompt optimization"
section of the README for what is being optimized and how sessions are scored.

## 1. Sync

```bash
git checkout main && git pull
uv sync --group gepa
```

## 2. Pinned checkouts to ask about

Prints one `NAME=PATH` line per repository (nano-rlm is pinned to a commit, the others
to release tags). Sessions run inside a checkout and may write to it, so even this
repository is asked about through its copy.

```bash
uv run python scripts/gepa/workspace.py
```

## 3. Build the task set

Paste the `NAME=PATH` lines from step 2 as `--repo` arguments. 15 per repository gives
60 tasks; `optimize.py` holds out 33% by task id, so 20 validation / 40 training.

```bash
uv run python scripts/gepa/tasks.py \
  --repo nano-rlm=<path> --repo itsdangerous=<path> --repo click=<path> --repo attrs=<path> \
  --out scripts/gepa/tasks.jsonl --per-repo 15 --seed 0
```

A quick look at what was generated:

```bash
uv run python -c "import json; [print(t['id'], [q['kind'] for q in t['questions']]) for t in map(json.loads, open('scripts/gepa/tasks.jsonl'))]"
```

## 4. Credentials

```bash
export RLM_API_KEY=...
export RLM_BASE_URL=https://openrouter.ai/api/v1
```

## 5. Run

In `tmux` or under `nohup`; a full run takes hours. Raise the open-file limit first:
each concurrent rollout is an IPython kernel with several sockets, and macOS shells
default to 256.

```bash
ulimit -n 4096
nohup uv run --group gepa python scripts/gepa/optimize.py \
  --tasks scripts/gepa/tasks.jsonl --run-dir scripts/gepa/runs/first \
  --model deepseek/deepseek-v4.1-flash --reflection-model anthropic/claude-fable-5.1 \
  --max-metric-calls 300 --minibatch 3 --concurrency 8 \
  > scripts/gepa/runs/first.log 2>&1 < /dev/null &
disown
```

`nohup` ignores SIGHUP only; detaching stdin and `disown` keep a stray Ctrl-C or a
closing terminal from reaching the process.

Defaults this accepts: `--summarize-at 10000`, `--tail 1000`, `--max-depth 1`,
`--timeout 900` per question, and the first-run components (`task`, `repl_doctrine`,
`delegation_doctrine`, `checkpoint`, `rollup`, `staircase_framing`; `--components`
selects others, including the reference texts).

Wall time: plan on 10-15 hours for the ~500 rollouts a 300-call budget produces at
concurrency 8. Each rollout is its own IPython kernel subprocess (plus a child session
when the model delegates), so 8 is comfortable on a laptop; provider rate limits are the
more likely ceiling. If `run_log_stderr.txt` shows 429 retries, drop to 6.

Cost: on the order of $15-35 per 300-call run with the models above (the task model's
input tokens dominate; the reflection model is a few dollars). For a cheaper feel-out,
`--max-metric-calls 60` gives the seed evaluation plus 5-8 iterations for a few dollars.

## 6. Monitor

```bash
tail -f scripts/gepa/runs/first.log                    # the process itself (tracebacks land here)
tail -f scripts/gepa/runs/first/run_log.txt            # iterations, proposals, accept/reject
wc -l scripts/gepa/runs/first/sessions/rollouts.jsonl  # rollouts completed
uv run python -c "
import json; r=[json.loads(l) for l in open('scripts/gepa/runs/first/sessions/rollouts.jsonl')]
print(len(r),'rollouts', sum(x['prompt_tokens'] for x in r),'in', sum(x['completion_tokens'] for x in r),'out', round(sum(x['correctness'] for x in r)/len(r),3),'mean correctness')"
```

## 7. Stop and resume

- Graceful stop: `touch scripts/gepa/runs/first/gepa.stop` (checked between iterations).
- Resume after a stop, crash or Ctrl-C: re-run the exact command from step 5. State is
  loaded from `gepa_state.bin` and the seed is not re-evaluated.

## 8. Results

In `scripts/gepa/runs/first/`:

- `report.md`: seed versus best score per validation task, a diff per component, and
  the reflection model's token totals.
- `best_prompt_overrides.json`: the `prompt_overrides` object (only the components that
  changed), which any ACP host can send in the runtime contract.
- `sessions/<task>/<id>/messages.jsonl`: every rollout's ledger, for reading what a
  candidate actually did.

To try a winner without touching the repository, send the JSON as `prompt_overrides` in
the `ai.prime.rlm/runtime-v1` object. To land it, edit the corresponding constants in
`src/rlm/prompt.py` / `src/rlm/compaction.py` in a PR with the report attached. Read the
diffs; never copy a candidate blindly.

## 9. Cleanup

`scripts/gepa/runs/` and `scripts/gepa/workspace/` are gitignored. The sessions
directory reaches a few GB after 500 rollouts; delete it when the run is reviewed.
