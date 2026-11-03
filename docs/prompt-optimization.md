# Prompt optimization

`scripts/gepa/` optimizes the registry texts behind [`prompt_overrides`](runtime-contract.md#prompts) with
[GEPA](https://github.com/gepa-ai/gepa): a candidate is a `{name: text}` dict, a rollout
is one nano-rlm session, and a reflection model rewrites one text at a time from the
session ledgers and their scores. Nothing here runs inside a session; it needs the
`gepa` dependency group (`uv sync --group gepa`).

Tasks are question sessions about checked-out repositories whose answers are computed
from the AST or the filesystem (`tasks.py`: definition counts, parameter defaults,
callers, importers, decorator users, test locations, line counts) plus `api` questions
that can only be answered by using a documented runtime surface (`rlm.shell.run`,
`rlm.agent.spawn`, `h.create_memory`, the history API) and are checked against the
session ledger as well as the answer. The history question that quotes question 1 is
only asked once the session has compacted, since before that the question is still in
context. A session's score is the mean question score minus a small token penalty.
Rollouts run with a `summarize_at_tokens` far below production so sessions compact and
the compaction texts get evidence.

```bash
uv run python scripts/gepa/workspace.py                     # pinned checkouts, one NAME=PATH per line
uv run python scripts/gepa/tasks.py --repo nano-rlm=<path> --repo click=<path> --per-repo 20
RLM_API_KEY=... RLM_BASE_URL=... uv run --group gepa python scripts/gepa/optimize.py \
    --tasks scripts/gepa/tasks/tasks.jsonl --run-dir scripts/gepa/runs/first \
    --model deepseek/deepseek-v4.1-flash --reflection-model openai/gpt-6-astra \
    --max-metric-calls 300
```

The first run optimizes `task`, `repl_doctrine`, `delegation_doctrine`, `checkpoint`,
`rollup` and `staircase_framing` (`--components` selects others). Every proposed
candidate is checked before any rollout is spent: a text that is empty, longer than its
allowance (twice the seed text or 1,200 characters, whichever is larger), or missing a
token the runtime depends on (`components.py`, e.g.
`hist.expand(` in `history`, `rlm.shell.run` in `runtime_reference`), or mentioning what
exists only in these sessions (scoring, checks, the `ANSWER:` format) scores zero with
feedback naming the problem. The component selector only proposes texts the minibatch
actually exercised. A run is resumable from its `--run-dir` and ends with
`best_prompt_overrides.json`, a `prompt_overrides` object any host can send, and
`report.md` with per-task validation scores and a diff per component. Landing a
winner in `rlm.prompt` is a reviewed change. `scripts/gepa/RUNBOOK.md` is the
step-by-step for a full run. Expect roughly 100-250k tokens per rollout
and a few hundred rollouts per run. Sessions run inside a checkout and may write to it,
so ask about this repository through its pinned workspace copy, never the live tree.
