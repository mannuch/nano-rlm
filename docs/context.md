# Context and history

A session has no fixed length. When the conversation outgrows the model's context window, the engine compacts it into a handoff summary and carries on, while the kernel keeps every variable and the full record stays on disk, readable from Python. The model loses nothing it cannot get back.

## Compaction

Compaction is on by default and unlimited: `compaction: false` turns it off and `max_compactions` caps it. The default `max_total_tokens` budget still bounds the session, since each compaction spends new tokens too.

**When.** The engine reads the model's context window from the provider's `/models` response and compacts once 16k tokens remain; small windows keep at least half. `summarize_at_tokens` pins the threshold instead. Without a known window or explicit threshold, compaction happens only when the provider reports an overflow (a 400 or 413 naming a context limit), which also triggers it with a threshold set.

**How.** Compaction is tiered, after headlong's memory pyramid: instead of replacing the window with one summary that every later compaction re-summarizes, the engine keeps the whole session in context at decreasing resolution and keeps the current prompt and the newest messages verbatim. Each compaction summarizes the context since the previous one; the sections below walk through the pieces.

The kernel keeps running through compaction, so variables, imports and in-memory data survive it. The same policy applies to every agent in the tree.

### Windows versus branches

Two words come up throughout this section and they are easy to conflate.

Think of the session as a long strip of paper: the ledger (`messages.jsonl`). Every message ever sent to or received from the model is written on it once, in order, and gets a number that never changes. Nothing is ever erased.

A **window** is a frame laid over that strip: the set of messages the model can currently see. When something replaces the model's context (session start, compaction, rollback, or a harness refinement swapping the system prompt) the engine puts down a new frame and gives it the next number. Frames are how the ledger answers "what exactly did the model see at request 57?"

A **branch** is a stretch of the strip between two compactions: the part of the story that one tier-1 block summarizes. Only compaction ends a branch.

Most of the time one branch is one window:

```text
ledger   0 ──── 1 ──── 2 ──── 3 ──── 4 ──── 5 ──── 6 ──── 7 ──── 8
         sys   task   call   res    call   res   |stair|  call   res
                                                 |  A  |
window   ├──────────── window 0 ─────────────────┤├──── window 1 ────
branch   ├──────────── branch 0 ─────────────────┤├──── branch 1 ────
                                                 ^ compaction
```

They differ in two situations. A harness refinement opens a new window without compacting, because the system prompt changed in place while every other message kept its number; the branch keeps going, so a block records a window *range*:

```text
ledger   0 ─── 1 ─── 2 ─── 3 ─── 9 ─── 4 ─── 5 ─── 6
         sys  task  call  res   sys'  call  res   ...
window   ├── window 0 ─────────┤├──── window 1 ────────
branch   ├──────────────── branch 0 ───────────────────
                               ^ refinement: new system message (9), same branch
```

And with the verbatim tail, a window after compaction *starts* with the last few messages of the branch before it. Those messages keep their numbers and belong to the next block, so the blocks tile the strip without gaps:

```text
ledger   0 ─── 1 ─── 2 ─── 3 ─── 4 ─── 5 ─── 6 ─── 7
         sys  task  call  res  stair  call  res  stair
                     └─ tail ─┘
window 1 seeds: [0, 1, 4, 2, 3]   (system, pinned task, staircase, then the tail by number)
block A  covers 1-1            (the branch before the tail)
block B  covers 2-4            (the tail, the staircase message, and what followed)
```

A rollback also opens a window, restoring the pre-prompt frame, and restores the branch bookkeeping with it. So: windows are how the ledger addresses context; branches are how compaction chunks time; a branch spans one or more windows, and a window can hold the end of one branch and the start of the next.

### Branches and blocks

The context since the last compaction is a *branch*. When a branch is compacted, the engine asks the model, in context, for a plain-text handoff summary of that branch (reasoning is never part of it) and records it as a tier-1 *block*. The checkpoint prompt asks for only the work since the previous block, well under 2,500 characters and most critical first: the current request, the best answer so far, conclusions from the evidence, and ongoing orchestration (children, jobs, subscriptions, pending events, output cursors and history references). The model is told to name the important kernel variables so the resumed branch knows what exists. A block is immutable and is a pointer into the ledger: its header names the message, window, and turn ranges it summarizes, so the raw `messages.jsonl` stays the source of truth and the summary is an index into it.

```text
[tier 1 | branch 3 | window 3 | messages 412-1180 | turns 88-131]
Ran the integration suite against the new parser; 3 failures remain in
tests/test_edges.py (see hist.messages[1104]). Fixed the tokenizer off-by-one in
src/lex.py. Open: the `--strict` flag is still unhandled.
```

The ranges are supplied by the engine, never by the model, so they are exact.

### Rollups

Every `compaction_fanout` (default 5, written `F` below) consecutive blocks of one tier are merged into one block of the next tier by a side call over those summaries alone: tier 2 summarizes `F` branches, tier 3 summarizes `F^2`, tier k summarizes `F^(k-1)`. A rollup never sees the live context, only the child blocks, so it costs about `F` summaries' worth of input regardless of how long the session is.

| tier | one block summarizes  | one block covers   | built from     |
| ---- | --------------------- | ------------------ | -------------- |
| 1    | the branch itself     | 1 branch           | the live context |
| 2    | `F` tier-1 blocks     | `F` branches       | tier 1         |
| 3    | `F` tier-2 blocks     | `F^2` branches     | tier 2         |
| k    | `F` tier-(k-1) blocks | `F^(k-1)` branches | tier k-1       |

A rollup seals as soon as its `F` children exist and is then never rebuilt; only the frontier is ever summarized, so the lifetime cost is about `N / (F - 1)` small calls for `N` compactions. A rolled-up range keeps its children's message range: a tier-3 block over branches 0-24 covers exactly the messages those branches covered.

### The staircase

The fresh window after a compaction is `[system, prompt, staircase, tail]`. The staircase lists blocks oldest first at decreasing resolution: the branch count `N` is decomposed in base `F`, and tier k contributes the aligned blocks just older than the finer tiers, at most `F - 1` of them. The coarsest block always starts at branch 0, so nothing in the session is unrepresented, and the whole staircase is bounded by `(F - 1) * log_F(N)` blocks.

With `F = 2` the layout after each compaction is:

```text
compactions  staircase (oldest first)
     1       [t1: 0]
     2       [t2: 0-1]
     3       [t2: 0-1] [t1: 2]
     4       [t3: 0-3]
     5       [t3: 0-3] [t1: 4]
     6       [t3: 0-3] [t2: 4-5]
     7       [t3: 0-3] [t2: 4-5] [t1: 6]
     8       [t4: 0-7]
     9       [t4: 0-7] [t1: 8]
```

Read left to right as decreasing age and increasing resolution: the far past is a single paragraph, the middle is folded into pairs, and the most recent branch is a full summary. With the default `F = 5` a session's first four compactions show every branch summary verbatim; the fifth folds them into one tier-2 block.

If a rollup is missing (it failed, or has not been attempted yet) the staircase shows its children instead, so coverage is never lost: at 4 compactions with `[t3: 0-3]` unsealed the staircase reads `[t2: 0-1] [t2: 2-3]`, and with one of those missing too it reads `[t1: 0] [t1: 1] [t2: 2-3]`.

### The verbatim tail

The staircase is followed by the newest run of messages that fits `compaction_tail_tokens` (default 12k, capped at a quarter of the compaction threshold; 0 keeps none), so the exact last tool results survive compaction instead of being paraphrased. The tail opens with an assistant or user message so no tool result is separated from its call, and never reaches the branch's first message. It is re-attached by ledger index rather than copied: the new `context_window` record lists those existing indices after the staircase message, exactly as a rollback re-lists restored messages.

```text
ledger index   0       1     2     3       4        5     6       7
record         system  task  call  result  stairc.  call  result  stairc.

window 0       [0, 1, 2, 3]                       <- first branch
compaction 1   block A = messages 1-1, tail = [2, 3]
window 1       [0, 1, 4, 2, 3, 5, 6]              <- system, pinned task, staircase, tail, new work
compaction 2   block B = messages 2-4, tail = [5, 6]
window 2       [0, 1, 7, 5, 6]
```

This toy session has one call per branch, so the tail takes nearly all of it; in practice a branch holds far more than its tail. The summary request is told how many messages will stay verbatim so it covers what precedes them; the branch block ends just before the tail, and the kept tail opens the next branch, so block message ranges stay gapless (block B above starts where block A's tail started).

### The pinned prompt

The current prompt (the latest `user` or parent instruction, not a supervisor notice) is re-attached by ledger index between the system message and the staircase whenever it fits `compaction_prompt_tokens` (default 4k estimated tokens; 0 never pins), so the task is never reconstructed from a summary and the `[system, prompt]` prefix stays stable across compactions. A larger prompt is referenced by ledger index in the staircase message instead (`hist.messages[i]`); a prompt already inside the kept tail is not duplicated. The `compaction` record names the `pinned_prompt_index`.

```text
window after compaction:  [system] [prompt] [staircase] [tail...]
                           ^ stable prefix   ^ new         ^ re-attached by index
```

### Drill-down

The compaction message names the ledger path and the history API, and every block header is a range the model can expand:

```python
from rlm import history

hist = await history()
hist.blocks                                  # [{"tier": 1, "branches": [0, 1], "messages": [1, 1], ...}, ...]
hist.expand(-1)                              # the messages the newest block summarizes
hist.expand(hist.blocks[0])                  # a block record works too
```

`hist.blocks` lists the blocks in ledger order and excludes those of rolled-back prompt attempts; `hist.expand` returns the slice of `hist.messages` a block covers, which for a rolled-up block is everything its children covered. A `compaction` record carries its tier-1 `block`, the `rollups_sealed` in that cycle, and the `tail_message_indices` it kept; each higher-tier block is a `rollup` record.

### Budgets and failure

Rollups are best effort: one compaction spends at most `max_compaction_attempts` rollup calls in total (the usual cycle needs none or one), a rollup that yields no usable reply or does not fit that budget is skipped, and the next compaction retries it, newest ranges first, so a range that keeps failing cannot starve fresh ones.

A provider overflow (a 400 or 413 naming a context limit) triggers the same compaction reactively from the current state. A rejected checkpoint request falls back to the last state that passed a threshold check - by definition a state with a full reserve of room - and an empty or tool-calling reply is resampled; after `max_compaction_attempts` failed attempts the run ends cleanly with what the conversation holds. An overflow with no history beyond the task propagates: the task alone approaches the window and there is nothing to reclaim. Summary and rollup calls are side requests: they spend tokens but are not work turns, and each carries a `compaction_attempt` semantic edge from the request that triggered the compaction, with the single `compaction` edge sourced from the branch summary.

For training, set `max_compaction_attempts: 1`. An unusable final summary (empty, tool-calling, or not normally terminated) ends the prompt with `format_violation`. Its terminal attempt is advertised as a `format_violation` semantic edge so a trainer can retain those tokens and assign zero reward. Earlier discarded attempts remain separate. A rejected provider request or an exhausted context-recovery path remains `compaction_failed`, without a format-violation edge. Only the branch summary can end a prompt this way: an unusable rollup is skipped as above.

## Tool output

A tool result longer than 20 KB (`max_tool_output_bytes`) enters the conversation as its first and last 10 KB around a notice with the whole output's token estimate, line count and omitted bytes. The full result stays in the ledger.

A command or cell always runs to its end, however much it prints. Its result keeps at most the first and last 512 KiB of output (bash divides that between stdout and stderr) and records how much more the whole output measured, so the notice reports the whole output's size either way. The ledger keeps the retained ends with a marker where the middle was dropped. `rlm.shell` jobs keep up to 16 MiB each in their own files ([Shell jobs](python-api.md#shell-jobs)).

## The ledger

Each agent appends its conversation to `messages.jsonl` in its session directory. The ledger serves the agent's own recovery: it is what `history()` reads and what compacted contexts point to. Clients that trace model calls record requests and responses at their own proxy and receive metrics and semantic edges over ACP, independent of this file.

- **Messages.** Message records carry a stable event `id`, a zero-based `message_index` and a `message` with the role, content and tool calls. System messages and installed summaries are recorded; a `compaction` record carries its tier-1 `block` and a `rollup` record each higher-tier block. Checkpoint prompts and raw checkpoint responses are not. A tool result truncated for the context has a separate record linking back through `source_message_index`.
- **Windows.** A `context_window` record opens a window with a zero-based `window` index, a `reason` (`start`, `compaction`, `rollback`, or `harness` for a [refinement](continual-harness.md#refinement)) and the ordered `message_indices` that seed it. Later records with a `window` field append to it, and closed windows never change. A failed or cancelled prompt rolls back by opening a new window with the restored context, so addresses in the failed window stay valid; a compaction window likewise re-lists the indices of its verbatim tail.
- **Prompts and turns.** A `user` record begins a prompt attempt, and `prompt_rollback.prompt_id` marks one that failed or was cancelled; its records remain as history. The `turn` field counts loop iterations and is not an index.
- **Other records.** Lifecycle records such as `kernel_recovery`, child spawns and delivered inbox notifications are in the same file.

Indices are local to each agent. A session opens a fresh ledger and refuses to reopen an existing `messages.jsonl` for writing. A failed write or flush makes the session refuse further prompts; rollback restores the in-memory context even if its own record cannot be written.

## Reading history

```python
from rlm import history

hist = await history()                     # this agent's ledger ($RLM_SESSION_DIR)
requests = hist.user_messages()             # original inputs, rolled-back attempts included
message = hist.messages[3]                  # by message index
earlier = hist.windows[0].messages          # the first context window
other = await history(session_dir="/path/to/session")
message = other.windows[4].messages[2]      # if that session has reached window 4
blocks = hist.blocks                        # compaction blocks with the ranges they cover
originals = hist.expand(blocks[0])          # the messages the first block summarizes
```

A snapshot holds the complete records as of the read; read again to see new activity. `hist.events` holds the lifecycle records, including each child's spawn prompt and rollback markers. For a child, `await child.history()` goes through the supervisor ([Agents](python-api.md#agents)). Compacted contexts point to this API, so the model can recover details without putting the transcript back in context.

## Session directory

Every agent writes `$RLM_HOME/sessions/<id>/`, and children nest under their parent, mirroring the call tree:

```text
~/.rlm/sessions/abc123/
├── meta.json
├── messages.jsonl
├── inbox.jsonl
├── jobs/<job-id>/{output.bin, meta.json}
├── harness/
│   └── harness_state.json
├── sub-d4e5/
│   ├── meta.json
│   ├── messages.jsonl
│   └── sub-f6g7/
└── sub-h8i9/
```

These files are meant for debugging, visualization and extracting training data. `harness/` is the agent's local [continual harness](continual-harness.md) store; the optional global store lives outside any session directory ([The global store](continual-harness.md#the-global-store)).
