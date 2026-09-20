# nano-rlm

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents. For a full-fledged coding agent built on the same RLM principles, see [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent).

By default the model gets a single built-in tool, `ipython`: a persistent IPython kernel for Python, and Bash commands and background jobs via `rlm.shell.run`. File edits, shell work, and orchestration all go through it. The runtime contract's `builtin_tools` list can select a different tool set (`bash`, `edit`, `fetch`, `ipython`) for native tool-calling runs.

For convenience, rlm ships built-in _skills_ that can be enabled per session via the runtime contract's `skills` list (off by default): `edit` (single-occurrence string replacement), `search` (web search via Serper, needs `SERPER_API_KEY`), and `fetch` (retrieve a URL as cleaned text). Enabled skills are pre-imported into the IPython kernel like any other skill (see [Skills](#skills)), so the agent calls `await edit(path=..., old_str=..., new_str=...)`, `await search(query=...)`, or `await fetch(url=...)`. `fetch` also exists as a native builtin tool with the same semantics, for tool-calling runs (opt-in via the contract's `builtin_tools`).

Context compaction is on by default: the engine compacts when 16k tokens remain below an advertised model context window. Termination comes from the default tree-wide budget of 1M new tokens (`max_total_tokens`). The policy can set an explicit `summarize_at_tokens` threshold. The IPython kernel keeps running across compaction, so REPL state survives (see [Compaction](#compaction)).

Inside IPython, the `rlm` package is available in the namespace. When recursion is allowed, `await rlm.agent.spawn(task=...)` returns a handle to a supervisor-owned agent that continues across cells. Skills supplied by the host environment (see [Skills](#skills)) are importable directly by name, e.g. `import websearch`.

## Install

```bash
git clone https://github.com/mannuch/nano-rlm.git
cd nano-rlm
uv sync
source .venv/bin/activate
```

## CLI

rlm runs exclusively as an [Agent Client Protocol](https://agentclientprotocol.com/) agent:

```bash
rlm --acp
```

There is no standalone prompt mode: every session is created by an ACP client that
supplies the full runtime configuration (see below). Skill CLIs provided by the host
environment are on `$PATH` inside the kernel (e.g. `websearch --queries "..."` when
the `websearch` skill is installed).

## Agent Client Protocol

Each ACP session owns one persistent RLM engine. Repeated `session/prompt`
requests retain both the model conversation and the live IPython kernel. The
agent accepts text prompts and stdio or streamable HTTP MCP servers, including
HTTP request headers. It supports cancellation and `session/close`; cancelling
a turn leaves its session available for the next prompt. It intentionally does
not advertise `session/load`: an arbitrary live Python kernel cannot be
reconstructed after the ACP process exits, so clients must keep the process
alive for the lifetime of a session.

RLM's ACP surface is a versioned training contract. `initialize` advertises the exact
`ai.prime.rlm/contract-v1` marker in its response `_meta`; clients must require
it, then provide one complete `ai.prime.rlm/runtime-v1` object in
`session/new._meta`. The runtime object contains the ACP session ID, model,
provider, execution policy, prompt configuration, enabled built-in skills,
optional builtin tool selection,
explicit kernel environment, and optional search credential. Nullable and
disabled values are sent explicitly as `null` or empty collections. Missing,
partial, unknown, or unsupported contracts are rejected; ACP sessions never
fall back to process environment configuration.

Credentials travel over the private ACP stdio channel and are never echoed.
The `session/close` response carries one authoritative, credential-free
snapshot of cumulative usage, metrics, tool-call stats, supervisor counters,
and limits under `ai.prime.rlm/session-v1`.

Every actual model call carries a standard HTTP `Idempotency-Key` header that
stays stable across SDK and outer retries (retry attempts are distinguished by
`x-stainless-retry-count`), so an inference proxy can deduplicate replayed
requests. Both header names are reserved and rejected in provider
configuration.

Model calls also carry a private `X-ACP-Model-Request-ID` correlation header.
RLM publishes sparse, labeled relationships between those request IDs under
`ai.prime.acp/semantic-edges-v1`: `continuation`, `subagent_call`,
`subagent_return`, `compaction`, `refinement_attempt`, and `refinement`. `continuation` preserves same-agent causal
order even when a consumer's physical token-prefix graph splits. ACP consumers
can resolve the request IDs onto their own message nodes while harnesses that do
not understand the extension ignore it.

## Python API (inside a session)

`rlm.agent` exposes `spawn`, `list`, and `get` inside a running session. These async calls contact the session supervisor; spawning waits only for registration, not task completion. `history()` also supports explicit session-directory loading outside a running session.

## Configuration

All runtime configuration enters through the `ai.prime.rlm/runtime-v1` contract object
(model, provider credentials, execution policy, prompt configuration, skills, builtin tools, kernel
environment, search credential, and an optional `harness` object; see [Continual harness](#continual-harness)). Prompt configuration is role-aware: optional
`subagent_append_to_system_prompt` (nodes that can still recurse) and
`leaf_append_to_system_prompt` (depth == max_depth) override `append_to_system_prompt`
for sub-agents, each falling back to the next-more-general tier. Recursive children inherit the parent's configuration
in-memory (`model_copy`); nothing is re-read from the process environment.

`system_prompt_path` supplies task instructions in place of the default task role.
The runtime guide is always appended, including when a custom prompt file is used.
The role-appropriate append instructions are included between the task instructions
and runtime guide. This keeps tool/API documentation and lifecycle rules available
to root agents, persistent children, and leaves. Credentials are not included in
agent identity metadata.

The generated guide distinguishes Python state from supervisor-owned resources,
shows inbox dictionaries versus handle/metadata objects, and explains waiting,
completion, history recovery, jobs, and subscriptions. Delegation instructions are
shown only when IPython and recursion are available. Tool descriptions follow the
same shell execution guidance.

Checkpoint prompts preserve task requirements, evidence, outstanding assignments,
jobs, subscriptions, event actions, output cursors, and history references. Commands
and edits are included only when relevant. Compaction thresholds, summary validation,
and context retention policy are unchanged.

The process environment configures only process infrastructure:

| Variable   | Default  | Description                          |
| ---------- | -------- | ------------------------------------ |
| `RLM_HOME` | `~/.rlm` | Root directory for sessions and data |

## Recursion

The supervisor owns each agent's identity, task, runtime, and lifetime. Python variables hold handles, so losing a variable or ending a cell does not stop its agent.

```python
researcher = await rlm.agent.spawn(
    task="Check authentication behavior", name="researcher"
)
other = await rlm.agent.spawn(task="Check login behavior", name="login")

agents = await rlm.agent.list()          # Direct children, including completed agents
info = await researcher.info()          # Fresh metadata snapshot
h = await researcher.history()          # Fresh conversation snapshot
researcher = await rlm.agent.get("researcher")  # Recover by sibling name or ID
```

Names are unique among siblings and remain reserved for the session, including after completion. Immutable IDs are shown alongside names. Metadata includes the parent ID, initial task, status, persistence flag, creation time, elapsed lifetime in seconds, session directory, and any failure. `list(recursive=True)` includes descendants, but only direct children can be retrieved as handles or controlled through the supervisor.

```python
status = await researcher.wait(timeout=30)
result = await researcher.result()
if result is not None:
    print(result.answer)
```

`result()` returns the latest successful `RLMResult`, including while a persistent agent is running again, or `None` before its first answer. Terminal failure or cancellation raises. Use `info()` or `wait()` for current activity; a retained answer does not mean a follow-up has finished. `wait()` returns current metadata after an outcome or its timeout (default 30 seconds, range 0–300). It waits inside the Python cell and uses the cell's normal execution timeout. Cancelling or timing out a wait does not cancel the agent.

`await researcher.cancel()` terminates the agent and its descendants and waits for cleanup. Ordinary agents release their kernels after answering. An agent spawned with `persistent=True` becomes idle after answering and retains its conversation and kernel; a parent instruction or new inbox event wakes it. Parent termination tears down all descendants, including persistent agents. Closing the ACP session tears down the tree. Cancelling an individual prompt or cell leaves its accepted children registered and recoverable.

Status is `starting` while waiting for capacity, `running` during execution, `waiting` during a native event wait, `idle` for a persistent agent that has answered, or `completed`, `failed`, or `cancelled` after termination. Completed metadata, results, and history remain accessible for the supervisor's lifetime. Shared filesystem access is trusted; handle permissions are orchestration controls, not filesystem isolation.

Depth, concurrency, total-spawn, and shared token/turn limits remain supervisor-enforced. Rejected spawns raise before registration. Persistent idle kernels release inference capacity but still count toward the session's total-spawn limit. A child return enters the ACP semantic trace when the parent retrieves its result, so background completion is not mistaken for result consumption.

### Messages and event waiting

```python
await researcher.send("Also check logout")   # Deliver after its answer or native wait
await researcher.steer("Focus on login first")  # Deliver at the next model/tool boundary

# Inside the child:
await rlm.agent.send_to_parent("Found a missing permission check")

# Inside its parent:
for event in await rlm.inbox.list():
    report = await rlm.inbox.read(event["id"])
    print(report["type"], report["content"])
```

Parent instructions are pushed into the child's conversation. Steering does not interrupt a running model request or tool. Queued messages wait until the child answers or calls the native `wait` tool. Both operations wake an idle persistent child; sending to a terminated child raises.

Reports and `agent.completed` events enter the parent's inbox at every depth. The model sees the unread count whenever it changes or new events arrive, and chooses when to retrieve payloads. `list()` returns metadata without marking events read; `read(id)` returns the payload and marks it read. `list(unread_only=False)` includes previously read events. Completion payloads identify the agent and status; retrieve the answer with its handle's `result()`.

ACP dependency edges use `agent_message` for delivered instructions and explicitly read inbox events. Spawning uses `subagent_call`; retrieving an answer uses `subagent_return`. Delivery policy and event type remain in the event records. An unread-count notification alone creates no `agent_message` edge.

The native `wait` tool suspends inference without occupying an IPython cell, until a new inbox arrival, parent instruction, or timeout (default 300 seconds, range 0–300). It also yields to queued instructions. Already-announced unread events do not repeatedly wake it. A final root answer returns control to the ACP caller; use `wait` to keep the current prompt available for events.

The supervisor owns inboxes and instruction queues. Each agent's `inbox.jsonl` records arrivals, instruction delivery, and explicit reads; delivered messages and notifications also enter `messages.jsonl`. Message payloads are limited to 65,536 JSON-encoded UTF-8 bytes. Reports are rejected once an inbox contains 10,000 events, and pending instruction queues have the same limit; lifecycle events remain recordable. This state lasts while the supervisor lives. Restarting a crashed IPython kernel and restoring a crashed supervisor are separate features.

## Compaction

There is no model-driven compaction tool. Compaction is on by default and unlimited (the policy's `compaction` field turns it off, `max_compactions` caps it); the default 1M `max_total_tokens` tree budget keeps sessions bounded, since each compaction cycle itself spends new tokens. The engine reads the model context window from the provider's `/models` response and compacts when 16k tokens remain below it; small windows keep at least half. Set `summarize_at_tokens` to pin the threshold explicitly. Without a known window or explicit threshold, proactive compaction stays off, but a provider overflow still triggers it reactively. A tool result larger than 20KB is truncated to its head and tail before it enters the conversation, with a warning naming the original size.

Compaction is tiered, after headlong's memory pyramid: instead of replacing the window with one summary that every later compaction re-summarizes, the engine keeps the whole session in context at decreasing resolution and keeps the current prompt and the newest messages verbatim.

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

The context since the last compaction is a *branch*. When a branch is compacted, the engine asks the model, in context, for a plain-text handoff summary of that branch (reasoning is never part of it) and records it as a tier-1 *block*. A block is immutable and is a pointer into the ledger: its header names the message, window, and turn ranges it summarizes, so the raw `messages.jsonl` stays the source of truth and the summary is an index into it.

```text
[tier 1 | branch 3 | window 3 | messages 412-1180 | turns 88-131]
Ran the integration suite against the new parser; 3 failures remain in
tests/test_edges.py (see h.messages[1104]). Fixed the tokenizer off-by-one in
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

The current prompt (the latest `user` or parent instruction, not a supervisor notice) is re-attached by ledger index between the system message and the staircase whenever it fits `compaction_prompt_tokens` (default 4k estimated tokens; 0 never pins), so the task is never reconstructed from a summary and the `[system, prompt]` prefix stays stable across compactions. A larger prompt is referenced by ledger index in the staircase message instead (`h.messages[i]`); a prompt already inside the kept tail is not duplicated. The `compaction` record names the `pinned_prompt_index`.

```text
window after compaction:  [system] [prompt] [staircase] [tail...]
                           ^ stable prefix   ^ new         ^ re-attached by index
```

### Drill-down

The compaction message names the ledger path and the history API, and every block header is a range the model can expand:

```python
from rlm import history

h = await history()
h.blocks                                     # [{"tier": 1, "branches": [0, 1], "messages": [1, 1], ...}, ...]
h.expand(-1)                                 # the messages the newest block summarizes
h.expand(h.blocks[0])                        # a block record works too
```

`h.blocks` lists the blocks in ledger order and excludes those of rolled-back prompt attempts; `h.expand` returns the slice of `h.messages` a block covers, which for a rolled-up block is everything its children covered. A `compaction` record carries its tier-1 `block`, the `rollups_sealed` in that cycle, and the `tail_message_indices` it kept; each higher-tier block is a `rollup` record.

### Budgets and failure

Rollups are best effort: one compaction spends at most `max_compaction_attempts` rollup calls in total (the usual cycle needs none or one), a rollup that yields no usable reply or does not fit that budget is skipped, and the next compaction retries it, newest ranges first, so a range that keeps failing cannot starve fresh ones.

A provider overflow (a 400 or 413 naming a context limit) triggers the same compaction reactively from the current state. A rejected checkpoint request falls back to the last state that passed a threshold check - by definition a state with a full reserve of room - and an empty or tool-calling reply is resampled; after `max_compaction_attempts` failed attempts the run ends cleanly with what the conversation holds. An overflow with no history beyond the task propagates: the task alone approaches the window and there is nothing to reclaim. Summary and rollup calls are side requests: they spend tokens but are not work turns, and each carries a `compaction_attempt` semantic edge from the request that triggered the compaction, with the single `compaction` edge sourced from the branch summary.

The IPython kernel keeps running across the compaction, so all variables, imports, and in-memory data are preserved. The model is told to mention important variable names in its summary so the resumed branch knows what is available. The same policy applies to the main agent and all recursive agents.

### The ledger

A session creates a fresh ledger and refuses to reopen an existing `messages.jsonl` for writing. Existing histories remain readable. The session owns active messages and their indices; the engine uses context snapshots. A ledger write or flush failure makes the writer unusable and prevents further prompts. Rollback restores in-memory context and counters even if its ledger write fails. Supervisor crash recovery is not supported.

The working conversation remains in the session's append-only `messages.jsonl`, which the model can search and process from Python. Message records have stable event `id` fields, zero-based `message_index` values, and a `message` object containing the role, content, and tool-call fields. Full tool outputs and shortened context versions have separate indices; the shortened tool record links to its `source_message_index`. System messages and installed summaries are also logged; a `compaction` record carries its tier-1 `block` and a `rollup` record each higher-tier block. Checkpoint prompts and raw checkpoint responses are omitted: this ledger serves agent context recovery. Verifiers, the ACP consumer, records model requests/responses through its interception endpoint and receives semantic links and metrics over ACP, independently of this file. A `user` record begins a prompt attempt; `prompt_rollback.prompt_id` identifies a failed or cancelled attempt, whose records remain available as history.

Each `context_window` record declares a zero-based `window` index, a `reason` (`start`, `compaction`, or `rollback`), and the ordered `message_indices` that seed that window. Subsequent message records with a `window` field append to it. Closed windows never change. Rollback opens a new window containing the restored context, preserving addresses in the failed window; a compaction window likewise re-lists the indices of its verbatim tail. The `turn` field is an execution-loop counter, not a message or window index.

Each agent writes its own `messages.jsonl` in its session directory. Message and context-window indices are local to that agent. Agent discovery belongs to the supervisor; the history reader accepts an explicit session directory.

The kernel can inspect its own history or a child's, including while the child is running:

```python
from rlm import history

h = await history()                  # Defaults to $RLM_SESSION_DIR
requests = h.user_messages()          # Original inputs, including rolled-back attempts
message = h.messages[3]               # Session-wide message index
earlier = h.windows[0].messages       # Initial working context
child_history = history(session_dir="/path/to/child-session")
message = child_history.windows[4].messages[2]  # If that child has reached window 4
blocks = h.blocks                     # Compaction blocks with the ranges they cover
originals = h.expand(blocks[0])       # The messages the first block summarizes
```

Snapshots contain complete records as of the read; call `history(...)` again to observe new activity. `h.events` exposes lifecycle records, including each child's spawn prompt and rollback markers. The compacted context points to this API so the model can retrieve omitted details without putting the whole transcript back in context.

## Session Directory

Every invocation writes to `$RLM_HOME/sessions/<id>/`. Nested session directories mirror the call tree.

```text
.rlm/sessions/abc123/
├── meta.json
├── messages.jsonl
├── harness/
│   └── harness_state.json
├── sub-d4e5/
│   ├── meta.json
│   ├── messages.jsonl
│   └── sub-f6g7/
└── sub-h8i9/
```

These artifacts are consumable for debugging, visualization, or training-data extraction.

The only cross-session file is the optional global harness store, which lives wherever
`harness.global_dir` points, outside any session directory:

```text
<global_dir>/
├── harness_state.json     # global prompt/memory/skill/subagent entries and, when
│                          # record_episodes is on, one episode entry per root session
└── refinements.jsonl
```

An episode entry's `path` field (`episodes/<YYYY-MM>/…`) is a facet inside that JSON file,
not a directory; the entry points back at its session through `metadata.session_dir`, and
the session's `messages.jsonl` records the `global_dir` it published to. Nothing under
`sessions/<id>/` changes.

## Continual harness

The continual harness is durable state that supplements the immutable system prompt: `prompt`
notes (narrow behavioural policies), `memory` entries (facts, decisions, failures), `skill`
entries (descriptions of how to call an importable module, with a `reference` and an
`arguments` contract), `subagent` specs (reusable delegation roles) and, in a global store,
`episode` records of past sessions written by the engine (see [Episodes](#episodes)). Each agent's local
store lives at `<session>/harness/harness_state.json`. A child reads its ancestors' local
stores read-only alongside its own. The contract's `harness` object controls the feature:

```json
"harness": {
  "enabled": true,
  "global_dir": null,
  "max_prompt_entries_per_kind": 6,
  "max_prompt_content_chars": 180,
  "max_prompt_refinements": 5,
  "auto_refine": false,
  "refine_turn_interval": 12,
  "refine_cooldown_seconds": 300,
  "max_refinements": null,
  "max_refinement_attempts": 3,
  "skills_dir": null,
  "record_episodes": false
}
```

`global_dir` names a store shared across sessions; `null` (the default) keeps every session
hermetic. The system prompt carries a compact `## Continual harness` block: per-kind counts,
the most relevant entries within the caps (ranked against the task text when a kind
overflows) and recent refinement events. Entries are displayed as `[local:id]`,
`[ancestor:id]` or `[global:id]`.

Inside the kernel, `rlm.harness` is the synchronous Python API:

```python
h = rlm.harness.harness()                      # the view this agent sees
print(h.overview())
h.search("pytest venv")                        # ranked by term overlap
h.get("memory", "ancestor:parent_lesson")      # ids exactly as displayed
h.create_memory("Project venv", "run tests with ./.venv/bin/python -m pytest")
h.create_skill("Release lookup", "query websearch with '<pkg> release'",
               reference={"type": "python", "import": "websearch", "callable": "run",
                          "call_pattern": "await websearch(queries=[...])"},
               arguments={"queries": {"type": "array", "required": True}})
h.update_memory("project_venv", "Project venv", "...", global_=True)   # needs global_dir
```

`update_*`/`delete_*` take the bare id or a `local:`/`global:` prefix; ancestor entries raise
`PermissionError`. A `skill` entry describes an already-importable module; it is not a
package (see [Skills](#skills) for the on-disk skill contract). Stores are rewritten
atomically under a file lock and reloaded when another writer changed them, so the engine
and the kernel share one file safely. `harness(session_dir=...)` loads a session's local
store outside a running session.

### Entry schema

Every entry is one `HarnessEntry` record (a Pydantic model in `rlm.harness`) with the same
base fields for all kinds, and kind-specific structure inside three payload dicts that is
validated whenever an entry is created, updated, restored or loaded from disk:

| field | type | notes |
| --- | --- | --- |
| `id`, `kind`, `title`, `content` | `str` | `id` is a slug of the title unless given; `kind` is one of `prompt`, `memory`, `skill`, `subagent`, `episode` |
| `path` | `str` | grouping facet, default `general`; not a filesystem path |
| `scope` | `local` \| `global` | the store the entry lives in |
| `reference`, `arguments`, `metadata` | `dict` | payloads, see below |
| `source` | `str` | `agent`, `refinement` or `engine` |
| `created_at`, `updated_at`, `version` | | maintained by the store |

| kind | payload | model |
| --- | --- | --- |
| `skill` | `reference` | `SkillReference`: `type: "python"`, `import` (module), and `callable` and/or `call_pattern` |
| `skill` | `arguments` | `{name: SkillArgument}` with `type`, `required` (default false), `default`, `description` |
| `episode` | `metadata` | `EpisodeMetadata`: `session_dir`, `session_id`, `stop_reason`, `prompts`, `turns`, `prompt_tokens`, `completion_tokens`, `cwd`, `blocks` (`BlockRecord`s), `started_at`, `ended_at` |
| others | `metadata` | free-form |

Payload models keep unknown keys, so a store written by a newer build still loads, and
fill in optional ones, so `entry.reference["call_pattern"]` is always present on a skill.
A violation raises `ValueError` naming the field (`arguments.queries: Input should be a
valid dictionary ...`) from the kernel API, is reported per edit by a refinement pass, and
fails `load()` with the store path and entry id when a state file is inconsistent.

### Skill entries versus skill packages

Three things share the word "skill", and they are different kinds of object:

|                 | Installed skill                            | Authored skill package                                        | Harness `skill` entry                                                                                                   |
| --------------- | ------------------------------------------ | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| What it is      | A Python package installed into rlm's venv | A Python package under `skills_dir`, imported from `sys.path` | A JSON record in `harness_state.json`                                                                                   |
| Contains        | Code: `async def run(...)`                 | Code: `async def run(...)`                                    | Text: title, when/how to call something, a `reference` (`import`, `callable`, `call_pattern`) and an `arguments` schema |
| Created by      | A human, via `install.sh` at image build   | The agent, mid-session, by writing files                      | The agent (`h.create_skill(...)`) or the refinement pass                                                                |
| Lives           | The venv; every session                    | `<skills_dir>/<name>/`; every session sharing that dir        | One store: session-local, an ancestor's, or global                                                                      |
| The kernel sees | `await websearch(...)`                     | `await greeter(...)`                                          | Nothing executable; it is rendered into the system prompt                                                               |
| Refinement can  | Never touch it                             | Never touch it                                                | Create, update, delete, roll back                                                                                       |

The relationship is pointer to target. An entry's `reference.import` names a module that
must already be importable (an installed skill, an authored package, an MCP proxy module,
or `rlm` itself); apply-time validation rejects an entry that points anywhere else. The
entry adds what code cannot carry: when this call is the right move, which arguments
matter for a kind of task, gotchas learned from use. One package can be described by
several entries, and a package with no entry is still callable, just not surfaced as a
routing hint.

The split gives the two layers different lifecycles. Packages are code, so they go
through the main loop: the agent writes them, `load_skills()` imports them, it tests
them, and mistakes are observed and iterated on. Refinement, a one-shot JSON proposal
with `before`/`after` snapshots, is kept away from them because a file write cannot be
undone by replaying a snapshot and a side model call should not mint code that
auto-executes in future sessions. Entries are state, so they go through refinement:
small, validated, snapshotted, reversible; a wrong one costs a bad hint in the prompt,
not a broken kernel. The typical arc is: notice a repeated procedure, write a package,
`load_skills` and test it, record a `skill` entry pointing at it; future sessions see the
entry in their system prompt with the package already pre-imported.

### Authored skill packages

`skills_dir` gives agent-written packages a place that survives the session. An authored
package is an installed skill minus installation — the [skill contract](#skill-contract)
applies with these adjustments:

- Required: `<skills_dir>/<name>/SKILL.md` and `<skills_dir>/<name>/src/<name>/__init__.py`
  defining `async def run(...)`. The kernel checks both; `run`'s docstring falls back to
  `SKILL.md` for `help(<name>)`.
- Optional: `pyproject.toml`; when present its `[project] name` must be `rlm-skill-<name>`
  so the package can be promoted to an installed skill unchanged. Nothing is installed, so
  dependencies are limited to what is already in the kernel venv and there is no console
  script: authored packages are IPython-only.

At kernel start every authored package's `src/` goes on `sys.path` and the module is
pre-imported by name with the same `await <name>(...)` wrapper as an installed skill. A
package that breaks the contract or fails to import binds a placeholder whose call raises
the reason, so a bad package never breaks the kernel. Names must not collide with installed
or MCP-generated skills. A package written or edited during the session is brought in
without a restart by `rlm.harness.load_skills("<name>")` (no names = all), which
reloads and rebinds it exactly as kernel start does and returns `{name: reason}` with
`None` for a usable skill. The system prompt states the contract and this workflow;
`RLM_HARNESS_SKILLS_DIR` names the directory. `null` (the default) keeps authored
packages session-local.

### Refinement

Refinement is a side model call, like compaction: the live conversation is extended with
one user message asking for a JSON proposal of create/update/delete edits, the proposal is
validated and applied under the store's lock, the system prompt is rebuilt in place (a new
`context_window` with `reason="harness"` that keeps every other message index), and a
`<runtime_event kind="refinement">` notice tells the model what changed. Rejected edits
(unknown module in a `skill` reference, edits to the base system prompt, entries that
changed while planning, …) are recorded with an error and the rest still apply. Each
store's `refinements.jsonl` holds one full result per pass with per-edit `before`/`after`
snapshots; a rollback replays those snapshots in reverse and needs no model call.

Three triggers, all of which run between model calls and never inside a cell:

- **Kernel**: `await rlm.refine.run(instructions=None, global_=False, rollback_id=None)`
  returns `{"scheduled": True}` (or a reason) and the pass runs at the next boundary;
  `await rlm.refine.status()` reports `pending`/`in_flight`.
- **Host**: `session/prompt` may carry `ai.prime.rlm/refine-v1` in `_meta`:
  `{"instructions": "...", "global": false, "rollback_id": null}`. The pass runs before
  the turn; with an empty prompt it is the whole turn and the notice is the answer
  (`stop_reason` `refined`). The key is refused when the harness is disabled.
- **Auto** (`auto_refine`, off by default, root agent only): every `refine_turn_interval`
  work turns and after each compaction, subject to `refine_cooldown_seconds`, a cheap
  review call decides whether the trajectory holds evidence worth persisting; only an
  approving review triggers a plan. After a compaction the review also receives the blocks
  that compaction produced (the branch summary and any rollups it sealed) as a
  `<compaction_blocks>` section, with the hint that a failure, tactic or fact recurring
  across blocks is evidence for a durable entry while one branch's progress is not; an
  approved review hands the same blocks to the plan as `<evidence>`. Both are side calls
  with `tool_choice="none"`, so the evidence is the block text itself, never a pointer to
  follow. The `refinement_review` ledger record lists the block keys.

`max_refinements` caps passes per engine; `max_refinement_attempts` bounds how often an
unusable reply (truncated JSON, prose, a tool call) is resampled before the pass is
reported as failed in the conversation and the run continues. Plan and review calls carry
`refinement_attempt` semantic edges from the last work request; the applied plan request
becomes the source of a `refinement` edge into the next work request, which also keeps its
ordinary `continuation` edge. Refinement counts and edit totals appear in the session
metrics; the `session-v1` snapshot carries per-scope entry counts under `harness`.

### Episodes

Trajectories are per agent directory and there is no global trajectory: `$RLM_HOME` is only
the default parent of `sessions/<id>/`, and no session discovers or reads another unless
it is handed a path. Episodes are the opt-in episodic half of cross-session memory, next to
the semantic entries above. With `record_episodes: true` **and** a `global_dir`, the root
engine writes one `episode` entry into the global store when the session closes, without
any model call:

- `title`: the first line of the first prompt; the entry's `path` field (a facet, not a
  directory): `episodes/<YYYY-MM>/<DD>T<HHMMSS.mmm>-<session id>` from the session's start
  time, so episodes list chronologically within the kind and a month or day is a
  searchable term.
- `content`: `Prompts (n):` with the first line of every prompt (the first ten listed,
  the rest counted), then the compaction staircase as it stood at close (the coarsest
  blocks, or `(no compaction)` for a session that fit one window), then
  `Outcome (<stop_reason>): <answer>` for the last prompt.
- `metadata`: `session_dir`, `session_id`, `stop_reason`, every prompt line under
  `prompts`, `turns`, token totals, `cwd`, the block records, and start/end times.

The entry is keyed by session id, so closing twice rewrites the same record. The system
prompt lists episodes like any other kind, ranked against the task text when there are more
than `max_prompt_entries_per_kind`; from the kernel, `h.search(query, kind="episode")` finds
one and `history(session_dir=h.get("episode", id).metadata["session_dir"])` opens its ledger
for `.blocks` / `.expand(i)`. Each session also logs an `episode` record naming what it
published.

Episodes are engine-owned: `create`/`update`/`delete` of that kind raise `PermissionError`
from the kernel API and refinement edits of that kind are rejected, so nothing inside a
session can rewrite or remove another session's record. The engine never deletes them
either — a per-session setting must not prune a shared store — so trimming a long-lived
store is an operator action. Both gates default to off because a global store shared
across RL rollouts of one task would let a rollout read another's outcome. A store that
holds episodes cannot be read by builds that predate the kind.

## Skills

`rlm` ships a small set of built-in skills enabled per session via the runtime contract's `skills` list (`edit`, `search`, `fetch`; see [MCP tools as skills](#mcp-tools-as-skills) for the related MCP path). `edit` runs in the kernel. Credentialed `search` runs in the supervisor through the capability broker, so `SERPER_API_KEY` is unavailable to IPython and its subprocesses; it returns title/URL/snippet for a single query (`await search(query="...")`). Additional skills are supplied by the host environment: before `install.sh` runs, the environment places skill packages under `/task/rlm-skills/<name>/`, and `install.sh` installs them alongside `rlm` so they're both importable and on `$PATH`.

From IPython, import a skill and call its async `run(...)` entrypoint:

```python
import websearch

help(websearch)  # signature + docstring
results = await websearch(queries=["latest jupyter_client release"])
```

Uploaded `rlm-skill-*` packages also expose their declared console command. Session-generated built-in and MCP proxy skills are IPython-only:

```bash
websearch --queries "latest jupyter_client release"
```

### Skill contract

A skill is a normal Python package laid out like this:

```text
<name>/
├── SKILL.md
├── pyproject.toml
└── src/
    └── <name>/
        ├── __init__.py
        └── <name>.py
```

Required public surface:

- async `run(...)`: the single entrypoint. Type-annotated parameters and a Google-style docstring (`Args:`, `Returns:`) drive both the Python API and the CLI.

The `pyproject.toml` points its console script at the shared CLI entry:

```toml
[project.scripts]
<name> = "rlm.skill:cli"
```

`rlm.skill:cli` reads `sys.argv[0]`, imports the matching module, and uses [tyro](https://brentyi.github.io/tyro/) to build the argparse CLI from `run`'s signature. The return value of `run` is printed; a raise surfaces as a normal Python traceback.

Naming expectations (all match):

- skill directory name: `<name>`
- distribution name in `pyproject.toml`: `rlm-skill-<name>`
- import name: `<name>`
- console script name: `<name>`

Keyword arguments on `run(...)` and CLI flags line up automatically — `queries: list[str]` becomes `--queries` on the CLI.

Dependencies go in the skill's own `pyproject.toml`; declare `rlm` there so the shared CLI entry resolves. Version conflicts between skills installed side-by-side are the user's responsibility.

### Local development

For running `rlm` against a specific skill set outside of a sandbox-orchestrated environment, create a `/task/rlm-skills/` directory (or bind-mount one) and place skill packages there before running `install.sh`. The rlm repo ships no skills by default; look at the `rlm-swe` or `rlm-deepdive` environments for working skill packages to copy.

### MCP tools as skills

A host harness can wire task-specific [MCP](https://modelcontextprotocol.io) tool servers to `rlm` through the ACP session (a standard `mcpServers` config shape). Streamable HTTP and stdio transports are supported:

```json
{
  "mcpServers": {
    "remote": { "url": "http://127.0.0.1:8000/mcp" },
    "local": {
      "command": "/path/to/tool-server",
      "args": ["--stdio"],
      "env": { "API_KEY": "..." }
    }
  }
}
```

Programmatically, pass `mcp_servers={"tools": "http://127.0.0.1:8000/mcp"}` to `RLMEngine`. An HTTP server may instead be `{"url": "...", "headers": {"Authorization": "..."}}` when it needs request headers; stdio servers use the same `command` / `args` / `env` shape shown above.

At startup `rlm` connects to each server, lists its tools, and generates one skill per tool (named `<server>_<tool>`, e.g. `tools_add_event`). These join the installed skills — pre-imported into the IPython namespace as async functions the agent calls programmatically, with a signature built from the tool's input schema:

```python
help(tools_add_event)  # signature (typed from the schema) + the tool's description
await tools_add_event(day="monday", title="standup")
```

Each call connects using the configured transport, invokes the tool, and returns its text content (a tool-reported error is raised as `RuntimeError`). Discovery and invocation run in the session supervisor. The generated modules contain only the public tool schema and an opaque capability; server URLs, headers, commands, and environment variables are not copied into the kernel environment or session artifacts. The kernel imports these proxy modules from the session directory and reaches the supervisor over the session-local broker. Unlike installed skills, MCP skills are IPython-only — they're not exposed as shell commands.

## Kernel

The IPython kernel always runs in rlm's own Python (`sys.executable`). `install.sh` puts `rlm` and all discovered skills into the same `uv tool install` environment, so `from rlm import run`, `import edit`, etc. work natively from inside an IPython cell.

The kernel starts from a small platform environment (`PATH`, home/user/shell, locale, temporary-directory, certificate, and virtual-environment variables) plus the contract's explicit `kernel_env` mapping. It receives private Jupyter/IPython config directories and does not inherit the rest of the supervisor process environment. This de-ambients credentials; it is not hostile-code containment because the kernel still shares the sandbox user, filesystem, process namespace, and network with the supervisor.

To exercise packages from the target project's `.venv` (e.g. running its test suite), shell out from an IPython cell: `!./.venv/bin/python3 -m pytest`. The kernel itself stays isolated from whatever project venv the agent is working on — no cross-cell state involving sandbox packages.

## Developing

After `uv sync`, enable the pre-commit hooks:

```bash
uv run pre-commit install
```

Lint and format manually:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Testing

Install dev dependencies and run the suite:

```bash
uv sync --group dev
uv run pytest tests/
```

Agent execution status and cleanup are separate: `info.cleanup_error` reports failed resource cleanup without hiding a completed result. Calling `cancel()` again retries unfinished cleanup.

`send()` and `steer()` return acceptance IDs, not processing acknowledgements. They reject instructions when the tree budget is already exhausted. If an accepted instruction cannot be delivered because the budget runs out or the child terminates, the supervisor records `instruction_failed` in the child's inbox journal and sends the parent an `agent.delivery_failed` event containing `agent_id`, `message_id`, and `reason`. Failed instructions are removed from the pending queue.

### Bash commands and background jobs

For quick commands, wait for the result in the current cell:

```python
result = await rlm.shell.run("git status --short")
print(result.text, result.exit_code)
```

`run(command, cwd=..., env=..., yield_after=10, timeout=None)` returns a `ShellJob`: a snapshot with `running`,
`ok`, combined stdout/stderr `text` (up to 16 KiB: the first and last 8 KiB around an
omitted-range marker when longer), `exit_code`, `id`, `truncated`, `error` and `timed_out`, plus the
methods `result()`, `read()`, `info()` and `cancel()`. `yield_after` is how long `run()`
waits before yielding a handle instead of a result: 10 s by default, 0 returns at once, capped
at 300. A command still going then comes back with `running=True` and the output so far while
the job continues; nothing is killed by yielding. `timeout`, if given, kills the process group after that many seconds and
sets `timed_out`. `await job.result(yield_after=...)` waits (up to 300 s per call) and returns
the finished snapshot, again with `running=True` if the job is still not done; it is
repeatable and never consumes output. The command is a Bash string or an argv list. Nonzero
exit codes are returned; startup/capture errors populate `error`. If output is truncated,
`await job.read(cursor=..., max_bytes=...)` reads the retained output (16 MiB per job).
Cancelling the waiting cell leaves the job running and discoverable with `shell.list()`;
`shell.get(job_id)` recovers it. Only a job handed back with `running=True` publishes a quiet
`shell.completed` inbox event (job ID, status, exit code, completeness flags and the last
4 KiB of output); quiet means it wakes the native `wait` tool but is not counted in the unread
notice, and `result()` marks it read. `await rlm.shell.setenv(NAME='value')` sets variables
for every later `run()` of the agent (`getenv()` reads the overlay; per-call `env=` wins).
The kernel and Bash jobs inherit the launching process's environment (in a sandbox, the
image's ENV) minus a blocklist: credential-looking names, values with URL-embedded
credentials, variables that would redirect the kernel's own interpreter or venv (PYTHONHOME,
UV_PYTHON, ...), and agent/daemon sockets.

For long commands or work that should run alongside other tasks, pass `yield_after=0` and
collect with `result()`:

```python
job = await rlm.shell.run("uv run pytest tests/", cwd="/workspace/project", yield_after=0)
# ... other work ...
res = await job.result()
print(res.exit_code, res.text)

# Several jobs at once, and recovery of a job in a later cell:
a, b = await asyncio.gather(job_a.result(), job_b.result())
job = await rlm.shell.get(job_id)
chunk = await job.read(cursor=0, max_bytes=16384)   # continue from chunk.next_cursor
```

Use the native `wait` tool when there is no other work and you hold no running job; a held
job is collected with `result()`. A nonzero Bash exit code is a completed process;
startup/capture errors have status `failed`.

`await job.cancel()` terminates its process group. Jobs survive cell completion
and lost handle variables; owner termination cancels them. Commands run in
`/bin/bash --noprofile --norc -c`, with closed stdin and combined stdout/stderr.
Relative working directories resolve against the agent's working directory.
The task environment and Git-history policy also apply to these commands.

The supervisor permits 32 active jobs and 1,024 total jobs per session tree.
Each job retains up to 16 MiB in `<session>/jobs/<id>/output.bin`; excess output
is drained and discarded. Reads default to 16 KiB and allow at most 64 KiB.
Cursors count bytes, and text decodes as UTF-8 with replacement. Terminal metadata
is saved alongside output in `meta.json`. Output files remain after job cleanup.

Normal completion waits for stdout/stderr EOF. If descendants keep the pipe open
more than two seconds after Bash exits, capture stops and reports
`output_complete=False` and `output_truncated=True`. Remaining processes in the
job's process group are killed when the job ends. Keep Bash alive until its work
finishes; detached processes that escape the process group are not managed.

There is no PTY, stdin API, or redirection of IPython's `!` / `%%bash` magics.

### IPython crash recovery

If the kernel exits during a cell, the harness starts a fresh kernel under the
same agent identity and resumes the conversation with a recovery notice. A dead
kernel discovered before a cell starts is also restarted; that cell is returned
unexecuted so the model can first reconstruct its Python state.

The conversation, history, child agents, shell jobs, and inbox (including read
state) remain owned by the live supervisor. Recover handles with
`rlm.agent.list/get` and `rlm.shell.list/get`. Python variables, imports, and
in-kernel tasks are lost. The interrupted cell is never replayed automatically:
its file writes or accepted spawn/send/job requests may already have happened.
Inspect those resources before retrying an operation.

Recovery notices are separate `kernel_recovery` history records and active
conversation messages, so truncating tool output cannot hide them. A kernel
restart required after an unresponsive timeout/interrupt produces the same notice.
A successful interrupt that preserves the kernel does not claim variables were lost.

There are at most three recovery attempts without a subsequently completed cell.
A failed startup reports failure; a later IPython call may retry within that
budget. Exhausting the budget leaves IPython unavailable, while the agent can
still respond or use native waiting. Supervisor crash recovery is not supported.
Kernel death is detected during execution or at the next IPython call; this does
not introduce a background kernel-health subscription.

### Subscriptions

Watch future activity without keeping a Python cell running:

```python
activity = await rlm.watch.agent(researcher)
output = await rlm.watch.job(job)
files = await rlm.watch.path("/workspace/results", recursive=False)

subscriptions = await rlm.watch.list()
watch = await rlm.watch.get(subscriptions[0].id)
await watch.cancel()
```

Agent watches observe direct children after complete assistant/tool steps,
including final answers. Job watches observe newly captured stdout/stderr from
jobs owned by the caller. Path watches observe existing files or directories;
relative paths resolve against the agent's working directory. Hidden files are
included, and directory recursion is opt-in. Child/job completion already
notifies the owner automatically and needs no subscription.

Events enter the same pull-based inbox and carry `subscription_id`. Their
`content` contains a `target` (agent ID, job ID, or absolute path) plus:

| Event          | References                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------- |
| `watch.agent`  | `start` and exclusive `end` message indices: `(await researcher.history()).messages[start:end]` |
| `watch.job`    | `start` and exclusive `end` byte cursors for `job.read(cursor=start)`                           |
| `watch.path`   | Changed `paths` and a `truncated` flag                                                          |
| `watch.failed` | An `error` explaining why the subscription stopped                                              |

Subscriptions start with future activity, not historical replay. Agent and job
ranges can be coalesced across several steps/chunks; changes are batched over
200 ms. Agent ranges exclude records written before registration. Filesystem
registration waits until the backend is watching before returning the handle.
Observed removal of the watched path ends the subscription with a failure event;
register a new watch after recreating it.

Handles and subscriptions survive kernel restart. Cancellation stops future
events and drops any unpublished batch; existing inbox events remain readable.
Owner termination cancels its subscriptions. Registration and state transitions
are journaled in the owner's `inbox.jsonl`; supervisor restart recovery remains
out of scope.

The tree permits 64 active subscriptions and 1,024 total registrations. Changed
path lists are capped at 32 KiB of JSON with explicit truncation. A subscription
stops with a failure event when the inbox event limit is reached. There are no
model-authored callbacks, event selectors, or output predicates.

Activity subscriptions become `completed` after their target permanently terminates, flushing pending activity and releasing their active slot. Watches of idle persistent agents remain active. Watching an already-finished target returns a completed subscription.

## Acknowledgements

- [headlong](https://github.com/laude-institute/headlong) (Laude Institute) for the tiered memory design that the compaction staircase, rollups, and verbatim tail follow: `design/tiered_memory.md` and `bin/recap --context`.
- [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) (Prime Intellect) for the continual harness: the prompt/memory/skill/subagent entries, layered stores, refinement passes, and the skill contract that nano-rlm's harness adopts.
