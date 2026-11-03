# Continual harness

The continual harness is durable state that supplements the immutable system prompt: `prompt`
notes (narrow behavioural policies), `memory` entries (facts, decisions, failures), `skill`
entries (how to use an importable Python tool, with a `reference` and an
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
  "auto_refine_review": "judge",
  "refine_turn_interval": 12,
  "refine_cooldown_seconds": 300,
  "max_refinements": null,
  "max_refinement_attempts": 3,
  "refine_judge": null,
  "author_skills": false,
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
h.list("episode")                              # every visible entry of one kind
h.get("memory", "ancestor:parent_lesson")      # ids exactly as displayed
h.create_memory("Project venv", "run tests with ./.venv/bin/python -m pytest")
h.create_skill("Release lookup", "query websearch with '<pkg> release'",
               reference={"type": "python", "import": "websearch", "callable": "run",
                          "call_pattern": "await websearch(queries=[...])"},
               arguments={"queries": {"type": "array", "required": True}})
h.update_memory("project_venv", "Project venv", "...", global_=True)   # needs global_dir
```

`update_*`/`delete_*` take the bare id or a `local:`/`global:` prefix; ancestor entries raise
`PermissionError`. Every method carries a docstring, so `help(h)` (or `help(h.create_skill)`) is
the full reference; the system prompt only lists the essential calls. A `skill` entry describes how to use an already-importable Python tool; it is not a
package (see [Python tool packages](tools.md#python-tool-packages) for the on-disk package contract). Like
`rlm.agent`, the kernel's view is served by the session supervisor through the broker:
each call runs on the calling agent's own stores and needs an active cell, and the kernel
never names a store. The store objects themselves (`h.local`, `h.layers()`) are not
available in the kernel. Stores are rewritten atomically under a file lock and reloaded
when another writer changed them, so the engine and the supervisor share one file safely.
`harness(session_dir=...)` loads a session's local store directly, for example outside a
running session.

## The global store

The only cross-session file is the optional global store, which lives wherever
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

## Entry schema

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

## Skill entries versus Python tools

The harness works with three kinds of object. In [Tools and skills](tools.md)
terms, an installed package is a Python tool, an authored package is a skill that ships a
Python tool, and a harness `skill` entry is a skill kept as harness state:

|                 | Installed Python tool                      | Authored skill                                                | Harness `skill` entry                                                                                                   |
| --------------- | ------------------------------------------ | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| What it is      | A Python package installed into rlm's venv | A Python package under `$RLM_HOME/skills`, on `sys.path`      | A JSON record in `harness_state.json`                                                                                   |
| Contains        | Code: `async def run(...)`                 | Code: `async def run(...)`                                    | Text: title, when/how to call something, a `reference` (`import`, `callable`, `call_pattern`) and an `arguments` schema |
| Created by      | A human, via `install.sh` at image build   | The agent, mid-session, by writing files                      | The agent (`h.create_skill(...)`) or the refinement pass                                                                |
| Lives           | The venv; every session                    | `$RLM_HOME/skills/<name>/`; sessions sharing `RLM_HOME`       | One store: session-local, an ancestor's, or global                                                                      |
| The kernel sees | `await websearch(...)`                     | `await greeter(...)`                                          | Nothing executable; it is rendered into the system prompt                                                               |
| Refinement can  | Never touch it                             | Never touch it                                                | Create, update, delete, roll back                                                                                       |

The relationship is pointer to target. An entry's `reference.import` names a module that
must already be importable (an installed Python tool, an authored skill's tool, a
built-in Python tool from the contract's `skills` list, or `rlm` itself); apply-time validation rejects an entry that points anywhere else. The
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

## Authored skills

`author_skills` lets the agent write skills that survive the session into the skills root,
`$RLM_HOME/skills`, which also holds the host's [skills](tools.md#skills) and Python tool packages.
An authored skill is a `SKILL.md` that ships a Python tool package. The
[package contract](tools.md#python-tool-packages) applies, minus installation, with these
adjustments:

- Required: `$RLM_HOME/skills/<name>/SKILL.md` and
  `$RLM_HOME/skills/<name>/src/<name>/__init__.py` defining `async def run(...)`. The kernel
  checks both; `run`'s docstring falls back to `SKILL.md` for `help(<name>)`.
- Optional: `pyproject.toml`; when present its `[project] name` must be `rlm-skill-<name>`
  so the package can be promoted to an installed Python tool unchanged. Nothing is installed, so
  dependencies are limited to what is already in the kernel venv and there is no console
  script: authored packages are IPython-only.

At kernel start every authored package's `src/` goes on `sys.path` and the module is
pre-imported by name with the same `await <name>(...)` wrapper as an installed Python tool. A
package that breaks the contract or fails to import binds a placeholder whose call raises
the reason, so a bad package never breaks the kernel. Names must not collide with installed
Python tools or the built-in Python tools the session generates from the contract's `skills` list. A package written or edited during the session is brought in
without a restart by `rlm.harness.load_skills("<name>")` (no names = all), which
reloads and rebinds it exactly as kernel start does and returns `{name: reason}` with
`None` for a usable tool. The system prompt states the contract and this workflow;
`RLM_HARNESS_SKILLS_DIR` names the directory.

The `## Skills` catalog scans the root each time a system prompt is built, so a skill
written mid-session, with `name` and `description` frontmatter, is listed for sub-agents
spawned afterwards, for later sessions, and for the authoring agent once a refinement
rebuilds its prompt. Until then, every running agent is told at its next model step: a
`<runtime_event kind="skills">` lists each skill added after its prompt was built, once,
in the catalog's line format. With `author_skills` on, it also says that a tool another
agent wrote is bound with `load_skills`. Compaction keeps the announced skills in the
continuation summary until the prompt is rebuilt. The prompt is not rebuilt just to list
them, which would discard the prompt cache.
A package that `install.sh` installed from the root, editable from its own folder, is an
installed Python tool rather than an authored one: it is pre-imported from the venv,
`load_skills` declines to rebind it, and edits take effect at the next kernel start. A
folder with no `src/`, `pyproject.toml` or `__init__.py` is an instruction-only skill and is
never loaded as a package.

`author_skills` is off by default, and no authored packages load.

## Refinement

Refinement is a side model call, like compaction: the live conversation is extended with
one user message asking for a JSON proposal of create/update/delete edits, the proposal is
validated and applied under the store's lock, the system prompt is rebuilt in place (a new
`context_window` with `reason="harness"` that keeps every other message index), and a
`<runtime_event kind="refinement">` notice tells the model what changed. Rejected edits
(unknown module in a `skill` reference, edits to the base system prompt, entries that
changed while planning, …) are recorded with an error and the rest still apply. Each
store's `refinements.jsonl` holds one full result per pass with per-edit `before`/`after`
snapshots; a rollback replays those snapshots in reverse and needs no model call.

The planning call also decides whether to refine at all: a proposal with an empty `edits`
array declines the pass. A declined pass changes nothing (no store write, no rebuilt
window, no `refinement` edge) and is recorded as a `refinement_declined` ledger record with
the planner's rationale; only a kernel-requested pass tells the model, in context. An
applied pass is a `refinement` record. Both records carry the `trigger`, any compaction
`blocks` behind the pass and, when the TypeSafe judge ran, its verdict under `judge`.

Three triggers, all of which run between model calls and never inside a cell:

- **Kernel**: `await rlm.refine.run(instructions=None, global_=False, rollback_id=None)`
  returns `{"scheduled": True}` (or a reason) and the pass runs at the next boundary;
  `await rlm.refine.status()` reports `pending`/`in_flight`.
- **Host**: `session/prompt` may carry `ai.prime.rlm/refine-v1` in `_meta`:
  `{"instructions": "...", "global": false, "rollback_id": null, "review": false, "focus": false}`.
  The pass runs before the turn; with an empty prompt it is the whole turn and the notice
  (or `[refinement declined: <rationale>]`) is the answer (`stop_reason` `refined`).
  `review: true` lets the TypeSafe judge decline the pass before any model call;
  `focus` has the judge write the plan's focus instructions, with any host `instructions`
  appended after them. A rollback takes neither. The key is refused when the harness is
  disabled, and `review` or `focus` is refused without `refine_judge`.
- **Auto** (`auto_refine`, off by default, root agent only): every `refine_turn_interval`
  work turns and after each compaction, subject to `refine_cooldown_seconds`, the TypeSafe
  judge reviews the pass (see below) and, unless it declines, a planning
  call runs with an `<automatic_refinement>` section: the pass was not requested, so edit
  only on evidence useful to this session's future turns (a repeated failure, a reusable
  tactic, a repeated delegation role, a durable fact or preference, a user correction) and
  otherwise decline with no edits. After a compaction the plan also receives the blocks that
  compaction produced (the branch summary and any rollups it sealed) as `<evidence>`, with
  the hint that a failure, tactic or fact recurring across blocks is evidence for a durable
  entry while one branch's progress is not. The call runs with `tool_choice="none"`, so the
  evidence is the block text itself, never a pointer to follow.

### TypeSafe review judge

`refine_judge` puts TypeSafe's System One model (Jev) in front of the planning call. Jev
answers typed yes/no and choice questions with calibrated probabilities; the task model
still writes the plan and may still decline it. Automatic passes need the judge:
`auto_refine` without `refine_judge` fails validation at `session/new`. Only
`auto_refine_review: "planner"` lets the task model decide automatic passes alone, with
no judge call, e.g. as a baseline. For host passes the judge stays opt-in (`review`,
`focus`). It runs for the root agent only and sends a compact evidence state to the TypeSafe API: the messages since the last pass
(clipped, newest first, user turns kept ahead of the rest), exception counts, the
compaction blocks behind the pass, and the visible harness entries with the entries of
the store the pass writes first, alongside that store's refinement history.

```json
"refine_judge": {
  "api_key": "...",
  "base_url": null,
  "model": "jev-latest",
  "mode": "gate",
  "threshold": 0.7,
  "veto_threshold": 0.8,
  "home_confidence": 0.6,
  "timeout_s": 30.0
}
```

A review is two calls. The **gate** asks one yes/no question per kind of lesson
(repeated failure, reusable tactic, delegation role, durable fact, user correction, and,
in a local pass a compaction started, working notes: state from the turns just
summarized away that the rest of the task still needs) and one per entry the pass could
fix: is it contradicted by the conversation? That covers
entries of the store the pass writes and, in a local pass, read-only global or ancestor
entries, which get a local override. Nothing at `threshold` declines after that one
call. The **focus** call is asked only about the lessons that fired, stated as
premises: per lesson, whether it is already recorded (a veto at `veto_threshold`) and
which kind should hold it; per entry of the store the pass writes, whether it covers a
lesson; per turn, whether it is direct evidence. Questions are worded
for the pass's scope: a local pass serves later work in this session (the rest of the
current task, which may continue after older turns are summarized away, and any later
tasks), a global pass (host `global: true`) future sessions. A fact or technique that
served only a step already finished doesn't count. Code turns the answers into deterministic plan
instructions naming the home kind (one kind when the choice's confidence reaches
`home_confidence`), the entries to update or delete, the read-only entries a local entry
should override, and quotes of the strongest turns.
They reach the planner as `<refine_instructions>`, the same slot host and kernel
instructions use.

`mode: "gate"` lets the judge decline automatic passes before any model call; `"shadow"`
leaves the decision to the planner and only records the judge's verdict. The pass record's
`judge` holds the mode, the gate and focus probabilities, the evidence turns it chose and
TypeSafe token usage per call. A judge failure fails a gated pass; where the judge only
informs the plan (`focus`, `shadow`) the plan runs without it and the failure is recorded
as `judge.error`. `scripts/refine_eval/run.py` scores these pieces on labeled, steered
scenarios: decisions against labels, who declined, focus against expected kinds and
entries, and applied edits and follow-up probes per arm (`force`, `force+focus`,
`typesafe`, `none`). Scenarios can seed either store and run a global pass, covering a
stale global entry, a global entry a local pass must override, and a session-only fact
a global pass should decline. `scripts/refine_eval/RUNBOOK.md` covers running it and
reading the report.

`max_refinements` caps applied passes per engine; `max_refinement_attempts` bounds how
often an unusable reply (truncated JSON, prose, a tool call) is resampled before the pass is
reported as failed in the conversation and the run continues. A pass is one planning
request (plus any resampled attempts), each carrying a `refinement_attempt` semantic edge
from the last work request; an applied plan becomes the source of a `refinement` edge into
the next work request, which also keeps its ordinary `continuation` edge. A declined plan
is a dead end, and a pass the judge declines makes no request at all. Refinement counts,
edit totals, declines by reason and the judge's reviews, errors, input tokens and
disagreements with the planner appear in the session metrics; the `session-v1` snapshot
carries per-scope entry counts under `harness`.

## Episodes

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
than `max_prompt_entries_per_kind`; from the kernel, `h.list("episode")` lists them,
`h.search(query, kind="episode")` ranks them (an empty query returns the newest), and
`history(session_dir=h.get("episode", id).metadata["session_dir"])` opens its ledger for
`.blocks` / `.expand(i)`. Each session also logs an `episode` record naming what it
published.

Episodes are engine-owned: `create`/`update`/`delete` of that kind raise `PermissionError`
from the kernel API and refinement edits of that kind are rejected, so nothing inside a
session can rewrite or remove another session's record. The engine never deletes them
either — a per-session setting must not prune a shared store — so trimming a long-lived
store is an operator command run outside any session:

```bash
rlm harness prune --global-dir ~/.rlm/global --older-than 30d          # dry run
rlm harness prune --global-dir ~/.rlm/global --older-than 30d --keep 200 --apply
```

`--older-than` (`30d`, `12h`, `4w`) selects by the session's start time and `--keep N`
drops everything beyond the N newest; without `--apply` the command only prints what would
go. `--global-dir` defaults to `$RLM_HARNESS_GLOBAL_DIR`. The removal runs under the store
lock and is recorded like a refinement — an `operator:prune` event in `harness_state.json`
and a result in `refinements.jsonl` with each removed entry's snapshot — so it is
inspectable and reversible by hand, while no in-session path can recreate an episode. Both gates default to off because a global store shared
across RL rollouts of one task would let a rollout read another's outcome. A store that
holds episodes cannot be read by builds that predate the kind.

## In split execution

With [split execution](split-execution.md) the harness keeps working as
described above, and its stores stay with the engine.

The worker keeps a mirror of `$RLM_HOME/skills` in its session directory, filled before
the kernel starts and updated before each tool call, so every skill the prompt lists is
readable there. The prompt's skill paths, including `RLM_HARNESS_SKILLS_DIR`, name the
mirror. With `author_skills` on, the kernel binds and writes authored packages in the
mirror, and the files it changes return with each tool result to `$RLM_HOME/skills`.
They persist there, and the engine announces them to every agent. The receiving side
writes only regular files inside skill folders. It rejects absolute paths, `..`, and
paths through symlinks, and never imports what it receives. Files over 4 MiB and
bytecode are not mirrored.

`history(session_dir)` and `harness(session_dir)` for a path the worker doesn't have ask
the supervisor. The supervisor serves the ledger in pages, or the session's local store
read-only, when the path is the `session_dir` of an episode in the agent's harness, and
refuses any other path.
