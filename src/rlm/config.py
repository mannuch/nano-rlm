"""Validated runtime configuration for RLM engines.

Configuration enters an rlm process exactly once, through the versioned ACP
runtime contract (``ai.prime.rlm/runtime-v1``); recursive children inherit it
in-memory via ``model_copy``. There is no environment-variable resolution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from rlm.semantic import ACP_EXTENSION_HEADER_NAMES


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProviderConfig(_ConfigModel):
    """Credentials and transport options for one inference provider."""

    base_url: str | None
    api_key: str = Field(min_length=1, repr=False)
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    max_retries: int = Field(default=5, ge=0)

    @field_validator("headers")
    @classmethod
    def _reserve_transport_headers(cls, headers: dict[str, str]) -> dict[str, str]:
        reserved_names = {
            "idempotency-key",
            "x-stainless-retry-count",
            *(header.lower() for header in ACP_EXTENSION_HEADER_NAMES),
        }
        reserved = sorted(name for name in headers if name.lower() in reserved_names)
        if reserved:
            raise ValueError(f"provider headers contain reserved names: {reserved}")
        return headers


class InvocationContext(_ConfigModel):
    """Trusted identity of one engine within a recursive session tree."""

    depth: int = Field(default=0, ge=0)
    ancestor_harness_dirs: tuple[str, ...] = ()
    """Local harness directories of every ancestor, nearest first; children read
    them but never write to them."""

    def child(self, harness_dir: str | None = None) -> InvocationContext:
        ancestors = self.ancestor_harness_dirs
        if harness_dir is not None:
            ancestors = (harness_dir, *ancestors)
        return InvocationContext(depth=self.depth + 1, ancestor_harness_dirs=ancestors)


class RefineJudgeConfig(_ConfigModel):
    """TypeSafe System One judge for refinement reviews: typed yes/no signals decide
    whether to refine and what the refinement should focus on; the task model still
    plans the edits."""

    api_key: str = Field(min_length=1, repr=False)
    base_url: str | None = None
    model: str = "jev-latest"
    mode: Literal["gate", "shadow"] = "gate"
    """``gate``: the judge replaces the model's auto-refine review. ``shadow``: the
    model review still decides and the judge's verdict is only logged."""
    threshold: float = Field(default=0.7, ge=0, le=1)
    """Probability at which a gate signal fires or an entry is flagged."""
    veto_threshold: float = Field(default=0.8, ge=0, le=1)
    """Probability at which a fired lesson counts as already recorded."""
    home_confidence: float = Field(default=0.6, ge=0, le=1)
    """Choice confidence needed to name a single harness kind for a lesson."""
    timeout_s: float = Field(default=30.0, gt=0)


class HarnessConfig(_ConfigModel):
    """Continual harness: durable prompt notes, memories, skill descriptions and
    sub-agent specs rendered into the system prompt as a compact block."""

    enabled: bool = True
    global_dir: str | None = None
    """Directory of a store shared across sessions. None (default) keeps every
    session hermetic: only the session-local store and ancestors are visible."""
    max_prompt_entries_per_kind: int = Field(default=6, gt=0)
    max_prompt_content_chars: int = Field(default=180, gt=0)
    max_prompt_refinements: int = Field(default=5, ge=0)
    auto_refine: bool = False
    """Let the root engine review its own trajectory every ``refine_turn_interval``
    work turns (and after each compaction) and refine when the review approves."""
    auto_refine_review: Literal["judge", "planner"] = "judge"
    """Who decides an automatic pass. ``judge`` needs ``refine_judge`` whenever
    ``auto_refine`` is on, and the judge's ``mode`` gates or shadows the planning call.
    ``planner`` leaves the decision to the task model alone and never consults the
    judge, e.g. for a baseline without it."""
    refine_turn_interval: int = Field(default=12, gt=0)
    refine_cooldown_seconds: int = Field(default=300, ge=0)
    max_refinements: int | None = Field(default=None, gt=0)
    """Refinement passes per engine before further requests are declined."""
    max_refinement_attempts: int = Field(default=3, gt=0)
    """Proposal attempts within one pass; an unusable reply is resampled."""
    refine_judge: RefineJudgeConfig | None = None
    """TypeSafe judge for auto-refine reviews and host-requested reviews. Required by
    ``auto_refine`` unless ``auto_refine_review`` is ``planner``."""
    skills_dir: str | None = None
    """Persistent directory of agent-authored skill packages, put on the kernel's
    sys.path at start. None (default) keeps authored packages session-local."""
    record_episodes: bool = False
    """Write one ``episode`` entry per root session into the global store at close:
    the task, the coarsest compaction blocks, the outcome and the session directory.
    Requires ``global_dir``. Off by default because a global store shared across RL
    rollouts would let one rollout read another's outcome."""

    @model_validator(mode="after")
    def _validate_auto_refine_review(self) -> Self:
        if (
            self.auto_refine
            and self.auto_refine_review == "judge"
            and self.refine_judge is None
        ):
            raise ValueError(
                "auto_refine needs refine_judge; set auto_refine_review to "
                "'planner' to let the task model decide automatic passes alone"
            )
        return self


class ExecutionPolicy(_ConfigModel):
    """Resource and context-management policy for one RLM engine."""

    max_depth: int = Field(default=1, ge=0)
    delegation_prompt: bool = False
    """Append the delegation guidance (when to spawn, how to brief, watch, collect and
    reconcile children) to the system prompt of every agent that can still delegate."""
    max_total_turns: int | None = Field(default=None, gt=0)
    """Tree-total turn budget (one turn = one work-loop model call, any engine; compaction
    calls don't count). Once reached, every engine stops before its next model call
    (stop_reason=max_total_turns). None = uncapped."""
    max_total_tokens: int | None = Field(default=1_000_000, gt=0)
    """Tree-total budget of NEW tokens across the whole recursive session tree, live: each
    model call contributes its completion plus uncached prompt tokens (the cached context
    prefix re-billed every call is not new work). Once reached, every engine stops before
    its next model call (stop_reason=max_total_tokens) and no further sub-agents are
    spawned. This budget is the default terminator (compaction never runs out of context,
    so without it a stuck session would run forever). None = unbounded."""
    max_tool_output_bytes: int | None = Field(default=None, gt=0)
    """Byte budget for a single tool result entering the conversation (middle truncation,
    head + tail, with a warning naming the original size). None = the built-in 20KB
    default; an explicit value overrides it in either direction."""
    exec_timeout: int = Field(default=300, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    compaction: bool = True
    """Compact the context once it outgrows ``summarize_at_tokens`` (and recover from
    provider context-overflow errors by checkpointing). On by default; set False to
    let an overflowing session fail instead."""
    summarize_at_tokens: int | None = Field(default=None, gt=0)
    """Compaction threshold. None = auto-discover from the provider's advertised
    context window (~16k tokens of headroom); when the provider advertises no
    window, compaction stays overflow-reactive only."""
    max_compactions: int | None = Field(default=None, gt=0)
    """Compactions per session before the engine stops compacting. None (default) =
    unlimited: every compaction cycle itself spends new tokens, so ``max_total_tokens``
    still bounds the session."""
    max_compaction_attempts: int = Field(default=5, gt=0)
    """Summary-generation attempts within one compaction cycle."""
    compaction_fanout: int = Field(default=5, ge=2)
    """Blocks per rollup in the compaction staircase: each compaction adds a tier-1 block
    (the branch summary) and every ``fanout`` consecutive blocks of one tier merge into
    one block of the next, so tier k covers fanout^(k-1) branches."""
    compaction_tail_tokens: int = Field(default=12_000, ge=0)
    """Estimated tokens of the most recent messages kept verbatim after the staircase
    when a window is compacted, capped at a quarter of the compaction threshold. 0
    keeps none: the compacted window ends with the staircase."""
    compaction_prompt_tokens: int = Field(default=4_000, ge=0)
    """Estimated-token cap for keeping the current prompt verbatim in a compacted
    window, ahead of the staircase. A larger prompt is referenced by ledger index in
    the staircase message instead; 0 never keeps it."""
    max_concurrent_subagents: int = Field(default=4, gt=0)
    max_subagent_calls: int | None = Field(default=None, gt=0)
    """Tree-total cap on sub-agent spawns. None (default) = uncapped; ``max_total_tokens``
    still bounds the tree."""
    allow_git: bool = False

    @model_validator(mode="after")
    def _validate_concurrency(self) -> Self:
        if self.max_concurrent_subagents < self.max_depth:
            raise ValueError("max_concurrent_subagents must be at least max_depth")
        return self


def validate_prompt_overrides(overrides: dict[str, str]) -> dict[str, str]:
    """Check override names against ``rlm.prompt.DEFAULT_PROMPTS`` and that each text
    keeps the markers the runtime fills in."""
    from rlm.prompt import DEFAULT_PROMPTS, REQUIRED_PROMPT_MARKERS

    for name, text in overrides.items():
        if name not in DEFAULT_PROMPTS:
            raise ValueError(
                f"unknown prompt {name!r}; overridable prompts: {sorted(DEFAULT_PROMPTS)}"
            )
        if not text.strip():
            raise ValueError(f"prompt override {name!r} is empty")
        missing = [m for m in REQUIRED_PROMPT_MARKERS.get(name, ()) if m not in text]
        if missing:
            raise ValueError(f"prompt override {name!r} must contain {missing}")
    return overrides


class RuntimeConfig(_ConfigModel):
    """Configuration resolved once at an RLM process boundary."""

    model: str = Field(min_length=1)
    provider: ProviderConfig
    invocation: InvocationContext
    policy: ExecutionPolicy
    system_prompt_path: str | None = None
    append_to_system_prompt: str | None = None
    subagent_append_to_system_prompt: str | None = None
    leaf_append_to_system_prompt: str | None = None
    skills: tuple[str, ...] = ()
    builtin_tools: tuple[str, ...] | None = None
    """Builtin tool set for every engine in the tree; None = the registry default
    (`ipython` alone). Validated against the registry when the engine starts."""
    kernel_env: tuple[tuple[str, str], ...] = Field(default=(), repr=False)
    search_api_key: str | None = Field(default=None, repr=False)
    harness: HarnessConfig = HarnessConfig()
    prompt_overrides: dict[str, str] = Field(default_factory=dict)
    """Replacement texts for named runtime prompts (see ``rlm.prompt.DEFAULT_PROMPTS``),
    applied to every engine in the tree."""

    @field_validator("prompt_overrides")
    @classmethod
    def _known_prompts(cls, overrides: dict[str, str]) -> dict[str, str]:
        return validate_prompt_overrides(overrides)

    @property
    def resolved_append_to_system_prompt(self) -> str | None:
        """The append for this engine's role in the session tree.

        root (depth 0)                            -> append_to_system_prompt
        node (depth >= 1, can still recurse)      -> subagent_append_to_system_prompt
        leaf (depth == max_depth, cannot recurse) -> leaf_append_to_system_prompt

        Unset tiers fall back in order: leaf -> subagent -> root.
        """
        if self.invocation.depth == 0:
            return self.append_to_system_prompt
        if (
            self.invocation.depth >= self.policy.max_depth
            and self.leaf_append_to_system_prompt is not None
        ):
            return self.leaf_append_to_system_prompt
        if self.subagent_append_to_system_prompt is not None:
            return self.subagent_append_to_system_prompt
        return self.append_to_system_prompt
