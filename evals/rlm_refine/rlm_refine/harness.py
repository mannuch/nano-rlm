"""nano-rlm over ACP, installed from a git commit, with its harness config passed
through so auto-refinement and the TypeSafe judge can be turned on."""

import hashlib
import shlex
from typing import Any

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field
from verifiers.v1.harnesses.utils.install import ensure_installed

# nano-rlm's ACP contract (src/rlm/acp.py).
RUNTIME_METADATA_KEY = "ai.prime.rlm/runtime-v1"
SESSION_METADATA_KEY = "ai.prime.rlm/session-v1"
REFINEMENTS_METADATA_KEY = "ai.prime.rlm/refinements-v1"

TYPESAFE_API_KEY = "TYPESAFE_API_KEY"
STATE_DIR = "/tmp/rlm-refine-state"

# nano-rlm stop reasons that truncate the rollout, as verifiers stop conditions.
TRUNCATION_STOPS = {
    "max_total_tokens": "max_total_tokens",
    "max_total_turns": "max_turns",
    "token_budget": "max_output_tokens",
    "compaction_failed": "compaction_failed",
}


class _SessionSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    session_id: str
    metrics: dict[str, int | float]
    last_stop_reason: str | None = None


class RlmRefineHarnessConfig(vf.HarnessConfig):
    repo: str = "https://github.com/mannuch/nano-rlm.git"
    version: str | None = None
    """The nano-rlm commit to install (required). A SHA: installs are cached by it."""
    policy: dict[str, Any] = Field(default_factory=dict)
    """nano-rlm's `ExecutionPolicy`, sent as runtime-v1 `policy`."""
    harness: dict[str, Any] | None = None
    """nano-rlm's `HarnessConfig`, sent as runtime-v1 `harness`. A `refine_judge` gets
    its `api_key` from the forwarded `TYPESAFE_API_KEY`."""
    append_to_system_prompt: str | None = None


class RlmRefineHarness(vf.ACPHarness[RlmRefineHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime: vf.Runtime) -> None:
        if not self.config.version:
            raise ValueError(
                "set the harness version to the nano-rlm commit to install"
            )
        if self._judged() and not self.config.resolved_env.get(TYPESAFE_API_KEY):
            raise ValueError(
                f"harness.refine_judge needs {TYPESAFE_API_KEY}: set it and add it to "
                "forward_env"
            )
        directory = self._install_dir()
        checkout = f"{directory}/checkout"
        ready = f"{directory}/.ready"
        install = (
            f"rm -f {ready} && "
            "{ command -v git >/dev/null 2>&1 || "
            "{ apt-get update -qq && apt-get install -y -qq git; }; } && "
            f"rm -rf {checkout} && git init -q {checkout} && "
            f"git -C {checkout} fetch -q --depth 1 {shlex.quote(self.config.repo)} "
            f"{shlex.quote(self.config.version)} && "
            f"git -C {checkout} checkout -q FETCH_HEAD && "
            f"UV_INSTALL_DIR={directory}/bin UV_TOOL_BIN_DIR={directory}/bin "
            f"UV_TOOL_DIR={directory}/tools RLM_CHECKOUT_PATH={checkout} "
            f"bash {checkout}/install.sh && touch {ready}"
        )
        await ensure_installed(
            runtime,
            directory=directory,
            ready=f"[ -f {ready} ] && [ -x {directory}/bin/rlm ]",
            install=install,
            env=self.config.resolved_env,
            label="rlm",
        )
        await super().setup(runtime)

    async def prepare_acp(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
    ) -> vf.ACPConfig:
        system_prompt, prompt = self.resolve_prompt(data)
        appends = [
            text
            for text in (system_prompt, self.config.append_to_system_prompt)
            if text
        ]
        harness = self.config.harness
        if harness and harness.get("refine_judge"):
            judge = {
                **harness["refine_judge"],
                "api_key": self.config.resolved_env[TYPESAFE_API_KEY],
            }
            harness = {**harness, "refine_judge": judge}
        payload = {
            "session_id": trace.id,
            "model": ctx.model,
            "provider": {"base_url": endpoint, "api_key": secret},
            "policy": self.config.policy,
            "system_prompt_path": None,
            "append_to_system_prompt": "\n\n".join(appends) or None,
            "skills": [],
            "kernel_env": runtime.env,
            "search_api_key": None,
            "harness": harness,
        }
        return vf.ACPConfig(
            env={**self.config.resolved_env, "RLM_HOME": f"{self._state(trace)}/home"},
            command=[f"{self._install_dir()}/bin/rlm", "--acp"],
            prompt=prompt,
            session_meta={RUNTIME_METADATA_KEY: payload},
        )

    def acp_turn_result(self, trace: vf.Trace, result: vf.ACPTurn) -> None:
        self._record_snapshot(trace, result.response_metadata)

    def acp_close_result(
        self, trace: vf.Trace, response_metadata: dict[str, Any]
    ) -> None:
        if SESSION_METADATA_KEY in response_metadata:
            self._record_snapshot(trace, response_metadata)
        if REFINEMENTS_METADATA_KEY in response_metadata:
            trace.info["rlm_refinements"] = response_metadata[REFINEMENTS_METADATA_KEY]

    async def cleanup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.run(["rm", "-rf", self._state(trace)], {})

    def _record_snapshot(self, trace: vf.Trace, metadata: dict[str, Any]) -> None:
        snapshot = _SessionSnapshot.model_validate(metadata.get(SESSION_METADATA_KEY))
        if snapshot.session_id != trace.id:
            raise ValueError("RLM session snapshot does not match the rollout")
        trace.record_metrics(snapshot.metrics)
        if condition := TRUNCATION_STOPS.get(snapshot.last_stop_reason or ""):
            trace.stop(condition)

    def _judged(self) -> bool:
        return bool(self.config.harness and self.config.harness.get("refine_judge"))

    def _install_dir(self) -> str:
        key = f"{self.config.repo}@{self.config.version}".encode()
        return f"/tmp/rlm-refine-{hashlib.sha256(key).hexdigest()[:16]}"

    @staticmethod
    def _state(trace: vf.Trace) -> str:
        return f"{STATE_DIR}/{trace.id}"
