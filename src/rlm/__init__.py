"""rlm — A minimalistic CLI agent for true recursion."""

from rlm import agent, harness, hints, inbox, peer, refine, services, shell, watch
from rlm.config import ExecutionPolicy, InvocationContext, ProviderConfig, RuntimeConfig
from rlm.engine import RLMEngine
from rlm.history import History, history
from rlm.types import RLMMetrics, RLMResult

__all__ = [
    "agent",
    "harness",
    "hints",
    "inbox",
    "peer",
    "refine",
    "services",
    "shell",
    "watch",
    "history",
    "History",
    "ExecutionPolicy",
    "InvocationContext",
    "ProviderConfig",
    "RLMEngine",
    "RLMMetrics",
    "RLMResult",
    "RuntimeConfig",
]


def __getattr__(name: str):
    if name == "scratchpad":
        import os
        from pathlib import Path

        path = os.environ.get("RLM_SCRATCHPAD")
        if not path:
            raise AttributeError(
                "rlm.scratchpad is unavailable: the mesh is disabled (policy.max_peers is 0)"
            )
        return Path(path)
    raise AttributeError(f"module 'rlm' has no attribute {name!r}")
