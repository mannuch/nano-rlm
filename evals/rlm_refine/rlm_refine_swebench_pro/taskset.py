from collections.abc import Iterator

import verifiers.v1 as vf
from pydantic import Field
from swebench_pro.taskset import SWEBenchProConfig, SWEBenchProTask, SWEBenchProTaskset

from rlm_refine.taskset import TYPESAFE_HOST, allowlisted


class RlmRefineSWEBenchProConfig(SWEBenchProConfig):
    allow_hosts: list[str] = Field(default_factory=lambda: [TYPESAFE_HOST])
    """Hosts added to every task's allowlist: the agent box is otherwise offline."""


# verifiers finds the config type from this base; ty rejects re-parameterizing a
# generic base, as it does for SWEBenchProTaskset itself.
class RlmRefineSWEBenchProTaskset(  # ty: ignore[invalid-generic-class]
    SWEBenchProTaskset,
    vf.Taskset[SWEBenchProTask, RlmRefineSWEBenchProConfig],  # ty: ignore[invalid-type-arguments]
):
    """SWE-bench Pro V2, unchanged except that the offline agent box may also reach
    `allow_hosts`. Grading still replays the agent's diff in a pristine box."""

    config: RlmRefineSWEBenchProConfig

    def load(self) -> Iterator[SWEBenchProTask]:
        return allowlisted(
            super().load(), self.config.allow_hosts, "rlm-refine-swebench-pro"
        )
