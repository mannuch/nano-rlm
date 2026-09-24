import logging
from collections.abc import Iterator

import verifiers.v1 as vf
from openthoughts_tblite.taskset import (
    OpenThoughtsTBLiteConfig,
    OpenThoughtsTBLiteTaskset,
)
from pydantic import Field
from verifiers.v1.tasksets.harbor import HarborTask

logger = logging.getLogger(__name__)


class RlmRefineConfig(OpenThoughtsTBLiteConfig):
    allow_hosts: list[str] = Field(default_factory=lambda: ["api.typesafe.ai"])
    """Hosts added to every network-restricted task's allowlist: the refinement judge
    is called from inside the sandbox."""


# verifiers finds the config type from this base; ty rejects re-parameterizing a
# generic base, as it does for OpenThoughtsTBLiteTaskset itself.
class RlmRefineTaskset(  # ty: ignore[invalid-generic-class]
    OpenThoughtsTBLiteTaskset,
    vf.Taskset[HarborTask, RlmRefineConfig],  # ty: ignore[invalid-type-arguments]
):
    """OpenThoughts TBLite, unchanged except that network-restricted tasks may also
    reach `allow_hosts`. A runtime allowlist can't do this: verifiers intersects it
    with the task's own policy, and an offline task stays offline. At the pinned
    prime-envs commit every task is public, so this changes nothing until TBLite's
    tasks are made offline."""

    config: RlmRefineConfig

    def load(self) -> Iterator[HarborTask]:
        widened = 0
        for task in super().load():
            allow = task.data.network_allow
            if "*" not in allow:
                extra = [h for h in self.config.allow_hosts if h not in allow]
                task.data = task.data.model_copy(
                    update={"network_allow": [*allow, *extra]}
                )
                widened += 1
            yield task
        logger.info(
            "rlm-refine: allowlisted %s in %d restricted tasks",
            self.config.allow_hosts,
            widened,
        )
