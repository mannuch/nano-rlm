"""The continual harness across split execution.

In split execution the kernel runs in a workspace worker while the harness stores stay
on the engine's machine. ``RemoteHarnessView`` gives the kernel the ``HarnessView`` API
over the broker, and the supervisor answers each call with :func:`dispatch` against the
calling agent's own view, so the kernel never names a store path.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, get_args

from rlm import broker
from rlm.config import HarnessConfig
from rlm.harness import (
    HarnessEntry,
    HarnessView,
    RefinementEvent,
    build_view,
    local_dir,
)

REMOTE_ENV = "RLM_HARNESS_REMOTE"
"""Set to ``"1"`` in a split-execution kernel whose harness lives with the engine."""

METHODS: tuple[str, ...] = get_args(broker.HarnessMethod)

_RAISED: tuple[type[Exception], ...] = (
    PermissionError,
    KeyError,
    TypeError,
    ValueError,
    RuntimeError,
)
"""Exceptions a harness call re-raises in the kernel with their type; a subclass comes
back as the first of these it inherits from."""

_STORE_ATTRIBUTES = frozenset({"local", "global_", "ancestors", "layers"})


def agent_view(
    session_dir: str | Path, config: HarnessConfig, ancestors: tuple[str, ...]
) -> HarnessView | None:
    """The view an agent's engine builds for itself, or None when the harness is off."""
    if not config.enabled:
        return None
    return build_view(
        local_dir(session_dir), global_dir=config.global_dir, ancestor_dirs=ancestors
    )


def encode(value: Any) -> Any:
    if isinstance(value, HarnessEntry):
        return {"__harness__": "entry", "value": value.model_dump(mode="json")}
    if isinstance(value, RefinementEvent):
        return {"__harness__": "refinement", "value": value.model_dump(mode="json")}
    if isinstance(value, tuple):
        return {"__harness__": "tuple", "value": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    return value


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("__harness__")
    if tag == "entry":
        return HarnessEntry.model_validate(value["value"])
    if tag == "refinement":
        return RefinementEvent.model_validate(value["value"])
    if tag == "tuple":
        return tuple(decode(item) for item in value["value"])
    return {key: decode(item) for key, item in value.items()}


def dispatch(
    view: HarnessView | None, method: str, args: list[Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Run one kernel harness call on ``view``: ``{"value": ...}``, or ``{"raised":
    {"type", "message"}}`` for an error the kernel should see with its type."""
    if method not in METHODS:
        raise ValueError(f"unknown harness method {method!r}")
    try:
        if view is None:
            raise RuntimeError("the continual harness is disabled for this session")
        result = getattr(view, method)(*args, **kwargs)
    except _RAISED as error:
        kind = next(cls for cls in _RAISED if isinstance(error, cls))
        message = error.args[0] if isinstance(error, KeyError) else str(error)
        return {"raised": {"type": kind.__name__, "message": message}}
    return {"value": encode(result)}


class RemoteHarnessView:
    """The harness layers visible to this agent, kept on the engine's machine: each
    call runs on them through the broker, with ``HarnessView``'s signature, behaviour
    and errors."""

    def __getattr__(self, name: str) -> Any:
        if name in _STORE_ATTRIBUTES:
            raise AttributeError(
                f"HarnessView.{name} holds the engine's stores, which split execution "
                "keeps out of the kernel; use the view's read and write methods"
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


def _remote_method(name: str):
    @functools.wraps(getattr(HarnessView, name))
    def call(self: RemoteHarnessView, *args: Any, **kwargs: Any) -> Any:
        response = broker.request_sync(
            "harness.call", method=name, args=list(args), kwargs=kwargs
        )
        if "raised" in response:
            raised = response["raised"]
            error = {cls.__name__: cls for cls in _RAISED}[raised["type"]]
            raise error(raised["message"])
        return decode(response["value"])

    return call


for _name in METHODS:
    setattr(RemoteHarnessView, _name, _remote_method(_name))
