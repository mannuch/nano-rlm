"""Peer roots and agent-to-agent messaging across the session's mesh of trees."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from pathlib import Path

from rlm import broker
from rlm.agent import AgentInfo
from rlm.history import History, history
from rlm.types import AgentResult


@dataclass(frozen=True)
class PeerHandle:
    """Any agent in the session: a peer, the root, or another tree's agent by id."""

    id: str
    session_dir: Path
    name: str | None = None

    async def info(self) -> AgentInfo:
        """Read current metadata from the supervisor."""
        return AgentInfo.from_payload(
            await broker.agent_request("peer.info", agent_id=self.id)
        )

    async def history(self) -> History:
        """Read a fresh snapshot of this agent's conversation history."""
        return await history(session_dir=self.session_dir)

    async def send(
        self, message: str, *, in_reply_to: str | None = None, wake: bool = True
    ) -> str:
        """Put a peer.message in this agent's inbox and return its message ID.

        ``in_reply_to`` names the message you are answering. ``wake=False`` leaves an
        idle receiver asleep: it sees the message at its next turn. Use it for FYIs and
        broadcasts, since every wake costs the receiver a model turn.
        """
        return await broker.agent_request(
            "peer.send",
            agent_id=self.id,
            message=message,
            in_reply_to=in_reply_to,
            wake=wake,
        )

    async def result(self, *, yield_after: float = 300) -> AgentResult:
        """Wait up to yield_after seconds for this agent's current turn, then return its
        latest answer. Terminal failure/cancellation raises."""
        payload = await broker.agent_request(
            "peer.result", agent_id=self.id, yield_after=yield_after
        )
        return broker.result_from_payload(payload)

    async def wait(self, timeout: float = 30) -> AgentInfo:
        """Wait up to timeout seconds for this agent to finish its turn, then return
        current metadata. Prefer the native wait tool, which holds no cell."""
        return AgentInfo.from_payload(
            await broker.agent_request("peer.wait", agent_id=self.id, timeout=timeout)
        )

    async def cancel(self) -> AgentInfo:
        """Terminate this peer and its subtree. Only its owner or the root may."""
        return AgentInfo.from_payload(
            await broker.agent_request("peer.cancel", agent_id=self.id)
        )


def _handle(info: AgentInfo) -> PeerHandle:
    return PeerHandle(info.id, info.session_dir, info.name)


async def spawn(task: str, *, name: str) -> PeerHandle:
    """Start a peer root of its own tree and return its handle before it answers.

    ``name`` is mesh-unique and should say what the peer is for, so others can find it.
    A peer is persistent, outlives you, and has the full recursion depth for its own
    subtree; you become its owner and receive its peer.completed events.
    """
    info = AgentInfo.from_payload(
        await broker.agent_request("peer.spawn", task=task, name=name)
    )
    return _handle(info)


async def get(name_or_id: str) -> PeerHandle:
    """A handle to a peer by name (``root`` for the root) or to any agent by id."""
    info = AgentInfo.from_payload(
        await broker.agent_request("peer.get", name_or_id=name_or_id)
    )
    return _handle(info)


async def list(*, all: bool = False) -> builtins.list[AgentInfo]:
    """The root and every peer, with their names, tasks, owners and status. ``all=True``
    adds every other agent in the session, finished ones included."""
    payload = await broker.agent_request("peer.list", all=all)
    return [AgentInfo.from_payload(item) for item in payload]
