"""Read-only snapshots of a session's message ledger and context windows."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


_BLOCK_FIELDS = (
    "tier",
    "branches",
    "messages",
    "windows",
    "turns",
    "summary",
    "request_id",
)


def read_records(path: Path) -> Iterator[dict]:
    with path.open("rb") as stream:
        for line in stream:
            # A live writer may not have finished its final record yet.
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


@dataclass
class ContextWindow:
    index: int
    reason: str
    message_indices: list[int]
    _messages: list[dict] = field(repr=False)

    @property
    def messages(self) -> list[dict]:
        return [self._messages[index] for index in self.message_indices]


class History:
    """A point-in-time snapshot. Read again to observe subsequent live activity."""

    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir).resolve()
        self.events = list(read_records(self.session_dir / "messages.jsonl"))
        self.messages: list[dict] = []
        self.windows: list[ContextWindow] = []
        for event in self.events:
            if "message_index" in event:
                index = event["message_index"]
                if index != len(self.messages):
                    raise ValueError("non-contiguous message indices in ledger")
                self.messages.append(event["message"])
                if "window" in event:
                    self.windows[event["window"]].message_indices.append(index)
            if event["type"] == "context_window":
                if event["window"] != len(self.windows):
                    raise ValueError("non-contiguous context windows in ledger")
                indices = event["message_indices"]
                if any(index < 0 or index >= len(self.messages) for index in indices):
                    raise ValueError("context window references an unknown message")
                self.windows.append(
                    ContextWindow(
                        event["window"], event["reason"], list(indices), self.messages
                    )
                )

    @property
    def blocks(self) -> list[dict]:
        """Compaction staircase blocks in ledger order, excluding those of rolled-back
        prompt attempts: each names the ``tier`` and the ``branches``, ``messages``,
        ``windows``, and ``turns`` ranges its ``summary`` covers."""
        blocks: list[tuple[int, dict]] = []
        positions: dict[str, int] = {}
        for position, event in enumerate(self.events):
            positions.setdefault(event["id"], position)
            if event["type"] == "compaction" and "block" in event:
                blocks.append((position, event["block"]))
            elif event["type"] == "rollup":
                blocks.append((position, {key: event[key] for key in _BLOCK_FIELDS}))
            elif event["type"] == "prompt_rollback":
                attempt = positions.get(event["prompt_id"], position)
                blocks = [item for item in blocks if item[0] < attempt]
        return [block for _, block in blocks]

    def expand(self, block: dict | int) -> list[dict]:
        """The ledger messages a compaction block summarizes: pass a record from
        ``blocks`` or its position in that list."""
        if isinstance(block, int):
            block = self.blocks[block]
        first, last = block["messages"]
        return self.messages[first : last + 1]

    def user_messages(self) -> list[dict]:
        """Original user inputs, including attempts identified by rollback events."""
        return [
            self.messages[event["message_index"]]
            for event in self.events
            if event["type"] == "user" and "message_index" in event
        ]


async def history(session_dir: str | Path | None = None) -> History:
    """Read a session's ledger; defaults to this kernel's RLM session directory.
    Async like every other rlm call, though it reads the local ledger without the broker."""
    if session_dir is None:
        session_dir = os.environ.get("RLM_SESSION_DIR")
        if not session_dir:
            raise RuntimeError("pass a session directory when outside an RLM kernel")
    return History(session_dir)
