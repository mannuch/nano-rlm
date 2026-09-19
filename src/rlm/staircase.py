"""Tiered compaction blocks.

A branch is the context window between two compactions. Its in-context summary is a
tier-1 block; every ``fanout`` consecutive blocks of one tier roll up into one block of
the next tier. The staircase rendered into a compacted window lists blocks oldest first
at decreasing resolution and always reaches back to the first message, so the summary
of a long session is bounded by ``(fanout - 1) * log(branches)`` blocks. Blocks are
immutable and keyed by ledger message ranges, so the raw ledger stays the source of
truth and every block is a pointer into it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    tier: int
    branches: tuple[int, int]
    """Half-open range of branch indices."""
    messages: tuple[int, int]
    """Inclusive range of ledger message indices."""
    windows: tuple[int, int]
    """Inclusive range of context window indices."""
    turns: tuple[int, int]
    """Inclusive range of engine turns."""
    summary: str
    request_id: str

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.tier, *self.branches)

    def header(self) -> str:
        start, end = self.branches
        span = f"branch {start}" if end == start + 1 else f"branches {start}-{end - 1}"
        return (
            f"[tier {self.tier} | {span} | {_range('window', self.windows)}"
            f" | {_range('message', self.messages)} | {_range('turn', self.turns)}]"
        )

    def to_record(self) -> dict:
        return {
            "tier": self.tier,
            "branches": list(self.branches),
            "messages": list(self.messages),
            "windows": list(self.windows),
            "turns": list(self.turns),
            "summary": self.summary,
            "request_id": self.request_id,
        }


def _range(noun: str, span: tuple[int, int]) -> str:
    first, last = span
    return f"{noun} {first}" if first == last else f"{noun}s {first}-{last}"


class Staircase:
    def __init__(self, fanout: int):
        if fanout < 2:
            raise ValueError("fanout must be at least 2")
        self.fanout = fanout
        self.blocks: dict[tuple[int, int, int], Block] = {}
        self.branch_count = 0

    def snapshot(self) -> tuple[dict[tuple[int, int, int], Block], int]:
        return dict(self.blocks), self.branch_count

    def restore(self, state: tuple[dict[tuple[int, int, int], Block], int]) -> None:
        self.blocks, self.branch_count = dict(state[0]), state[1]

    def add_branch(self, block: Block) -> None:
        if block.tier != 1 or block.branches != (
            self.branch_count,
            self.branch_count + 1,
        ):
            raise ValueError("a branch block must be the next tier-1 block")
        self.blocks[block.key] = block
        self.branch_count += 1

    def children(self, tier: int, start: int, end: int) -> list[Block]:
        size = self.fanout ** (tier - 2)
        return [self.blocks[(tier - 1, c, c + size)] for c in range(start, end, size)]

    def unsealed(self) -> list[tuple[int, int, int]]:
        """Aligned ranges whose children all exist but which have no block yet,
        finest tier first."""
        ranges = []
        tier, size = 2, self.fanout
        while size <= self.branch_count:
            child = size // self.fanout
            for start in range(0, self.branch_count - size + 1, size):
                key = (tier, start, start + size)
                if key in self.blocks:
                    continue
                if all(
                    (tier - 1, c, c + child) in self.blocks
                    for c in range(start, start + size, child)
                ):
                    ranges.append(key)
            tier, size = tier + 1, size * self.fanout
        return ranges

    def seal(self, block: Block) -> None:
        if block.tier < 2 or block.key in self.blocks:
            raise ValueError("a rollup seals one unsealed range of tier 2 or higher")
        self.blocks[block.key] = block

    def segments(self) -> list[Block]:
        """The blocks to show, coarsest first and chronological within a tier.

        The branch count is decomposed in base ``fanout``: tier k contributes the
        aligned blocks just older than the finer tiers, at most ``fanout - 1`` of them.
        A range whose rollup is missing is shown through its children instead, so
        coverage is never lost to a failed rollup.
        """
        by_tier: list[list[tuple[int, int, int]]] = []
        remaining, tier, size = self.branch_count, 1, 1
        while remaining > 0:
            coarser = size * self.fanout
            chunk = remaining % coarser
            by_tier.append(
                [(tier, x, x + size) for x in range(remaining - chunk, remaining, size)]
            )
            remaining -= chunk
            tier, size = tier + 1, coarser
        shown: list[Block] = []
        for ranges in reversed(by_tier):
            for key in ranges:
                self._emit(key, shown)
        return shown

    def _emit(self, key: tuple[int, int, int], shown: list[Block]) -> None:
        block = self.blocks.get(key)
        if block is not None:
            shown.append(block)
            return
        tier, start, end = key
        child = self.fanout ** (tier - 2)
        for c in range(start, end, child):
            self._emit((tier - 1, c, c + child), shown)

    def render(self) -> str:
        return "\n\n".join(
            f"{block.header()}\n{block.summary}" for block in self.segments()
        )
