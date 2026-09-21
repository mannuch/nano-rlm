"""CLI entry point: an Agent Client Protocol agent over stdio, plus operator
utilities for a harness store that never run inside a session."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rlm",
        description="A minimalistic recursive agent, served over the Agent Client Protocol.",
    )
    parser.add_argument(
        "--acp",
        action="store_true",
        help="Serve as an Agent Client Protocol agent over stdio",
    )
    commands = parser.add_subparsers(dest="command")
    harness = commands.add_parser(
        "harness", help="Operator maintenance of a harness store"
    ).add_subparsers(dest="harness_command", required=True)
    prune = harness.add_parser(
        "prune",
        help="Remove old episode records from a global harness store",
        description=(
            "Select episodes by age and/or count and print what would be removed; "
            "pass --apply to remove them. The prune is recorded in the store's "
            "refinements with a snapshot of every removed entry."
        ),
    )
    prune.add_argument(
        "--global-dir",
        default=os.environ.get("RLM_HARNESS_GLOBAL_DIR") or None,
        help="The global harness store (default: $RLM_HARNESS_GLOBAL_DIR)",
    )
    prune.add_argument(
        "--older-than",
        metavar="DURATION",
        help="Remove episodes whose session started more than this long ago (30d, 12h, 4w)",
    )
    prune.add_argument(
        "--keep", type=int, metavar="N", help="Keep only the N newest episodes"
    )
    prune.add_argument(
        "--apply", action="store_true", help="Remove the selected episodes"
    )
    args = parser.parse_args(argv)

    if args.command == "harness":
        return _prune(prune, args)
    if not args.acp:
        parser.error("rlm runs as an ACP agent: use `rlm --acp` (or `rlm harness ...`)")
    from rlm.acp import serve_acp

    asyncio.run(serve_acp())
    return 0


def _prune(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from rlm.harness import HarnessStore
    from rlm.harness_ops import parse_duration, prune_episodes, select_prunable

    if not args.global_dir:
        parser.error("--global-dir is required (or set RLM_HARNESS_GLOBAL_DIR)")
    if args.older_than is None and args.keep is None:
        parser.error("pass --older-than and/or --keep to select episodes")
    if args.keep is not None and args.keep < 0:
        parser.error("--keep must be zero or more")
    older_than = parse_duration(args.older_than) if args.older_than else None
    store = HarnessStore(args.global_dir, scope="global").load()
    total = store.count("episode")
    selected = select_prunable(store, older_than=older_than, keep=args.keep)
    verb = "Removing" if args.apply else "Would remove"
    print(f"{verb} {len(selected)} of {total} episode(s) from {store.path}")
    for entry in selected:
        stop = entry.metadata.get("stop_reason", "?")
        print(f"  {entry.id}  {entry.path}  {entry.title!r}  ({stop})")
    if not selected:
        return 0
    if not args.apply:
        print("Dry run: pass --apply to remove them.")
        return 0
    result = prune_episodes(store, selected)
    print(
        f"Recorded as refinement {result.id}; {total - len(selected)} episode(s) remain."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
