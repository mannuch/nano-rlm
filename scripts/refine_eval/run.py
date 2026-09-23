"""Run the refinement eval: every scenario x arm x repeat is one live session that is
steered into a situation, refined through the host path, optionally probed, and graded.

    uv run python scripts/refine_eval/run.py --model <model> --run-dir <dir>

Credentials come from the environment: RLM_API_KEY (or OPENAI_API_KEY) and
RLM_BASE_URL for the task model, TYPESAFE_API_KEY for the judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm.config import (
    ExecutionPolicy,
    HarnessConfig,
    InvocationContext,
    ProviderConfig,
    RefineJudgeConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.harness import HarnessStore, local_dir
from rlm.history import read_records
from rlm.session import Session

from grade import grade_case, report
from scenarios import BY_ID, SCENARIOS, Scenario

ARMS: dict[str, dict[str, Any] | None] = {
    "none": None,
    "force": {},
    "force+focus": {"focus": True},
    "typesafe": {"review": True},
}
"""Host refine requests. In every refining arm the planner may still decline with no
edits; ``typesafe`` lets the judge decline first. ``none`` is a probe control."""
SWEEP = [round(0.3 + 0.1 * i, 2) for i in range(7)]


@dataclass
class Settings:
    model: str
    api_key: str
    base_url: str | None
    typesafe_api_key: str
    threshold: float = 0.7
    max_depth: int = 1
    max_total_tokens: int = 300_000
    exec_timeout: int = 60
    timeout_s: float = 600.0


def _config(settings: Settings, global_dir: Path) -> RuntimeConfig:
    return RuntimeConfig(
        model=settings.model,
        provider=ProviderConfig(base_url=settings.base_url, api_key=settings.api_key),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(
            max_depth=settings.max_depth,
            max_total_tokens=settings.max_total_tokens,
            exec_timeout=settings.exec_timeout,
        ),
        harness=HarnessConfig(
            global_dir=str(global_dir),
            refine_judge=RefineJudgeConfig(
                api_key=settings.typesafe_api_key, threshold=settings.threshold
            ),
        ),
    )


async def run_case(
    scenario: Scenario, arm: str, repeat: int, settings: Settings, root: Path
) -> dict[str, Any]:
    """One graded case; failures are recorded on the row, never raised."""
    case = f"{scenario.id}.{arm}.{repeat}"
    workspace = root / "workspaces" / case
    session_dir = root / "sessions" / case
    global_dir = root / "global" / case
    scenario.build(workspace)
    stores = {
        "local": HarnessStore(local_dir(session_dir)),
        "global": HarnessStore(global_dir, scope="global"),
    }
    for seed in scenario.seed:
        stores[seed.scope].create(
            seed.kind, seed.title, seed.content, id=seed.id, source="eval"
        )

    engine: RLMEngine | None = None
    probe_answer = None
    error = None
    try:
        engine = RLMEngine(
            cwd=str(workspace),
            session=Session(session_dir),
            runtime_config=_config(settings, global_dir),
        )
        for text in scenario.steer:
            await asyncio.wait_for(engine.prompt(text), timeout=settings.timeout_s)
        request = ARMS[arm]
        if request is not None:
            await asyncio.wait_for(
                engine.prompt(
                    "",
                    refine={
                        "instructions": None,
                        "global_": scenario.scope == "global",
                        "rollback_id": None,
                        **request,
                    },
                ),
                timeout=settings.timeout_s,
            )
        if scenario.probe is not None:
            result = await asyncio.wait_for(
                engine.prompt(scenario.probe.prompt), timeout=settings.timeout_s
            )
            probe_answer = result.answer
    except Exception as exc:  # noqa: BLE001 - one failed case must not end the run
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
    finally:
        if engine is not None:
            try:
                await engine.aclose()
            except Exception as exc:  # noqa: BLE001 - keep the case's graded row
                error = (error or "") + f"\nclose failed: {exc!r}"

    records = list(read_records(session_dir / "messages.jsonl"))
    final = {
        scope: [e.model_dump() for e in store.list()] for scope, store in stores.items()
    }
    row = grade_case(scenario, arm, records, final, probe_answer, settings.threshold)
    row.update(
        repeat=repeat,
        session_dir=str(session_dir),
        tokens=(
            engine._total_usage.prompt_tokens + engine._total_usage.completion_tokens
        )
        if engine is not None
        else None,
        error=error,
    )
    return row


async def run_all(
    scenarios: list[Scenario],
    arms: list[str],
    repeats: int,
    settings: Settings,
    root: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    gate = asyncio.Semaphore(concurrency)
    results = root / "results.jsonl"

    async def one(scenario: Scenario, arm: str, repeat: int) -> dict[str, Any]:
        async with gate:
            row = await run_case(scenario, arm, repeat, settings, root)
        with results.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        status = "error" if row["error"] else f"decision={row['decision']}"
        print(f"{scenario.id} {arm} #{repeat}: {status}", flush=True)
        return row

    return await asyncio.gather(
        *(
            one(scenario, arm, repeat)
            for scenario in scenarios
            for arm in arms
            if arm != "none" or scenario.probe is not None
            for repeat in range(repeats)
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="Task model; required unless --report-only")
    parser.add_argument("--base-url", default=os.environ.get("RLM_BASE_URL"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--scenarios",
        default=",".join(s.id for s in SCENARIOS),
        help=f"Comma-separated scenario ids: {', '.join(BY_ID)}",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"Comma-separated arms: {', '.join(ARMS)}",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="Seconds per prompt"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild report.md from an existing results.jsonl",
    )
    args = parser.parse_args(argv)

    root = Path(args.run_dir)
    if args.report_only:
        rows = [json.loads(line) for line in (root / "results.jsonl").open()]
        (root / "report.md").write_text(report(rows, SWEEP))
        print(root / "report.md")
        return 0

    if not args.model:
        parser.error("--model is required unless --report-only")
    api_key = os.environ.get("RLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    typesafe_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        parser.error("set RLM_API_KEY or OPENAI_API_KEY")
    if not typesafe_key:
        parser.error("set TYPESAFE_API_KEY")
    ids = [i.strip() for i in args.scenarios.split(",") if i.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [i for i in ids if i not in BY_ID] + [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown scenarios or arms: {unknown}")
    if root.exists() and any(root.iterdir()):
        parser.error(f"{root} is not empty")
    root.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        typesafe_api_key=typesafe_key,
        threshold=args.threshold,
        timeout_s=args.timeout,
    )
    rows = asyncio.run(
        run_all(
            [BY_ID[i] for i in ids],
            arms,
            args.repeats,
            settings,
            root,
            args.concurrency,
        )
    )
    (root / "report.md").write_text(report(rows, SWEEP))
    print(root / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
