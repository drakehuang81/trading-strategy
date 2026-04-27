"""Pre-Live Gate CLI — spec §10.

Runs all 8 gates against current SQLite + filesystem state.
Exits 0 if every gate passes, 1 if any fails.

Usage:
    python -m scripts.pre_live_gate \
        --sqlite-path data/state.db \
        --model-dir models \
        --watchdog-log data/watchdog_pings.log \
        --brier-threshold 0.24
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from execution.pre_live_gate import (
    GateContext,
    NoRepaintingGate,
    BacktestDSRGate,
    CalibrationBrierGate,
    PaperRuntimeGate,
    ReconciliationGate,
    DriftStabilityGate,
    WatchdogUptimeGate,
    HaltDiversityGate,
    run_all_gates,
)


_ALL_GATES = [
    NoRepaintingGate(),
    BacktestDSRGate(),
    CalibrationBrierGate(),
    PaperRuntimeGate(),
    ReconciliationGate(),
    DriftStabilityGate(),
    WatchdogUptimeGate(),
    HaltDiversityGate(),
]


async def main_async(args: argparse.Namespace) -> int:
    ctx = GateContext(
        sqlite_path=args.sqlite_path,
        brier_threshold=args.brier_threshold,
        watchdog_log_path=args.watchdog_log,
        model_dir=args.model_dir,
    )
    results = run_all_gates(_ALL_GATES, ctx)
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    print(f"\nPre-Live Gate: {n_pass}/{n_total} passed\n")
    print(f"{'GATE':<22} {'PASS':<6} REASON")
    print("-" * 80)
    for r in results:
        flag = "✅" if r.passed else "❌"
        print(f"{r.name:<22} {flag:<6} {r.reason}")
    print()
    failed = [r.name for r in results if not r.passed]
    if failed:
        print(f"FAILED gates: {failed}", file=sys.stderr)
        return 1
    print("All gates green — live mode is allowed.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite-path", default="data/state.db")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--watchdog-log", default="data/watchdog_pings.log")
    ap.add_argument("--brier-threshold", type=float, default=0.24)
    args = ap.parse_args()
    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
