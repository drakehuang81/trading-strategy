# Plan 5B-4 STATUS — Pre-Live Gate

**Date**: 2026-04-27
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: `a879cfd` (Plan 5B-4 doc commit)
**Head commit**: `74c7ef0` (CLI + STATUS)

## Summary

`src/execution/pre_live_gate.py` ships all 8 gates from spec §10 (correctness 1-3, operations 4-8). `scripts/pre_live_gate.py` CLI runs all gates and exits 0/1. `scripts/heartbeat_watchdog.py` now appends each ping to `data/watchdog_pings.log` so gate 7 has evidence. The CLI is the operational gatekeeper that lets the orchestrator's `live` mode wiring (Plan 5D) refuse to boot when any gate is red.

Test count: **359 passed** (Plan 5B-3 baseline 333 + 26 new across 5 tasks).

## Task table

| # | Title | Commit | Files |
|---|-------|--------|-------|
| 1 | Foundation (GateResult, GateContext, Protocol, driver) | `9871931` | `pre_live_gate.py`, test |
| 2 | Correctness gates 1-3 | `29ce4bb` | `pre_live_gate.py`, test |
| 3 | Operations gates 4-6 | `65cf983` | `pre_live_gate.py`, test |
| 4 | Operations gates 7-8 + watchdog log | `556ad17` | `pre_live_gate.py`, `heartbeat_watchdog.py`, test |
| 5 | CLI + STATUS | `74c7ef0` | `scripts/pre_live_gate.py`, test |

## Manual smoke results

<!-- TODO controller: paste actual `python -m scripts.pre_live_gate ...` output + per-gate analysis -->

## Decisions landed

- **Gate 1 (no_repainting) shells out to pytest** — pytest is source of truth; ~5s overhead acceptable.
- **Gate 2 (DSR) reads `backtest_runs.deflated_sharpe` directly** — assumes Plan 5B-3 Task 1's probability-form storage.
- **Gate 3 (Brier) reads `models/meta_*.json`** by mtime; chosen calibrator's Brier value compared to `brier_threshold` (default 0.24).
- **Gate 6 (drift) uses HALT proxy** — no `drift_state_history` table exists; absence of feature_drift HALT in last 30d is the operational evidence.
- **Gate 7 (watchdog) uses new `data/watchdog_pings.log`** — append-only timestamp log written by `heartbeat_watchdog.py` on each run.
- **Gate 4 (paper runtime) tolerance**: span check has tolerance `max_gap_minutes / (24*60)` days to handle integer-row arithmetic at 5-min intervals.
- **Gate 4 fills query uses `>= min_ts`**, not `BETWEEN min_ts AND max_ts`. Reason: avoids skew between two `datetime.now()` calls in test seeders. Mild spec deviation; flagged as follow-up.
- **All gates run; no short-circuit** — operators see every red at once.
- **CLI uses `python -m scripts.pre_live_gate`** invocation (not `python scripts/pre_live_gate.py`) to avoid the `scripts/backtest.py` ↔ `src/backtest/` shadowing pattern (Plan 5B-3 footgun applied here too — `scripts/pre_live_gate.py` shadows `src/execution/pre_live_gate.py`'s package access if invoked directly).

## What is NOT done (Plan 5D scope)

- **Live-mode boot guard wiring**: `src/wiring.py`'s `live` branch should call `pre_live_gate` and refuse to boot if any gate is red. Not in this plan.
- **`drift_state_history` table** for stronger gate 6 evidence.
- **`make pre-live-check` Make target** (spec §9.7).
- **Gate 5 dust-threshold semantic**: today treats "halted" as bad and "auto_repaired" as good. The spec's "above dust threshold" semantic is not implemented as a JSON parse — assumed encoded in `resolution` value.
- **Gate 7 watchdog log rotation** — log grows unbounded; needs rotation policy.

## Known follow-ups

- **Gate 4 fills query upper bound**: tighten to `BETWEEN min_ts AND max_ts` once test seeders share a single `now` anchor.
- **Gate 6 drift evidence**: add a `drift_state_history` table that the drift monitor writes to on every check; gate then queries that table for "30 consecutive green" evidence rather than "0 drift halts" proxy.
- **`script.pre_live_gate` ↔ `execution.pre_live_gate` shadowing**: rename script to `scripts/check_pre_live.py` to avoid confusion (same pattern as `scripts/backtest.py` ↔ `src/backtest/`).
- **Brier threshold default 0.24**: arbitrary. Plan 5C should derive threshold from baseline-comparison or Pre-Live Gate's actual cost analysis.
