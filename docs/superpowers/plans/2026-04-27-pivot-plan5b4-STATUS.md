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

`python -m scripts.pre_live_gate --sqlite-path data/state.db --model-dir models --watchdog-log data/watchdog_pings.log --brier-threshold 0.24` exits **1** with **4/8 passed**:

```
GATE                   PASS   REASON
no_repainting          ✅      all repainting tests passed
backtest_dsr           ✅      DSR 0.5755528922172057 > 0.5
calibration_brier      ❌      chosen calibrator platt Brier 0.2505 >= threshold 0.24
paper_runtime          ❌      no heartbeat rows
reconciliation         ✅      0 unresolved diffs in last 14d
drift_stability        ✅      0 drift HALTs in last 30d
watchdog_uptime        ❌      watchdog log not found at data/watchdog_pings.log
halt_diversity         ❌      missing trigger_source(s): ['daily_loss_kill_switch', 'feature_drift', 'broker_desync']

FAILED gates: ['calibration_brier', 'paper_runtime', 'watchdog_uptime', 'halt_diversity']
```

Per-gate analysis:

- **✅ no_repainting** — Real signal. Pytest no_repainting suite passes (Plan 1's invariant holds).
- **✅ backtest_dsr** — Real signal. Reads Plan 5B-3's `0a5e78d23a22` row (DSR 0.576). Note: this run used loose `long_threshold=0.51`; production runs must use `>=0.58`.
- **❌ calibration_brier** — **Real fail**. Model `92fddb72f14b` has Platt Brier 0.2505, threshold 0.24. Confirms Plan 5B-1/5B-3 conclusion: model has near-zero alpha. Plan 5C model improvements (gap=horizon, NaN handling, triple-barrier labels) needed before this passes.
- **❌ paper_runtime** — Real fail. Orchestrator has never been left running; 0 heartbeat rows. Need 60 days continuous paper mode operation.
- **✅ reconciliation** — **Vacuously true** (no diffs because no paper trading). False positive in spirit; will become real signal once paper trading runs and reconciliation actually executes.
- **✅ drift_stability** — **Vacuously true** (no drift HALTs because no paper trading). Same false-positive pattern as reconciliation. Plan 5C should add `drift_state_history` table for explicit positive evidence rather than negative proxy.
- **❌ watchdog_uptime** — Real fail. `data/watchdog_pings.log` doesn't exist; watchdog has never been deployed (launchd/cron not configured).
- **❌ halt_diversity** — Real fail. 0 halt_events; spec requires at least one row per `daily_loss_kill_switch` / `feature_drift` / `broker_desync` plus at least one followed by `/resume`. Need fire-drill (manually trigger each HALT type once).

**Operational meaning**: today's state would be blocked from live mode by 4 gates. Even after Plan 5C improves the model (likely fixes `calibration_brier`), 3 ops gates require actual deployment time:
1. Deploy watchdog via launchd/cron → wait 7 days
2. Run orchestrator paper-mode for 60 consecutive days with ≥30 fills
3. Manually trigger each HALT trigger family once (fire drill)

Earliest possible green: **~10 weeks from when watchdog + paper mode start running**.

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
