# Plan 5B-2 STATUS — ReplayBroker + Broker Contract

**Date**: 2026-04-26
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: `fd02eb7` (Plan 5B-2 plan doc)
**Head commit**: `0d1163b` (feat(wiring): broker_kind selects paper/replay/live broker at boot)

## Summary

`PaperBroker`'s slippage + fee math extracted to `src/execution/cost_model.py` (3 pure functions). `ReplayBroker` (deterministic, external clock) and `LiveBroker` (refuses orders) added; both share the cost model. `tests/contracts/test_broker_contract.py` now runs against PaperBroker AND ReplayBroker. `OrchestratorConfig.broker_kind` selects which broker the wiring constructs at boot.

Test count: **320 passed** (Plan 5B-1 baseline 296 + 24 new across 5 tasks).

## Task table

| # | Title | Commits | Files |
|---|-------|---------|-------|
| 1 | Extract `cost_model.py`; PaperBroker refactor | `a316c1f`, `673a941` | `cost_model.py`, `paper_broker.py`, test |
| 2 | `ReplayBroker` | `fd951eb`, `d8d32e5` | `replay_broker.py`, test |
| 3 | `LiveBroker` stub | `405d5ae` | `live_broker.py`, test |
| 4 | Broker contract: ReplayBroker added | `963be6b` | `tests/contracts/test_broker_contract.py` |
| 5 | Wiring `broker_kind` switch + STATUS | `0d1163b` (this commit) | `orchestrator.py`, `wiring.py`, test |

## Decisions landed

- **Cost model is shared** (`execution.cost_model`). Backtest fills now provably match paper fills bit-for-bit (spec §7.2).
- **`PaperBrokerConfig` composes `SlippageConfig`** (Task 1 fix `673a941`) — single source of truth, no field duplication.
- **ReplayBroker has zero realism noise** — no latency, no partial fill, no rejection. Determinism over realism for backtest.
- **External clock**: `ReplayBroker.set_time(ts)` is the only mutator of internal time. The Plan 5B-3 backtest harness will own the loop.
- **ReplayBroker is single-symbol by construction** (Task 2 fix `d8d32e5`) — `symbol` field + `submit` guard. Multi-symbol replay deferred.
- **Cross-zero entry basis reset** (Task 2 fix `d8d32e5`) — when a fill flips position direction, new avg_entry is the cross-zero fill price.
- **Funding mark price uses funding-tick close** (Task 2 fix `d8d32e5`) — not `_current_ts` close, which would use stale prices.
- **LiveBroker raises `LiveBrokerNotImplemented`** on submit/cancel; positions/balance return empty/zero so wiring boot doesn't crash. Plan 5D replaces.

## Bugs caught by review

1. **Cross-zero entry basis** — Critical bug in initial ReplayBroker: long → larger sell flip kept the long's avg_entry. Code reviewer caught it. Fixed with explicit branch + test.
2. **Funding stale mark price** — Critical bug: `_charge_funding` used `_current_ts` (the OLD time at the moment of the call inside `set_time`) instead of the funding-tick time. Would silently corrupt PnL on multi-bar advances. Fixed with `_close_at(ts)` helper + test.
3. **Symbol arg ignored in `_current_close`** — Important: BTCUSDT order silently filled at ETH price. Fixed by storing `self.symbol` + guard in `submit`.
4. **Config field duplication** (Task 1) — `PaperBrokerConfig` and `SlippageConfig` both had the 3 slippage params. Fixed via composition.

All bugs were caught BEFORE the wiring switch landed. Backtest harness (Plan 5B-3) will inherit a correct broker.

## What is NOT done (Plan 5B-3+ scope)

- **Plan 5B-3**: Walk-forward backtest harness that drives `ReplayBroker.set_time` over historical klines + records `backtest_runs` rows + computes Deflated Sharpe.
- **Plan 5B-4**: Pre-Live Gate module (§10 8 gates).
- **Plan 5D**: Replace `LiveBroker` stub with real Binance live integration.

## Known follow-ups

- **`ReplayBroker` is single-symbol** by construction. Multi-symbol replay would require `klines: dict[symbol, df]` and per-symbol `_current_close`. Defer.
- **`LiveBroker.events()` never yields** — caller must handle the empty-async-iterator case. Plan 5D will replace with real stream.
- **`replay_funding_path` defaults to `data/funding/ETHUSDT.parquet`** — Plan 5B-1's backfilled parquet. After Plan 5C (multi-symbol), this needs templating per symbol.
