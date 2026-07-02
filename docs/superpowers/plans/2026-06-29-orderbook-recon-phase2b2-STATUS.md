# Phase 2b-2 STATUS — BTC→ETH Cross-Asset Lead-Lag: NEGATIVE RESULT (2026-06-29)

**Branch**: `worktree-recon-phase2b1` · **Tests**: 37 passed · **Gate**: full window, 0 days skipped.

## The gate result (BTC book → ETH 1h, 2023-05-16 → 2024-03-30, 7,353 hours)

```
ic_all:              0.042
ic_train / ic_test:  0.048 / 0.040   → fails |ic_test| > 0.1
ic_momentum (ETH):  -0.067  |  ic_momentum (BTC): -0.081
                     → BOTH controls carry MORE information than the signal
gross_bps:           0.90   → net after 8 bps taker: -7.10
nw_tstat:           -0.65
monotone:            False
verdict:             FAILED — OOS, vs-controls, post-cost, monotone
```

## Interpretation

BTC's book tells you essentially the same (near-nothing) about ETH's next hour as ETH's own book did in Phase 2b-1 (IC 0.042 vs 0.046) — unsurprising, the two books are highly correlated. Even BTC's own **price momentum** (|IC| 0.081) carries more information about ETH's next hour than BTC's book does, and that momentum itself is far below tradeable. The dual-control design did its job: there is no order-book information here beyond what price already shows, and price itself shows ~nothing at 1h.

(Side observation, not actionable: both momentum ICs are mildly **negative** — weak 1h mean-reversion in this window — but |0.07| is nowhere near the 0.1 OOS bar, let alone post-cost viability.)

## The recon program is now COMPLETE — final scoreboard

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | TA features → ML direction (Plan 5A→5E) | dead (Brier ≥ coin-flip baseline) |
| 2 | BTC/ETH pair trade (mean reversion) | dead (ratio trends; half-life 151d) |
| 3 | Low-vol breakout | dead (OOS collapse) |
| 4 | L1 queue imbalance @ seconds | real IC (~0.37) but **taker-fee-dead**; maker-only |
| 5 | depth_imbalance @ 15m–1h (own book) | dead (full-window: regime artifact) |
| 6 | BTC book → ETH (cross-asset) | **dead (this plan)** |

Every free-public-data directional hypothesis on ETH has now been tested against pre-committed gates and refuted. This is a clean, honest conclusion, not a failure of process — six negative results, each documented, none re-runnable by accident.

## What the project owns now

- A **generic signal-validation harness** (`depth_study` + `depth_validation` + `cross_validation`): any (signal × symbol × window) → four-gate verdict in one command. 37 tests.
- Full order-book data pipeline (bookTicker/bookDepth/aggTrades/klines loaders, header-tolerant), ~320 days × 2 symbols cached locally.
- The production 1h assistant infrastructure (Plan 1–5E): broker, backtest, Pre-Live Gate, Telegram — intact, waiting for a model with real edge.

## Remaining paths (all structurally different — strategic choice, not another quick test)

1. **qi maker/HF market-making** — the one signal with genuine information content; requires maker-fill simulation, queue-position modeling, inventory risk. A separate project.
2. **Different market** — same harness, venue where fees/volatility ratio is friendlier or signals less arbitraged.
3. **Stop here** — run the 1h assistant as a paper-trading learning system on infrastructure that is now thoroughly battle-tested.

## Commits (2b-2, this branch)

`9499d71` plan · `96492f5` build_hourly_cross · `cabe7f0` lead-momentum control from complete klines · `8d3a204` cross driver (dual controls) · this STATUS.
