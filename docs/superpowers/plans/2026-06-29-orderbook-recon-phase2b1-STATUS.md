# Phase 2b-1 STATUS — Depth @ 1h Validation: NEGATIVE RESULT (2026-06-29)

**Branch**: `worktree-recon-phase2b1` · **Tests**: 35 passed · **Gate**: run on full window, verdict below.

## The gate result (ETHUSDT, 2023-05-16 → 2024-03-30, 7,353 hours, 0 days skipped)

```
ic_all:             0.046      (single-day 2023-06-01 was 0.497)
ic_train / ic_test: 0.044 / 0.036   → fails |ic_test| > 0.1
ic_momentum:       -0.067      → |momentum| > |depth| — depth does NOT beat the trend control
gross_bps:          0.86       → net after 8 bps taker: -7.14
nw_tstat:          -0.47       (median-split strategy return ≈ statistically zero)
monotone:           False
verdict:            FAILED — OOS, vs-momentum, post-cost, monotone
```

**All four pre-committed gates failed.** Not marginal — decisive.

## Interpretation

The single-day finding (IC 0.50, +14 bps net @ 1h on 2023-06-01) was **exactly the trap the plan was built to catch**: one day where the book happened to lean into a large move. Over 10.5 months the effect is ~0.05 IC, sub-bps decile edge, and the trivial momentum control carries MORE information than depth. depth_imbalance @ 1h is a regime artifact, not alpha. Same failure shape as Plan 5E's H=96 "Sharpe 1.61" (buy-and-hold beta in disguise).

## Where the whole recon line now stands

| Attempt | Result |
|---|---|
| TA features, ML (Plan 5A→5E) | no edge (Brier ≥ 0.25 baseline) |
| BTC/ETH pair trade | ratio trends, not mean-reverting — dead |
| Low-vol breakout | in-sample overfit, OOS collapse — dead |
| Order book: qi (L1) @ seconds | real IC (~0.37) but edge << taker fee — **maker-only**, deferred |
| Order book: depth @ 15m–1h | **this plan — FAILED full-window validation** |

**Honest read**: single-asset ETH *directional* prediction with free public data is now exhausted across four independent signal families. The remaining unexplored paths are structurally different:
1. **qi maker/HF path** — the one signal with real information content; needs a market-making architecture (maker fills, queue position, inventory risk), a different project from the 1h assistant.
2. **BTC→ETH cross-asset lead-lag** — never tested (Phase 2b-2 scope, deferred).
3. **Different market/instrument** — where taker fees are lower relative to volatility, or where the same signals aren't arbitraged out.

## What survives regardless (infrastructure)

- `depth_study.py` (hourly dataset, OOS split, IC/decile/Newey-West metrics) and `depth_validation.py` (full-window driver with pre-committed verdict) are **reusable for any signal × any symbol × any window** — this is now a generic "is this signal real?" harness.
- Fix landed en route: um-futures daily kline CSVs ship header rows (all 320 files) — `load_klines_1h` now tolerates both vintages.
- 320 days of ETHUSDT bookDepth + klines cached under `data/orderbook/_fw/` (gitignored).

## Commits (this branch)

`7a79100` recon_multi column-name robustness · `ece27a2` klines loader · `e99ff0e` hourly dataset + OOS split · `f9bbbc6` IC/decile/NW metrics · `dc62bda` driver + verdict · `89ed836` header-row fix.
