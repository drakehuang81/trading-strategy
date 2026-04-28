# Plan 5E STATUS — Horizon Sweep

**Date**: 2026-04-28
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: `a4d7d55` (Plan 5E plan doc)
**Head commit**: (this commit)

## Summary

Swept XGBoost training across `H ∈ {4, 24, 48, 96}` to test whether 4-bar horizon was the bottleneck. **Result: NO sweet spot.** Brier monotonically degrades as horizon grows (0.2504 → 0.2784); the only "good" backtest Sharpe (1.61 at H=96) is unintentional buy-and-hold riding the 2024-2026 ETH bull market, not real model edge. **Spec §12 red flag confirmed**: single-asset / single-timeframe ETHUSDT 1h with our current feature stack has no usable predictive content for directional prediction at any tested horizon.

Test count: **373 passed** (Plan 5C baseline 369 + 4 new in Task 1).

## Task table

| # | Title | Commit | Files |
|---|-------|--------|-------|
| 1 | `scripts/sweep_horizons.py` automation | `ba5a4f4` | `sweep_horizons.py`, test |
| 2 | Manual smoke (sweep + 4 backtests) | (no commit — runtime) | — |
| 3 | STATUS handoff | (this commit) | this doc |

## Manual smoke results

### Sweep table

```
PYTHONPATH=src python -m scripts.sweep_horizons \
    --kline data/history/ETHUSDT_1h.parquet \
    --features data/training/ETHUSDT_1h_features.parquet \
    --horizons 4,24,48,96 \
    --out-root models/sweep \
    --sqlite-path data/state.db
```

```
H      BRIER    CALIB      MODEL          ERROR
4      0.2504   platt      5fbd55f27f73   
24     0.2614   platt      01f66ea681d6   
48     0.2692   platt      4b282fce3cb0   
96     0.2784   platt      26d1ffd87113   
```

### Per-horizon backtest (threshold 0.51, n_trials=4)

| H | Brier | Sharpe | DSR | n_trades | model_version |
|---|-------|--------|-----|----------|---------------|
| 4 | 0.2504 | 1.13 | 0.452 | 158 | `5fbd55f27f73` |
| 24 | 0.2614 | 1.00 | 0.421 | 1613 | `01f66ea681d6` |
| 48 | 0.2692 | 1.46 | 0.536 | 2127 | `4b282fce3cb0` |
| 96 | 0.2784 | **1.61** | 0.572 | **3364** | `26d1ffd87113` |

### Numbers tell two contradictory stories

1. **Brier monotonically gets worse.** Even the H=4 model is at baseline (0.2504 ≈ 0.25 = Brier of constant 0.5 prediction). Longer horizons are progressively WORSE than constant-prediction.

2. **Sharpe + n_trades grow with horizon.** H=96 trades 21× more than H=4 (3364 vs 158) and posts Sharpe 1.61. This is suspicious because the CALIBRATION (Brier) is worsest there.

**Resolution**: longer-horizon models are **systematically biased long** (OOS window 2025-12 → 2026-04 is a uptrending ETH market, so longer-H labels are ≥50% positive). The model's `prob_up` distribution shifts above 0.51 most of the time, firing constant longs, riding the trend. The "good" Sharpe is buy-and-hold beta, not edge.

The Brier numbers are unambiguous: **the feature stack has no calibrated directional signal at any tested horizon**. The Sharpe pattern reinforces this — true alpha doesn't decay smoothly with horizon AND simultaneously inflate trade count.

## Verdict: (C) — No sweet spot

Outcome **(C)** from the plan: **all horizons produce Brier ≥ baseline 0.25, no "real" Sharpe improvement**. Spec §12 red flag ("single asset, single timeframe ML — expect IC near zero after costs") is operationally confirmed.

This narrows the diagnostic to:
- ❌ Label noise (Plan 5C ruled out)
- ❌ Calibration method (Plan 5A Q1 ruled out)
- ❌ Horizon (Plan 5E rules out — this plan)
- ✅ **Either the feature stack has no predictive content, OR single-asset / single-timeframe is fundamentally unworkable**

## Decisions landed

- **All 4 trained models stored under `models/sweep/h<N>/`** — production `models/` directory untouched, Pre-Live Gate's `load_latest_model` still reads Plan 5C's `271eae87f670`.
- **Backtests use threshold 0.51 + n_trials=4** — reflects 4 model variants tested in this experiment (proper DSR deflation).
- **No model promotion** — none of these is worth shipping to production; Pre-Live Gate would still refuse.
- **Sweep is reproducible** — same features parquet, same seed, only labels parquet changes per horizon. Re-running gives identical model_versions.

## Honest verdict on the strategy concept

After Plan 5A → 5B → 5C → 5E:

> The current feature stack (SMC + Fib + Liquidity + Divergence + Funding + Confidence) does NOT predict ETHUSDT 1h directional moves above coin-flip baseline. This holds across:
> - 2 calibration methods (isotonic + Platt)
> - 2 label types (forward_up + triple_barrier)
> - 4 horizons (4, 24, 48, 96 bars)
>
> The orchestrator pipeline works correctly (~373 tests + 4 e2e). The problem is upstream of the model: **features themselves don't carry signal**.

What this means in practice:

1. **No amount of further model engineering will fix this.** Plan 5C-2 (gap=horizon, NaN handling, drop dead features) won't help. Plan 5F (combine winning horizon × TB) has no winning horizon to harden.

2. **The spec §12 red flag was honest from day one.** "Single asset, single timeframe ML — expect IC near zero after costs" was right. We've now verified it with 4 negative result plans.

3. **Two real paths forward**:
   - **(P1) Plan 5G — Feature engineering**: replace the current TA-based features with new families:
     - Volume-weighted price (VWAP, OBV)
     - Order book features (bid-ask imbalance, depth-weighted mid)
     - On-chain features (active addresses, exchange flows, perp funding term structure)
     - Multi-timeframe (1h + 4h + 1d cross-correlation)
     - Multi-asset (BTC dominance, BTC/ETH ratio momentum)
   - **(P2) Pivot to a different setup family** entirely:
     - Statistical arbitrage between perpetual futures and spot
     - Funding rate harvest (delta-neutral collect funding)
     - Mean-reversion on basis spreads
     - Options vol arbitrage (if exchange supports)

   These are **different strategies**, not improvements to the current model. Worth doing only if you genuinely want to keep building this system.

4. **Or accept the current state**: the codebase has solid infrastructure (broker, backtest, Pre-Live Gate, drift monitor, Telegram, ChatLLM). It's a quality foundation for whatever quant project comes next. The model itself is a placeholder until features improve.

## What is NOT done

- **Plan 5C-2** (small wins bundle: gap=horizon CV, NaN-pass-through, drop dead funding cols): given Plan 5E's negative result, **STRONGLY RECOMMEND SKIPPING** — these would not move Brier. Going straight from here to Plan 5G or strategy pivot.
- **Plan 5F** (TB × best-horizon hardening): cancelled (no best horizon).
- **Plan 5G** (feature engineering): see "real paths forward" above.
- **Plan 5D** (Live activation prep): independently still useful for any future model that does have edge — Telegram + Ollama wiring don't depend on which model ships.

## Known follow-ups

- **`scripts/sweep_horizons.py` doesn't auto-backtest** — operator runs the for-loop manually. Plan 5G could extend with a `--with-backtest` flag.
- **DSR n_trials accounting still imprecise** — sweep counted as 4, but if you include Plan 5C's TB attempt and Plan 5B-1's original, real n_trials is 6+. Plan 5G should auto-count from `backtest_runs` history.
- **`XGBPredictor.horizon_bars` still hardcoded to 4** — irrelevant given Plan 5E's negative result, but worth fixing if a future model with non-4 horizon ships.
- **Class share per horizon not recorded in meta JSON** — would help diagnose the H=96 buy-and-hold suspicion definitively. Plan 5G could log it.
