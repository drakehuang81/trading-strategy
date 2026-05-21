# Plan 5C STATUS — Triple-Barrier Labels

**Date**: 2026-04-27
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: `d78ae5e` (Plan 5C plan doc)
**Head commit**: (this commit)

## Summary

Triple-barrier labels (López de Prado ch. 3) implemented as `src/labels/triple_barrier.py` and wired into `scripts/build_labels.py` via `--label-type` switch. Retrained XGBoost on 2-year ETHUSDT 1h with TB labels.

**The intended outcome — Brier dropping below the 0.24 calibration_brier gate threshold — did NOT happen** (Brier moved from 0.2505 → 0.2510, within noise). However, the Pre-Live Gate's `backtest_dsr` value improved dramatically (0.576 → 0.903) because TB labels produce a model with better directional ranking quality even though Brier (calibration accuracy) stayed flat. **Pre-Live Gate result: still 4/8 passed**, same gates failing as Plan 5B-4.

Test count: **369 passed** (Plan 5B-4 baseline 359 + 10 new across Tasks 1-2).

## Task table

| # | Title | Commit | Files |
|---|-------|--------|-------|
| 1 | `triple_barrier_labels` pure module | `b6df526` | `src/labels/triple_barrier.py`, test |
| 2 | `build_labels --label-type` dispatch | `a145378` | `scripts/build_labels.py`, test |
| 3 | Manual smoke (labels + retrain + 2 backtests + gate) | (no commit — runtime) | — |
| 4 | STATUS handoff | (this commit) | this doc |

## Manual smoke results

### Step 1: Label generation

```
labels[forward_up]:     17516 rows, positive share = 0.5046
labels[triple_barrier]: 17519 rows, positive share = 0.5021
```

(Forward_up was generated against Plan 5A's 17520-row klines; TB against Plan 5B-1's 17537-row backfilled klines. Difference = 14 atr_window warmup vs 4 trailing).

Triple-barrier labels are slightly more balanced (50.21% vs 50.46%) — consistent with symmetric TP/SL barriers.

### Step 2: Retrain comparison

| Model | Labels | Calibrator | Brier_iso | Brier_platt |
|-------|--------|-----------|-----------|-------------|
| `92fddb72f14b` (Plan 5B-1) | forward_up | platt | 0.2627 | **0.2505** |
| `271eae87f670` (Plan 5C) | triple_barrier | platt | 0.2695 | **0.2510** |

Brier_platt moved by **+0.0005** — within noise. Both models' Brier sits at trivial baseline (`Brier(p=0.5) = 0.25`), meaning the model's calibrated probabilities are essentially "predict 50% for everything" with minor variation. Calibration gate threshold (0.24) requires beating baseline by 4%; we're not there.

### Step 3: Backtest comparison (5-month OOS, 2025-12 → 2026-04)

| Threshold | Plan 5B-1 (FU) | Plan 5C (TB) | Δ |
|-----------|---------------|---------------|----|
| 0.58 (spec default) | 0 trades | **0 trades** | — |
| 0.51 (looser) | n=44, Sharpe=0.99, DSR=0.576 | **n=74, Sharpe=2.75, DSR=0.903** | +75% trades, **2.7x Sharpe** |

The 0.51-threshold improvement is real but suspicious:
- 74 trades is still small sample; Sharpe estimates are noisy.
- DSR `n_trials=2` under-counts. We've now tried (FU+0.58, FU+0.51, TB+0.58, TB+0.51) = 4 trials. Real DSR with N=4 would be lower than reported.
- Threshold 0.51 is itself cherry-picked; production is supposed to use ≥0.58.

At spec-default threshold 0.58, both models produce **zero trades** — model never crosses 58% confidence. So **the model is not tradeable in production-spec config**, regardless of Brier or DSR with looser thresholds.

### Step 4: Pre-Live Gate result

Same 4/8 passed:

```
✅ no_repainting     all repainting tests passed
✅ backtest_dsr      DSR 0.9033 > 0.5  (was 0.576 in Plan 5B-4)
❌ calibration_brier platt Brier 0.2510 >= threshold 0.24
✅ reconciliation    0 unresolved diffs in last 14d  (vacuously true)
✅ drift_stability   0 drift HALTs in last 30d        (vacuously true)
❌ paper_runtime     no heartbeat rows
❌ watchdog_uptime   watchdog log not found
❌ halt_diversity    missing 3 trigger_sources
```

No gate flipped — same 4 fails. `backtest_dsr` is greener but was already passing.

## Decisions landed

- **Symmetric barriers `tp=sl=1.5`** chosen — preserves clean binary semantic.
- **Simplified ATR (mean H-L)** — consistent with `RollingKlineCache.atr` (Plan 5B-2). Not Wilder.
- **Output column `y_triple_barrier_4`** — train script identifies by index "as_of" join, so column name is informational only.
- **Warmup convention**: first `atr_window=14` bars are NaN (one bar more conservative than what `rolling().mean()` natively gives — implementer added explicit guard, accepted by spec reviewer).
- **Old `_labels.parquet` (forward_up) preserved** — A/B comparison possible without rerun.

## Sanity check on the new model

**The model still has no useful edge at production-spec threshold (0.58).** Both labels (forward_up and triple_barrier) trained models that:
1. Produce Brier ≈ baseline 0.25 (no calibration gain).
2. Never cross the 0.58 spec-default threshold (no production-spec trades).

The 0.51-threshold backtest improvement (Sharpe 0.99 → 2.75) is real but:
- Not at the production threshold.
- Still small-sample (74 trades).
- DSR is under-corrected (n_trials should be 4+).

> ⚠️ **DO NOT enable `cfg.use_trained_model=True` based on Plan 5C results.** The Pre-Live Gate correctly refuses (`calibration_brier` red). Triple-barrier labels alone don't fix the model.

**The actual bottleneck**: features themselves don't have signal for 4-bar direction prediction on ETHUSDT 1h. Plan 5C's negative result narrows the diagnostic:
- ❌ Label noise (Plan 5C tested — not the cause).
- ❌ Calibration method (Plan 5A Q1 tested — Platt vs isotonic both ~baseline).
- ⚠️ Possibly: features stack itself lacks predictive content.
- ⚠️ Possibly: 4-bar horizon is too short for these features (try 24, 48, 96 bars).
- ⚠️ Possibly: single-asset / single-timeframe is fundamentally hard (spec §12 red flag).

## What is NOT done (Plan 5C-2 / 5D / 5E scope)

- **Plan 5C-2** (small wins bundle): `gap=horizon` in TimeSeriesSplit (label leakage), NaN-pass-through, drop 4 dead-weight funding columns. Likely won't move Brier either, but cheap to ship.
- **Plan 5D** (live activation prep): Ollama / Gemma activation, LiveConfirmViaTelegram, mypy strict, wire `pre_live_gate` into `wiring.py`'s live branch.
- **Plan 5E** (real model improvement, big ticket):
  - Try longer horizons (24, 48, 96 bars) — possibly more signal, less noise.
  - Add new feature families (volume profile, order book imbalance, on-chain).
  - Multi-symbol training (BTCUSDT + ETHUSDT shared model).
  - Switch from XGBoost to LightGBM or LSTM for sequence-aware prediction.
  - **Honest verdict**: spec §12 ("expect IC near zero after costs") may be the operative reality. If Plan 5E experiments also produce Brier ≈ 0.25, conclude that this strategy concept doesn't have edge and pivot to a different setup family.

## Known follow-ups

- **Calibration gate threshold review**: spec §10.1.3 says "below threshold" without naming a number. We picked 0.24. If the project decides the strategy doesn't need calibration improvement to ship (relies on backtest DSR + paper runtime gates), the threshold could be relaxed to 0.249. This is a strategy decision, not an engineering one.
- **DSR n_trials accounting**: today's `--n-trials 2` is wrong now that we've tested 4 (model+threshold) combos. Plan 5E should add `--n-trials 4` to backtest CLI or auto-count from `backtest_runs` history.
- **Triple-barrier `tp_mult`/`sl_mult` sweep**: we picked `1.5/1.5` blindly. Plan 5E could sweep 1.0/2.0/3.0 and pick by validation OOS Brier or by some other criterion.
- **TB labels use horizon=4 same as forward_up** — but TB barriers may resolve much earlier than horizon. Effective horizon distribution would be informative; emit it as a stat in the build_labels script.
