# Plan 5B-3 STATUS — Walk-forward Backtest + DSR

**Date**: 2026-04-27
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: `d5d433b` (Plan 5B-3 plan doc)
**Head commit**: (this commit)

## Summary

`scripts/backtest.py` runs the trained model over an OOS window of historical klines, drives `ReplayBroker.set_time` bar-by-bar, computes Sharpe + Deflated Sharpe Ratio (Bailey & López de Prado 2014 probability form), and writes one row to `backtest_runs`. Manual smoke produced the first quantitative answer to "does this model have any edge?" — **DSR ≈ 0.576 with looser thresholds, 0 trades at spec default threshold 0.58**.

Test count: **333 passed** (Plan 5B-2 baseline 320 + 13 new across 4 code tasks).

## Task table

| # | Title | Commits | Files |
|---|-------|---------|-------|
| 1 | Sharpe + DSR module | `8664144`, `c2dc8af` | `src/observability/sharpe.py`, test |
| 2 | `ReplayBroker.drain_events()` | `8ee0291` | `src/execution/replay_broker.py`, test |
| 3 | `equity_curve` from broker events | `3d9970c`, `807ae9e` | `src/backtest/equity.py`, test |
| 4 | `BacktestRunner` + `scripts/backtest.py` | `80acc99`, `f5210e4` | `runner.py`, `backtest.py`, 2 tests |
| 5 | Manual smoke + STATUS | (this commit) | this doc |

## Manual smoke results

> ⚠️ **Invocation gotcha**: `scripts/backtest.py` naming collides with the `src/backtest/` package. Run via `python -m scripts.backtest` from project root, NOT `python scripts/backtest.py`. The latter shadows the package and raises `ModuleNotFoundError: No module named 'backtest.runner'`. Future plan can rename the script to `scripts/run_backtest.py` to avoid this.

**Run 1 — default thresholds (long=short=0.58, matching spec §4.4)**:
```
PYTHONPATH=src python -m scripts.backtest \
    --kline data/history/ETHUSDT_1h.parquet \
    --features data/training/ETHUSDT_1h_features.parquet \
    --funding data/funding/ETHUSDT.parquet \
    --model-dir models \
    --sqlite-path data/state.db \
    --oos-fraction 0.2
→ run_id=fd6bb0e25647 sharpe=None dsr=None n_trades=0
  oos window: 2025-12-01 → 2026-04-26 (3507 bars ≈ 5 months)
```

The trained model `92fddb72f14b` never produces `prob_up > 0.58` over the OOS window. This is consistent with Plan 5B-1's finding (Brier 0.2505 ≈ baseline 0.25) — the model's calibrated probabilities cluster tightly around 0.5.

**Run 2 — looser thresholds (long=short=0.51, "near always fire")**:
```
... --long-threshold 0.51 --short-threshold 0.51
→ run_id=0a5e78d23a22 sharpe=0.9875 dsr=0.5756 n_trades=44
  initial=$10,000.00 final=$10,147.61
```

DSR = 0.576 marginally passes spec §10.1.2 ("DSR > 0.5"). Annualised return ≈ 3.6%. Sharpe 0.99 over 5 months on 44 trades.

## Decisions landed

- **DSR returns probability `Phi(z)`** (Bailey-de Prado 2014 §4 convention) — Task 1 deviation from plan text, accepted by spec reviewer.
- **`PERIODS_PER_YEAR_1H = 8760` constant** centralised in `src/observability/sharpe.py`.
- **`ReplayBroker.current_ts` + `current_mid()` public properties** (Task 4 fix `f5210e4`) — runner no longer reaches into `_current_ts`.
- **Per-bar portfolio refresh** in `BacktestRunner.run` (Task 4 fix) — future risk-aware policies see live broker state.
- **`run_id = uuid.uuid4().hex[:12]`** (Task 4 fix) — content-addressed sha256 collided on rerun.
- **Single-fold OOS = last 20%** of features parquet. Multi-fold rolling deferred to Plan 5C.
- **No LLM in backtest** (spec §9.6 — "OOS holdout never seen by LLM").
- **`backtest_runs.cost_model_version="plan5b2_v1"`** hardcoded — future task should hash actual `PaperBrokerConfig`.
- **`atr_provider=lambda: 15.0`** hardcoded for ETHUSDT (Plan 5C should compute true ATR).
- **`fixed_position_size=0.01` ETH** per trade — for $10k equity at $3k mid that's a $30 position, 0.3% gross exposure. Way too small to move the needle for live, but appropriate for a sanity-check backtest.

## Sanity check on the model

**Read 1**: With spec-default threshold 0.58, the model produces **zero proposals over 5 months of OOS data**. This means the trained calibration never assigns >58% confidence in either direction. For practical purposes, **this model has no usable signal at the spec-prescribed threshold**.

**Read 2**: With threshold 0.51 (i.e., trade on any directional lean), DSR 0.576 = "57.6% probability the true Sharpe > 0". That's marginal — half a standard deviation above the null. Better than coin flip, but with significant multiple-testing concern (we tried 2 thresholds, n_trials=2 in DSR is now under-counting).

**Read 3**: The ~$148 profit on $10k initial equity over 5 months = 0.36% per month. Even before subtracting opportunity cost, this is below most savings accounts. With realistic capital deployment overhead and the risk that the next 5 months produce a similar-magnitude loss (Sharpe 0.99 is consistent with that), **this is not a tradeable strategy**.

> ⚠️ **DO NOT enable `cfg.use_trained_model=True` in any live or paper-money configuration based on Plan 5B-3 results.** The Brier from Plan 5B-1, the zero-trade behavior at spec default threshold, AND the marginal DSR at relaxed threshold all converge on the same answer: this model needs Plan 5C improvements (gap=horizon CV, NaN-pass-through, triple-barrier labels, multi-symbol features) before any live discussion is warranted.

## Bugs caught by review

1. **DSR z-score vs probability** (Task 1): Plan said "return raw z" but docstring + test + spec §10.1.2 threshold ("DSR > 0.5") all imply probability. Implementer caught it; returned `norm.cdf(z)`.
2. **Position-math basis-stays partial-reduce comment** (Task 3): code reviewer requested explicit comment so future readers don't "fix" the intentional behavior.
3. **`mark_prices` naive datetime silent misalign** (Task 3): added tz-aware guard with `ValueError`.
4. **`run_id` PK collision on rerun** (Task 4 — Critical): content-hashed run_id collided when rerunning the same command. Fixed with `uuid.uuid4().hex[:12]`.
5. **Stale portfolio in runner loop** (Task 4 — Important): future risk-aware policies would silently see initial state. Fixed with per-bar refresh.
6. **Private `_current_ts` access + duplicate mid logic** (Task 4 — Important): added public `current_ts` + `current_mid()` to ReplayBroker.

All bugs caught BEFORE the manual smoke produced backtest_runs rows that would feed Plan 5B-4's Pre-Live Gate. Important: the gate evaluates `backtest_runs.deflated_sharpe`, so DSR getting the wrong number type (z-score vs probability) would have silently broken the gate.

## What is NOT done (Plan 5B-4+ scope)

- **Plan 5B-4**: Pre-Live Gate module (§10 with all 8 gates) — uses these `backtest_runs` rows + `model_versions` + `heartbeat` + `halt_events` to produce a binary "can we go live?" decision.
- **Plan 5C** (model quality): `gap=horizon` in TimeSeriesSplit, NaN-pass-through, triple-barrier labels, multi-symbol watchlist, drop dead-weight funding sub-features. Will likely produce a model worth backtesting again.
- **Plan 5D** (live activation): Ollama / Gemma activation, LiveConfirmViaTelegram, real LiveBroker.
- **Multi-fold rolling backtest**: this plan delivered single-fold OOS. Plan 5C should add proper walk-forward with purged-CV embargo (López de Prado).
- **Hyperparameter sweep over thresholds**: today's smoke tried 2 manually; Plan 5C should sweep + record all to backtest_runs with proper N for DSR.

## Known follow-ups

- **Script-name shadowing** (`scripts/backtest.py` ↔ `src/backtest/`): rename script to `scripts/run_backtest.py` (1 commit, 1 test rename).
- **`atr_provider=lambda: 15.0`** hardcoded — should compute trailing ATR from klines.
- **`cost_model_version="plan5b2_v1"` string literal** — should hash actual PaperBrokerConfig snapshot for drift detection.
- **`day_pnl_r=0.0, consecutive_wins=0` in portfolio** — backtest doesn't track these; future risk policies that read them will get stale data.
- **Position-math duplication** between `ReplayBroker._update_position` and `equity_curve`: same 3-branch logic in two places. Risk of drift if either changes. Extract to shared helper.
- **`mid_provider` lambda capture issue**: today's lambda closes over `self.broker.current_mid()` which is fine, but if multiple symbols ever flow through, the lambda would always read `self.broker`'s clock — single-symbol bug latent.
- **Test `test_runner_produces_equity_series`** asserts `n_trades >= 1` but doesn't verify equity actually changed. Strengthen to `assert result.final_equity != result.initial_equity`.
