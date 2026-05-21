# Plan 5A STATUS — Real Data + Real Model

**Date**: 2026-04-26
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: `f86b41f` (Plan 5A doc commit)
**Head commit**: `63109bd` (E2E smoke + this STATUS)

## Summary

All 11 tasks complete. The orchestrator can now boot with `use_trained_model=True`, load a real XGBoost bundle from `models/`, fetch real klines from Binance, compute features, predict via the trained model + Platt calibration, and persist a proposal to `proposals` table. The Plan 5A goal "first real prediction in paper mode" is achieved.

Test count: **288 passed** (excluding any pre-existing env failures).

## Task table

| # | Title | Commit | Files |
|---|-------|--------|-------|
| 1 | Async `build_scan_context` + BinanceKline lifecycle | `ebd2fbc`, `1bc1ea8`, `46f50f7` | `wiring.py`, `orchestrator.py`, 4 test files |
| 2 | `RollingKlineCache` | `0501b64`, `ab85249` | `data/kline_cache.py` |
| 3 | Cache-backed mid/atr/spread providers | `8326744` | `data/providers.py` |
| 4 | Wire cache + 1-min refresh loop | `2fc7821`, `3e72077` | `wiring.py`, `orchestrator.py` |
| 5 | `download_history.py` paginator + funding | `4c9fa65`, `13c6d38` | `scripts/download_history.py` |
| 6 | `build_training_set.py` + funding_rate.py fix + `flatten_features` promotion | `67e5696`, `836849e` | `scripts/build_training_set.py`, `features/registry.py`, `models/xgb_predictor.py` |
| 7 | `build_labels.py` (4-bar forward up) | `18f8151` | `scripts/build_labels.py` |
| 8 | Walk-forward XGBoost + isotonic vs Platt + datetime/validation fix | `c569ff6`, `dcb3e2a` | `scripts/train_xgb.py`, `models/xgb_predictor.py` |
| 9 | `models/registry.py` (`load_latest_model`) | `fe659f8` | `models/registry.py` |
| 10 | Drift reference baseline (writer + loader) | `0ae4356` | `scripts/train_xgb.py`, `wiring.py` |
| 11 | E2E smoke + STATUS | `63109bd` | `tests/e2e/test_real_data_smoke.py`, this STATUS |

## Manual smoke results

**Task 5** — `python scripts/download_history.py --years 2`:
- klines: 17520 bars (2024-04-25 → 2026-04-25) → `data/history/ETHUSDT_1h.parquet` (734 KB)
- funding: 200 rows (2026-02-18 → 2026-04-25) → `data/funding/ETHUSDT.parquet` (5.6 KB) — Binance public API limit; older funding history requires backfill paginator (Plan 5B).

**Task 6** — `python scripts/build_training_set.py ...`:
- features: 17320 rows × 37 cols → `data/training/ETHUSDT_1h_features.parquet` (937 KB)

**Task 7** — `python scripts/build_labels.py ...`:
- labels: 17516 rows (4 trailing NaN dropped); positive class share = **0.5046** → balanced ✅

**Task 8** — `python scripts/train_xgb.py --features ... --labels ... --out models`:
- model_version: `ece8d16d4a29`
- calibration: **platt** (Brier 0.2502 < isotonic 0.2577)
- training window: 2024-05-04 → 2026-04-25 (~2 years)
- training rows: 17316 (after as_of inner join)
- features: 37
- `model_versions` SQL row inserted with real datetime types ✅
- **drift_reference.json NOT generated** in this run because Task 10 (writer) committed AFTER training. Re-run training to backfill the reference (~3-5 min) — see "What is NOT done" below.

## Decisions landed

- **Calibration method**: Platt won (lower OOS Brier on walk-forward CV). Recorded in `model_versions.calibration_method`. Spec §14 Q1 resolved.
- **Label rule**: binary `close[t+4] > close[t]`. Triple barrier deferred to Plan 5B.
- **Training window**: ETHUSDT 1h × 2 years. Walk-forward 5-fold TimeSeriesSplit with internal 80/20 fit/calib split.
- **`flatten_features` is now public** in `features/registry.py` — single source of truth for train→serve flatten contract.
- **`_run_model` dispatches by `hasattr(transform)`** for Isotonic vs Platt. Plan 5B should consider explicit `meta["calibrator_kind"]` tag for robustness.
- **gap=horizon NOT applied** in TimeSeriesSplit. TODO comment in `train_xgb.py` flags label leakage between train/calib folds. Defer to Plan 5B.
- **fillna(0.0) NOT NaN-pass-through**. TODO in `_run_model` and `build_training_set`. Plan 5B should switch to NaN for proper XGBoost native missing-value handling.
- **OllamaClient scheduler must be started before scan**: both `Ensemble` (via `GemmaContextProvider.flags` → `complete()`) and `ChatLLM.explain` (via `chat()`) call `_acquire()` which blocks on `ticket.ready.wait()` forever if `start()` was not called. Tests that invoke `scheduled_macro_scan` directly (without `orch.run()`) must patch both `OllamaClient.complete` and `OllamaClient.chat` to avoid hangs.

## E2E smoke test outcome (Task 11)

The smoke test (`tests/e2e/test_real_data_smoke.py`) exercises the full pipeline:

1. `build_default_registry().compute_all(_fake_klines())` → derive `feature_order` (27 features)
2. Seed a 200-row XGBoost + isotonic calibrator in `tmp_path/models/`
3. Boot orchestrator with `use_trained_model=True`, `long_threshold=0.0`, `short_threshold=0.0`
4. Patch `BinanceKline.open` → return fake klines
5. Patch `OllamaClient.complete` + `OllamaClient.chat` → raise `ConnectionRefusedError`
6. Call `scheduled_macro_scan(ctx, trace_id="smoke")`
7. Assert `SELECT COUNT(*) FROM proposals >= 1`

Observed: **1 passed, 2.91s**. Log output confirmed:
- `llm_context_failed_falling_back_to_ml_only` (Ensemble exercised ML-only path)
- `rationale_generation_failed` (ChatLLM.explain skipped via except-branch)
- `order_submitted` (proposal accepted and executed through PaperBroker)

## Sanity check on the model

Brier 0.2502 vs baseline 0.25 (Brier of `p=0.5`) means the model has **near-zero alpha** on raw 4-bar direction prediction. This matches spec §12's red flag ("single asset, single timeframe ML — expect IC near zero after costs"). Plan 5A's goal was to ship the pipeline, not to ship a profitable model. Plan 5B's training improvements (triple barrier, gap=horizon, NaN handling, more features) are needed before the §10.1.3 calibration gate can plausibly pass.

> ⚠️ **DO NOT enable `cfg.use_trained_model=True` in any live or paper-money configuration based on Plan 5A artifacts.** The current model has no demonstrated edge. Live-mode promotion requires Plan 5B's calibration gate (§10.1.3) + walk-forward Deflated Sharpe (§10.1.2) + 60-day paper runtime (§10.2.4) — all explicitly out of Plan 5A scope.

## What is NOT done (Plan 5B scope)

**Step 0 of Plan 5B work** — re-run training to backfill `drift_reference.json` (~3-5 min, no code change needed):
```bash
python scripts/train_xgb.py \
    --features data/training/ETHUSDT_1h_features.parquet \
    --labels   data/training/ETHUSDT_1h_labels.parquet \
    --out      models
```

Plan 5B itself:
- Funding rate backfill paginator (currently only ~2 months of funding history; need 2 years).
- ReplayBroker + LiveBroker contract suite.
- Walk-forward backtest harness + Deflated Sharpe report (§10.1.2).
- Pre-Live Gate module (§10) with all 8 gates.
- LiveConfirmViaTelegram reconciliation (§7.4).
- Heartbeat-DB-failure HALT escalation (Plan 4 TODO).
- mypy `--strict` global pass.
- Ollama / Gemma activation (currently stubbed via Ensemble's `LLM_UNAVAILABLE_MARKER` fallback).
- `gap=horizon` in TimeSeriesSplit, NaN-pass-through, triple-barrier labels.
- Multi-symbol watchlist (currently 1 symbol hardcoded in cache + refresh loop).

## Known follow-ups (deferred during Plan 5A)

- **OllamaClient scheduler requirement in tests**: any test that calls `scheduled_macro_scan` without `orch.run()` must patch both `OllamaClient.complete` and `OllamaClient.chat` to avoid blocking on `_acquire()`. Plan 5B should document this contract explicitly or add a `start_nowait()` mode that short-circuits the queue.
- **Scope-discipline incidents**: Task 6 + Task 9 implementers modified files outside their declared scope (`funding_rate.py` and `xgb_predictor.py` respectively). Both fixes were correct and necessary, but the pattern violates user CLAUDE.md `Subagent 修改範圍限制`. Plan 5B should pre-list these "may also touch" files explicitly so subagents don't have to choose between BLOCKED and silent drift.
- **`_register` argument types**: tightened from `str` → `datetime` in commit `dcb3e2a`. The first test row in `model_versions` still has `training_window_start=0` (int from old `str("0")` cast) — left in place as evidence of the bug; Plan 5B can either purge or accept as historical.
- **`np.random.default_rng(0)` reproducibility comment** in `write_drift_reference` (deterministic per-run, not stable across `max_samples` changes).
- **Silent missing-file warning** in wiring's drift reference loader.
- **Cap-trim test** for `write_drift_reference` (current test only covers 500 rows < cap).
