# Plan 5B-1 STATUS — Funding Backfill

**Date**: 2026-04-26
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: `279ad41` (Plan 5B-1 plan doc)
**Head commit**: (this commit)

## Summary

`FundingRateWriter` gains a `backfill(symbol, *, since, until=None)` method that **forward-paginates** Binance's `futures_funding_rate` endpoint via `startTime` cursor advancing from `since` toward `until`. `scripts/download_history.py` invokes it whenever the on-disk parquet doesn't already cover the requested `--years` window. Manual smoke replaced the ~200-row stub funding parquet with the full **2190 rows** for 2 years of ETHUSDT (range 2024-04-26 → 2026-04-26).

Subsequent retrain produced a new model (`92fddb72f14b`, Brier_platt 0.2505) and finally generated `models/drift_reference.json` (37 cols × 5000 samples each), closing Plan 5A's "Step 0" loose end.

Test count: **296 passed** (Plan 5A baseline 288 + 8 new across Task 1 + Task 2).

## Task table

| # | Title | Commits | Files |
|---|-------|---------|-------|
| 1 | `FundingRateWriter.backfill` (initial impl) | `3f3d228`, `1b348db`, `744762f`, `a68a316` | `src/data/funding.py`, `tests/unit/data/test_funding_backfill.py` |
| 2 | Wire backfill into download_history + manual smoke | `40872df`, `90b350e` | `scripts/download_history.py`, `tests/unit/scripts/test_download_history.py` |

## Manual smoke results

**Funding backfill** (`python scripts/download_history.py --years 2`):
- klines: 17520 bars (unchanged from Plan 5A)
- **funding: 2190 rows** (vs Plan 5A's 200 rows) covering 2024-04-26 → 2026-04-26
- funding rates verified non-trivial: mean 5.2e-5, std 6.5e-5, range [-3.6e-4, 5.8e-4]

**Features rebuild** (`python scripts/build_training_set.py ...`, ~50 min CPU):
- 17337 rows × 37 cols (vs Plan 5A's 17320 — minor as_of boundary difference)
- `funding.rate` column now varies meaningfully (was mostly 0.0/None before)
- 4 derived funding columns (`funding.evaluation.risk_multiplier`, `funding.position_adj.*`) remain constants because all rates fell within NORMAL ±0.05% range — these are dead-weight features for training; address in Plan 5C

**Retrain** (`python scripts/train_xgb.py ...`, ~5 min):
- new model_version: **`92fddb72f14b`**
- calibration: **platt** (Brier 0.2505 vs isotonic 0.2627)
- training rows: 17316 (after as_of inner join with labels)
- features: 37
- `model_versions` row inserted with real datetime types
- **`models/drift_reference.json` generated** (1.4 MB, 37 cols × 5000 samples) ✅

## Decisions landed

- **`backfill` is forward, not backward**. Initial design used backward `endTime` pagination — Binance behavior mismatch (returns oldest 1000 ≤ endTime, not newest 1000) was caught by manual smoke producing wrong-window parquet. Rewrote to forward `startTime` cursor.
- **`backfill` separate from `update`** retained: same algorithm internally, different start cursor (`since` vs `existing.max + 1`); separate names communicate intent.
- **`until: datetime | None = None` parameter**: defaults to `datetime.now(timezone.utc)`; tests pass explicit `until` for determinism.
- **`since` is keyword-only**, both `since` and `until` must be tz-aware (raises `ValueError` otherwise — caught one bug class before it could land).
- **`FakeFundingClient` in tests now models real Binance** (not what we wished it did).
- **download_history uses simple `min ≤ since` gap check** — gap-detection is intentionally naive; assumes parquet contiguity. Valid for Plan 5B-1 but documented as caveat.

## Calibration impact: minimal

| Metric | Plan 5A model | Plan 5B-1 model |
|---|---|---|
| Brier (winner) | 0.2502 (platt) | **0.2505 (platt)** |
| Brier isotonic | 0.2577 | 0.2627 |
| Calibrator winner | platt | platt |

Funding history coverage went from ~2 months to 2 years, but Brier moved by 0.0003 — within noise. Confirms spec §12's red flag: single-asset / single-timeframe / 4-bar binary direction has near-zero alpha. Plan 5C (model quality) is needed before any meaningful calibration improvement.

> ⚠️ **DO NOT enable `cfg.use_trained_model=True` in any live or paper-money configuration based on Plan 5B-1 artifacts.** Brier 0.2505 vs baseline 0.25 means the model still has no demonstrated edge. Live-mode promotion still requires Plan 5B-3 walk-forward DSR + Plan 5B-4 Pre-Live Gate + 60-day paper runtime.

## Bugs found and fixed during this plan

1. **Naive datetime silent-wrong** (caught by code review): `since.timestamp()` on a naive datetime returns local-time epoch. Added tz-aware assertion + test (commit `1b348db`).
2. **Backward pagination wrong against real Binance** (caught by Task 2's manual smoke): `endTime=None` returned the OLDEST 1000 rows starting 2019-11-27, not most-recent. First fix attempt initialized `end_cursor=now_ms` (still wrong because Binance with `endTime=X` returns oldest 1000 ≤ X, not newest 1000 ≤ X). Final fix in `a68a316` switched to forward pagination.

Both were caught BEFORE the model retrain landed; no corrupted artifacts shipped.

## What is NOT done (Plan 5B-2+ scope)

- **Plan 5B-2**: ReplayBroker + Broker contract test suite
- **Plan 5B-3**: Walk-forward backtest harness + Deflated Sharpe report (§10.1.2)
- **Plan 5B-4**: Pre-Live Gate module (§10 with all 8 gates)
- **Plan 5C** (model quality, parallel): gap=horizon TimeSeriesSplit, NaN-pass-through, triple-barrier labels, multi-symbol watchlist, drop dead-weight funding sub-features
- **Plan 5D** (live activation): Ollama / Gemma activation, LiveConfirmViaTelegram, HALT-DB escalation, mypy strict

## Known follow-ups

- **Stale `model_versions` rows**: `data/state.db` contains `50294eaae018` (Task 8 unit-test residue) and `ece8d16d4a29` (Plan 5A model, now superseded). `load_latest_model` picks `92fddb72f14b` by mtime — the older bundles are inert. Cleanup deferred.
- **Dead-weight funding sub-features**: 4 of 5 funding columns are constants on this 2-year ETHUSDT window. Drop or re-engineer in Plan 5C.
- **Gap detection in download_history**: current `min ≤ since` check trusts parquet contiguity. If a future ingest run is interrupted, gaps in the middle won't trigger backfill. Plan 5B-2/3 may need stronger gap detection if the backtest grows pickier.
- **`backfill` returns "rows fetched from API", not "net rows added"**: documented in docstring + pinned in test. Alternative semantic deferred.
