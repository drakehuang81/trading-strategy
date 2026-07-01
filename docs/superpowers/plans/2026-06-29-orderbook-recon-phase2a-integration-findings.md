# Phase 2a Integration Findings — First Real Order-Book IC (2026-06-29)

> Phase 2b 動工前必讀。這是 recon 專案第一個真實數據結果,決定 Phase 2b 的重點。

## Setup

- **Data**: ETHUSDT USD-M perp, **2023-06-01 (single day, in-sample)**.
- Raw sizes: bookTicker **6,400,396** rows, bookDepth 20,270 (12 levels × ~1,689 snapshots), aggTrades 679,324.
- Pipeline: `to_mid_grid(bookTicker, every="1s")` → 86,400 grid rows; 5 signals as-of joined; `recon_multi` Spearman IC.
- Signal params: `ofi(window=100)`, `taker_imbalance(window=500)`, others default.
- Horizons: 1s, 5s, 10s, 30s, 60s, 5m, 15m, 1h.

## IC-vs-horizon (Spearman)

| signal | 1s | 5s | 10s | 30s | 60s | 5m | 15m | 1h |
|---|---|---|---|---|---|---|---|---|
| **qi** (L1 imbalance) | **0.366** | **0.367** | 0.312 | 0.182 | 0.120 | 0.065 | 0.040 | 0.032 |
| ofi | -0.014 | -0.011 | -0.015 | -0.005 | 0.007 | 0.016 | 0.017 | 0.007 |
| **depth_imbalance** | 0.022 | 0.032 | 0.039 | 0.056 | 0.079 | 0.163 | 0.304 | **0.497** |
| book_slope | -0.005 | -0.012 | -0.015 | -0.024 | -0.029 | -0.073 | -0.193 | -0.367 |
| taker_imbalance | 0.025 | 0.018 | 0.003 | -0.024 | -0.035 | -0.019 | 0.009 | -0.048 |

## Findings

1. **qi (L1 queue imbalance) is strong at the second scale** — IC ~0.37 @ 1–5s, decaying fast (0.18 @ 30s, 0.03 @ 1h). Textbook microstructure alpha shape (strong + fast-decaying).
2. **depth_imbalance is strong at the long scale** — IC rises monotonically to 0.50 @ 1h; `book_slope` mirrors it negatively (-0.37 @ 1h). Book shape predicts longer-horizon direction.
3. **ofi and taker_imbalance are near-zero** (|IC| < 0.02). OFI is theoretically strong — suspect `window=100` events is wrong, or an alignment issue. Phase 2b must sweep OFI window and re-check.

## ⚠️ Caveats (do NOT get optimistic yet — Plan 5E lesson)

These IC numbers are **suspiciously high** and are **single-day, in-sample, pre-cost, no-OOS**:
- **qi 0.37 likely contains mechanical bid-ask bounce** — L1 imbalance "predicting" next-tick mid is partly the arithmetic of `mid=(bid+ask)/2`, not necessarily tradeable alpha.
- **depth 0.50 @ 1h may be a slow-moving contemporaneous correlation** (depth drifts with trend), not causal prediction.

**What survives spread + taker fee, and holds out-of-sample, is the real question** — exactly Phase 2b's job.

## Decision-rule pointer (spec §10, provisional)

If Phase 2b validates:
- **qi edge lives at <10s → high-frequency subsystem** (departs from 1h + LLM architecture).
- **depth_imbalance edge lives at ≥15min → can aggregate into the existing 1h architecture.**
- The two signals live at opposite time scales — potentially both useful, for different subsystems.

## Phase 2b must-haves (updated from this run)

1. **Full-window study** over 2023-05-16 → 2024-03-30 (not one day), multi-day chunked driver.
2. **Bounce/cost correction for qi** — subtract spread; test whether qi alpha survives crossing it. This is the make-or-break for the headline 0.37.
3. **OFI window sweep** — the near-zero IC is a red flag; sweep window (e.g. 10/50/200/1000 events) and verify the Cont computation on real ticks.
4. **OOS holdout** + **Newey-West / block-bootstrap t-stats** (86k autocorrelated 1s rows massively inflate naive significance).
5. **Quantile layering** on the real signals (monotone? or driven by tails?).
6. **`recon_multi` robustness** — it silently requires `signals` dict keys == signal column names (the integration driver hit a ColumnNotFoundError). Make it derive the column name or assert clearly.
7. **BTC→ETH cross-asset** signals (deferred from 2a).

## UPDATE — Cost check with taker fee (the real make-or-break)

The IC/decile numbers above are pre-cost. Re-ran decile edge vs REALISTIC
Binance USD-M perp taker fee (round-trip ~3.6 bps BNB-VIP, ~8 bps standard).
**Spread is negligible** (median 0.054 bps — ETH perp is ultra-liquid); the
**taker fee is the binding cost**, which the first check wrongly omitted.

Net decile edge (bps) after taker round-trip:

| horizon | qi net @VIP / std | depth_imbalance net @VIP / std |
|---|---|---|
| 1–60s | all negative | all negative |
| 5m | −2.3 / −6.7 | +0.5 / −3.9 |
| **15m** | −1.6 / −6.0 | **+11.5 / +7.1** |
| **1h** | +1.6 / −2.8 | **+18.7 / +14.3** |

**Verdict (supersedes the provisional pointer above):**
- **qi (L1) is NOT taker-tradeable** — strong IC is at the second scale where
  edge (0.2–0.9 bps) << taker fee. Only viable as a maker / HF market-maker
  (separate architecture). **Deferred.**
- **depth_imbalance @ 15m–1h is the first signal net-positive after realistic
  taker fee** (+7 to +14 bps @ standard taker), IC 0.5, monotone. This horizon
  fits the existing 1h + LLM architecture. **This is the lead.**

### Phase 2b (focused) must-haves — validate depth_imbalance @ 15m–1h
1. **Full-window study** 2023-05→2024-03, multi-day chunked driver (not one day).
2. **Trend control (#1 risk)** — is depth@1h causal, or just "buy-orders pile up
   while price trends"? Compare depth-signal PnL vs buy-and-hold beta; test
   up/down/chop regimes separately.
3. **OOS holdout** (time split) — headline edge must survive out-of-sample.
4. **Full-bucket monotonicity** (not just extreme deciles) + tail-robustness
   (is +22 bps driven by a few big 1h moves?).
5. **Newey-West / block-bootstrap t-stat** (only ~24 non-overlapping 1h
   windows/day — small effective N).
6. **Capacity / turnover** — decile-strategy trade frequency + notional.
7. `recon_multi` robustness (derive signal column name — hit this bug twice).

ofi / book_slope / taker_imbalance, qi's maker-only path, and BTC→ETH
cross-asset are all deferred to a later phase.

## Reproduce

Driver logic (ad-hoc, scratchpad): download bookTicker/bookDepth/aggTrades for the date via `download.build_url/download_zip/extract_zip_to_parquet`, load via `load_book_ticker/load_book_depth/load_agg_trades`, build grid via `to_mid_grid`, build the 5 signals, call `recon_multi(grid, signals, horizons_secs=[1,5,10,30,60,300,900,3600])`. Phase 2b turns this into a proper multi-day CLI.
