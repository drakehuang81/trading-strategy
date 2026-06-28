# Order Book Microstructure — Edge Reconnaissance Design Spec

- **Date**: 2026-06-28
- **Status**: Draft (pending user review)
- **Scope**: a self-contained **research experiment** that quantifies the short-horizon predictive power of order book microstructure signals on ETH/BTC USD-M perpetuals, and produces an IC-vs-horizon curve that decides the next architecture step. It does NOT trade, touch the broker, or use the LLM.
- **Precedes**: any order-book trading system. This experiment gates that decision.

---

## 1. Context

Plan 5A→5E established that the current TA feature stack (SMC/Fib/Liquidity/Divergence/Funding/Confidence) has **no calibrated directional edge** on ETHUSDT 1h (Brier ≥ 0.25 baseline across 2 calibration methods, 2 label types, 4 horizons). Two follow-up non-directional probes (BTC/ETH pair trade, low-vol breakout) also failed — the pair trade because the BTC/ETH ratio is strongly *trending* (OU half-life ~151 days), the breakout because it was in-sample overfit (OOS Sharpe −0.25).

The user's chosen direction is to **seriously hunt for an edge**, accepting that a new data source is required. Among candidates (multi-asset momentum, order book microstructure, on-chain), the user selected **order book microstructure** — the signal family with the strongest theoretical/empirical claim to persistent short-horizon predictive power (order flow imbalance).

The central tension: microstructure alpha lives at the second-to-minute scale and decays fast, but this project's architecture is explicitly **1h macro scan** with an LLM in the loop (spec §2 lists "sub-minute / streaming tick trading" as a Non-Goal). Rather than bet an architecture blind, the agreed first step is a **cheap reconnaissance experiment**: measure where the edge lives (which horizon, how strong), then let the data choose the architecture.

References:
- Personal Trading Assistant design: [docs/superpowers/specs/2026-04-18-personal-trading-assistant-design.md](2026-04-18-personal-trading-assistant-design.md)
- Plan 5E negative-result STATUS (in `pivot/foundation` history): horizon sweep, no sweet spot
- Negative-result probes: `scripts/btc_eth_ratio_analysis.py`, `scripts/vol_breakout_analysis.py`, `scripts/vol_breakout_oos.py`

## 2. Goals & Non-Goals

### Goals

1. **Measure predictive power, not build a strategy.** Produce an IC-vs-horizon table/curve for each microstructure signal on ETH (and BTC as cross-asset predictor).
2. **Data-driven architecture decision.** The output maps directly to a follow-up path via a pre-committed decision rule (§10).
3. **Local-first, $0 spend.** Runs on M1 Pro 16GB; data is free from `data.binance.vision`.
4. **Reuse project discipline.** Point-in-time / no-repainting, append-only artifacts, honest negative results.

### Non-Goals (YAGNI — stated up front)

- No order submission, no broker integration, no PaperBroker.
- No full strategy backtest (that is post-recon, only if an edge is found).
- No high-frequency execution system.
- No LLM involvement.
- No precise cost model — recon uses only raw IC plus a *coarse* cost sensitivity (spread + taker fee).
- No live recording dependency — recon backtests on **downloaded history**, not on TickRecorder output.

### Constraints

- **Market**: USD-M perpetuals (ETHUSDT, BTCUSDT). Order book history exists only for perps, and this matches the project's existing funding-rate context.
- **Hardware**: M1 Pro 16GB. Multi-level depth × 2 symbols × ~2 months may be tens of GB → streaming/chunked processing is mandatory.
- **Isolation**: lives in a dedicated research area; must not import from or pollute production `decision/`, `execution/`, `models/`.

## 3. Data

| Data type | Content | Use |
|---|---|---|
| `bookTicker` | best bid/ask price + qty, event-level | L1 queue imbalance, OFI |
| `bookDepth` | multi-level depth snapshots (granularity confirmed in Step 0, §3.1) | depth imbalance, book slope |
| `aggTrades` | aggregated trades incl. `is_buyer_maker` | taker buy/sell imbalance |

- **Symbols**: ETHUSDT (target), BTCUSDT (cross-asset predictor).
- **Range**: ~2 months, chosen to span more than one volatility regime. Tunable.
- **Source**: `data.binance.vision` (USD-M futures daily/monthly archives), downloaded, decompressed, column-pruned, stored as parquet.

### 3.1 Step 0 — de-risk before building (blocking)

`bookDepth` actual granularity (snapshot frequency, number of levels) is **not yet verified**. The experiment's first action is to download **1 day** of each data type for ETHUSDT, confirm the schema and granularity, and only then finalize the multi-level feature definitions.

**Fallback**: if `bookDepth` turns out unavailable or too coarse for ETH perp, the experiment drops to `bookTicker` (L1) + `aggTrades`, which still supports L1 imbalance, OFI, taker imbalance, and cross-asset lead-lag — i.e. the full recon minus the multi-level depth group. This fallback is a first-class supported outcome, not a failure.

## 4. Components & Data Flow

```
data.binance.vision (S3)
   │  bookTicker / bookDepth / aggTrades   (ETH + BTC perp)
   ▼
┌──────────────┐  prune + convert   ┌───────────────┐
│ 1 Downloader │ ─────────────────▶ │ parquet cache │  + manifest
└──────────────┘                    └──────┬────────┘
                                           │ chunked / lazy
                                           ▼
                                  ┌────────────────────┐
                                  │ 2 Microstructure   │  L1 imbalance / OFI /
                                  │   features          │  depth imbalance / taker /
                                  └─────────┬──────────┘  BTC→ETH lead-lag
                                            ▼
                                  ┌────────────────────┐
                                  │ 3 Align + labels   │  grid + mid + forward
                                  └─────────┬──────────┘  returns @ horizons
                                            ▼
                                  ┌────────────────────┐
                                  │ 4 IC analysis      │  Spearman IC, t-stat,
                                  └─────────┬──────────┘  quantile layering, cost
                                            ▼
                                  ┌────────────────────┐
                                  │ 5 Report (nb + md) │  IC-vs-horizon + verdict
                                  └────────────────────┘
```

| # | Component | Responsibility | Interface (in → out) | Depends on |
|---|---|---|---|---|
| 1 | **Downloader** | fetch, decompress, prune columns, write parquet + manifest | `(symbol, data_type, date_range) → list[parquet_path]` | `data.binance.vision`; existing `download_history.py` pattern |
| 2 | **Microstructure features** | compute each imbalance signal, strictly point-in-time | `(book/trade df, window) → signal series` | (1) parquet |
| 3 | **Align + labels** | align event-level to a time grid, compute mid + forward returns | `(signals, mid, horizons) → aligned df` | (2) signals |
| 4 | **IC analysis** | IC, significance, quantile layering, coarse cost sensitivity per (signal × horizon) | `(aligned df) → IC table` | (3) aligned df |
| 5 | **Report** | render IC-vs-horizon plots, write conclusion | `(IC table) → notebook + markdown` | (4) IC table |

**Design decision — research features are lightweight functions.** They do NOT implement the production `Feature` Protocol (to avoid coupling research code to production contracts), but they DO keep the project's no-lookahead discipline: every signal has a point-in-time test.

## 5. Signal Definitions

All signals computed at multiple aggregation windows where applicable (e.g. 1s / 10s / 1m).

- **L1 queue imbalance (QI)** — from `bookTicker`:
  `QI = (bid_qty − ask_qty) / (bid_qty + ask_qty)`, range [−1, 1].
- **Order Flow Imbalance (OFI)** — Cont, Kukanov & Stoikov (2014), from consecutive best bid/ask updates:
  - per update `n`: `e_n = ΔW_bid − ΔW_ask`, where bid contribution is `+bid_qty_n` if `P_bid` rose, `−bid_qty_{n-1}` if it fell, signed-delta if unchanged (symmetric for ask);
  - `OFI = rolling_sum(e_n)` over the window.
- **Depth imbalance (DI)** — from `bookDepth`, top-K levels:
  `DI = (Σ bid_qty_i − Σ ask_qty_i) / (Σ bid_qty_i + Σ ask_qty_i)`; also a depth-weighted variant (weight by distance to mid).
- **Book slope** — how quickly cumulative depth grows with distance from mid (liquidity concentration); a single regression slope per side.
- **Taker imbalance (TI)** — from `aggTrades` over a rolling window:
  `TI = (taker_buy_vol − taker_sell_vol) / total_vol`; direction from `is_buyer_maker`.
- **Cross-asset lead-lag** — BTC's QI / OFI / TI as predictors of ETH forward return at lag ≥ 0 (does BTC microstructure lead ETH price?).

## 6. Alignment & Labels

- **Time grid**: a uniform grid (e.g. 1s) is the join base. `bookTicker` (event), `bookDepth` (snapshot), and `aggTrades` (trade) each have different time axes; each signal is **as-of joined** (backward) onto the grid — only information with timestamp ≤ t is used.
- **Mid-price**: `mid = (best_bid + best_ask) / 2` on the grid.
- **Forward returns**: `r(t → t+h) = mid_{t+h} / mid_t − 1` for `h ∈ {1s, 5s, 10s, 30s, 1m, 5m, 15m, 30m, 1h}`. These horizons are the x-axis of the IC curve.
- **No-lookahead**: signals use backward as-of; forward returns use strictly future grid points. Funding is negligible at these horizons and ignored.

## 7. IC Analysis Method

- **IC**: Spearman rank correlation between `signal_t` and `r(t → t+h)`, per (signal × horizon × window).
- **Significance**: Newey-West or block-bootstrap t-stat (naive t-stat overstates significance under the heavy autocorrelation of high-frequency data).
- **Quantile layering**: bucket the signal into 5–10 quantiles; report mean forward return per bucket and check monotonicity (a real edge is monotone, not just a nonzero IC number).
- **Coarse cost sensitivity**: at horizons with positive IC, subtract spread + taker fee to see whether the edge survives transaction costs even roughly.
- **Multiple-testing guard**: many signals × many horizons inflate false positives — the report states the number of tests run, and any headline conclusion is re-checked on an **OOS holdout** segment never inspected during exploration.

## 8. Constraints & Risk Mitigations

| Risk | Mitigation |
|---|---|
| Data volume (multi-level depth × 2 symbols × 2 months, tens of GB) | polars **lazy / per-day chunked** processing; column pruning; persist only aggregated results per day, never load the whole set into memory |
| `bookDepth` granularity unknown | Step 0 verifies on a 1-day sample before building multi-level features (§3.1) |
| `bookDepth` possibly unavailable for ETH perp | Fallback to `bookTicker` + `aggTrades` (§3.1) — a supported outcome |
| No-lookahead / alignment direction | backward as-of joins for signals; strictly-future forward returns; fixture tests |
| Three different time axes (event / snapshot / trade) | mid-price grid is the base; all signals as-of joined onto it |
| Spurious high-frequency IC significance (autocorrelation) | Newey-West / block bootstrap; require quantile monotonicity, not just IC magnitude |
| Multiple-testing false positives | report test count; OOS holdout for headline conclusions |

## 9. Testing Strategy

Reusing project discipline:

- **No-repainting / point-in-time** test per signal.
- **Formula correctness**: hand-built small book/trade fixtures with known expected imbalance values.
- **IC computation**: synthetic data with a known correlation → assert the IC computation recovers it.
- **Alignment**: fixture asserting as-of joins never pull future information onto the grid.
- **End-to-end smoke**: run a small sample through all five components to produce a report.

## 10. Decision Rule (pre-committed — the soul of the recon)

Stated before seeing results, to avoid post-hoc p-hacking:

| Recon result | Next architecture |
|---|---|
| Edge only at **< 1 minute** | **High-frequency subsystem** (departs from the 1h + LLM architecture) |
| Edge at **1–15 minutes** | **Event-triggered** path (adapt the existing on-demand route) |
| Edge at **≥ 1 hour** | **Aggregate into the existing 1h ensemble** |
| **No IC** after coarse costs | Honestly stop; record the negative result (like Plan 5E); revisit multi-asset momentum or another source |

"Edge" means: statistically significant IC with monotone quantile layering that survives coarse transaction costs on the OOS holdout.

## 11. Repository Layout

```
src/research/microstructure/      # research module (NOT production)
  __init__.py
  download.py                     # component 1
  signals.py                      # component 2
  align.py                        # component 3
  ic.py                           # component 4
scripts/recon/
  download_orderbook.py           # CLI wrapper for component 1
  run_recon.py                    # orchestrates 2→3→4, writes IC table
notebooks/
  orderbook_recon_report.ipynb    # component 5 (visual report)
data/orderbook/                   # downloaded parquet cache (gitignored)
tests/research/microstructure/    # point-in-time + formula + IC tests
docs/superpowers/specs/2026-06-28-orderbook-microstructure-recon-design.md
```

Research code is isolated under `src/research/` and imports nothing from production decision/execution layers.

## 12. Open Questions (resolved during Step 0 or noted as deferred)

- **Q1 · `bookDepth` granularity** — resolved by Step 0 sample download; determines the multi-level feature set (or triggers the L1 fallback).
- **Q2 · Exact date range** — default ~2 months; may extend if a regime is under-represented. Decided after Step 0 reveals per-day data size.
- **Q3 · Grid resolution** — default 1s; if data is coarser than 1s, the grid matches the coarsest needed resolution. Confirmed after Step 0.
- **Deferred** — multi-symbol expansion beyond BTC/ETH, and any actual trading system, are out of scope until the recon verdict is in.
