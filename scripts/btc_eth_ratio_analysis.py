"""BTC/ETH ratio pair-trading viability analysis.

Hypothesis: log(BTC) - log(ETH) is mean-reverting on short-medium timescales.
A pair-trade strategy can collect profits from spread mean reversion without
needing directional prediction.

Steps:
  1. Load both 1h kline parquets, align timestamps
  2. Compute log-ratio = log(BTC_close) - log(ETH_close)
  3. Estimate half-life via OU process fit (regress Δlog_ratio on log_ratio)
  4. ADF stationarity test
  5. Build rolling z-score (default 7-day window)
  6. Backtest simple z-score strategy:
       Entry:  |z| > entry_z (default 2.0) → long underpriced, short overpriced
       Exit:   |z| < exit_z (default 0.5) → close
       Stop:   |z| > stop_z (default 4.0) → close
  7. Apply realistic fees (taker 5 bps × 4 fills per round trip)
  8. Report Sharpe, hit rate, max DD, net APY

Run:
  PYTHONPATH=src python scripts/btc_eth_ratio_analysis.py \
      --btc data/history/BTCUSDT_1h.parquet \
      --eth data/history/ETHUSDT_1h.parquet

RESULT (2026-06-28 · ETHUSDT/BTCUSDT 1h · 2024-04 -> 2026-04) — DEAD END:
  OU half-life 151 days (far too slow; the ratio trends, it does not
  mean-revert). |z|>2 share 14.8% vs ~4.6% normal -> fat-tailed/trending.
  Backtest Sharpe -0.85, net APY -12.9%; fees ($3570) alone exceed gross
  PnL. Pair trade on BTC/ETH is not viable on this window.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def estimate_half_life(series: pd.Series) -> float:
    """Estimate OU half-life via OLS regression of Δx_t on x_{t-1}.

    For OU: dx = -k * x * dt + σ dW → β ≈ -k from OLS.
    half_life = ln(2) / k. Returns NaN if not mean-reverting (β >= 0).
    """
    x = series.dropna().to_numpy()
    if len(x) < 100:
        return float("nan")
    delta = x[1:] - x[:-1]
    lag = x[:-1]
    lag_centered = lag - lag.mean()
    cov = (lag_centered * delta).sum() / (lag_centered ** 2).sum()
    if cov >= 0:
        return float("nan")  # not mean-reverting
    k = -cov
    return math.log(2) / k


def adf_test(series: pd.Series) -> tuple[float, float]:
    """Simplified ADF: returns (test_stat, p_value_approx).

    Uses statsmodels if available, else returns (NaN, NaN).
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna().to_numpy(), maxlag=24)
        return float(result[0]), float(result[1])
    except ImportError:
        return float("nan"), float("nan")


def backtest_pair(
    log_ratio: pd.Series,
    *,
    z_window_hours: int = 168,         # 7 days
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    notional_usd: float = 10_000,
    taker_bps: float = 5.0,
) -> dict:
    """Simple z-score-based pair-trade backtest.

    Position state: 0 (flat), +1 (long-spread = long ETH, short BTC),
                              -1 (short-spread = long BTC, short ETH).
    "Spread" here = log(BTC) - log(ETH). Negative spread → ETH overpriced
    relative to BTC → short spread → long ETH, short BTC.

    PnL per bar: position * Δlog_ratio * notional. (Approximation: log-return
    pair PnL is ~Δlog_ratio for small moves.)

    Fees: taker_bps applied on notional at each entry & exit (one round trip
    = 2 transitions × 2 sides × taker_bps).
    """
    z = (log_ratio - log_ratio.rolling(z_window_hours).mean()) / log_ratio.rolling(z_window_hours).std()
    z = z.dropna()
    log_ret = log_ratio.diff().reindex(z.index).fillna(0.0)

    position = 0
    pnl_per_bar = []
    fees_per_bar = []
    n_trades = 0
    n_stops = 0
    pos_durations = []
    cur_duration = 0

    for ts, z_val in z.items():
        # Realise PnL on existing position based on this bar's log-return
        ret = log_ret.loc[ts]
        bar_pnl = position * (-ret) * notional_usd
        # NOTE: position +1 means SHORT spread (long ETH, short BTC).
        #       Spread = log(BTC) - log(ETH). If spread DROPS (ret < 0),
        #       that means ETH outperformed → our long-ETH-short-BTC pair gains.
        #       So PnL = -position * ret. Let me recompute:
        #         position +1 = long-spread (long BTC, short ETH) — gains when ret > 0
        #         position -1 = short-spread (long ETH, short BTC) — gains when ret < 0
        #       PnL = +position * ret. Reset.
        bar_pnl = position * ret * notional_usd
        bar_fee = 0.0

        cur_duration += 1 if position != 0 else 0

        new_position = position
        if position == 0:
            # Look for entry
            if z_val > entry_z:
                # Spread is too high → expect mean reversion (spread goes down)
                # → take SHORT spread = long ETH, short BTC. position = -1.
                new_position = -1
            elif z_val < -entry_z:
                # Spread too low → expect mean reversion up
                # → LONG spread = long BTC, short ETH. position = +1.
                new_position = 1
        else:
            # In a position; check exit / stop
            if abs(z_val) > stop_z:
                new_position = 0
                n_stops += 1
            elif abs(z_val) < exit_z:
                new_position = 0

        if new_position != position:
            # Pay fees: 2 sides each pay taker_bps on notional (each leg = notional)
            transition_fee = 2 * notional_usd * (taker_bps / 10_000)
            bar_fee += transition_fee
            if position == 0 and new_position != 0:
                n_trades += 1
                cur_duration = 0
            elif position != 0 and new_position == 0:
                pos_durations.append(cur_duration)
                cur_duration = 0
            position = new_position

        pnl_per_bar.append(bar_pnl - bar_fee)
        fees_per_bar.append(bar_fee)

    pnl_series = pd.Series(pnl_per_bar, index=z.index)
    fees_series = pd.Series(fees_per_bar, index=z.index)
    cum = pnl_series.cumsum()
    peak = cum.cummax()
    dd = cum - peak

    # Annualised metrics
    span_days = (z.index.max() - z.index.min()).total_seconds() / 86400
    bars_per_year = 365 * 24
    returns_per_bar = pnl_series / notional_usd
    mean_ret = returns_per_bar.mean()
    std_ret = returns_per_bar.std()
    sharpe = (mean_ret / std_ret * math.sqrt(bars_per_year)) if std_ret > 0 else float("nan")

    total_pnl = pnl_series.sum()
    total_fees = fees_series.sum()
    capital_deployed = 2 * notional_usd  # both legs at full notional
    net_apy = total_pnl / capital_deployed * (365 / span_days)

    return {
        "n_trades": n_trades,
        "n_stops": n_stops,
        "avg_duration_hours": float(np.mean(pos_durations)) if pos_durations else 0.0,
        "total_pnl": float(total_pnl),
        "total_fees": float(total_fees),
        "max_drawdown": float(dd.min()),
        "sharpe": float(sharpe),
        "span_days": float(span_days),
        "net_apy_pct": float(net_apy * 100),
        "capital_deployed": capital_deployed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc", type=Path, default=Path("data/history/BTCUSDT_1h.parquet"))
    ap.add_argument("--eth", type=Path, default=Path("data/history/ETHUSDT_1h.parquet"))
    ap.add_argument("--z-window-hours", type=int, default=168)
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=4.0)
    args = ap.parse_args()

    btc = pd.read_parquet(args.btc)["close"].rename("btc")
    eth = pd.read_parquet(args.eth)["close"].rename("eth")
    df = pd.concat([btc, eth], axis=1).dropna()
    print(f"\nData: {len(df)} aligned bars  range {df.index.min()} → {df.index.max()}\n")

    log_ratio = np.log(df["btc"]) - np.log(df["eth"])

    print("=== Mean-reversion characterization ===")
    hl = estimate_half_life(log_ratio)
    print(f"  OU half-life:      {hl:.1f} hours = {hl/24:.1f} days" if not math.isnan(hl) else "  OU half-life:      not mean-reverting (β ≥ 0)")
    adf_stat, adf_p = adf_test(log_ratio)
    if not math.isnan(adf_stat):
        print(f"  ADF stat:          {adf_stat:.3f}  p-value: {adf_p:.4f}")
        verdict = "stationary (mean-reverting)" if adf_p < 0.05 else "non-stationary (random walk)"
        print(f"  Verdict:           {verdict}")
    print(f"  log_ratio mean:    {log_ratio.mean():.4f}  std: {log_ratio.std():.4f}")
    print(f"  log_ratio range:   [{log_ratio.min():.4f}, {log_ratio.max():.4f}]")
    print()

    print("=== Z-score distribution ({}h window) ===".format(args.z_window_hours))
    z = (log_ratio - log_ratio.rolling(args.z_window_hours).mean()) / log_ratio.rolling(args.z_window_hours).std()
    z = z.dropna()
    print(f"  mean: {z.mean():+.3f}  std: {z.std():.3f}")
    print(f"  |z| > 2 share: {(z.abs() > 2).mean()*100:.2f}%")
    print(f"  |z| > 3 share: {(z.abs() > 3).mean()*100:.2f}%")
    print()

    print("=== Backtest (entry={} | exit={} | stop={}) ===".format(args.entry_z, args.exit_z, args.stop_z))
    result = backtest_pair(
        log_ratio,
        z_window_hours=args.z_window_hours,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
    )
    print(f"  n_trades:         {result['n_trades']} (of which {result['n_stops']} stops)")
    print(f"  avg duration:     {result['avg_duration_hours']:.1f} hours")
    print(f"  total gross PnL:  ${result['total_pnl']+result['total_fees']:.2f}")
    print(f"  total fees:       ${result['total_fees']:.2f}")
    print(f"  total net PnL:    ${result['total_pnl']:.2f}")
    print(f"  max drawdown:     ${result['max_drawdown']:.2f}")
    print(f"  Sharpe (annual):  {result['sharpe']:.3f}")
    print(f"  span:             {result['span_days']:.0f} days")
    print(f"  capital:          ${result['capital_deployed']:,}")
    print(f"  Net APY:          {result['net_apy_pct']:+.2f}%")


if __name__ == "__main__":
    main()
