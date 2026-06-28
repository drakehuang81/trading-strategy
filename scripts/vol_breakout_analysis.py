"""Volatility-regime breakout viability analysis.

Hypothesis: in low-volatility regimes, range breakouts produce profitable
trend-follow trades. In high-vol regimes, breakouts are noise.

Strategy:
  1. Compute rolling N-bar realised vol = std(log_returns) over N bars
  2. Identify "low-vol regime" = vol below pth percentile (default 25th)
  3. Inside low-vol regime, track recent N-bar high/low
  4. Entry: price breaks N-bar high (long) or N-bar low (short)
  5. Exit: trailing stop at K × ATR (default 2.0); take profit at 3 × ATR
  6. Apply realistic taker fees

Run:
  PYTHONPATH=src python scripts/vol_breakout_analysis.py \
      --kline data/history/ETHUSDT_1h.parquet

RESULT (2026-06-28 · ETHUSDT 1h · 2024-04 -> 2026-04) — IN-SAMPLE ONLY:
  Full-sample looks tradeable: Sharpe 1.45, APY +23%, hit 57.3%. But this is
  in-sample. See vol_breakout_oos.py: out-of-sample collapses to Sharpe
  -0.25. Do NOT trust the full-sample number.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def backtest_breakout(
    df: pd.DataFrame,
    *,
    vol_window: int = 168,             # 7 days
    vol_pct_threshold: float = 25.0,   # only enter when vol < 25th pct
    breakout_window: int = 24,         # break N-bar high/low
    atr_window: int = 14,
    stop_atr_mult: float = 2.0,
    target_atr_mult: float = 3.0,
    notional_usd: float = 10_000,
    taker_bps: float = 5.0,
) -> dict:
    """Long N-bar breakout in low-vol regime; trailing stop / fixed target."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Realised vol over rolling vol_window of log-returns
    log_ret = np.log(close).diff()
    realised_vol = log_ret.rolling(vol_window).std()
    # Threshold for low-vol regime (percentile expanding so we don't peek)
    vol_threshold = realised_vol.expanding(min_periods=vol_window * 2).quantile(
        vol_pct_threshold / 100.0,
    )

    # Rolling N-bar high/low (excluding current bar to avoid lookahead)
    rolling_high = high.shift(1).rolling(breakout_window).max()
    rolling_low = low.shift(1).rolling(breakout_window).min()

    # ATR (mean H-L over atr_window)
    atr = (high - low).rolling(atr_window).mean()

    position = 0  # +1 long, -1 short, 0 flat
    entry_price = 0.0
    entry_atr = 0.0
    pnl_per_bar = []
    fees_per_bar = []
    n_trades = 0
    n_wins = 0
    n_stops = 0
    n_targets = 0

    n = len(df)
    arr_close = close.to_numpy()
    arr_high = high.to_numpy()
    arr_low = low.to_numpy()
    arr_vol = realised_vol.to_numpy()
    arr_thr = vol_threshold.to_numpy()
    arr_rhigh = rolling_high.to_numpy()
    arr_rlow = rolling_low.to_numpy()
    arr_atr = atr.to_numpy()

    for i in range(n):
        bar_pnl = 0.0
        bar_fee = 0.0

        # Mark-to-market existing position via close
        if position != 0:
            ret = (arr_close[i] - arr_close[i - 1]) / arr_close[i - 1] if i > 0 else 0.0
            bar_pnl += position * ret * notional_usd

        # Check exits on existing position (intrabar, conservative)
        if position != 0:
            stop_price = entry_price - position * stop_atr_mult * entry_atr
            target_price = entry_price + position * target_atr_mult * entry_atr
            hit_stop = (position > 0 and arr_low[i] <= stop_price) or (position < 0 and arr_high[i] >= stop_price)
            hit_target = (position > 0 and arr_high[i] >= target_price) or (position < 0 and arr_low[i] <= target_price)
            if hit_stop or hit_target:
                # Realise PnL to exit price (approx)
                exit_price = stop_price if hit_stop else target_price
                exit_ret = (exit_price - arr_close[i]) / arr_close[i]
                bar_pnl += position * exit_ret * notional_usd
                bar_fee += notional_usd * (taker_bps / 10_000)
                if hit_stop:
                    n_stops += 1
                else:
                    n_targets += 1
                    n_wins += 1
                position = 0

        # Entry logic — only when flat AND in low-vol regime
        if (
            position == 0
            and not math.isnan(arr_vol[i])
            and not math.isnan(arr_thr[i])
            and arr_vol[i] < arr_thr[i]
            and not math.isnan(arr_rhigh[i])
            and not math.isnan(arr_atr[i])
            and arr_atr[i] > 0
        ):
            # Long breakout: this bar's high crosses prior N-bar high
            if arr_high[i] > arr_rhigh[i]:
                entry_price = arr_rhigh[i]  # assume fill at the breakout level
                entry_atr = arr_atr[i]
                position = 1
                bar_fee += notional_usd * (taker_bps / 10_000)
                n_trades += 1
            elif arr_low[i] < arr_rlow[i]:
                entry_price = arr_rlow[i]
                entry_atr = arr_atr[i]
                position = -1
                bar_fee += notional_usd * (taker_bps / 10_000)
                n_trades += 1

        pnl_per_bar.append(bar_pnl - bar_fee)
        fees_per_bar.append(bar_fee)

    pnl_series = pd.Series(pnl_per_bar, index=df.index)
    fees_series = pd.Series(fees_per_bar, index=df.index)
    cum = pnl_series.cumsum()
    peak = cum.cummax()
    dd = cum - peak

    span_days = (df.index.max() - df.index.min()).total_seconds() / 86400
    bars_per_year = 365 * 24
    returns_per_bar = pnl_series / notional_usd
    mean_ret = returns_per_bar.mean()
    std_ret = returns_per_bar.std()
    sharpe = (mean_ret / std_ret * math.sqrt(bars_per_year)) if std_ret > 0 else float("nan")

    total_pnl = pnl_series.sum()
    total_fees = fees_series.sum()
    net_apy = total_pnl / notional_usd * (365 / span_days)

    return {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_stops": n_stops,
        "n_targets": n_targets,
        "hit_rate_pct": (n_wins / n_trades * 100) if n_trades else 0.0,
        "total_pnl": float(total_pnl),
        "total_fees": float(total_fees),
        "max_drawdown": float(dd.min()),
        "sharpe": float(sharpe),
        "span_days": float(span_days),
        "net_apy_pct": float(net_apy * 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", type=Path, default=Path("data/history/ETHUSDT_1h.parquet"))
    ap.add_argument("--vol-pct", type=float, default=25.0,
                    help="vol percentile threshold (entries only below)")
    ap.add_argument("--breakout-window", type=int, default=24)
    ap.add_argument("--stop-atr", type=float, default=2.0)
    ap.add_argument("--target-atr", type=float, default=3.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.kline).sort_index()
    print(f"\nData: {len(df)} bars  range {df.index.min()} → {df.index.max()}\n")

    print(f"=== Backtest (vol<{args.vol_pct}pct | break {args.breakout_window}h | "
          f"stop {args.stop_atr}×ATR | target {args.target_atr}×ATR) ===")
    r = backtest_breakout(
        df,
        vol_pct_threshold=args.vol_pct,
        breakout_window=args.breakout_window,
        stop_atr_mult=args.stop_atr,
        target_atr_mult=args.target_atr,
    )
    print(f"  n_trades:         {r['n_trades']} ({r['n_targets']} targets, {r['n_stops']} stops)")
    print(f"  hit rate:         {r['hit_rate_pct']:.1f}%")
    print(f"  total gross PnL:  ${r['total_pnl']+r['total_fees']:.2f}")
    print(f"  total fees:       ${r['total_fees']:.2f}")
    print(f"  total net PnL:    ${r['total_pnl']:.2f}")
    print(f"  max drawdown:     ${r['max_drawdown']:.2f}")
    print(f"  Sharpe (annual):  {r['sharpe']:.3f}")
    print(f"  Net APY (on $10k):{r['net_apy_pct']:+.2f}%")
    print()


if __name__ == "__main__":
    main()
