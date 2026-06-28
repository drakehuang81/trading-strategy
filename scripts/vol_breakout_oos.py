"""OOS check for vol-breakout strategy.

Slice the kline parquet into TRAIN (first 70%) and TEST (last 30%).
Run the same strategy + same params on each. Compare metrics.

If TEST APY is dramatically worse than TRAIN APY, strategy is in-sample
overfit. If TEST APY is in the same ballpark (within ~50%), evidence
of robustness.

RESULT (2026-06-28 · ETHUSDT 1h · 70/30 split) — DEAD END (overfit):
  TRAIN Sharpe +2.15 / APY +31% / hit 58.8%  ->  TEST Sharpe -0.25 / APY
  -4.3% / hit 50.8% (coin-flip). Classic in-sample overfit; no real edge
  out-of-sample.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vol_breakout_analysis import backtest_breakout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", type=Path, default=Path("data/history/ETHUSDT_1h.parquet"))
    ap.add_argument("--train-frac", type=float, default=0.70)
    args = ap.parse_args()

    df = pd.read_parquet(args.kline).sort_index()
    cut = int(len(df) * args.train_frac)
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    print(f"\n{args.kline.name}: {len(df)} bars total")
    print(f"  TRAIN: {len(train)} bars  {train.index[0].date()} → {train.index[-1].date()}")
    print(f"  TEST:  {len(test)} bars  {test.index[0].date()} → {test.index[-1].date()}\n")

    for label, slice_df in [("TRAIN", train), ("TEST", test)]:
        r = backtest_breakout(slice_df)
        print(f"=== {label} ===")
        print(f"  trades: {r['n_trades']:>4} | hit%: {r['hit_rate_pct']:>5.1f} | "
              f"net PnL: ${r['total_pnl']:>+8.2f} | Sharpe: {r['sharpe']:>+.2f} | "
              f"APY: {r['net_apy_pct']:>+7.2f}% | maxDD: ${r['max_drawdown']:>+8.2f}")


if __name__ == "__main__":
    main()
