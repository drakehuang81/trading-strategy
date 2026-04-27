"""Builds binary labels for ML training.

Two label types:
  - forward_up (default, Plan 5A): y = 1 if close[t+H] > close[t] else 0
  - triple_barrier (Plan 5C):      walk bars t+1..t+H against TP/SL barriers
                                   computed from ATR; timeout falls back to
                                   close-sign

Usage:
    python scripts/build_labels.py \
        --kline data/history/ETHUSDT_1h.parquet \
        --out   data/training/ETHUSDT_1h_labels_tb.parquet \
        --horizon 4 \
        --label-type triple_barrier \
        --tp-mult 1.5 --sl-mult 1.5 --atr-window 14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from labels.triple_barrier import triple_barrier_labels


def compute_forward_up_labels(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    future = df["close"].shift(-horizon)
    label = (future > df["close"]).astype("float")
    label[future.isna()] = np.nan
    label.name = f"y_{horizon}bar_up"
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--label-type",
                    choices=["forward_up", "triple_barrier"],
                    default="forward_up")
    ap.add_argument("--tp-mult", type=float, default=1.5,
                    help="triple_barrier: TP barrier multiplier on ATR")
    ap.add_argument("--sl-mult", type=float, default=1.5,
                    help="triple_barrier: SL barrier multiplier on ATR")
    ap.add_argument("--atr-window", type=int, default=14,
                    help="triple_barrier: ATR rolling window in bars")
    args = ap.parse_args()

    df = pd.read_parquet(args.kline).sort_index()
    if args.label_type == "forward_up":
        y = compute_forward_up_labels(df, horizon=args.horizon)
    else:  # triple_barrier
        y = triple_barrier_labels(
            df,
            horizon=args.horizon,
            tp_mult=args.tp_mult,
            sl_mult=args.sl_mult,
            atr_window=args.atr_window,
        )
    out = y.dropna().to_frame()
    out.index.name = "as_of"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"labels[{args.label_type}]: {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
