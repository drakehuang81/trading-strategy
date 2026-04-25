"""Builds binary forward-return labels (Plan 5A Task 7).

Label rule: y_<H>bar_up = 1 if close[t+H] > close[t] else 0.
Horizon defaults to 4 (spec PredictionBundle.horizon_bars).

Usage:
    python scripts/build_labels.py \
        --kline data/history/ETHUSDT_1h.parquet \
        --out   data/training/ETHUSDT_1h_labels.parquet \
        --horizon 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
    args = ap.parse_args()

    df = pd.read_parquet(args.kline).sort_index()
    y = compute_forward_up_labels(df, horizon=args.horizon)
    out = y.dropna().to_frame()
    out.index.name = "as_of"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"labels: {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
