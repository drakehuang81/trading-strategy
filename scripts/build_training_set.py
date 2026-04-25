"""Computes flat features for every bar of a kline parquet (Plan 5A Task 6).

Reuses XGBPredictor._flatten so column names match what XGBPredictor.load
expects at inference time.

Usage:
    python scripts/build_training_set.py \
        --kline data/history/ETHUSDT_1h.parquet \
        --out   data/training/ETHUSDT_1h_features.parquet \
        --warmup 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features.registry import FeatureRegistry, build_default_registry
from models.xgb_predictor import _flatten


def build_training_set(
    df: pd.DataFrame,
    registry: FeatureRegistry,
    warmup_bars: int = 200,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    index: list = []
    for i, ts in enumerate(df.index):
        if i < warmup_bars:
            continue
        feats = registry.compute_all(df, as_of=ts)
        flat = _flatten(feats)
        rows.append(flat)
        index.append(ts)
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="as_of"))
    return out.fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--symbol", default="ETHUSDT")
    args = ap.parse_args()

    df = pd.read_parquet(args.kline)
    df = df.sort_index()
    reg = build_default_registry(symbol=args.symbol)
    out = build_training_set(df, reg, warmup_bars=args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"training set: {len(out)} rows x {len(out.columns)} cols -> {args.out}")


if __name__ == "__main__":
    main()
