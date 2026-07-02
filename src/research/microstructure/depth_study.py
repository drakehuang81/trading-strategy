"""Depth-imbalance @ 1h validation dataset + OOS split.

build_hourly: bookDepth (long) -> hourly mean depth_imbalance, aligned to 1h
kline close, with forward 1h return. time_split: strict time-ordered holdout.
"""
from __future__ import annotations

import polars as pl

from research.microstructure.signals import depth_imbalance


def build_hourly(depth: pl.DataFrame, klines_1h: pl.DataFrame) -> pl.DataFrame:
    """(depth long-form, klines (hour,close)) -> (hour, di, close, fwd_1h).

    di = mean per-snapshot depth_imbalance within the hour; fwd_1h = next
    hour's close / this close - 1. Rows without a forward close are dropped.
    """
    di = depth_imbalance(depth)  # (ts, depth_imbalance) per snapshot
    di_1h = (
        di.with_columns(pl.col("ts").dt.truncate("1h").alias("hour"))
        .group_by("hour")
        .agg(pl.col("depth_imbalance").mean().alias("di"))
        .sort("hour")
    )
    k = klines_1h.sort("hour").with_columns(
        (pl.col("close").shift(-1) / pl.col("close") - 1).alias("fwd_1h")
    )
    return di_1h.join(k, on="hour", how="inner").drop_nulls(["di", "fwd_1h"]).sort("hour")


def time_split(ds: pl.DataFrame, *, train_frac: float = 0.7) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Strict time-ordered holdout: earliest train_frac rows train, rest test."""
    ds = ds.sort("hour")
    cut = int(ds.height * train_frac)
    return ds.head(cut), ds.tail(ds.height - cut)
