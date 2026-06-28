"""Align event-level book onto a uniform time grid and build forward returns.

No-lookahead: grid mid uses a BACKWARD as-of join (latest book at/<= grid t);
forward returns use strictly future grid points.
"""
from __future__ import annotations

import polars as pl


def to_mid_grid(book: pl.DataFrame, *, every: str = "1s") -> pl.DataFrame:
    """Return a uniform grid (ts, mid) by backward as-of join onto the book."""
    b = book.with_columns(
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid")
    ).select(["ts", "mid"]).sort("ts")
    start, end = b["ts"].min(), b["ts"].max()
    grid = pl.datetime_range(start, end, every, eager=True).alias("ts").to_frame()
    return grid.join_asof(b, on="ts", strategy="backward")


def forward_returns(grid: pl.DataFrame, *, horizons_secs: list[int]) -> pl.DataFrame:
    """Add fwd_<h>s columns: mid_{t+h}/mid_t - 1. Grid must be uniform 1s."""
    out = grid
    for h in horizons_secs:
        out = out.with_columns(
            (pl.col("mid").shift(-h) / pl.col("mid") - 1).alias(f"fwd_{h}s")
        )
    return out
