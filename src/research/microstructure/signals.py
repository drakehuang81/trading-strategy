"""Microstructure signals. Phase 1 ships L1 queue imbalance only.

All signals are point-in-time: the value at row t uses only data at/before t.
"""
from __future__ import annotations

import polars as pl


def queue_imbalance(book: pl.DataFrame) -> pl.DataFrame:
    """L1 queue imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty).

    Input columns: ts, bid_qty, ask_qty (others ignored).
    Output: ts, qi  (qi is null when total depth is zero).
    """
    total = pl.col("bid_qty") + pl.col("ask_qty")
    qi = (
        pl.when(total > 0)
        .then((pl.col("bid_qty") - pl.col("ask_qty")) / total)
        .otherwise(None)
        .alias("qi")
    )
    return book.select(["ts", qi])


def depth_imbalance(book_depth: pl.DataFrame) -> pl.DataFrame:
    """Per-snapshot depth imbalance from percentage-distance depth.

    bid = sum(depth where percentage < 0), ask = sum(depth where percentage > 0).
    DI = (bid - ask) / (bid + ask); null when total is zero.
    Input: ts, percentage, depth (long form). Output: ts, depth_imbalance.
    """
    g = book_depth.group_by("ts").agg(
        pl.col("depth").filter(pl.col("percentage") < 0).sum().alias("bid"),
        pl.col("depth").filter(pl.col("percentage") > 0).sum().alias("ask"),
    )
    total = pl.col("bid") + pl.col("ask")
    di = (
        pl.when(total > 0)
        .then((pl.col("bid") - pl.col("ask")) / total)
        .otherwise(None)
        .alias("depth_imbalance")
    )
    return g.select(["ts", di]).sort("ts")


def book_slope(
    book_depth: pl.DataFrame, *, near_max: float = 1.0, far_min: float = 3.0
) -> pl.DataFrame:
    """Liquidity concentration: log(far_depth / near_depth) per snapshot.

    near = sum(depth where |percentage| <= near_max),
    far  = sum(depth where |percentage| >= far_min).
    Positive slope = depth concentrated away from mid. Null if either is 0.
    Input: ts, percentage, depth. Output: ts, book_slope.
    """
    absp = pl.col("percentage").abs()
    g = book_depth.group_by("ts").agg(
        pl.col("depth").filter(absp <= near_max).sum().alias("near"),
        pl.col("depth").filter(absp >= far_min).sum().alias("far"),
    )
    slope = (
        pl.when((pl.col("near") > 0) & (pl.col("far") > 0))
        .then((pl.col("far") / pl.col("near")).log())
        .otherwise(None)
        .alias("book_slope")
    )
    return g.select(["ts", slope]).sort("ts")
