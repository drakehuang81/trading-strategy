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
