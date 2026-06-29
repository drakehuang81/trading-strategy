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


def ofi(book: pl.DataFrame, *, window: int = 50) -> pl.DataFrame:
    """Order Flow Imbalance (Cont, Kukanov & Stoikov 2014) from best bid/ask.

    Per update n (vs n-1):
      bid term  = bid_qty_n                if bid_price_n  > bid_price_{n-1}
                  bid_qty_n - bid_qty_{n-1} if equal
                  -bid_qty_{n-1}           if bid_price_n  < bid_price_{n-1}
      ask term  = -ask_qty_n               if ask_price_n  > ask_price_{n-1}
                  ask_qty_n - ask_qty_{n-1} if equal
                  ask_qty_{n-1}            if ask_price_n  < ask_price_{n-1}
      e_n = bid term + ask term
    ofi = rolling_sum(e_n, window). First row is null (no previous update).
    Input: ts, bid_price, bid_qty, ask_price, ask_qty. Output: ts, ofi.
    """
    b = book.sort("ts")
    pbid, qbid = pl.col("bid_price"), pl.col("bid_qty")
    pask, qask = pl.col("ask_price"), pl.col("ask_qty")
    pbid0, qbid0 = pbid.shift(1), qbid.shift(1)
    pask0, qask0 = pask.shift(1), qask.shift(1)

    bid_term = (
        pl.when(pbid > pbid0).then(qbid)
        .when(pbid == pbid0).then(qbid - qbid0)
        .otherwise(-qbid0)
    )
    ask_term = (
        pl.when(pask > pask0).then(-qask)
        .when(pask == pask0).then(qask - qask0)
        .otherwise(qask0)
    )
    b = b.with_columns((bid_term + ask_term).alias("_e"))
    # rolling_sum over a window containing nulls can return null; fill the
    # per-update term with 0 for the sum, but keep the first row's ofi null
    # (no previous update exists there).
    b = b.with_columns(pl.col("_e").fill_null(0.0).alias("_e_filled"))
    out = b.with_columns(
        pl.col("_e_filled").rolling_sum(window_size=window, min_samples=1).alias("ofi")
    )
    out = out.with_columns(
        pl.when(pl.col("_e").is_null()).then(None).otherwise(pl.col("ofi")).alias("ofi")
    )
    return out.select(["ts", "ofi"])
