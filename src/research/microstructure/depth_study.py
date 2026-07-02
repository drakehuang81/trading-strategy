"""Depth-imbalance @ 1h validation dataset + OOS split.

build_hourly: bookDepth (long) -> hourly mean depth_imbalance, aligned to 1h
kline close, with forward 1h return. time_split: strict time-ordered holdout.
"""
from __future__ import annotations

import polars as pl
from scipy.stats import spearmanr

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


def ic(ds: pl.DataFrame, col: str, *, target: str = "fwd_1h") -> float:
    p = ds.select([col, target]).drop_nulls().filter(
        pl.col(col).is_finite() & pl.col(target).is_finite()
    )
    if p.height < 3:
        return float("nan")
    return float(spearmanr(p[col].to_numpy(), p[target].to_numpy())[0])


def decile_edge_bps(ds: pl.DataFrame, *, n_buckets: int = 10, taker_std_bps: float = 8.0) -> dict:
    """Mean fwd_1h (bps) per di quantile; gross = best of top-long/bottom-short."""
    labels = [str(i).zfill(len(str(n_buckets - 1))) for i in range(n_buckets)]
    g = (
        ds.drop_nulls(["di", "fwd_1h"])
        .with_columns(pl.col("di").qcut(n_buckets, labels=labels, allow_duplicates=True).alias("b"))
        .group_by("b").agg((pl.col("fwd_1h").mean() * 1e4).alias("r")).sort("b")
    )
    means = g["r"].to_list()
    gross = max(means[-1], -means[0])
    return {"means": means, "gross_bps": gross, "net_std_taker_bps": gross - taker_std_bps}


def newey_west_tstat(returns, *, lags: int = 5) -> float:
    """HAC t-stat of the mean of a return series (autocorrelation-robust)."""
    import numpy as np
    import statsmodels.api as sm
    y = np.asarray(returns, dtype=float)
    x = np.ones_like(y)
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.tvalues[0])


def build_hourly_cross(
    depth_lead: pl.DataFrame,
    klines_lead: pl.DataFrame,
    klines_lag: pl.DataFrame,
) -> pl.DataFrame:
    """Cross-asset hourly dataset: lead symbol's book vs lag symbol's return.

    Output (hour, di, past_1h_lead, fwd_1h, past_1h): di = LEAD symbol's
    hourly mean depth imbalance; past_1h_lead = lead's own trailing 1h return;
    fwd_1h / past_1h = LAG symbol's forward / trailing 1h return. fwd_1h (lag)
    is the prediction target.
    """
    lead = build_hourly(depth_lead, klines_lead).sort("hour")
    lead = lead.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("past_1h_lead")
    ).select(["hour", "di", "past_1h_lead"])
    lag = klines_lag.sort("hour").with_columns(
        (pl.col("close").shift(-1) / pl.col("close") - 1).alias("fwd_1h"),
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("past_1h"),
    ).select(["hour", "fwd_1h", "past_1h"])
    return (
        lead.join(lag, on="hour", how="inner")
        .drop_nulls(["di", "fwd_1h"])
        .sort("hour")
    )
