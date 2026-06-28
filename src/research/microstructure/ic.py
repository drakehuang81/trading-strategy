"""Information Coefficient (Spearman) per horizon, and quantile layering.

IC magnitude alone can mislead; a real edge also shows monotone mean forward
return across signal quantiles (checked by quantile_layering).
"""
from __future__ import annotations

import polars as pl
from scipy.stats import spearmanr


def compute_ic(
    df: pl.DataFrame, *, signal_col: str, horizon_cols: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for hcol in horizon_cols:
        pair = df.select([signal_col, hcol]).drop_nulls()
        if pair.height < 3:
            out[hcol] = float("nan")
            continue
        rho, _ = spearmanr(pair[signal_col].to_numpy(), pair[hcol].to_numpy())
        out[hcol] = float(rho)
    return out


def quantile_layering(
    df: pl.DataFrame, *, signal_col: str, horizon_col: str, n_buckets: int = 5
) -> pl.DataFrame:
    """Mean forward return per signal quantile bucket (ascending by signal)."""
    pair = df.select([signal_col, horizon_col]).drop_nulls()
    pair = pair.with_columns(
        pl.col(signal_col).qcut(n_buckets, labels=[str(i) for i in range(n_buckets)])
        .alias("bucket")
    )
    return (
        pair.group_by("bucket")
        .agg(pl.col(horizon_col).mean().alias("mean_fwd"))
        .sort("bucket")
    )
