"""Step 0 — de-risk: report the real schema/cadence of downloaded data.

For bookDepth specifically, the presence of a 'percentage' column means
Binance ships *percentage-distance depth* (not raw L2 levels), which decides
how Phase 2 defines depth imbalance / book slope.
"""
from __future__ import annotations

from typing import Any

import polars as pl


def summarize_schema(df: pl.DataFrame, *, ts_col: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": df.height,
        "columns": df.columns,
        "dtypes": {c: str(t) for c, t in zip(df.columns, df.dtypes)},
    }
    if ts_col in df.columns and df.height > 1:
        col = df[ts_col]
        if col.dtype == pl.String:
            # bookDepth ships ts as "YYYY-MM-DD HH:MM:SS" — parse to datetime
            col = col.str.to_datetime(strict=False)
        if col.dtype.is_temporal():
            gaps_ms = col.sort().diff().drop_nulls().dt.total_milliseconds()
            summary["median_gap_ms"] = int(gaps_ms.median())
        else:
            gaps = col.sort().diff().drop_nulls()
            summary["median_gap_ms"] = int(gaps.median())
    else:
        summary["median_gap_ms"] = None
    if "percentage" in df.columns:
        summary["distinct_percentage"] = sorted(
            df["percentage"].unique().to_list()
        )
    return summary
