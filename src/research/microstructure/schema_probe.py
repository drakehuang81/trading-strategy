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
        gaps = df[ts_col].sort().diff().drop_nulls()
        summary["median_gap_ms"] = int(gaps.median())
    else:
        summary["median_gap_ms"] = None
    if "percentage" in df.columns:
        summary["distinct_percentage"] = sorted(
            df["percentage"].unique().to_list()
        )
    return summary
