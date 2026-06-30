"""Multi-signal recon driver: attach each signal series onto a mid grid via
backward as-of join, compute forward returns, then IC per (signal x horizon).

Each signal df is (ts, <name>); the grid is (ts, mid) on a uniform 1s axis.
"""
from __future__ import annotations

import polars as pl

from research.microstructure.align import forward_returns
from research.microstructure.ic import compute_ic
from research.microstructure.report import render_ic_markdown


def recon_multi(
    grid: pl.DataFrame,
    signals: dict[str, pl.DataFrame],
    *,
    horizons_secs: list[int],
) -> tuple[str, dict[str, dict[str, float]]]:
    """grid: (ts, mid). signals: name -> (ts, name). Returns (markdown, ic_by_signal)."""
    g = forward_returns(grid.sort("ts"), horizons_secs=horizons_secs)
    hcols = [f"fwd_{h}s" for h in horizons_secs]
    ic_by_signal: dict[str, dict[str, float]] = {}
    for name, sig in signals.items():
        merged = g.join_asof(sig.sort("ts"), on="ts", strategy="backward")
        ic_by_signal[name] = compute_ic(merged, signal_col=name, horizon_cols=hcols)
    n_tests = len(signals) * len(hcols)
    md = render_ic_markdown(ic_by_signal, n_tests=n_tests)
    return md, ic_by_signal
