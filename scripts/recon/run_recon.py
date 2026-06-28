"""End-to-end thin recon pipeline: book -> qi -> grid -> IC -> markdown.

Phase 1 wires the full flow with the single L1 signal. Phase 2 extends the
signal set once Step 0 confirms schemas.

Run (after download):
    PYTHONPATH=src venv/bin/python -m scripts.recon.run_recon \
        --book data/orderbook/ETHUSDT-bookTicker-2026-06-01.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from research.microstructure.align import forward_returns, to_mid_grid
from research.microstructure.download import load_book_ticker
from research.microstructure.ic import compute_ic
from research.microstructure.report import render_ic_markdown
from research.microstructure.signals import queue_imbalance


def recon_from_book(
    book: pl.DataFrame, *, horizons_secs: list[int]
) -> tuple[str, dict[str, float]]:
    """Pure pipeline: normalized book df -> (markdown, ic dict)."""
    grid = to_mid_grid(book, every="1s")
    grid = forward_returns(grid, horizons_secs=horizons_secs)
    qi = queue_imbalance(book)
    # as-of attach qi onto the grid (backward), then IC
    merged = grid.join_asof(qi.sort("ts"), on="ts", strategy="backward")
    hcols = [f"fwd_{h}s" for h in horizons_secs]
    ic = compute_ic(merged, signal_col="qi", horizon_cols=hcols)
    md = render_ic_markdown({"qi": ic}, n_tests=len(hcols))
    return md, ic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="normalized-or-raw bookTicker parquet")
    ap.add_argument("--horizons", default="1,5,10,30,60,300,900,3600")
    ap.add_argument("--out", default="docs/superpowers/recon/phase1_ic.md")
    args = ap.parse_args()

    book = load_book_ticker(Path(args.book))
    horizons = [int(x) for x in args.horizons.split(",")]
    md, ic = recon_from_book(book, horizons_secs=horizons)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
