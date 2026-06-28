"""Step 0 CLI: download 1 day of each data type for one symbol, print schema.

Run (network — do this once manually, not in CI):
    PYTHONPATH=src venv/bin/python -m scripts.recon.probe_schema --symbol ETHUSDT --date 2026-06-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import polars as pl

from research.microstructure.download import (
    DATA_TYPES,
    build_url,
    download_zip,
    extract_zip_to_parquet,
)
from research.microstructure.schema_probe import summarize_schema

_TS_GUESS = {"bookTicker": "transaction_time", "bookDepth": "timestamp", "aggTrades": "transact_time"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out-dir", default="data/orderbook/_probe")
    args = ap.parse_args()
    date = dt.date.fromisoformat(args.date)
    out = Path(args.out_dir)

    for dtype in DATA_TYPES:
        url = build_url(args.symbol, dtype, date)
        zip_path = out / f"{args.symbol}-{dtype}-{date}.zip"
        pq = out / f"{args.symbol}-{dtype}-{date}.parquet"
        try:
            download_zip(url, zip_path)
            extract_zip_to_parquet(zip_path, pq)
            df = pl.read_parquet(pq)
            ts_col = _TS_GUESS.get(dtype, df.columns[0])
            print(f"\n=== {dtype} ===")
            print(json.dumps(summarize_schema(df, ts_col=ts_col), indent=2, default=str))
            print("head:", df.head(3).to_dicts())
        except Exception as e:  # noqa: BLE001 — probe should report, not crash
            print(f"\n=== {dtype} === FAILED: {e}")


if __name__ == "__main__":
    main()
