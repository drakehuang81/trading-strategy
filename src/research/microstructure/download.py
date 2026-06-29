"""Downloader for data.binance.vision USD-M futures order book archives.

Raw CSV schemas vary by data type and are confirmed by the Step-0 probe
(schema_probe.py). COLUMN_MAP below holds the *expected* raw->normalized
mapping; if Step 0 reveals different raw column names, update COLUMN_MAP
only — signal/align/ic code consumes the normalized schema and stays put.
"""
from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import polars as pl
import requests

BASE_URL = "https://data.binance.vision/data/futures/um/daily"
DATA_TYPES = ("bookTicker", "bookDepth", "aggTrades")

# Expected raw bookTicker columns (confirmed/adjusted by Step 0).
# Normalized target: ts, bid_price, bid_qty, ask_price, ask_qty
BOOK_TICKER_MAP = {
    "best_bid_price": "bid_price",
    "best_bid_qty": "bid_qty",
    "best_ask_price": "ask_price",
    "best_ask_qty": "ask_qty",
}
BOOK_TICKER_TS_COL = "transaction_time"  # epoch ms


def build_url(symbol: str, data_type: str, date: dt.date) -> str:
    if data_type not in DATA_TYPES:
        raise ValueError(f"unknown data_type {data_type!r}")
    fname = f"{symbol}-{data_type}-{date.isoformat()}.zip"
    return f"{BASE_URL}/{data_type}/{symbol}/{fname}"


def download_zip(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    return dest


def extract_zip_to_parquet(zip_path: Path, parquet_path: Path) -> Path:
    """Extract the single CSV inside a Binance daily zip into parquet."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected 1 csv in {zip_path}, found {names}")
        with zf.open(names[0]) as fh:
            df = pl.read_csv(fh)
    df.write_parquet(parquet_path)
    return parquet_path


def load_book_ticker(parquet_path: Path) -> pl.DataFrame:
    """Load + normalize a bookTicker parquet to the standard schema."""
    df = pl.read_parquet(parquet_path)
    df = df.rename(BOOK_TICKER_MAP)
    df = df.with_columns(
        pl.from_epoch(pl.col(BOOK_TICKER_TS_COL), time_unit="ms").alias("ts")
    )
    return df.select(["ts", "bid_price", "bid_qty", "ask_price", "ask_qty"])


def load_book_depth(parquet_path: Path) -> pl.DataFrame:
    """Load + normalize a bookDepth parquet (long form: one row per level).

    Output: ts(Datetime), percentage, depth, notional. Timestamp is parsed
    from the raw "YYYY-MM-DD HH:MM:SS" string. 12 symmetric percentage levels
    (+/-0.2/1/2/3/4/5) share each ts.
    """
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime(strict=False).alias("ts")
    )
    return df.select(["ts", "percentage", "depth", "notional"])
