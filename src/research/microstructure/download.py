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


def load_agg_trades(parquet_path: Path) -> pl.DataFrame:
    """Load + normalize aggTrades, splitting volume into taker buy/sell.

    is_buyer_maker=True  => buyer is maker => taker SOLD  -> taker_sell_qty
    is_buyer_maker=False => buyer is taker => taker BOUGHT -> taker_buy_qty
    Output: ts(Datetime), price, qty, taker_buy_qty, taker_sell_qty.
    """
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(
        pl.from_epoch(pl.col("transact_time"), time_unit="ms").alias("ts"),
        pl.col("quantity").alias("qty"),
    )
    df = df.with_columns(
        pl.when(pl.col("is_buyer_maker"))
        .then(0.0).otherwise(pl.col("qty")).alias("taker_buy_qty"),
        pl.when(pl.col("is_buyer_maker"))
        .then(pl.col("qty")).otherwise(0.0).alias("taker_sell_qty"),
    )
    return df.select(["ts", "price", "qty", "taker_buy_qty", "taker_sell_qty"])


KLINES_1H_BASE = "https://data.binance.vision/data/futures/um/daily/klines"


def klines_1h_url(symbol: str, date: dt.date) -> str:
    return f"{KLINES_1H_BASE}/{symbol}/1h/{symbol}-1h-{date.isoformat()}.zip"


def load_klines_1h(csv_or_parquet: Path) -> pl.DataFrame:
    """Load headerless Binance 1h klines CSV → (hour, close).

    Columns are positional: col 0 = open_time (epoch ms), col 4 = close.
    """
    df = pl.read_csv(csv_or_parquet, has_header=False)
    open_time, close = df.columns[0], df.columns[4]
    return df.select(
        pl.from_epoch(pl.col(open_time).cast(pl.Int64), time_unit="ms").alias("hour"),
        pl.col(close).cast(pl.Float64).alias("close"),
    )
