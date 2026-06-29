import datetime as dt
import zipfile
from pathlib import Path

import polars as pl

from research.microstructure.download import (
    build_url,
    extract_zip_to_parquet,
    load_book_ticker,
)


def test_build_url_book_ticker():
    url = build_url("ETHUSDT", "bookTicker", dt.date(2026, 1, 15))
    assert url == (
        "https://data.binance.vision/data/futures/um/daily/"
        "bookTicker/ETHUSDT/ETHUSDT-bookTicker-2026-01-15.zip"
    )


def test_extract_zip_to_parquet_roundtrip(tmp_path: Path):
    csv_bytes = (
        b"update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,"
        b"transaction_time,event_time\n"
        b"1,100.0,5.0,100.5,3.0,1700000000000,1700000000001\n"
    )
    zip_path = tmp_path / "ETHUSDT-bookTicker-2026-01-15.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ETHUSDT-bookTicker-2026-01-15.csv", csv_bytes)

    out = extract_zip_to_parquet(zip_path, tmp_path / "out.parquet")
    df = pl.read_parquet(out)
    assert df.height == 1
    assert "best_bid_price" in df.columns


def test_load_book_ticker_normalizes_columns(tmp_path: Path):
    raw = pl.DataFrame({
        "update_id": [1],
        "best_bid_price": [100.0],
        "best_bid_qty": [5.0],
        "best_ask_price": [100.5],
        "best_ask_qty": [3.0],
        "transaction_time": [1700000000000],
        "event_time": [1700000000001],
    })
    p = tmp_path / "bt.parquet"
    raw.write_parquet(p)

    df = load_book_ticker(p)
    assert df.columns == ["ts", "bid_price", "bid_qty", "ask_price", "ask_qty"]
    assert df["bid_price"][0] == 100.0
    assert df["ts"].dtype == pl.Datetime


def test_load_book_depth_normalizes(tmp_path: Path):
    raw = pl.DataFrame({
        "timestamp": ["2026-06-01 00:00:07", "2026-06-01 00:00:07"],
        "percentage": [-1.0, 1.0],
        "depth": [42881.4, 28062.9],
        "notional": [8.5e7, 5.6e7],
    })
    p = tmp_path / "bd.parquet"
    raw.write_parquet(p)
    from research.microstructure.download import load_book_depth
    df = load_book_depth(p)
    assert df.columns == ["ts", "percentage", "depth", "notional"]
    assert df["ts"].dtype == pl.Datetime
    assert df.height == 2


def test_load_agg_trades_normalizes_and_signs_taker(tmp_path: Path):
    raw = pl.DataFrame({
        "agg_trade_id": [1, 2],
        "price": [2006.3, 2006.4],
        "quantity": [0.36, 1.0],
        "first_trade_id": [10, 11],
        "last_trade_id": [10, 12],
        "transact_time": [1780272000067, 1780272000357],
        "is_buyer_maker": [True, False],
    })
    p = tmp_path / "agg.parquet"
    raw.write_parquet(p)
    from research.microstructure.download import load_agg_trades
    df = load_agg_trades(p)
    assert df.columns == ["ts", "price", "qty", "taker_buy_qty", "taker_sell_qty"]
    assert df["ts"].dtype == pl.Datetime
    # is_buyer_maker=True -> taker SOLD: taker_sell_qty=0.36, taker_buy_qty=0
    assert df["taker_sell_qty"][0] == 0.36
    assert df["taker_buy_qty"][0] == 0.0
    # is_buyer_maker=False -> taker BOUGHT
    assert df["taker_buy_qty"][1] == 1.0
    assert df["taker_sell_qty"][1] == 0.0
