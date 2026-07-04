"""Tests for the book-stream recorder's pure parts (routing + append + rotation)."""
import gzip
import json
import time
from pathlib import Path

from scripts.record_book import (
    append_jsonl,
    gzip_stale_files,
    route_message,
    split_streams,
)


def test_split_streams_by_shard():
    # Binance shards: aggTrade only on "market"; book streams only on legacy
    groups = split_streams(["BTCUSDT", "ETHUSDT"], ["bookTicker", "aggTrade", "depth5@500ms"])
    assert groups["market"] == ["btcusdt@aggTrade", "ethusdt@aggTrade"]
    assert groups[None] == [
        "btcusdt@bookTicker", "ethusdt@bookTicker",
        "btcusdt@depth5@500ms", "ethusdt@depth5@500ms",
    ]


def test_route_message_book_ticker():
    msg = {
        "stream": "btcusdt@bookTicker",
        "data": {"e": "bookTicker", "E": 1700000000123, "s": "BTCUSDT",
                 "b": "100.0", "B": "5", "a": "100.5", "A": "3"},
    }
    kind, symbol, ts = route_message(msg)
    assert kind == "bookTicker"
    assert symbol == "BTCUSDT"
    assert ts == 1700000000123


def test_route_message_depth_stream_safe_dir_and_symbol_fallback():
    # partial depth: kind contains "@" (speed suffix) -> path-safe dir name;
    # payload without "s" -> symbol falls back to the stream prefix
    msg = {"stream": "ethusdt@depth5@500ms",
           "data": {"E": 1700000000500, "bids": [], "asks": []}}
    kind, symbol, ts = route_message(msg)
    assert kind == "depth5_500ms"
    assert symbol == "ETHUSDT"
    assert ts == 1700000000500


def test_append_jsonl_daily_rollover(tmp_path: Path):
    d1 = 1700000000000            # 2023-11-14 UTC
    d2 = d1 + 86_400_000          # next UTC day
    p1 = append_jsonl(tmp_path, "bookTicker", "BTCUSDT", d1, {"x": 1})
    p2 = append_jsonl(tmp_path, "bookTicker", "BTCUSDT", d2, {"x": 2})
    assert p1 != p2
    assert p1.parent == tmp_path / "bookTicker" / "BTCUSDT"
    assert json.loads(p1.read_text().strip()) == {"x": 1}


def test_gzip_stale_files_compresses_past_days_only(tmp_path: Path):
    d1 = 1700000000000            # 2023-11-14 UTC
    d2 = d1 + 86_400_000          # 2023-11-15 UTC
    old = append_jsonl(tmp_path, "aggTrade", "BTCUSDT", d1, {"x": 1})
    today = append_jsonl(tmp_path, "aggTrade", "BTCUSDT", d2, {"x": 2})
    # pretend an hour has passed so `old` is outside the write-grace window
    done = gzip_stale_files(tmp_path, today="2023-11-15", now=time.time() + 3600)
    assert done == [old.with_name(old.name + ".gz")]
    assert not old.exists()                      # original removed
    assert today.exists()                        # today's file untouched
    with gzip.open(done[0], "rt") as fh:
        assert json.loads(fh.read().strip()) == {"x": 1}


def test_gzip_stale_files_skips_recently_written(tmp_path: Path):
    # a just-rolled-over writer may still flush late events: fresh mtime -> skip
    d1 = 1700000000000
    old = append_jsonl(tmp_path, "aggTrade", "BTCUSDT", d1, {"x": 1})
    assert gzip_stale_files(tmp_path, today="2023-11-15") == []
    assert old.exists()
