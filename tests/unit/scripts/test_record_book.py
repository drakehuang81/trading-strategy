"""Tests for the book-stream recorder's pure parts (routing + append)."""
import json
from pathlib import Path

from scripts.record_book import append_jsonl, route_message


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
