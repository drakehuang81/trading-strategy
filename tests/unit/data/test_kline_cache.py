from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from data.kline_cache import RollingKlineCache


def _row(ts: datetime, close: float, hi: float | None = None, lo: float | None = None) -> dict:
    return {
        "open": close, "high": hi or close, "low": lo or close,
        "close": close, "volume": 1.0,
    }


def _df(rows: list[tuple[datetime, dict]]) -> pd.DataFrame:
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
    df.index.name = "open_time"
    return df


def test_returns_none_when_empty():
    cache = RollingKlineCache(max_bars=200)
    assert cache.last_close("ETHUSDT", "1h") is None
    assert cache.atr("ETHUSDT", "1h", n=14) is None


def test_ingest_keeps_last_max_bars():
    cache = RollingKlineCache(max_bars=3)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [(base + timedelta(hours=i), _row(base + timedelta(hours=i), 100.0 + i)) for i in range(5)]
    cache.ingest("ETHUSDT", "1h", _df(rows))
    snap = cache.snapshot("ETHUSDT", "1h")
    assert len(snap) == 3
    assert snap["close"].iloc[-1] == 104.0


def test_ingest_dedupes_overlapping_timestamps():
    cache = RollingKlineCache(max_bars=10)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = _df([(base + timedelta(hours=i), _row(base + timedelta(hours=i), 100.0 + i)) for i in range(3)])
    overlap = _df([(base + timedelta(hours=2), _row(base + timedelta(hours=2), 999.0))])
    cache.ingest("ETHUSDT", "1h", first)
    cache.ingest("ETHUSDT", "1h", overlap)
    snap = cache.snapshot("ETHUSDT", "1h")
    assert len(snap) == 3
    # latest write wins for the duplicated bar
    assert snap["close"].iloc[-1] == 999.0


def test_atr_simple_average_of_high_low_range():
    cache = RollingKlineCache(max_bars=20)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(5):
        rows.append((base + timedelta(hours=i),
                     _row(base + timedelta(hours=i), close=100.0,
                          hi=110.0 + i, lo=90.0 + i)))
    cache.ingest("ETHUSDT", "1h", _df(rows))
    # Range = 20.0 every bar -> ATR over n=5 = 20.0
    assert cache.atr("ETHUSDT", "1h", n=5) == pytest.approx(20.0)


def test_spread_bps_recorded_and_returned():
    cache = RollingKlineCache(max_bars=10)
    cache.record_spread("ETHUSDT", bid=2999.0, ask=3001.0)
    # mid = 3000, spread = 2 -> 2/3000*1e4 = 6.6667 bps
    assert cache.spread_bps("ETHUSDT") == pytest.approx(2.0 / 3000.0 * 10_000)


def test_unsupported_symbol_returns_none_for_spread():
    cache = RollingKlineCache(max_bars=10)
    assert cache.spread_bps("BTCUSDT") is None
