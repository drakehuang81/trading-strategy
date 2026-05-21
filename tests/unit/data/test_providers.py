import pytest

from data.kline_cache import RollingKlineCache
from data.providers import (
    cache_backed_atr_provider,
    cache_backed_mid_provider,
    cache_backed_spread_bps_provider,
)
from datetime import datetime, timezone, timedelta
import pandas as pd


def _seed_cache(cache: RollingKlineCache, symbol: str, n: int) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    idx = []
    for i in range(n):
        idx.append(base + timedelta(hours=i))
        rows.append({"open": 3000, "high": 3010 + i, "low": 2990 - i,
                     "close": 3000 + i, "volume": 1.0})
    df = pd.DataFrame(rows, index=idx)
    cache.ingest(symbol, "1h", df)


def test_mid_provider_uses_last_close_when_warm():
    cache = RollingKlineCache(max_bars=200)
    _seed_cache(cache, "ETHUSDT", 5)
    mid = cache_backed_mid_provider(cache, timeframe="1h", fallback=3000.0)
    assert mid("ETHUSDT") == 3004.0   # last close = 3000 + 4


def test_mid_provider_returns_fallback_when_cold():
    cache = RollingKlineCache(max_bars=200)
    mid = cache_backed_mid_provider(cache, timeframe="1h", fallback=2500.0)
    assert mid("ETHUSDT") == 2500.0


def test_atr_provider_uses_cache_atr_when_warm():
    cache = RollingKlineCache(max_bars=200)
    _seed_cache(cache, "ETHUSDT", 30)
    atr = cache_backed_atr_provider(cache, timeframe="1h", n=14, fallback=15.0)
    assert atr("ETHUSDT") > 0.0
    assert atr("ETHUSDT") != 15.0


def test_atr_provider_returns_fallback_when_cold():
    cache = RollingKlineCache(max_bars=200)
    atr = cache_backed_atr_provider(cache, timeframe="1h", n=14, fallback=15.0)
    assert atr("ETHUSDT") == 15.0


def test_spread_provider_uses_recorded_spread_when_warm():
    cache = RollingKlineCache(max_bars=200)
    cache.record_spread("ETHUSDT", bid=2999.0, ask=3001.0)
    sp = cache_backed_spread_bps_provider(cache, fallback=0.0)
    assert sp("ETHUSDT") == pytest.approx(2.0 / 3000.0 * 10_000)


def test_spread_provider_returns_fallback_when_no_record():
    cache = RollingKlineCache(max_bars=200)
    sp = cache_backed_spread_bps_provider(cache, fallback=0.0)
    assert sp("ETHUSDT") == 0.0
