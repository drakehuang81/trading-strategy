"""Cache-backed callables for mid / ATR / spread (Plan 5A Task 3).

Returned functions match the signature `(symbol: str) -> float` consumed
by ThresholdPolicy, PaperBroker, and SpreadGate. Each returns the
configured `fallback` value when the cache has no data, so the
orchestrator can boot and run unit tests before the first refresh.
"""
from __future__ import annotations

from typing import Callable

from data.kline_cache import RollingKlineCache


def cache_backed_mid_provider(
    cache: RollingKlineCache,
    timeframe: str,
    fallback: float,
) -> Callable[[str], float]:
    def _mid(symbol: str) -> float:
        v = cache.last_close(symbol, timeframe)
        return v if v is not None else fallback
    return _mid


def cache_backed_atr_provider(
    cache: RollingKlineCache,
    timeframe: str,
    n: int,
    fallback: float,
) -> Callable[[str], float]:
    def _atr(symbol: str) -> float:
        v = cache.atr(symbol, timeframe, n=n)
        return v if v is not None else fallback
    return _atr


def cache_backed_spread_bps_provider(
    cache: RollingKlineCache,
    fallback: float,
) -> Callable[[str], float]:
    def _spread(symbol: str) -> float:
        v = cache.spread_bps(symbol)
        return v if v is not None else fallback
    return _spread
