"""In-process rolling kline + spread cache (Plan 5A Task 2).

Holds the last N bars per (symbol, timeframe). `mid_provider` and
`atr_provider` (Task 3) read from here so we do not hit Binance on every
risk-check call. Updated by a refresh loop owned by the orchestrator
(Task 4) and seeded by `BinanceKline.fetch_latest` at boot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RollingKlineCache:
    """In-process N-bar buffer per (symbol, timeframe) + last-known spread.

    Concurrency: single-writer (Task 4's refresh loop) / multi-reader
    (ThresholdPolicy / SpreadGate / PaperBroker). Safe under asyncio
    because no `await` lives between mutation and read inside this class.
    Do NOT make any method `async` without adding a lock.
    """

    max_bars: int = 200
    _frames: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    _spreads_bps: dict[str, float] = field(default_factory=dict)

    def ingest(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        key = (symbol, timeframe)
        existing = self._frames.get(key)
        if existing is None or existing.empty:
            combined = df
        else:
            combined = pd.concat([existing, df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        self._frames[key] = combined.iloc[-self.max_bars:]

    def snapshot(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._frames.get((symbol, timeframe), pd.DataFrame()).copy()

    def last_close(self, symbol: str, timeframe: str) -> float | None:
        snap = self._frames.get((symbol, timeframe))
        if snap is None or snap.empty:
            return None
        return float(snap["close"].iloc[-1])

    def atr(self, symbol: str, timeframe: str, n: int = 14) -> float | None:
        """Mean of (high - low) over the last n bars.

        NOT Wilder true-range ATR; bar-to-bar gaps are ignored. Sufficient
        for relative threshold sizing in Plan 5A; upgrade to Wilder if a
        future feature/risk check needs gap-aware volatility.
        """
        snap = self._frames.get((symbol, timeframe))
        if snap is None or len(snap) < n:
            return None
        rng = (snap["high"] - snap["low"]).iloc[-n:]
        return float(rng.mean())

    def record_spread(self, symbol: str, bid: float, ask: float) -> None:
        if bid <= 0 or ask <= 0 or ask < bid:
            return
        mid = 0.5 * (bid + ask)
        if mid <= 0:
            return
        self._spreads_bps[symbol] = (ask - bid) / mid * 10_000.0

    def spread_bps(self, symbol: str) -> float | None:
        return self._spreads_bps.get(symbol)
