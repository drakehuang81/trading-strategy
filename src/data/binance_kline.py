"""BinanceKline — first DataSource implementation (spec §4.1).

Wraps python-binance AsyncClient. Public constructor accepts an injected
client (for tests). Real client lifecycle is managed by
`BinanceKline.open(api_key, api_secret)` classmethod.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd


_VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}


class BinanceKline:
    name = "binance_kline"

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        """Underlying python-binance AsyncClient. Exposed so callers
        can construct adjacent writers (FundingRateWriter) without
        re-opening a second connection. Caller does NOT own the
        client lifecycle — `BinanceKline.close()` still closes it."""
        return self._client

    @classmethod
    async def open(cls, api_key: str = "", api_secret: str = "") -> "BinanceKline":
        from binance import AsyncClient
        client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
        return cls(client)

    async def close(self) -> None:
        await self._client.close_connection()

    def supports(self, symbol: str, timeframe: str) -> bool:
        return timeframe in _VALID_INTERVALS

    async def fetch_latest(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        raw = await self._client.get_klines(symbol=symbol, interval=timeframe, limit=n)
        return self._to_df(raw)

    async def fetch(
        self, symbol: str, timeframe: str, since: datetime, until: datetime
    ) -> pd.DataFrame:
        start_ms = int(since.timestamp() * 1000)
        end_ms = int(until.timestamp() * 1000)
        raw = await self._client.get_klines(
            symbol=symbol, interval=timeframe,
            startTime=start_ms, endTime=end_ms, limit=1000,
        )
        return self._to_df(raw)

    @staticmethod
    def _to_df(raw: Iterable[list[Any]]) -> pd.DataFrame:
        rows = list(raw)
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "n_trades", "tbv", "tqv", "_ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df[["open", "high", "low", "close", "volume"]]
