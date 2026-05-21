import pytest
import pandas as pd
from datetime import datetime, timezone

from data.binance_kline import BinanceKline


class FakeAsyncClient:
    async def get_klines(self, *, symbol, interval, startTime=None, endTime=None, limit=500):
        return [
            [1700000000000, "2000", "2010", "1990", "2005", "10", 1700003599999,
             "0", 100, "0", "0", "0"],
            [1700003600000, "2005", "2015", "2000", "2012", "12", 1700007199999,
             "0", 120, "0", "0", "0"],
        ]

    async def close_connection(self): ...


@pytest.mark.asyncio
async def test_fetch_latest_returns_dataframe():
    ds = BinanceKline(client=FakeAsyncClient())
    df = await ds.fetch_latest("ETHUSDT", "1h", n=2)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")
    assert df["open"].iloc[0] == 2000.0


def test_supports_only_known_intervals():
    ds = BinanceKline(client=None)
    assert ds.supports("ETHUSDT", "1h") is True
    assert ds.supports("ETHUSDT", "7m") is False
