import pandas as pd
import pytest
from pathlib import Path

from data.funding import FundingRateWriter, load_funding


class FakeFundingClient:
    async def futures_funding_rate(self, *, symbol, startTime=None, endTime=None, limit=1000):
        return [
            {"fundingTime": 1700000000000, "fundingRate": "0.0001"},
            {"fundingTime": 1700028800000, "fundingRate": "0.0002"},
        ]
    async def close_connection(self): ...


@pytest.mark.asyncio
async def test_writer_persists_parquet(tmp_path: Path):
    w = FundingRateWriter(client=FakeFundingClient(), out_dir=tmp_path)
    n = await w.update("ETHUSDT")
    assert n == 2
    df = load_funding(tmp_path / "ETHUSDT.parquet")
    assert list(df.columns) == ["funding_rate"]
    assert df.index.tz is not None
