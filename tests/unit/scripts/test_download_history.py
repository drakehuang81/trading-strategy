from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from scripts.download_history import (
    fetch_klines_paginated,
    upsert_parquet,
)


def _df(start: datetime, n: int, base_close: float = 3000.0) -> pd.DataFrame:
    idx = [start + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame({
        "open": [base_close] * n, "high": [base_close + 5] * n,
        "low": [base_close - 5] * n, "close": [base_close + i for i in range(n)],
        "volume": [1.0] * n,
    }, index=pd.DatetimeIndex(idx, name="open_time"))


@pytest.mark.asyncio
async def test_paginator_walks_in_1000_bar_chunks_until_until():
    chunks = [_df(datetime(2026, 1, 1, tzinfo=timezone.utc), 1000),
              _df(datetime(2026, 2, 11, 16, tzinfo=timezone.utc), 500),
              _df(datetime(2026, 1, 1, tzinfo=timezone.utc), 0)]  # empty -> stop
    fake = AsyncMock()
    fake.fetch = AsyncMock(side_effect=chunks)
    df = await fetch_klines_paginated(
        fake,
        symbol="ETHUSDT", timeframe="1h",
        since=datetime(2026, 1, 1, tzinfo=timezone.utc),
        until=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    assert len(df) == 1500
    assert df.index.is_monotonic_increasing
    # No duplicate indices.
    assert df.index.is_unique


@pytest.mark.asyncio
async def test_paginator_stops_on_empty_chunk():
    fake = AsyncMock()
    fake.fetch = AsyncMock(side_effect=[
        _df(datetime(2026, 1, 1, tzinfo=timezone.utc), 0),
    ])
    df = await fetch_klines_paginated(
        fake,
        symbol="ETHUSDT", timeframe="1h",
        since=datetime(2026, 1, 1, tzinfo=timezone.utc),
        until=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    assert df.empty


def test_upsert_parquet_dedupes_overlap(tmp_path):
    p = tmp_path / "ETHUSDT_1h.parquet"
    a = _df(datetime(2026, 1, 1, tzinfo=timezone.utc), 5, base_close=3000.0)
    b = _df(datetime(2026, 1, 1, 3, tzinfo=timezone.utc), 5, base_close=4000.0)
    upsert_parquet(p, a)
    upsert_parquet(p, b)
    out = pd.read_parquet(p)
    # 5 + 5 with 2 overlapping timestamps -> 8 unique
    assert len(out) == 8
    # Latest write wins on the overlap.
    assert out.loc[datetime(2026, 1, 1, 4, tzinfo=timezone.utc), "close"] == 4001


@pytest.mark.asyncio
async def test_funding_backfill_invoked_when_parquet_missing(tmp_path, monkeypatch):
    """download_history should call FundingRateWriter.backfill when no parquet exists."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from datetime import datetime, timedelta, timezone

    from scripts.download_history import main_async
    import argparse

    # Fake BinanceKline that returns one chunk of klines, then closes cleanly.
    fake_kline_df = _df(datetime(2026, 4, 1, tzinfo=timezone.utc), 5)
    fake_source = AsyncMock()
    fake_source.fetch = AsyncMock(side_effect=[fake_kline_df, _df(datetime(2026, 4, 2, tzinfo=timezone.utc), 0)])
    fake_source.close = AsyncMock()
    fake_source.client = MagicMock()  # accessed by FundingRateWriter

    backfill_mock = AsyncMock(return_value=42)
    update_mock = AsyncMock(return_value=0)

    args = argparse.Namespace(
        symbol="ETHUSDT", timeframe="1h", years=1,
        out_dir=str(tmp_path / "history"),
        funding_out_dir=str(tmp_path / "funding"),
    )

    with patch("scripts.download_history.BinanceKline.open",
               new=AsyncMock(return_value=fake_source)), \
         patch("scripts.download_history.FundingRateWriter") as FW:
        instance = MagicMock()
        instance.backfill = backfill_mock
        instance.update = update_mock
        FW.return_value = instance
        await main_async(args)

    backfill_mock.assert_awaited_once()
    update_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_funding_backfill_skipped_when_parquet_covers_since(tmp_path, monkeypatch):
    """If existing parquet already covers since, only update() runs (not backfill)."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from datetime import datetime, timezone, timedelta
    import argparse
    import pandas as pd

    from scripts.download_history import main_async

    funding_dir = tmp_path / "funding"
    funding_dir.mkdir()
    # Pre-seed parquet covering the full requested window
    base = datetime.now(tz=timezone.utc) - timedelta(days=400)
    idx = pd.DatetimeIndex(
        [base + timedelta(hours=i * 8) for i in range(50)],
        name="ts",
    )
    pd.DataFrame({"funding_rate": [0.0001] * 50}, index=idx).to_parquet(
        funding_dir / "ETHUSDT.parquet"
    )

    fake_kline_df = _df(datetime(2026, 4, 1, tzinfo=timezone.utc), 5)
    fake_source = AsyncMock()
    fake_source.fetch = AsyncMock(side_effect=[fake_kline_df, _df(datetime(2026, 4, 2, tzinfo=timezone.utc), 0)])
    fake_source.close = AsyncMock()
    fake_source.client = MagicMock()

    backfill_mock = AsyncMock(return_value=0)
    update_mock = AsyncMock(return_value=0)

    args = argparse.Namespace(
        symbol="ETHUSDT", timeframe="1h", years=1,
        out_dir=str(tmp_path / "history"),
        funding_out_dir=str(funding_dir),
    )

    with patch("scripts.download_history.BinanceKline.open",
               new=AsyncMock(return_value=fake_source)), \
         patch("scripts.download_history.FundingRateWriter") as FW:
        instance = MagicMock()
        instance.backfill = backfill_mock
        instance.update = update_mock
        FW.return_value = instance
        await main_async(args)

    backfill_mock.assert_not_awaited()
    update_mock.assert_awaited_once()
