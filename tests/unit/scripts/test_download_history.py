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
