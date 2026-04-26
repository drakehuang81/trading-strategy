"""Tests for FundingRateWriter.backfill — Plan 5B-1 Task 1."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data.funding import FundingRateWriter, load_funding


class _ChunkedFakeFundingClient:
    """Fake Binance AsyncClient that returns predictable chunks based on endTime.

    Holds an ordered list of (ts_ms, rate) rows representing the full universe
    of available funding history. Each `futures_funding_rate(endTime=X)` call
    returns up to `limit` rows with `fundingTime <= X`, ordered ascending by
    fundingTime. When the universe is exhausted (no rows at or before
    endTime), returns []. Tracks call count so tests can assert pagination
    actually happened.
    """

    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self._rows = sorted(rows)  # (ts_ms, rate) tuples, ascending
        self.calls: list[dict[str, Any]] = []

    async def futures_funding_rate(
        self,
        *,
        symbol: str,
        startTime: int | None = None,
        endTime: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        self.calls.append({"endTime": endTime, "limit": limit})
        if endTime is None:
            chosen = self._rows[-limit:]
        else:
            chosen = [r for r in self._rows if r[0] <= endTime][-limit:]
        return [{"fundingTime": ts, "fundingRate": str(rate)} for ts, rate in chosen]

    async def close_connection(self) -> None:
        pass


def _make_universe(start: datetime, n: int, step_hours: int = 8) -> list[tuple[int, float]]:
    """N funding rows starting at `start`, every `step_hours` hours."""
    rows = []
    for i in range(n):
        ts = start + timedelta(hours=i * step_hours)
        rows.append((int(ts.timestamp() * 1000), 0.0001 + i * 1e-7))
    return rows


@pytest.mark.asyncio
async def test_backfill_paginates_until_since(tmp_path: Path):
    """3 pages of 1000-row chunks should cover ~3000 funding ticks."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    universe = _make_universe(base, n=2200)  # ~2 years of 8h ticks
    client = _ChunkedFakeFundingClient(universe)
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    since = base                                       # ask for the full universe
    n_added = await writer.backfill("ETHUSDT", since=since)

    assert n_added == 2200
    df = load_funding(tmp_path / "ETHUSDT.parquet")
    assert len(df) == 2200
    assert df.index.min().to_pydatetime() == base
    # 2200 rows requires at least 3 pages of limit=1000
    assert len(client.calls) >= 3


@pytest.mark.asyncio
async def test_backfill_stops_when_chunk_predates_since(tmp_path: Path):
    """When chunk earliest row is < since, we stop without paging further."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    universe = _make_universe(base, n=2200)
    client = _ChunkedFakeFundingClient(universe)
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    # since = 6 months in (~540 funding ticks from start), so we should grab
    # ~1660 rows (the slice from since to now, +/- one chunk worth of overshoot
    # because we don't slice mid-chunk).
    since = base + timedelta(days=180)
    n_added = await writer.backfill("ETHUSDT", since=since)

    df = load_funding(tmp_path / "ETHUSDT.parquet")
    # We must cover everything from `since` onward (no gap).
    assert df.index.min().to_pydatetime() <= since
    assert df.index.max().to_pydatetime() >= base + timedelta(hours=2199 * 8)
    # We did at least 2 pages (because page 1 alone can't reach 1660 rows).
    assert len(client.calls) >= 2
    # Stop condition fired: we did NOT keep paging for ages.
    assert len(client.calls) <= 4
    assert n_added == len(df)


@pytest.mark.asyncio
async def test_backfill_stops_on_empty_chunk(tmp_path: Path):
    """When Binance returns [], backfill stops even if we haven't reached since."""
    client = _ChunkedFakeFundingClient(rows=[])  # empty universe
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    n = await writer.backfill("ETHUSDT", since=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert n == 0
    # No parquet should be written when no rows came back.
    assert not (tmp_path / "ETHUSDT.parquet").exists()


@pytest.mark.asyncio
async def test_backfill_dedupes_against_existing_parquet(tmp_path: Path):
    """Re-running backfill on top of existing parquet must not duplicate rows."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    universe = _make_universe(base, n=500)
    client = _ChunkedFakeFundingClient(universe)
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    n1 = await writer.backfill("ETHUSDT", since=base)
    assert n1 == 500
    # Run again — should produce 0 new rows, parquet stays at 500.
    n2 = await writer.backfill("ETHUSDT", since=base)
    df = load_funding(tmp_path / "ETHUSDT.parquet")
    assert len(df) == 500
    # n2 reports rows fetched from API (still 500), but the parquet doesn't grow.
    # We let the implementation define n2's semantic: it's documentation of API
    # cost, not of new-row count. Either contract is acceptable; pin whatever
    # we ship.
    assert isinstance(n2, int)
