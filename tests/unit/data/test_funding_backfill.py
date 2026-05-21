"""Tests for FundingRateWriter.backfill — Plan 5B-1 Task 1."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data.funding import FundingRateWriter, load_funding


class _ChunkedFakeFundingClient:
    """Fake Binance AsyncClient that models real Binance startTime/endTime semantics.

    Holds an ordered list of (ts_ms, rate) rows representing the full universe
    of available funding history. Each `futures_funding_rate` call returns up to
    `limit` rows filtered by [startTime, endTime], oldest-first — matching real
    Binance behavior. Tracks calls so tests can assert pagination actually happened.
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
        self.calls.append({"startTime": startTime, "endTime": endTime, "limit": limit})
        # Real Binance semantics:
        # - returns rows where (startTime is None or ts >= startTime) AND (endTime is None or ts <= endTime)
        # - sorted oldest-first
        # - capped at limit
        rows = self._rows
        if startTime is not None:
            rows = [r for r in rows if r[0] >= startTime]
        if endTime is not None:
            rows = [r for r in rows if r[0] <= endTime]
        chosen = rows[:limit]
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
    universe_max = base + timedelta(hours=2199 * 8)
    n_added = await writer.backfill("ETHUSDT", since=since, until=universe_max)

    assert n_added == 2200
    df = load_funding(tmp_path / "ETHUSDT.parquet")
    assert len(df) == 2200
    assert df.index.min().to_pydatetime() == base
    # 2200 rows requires at least 3 pages of limit=1000
    assert len(client.calls) >= 3


@pytest.mark.asyncio
async def test_backfill_starts_at_since_and_walks_forward(tmp_path: Path):
    """Forward pagination from since should fetch rows from since onward."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    universe = _make_universe(base, n=2200)
    client = _ChunkedFakeFundingClient(universe)
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    # since = 6 months in (~540 ticks). Should fetch from there to universe end.
    since = base + timedelta(days=180)
    universe_max = base + timedelta(hours=2199 * 8)
    n_added = await writer.backfill("ETHUSDT", since=since, until=universe_max)

    df = load_funding(tmp_path / "ETHUSDT.parquet")
    # All rows >= since (forward paginate doesn't fetch older).
    assert df.index.min().to_pydatetime() >= since
    # Coverage extends to universe end.
    assert df.index.max().to_pydatetime() == universe_max
    # We did at least 2 pages (1660 rows requires 2 pages of 1000).
    assert len(client.calls) >= 2
    assert n_added == len(df)


@pytest.mark.asyncio
async def test_backfill_stops_on_empty_chunk(tmp_path: Path):
    """When Binance returns [], backfill stops even if we haven't reached since."""
    client = _ChunkedFakeFundingClient(rows=[])  # empty universe
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    until = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = await writer.backfill("ETHUSDT", since=datetime(2024, 1, 1, tzinfo=timezone.utc), until=until)
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

    universe_max = base + timedelta(hours=499 * 8)
    n1 = await writer.backfill("ETHUSDT", since=base, until=universe_max)
    assert n1 == 500
    # Run again — should produce 0 new rows, parquet stays at 500.
    n2 = await writer.backfill("ETHUSDT", since=base, until=universe_max)
    df = load_funding(tmp_path / "ETHUSDT.parquet")
    assert len(df) == 500
    # n2 reports rows fetched from API (still 500), but the parquet doesn't grow.
    # We let the implementation define n2's semantic: it's documentation of API
    # cost, not of new-row count. Either contract is acceptable; pin whatever
    # we ship.
    assert n2 == 500   # rows fetched from API; net rows added to disk = 0


@pytest.mark.asyncio
async def test_backfill_rejects_naive_datetime(tmp_path: Path):
    """Naive datetime would silently produce wrong epoch — must raise."""
    client = _ChunkedFakeFundingClient(rows=[])
    writer = FundingRateWriter(client=client, out_dir=tmp_path)
    naive = datetime(2024, 1, 1)  # NO tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        await writer.backfill("ETHUSDT", since=naive)


@pytest.mark.asyncio
async def test_backfill_rejects_naive_until(tmp_path: Path):
    """Naive until would silently produce wrong epoch — must raise."""
    client = _ChunkedFakeFundingClient(rows=[])
    writer = FundingRateWriter(client=client, out_dir=tmp_path)
    aware_since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    naive_until = datetime(2026, 4, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        await writer.backfill("ETHUSDT", since=aware_since, until=naive_until)
