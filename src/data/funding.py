"""Funding rate ingestion — spec §4.1.

Binance futures funding rate every 8h. Stored at data/funding/<symbol>.parquet
with DatetimeIndex (UTC) and single column funding_rate (float).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class FundingRateWriter:
    def __init__(self, client: Any, out_dir: Path) -> None:
        self._client = client
        self._out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    async def update(self, symbol: str) -> int:
        """Fetch new funding rows since last persisted ts; upsert to parquet."""
        out = self._out_dir / f"{symbol}.parquet"
        existing = load_funding(out) if out.exists() else pd.DataFrame()
        start_ms = int(existing.index.max().timestamp() * 1000) + 1 if not existing.empty else None
        raw = await self._client.futures_funding_rate(symbol=symbol, startTime=start_ms, limit=1000)
        if not raw:
            return 0
        new_df = pd.DataFrame(raw)
        new_df["ts"] = pd.to_datetime(new_df["fundingTime"], unit="ms", utc=True)
        new_df = new_df.set_index("ts")
        new_df["funding_rate"] = new_df["fundingRate"].astype(float)
        new_df = new_df[["funding_rate"]]
        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(out)
        return len(new_df)

    async def backfill(self, symbol: str, *, since: datetime) -> int:
        """Paginate Binance funding history backward until reaching `since`.

        Walks endTime cursor in 1000-row chunks. Stops on either:
          - chunk earliest row predates `since` (we've covered the requested range), or
          - chunk is empty (no more history available from the exchange).

        Idempotent: re-running on top of an existing parquet dedupes by ts.

        Returns the number of rows fetched from the API across all pages
        (NOT the number of *new* rows added to disk — re-running still returns
        a positive number on the second call). For a precise "rows added to
        disk" count, diff `len(load_funding(...))` before and after.
        """
        if since.tzinfo is None:
            raise ValueError("backfill requires timezone-aware `since` (use UTC)")

        out = self._out_dir / f"{symbol}.parquet"
        existing = load_funding(out) if out.exists() else pd.DataFrame()

        since_ms = int(since.timestamp() * 1000)
        end_cursor: int | None = None
        all_rows: list[dict[str, Any]] = []
        seen_ms: set[int] = set()
        while True:
            chunk = await self._client.futures_funding_rate(
                symbol=symbol, endTime=end_cursor, limit=1000,
            )
            if not chunk:
                break
            new_chunk = [r for r in chunk if int(r["fundingTime"]) not in seen_ms]
            for r in new_chunk:
                seen_ms.add(int(r["fundingTime"]))
            all_rows.extend(new_chunk)

            chunk_min_ms = min(int(r["fundingTime"]) for r in chunk)
            # may overshoot since by up to limit-1 rows; intentional — we don't slice mid-chunk
            if chunk_min_ms <= since_ms:
                break
            # Page back: next chunk's endTime is one ms before this chunk's earliest.
            end_cursor = chunk_min_ms - 1

        if not all_rows:
            return 0

        new_df = pd.DataFrame(all_rows)
        new_df["ts"] = pd.to_datetime(new_df["fundingTime"], unit="ms", utc=True)
        new_df = new_df.set_index("ts")
        new_df["funding_rate"] = new_df["fundingRate"].astype(float)
        new_df = new_df[["funding_rate"]]
        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(out)
        return len(new_df)


def load_funding(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
