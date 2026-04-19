"""Funding rate ingestion — spec §4.1.

Binance futures funding rate every 8h. Stored at data/funding/<symbol>.parquet
with DatetimeIndex (UTC) and single column funding_rate (float).
"""
from __future__ import annotations

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


def load_funding(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
