"""Data layer Protocol — spec §4.1."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    """Historical + latest candles abstraction. Every implementation
    must be async-safe and return a DataFrame indexed by UTC timestamp."""

    name: str

    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame: ...

    async def fetch_latest(
        self,
        symbol: str,
        timeframe: str,
        n: int,
    ) -> pd.DataFrame: ...

    def supports(self, symbol: str, timeframe: str) -> bool: ...
