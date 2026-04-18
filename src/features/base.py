"""Feature layer Protocol — spec §4.2.

Every Feature MUST only use df[df.index <= as_of]. The no-repainting
test (§9.2) enforces this at CI time."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

import pandas as pd


class Feature(Protocol):
    """Point-in-time feature computation."""

    name: str
    version: str           # bump on logic change
    required_lookback: int

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]: ...
