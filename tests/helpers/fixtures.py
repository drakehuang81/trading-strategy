"""Shared test fixtures — ETH OHLCV sample data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def eth_1h_df() -> pd.DataFrame:
    """Small (~400 rows) ETHUSDT 1h sample covering enough history for
    any single feature's required_lookback."""
    df = pd.read_csv(FIXTURES_DIR / "ethusdt_1h_sample.csv", parse_dates=["open_time"])
    df = df.set_index("open_time").sort_index()
    return df
