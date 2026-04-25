from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.build_training_set import build_training_set
from features.registry import build_default_registry


def _kline_df(n: int) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex([base + timedelta(hours=i) for i in range(n)],
                           name="open_time")
    return pd.DataFrame({
        "open":   [3000.0 + i for i in range(n)],
        "high":   [3010.0 + i for i in range(n)],
        "low":    [2990.0 + i for i in range(n)],
        "close":  [3005.0 + i for i in range(n)],
        "volume": [1.0] * n,
    }, index=idx)


def test_build_training_set_skips_lookback_warmup():
    df = _kline_df(250)
    out = build_training_set(df, registry=build_default_registry(),
                             warmup_bars=200)
    # 250 - 200 = 50 rows
    assert len(out) == 50
    # First row's `as_of` is bar #200 (zero-indexed) = base + 200h
    assert out.index[0] == df.index[200]
    # Index name preserved.
    assert out.index.name == "as_of"


def test_build_training_set_emits_one_column_per_flat_feature():
    df = _kline_df(220)
    out = build_training_set(df, registry=build_default_registry(),
                             warmup_bars=200)
    # smc, fib, liquidity, divergence, funding, confidence each contribute >=1 column
    # (some may flatten to multiple sub-keys; we only require >0).
    assert len(out.columns) > 0
    # Column names are dot-prefixed by feature name.
    assert any(c.startswith("smc.") for c in out.columns)
