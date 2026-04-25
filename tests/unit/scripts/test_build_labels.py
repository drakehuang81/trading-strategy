from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.build_labels import compute_forward_up_labels


def _df(closes: list[float]) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex([base + timedelta(hours=i) for i in range(len(closes))],
                           name="open_time")
    return pd.DataFrame({"close": closes}, index=idx)


def test_label_is_one_when_future_close_higher():
    df = _df([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    out = compute_forward_up_labels(df, horizon=4)
    # only index 0 is valid (needs t+4)
    assert int(out.iloc[0]) == 1
    # index 1 still in range -> 102 -> 105 +ve
    assert int(out.iloc[1]) == 1
    # last 4 rows have no future close -> NaN
    assert out.iloc[-4:].isna().all()


def test_label_is_zero_when_future_close_equal_or_lower():
    df = _df([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    out = compute_forward_up_labels(df, horizon=4)
    # equal does NOT count as up
    assert int(out.iloc[0]) == 0


def test_label_handles_negative_returns():
    df = _df([100.0, 99.0, 98.0, 97.0, 96.0])
    out = compute_forward_up_labels(df, horizon=4)
    assert int(out.iloc[0]) == 0
