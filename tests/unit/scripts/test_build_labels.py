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


def test_main_dispatches_triple_barrier_label_type(tmp_path, monkeypatch):
    """Calling main() with --label-type=triple_barrier writes a parquet
    whose only column is y_triple_barrier_<H>."""
    import sys
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = 50
    closes = [3000.0 + (i * 0.5) for i in range(n)]
    df = pd.DataFrame({
        "open":   closes,
        "high":   [c + 8.0 for c in closes],
        "low":    [c - 8.0 for c in closes],
        "close":  closes,
        "volume": [1.0] * n,
    }, index=pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(n)], name="open_time",
    ))
    kline_path = tmp_path / "k.parquet"
    out_path = tmp_path / "labels.parquet"
    df.to_parquet(kline_path)

    monkeypatch.setattr(sys, "argv", [
        "build_labels",
        "--kline", str(kline_path),
        "--out", str(out_path),
        "--horizon", "4",
        "--label-type", "triple_barrier",
        "--tp-mult", "1.5",
        "--sl-mult", "1.5",
        "--atr-window", "14",
    ])
    from scripts.build_labels import main
    main()

    out = pd.read_parquet(out_path)
    assert list(out.columns) == ["y_triple_barrier_4"]
    # Most rows should be labeled (we seeded a strong uptrend).
    assert len(out) > 20


def test_main_default_label_type_stays_forward_up(tmp_path, monkeypatch):
    """Existing callers (no --label-type) get the forward_up behavior."""
    import sys
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
                      index=pd.DatetimeIndex(
                          [base + timedelta(hours=i) for i in range(6)],
                          name="open_time"))
    kline_path = tmp_path / "k.parquet"
    out_path = tmp_path / "labels.parquet"
    df.to_parquet(kline_path)

    monkeypatch.setattr(sys, "argv", [
        "build_labels",
        "--kline", str(kline_path),
        "--out", str(out_path),
        "--horizon", "4",
    ])
    from scripts.build_labels import main
    main()

    out = pd.read_parquet(out_path)
    assert list(out.columns) == ["y_4bar_up"]
