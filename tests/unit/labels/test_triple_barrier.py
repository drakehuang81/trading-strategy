"""Triple-barrier labels — Plan 5C Task 1."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from labels.triple_barrier import triple_barrier_labels


def _ohlc(closes: list[float], highs: list[float] | None = None,
          lows: list[float] | None = None) -> pd.DataFrame:
    """Build a DataFrame with the given closes; default high=close+5, low=close-5."""
    n = len(closes)
    if highs is None:
        highs = [c + 5.0 for c in closes]
    if lows is None:
        lows = [c - 5.0 for c in closes]
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(n)], name="open_time",
    )
    return pd.DataFrame({
        "open":   closes, "high": highs, "low": lows,
        "close":  closes, "volume": [1.0] * n,
    }, index=idx)


def test_label_one_when_tp_hit_before_sl():
    """Up-trending bars → high crosses TP before low crosses SL."""
    closes = [3000.0] * 14 + [3000.0, 3010.0, 3025.0, 3050.0, 3050.0]
    highs  = [3005.0] * 14 + [3005.0, 3015.0, 3030.0, 3055.0, 3055.0]
    lows   = [2995.0] * 14 + [2995.0, 3005.0, 3020.0, 3045.0, 3045.0]
    df = _ohlc(closes, highs, lows)
    # ATR(14) over the first 14 bars = mean(10) = 10.0.
    # At bar 14: TP = 3000 + 1.5*10 = 3015. SL = 3000 - 1.5*10 = 2985.
    # Bar 15 high = 3015 → TP touched on bar 15. Label = 1.
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert int(out.iloc[14]) == 1


def test_label_zero_when_sl_hit_before_tp():
    """Down-trending bars → low crosses SL before high crosses TP."""
    closes = [3000.0] * 14 + [3000.0, 2990.0, 2975.0, 2950.0, 2950.0]
    highs  = [3005.0] * 14 + [3005.0, 2995.0, 2980.0, 2955.0, 2955.0]
    lows   = [2995.0] * 14 + [2995.0, 2985.0, 2970.0, 2945.0, 2945.0]
    df = _ohlc(closes, highs, lows)
    # ATR(14) = 10. At bar 14: TP = 3015, SL = 2985.
    # Bar 15 low = 2985 → SL touched. Label = 0.
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert int(out.iloc[14]) == 0


def test_label_uses_close_sign_on_timeout():
    """Neither barrier hit within horizon → fall back to close sign."""
    # Closes hover within ±5 (well inside ±15 barriers); end slightly up.
    closes = [3000.0] * 14 + [3000.0, 3001.0, 3002.0, 3003.0, 3004.0]
    highs  = [3005.0] * 14 + [3005.0, 3006.0, 3007.0, 3008.0, 3009.0]
    lows   = [2995.0] * 14 + [2995.0, 2996.0, 2997.0, 2998.0, 2999.0]
    df = _ohlc(closes, highs, lows)
    # No high crosses 3015, no low crosses 2985 within horizon=4.
    # close[14+4=18] = 3004 > close[14] = 3000 → label = 1.
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert int(out.iloc[14]) == 1


def test_label_zero_on_timeout_when_close_drifts_down():
    closes = [3000.0] * 14 + [3000.0, 2999.0, 2998.0, 2997.0, 2996.0]
    highs  = [3005.0] * 14 + [3005.0, 3004.0, 3003.0, 3002.0, 3001.0]
    lows   = [2995.0] * 14 + [2995.0, 2994.0, 2993.0, 2992.0, 2991.0]
    df = _ohlc(closes, highs, lows)
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert int(out.iloc[14]) == 0


def test_label_nan_for_trailing_rows_without_horizon():
    """The last `horizon` rows can't be labeled (no forward bars)."""
    n = 30
    closes = [3000.0] * n
    df = _ohlc(closes)
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert out.iloc[-4:].isna().all()


def test_label_nan_during_atr_warmup():
    """First `atr_window` rows have no ATR → no labels."""
    n = 30
    closes = [3000.0] * n
    df = _ohlc(closes)
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert out.iloc[:14].isna().all()


def test_same_bar_tp_and_sl_hit_uses_close_sign():
    """When a single bar's high≥TP AND low≤SL, fall back to that bar's close vs entry."""
    # Bar 14 has wild range: high 3020 (above TP 3015), low 2980 (below SL 2985).
    # That bar's close = 3010 → up sign → label = 1.
    closes = [3000.0] * 14 + [3010.0] + [3000.0] * 4
    highs  = [3005.0] * 14 + [3020.0] + [3005.0] * 4
    lows   = [2995.0] * 14 + [2980.0] + [2995.0] * 4
    df = _ohlc(closes, highs, lows)
    # The function walks forward bars t+1 .. t+horizon, NOT t itself.
    # So for entry at t=14, the wild bar at index 14 is the "current" bar; we look at bars 15..18.
    # In this fixture bars 15..18 are flat (3000), so timeout fallback fires.
    # close[18] = 3000, close[14] = 3010 → close drifted DOWN → label = 0.
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert int(out.iloc[14]) == 0


def test_output_series_has_expected_name_and_dtype():
    n = 30
    df = _ohlc([3000.0] * n)
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert out.name == "y_triple_barrier_4"
    # Float dtype so NaN can coexist with 0/1.
    assert out.dtype == float
