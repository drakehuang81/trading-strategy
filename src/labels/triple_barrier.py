"""Triple-barrier labels — Plan 5C Task 1.

For each bar t with valid ATR[t]:
  TP = close[t] + tp_mult * ATR[t]
  SL = close[t] - sl_mult * ATR[t]

Walk bars t+1 .. t+horizon:
  - first bar with high >= TP wins → label = 1
  - first bar with low  <= SL wins → label = 0
  - same bar with both barriers crossed → use that bar's close sign vs entry
  - no bar crosses (timeout) → label = (close[t+horizon] > close[t])

Returns Series of 0.0/1.0 with NaN for trailing `horizon` rows AND
the leading `atr_window` rows (no ATR yet).

ATR is the simplified mean(high-low) over `atr_window` bars (matching
RollingKlineCache.atr from Plan 5B-2). Not Wilder; gaps ignored.
Documented limitation; consistent with the rest of the codebase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon: int = 4,
    tp_mult: float = 1.5,
    sl_mult: float = 1.5,
    atr_window: int = 14,
) -> pd.Series:
    if not {"close", "high", "low"}.issubset(df.columns):
        raise ValueError("df must have close/high/low columns")

    n = len(df)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    # Simple ATR = rolling mean of (high - low). Index t's value uses the
    # window ending AT t inclusive, matching RollingKlineCache.atr.
    range_hl = (df["high"] - df["low"]).to_numpy()
    atr = pd.Series(range_hl).rolling(atr_window, min_periods=atr_window).mean().to_numpy()

    out = np.full(n, np.nan, dtype=float)

    for t in range(n):
        if np.isnan(atr[t]):
            continue
        # Skip the first atr_window rows: bar t=atr_window-1 has the first valid
        # rolling ATR but the plan convention treats those as "warm-up" rows.
        if t < atr_window:
            continue
        if t + horizon >= n:
            continue   # not enough lookahead

        entry = close[t]
        tp = entry + tp_mult * atr[t]
        sl = entry - sl_mult * atr[t]

        labeled = False
        for k in range(1, horizon + 1):
            tp_hit = high[t + k] >= tp
            sl_hit = low[t + k] <= sl
            if tp_hit and sl_hit:
                # Same-bar tie: use that bar's close sign vs entry.
                out[t] = 1.0 if close[t + k] > entry else 0.0
                labeled = True
                break
            if tp_hit:
                out[t] = 1.0
                labeled = True
                break
            if sl_hit:
                out[t] = 0.0
                labeled = True
                break
        if not labeled:
            # Timeout: fall back to sign of close at horizon end.
            out[t] = 1.0 if close[t + horizon] > entry else 0.0

    return pd.Series(out, index=df.index, name=f"y_triple_barrier_{horizon}")
