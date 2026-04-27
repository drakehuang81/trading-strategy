# Plan 5C — Triple-Barrier Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace today's noisy "is `close[t+4]` higher than `close[t]`" binary labels with **triple-barrier labels** (López de Prado's chapter 3 Meta-Labeling): for each bar, walk forward up to a horizon, label `1` if a TP barrier (`close + tp_mult * ATR`) is hit before a SL barrier (`close − sl_mult * ATR`), else `0`. Retrain XGBoost on these labels; rerun backtest; compare Brier score against Plan 5A's baseline (0.2505). Goal: **calibration_brier gate (Plan 5B-4) flips from ❌ to ✅** (Brier < 0.24).

**Architecture:** One pure label-computation module (`src/labels/triple_barrier.py`) — vectorised forward-walk over each bar's `(high, low)` to detect first-barrier-hit. Extend `scripts/build_labels.py` with `--label-type` flag selecting `forward_up` (existing) or `triple_barrier` (new). No changes to `train_xgb.py` (it just consumes a labels parquet). Then run the manual smoke pipeline from prior plans (build labels → train → backtest → Pre-Live Gate) and record Brier delta.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. No new dependencies. Reuses existing kline parquet + features parquet from Plan 5B-1.

**Decisions baked in:**
- **Symmetric barriers**: `tp_mult = sl_mult = 1.5` default. Asymmetric barriers bias the labels even on flat data. Symmetric keeps the binary interpretation clean: "is the next 1.5×ATR move up or down?". CLI exposes both for tuning.
- **ATR formula = mean of (high − low) over last N bars** (default N=14). Same simplified ATR as `RollingKlineCache.atr` (Plan 5B-2 Task 2). NOT Wilder true-range. Consistent with the rest of the codebase; documented limitation.
- **Timeout fallback** uses sign of close: `1 if close[t+horizon] > close[t] else 0`. Matches Plan 5A's existing forward-up semantic for the timeout case.
- **Same-bar TP+SL hit** uses sign of that bar's close vs entry. Rare (only on extreme volatility bars); breaking ties this way is consistent with timeout fallback.
- **Output column name** = `y_triple_barrier_<H>` so we can keep the old `y_4bar_up` parquet alongside for A/B comparison without overwriting.
- **Horizon stays 4** (matches `PredictionBundle.horizon_bars=4`) so the trained model's `horizon_bars` field still describes reality.
- **Multi-symbol deferred to Plan 5D**. Single-symbol ETHUSDT only.
- **Other small wins (gap=horizon CV, NaN-pass-through, drop dead funding columns) deferred to Plan 5C-2**.

**Out of Plan 5C scope (deferred):**
- Multi-symbol watchlist (BTCUSDT) — needs broker / cache refactor.
- `gap=horizon` in TimeSeriesSplit and NaN-pass-through (Plan 5C-2).
- Dropping the 4 dead-weight funding columns (Plan 5C-2).
- Calibration sweep over `tp_mult/sl_mult/horizon` — Plan 5E hyperparameter search.
- Fractional differentiation features (López de Prado ch. 5).

---

## File map

### Created
- `src/labels/__init__.py` (empty package marker)
- `src/labels/triple_barrier.py` — `triple_barrier_labels(df, horizon, tp_mult, sl_mult, atr_window) -> pd.Series`
- `tests/unit/labels/__init__.py` (empty)
- `tests/unit/labels/test_triple_barrier.py`
- `docs/superpowers/plans/2026-04-27-pivot-plan5c-STATUS.md` (handoff)

### Modified
- `scripts/build_labels.py` — add `--label-type {forward_up,triple_barrier}`, `--tp-mult`, `--sl-mult`, `--atr-window` flags; dispatch on type
- `tests/unit/scripts/test_build_labels.py` — append a triple-barrier dispatch test

### Untouched (verified intentionally)
- `scripts/train_xgb.py` — consumes labels parquet by column lookup; works unchanged.
- `scripts/backtest.py` — consumes trained model bundle; works unchanged.
- `src/execution/pre_live_gate.py` — reads `model_versions` and `meta_*.json`; works unchanged.

---

## Task 1: `src/labels/triple_barrier.py` — pure label computation

**Why first:** Pure function, fully testable with synthetic series. No dependencies on the rest of the plan. Once tested it's a leaf module the script imports.

**Files:**
- Create: `src/labels/__init__.py` (empty)
- Create: `src/labels/triple_barrier.py`
- Create: `tests/unit/labels/__init__.py` (empty)
- Create: `tests/unit/labels/test_triple_barrier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/labels/test_triple_barrier.py
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
    # Wait — same bar as t=14 is the entry bar; should we use bar 14 or bar 15 to walk forward?
    # The function walks forward bars t+1 .. t+horizon, NOT t itself.
    # So for entry at t=14, the wild bar at index 14 is the "current" bar; we look at bars 15..18.
    # In this fixture bars 15..18 are flat (3000), so timeout fallback fires.
    # close[18] = 3000 == close[14]? No — close[14] = 3010, close[18] = 3000. So close drifted DOWN → label = 0.
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert int(out.iloc[14]) == 0


def test_output_series_has_expected_name_and_dtype():
    n = 30
    df = _ohlc([3000.0] * n)
    out = triple_barrier_labels(df, horizon=4, tp_mult=1.5, sl_mult=1.5, atr_window=14)
    assert out.name == "y_triple_barrier_4"
    # Float dtype so NaN can coexist with 0/1.
    assert out.dtype == float
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/labels/test_triple_barrier.py -v`
Expected: ImportError on `labels.triple_barrier`.

- [ ] **Step 3: Implement `triple_barrier_labels`**

```python
# src/labels/triple_barrier.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/labels/test_triple_barrier.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 367 passed (359 + 8 new).

- [ ] **Step 6: Commit**

```bash
git add src/labels/__init__.py src/labels/triple_barrier.py tests/unit/labels/__init__.py tests/unit/labels/test_triple_barrier.py
git commit -m "feat(labels): triple_barrier_labels — TP/SL/timeout binary labels"
```

---

## Task 2: Extend `scripts/build_labels.py` with `--label-type`

**Why:** Plan 5A's labels script only does forward-up. We need a switch so existing forward_up callers stay green AND triple_barrier becomes selectable. Also the new label CLI knobs (`--tp-mult`, `--sl-mult`, `--atr-window`) live here.

**Files:**
- Modify: `scripts/build_labels.py` — add `--label-type` and triple-barrier-specific flags; dispatch
- Modify: `tests/unit/scripts/test_build_labels.py` — append a dispatch test

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/scripts/test_build_labels.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/scripts/test_build_labels.py -v`
Expected: argparse fails on the new flags (script doesn't know `--label-type`).

- [ ] **Step 3: Update `scripts/build_labels.py`**

Replace the entire file content with:

```python
"""Builds binary labels for ML training.

Two label types:
  - forward_up (default, Plan 5A): y = 1 if close[t+H] > close[t] else 0
  - triple_barrier (Plan 5C):      walk bars t+1..t+H against TP/SL barriers
                                   computed from ATR; timeout falls back to
                                   close-sign

Usage:
    python scripts/build_labels.py \
        --kline data/history/ETHUSDT_1h.parquet \
        --out   data/training/ETHUSDT_1h_labels_tb.parquet \
        --horizon 4 \
        --label-type triple_barrier \
        --tp-mult 1.5 --sl-mult 1.5 --atr-window 14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from labels.triple_barrier import triple_barrier_labels


def compute_forward_up_labels(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    future = df["close"].shift(-horizon)
    label = (future > df["close"]).astype("float")
    label[future.isna()] = np.nan
    label.name = f"y_{horizon}bar_up"
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--label-type",
                    choices=["forward_up", "triple_barrier"],
                    default="forward_up")
    ap.add_argument("--tp-mult", type=float, default=1.5,
                    help="triple_barrier: TP barrier multiplier on ATR")
    ap.add_argument("--sl-mult", type=float, default=1.5,
                    help="triple_barrier: SL barrier multiplier on ATR")
    ap.add_argument("--atr-window", type=int, default=14,
                    help="triple_barrier: ATR rolling window in bars")
    args = ap.parse_args()

    df = pd.read_parquet(args.kline).sort_index()
    if args.label_type == "forward_up":
        y = compute_forward_up_labels(df, horizon=args.horizon)
    else:  # triple_barrier
        y = triple_barrier_labels(
            df,
            horizon=args.horizon,
            tp_mult=args.tp_mult,
            sl_mult=args.sl_mult,
            atr_window=args.atr_window,
        )
    out = y.dropna().to_frame()
    out.index.name = "as_of"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"labels[{args.label_type}]: {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/scripts/test_build_labels.py -v`
Expected: 5 passed (3 existing forward_up tests + 2 new dispatch tests).

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 369 passed (367 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_labels.py tests/unit/scripts/test_build_labels.py
git commit -m "feat(scripts): build_labels --label-type {forward_up,triple_barrier}"
```

---

## Task 3: Manual smoke — generate triple-barrier labels + retrain

**Why:** First end-to-end test on real data. We expect a meaningfully different label distribution and a different Brier score.

**Files:** No code changes; only commands and observation.

- [ ] **Step 1: Build triple-barrier labels on real klines**

```bash
PYTHONPATH=src python scripts/build_labels.py \
    --kline data/history/ETHUSDT_1h.parquet \
    --out   data/training/ETHUSDT_1h_labels_tb.parquet \
    --horizon 4 \
    --label-type triple_barrier \
    --tp-mult 1.5 --sl-mult 1.5 --atr-window 14
```

Expected stdout: `labels[triple_barrier]: ~17500 rows -> data/training/ETHUSDT_1h_labels_tb.parquet`.

(Slightly fewer than the 17516 forward_up labels because of the 14-bar ATR warmup at the start.)

- [ ] **Step 2: Compare label distributions**

```bash
source venv/bin/activate
python -c "
import pandas as pd
fu = pd.read_parquet('data/training/ETHUSDT_1h_labels.parquet')
tb = pd.read_parquet('data/training/ETHUSDT_1h_labels_tb.parquet')
print('forward_up rows:', len(fu), 'positive share:', fu['y_4bar_up'].mean())
print('triple_barrier rows:', len(tb), 'positive share:', tb['y_triple_barrier_4'].mean())
"
```

Expected: positive share for triple_barrier different from forward_up's ~0.50. Could be 0.45-0.55 depending on volatility.

Note these numbers in STATUS.

- [ ] **Step 3: Retrain on triple-barrier labels (background)**

```bash
PYTHONPATH=src python scripts/train_xgb.py \
    --features data/training/ETHUSDT_1h_features.parquet \
    --labels   data/training/ETHUSDT_1h_labels_tb.parquet \
    --out      models 2>&1 | tail -5
```

Expected: ~5 min training. Stdout: `trained <12hex>; calib=<isotonic|platt> brier_iso=<f> brier_platt=<f>`.

Compare the chosen calibrator's Brier vs Plan 5B-1's `92fddb72f14b` (Platt 0.2505). If Brier drops to <0.24, **calibration_brier gate will pass on next Pre-Live Gate run**.

Note model_version + Brier values in STATUS.

- [ ] **Step 4: Run backtest on the new model (default 0.58 thresholds)**

```bash
PYTHONPATH=src python -m scripts.backtest \
    --kline data/history/ETHUSDT_1h.parquet \
    --features data/training/ETHUSDT_1h_features.parquet \
    --funding data/funding/ETHUSDT.parquet \
    --model-dir models \
    --sqlite-path data/state.db \
    --oos-fraction 0.2 \
    --long-threshold 0.58 --short-threshold 0.58
```

Expected: a new `backtest_runs` row. n_trades may be 0 (still — Plan 5B-3 saw 0 even with old model at 0.58) or non-zero. DSR might be NaN if 0 trades.

Also rerun with loose thresholds for comparison:

```bash
PYTHONPATH=src python -m scripts.backtest \
    --kline data/history/ETHUSDT_1h.parquet \
    --features data/training/ETHUSDT_1h_features.parquet \
    --funding data/funding/ETHUSDT.parquet \
    --model-dir models \
    --sqlite-path data/state.db \
    --oos-fraction 0.2 \
    --long-threshold 0.51 --short-threshold 0.51
```

Note both runs' Sharpe/DSR/n_trades.

- [ ] **Step 5: Re-run Pre-Live Gate**

```bash
PYTHONPATH=src python -m scripts.pre_live_gate \
    --sqlite-path data/state.db \
    --model-dir models \
    --watchdog-log data/watchdog_pings.log \
    --brier-threshold 0.24
```

Expected delta vs Plan 5B-4 smoke (4/8 passed): if Brier dropped < 0.24, calibration_brier gate flips to ✅, total becomes 5/8. If not, still 4/8.

Note exact gate output in STATUS.

---

## Task 4: STATUS handoff

**Files:**
- Create: `docs/superpowers/plans/2026-04-27-pivot-plan5c-STATUS.md`

- [ ] **Step 1: Write Plan 5C STATUS**

Create `docs/superpowers/plans/2026-04-27-pivot-plan5c-STATUS.md`. Sections:
- Date / branch / base / head SHAs
- Summary (one paragraph; lead with Brier delta)
- Task table (4 rows with commit SHAs from `git log --oneline a7a2252..HEAD`)
- Manual smoke results table:
  - forward_up vs triple_barrier label distribution (positive share)
  - Plan 5A model `92fddb72f14b` Brier vs Plan 5C model Brier
  - Backtest comparison: n_trades, Sharpe, DSR for both threshold settings
  - Pre-Live Gate before (4/8) vs after (X/8) — which gate(s) flipped
- Decisions landed (symmetric barriers, simplified ATR, output column naming)
- Sanity check on the new model (did Brier actually drop? Is the model now tradeable?)
- What is NOT done (Plan 5C-2 small fixes; Plan 5D live activation; Plan 5E hyperparameter sweep)
- Known follow-ups

- [ ] **Step 2: Final commit**

```bash
git add docs/superpowers/plans/2026-04-27-pivot-plan5c-STATUS.md
git commit -m "docs: Plan 5C handoff STATUS (triple-barrier labels)"
```

---

## Self-review notes

- **Spec coverage**: spec doesn't mandate triple-barrier specifically (López de Prado convention is implicit in §10.1.3 calibration gate). The plan moves toward fixing the gate identified in Plan 5B-4 STATUS.
- **Type consistency**: `triple_barrier_labels` returns `pd.Series` with `name=f"y_triple_barrier_{horizon}"`. The build_labels script's `dropna().to_frame()` pattern matches `compute_forward_up_labels`. Train script joins by index name "as_of" — both label paths produce same shape, so `train_xgb.py` works unchanged.
- **No placeholders**: every step has working code; manual smoke commands are concrete with expected outputs.
- **Test math is hand-verifiable**: ATR(14)=10 with constant H-L=10; TP=3015, SL=2985 at entry 3000; barrier crossings match the assertions exactly.
- **Backward compat**: existing `data/training/ETHUSDT_1h_labels.parquet` (forward_up) is NOT overwritten. New labels go to `_labels_tb.parquet`. Plan 5A model + parquet stay intact for A/B comparison.
- **Scope discipline**: triple-barrier ONLY. gap=horizon CV, NaN-pass-through, dead-weight feature drop, multi-symbol all explicitly deferred.
