# Order Book Recon — Phase 2a (Data Layer + Single-Asset Signals) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the bookDepth/aggTrades data loaders and the four single-asset microstructure signals (OFI, depth imbalance, book slope, taker imbalance), fix the two final-review robustness issues + the Step-0 probe bug, and wire all signals into a multi-signal IC pipeline — so we can produce an IC-vs-horizon table across every single-asset signal on the bookTicker-overlap window.

**Architecture:** Extends the Phase-1 `research.microstructure` module (already merged to main). Each signal is a lightweight pure function taking a normalized DataFrame and returning `(ts, <signal>)`; all keep the no-lookahead discipline with a point-in-time test. bookDepth is loaded in long form (one row per (ts, percentage) level) and signals aggregate per timestamp. The multi-signal pipeline as-of joins every signal onto the mid grid and computes IC per (signal × horizon).

**Tech Stack:** Python 3.11, polars (lazy/chunked), scipy, pytest. Spec: [docs/superpowers/specs/2026-06-28-orderbook-microstructure-recon-design.md](../specs/2026-06-28-orderbook-microstructure-recon-design.md). Builds on Phase 1 (merged at `f5f6285`).

---

## Step-0 Findings (parameterize this plan)

- **Recon window is locked to 2023-05-16 → 2024-03-30** (bookTicker only exists in that range; bookDepth covers 2023-01→2026+, aggTrades 2019→2026+). All cross-signal comparison happens inside this overlap.
- **bookDepth is percentage-distance depth, not raw L2**: columns `timestamp`(str like `"2026-06-01 00:00:07"`), `percentage`(f64), `depth`(f64), `notional`(f64); 12 symmetric levels `±0.2/1/2/3/4/5%`; ~33s per snapshot (12 rows share one timestamp).
- **aggTrades** columns: `agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time`(epoch ms int), `is_buyer_maker`(bool). `is_buyer_maker=True` ⇒ the buyer was the maker ⇒ the **taker sold**; `False` ⇒ taker bought.

## Environment notes (read first)

- Work in a clean worktree off latest `main` with a fresh 3.11 venv (main repo venv is stale 3.9). Phase-1 deps + `polars` are required: `pandas numpy pyarrow scipy pytest pytest-asyncio requests polars`.
- Verify imports via **pytest** (the `pythonpath=[".","src"]` is pytest-only). Run ONLY `./venv/bin/pytest tests/research/microstructure/ -v` (scoped — full suite needs production deps absent from the lean venv).
- Commit convention: conventional-commits prefix + trailing `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- No imports from production layers (`decision/`, `execution/`, `models/`, etc.) — research-only.

## File Structure

| Path | Responsibility | Change |
|---|---|---|
| `src/research/microstructure/schema_probe.py` | str-timestamp robust gap calc | modify |
| `src/research/microstructure/download.py` | add `load_book_depth`, `load_agg_trades` + maps | modify |
| `src/research/microstructure/signals.py` | add `ofi`, `depth_imbalance`, `book_slope`, `taker_imbalance` | modify |
| `src/research/microstructure/ic.py` | NaN filtering in `compute_ic` | modify |
| `src/research/microstructure/report.py` | union horizons + empty/missing guard | modify |
| `src/research/microstructure/pipeline.py` | `recon_multi` over many signals + multi-day | **create** |
| `tests/research/microstructure/test_*.py` | one test file per change | modify/create |

Note: `recon_from_book` (Phase 1) currently lives in `scripts/recon/run_recon.py`. This plan adds the multi-signal driver in a new `src/research/microstructure/pipeline.py` (research module, importable + testable) and leaves the Phase-1 thin CLI untouched.

---

### Task 1: Fix schema_probe for string timestamps

**Files:**
- Modify: `src/research/microstructure/schema_probe.py`
- Test: `tests/research/microstructure/test_schema_probe.py`

- [ ] **Step 1: Add a failing test for str-timestamp cadence**

Append to `tests/research/microstructure/test_schema_probe.py`:

```python
def test_summarize_schema_handles_string_timestamps():
    # bookDepth ships timestamp as a string; gap calc must not crash
    df = pl.DataFrame({
        "timestamp": ["2026-06-01 00:00:00", "2026-06-01 00:00:33", "2026-06-01 00:01:06"],
        "percentage": [1.0, 1.0, 1.0],
        "depth": [10.0, 11.0, 12.0],
    })
    summary = summarize_schema(df, ts_col="timestamp")
    assert summary["n_rows"] == 3
    assert summary["median_gap_ms"] == 33_000  # parsed, not crashed
```

- [ ] **Step 2: Run to verify it fails**

Run: `./venv/bin/pytest tests/research/microstructure/test_schema_probe.py::test_summarize_schema_handles_string_timestamps -v`
Expected: FAIL with `polars` error `sub operation not supported for dtypes str and str`.

- [ ] **Step 3: Make the gap calc parse strings first**

In `src/research/microstructure/schema_probe.py`, replace the gap-calc block inside `summarize_schema`:

```python
    if ts_col in df.columns and df.height > 1:
        col = df[ts_col]
        if col.dtype == pl.String:
            # bookDepth ships ts as "YYYY-MM-DD HH:MM:SS" — parse to datetime
            col = col.str.to_datetime(strict=False)
        if col.dtype.is_temporal():
            gaps_ms = col.sort().diff().drop_nulls().dt.total_milliseconds()
            summary["median_gap_ms"] = int(gaps_ms.median())
        else:
            gaps = col.sort().diff().drop_nulls()
            summary["median_gap_ms"] = int(gaps.median())
    else:
        summary["median_gap_ms"] = None
```

- [ ] **Step 4: Run to verify PASS** (both old int-ts and new str-ts tests)

Run: `./venv/bin/pytest tests/research/microstructure/test_schema_probe.py -v`
Expected: PASS. If `dt.total_milliseconds()` differs in polars 1.42, use `.dt.total_nanoseconds() // 1_000_000`.

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/schema_probe.py tests/research/microstructure/test_schema_probe.py
git commit -m "$(printf 'fix(recon): schema_probe handles string timestamps (bookDepth)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: bookDepth loader (parse + normalize, long form)

**Files:**
- Modify: `src/research/microstructure/download.py`
- Test: `tests/research/microstructure/test_download.py`

- [ ] **Step 1: Failing test**

Append to `tests/research/microstructure/test_download.py`:

```python
def test_load_book_depth_normalizes(tmp_path: Path):
    raw = pl.DataFrame({
        "timestamp": ["2026-06-01 00:00:07", "2026-06-01 00:00:07"],
        "percentage": [-1.0, 1.0],
        "depth": [42881.4, 28062.9],
        "notional": [8.5e7, 5.6e7],
    })
    p = tmp_path / "bd.parquet"
    raw.write_parquet(p)
    from research.microstructure.download import load_book_depth
    df = load_book_depth(p)
    assert df.columns == ["ts", "percentage", "depth", "notional"]
    assert df["ts"].dtype == pl.Datetime
    assert df.height == 2
```

- [ ] **Step 2: Run to verify FAIL** (ImportError).

Run: `./venv/bin/pytest tests/research/microstructure/test_download.py::test_load_book_depth_normalizes -v`

- [ ] **Step 3: Implement `load_book_depth`**

Append to `src/research/microstructure/download.py`:

```python
def load_book_depth(parquet_path: Path) -> pl.DataFrame:
    """Load + normalize a bookDepth parquet (long form: one row per level).

    Output: ts(Datetime), percentage, depth, notional. Timestamp is parsed
    from the raw "YYYY-MM-DD HH:MM:SS" string. 12 symmetric percentage levels
    (+/-0.2/1/2/3/4/5) share each ts.
    """
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime(strict=False).alias("ts")
    )
    return df.select(["ts", "percentage", "depth", "notional"])
```

- [ ] **Step 4: Run to verify PASS.**

Run: `./venv/bin/pytest tests/research/microstructure/test_download.py -v`
If `str.to_datetime` needs a format, use `format="%Y-%m-%d %H:%M:%S"`.

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/download.py tests/research/microstructure/test_download.py
git commit -m "$(printf 'feat(recon): bookDepth loader (parse str ts, long form)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 3: aggTrades loader (taker side)

**Files:**
- Modify: `src/research/microstructure/download.py`
- Test: `tests/research/microstructure/test_download.py`

- [ ] **Step 1: Failing test**

Append to `tests/research/microstructure/test_download.py`:

```python
def test_load_agg_trades_normalizes_and_signs_taker(tmp_path: Path):
    raw = pl.DataFrame({
        "agg_trade_id": [1, 2],
        "price": [2006.3, 2006.4],
        "quantity": [0.36, 1.0],
        "first_trade_id": [10, 11],
        "last_trade_id": [10, 12],
        "transact_time": [1780272000067, 1780272000357],
        "is_buyer_maker": [True, False],
    })
    p = tmp_path / "agg.parquet"
    raw.write_parquet(p)
    from research.microstructure.download import load_agg_trades
    df = load_agg_trades(p)
    assert df.columns == ["ts", "price", "qty", "taker_buy_qty", "taker_sell_qty"]
    assert df["ts"].dtype == pl.Datetime
    # is_buyer_maker=True -> taker SOLD: taker_sell_qty=0.36, taker_buy_qty=0
    assert df["taker_sell_qty"][0] == 0.36
    assert df["taker_buy_qty"][0] == 0.0
    # is_buyer_maker=False -> taker BOUGHT
    assert df["taker_buy_qty"][1] == 1.0
    assert df["taker_sell_qty"][1] == 0.0
```

- [ ] **Step 2: Run to verify FAIL** (ImportError).

- [ ] **Step 3: Implement `load_agg_trades`**

Append to `src/research/microstructure/download.py`:

```python
def load_agg_trades(parquet_path: Path) -> pl.DataFrame:
    """Load + normalize aggTrades, splitting volume into taker buy/sell.

    is_buyer_maker=True  => buyer is maker => taker SOLD  -> taker_sell_qty
    is_buyer_maker=False => buyer is taker => taker BOUGHT -> taker_buy_qty
    Output: ts(Datetime), price, qty, taker_buy_qty, taker_sell_qty.
    """
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(
        pl.from_epoch(pl.col("transact_time"), time_unit="ms").alias("ts"),
        pl.col("quantity").alias("qty"),
    )
    df = df.with_columns(
        pl.when(pl.col("is_buyer_maker"))
        .then(0.0).otherwise(pl.col("qty")).alias("taker_buy_qty"),
        pl.when(pl.col("is_buyer_maker"))
        .then(pl.col("qty")).otherwise(0.0).alias("taker_sell_qty"),
    )
    return df.select(["ts", "price", "qty", "taker_buy_qty", "taker_sell_qty"])
```

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/download.py tests/research/microstructure/test_download.py
git commit -m "$(printf 'feat(recon): aggTrades loader with taker buy/sell split\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 4: Depth imbalance signal

**Files:**
- Modify: `src/research/microstructure/signals.py`
- Test: `tests/research/microstructure/test_signals.py`

- [ ] **Step 1: Failing tests**

Append to `tests/research/microstructure/test_signals.py`:

```python
def _depth(ts_rows):
    # ts_rows: list of (ts, percentage, depth)
    import datetime as _dt
    return pl.DataFrame({
        "ts": [r[0] for r in ts_rows],
        "percentage": [r[1] for r in ts_rows],
        "depth": [r[2] for r in ts_rows],
    })


def test_depth_imbalance_per_snapshot():
    from research.microstructure.signals import depth_imbalance
    t = dt.datetime(2026, 1, 1, 0, 0, 0)
    # bid side (neg pct) total depth 30, ask side (pos pct) total depth 10
    book = _depth([
        (t, -1.0, 20.0), (t, -0.2, 10.0), (t, 0.2, 4.0), (t, 1.0, 6.0),
    ])
    di = depth_imbalance(book)
    # (30 - 10) / (30 + 10) = 0.5
    assert di.height == 1
    assert abs(di["depth_imbalance"][0] - 0.5) < 1e-12


def test_depth_imbalance_zero_total_is_null():
    from research.microstructure.signals import depth_imbalance
    t = dt.datetime(2026, 1, 1, 0, 0, 0)
    book = _depth([(t, -1.0, 0.0), (t, 1.0, 0.0)])
    di = depth_imbalance(book)
    assert di["depth_imbalance"][0] is None
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement `depth_imbalance`**

Append to `src/research/microstructure/signals.py`:

```python
def depth_imbalance(book_depth: pl.DataFrame) -> pl.DataFrame:
    """Per-snapshot depth imbalance from percentage-distance depth.

    bid = sum(depth where percentage < 0), ask = sum(depth where percentage > 0).
    DI = (bid - ask) / (bid + ask); null when total is zero.
    Input: ts, percentage, depth (long form). Output: ts, depth_imbalance.
    """
    g = book_depth.group_by("ts").agg(
        pl.col("depth").filter(pl.col("percentage") < 0).sum().alias("bid"),
        pl.col("depth").filter(pl.col("percentage") > 0).sum().alias("ask"),
    )
    total = pl.col("bid") + pl.col("ask")
    di = (
        pl.when(total > 0)
        .then((pl.col("bid") - pl.col("ask")) / total)
        .otherwise(None)
        .alias("depth_imbalance")
    )
    return g.select(["ts", di]).sort("ts")
```

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/signals.py tests/research/microstructure/test_signals.py
git commit -m "$(printf 'feat(recon): depth imbalance signal (percentage-depth)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 5: Book slope signal

**Files:**
- Modify: `src/research/microstructure/signals.py`
- Test: `tests/research/microstructure/test_signals.py`

- [ ] **Step 1: Failing test**

Append to `tests/research/microstructure/test_signals.py`:

```python
def test_book_slope_log_far_near_ratio():
    from research.microstructure.signals import book_slope
    t = dt.datetime(2026, 1, 1, 0, 0, 0)
    # near (|pct| <= 1) total depth = 10+10 = 20; far (|pct| >= 3) = 40+40 = 80
    book = _depth([
        (t, -1.0, 10.0), (t, 1.0, 10.0),
        (t, -3.0, 40.0), (t, 3.0, 40.0),
    ])
    bs = book_slope(book)
    # log(80 / 20) = log(4)
    import math
    assert abs(bs["book_slope"][0] - math.log(4.0)) < 1e-9
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement `book_slope`**

Append to `src/research/microstructure/signals.py`:

```python
def book_slope(
    book_depth: pl.DataFrame, *, near_max: float = 1.0, far_min: float = 3.0
) -> pl.DataFrame:
    """Liquidity concentration: log(far_depth / near_depth) per snapshot.

    near = sum(depth where |percentage| <= near_max),
    far  = sum(depth where |percentage| >= far_min).
    Positive slope = depth concentrated away from mid. Null if either is 0.
    Input: ts, percentage, depth. Output: ts, book_slope.
    """
    absp = pl.col("percentage").abs()
    g = book_depth.group_by("ts").agg(
        pl.col("depth").filter(absp <= near_max).sum().alias("near"),
        pl.col("depth").filter(absp >= far_min).sum().alias("far"),
    )
    slope = (
        pl.when((pl.col("near") > 0) & (pl.col("far") > 0))
        .then((pl.col("far") / pl.col("near")).log())
        .otherwise(None)
        .alias("book_slope")
    )
    return g.select(["ts", slope]).sort("ts")
```

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/signals.py tests/research/microstructure/test_signals.py
git commit -m "$(printf 'feat(recon): book slope signal (far/near depth ratio)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 6: OFI signal (Cont 2014, from bookTicker)

**Files:**
- Modify: `src/research/microstructure/signals.py`
- Test: `tests/research/microstructure/test_signals.py`

- [ ] **Step 1: Failing test**

Append to `tests/research/microstructure/test_signals.py`:

```python
def test_ofi_cont_contributions():
    from research.microstructure.signals import ofi
    # rows: (bid_price, bid_qty, ask_price, ask_qty)
    book = _book([
        (100.0, 5.0, 100.5, 5.0),   # t0 baseline
        (100.0, 8.0, 100.5, 5.0),   # bid_qty up, bid_price same -> +3 bid contrib
        (101.0, 2.0, 101.5, 4.0),   # bid_price up -> +bid_qty(2); ask_price up -> +prev ask_qty(5)
    ])
    out = ofi(book, window=10)
    # rolling-sum OFI; first row null (no prev). Check it's defined for later rows.
    assert "ofi" in out.columns
    assert out["ofi"][0] is None
    # e_1 = +3 (bid up by 3, ask unchanged contributes 0)
    assert abs(out["ofi"][1] - 3.0) < 1e-9
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement `ofi`**

Append to `src/research/microstructure/signals.py` (uses the Cont-Kukanov-Stoikov order-flow per-update term, then a rolling sum):

```python
def ofi(book: pl.DataFrame, *, window: int = 50) -> pl.DataFrame:
    """Order Flow Imbalance (Cont, Kukanov & Stoikov 2014) from best bid/ask.

    Per update n (vs n-1):
      bid term  = bid_qty_n            if bid_price_n  > bid_price_{n-1}
                  bid_qty_n - bid_qty_{n-1} if equal
                  -bid_qty_{n-1}       if bid_price_n  < bid_price_{n-1}
      ask term  = -ask_qty_n           if ask_price_n  > ask_price_{n-1}  (mirror)
                  ask_qty_n - ask_qty_{n-1} if equal
                  ask_qty_{n-1}        if ask_price_n  < ask_price_{n-1}
      e_n = bid term + ask term
    ofi = rolling_sum(e_n, window). First row is null (no previous update).
    Input: ts, bid_price, bid_qty, ask_price, ask_qty. Output: ts, ofi.
    """
    b = book.sort("ts")
    pbid, qbid = pl.col("bid_price"), pl.col("bid_qty")
    pask, qask = pl.col("ask_price"), pl.col("ask_qty")
    pbid0, qbid0 = pbid.shift(1), qbid.shift(1)
    pask0, qask0 = pask.shift(1), qask.shift(1)

    bid_term = (
        pl.when(pbid > pbid0).then(qbid)
        .when(pbid == pbid0).then(qbid - qbid0)
        .otherwise(-qbid0)
    )
    ask_term = (
        pl.when(pask > pask0).then(-qask)
        .when(pask == pask0).then(qask - qask0)
        .otherwise(qask0)
    )
    b = b.with_columns((bid_term + ask_term).alias("_e"))
    # rolling_sum over a window containing nulls can return null; fill the
    # per-update term with 0 for the sum, but keep the first row's ofi null
    # (no previous update exists there).
    b = b.with_columns(pl.col("_e").fill_null(0.0).alias("_e_filled"))
    out = b.with_columns(
        pl.col("_e_filled").rolling_sum(window_size=window, min_periods=1).alias("ofi")
    )
    out = out.with_columns(
        pl.when(pl.col("_e").is_null()).then(None).otherwise(pl.col("ofi")).alias("ofi")
    )
    return out.select(["ts", "ofi"])
```

- [ ] **Step 4: Run to verify PASS.**

Run: `./venv/bin/pytest tests/research/microstructure/test_signals.py::test_ofi_cont_contributions -v`
If `rolling_sum` kwarg names differ in polars 1.42, use `.rolling_sum(window)` positionally.

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/signals.py tests/research/microstructure/test_signals.py
git commit -m "$(printf 'feat(recon): OFI signal (Cont 2014) from bookTicker\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 7: Taker imbalance signal (aggTrades)

**Files:**
- Modify: `src/research/microstructure/signals.py`
- Test: `tests/research/microstructure/test_signals.py`

- [ ] **Step 1: Failing test**

Append to `tests/research/microstructure/test_signals.py`:

```python
def test_taker_imbalance_rolling():
    from research.microstructure.signals import taker_imbalance
    t0 = dt.datetime(2026, 1, 1, 0, 0, 0)
    trades = pl.DataFrame({
        "ts": [t0 + dt.timedelta(seconds=s) for s in range(3)],
        "taker_buy_qty": [3.0, 0.0, 6.0],
        "taker_sell_qty": [1.0, 4.0, 0.0],
    })
    out = taker_imbalance(trades, window=3)
    # cumulative within window=3 at last row: buy=9, sell=5 -> (9-5)/14
    assert abs(out["taker_imbalance"][2] - (4.0 / 14.0)) < 1e-12
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement `taker_imbalance`**

Append to `src/research/microstructure/signals.py`:

```python
def taker_imbalance(trades: pl.DataFrame, *, window: int = 100) -> pl.DataFrame:
    """Rolling taker buy/sell imbalance from aggTrades.

    TI = (rolling_buy - rolling_sell) / (rolling_buy + rolling_sell) over the
    last `window` trades; null when the rolling total is zero.
    Input: ts, taker_buy_qty, taker_sell_qty. Output: ts, taker_imbalance.
    """
    t = trades.sort("ts").with_columns(
        pl.col("taker_buy_qty").rolling_sum(window_size=window, min_periods=1).alias("_b"),
        pl.col("taker_sell_qty").rolling_sum(window_size=window, min_periods=1).alias("_s"),
    )
    total = pl.col("_b") + pl.col("_s")
    ti = (
        pl.when(total > 0)
        .then((pl.col("_b") - pl.col("_s")) / total)
        .otherwise(None)
        .alias("taker_imbalance")
    )
    return t.select(["ts", ti])
```

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/signals.py tests/research/microstructure/test_signals.py
git commit -m "$(printf 'feat(recon): taker imbalance signal (aggTrades rolling)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 8: Fix render_ic_markdown (final-review #2)

**Files:**
- Modify: `src/research/microstructure/report.py`
- Test: `tests/research/microstructure/test_report.py`

- [ ] **Step 1: Failing tests**

Append to `tests/research/microstructure/test_report.py`:

```python
def test_render_ic_markdown_heterogeneous_horizons():
    from research.microstructure.report import render_ic_markdown
    ic = {"qi": {"fwd_1s": 0.03, "fwd_60s": -0.01}, "ofi": {"fwd_1s": 0.05}}
    md = render_ic_markdown(ic, n_tests=3)
    assert "fwd_1s" in md and "fwd_60s" in md
    assert "nan" in md.lower()  # ofi missing fwd_60s -> nan cell


def test_render_ic_markdown_empty_dict():
    from research.microstructure.report import render_ic_markdown
    md = render_ic_markdown({}, n_tests=0)
    assert "no signals" in md.lower()
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Rewrite `render_ic_markdown`**

Replace the body of `src/research/microstructure/report.py`'s `render_ic_markdown`:

```python
def render_ic_markdown(
    ic_by_signal: dict[str, dict[str, float]], *, n_tests: int
) -> str:
    if not ic_by_signal:
        return "_no signals to report_"
    # union of all horizons, preserving first-seen order
    horizons: list[str] = list(
        dict.fromkeys(h for ic in ic_by_signal.values() for h in ic)
    )
    header = "| signal | " + " | ".join(horizons) + " |"
    sep = "|" + "---|" * (len(horizons) + 1)
    lines = [header, sep]
    for sig, ic in ic_by_signal.items():
        cells = " | ".join(f"{ic.get(h, float('nan')):.3f}" for h in horizons)
        lines.append(f"| {sig} | {cells} |")
    lines.append("")
    lines.append(f"_tests run: {n_tests} (multiple-testing guard — see spec §7)_")
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify PASS** (old + new report tests).

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/report.py tests/research/microstructure/test_report.py
git commit -m "$(printf 'fix(recon): render_ic_markdown union horizons + empty guard (review #2)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 9: NaN filtering in compute_ic (final-review #3)

**Files:**
- Modify: `src/research/microstructure/ic.py`
- Test: `tests/research/microstructure/test_ic.py`

- [ ] **Step 1: Failing test**

Append to `tests/research/microstructure/test_ic.py`:

```python
def test_compute_ic_drops_nan_not_just_null():
    from research.microstructure.ic import compute_ic
    # one NaN in the signal must be excluded, not poison the whole horizon
    sig = [float("nan")] + list(np.linspace(-1, 1, 100))
    fwd = [0.0] + list(np.linspace(-1, 1, 100) * 2.0)
    df = pl.DataFrame({"qi": sig, "fwd_1s": fwd})
    ic = compute_ic(df, signal_col="qi", horizon_cols=["fwd_1s"])
    assert abs(ic["fwd_1s"] - 1.0) < 1e-6  # ~1.0, not nan
```

- [ ] **Step 2: Run to verify FAIL** (IC comes back nan).

- [ ] **Step 3: Add finite filtering in `compute_ic`**

In `src/research/microstructure/ic.py`, change the pair-building line inside `compute_ic`:

```python
        pair = (
            df.select([signal_col, hcol])
            .drop_nulls()
            .filter(pl.col(signal_col).is_finite() & pl.col(hcol).is_finite())
        )
```

- [ ] **Step 4: Run to verify PASS** (old IC tests + new).

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/ic.py tests/research/microstructure/test_ic.py
git commit -m "$(printf 'fix(recon): compute_ic filters NaN, not just null (review #3)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 10: Multi-signal pipeline (recon_multi)

**Files:**
- Create: `src/research/microstructure/pipeline.py`
- Test: `tests/research/microstructure/test_pipeline.py`

- [ ] **Step 1: Failing test**

Create `tests/research/microstructure/test_pipeline.py`:

```python
import datetime as dt

import polars as pl

from research.microstructure.pipeline import recon_multi


def test_recon_multi_attaches_each_signal_and_computes_ic():
    # grid with mid + one signal series ("qi") already aligned
    ts = [dt.datetime(2026, 1, 1, 0, 0, s) for s in range(60)]
    grid = pl.DataFrame({"ts": ts, "mid": [100.0 + (s % 2) for s in range(60)]})
    qi = pl.DataFrame({"ts": ts, "qi": [0.5 if s % 2 == 0 else -0.5 for s in range(60)]})
    md, ic = recon_multi(grid, {"qi": qi}, horizons_secs=[1, 5])
    assert set(ic.keys()) == {"qi"}
    assert "fwd_1s" in ic["qi"] and "fwd_5s" in ic["qi"]
    assert "signal" in md
```

- [ ] **Step 2: Run to verify FAIL** (ImportError).

- [ ] **Step 3: Implement `recon_multi`**

Create `src/research/microstructure/pipeline.py`:

```python
"""Multi-signal recon driver: attach each signal series onto a mid grid via
backward as-of join, compute forward returns, then IC per (signal x horizon).

Each signal df is (ts, <name>); the grid is (ts, mid) on a uniform 1s axis.
"""
from __future__ import annotations

import polars as pl

from research.microstructure.align import forward_returns
from research.microstructure.ic import compute_ic
from research.microstructure.report import render_ic_markdown


def recon_multi(
    grid: pl.DataFrame,
    signals: dict[str, pl.DataFrame],
    *,
    horizons_secs: list[int],
) -> tuple[str, dict[str, dict[str, float]]]:
    """grid: (ts, mid). signals: name -> (ts, name). Returns (markdown, ic_by_signal)."""
    g = forward_returns(grid.sort("ts"), horizons_secs=horizons_secs)
    hcols = [f"fwd_{h}s" for h in horizons_secs]
    ic_by_signal: dict[str, dict[str, float]] = {}
    for name, sig in signals.items():
        merged = g.join_asof(sig.sort("ts"), on="ts", strategy="backward")
        ic_by_signal[name] = compute_ic(merged, signal_col=name, horizon_cols=hcols)
    n_tests = len(signals) * len(hcols)
    md = render_ic_markdown(ic_by_signal, n_tests=n_tests)
    return md, ic_by_signal
```

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Run the full Phase-2a suite**

Run: `./venv/bin/pytest tests/research/microstructure/ -v`
Expected: all pass (Phase-1 tests + all new Task 1-10 tests).

- [ ] **Step 6: Commit**

```bash
git add src/research/microstructure/pipeline.py tests/research/microstructure/test_pipeline.py
git commit -m "$(printf 'feat(recon): multi-signal recon pipeline (recon_multi)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Manual integration check (after Task 10, network — not in CI)

Download a few days in the window, build the mid grid from bookTicker, build each signal, and run `recon_multi`. This is the first real multi-signal IC read (single-day sanity, not the full study — that is Phase 2b):

```bash
# example for one day; a driver CLI over the full window is Phase 2b
PYTHONPATH=src venv/bin/python -m scripts.recon.probe_schema --symbol ETHUSDT --date 2023-06-01
# (bookTicker exists for 2023-06-01 — confirm the thin pipeline + new signals run)
```

Record per-signal IC sanity in the Phase 2b plan kickoff.

## Phase 2b (separate plan)

Not in this plan: BTC→ETH cross-asset lead-lag; multi-day/multi-month chunked driver over the full 2023-05→2024-03 window; Newey-West / block-bootstrap significance; OOS holdout; coarse cost sensitivity; plotly/notebook visual report; and the §10 decision-rule verdict that maps the IC-vs-horizon result to the next architecture.
