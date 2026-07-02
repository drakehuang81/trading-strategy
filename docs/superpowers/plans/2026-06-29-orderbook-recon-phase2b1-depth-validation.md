# Order Book Recon — Phase 2b-1 (Depth-Imbalance @ 1h Validation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether `depth_imbalance @ 1h` is real, tradeable alpha or an artifact — via a full-window (2023-05-16 → 2024-03-30) out-of-sample study with trend control, full-bucket monotonicity, tail-robustness, and Newey-West significance. Task 3 (OOS gate) is make-or-break: if it fails, stop and record the negative result; do not build the later tasks.

**Architecture:** Extends `research.microstructure` (merged). Focus on depth means light data: bookDepth (~545 KB/day) + 1h klines (~tiny), NOT the 6.4M-row/day bookTicker. A per-day builder turns bookDepth → hourly `depth_imbalance`, aligns to 1h kline close, and computes forward 1h return. A driver concatenates the full window and runs OOS + diagnostics.

**Tech Stack:** Python 3.11, polars, scipy, statsmodels (new dep, Newey-West), pytest. Findings: [2026-06-29-orderbook-recon-phase2a-integration-findings.md](2026-06-29-orderbook-recon-phase2a-integration-findings.md). Builds on Phase 2a (merged).

---

## Context: why this scope

Phase 2a integration + cost check found: qi (L1) has strong second-scale IC but dies to taker fee (maker-only, deferred); **depth_imbalance @ 15m–1h nets +7 to +14 bps after realistic taker fee, IC 0.5, monotone** — the first net-positive-after-cost signal, at a horizon that fits the 1h architecture. BUT single-day, in-sample. This plan validates it. The #1 risk is that depth@1h is a **trend proxy** ("buy orders pile up while price trends"), not causal.

## Environment notes (read first)

- Clean worktree off latest `main` + fresh 3.11 venv. Deps: `pandas numpy pyarrow scipy pytest pytest-asyncio requests polars statsmodels`.
- Verify via **pytest** (`pythonpath=[".","src"]` is pytest-only). Run ONLY `./venv/bin/pytest tests/research/microstructure/ -v` (scoped).
- Commit: conventional prefix + trailing `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Research-only: no imports from production layers.

## File Structure

| Path | Responsibility | Change |
|---|---|---|
| `src/research/microstructure/pipeline.py` | `recon_multi` derives signal col name (robustness) | modify |
| `src/research/microstructure/download.py` | `load_klines_1h` (robust, headerless) | modify |
| `src/research/microstructure/depth_study.py` | hourly dataset builder + OOS split + trend/monotonicity/stats | **create** |
| `scripts/recon/depth_validation.py` | full-window driver CLI (download loop → dataset → gate + diagnostics) | **create** |
| `tests/research/microstructure/test_depth_study.py` | dataset builder + metrics tests | **create** |
| tests for pipeline/download changes | | modify |

---

### Task 1: recon_multi robustness (derive signal column name)

**Files:** Modify `src/research/microstructure/pipeline.py`; Test `tests/research/microstructure/test_pipeline.py`

- [ ] **Step 1: Add a failing test** (append to `test_pipeline.py`):

```python
def test_recon_multi_derives_column_name_not_dict_key():
    import datetime as dt
    ts = [dt.datetime(2026, 1, 1, 0, 0, s) for s in range(60)]
    grid = pl.DataFrame({"ts": ts, "mid": [100.0 + (s % 2) for s in range(60)]})
    # dict key "sig" differs from the actual column "depth_imbalance"
    sig = pl.DataFrame({"ts": ts, "depth_imbalance": [0.5 if s % 2 == 0 else -0.5 for s in range(60)]})
    md, ic = recon_multi(grid, {"sig": sig}, horizons_secs=[1])
    assert "fwd_1s" in ic["sig"]  # must not KeyError on the column name
```

- [ ] **Step 2: Run, verify FAIL** (ColumnNotFoundError): `./venv/bin/pytest tests/research/microstructure/test_pipeline.py -v`

- [ ] **Step 3: Fix `recon_multi`.** In `src/research/microstructure/pipeline.py`, change the loop body to derive the column name from the signal df (the non-`ts` column), not the dict key:

```python
    for name, sig in signals.items():
        col = next(c for c in sig.columns if c != "ts")
        merged = g.join_asof(sig.sort("ts"), on="ts", strategy="backward")
        ic_by_signal[name] = compute_ic(merged, signal_col=col, horizon_cols=hcols)
```

- [ ] **Step 4: Run, verify PASS** (new + existing pipeline tests).

- [ ] **Step 5: Commit**
```bash
git add src/research/microstructure/pipeline.py tests/research/microstructure/test_pipeline.py
git commit -m "$(printf 'fix(recon): recon_multi derives signal column name (findings #6)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: load_klines_1h (robust, headerless)

**Files:** Modify `src/research/microstructure/download.py`; Test `tests/research/microstructure/test_download.py`

Binance daily 1h kline CSVs are **headerless**, 12 columns: `open_time(ms), open, high, low, close, volume, close_time, quote_vol, count, taker_buy_vol, taker_buy_quote, ignore`.

- [ ] **Step 1: Failing test** (append to `test_download.py`):

```python
def test_load_klines_1h_headerless(tmp_path: Path):
    # headerless CSV -> parquet, as extract_zip_to_parquet would NOT work
    # (that assumes header); load_klines_1h reads with has_header=False.
    import csv
    csv_path = tmp_path / "k.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([1685577600000, 1900, 1910, 1890, 1905, 100, 1685581199999, 0, 0, 0, 0, 0])
        w.writerow([1685581200000, 1905, 1915, 1900, 1912, 120, 1685584799999, 0, 0, 0, 0, 0])
    from research.microstructure.download import load_klines_1h
    df = load_klines_1h(csv_path)
    assert df.columns == ["hour", "close"]
    assert df["hour"].dtype == pl.Datetime
    assert df["close"][0] == 1905.0
```

- [ ] **Step 2: Run, verify FAIL** (ImportError).

- [ ] **Step 3: Implement** (append to `download.py`):

```python
KLINES_1H_BASE = "https://data.binance.vision/data/futures/um/daily/klines"


def klines_1h_url(symbol: str, date: "dt.date") -> str:
    return f"{KLINES_1H_BASE}/{symbol}/1h/{symbol}-1h-{date.isoformat()}.zip"


def load_klines_1h(csv_or_parquet: Path) -> pl.DataFrame:
    """Load headerless Binance 1h klines CSV → (hour, close).

    Columns are positional: col 0 = open_time (epoch ms), col 4 = close.
    """
    df = pl.read_csv(csv_or_parquet, has_header=False)
    open_time, close = df.columns[0], df.columns[4]
    return df.select(
        pl.from_epoch(pl.col(open_time).cast(pl.Int64), time_unit="ms").alias("hour"),
        pl.col(close).cast(pl.Float64).alias("close"),
    )
```

Add `import datetime as dt` at the top if not present (for the `klines_1h_url` type hint; it is already imported in download.py from Task-2 Phase-2a work — verify).

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/research/microstructure/download.py tests/research/microstructure/test_download.py
git commit -m "$(printf 'feat(recon): load_klines_1h (headerless) + klines_1h_url\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 3: Hourly dataset builder + OOS split (THE GATE)

**Files:** Create `src/research/microstructure/depth_study.py`; Test `tests/research/microstructure/test_depth_study.py`

- [ ] **Step 1: Failing tests**

```python
# tests/research/microstructure/test_depth_study.py
import datetime as dt
import polars as pl

from research.microstructure.depth_study import build_hourly, time_split


def test_build_hourly_aligns_depth_to_kline_forward_return():
    # two hours of depth snapshots + 1h klines
    h0 = dt.datetime(2023, 6, 1, 0, 0, 0)
    h1 = dt.datetime(2023, 6, 1, 1, 0, 0)
    h2 = dt.datetime(2023, 6, 1, 2, 0, 0)
    # depth long-form: hour 0 bid-heavy (di>0), hour 1 ask-heavy (di<0)
    depth = pl.DataFrame({
        "ts": [h0, h0, h1, h1],
        "percentage": [-1.0, 1.0, -1.0, 1.0],
        "depth": [30.0, 10.0, 10.0, 30.0],
    })
    klines = pl.DataFrame({"hour": [h0, h1, h2], "close": [100.0, 110.0, 99.0]})
    ds = build_hourly(depth, klines)
    # hour 0: di = (30-10)/40 = 0.5; fwd_1h = 110/100-1 = 0.10
    row0 = ds.filter(pl.col("hour") == h0)
    assert abs(row0["di"][0] - 0.5) < 1e-9
    assert abs(row0["fwd_1h"][0] - 0.10) < 1e-9
    # last hour has no forward kline -> dropped
    assert ds.filter(pl.col("hour") == h2).height == 0


def test_time_split_70_30_by_date():
    hours = [dt.datetime(2023, 6, d, 0) for d in range(1, 11)]
    ds = pl.DataFrame({"hour": hours, "di": [0.0] * 10, "fwd_1h": [0.0] * 10})
    train, test = time_split(ds, train_frac=0.7)
    assert train.height == 7 and test.height == 3
    assert train["hour"].max() < test["hour"].min()  # strictly time-ordered
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement `depth_study.py`**

```python
"""Depth-imbalance @ 1h validation dataset + OOS split.

build_hourly: bookDepth (long) -> hourly mean depth_imbalance, aligned to 1h
kline close, with forward 1h return. time_split: strict time-ordered holdout.
"""
from __future__ import annotations

import polars as pl

from research.microstructure.signals import depth_imbalance


def build_hourly(depth: pl.DataFrame, klines_1h: pl.DataFrame) -> pl.DataFrame:
    """(depth long-form, klines (hour,close)) -> (hour, di, fwd_1h).

    di = mean per-snapshot depth_imbalance within the hour; fwd_1h = next
    hour's close / this close - 1. Rows without a forward close are dropped.
    """
    di = depth_imbalance(depth)  # (ts, depth_imbalance) per snapshot
    di_1h = (
        di.with_columns(pl.col("ts").dt.truncate("1h").alias("hour"))
        .group_by("hour")
        .agg(pl.col("depth_imbalance").mean().alias("di"))
        .sort("hour")
    )
    k = klines_1h.sort("hour").with_columns(
        (pl.col("close").shift(-1) / pl.col("close") - 1).alias("fwd_1h")
    )
    return di_1h.join(k, on="hour", how="inner").drop_nulls(["di", "fwd_1h"]).sort("hour")


def time_split(ds: pl.DataFrame, *, train_frac: float = 0.7) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Strict time-ordered holdout: earliest train_frac rows train, rest test."""
    ds = ds.sort("hour")
    cut = int(ds.height * train_frac)
    return ds.head(cut), ds.tail(ds.height - cut)
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/research/microstructure/depth_study.py tests/research/microstructure/test_depth_study.py
git commit -m "$(printf 'feat(recon): depth @ 1h hourly dataset builder + OOS split\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 4: Trend control + monotonicity + Newey-West metrics

**Files:** Modify `src/research/microstructure/depth_study.py`; Test `tests/research/microstructure/test_depth_study.py`

- [ ] **Step 1: Failing tests** (append):

```python
def test_decile_edge_bps_monotone_and_tail():
    import numpy as np
    rng = np.random.default_rng(0)
    di = rng.normal(size=4000)
    fwd = di * 0.001 + rng.normal(scale=0.002, size=4000)  # di predicts fwd
    ds = pl.DataFrame({"di": di, "fwd_1h": fwd})
    from research.microstructure.depth_study import decile_edge_bps
    res = decile_edge_bps(ds, n_buckets=10)
    assert res["means"] == sorted(res["means"])          # full-bucket monotone
    assert res["net_std_taker_bps"] == res["gross_bps"] - 8.0


def test_newey_west_tstat_runs():
    import numpy as np
    rng = np.random.default_rng(1)
    r = rng.normal(loc=0.5, scale=1.0, size=200)  # positive-mean series
    from research.microstructure.depth_study import newey_west_tstat
    t = newey_west_tstat(r, lags=5)
    assert t > 3.0  # clearly positive


def test_momentum_control_ic():
    # depth IC vs momentum(past return) IC — if equal, depth is trend proxy
    import numpy as np
    rng = np.random.default_rng(2)
    di = rng.normal(size=1000)
    past = rng.normal(size=1000)
    fwd = di * 0.001 + rng.normal(scale=0.002, size=1000)  # driven by di, not past
    ds = pl.DataFrame({"di": di, "past_1h": past, "fwd_1h": fwd})
    from research.microstructure.depth_study import ic
    assert abs(ic(ds, "di")) > abs(ic(ds, "past_1h")) + 0.1
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement** (append to `depth_study.py`):

```python
from scipy.stats import spearmanr


def ic(ds: pl.DataFrame, col: str, *, target: str = "fwd_1h") -> float:
    p = ds.select([col, target]).drop_nulls().filter(
        pl.col(col).is_finite() & pl.col(target).is_finite()
    )
    if p.height < 3:
        return float("nan")
    return float(spearmanr(p[col].to_numpy(), p[target].to_numpy())[0])


def decile_edge_bps(ds: pl.DataFrame, *, n_buckets: int = 10, taker_std_bps: float = 8.0) -> dict:
    """Mean fwd_1h (bps) per di quantile; gross = best of top-long/bottom-short."""
    labels = [str(i).zfill(len(str(n_buckets - 1))) for i in range(n_buckets)]
    g = (
        ds.drop_nulls(["di", "fwd_1h"])
        .with_columns(pl.col("di").qcut(n_buckets, labels=labels, allow_duplicates=True).alias("b"))
        .group_by("b").agg((pl.col("fwd_1h").mean() * 1e4).alias("r")).sort("b")
    )
    means = g["r"].to_list()
    gross = max(means[-1], -means[0])
    return {"means": means, "gross_bps": gross, "net_std_taker_bps": gross - taker_std_bps}


def newey_west_tstat(returns, *, lags: int = 5) -> float:
    """HAC t-stat of the mean of a return series (autocorrelation-robust)."""
    import numpy as np
    import statsmodels.api as sm
    y = np.asarray(returns, dtype=float)
    x = np.ones_like(y)
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.tvalues[0])
```

- [ ] **Step 4: Add `statsmodels>=0.14` to `requirements.txt`** (append under the `# Recon Phase 1 additions` section, or a new `# Recon Phase 2b additions` line), then run, verify PASS: `./venv/bin/pytest tests/research/microstructure/test_depth_study.py -v` (requires `statsmodels` installed in the venv).

- [ ] **Step 5: Commit**
```bash
git add src/research/microstructure/depth_study.py tests/research/microstructure/test_depth_study.py requirements.txt
git commit -m "$(printf 'feat(recon): decile edge, IC, momentum control, Newey-West tstat\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 5: Full-window driver CLI (runs the gate + diagnostics)

**Files:** Create `scripts/recon/depth_validation.py`; Test `tests/research/microstructure/test_depth_validation.py`

- [ ] **Step 1: Failing test for the pure aggregation** (the network loop is manual):

```python
# tests/research/microstructure/test_depth_validation.py
import datetime as dt
import polars as pl

from scripts.recon.depth_validation import summarize


def test_summarize_reports_train_test_and_control():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 2000
    di = rng.normal(size=n)
    ds = pl.DataFrame({
        "hour": [dt.datetime(2023, 6, 1) + dt.timedelta(hours=i) for i in range(n)],
        "di": di,
        "past_1h": rng.normal(size=n),
        "fwd_1h": di * 0.001 + rng.normal(scale=0.002, size=n),
    })
    rep = summarize(ds)
    assert "ic_all" in rep and "ic_train" in rep and "ic_test" in rep
    assert "ic_momentum" in rep and "gross_bps" in rep and "nw_tstat" in rep
    assert isinstance(rep["verdict"], str)
```

- [ ] **Step 2: Run, verify FAIL** (ImportError).

- [ ] **Step 3: Implement `scripts/recon/depth_validation.py`**

```python
"""Full-window depth_imbalance @ 1h validation driver.

Downloads bookDepth + 1h klines per day over the window, builds the hourly
dataset, and reports the OOS gate + trend/monotonicity/significance verdict.

Run (network — manual):
    PYTHONPATH=src venv/bin/python -m scripts.recon.depth_validation \
        --symbol ETHUSDT --start 2023-05-16 --end 2024-03-30
"""
from __future__ import annotations

import argparse
import datetime as dt
import zipfile
from pathlib import Path

import polars as pl

from research.microstructure.depth_study import (
    build_hourly, time_split, ic, decile_edge_bps, newey_west_tstat,
)
from research.microstructure.download import (
    build_url, download_zip, extract_zip_to_parquet,
    load_book_depth, klines_1h_url, load_klines_1h,
)


def summarize(ds: pl.DataFrame) -> dict:
    """Pure: hourly dataset (hour, di, past_1h, fwd_1h) -> verdict report."""
    train, test = time_split(ds, train_frac=0.7)
    edge = decile_edge_bps(ds)
    ic_all, ic_tr, ic_te = ic(ds, "di"), ic(train, "di"), ic(test, "di")
    ic_mom = ic(ds, "past_1h") if "past_1h" in ds.columns else float("nan")
    # NW t-stat on the DEPTH STRATEGY's per-hour return (long above median di,
    # short below) — NOT raw fwd_1h, which would just measure buy-and-hold drift.
    d2 = ds.drop_nulls(["di", "fwd_1h"])
    thr = d2["di"].median()
    strat = d2.with_columns(
        pl.when(pl.col("di") > thr).then(pl.col("fwd_1h"))
        .otherwise(-pl.col("fwd_1h")).alias("strat_ret")
    )
    nw = newey_west_tstat(strat["strat_ret"].to_numpy(), lags=5)
    monotone = edge["means"] == sorted(edge["means"])
    beats_momentum = abs(ic_all) > abs(ic_mom) + 0.05
    survives_oos = (ic_tr * ic_te > 0) and abs(ic_te) > 0.1
    tradeable = edge["net_std_taker_bps"] > 0
    verdict = (
        "REAL-ALPHA candidate" if (survives_oos and beats_momentum and tradeable and monotone)
        else "FAILED — " + ", ".join(
            x for x, ok in [
                ("OOS", survives_oos), ("vs-momentum", beats_momentum),
                ("post-cost", tradeable), ("monotone", monotone),
            ] if not ok
        )
    )
    return {
        "ic_all": ic_all, "ic_train": ic_tr, "ic_test": ic_te,
        "ic_momentum": ic_mom, "gross_bps": edge["gross_bps"],
        "net_std_taker_bps": edge["net_std_taker_bps"], "nw_tstat": nw,
        "monotone": monotone, "verdict": verdict, "n_hours": ds.height,
    }


def _daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def build_window(symbol: str, start: dt.date, end: dt.date, out: Path) -> pl.DataFrame:
    parts = []
    for d in _daterange(start, end):
        try:
            bdp = out / f"bd-{d}.parquet"
            if not bdp.exists():
                extract_zip_to_parquet(download_zip(build_url(symbol, "bookDepth", d), out / f"bd-{d}.zip"), bdp)
            # klines: keep the raw csv (extract_zip_to_parquet assumes a header
            # row, which klines lack) — cache check on the csv we actually write
            klc = out / f"kl-{d}.csv"
            if not klc.exists():
                klz = out / f"kl-{d}.zip"
                download_zip(klines_1h_url(symbol, d), klz)
                with zipfile.ZipFile(klz) as zf:
                    name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                    zf.extract(name, out)
                    (out / name).rename(klc)
            depth = load_book_depth(bdp)
            klines = load_klines_1h(klc)
            day = build_hourly(depth, klines)
            day = day.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("past_1h"))
            parts.append(day)
        except Exception as e:  # noqa: BLE001
            print(f"  {d} skipped: {e}")
    return pl.concat(parts).sort("hour")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out-dir", default="data/orderbook/_fw")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = build_window(args.symbol, dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end), out)
    rep = summarize(ds)
    print("\n=== depth_imbalance @ 1h — full-window validation ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, verify PASS** on the pure `summarize` test.

- [ ] **Step 5: Run the full scoped suite:** `./venv/bin/pytest tests/research/microstructure/ -v` (all Phase 1/2a + new pass).

- [ ] **Step 6: Commit**
```bash
git add scripts/recon/depth_validation.py tests/research/microstructure/test_depth_validation.py
git commit -m "$(printf 'feat(recon): full-window depth @ 1h validation driver + verdict\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## THE GATE (manual, after Task 5 — network, ~5-15 min for ~320 days of light bookDepth+klines)

```bash
PYTHONPATH=src venv/bin/python -m scripts.recon.depth_validation \
    --symbol ETHUSDT --start 2023-05-16 --end 2024-03-30
```

Read the `verdict`:
- **REAL-ALPHA candidate** (OOS holds + beats momentum + net-positive post-taker + monotone) → depth@1h is the lead. Proceed to Phase 2b-2: wire depth_imbalance as a 1h feature into the existing architecture; add capacity/turnover + regime breakdown; BTC→ETH cross-asset; visual report.
- **FAILED (any reason)** → record the negative result like Plan 5E. Most likely failure is `vs-momentum` (depth@1h is a trend proxy) — if so, order-book directional recon on ETH is largely exhausted; revisit qi's maker/HF path or a different market.

## Deferred to Phase 2b-2 (only if the gate passes)

Capacity/turnover of the decile strategy; regime breakdown (up/down/chop); BTC→ETH cross-asset lead-lag; plotly/notebook visual report; and wiring depth_imbalance into the production 1h decision path per spec §10.
