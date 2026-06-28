# Order Book Recon — Phase 1 (Step 0 + End-to-End Thin Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the research module, a generic data.binance.vision downloader, a Step-0 schema probe, and an end-to-end thin pipeline (download → L1 queue-imbalance signal → align → IC → markdown report) that proves the whole flow on real ETH perp data using the one signal whose source format is stable.

**Architecture:** Research-only code under `src/research/microstructure/`, isolated from production. Signal logic consumes a *normalized* DataFrame (`ts, bid_price, bid_qty, ask_price, ask_qty`); the raw→normalized column map lives in the load layer and is locked by the Step-0 probe. This decouples signal code from the unverified raw schema. Phase 2 (after Step 0 confirms `bookDepth` semantics) adds OFI, depth imbalance, book slope, taker imbalance, and cross-asset.

**Tech Stack:** Python 3.11, polars (new dep, lazy/chunked), requests (download), scipy (Spearman), pytest. Spec: [docs/superpowers/specs/2026-06-28-orderbook-microstructure-recon-design.md](../specs/2026-06-28-orderbook-microstructure-recon-design.md).

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/research/__init__.py` | research namespace package marker |
| `src/research/microstructure/__init__.py` | module marker |
| `src/research/microstructure/download.py` | build data.binance.vision URLs, download+extract zip → parquet; normalized loaders + `COLUMN_MAP` |
| `src/research/microstructure/schema_probe.py` | Step 0: summarize raw CSV schema, dtypes, row cadence, bookDepth semantics |
| `src/research/microstructure/signals.py` | Phase 1: `queue_imbalance` (L1) only |
| `src/research/microstructure/align.py` | mid-price grid (as-of), forward returns at horizons |
| `src/research/microstructure/ic.py` | Spearman IC per horizon, quantile layering |
| `src/research/microstructure/report.py` | render IC table → markdown |
| `scripts/recon/download_orderbook.py` | CLI for download.py |
| `scripts/recon/probe_schema.py` | CLI for schema_probe.py (Step 0) |
| `scripts/recon/run_recon.py` | CLI: end-to-end thin pipeline → markdown report |
| `tests/research/microstructure/test_download.py` | URL build + extract/normalize tests |
| `tests/research/microstructure/test_signals.py` | queue_imbalance point-in-time + formula |
| `tests/research/microstructure/test_align.py` | grid as-of + forward returns no-lookahead |
| `tests/research/microstructure/test_ic.py` | Spearman recovery + layering |
| `tests/research/microstructure/test_report.py` | markdown rendering |

**Convention notes:** `pythonpath=[".","src"]`, `testpaths=["tests"]`, `asyncio_mode=auto`. `data/orderbook/` is gitignored (large). Mark any test that hits the network with `@pytest.mark.slow` and never run it in the default suite — all tests below use fixtures/tmp_path, no real network.

---

### Task 1: Module skeleton + polars dependency

**Files:**
- Create: `src/research/__init__.py`, `src/research/microstructure/__init__.py`
- Create: `tests/research/__init__.py`, `tests/research/microstructure/__init__.py`
- Modify: `requirements.txt` (add polars)

- [ ] **Step 1: Add polars to requirements and install**

Append a line to `requirements.txt`:

```
polars>=1.0
```

Run: `venv/bin/pip install "polars>=1.0"`
Expected: installs successfully; `venv/bin/python -c "import polars; print(polars.__version__)"` prints a 1.x version.

- [ ] **Step 2: Create empty package markers**

Create these four files, each empty:
- `src/research/__init__.py`
- `src/research/microstructure/__init__.py`
- `tests/research/__init__.py`
- `tests/research/microstructure/__init__.py`

- [ ] **Step 3: Verify import path works**

Run: `venv/bin/python -c "import research.microstructure; print('ok')"`
Expected: prints `ok` (because `pythonpath` includes `src`).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src/research tests/research
git commit -m "chore(recon): research.microstructure skeleton + polars dep"
```

---

### Task 2: Downloader (URL build + extract + normalized load)

**Files:**
- Create: `src/research/microstructure/download.py`
- Test: `tests/research/microstructure/test_download.py`

- [ ] **Step 1: Write failing tests for URL building and normalization**

```python
# tests/research/microstructure/test_download.py
import datetime as dt
import io
import zipfile
from pathlib import Path

import polars as pl
import pytest

from research.microstructure.download import (
    build_url,
    extract_zip_to_parquet,
    load_book_ticker,
)


def test_build_url_book_ticker():
    url = build_url("ETHUSDT", "bookTicker", dt.date(2026, 1, 15))
    assert url == (
        "https://data.binance.vision/data/futures/um/daily/"
        "bookTicker/ETHUSDT/ETHUSDT-bookTicker-2026-01-15.zip"
    )


def test_extract_zip_to_parquet_roundtrip(tmp_path: Path):
    # Build an in-memory zip holding one CSV, mimicking data.binance.vision
    csv_bytes = (
        b"update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,"
        b"transaction_time,event_time\n"
        b"1,100.0,5.0,100.5,3.0,1700000000000,1700000000001\n"
    )
    zip_path = tmp_path / "ETHUSDT-bookTicker-2026-01-15.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ETHUSDT-bookTicker-2026-01-15.csv", csv_bytes)

    out = extract_zip_to_parquet(zip_path, tmp_path / "out.parquet")
    df = pl.read_parquet(out)
    assert df.height == 1
    assert "best_bid_price" in df.columns


def test_load_book_ticker_normalizes_columns(tmp_path: Path):
    raw = pl.DataFrame({
        "update_id": [1],
        "best_bid_price": [100.0],
        "best_bid_qty": [5.0],
        "best_ask_price": [100.5],
        "best_ask_qty": [3.0],
        "transaction_time": [1700000000000],
        "event_time": [1700000000001],
    })
    p = tmp_path / "bt.parquet"
    raw.write_parquet(p)

    df = load_book_ticker(p)
    assert df.columns == ["ts", "bid_price", "bid_qty", "ask_price", "ask_qty"]
    assert df["bid_price"][0] == 100.0
    # ts parsed from transaction_time (ms) to datetime
    assert df["ts"].dtype == pl.Datetime
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/research/microstructure/test_download.py -v`
Expected: FAIL with `ImportError` (module/functions not defined).

- [ ] **Step 3: Implement download.py**

```python
# src/research/microstructure/download.py
"""Downloader for data.binance.vision USD-M futures order book archives.

Raw CSV schemas vary by data type and are confirmed by the Step-0 probe
(schema_probe.py). COLUMN_MAP below holds the *expected* raw→normalized
mapping; if Step 0 reveals different raw column names, update COLUMN_MAP
only — signal/align/ic code consumes the normalized schema and stays put.
"""
from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import polars as pl
import requests

BASE_URL = "https://data.binance.vision/data/futures/um/daily"
DATA_TYPES = ("bookTicker", "bookDepth", "aggTrades")

# Expected raw bookTicker columns (confirmed/adjusted by Step 0).
# Normalized target: ts, bid_price, bid_qty, ask_price, ask_qty
BOOK_TICKER_MAP = {
    "best_bid_price": "bid_price",
    "best_bid_qty": "bid_qty",
    "best_ask_price": "ask_price",
    "best_ask_qty": "ask_qty",
}
BOOK_TICKER_TS_COL = "transaction_time"  # epoch ms


def build_url(symbol: str, data_type: str, date: dt.date) -> str:
    if data_type not in DATA_TYPES:
        raise ValueError(f"unknown data_type {data_type!r}")
    fname = f"{symbol}-{data_type}-{date.isoformat()}.zip"
    return f"{BASE_URL}/{data_type}/{symbol}/{fname}"


def download_zip(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    return dest


def extract_zip_to_parquet(zip_path: Path, parquet_path: Path) -> Path:
    """Extract the single CSV inside a Binance daily zip into parquet."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected 1 csv in {zip_path}, found {names}")
        with zf.open(names[0]) as fh:
            df = pl.read_csv(fh)
    df.write_parquet(parquet_path)
    return parquet_path


def load_book_ticker(parquet_path: Path) -> pl.DataFrame:
    """Load + normalize a bookTicker parquet to the standard schema."""
    df = pl.read_parquet(parquet_path)
    df = df.rename(BOOK_TICKER_MAP)
    df = df.with_columns(
        pl.from_epoch(pl.col(BOOK_TICKER_TS_COL), time_unit="ms").alias("ts")
    )
    return df.select(["ts", "bid_price", "bid_qty", "ask_price", "ask_qty"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/research/microstructure/test_download.py -v`
Expected: PASS (3 tests). If `from_epoch` signature differs in the installed polars, adjust to `pl.col(...).cast(pl.Datetime("ms"))`.

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/download.py tests/research/microstructure/test_download.py
git commit -m "feat(recon): data.binance.vision downloader + normalized bookTicker loader"
```

---

### Task 3: Step-0 schema probe

**Files:**
- Create: `src/research/microstructure/schema_probe.py`
- Create: `scripts/recon/__init__.py`, `scripts/recon/probe_schema.py`
- Test: `tests/research/microstructure/test_schema_probe.py`

- [ ] **Step 1: Write failing test for the schema summary**

```python
# tests/research/microstructure/test_schema_probe.py
import polars as pl

from research.microstructure.schema_probe import summarize_schema


def test_summarize_schema_reports_columns_and_cadence():
    df = pl.DataFrame({
        "timestamp": [1700000000000, 1700000001000, 1700000002000],
        "percentage": [1, 1, 1],
        "depth": [10.0, 11.0, 12.0],
        "notional": [1000.0, 1100.0, 1200.0],
    })
    summary = summarize_schema(df, ts_col="timestamp")
    assert summary["n_rows"] == 3
    assert "percentage" in summary["columns"]
    # median gap between consecutive timestamps (ms)
    assert summary["median_gap_ms"] == 1000
    assert summary["distinct_percentage"] == [1]  # bookDepth semantics hint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/research/microstructure/test_schema_probe.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement schema_probe.py**

```python
# src/research/microstructure/schema_probe.py
"""Step 0 — de-risk: report the real schema/cadence of downloaded data.

For bookDepth specifically, the presence of a 'percentage' column means
Binance ships *percentage-distance depth* (not raw L2 levels), which decides
how Phase 2 defines depth imbalance / book slope.
"""
from __future__ import annotations

from typing import Any

import polars as pl


def summarize_schema(df: pl.DataFrame, *, ts_col: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": df.height,
        "columns": df.columns,
        "dtypes": {c: str(t) for c, t in zip(df.columns, df.dtypes)},
    }
    if ts_col in df.columns and df.height > 1:
        gaps = df[ts_col].sort().diff().drop_nulls()
        summary["median_gap_ms"] = int(gaps.median())
    else:
        summary["median_gap_ms"] = None
    if "percentage" in df.columns:
        summary["distinct_percentage"] = sorted(
            df["percentage"].unique().to_list()
        )
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/research/microstructure/test_schema_probe.py -v`
Expected: PASS.

- [ ] **Step 5: Write the Step-0 CLI**

```python
# scripts/recon/probe_schema.py
"""Step 0 CLI: download 1 day of each data type for one symbol, print schema.

Run (network — do this once manually, not in CI):
    PYTHONPATH=src venv/bin/python -m scripts.recon.probe_schema --symbol ETHUSDT --date 2026-06-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import polars as pl

from research.microstructure.download import (
    DATA_TYPES,
    build_url,
    download_zip,
    extract_zip_to_parquet,
)
from research.microstructure.schema_probe import summarize_schema

_TS_GUESS = {"bookTicker": "transaction_time", "bookDepth": "timestamp", "aggTrades": "transact_time"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out-dir", default="data/orderbook/_probe")
    args = ap.parse_args()
    date = dt.date.fromisoformat(args.date)
    out = Path(args.out_dir)

    for dtype in DATA_TYPES:
        url = build_url(args.symbol, dtype, date)
        zip_path = out / f"{args.symbol}-{dtype}-{date}.zip"
        pq = out / f"{args.symbol}-{dtype}-{date}.parquet"
        try:
            download_zip(url, zip_path)
            extract_zip_to_parquet(zip_path, pq)
            df = pl.read_parquet(pq)
            ts_col = _TS_GUESS.get(dtype, df.columns[0])
            print(f"\n=== {dtype} ===")
            print(json.dumps(summarize_schema(df, ts_col=ts_col), indent=2, default=str))
            print("head:", df.head(3).to_dicts())
        except Exception as e:  # noqa: BLE001 — probe should report, not crash
            print(f"\n=== {dtype} === FAILED: {e}")


if __name__ == "__main__":
    main()
```

Also create empty `scripts/recon/__init__.py`.

- [ ] **Step 6: Commit**

```bash
git add src/research/microstructure/schema_probe.py scripts/recon tests/research/microstructure/test_schema_probe.py
git commit -m "feat(recon): Step-0 schema probe + CLI"
```

---

### Task 4: L1 queue imbalance signal

**Files:**
- Create: `src/research/microstructure/signals.py`
- Test: `tests/research/microstructure/test_signals.py`

- [ ] **Step 1: Write failing tests (formula + point-in-time)**

```python
# tests/research/microstructure/test_signals.py
import datetime as dt

import polars as pl

from research.microstructure.signals import queue_imbalance


def _book(rows):
    return pl.DataFrame(
        {
            "ts": [dt.datetime(2026, 1, 1, 0, 0, i) for i in range(len(rows))],
            "bid_price": [r[0] for r in rows],
            "bid_qty": [r[1] for r in rows],
            "ask_price": [r[2] for r in rows],
            "ask_qty": [r[3] for r in rows],
        }
    )


def test_queue_imbalance_formula():
    # QI = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    book = _book([(100.0, 6.0, 100.5, 2.0)])  # (6-2)/(6+2) = 0.5
    qi = queue_imbalance(book)
    assert abs(qi["qi"][0] - 0.5) < 1e-12


def test_queue_imbalance_point_in_time():
    # QI at row i must depend ONLY on row i (no lookahead): computing on a
    # truncated prefix yields identical values for the rows present.
    book = _book([(100.0, 6.0, 100.5, 2.0), (101.0, 1.0, 101.5, 3.0)])
    full = queue_imbalance(book)
    prefix = queue_imbalance(book.head(1))
    assert abs(full["qi"][0] - prefix["qi"][0]) < 1e-12


def test_queue_imbalance_zero_depth_is_null():
    book = _book([(100.0, 0.0, 100.5, 0.0)])
    qi = queue_imbalance(book)
    assert qi["qi"][0] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/research/microstructure/test_signals.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement signals.py**

```python
# src/research/microstructure/signals.py
"""Microstructure signals. Phase 1 ships L1 queue imbalance only.

All signals are point-in-time: the value at row t uses only data at/before t.
"""
from __future__ import annotations

import polars as pl


def queue_imbalance(book: pl.DataFrame) -> pl.DataFrame:
    """L1 queue imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty).

    Input columns: ts, bid_qty, ask_qty (others ignored).
    Output: ts, qi  (qi is null when total depth is zero).
    """
    total = pl.col("bid_qty") + pl.col("ask_qty")
    qi = (
        pl.when(total > 0)
        .then((pl.col("bid_qty") - pl.col("ask_qty")) / total)
        .otherwise(None)
        .alias("qi")
    )
    return book.select(["ts", qi])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/research/microstructure/test_signals.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/signals.py tests/research/microstructure/test_signals.py
git commit -m "feat(recon): L1 queue imbalance signal (point-in-time)"
```

---

### Task 5: Align to grid + forward returns

**Files:**
- Create: `src/research/microstructure/align.py`
- Test: `tests/research/microstructure/test_align.py`

- [ ] **Step 1: Write failing tests (as-of backward + forward returns)**

```python
# tests/research/microstructure/test_align.py
import datetime as dt

import polars as pl

from research.microstructure.align import forward_returns, to_mid_grid


def _book_with_mid():
    # bookTicker rows at irregular seconds; mid = (bid+ask)/2
    ts = [dt.datetime(2026, 1, 1, 0, 0, s) for s in (0, 2, 5)]
    return pl.DataFrame({
        "ts": ts,
        "bid_price": [100.0, 102.0, 104.0],
        "ask_price": [100.0, 102.0, 104.0],
        "bid_qty": [1.0, 1.0, 1.0],
        "ask_qty": [1.0, 1.0, 1.0],
    })


def test_to_mid_grid_backward_asof():
    grid = to_mid_grid(_book_with_mid(), every="1s")
    # at t=1s the latest book <= 1s is the t=0 row → mid 100
    row1 = grid.filter(pl.col("ts") == dt.datetime(2026, 1, 1, 0, 0, 1))
    assert row1["mid"][0] == 100.0
    # at t=3s the latest book <= 3s is the t=2 row → mid 102
    row3 = grid.filter(pl.col("ts") == dt.datetime(2026, 1, 1, 0, 0, 3))
    assert row3["mid"][0] == 102.0


def test_forward_returns_uses_future_only():
    grid = pl.DataFrame({
        "ts": [dt.datetime(2026, 1, 1, 0, 0, s) for s in range(4)],
        "mid": [100.0, 110.0, 121.0, 121.0],
    })
    out = forward_returns(grid, horizons_secs=[1])
    # r(t->t+1s) at t0 = 110/100 - 1 = 0.10
    assert abs(out["fwd_1s"][0] - 0.10) < 1e-12
    # last row has no future point → null
    assert out["fwd_1s"][3] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/research/microstructure/test_align.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement align.py**

```python
# src/research/microstructure/align.py
"""Align event-level book onto a uniform time grid and build forward returns.

No-lookahead: grid mid uses a BACKWARD as-of join (latest book at/<= grid t);
forward returns use strictly future grid points.
"""
from __future__ import annotations

import polars as pl


def to_mid_grid(book: pl.DataFrame, *, every: str = "1s") -> pl.DataFrame:
    """Return a uniform grid (ts, mid) by backward as-of join onto the book."""
    b = book.with_columns(
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid")
    ).select(["ts", "mid"]).sort("ts")
    start, end = b["ts"].min(), b["ts"].max()
    grid = pl.datetime_range(start, end, interval=every, eager=True).alias("ts").to_frame()
    return grid.join_asof(b, on="ts", strategy="backward")


def forward_returns(grid: pl.DataFrame, *, horizons_secs: list[int]) -> pl.DataFrame:
    """Add fwd_<h>s columns: mid_{t+h}/mid_t - 1. Grid must be uniform 1s."""
    out = grid
    for h in horizons_secs:
        out = out.with_columns(
            (pl.col("mid").shift(-h) / pl.col("mid") - 1).alias(f"fwd_{h}s")
        )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/research/microstructure/test_align.py -v`
Expected: PASS (2 tests). If `datetime_range` errors on signature, use `pl.datetime_range(start, end, every, eager=True)`.

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/align.py tests/research/microstructure/test_align.py
git commit -m "feat(recon): mid-price grid (backward as-of) + forward returns"
```

---

### Task 6: IC computation + quantile layering

**Files:**
- Create: `src/research/microstructure/ic.py`
- Test: `tests/research/microstructure/test_ic.py`

- [ ] **Step 1: Write failing tests (Spearman recovery + layering)**

```python
# tests/research/microstructure/test_ic.py
import numpy as np
import polars as pl

from research.microstructure.ic import compute_ic, quantile_layering


def test_compute_ic_recovers_known_rank_correlation():
    # signal perfectly rank-correlated with forward return → IC ~ 1.0
    n = 500
    sig = np.linspace(-1, 1, n)
    fwd = sig * 2.0  # monotone increasing → Spearman 1.0
    df = pl.DataFrame({"qi": sig, "fwd_1s": fwd, "fwd_5s": -fwd})
    ic = compute_ic(df, signal_col="qi", horizon_cols=["fwd_1s", "fwd_5s"])
    assert abs(ic["fwd_1s"] - 1.0) < 1e-6
    assert abs(ic["fwd_5s"] + 1.0) < 1e-6  # perfectly anti-correlated


def test_quantile_layering_is_monotone_for_real_signal():
    n = 1000
    rng = np.random.default_rng(0)
    sig = rng.normal(size=n)
    fwd = sig * 0.5 + rng.normal(scale=0.1, size=n)  # signal predicts fwd
    df = pl.DataFrame({"qi": sig, "fwd_1s": fwd})
    buckets = quantile_layering(df, signal_col="qi", horizon_col="fwd_1s", n_buckets=5)
    means = buckets["mean_fwd"].to_list()
    assert means == sorted(means)  # monotone increasing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/research/microstructure/test_ic.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement ic.py**

```python
# src/research/microstructure/ic.py
"""Information Coefficient (Spearman) per horizon, and quantile layering.

IC magnitude alone can mislead; a real edge also shows monotone mean forward
return across signal quantiles (checked by quantile_layering).
"""
from __future__ import annotations

import polars as pl
from scipy.stats import spearmanr


def compute_ic(
    df: pl.DataFrame, *, signal_col: str, horizon_cols: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for hcol in horizon_cols:
        pair = df.select([signal_col, hcol]).drop_nulls()
        if pair.height < 3:
            out[hcol] = float("nan")
            continue
        rho, _ = spearmanr(pair[signal_col].to_numpy(), pair[hcol].to_numpy())
        out[hcol] = float(rho)
    return out


def quantile_layering(
    df: pl.DataFrame, *, signal_col: str, horizon_col: str, n_buckets: int = 5
) -> pl.DataFrame:
    """Mean forward return per signal quantile bucket (ascending by signal)."""
    pair = df.select([signal_col, horizon_col]).drop_nulls()
    pair = pair.with_columns(
        pl.col(signal_col).qcut(n_buckets, labels=[str(i) for i in range(n_buckets)])
        .alias("bucket")
    )
    return (
        pair.group_by("bucket")
        .agg(pl.col(horizon_col).mean().alias("mean_fwd"))
        .sort("bucket")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/research/microstructure/test_ic.py -v`
Expected: PASS (2 tests). If `qcut` signature differs, use `pl.col(signal_col).qcut(n_buckets)` and cast bucket to string.

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/ic.py tests/research/microstructure/test_ic.py
git commit -m "feat(recon): Spearman IC + quantile layering"
```

---

### Task 7: Markdown report renderer

**Files:**
- Create: `src/research/microstructure/report.py`
- Test: `tests/research/microstructure/test_report.py`

- [ ] **Step 1: Write failing test**

```python
# tests/research/microstructure/test_report.py
from research.microstructure.report import render_ic_markdown


def test_render_ic_markdown_table():
    ic_by_signal = {"qi": {"fwd_1s": 0.031, "fwd_60s": -0.004}}
    md = render_ic_markdown(ic_by_signal, n_tests=2)
    assert "| signal | fwd_1s | fwd_60s |" in md
    assert "0.031" in md
    assert "tests run: 2" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/research/microstructure/test_report.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement report.py**

```python
# src/research/microstructure/report.py
"""Render an IC-by-(signal x horizon) table to markdown."""
from __future__ import annotations


def render_ic_markdown(
    ic_by_signal: dict[str, dict[str, float]], *, n_tests: int
) -> str:
    horizons = list(next(iter(ic_by_signal.values())).keys())
    header = "| signal | " + " | ".join(horizons) + " |"
    sep = "|" + "---|" * (len(horizons) + 1)
    lines = [header, sep]
    for sig, ic in ic_by_signal.items():
        cells = " | ".join(f"{ic[h]:.3f}" for h in horizons)
        lines.append(f"| {sig} | {cells} |")
    lines.append("")
    lines.append(f"_tests run: {n_tests} (multiple-testing guard — see spec §7)_")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/research/microstructure/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/research/microstructure/report.py tests/research/microstructure/test_report.py
git commit -m "feat(recon): markdown IC report renderer"
```

---

### Task 8: End-to-end thin pipeline CLI + smoke

**Files:**
- Create: `scripts/recon/run_recon.py`
- Test: `tests/research/microstructure/test_run_recon.py`

- [ ] **Step 1: Write failing test for the pure pipeline function**

```python
# tests/research/microstructure/test_run_recon.py
import datetime as dt

import polars as pl

from scripts.recon.run_recon import recon_from_book


def test_recon_from_book_end_to_end():
    # 120s of synthetic book where qi predicts next-second up-move
    ts = [dt.datetime(2026, 1, 1, 0, 0, 0) + dt.timedelta(seconds=s) for s in range(120)]
    bid_qty = [5.0 if s % 2 == 0 else 1.0 for s in range(120)]
    ask_qty = [1.0 if s % 2 == 0 else 5.0 for s in range(120)]
    price = [100.0 + (1 if s % 2 == 0 else 0) for s in range(120)]
    book = pl.DataFrame({
        "ts": ts,
        "bid_price": price, "ask_price": price,
        "bid_qty": bid_qty, "ask_qty": ask_qty,
    })
    md, ic = recon_from_book(book, horizons_secs=[1, 5])
    assert "fwd_1s" in ic
    assert isinstance(md, str) and "signal" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/research/microstructure/test_run_recon.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement run_recon.py (pure function + CLI)**

```python
# scripts/recon/run_recon.py
"""End-to-end thin recon pipeline: book -> qi -> grid -> IC -> markdown.

Phase 1 wires the full flow with the single L1 signal. Phase 2 extends the
signal set once Step 0 confirms schemas.

Run (after download):
    PYTHONPATH=src venv/bin/python -m scripts.recon.run_recon \
        --book data/orderbook/ETHUSDT-bookTicker-2026-06-01.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from research.microstructure.align import forward_returns, to_mid_grid
from research.microstructure.download import load_book_ticker
from research.microstructure.ic import compute_ic
from research.microstructure.report import render_ic_markdown
from research.microstructure.signals import queue_imbalance


def recon_from_book(
    book: pl.DataFrame, *, horizons_secs: list[int]
) -> tuple[str, dict[str, float]]:
    """Pure pipeline: normalized book df -> (markdown, ic dict)."""
    grid = to_mid_grid(book, every="1s")
    grid = forward_returns(grid, horizons_secs=horizons_secs)
    qi = queue_imbalance(book)
    # as-of attach qi onto the grid (backward), then IC
    merged = grid.join_asof(qi.sort("ts"), on="ts", strategy="backward")
    hcols = [f"fwd_{h}s" for h in horizons_secs]
    ic = compute_ic(merged, signal_col="qi", horizon_cols=hcols)
    md = render_ic_markdown({"qi": ic}, n_tests=len(hcols))
    return md, ic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="normalized-or-raw bookTicker parquet")
    ap.add_argument("--horizons", default="1,5,10,30,60,300,900,3600")
    ap.add_argument("--out", default="docs/superpowers/recon/phase1_ic.md")
    args = ap.parse_args()

    book = load_book_ticker(Path(args.book))
    horizons = [int(x) for x in args.horizons.split(",")]
    md, ic = recon_from_book(book, horizons_secs=horizons)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/research/microstructure/test_run_recon.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full Phase-1 suite**

Run: `venv/bin/pytest tests/research/microstructure/ -v`
Expected: all PASS (download, schema_probe, signals, align, ic, report, run_recon).

- [ ] **Step 6: Commit**

```bash
git add scripts/recon/run_recon.py tests/research/microstructure/test_run_recon.py
git commit -m "feat(recon): end-to-end thin pipeline (qi -> IC) + smoke"
```

---

## Manual Step 0 (run once, after Task 8 — network, not in CI)

This is the de-risk gate from spec §3.1. Run against a real recent date:

```bash
PYTHONPATH=src venv/bin/python -m scripts.recon.probe_schema --symbol ETHUSDT --date 2026-06-01
```

**Record in a STATUS doc:** the actual raw columns for each data type, the
`bookDepth` cadence + whether it has a `percentage` column (percentage-depth
vs raw levels), and per-day file sizes. If `bookTicker` raw columns differ
from `BOOK_TICKER_MAP`/`BOOK_TICKER_TS_COL` in `download.py`, update those two
constants (signal/align/ic code is unaffected). These findings parameterize
Phase 2.

Then run the thin pipeline on one real day to confirm end-to-end:

```bash
PYTHONPATH=src venv/bin/python -m scripts.recon.run_recon \
    --book data/orderbook/_probe/ETHUSDT-bookTicker-2026-06-01.parquet
```

---

## Phase 2 (separate plan, written after Step 0)

Not in this plan — requires Step-0 confirmed schemas to stay no-placeholder.
Will add: OFI (Cont 2014) from bookTicker deltas; depth imbalance + book slope
from bookDepth (definition depends on percentage-vs-levels finding); taker
imbalance from aggTrades; BTC→ETH cross-asset lead-lag; chunked multi-day +
multi-month processing; OOS holdout; Newey-West/block-bootstrap significance;
coarse cost sensitivity; the plotly/notebook visual report; and the §10
decision-rule writeup.
