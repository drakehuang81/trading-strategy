# Order Book Recon — Phase 2b-2 (BTC→ETH Cross-Asset Lead-Lag) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test the last untested free-data hypothesis: does BTC's order book (hourly depth imbalance) predict ETH's forward 1h return **beyond what BTC's own price momentum already tells you** — via the same pre-committed four-gate harness, over the same 2023-05-16 → 2024-03-30 window.

**Architecture:** Reuses the Phase 2b-1 validation harness end-to-end (`build_hourly`, `time_split`, `ic`, `decile_edge_bps`, `newey_west_tstat`). Adds one dataset builder (`build_hourly_cross`: lead symbol's book + lead klines + lag klines → aligned hourly rows) and one driver (`cross_validation.py`) whose `summarize_cross` upgrades the momentum control to **dual controls** — the signal's |IC| must beat BOTH the lag asset's own momentum AND the lead asset's momentum (the known BTC→ETH price lead-lag is the null hypothesis to reject).

**Tech Stack:** unchanged (polars, scipy, statsmodels, pytest). Same worktree/venv as Phase 2b-1.

**Pre-committed gates (before seeing results):** horizon = 1h only. Verdict = REAL-ALPHA candidate iff (a) OOS: train/test IC same sign and |ic_test| > 0.1; (b) vs-controls: |ic(di_btc)| > max(|ic(past_1h_eth)|, |ic(past_1h_btc)|) + 0.05; (c) post-cost: decile gross edge − 8 bps taker > 0; (d) full-bucket monotone.

---

## File Structure

| Path | Responsibility | Change |
|---|---|---|
| `src/research/microstructure/depth_study.py` | add `build_hourly_cross` | modify |
| `scripts/recon/cross_validation.py` | `summarize_cross` (dual controls) + per-day download loop + CLI | **create** |
| `tests/research/microstructure/test_depth_study.py` | cross builder test | modify |
| `tests/research/microstructure/test_cross_validation.py` | summarize_cross test | **create** |

---

### Task 1: build_hourly_cross

**Files:** Modify `src/research/microstructure/depth_study.py`; Test `tests/research/microstructure/test_depth_study.py`

- [ ] **Step 1: append failing test** to `test_depth_study.py`:

```python
def test_build_hourly_cross_lead_book_lag_return():
    from research.microstructure.depth_study import build_hourly_cross
    h0 = dt.datetime(2023, 6, 1, 0, 0, 0)
    h1 = dt.datetime(2023, 6, 1, 1, 0, 0)
    h2 = dt.datetime(2023, 6, 1, 2, 0, 0)
    depth_btc = pl.DataFrame({
        "ts": [h0, h0, h1, h1],
        "percentage": [-1.0, 1.0, -1.0, 1.0],
        "depth": [30.0, 10.0, 10.0, 30.0],
    })
    klines_btc = pl.DataFrame({"hour": [h0, h1, h2], "close": [200.0, 202.0, 201.0]})
    klines_eth = pl.DataFrame({"hour": [h0, h1, h2], "close": [100.0, 105.0, 99.0]})
    ds = build_hourly_cross(depth_btc, klines_btc, klines_eth)
    row0 = ds.filter(pl.col("hour") == h0)
    assert abs(row0["di"][0] - 0.5) < 1e-9            # di from the BTC book
    assert abs(row0["fwd_1h"][0] - 0.05) < 1e-9       # target = ETH 100->105
    row1 = ds.filter(pl.col("hour") == h1)
    assert abs(row1["past_1h"][0] - 0.05) < 1e-9      # ETH's own momentum
    assert abs(row1["past_1h_lead"][0] - 0.01) < 1e-9  # BTC momentum 200->202
```

- [ ] **Step 2: run, verify FAIL** (ImportError): `./venv/bin/pytest tests/research/microstructure/test_depth_study.py -v`

- [ ] **Step 3: implement.** Append to `src/research/microstructure/depth_study.py`:

```python
def build_hourly_cross(
    depth_lead: pl.DataFrame,
    klines_lead: pl.DataFrame,
    klines_lag: pl.DataFrame,
) -> pl.DataFrame:
    """Cross-asset hourly dataset: lead symbol's book vs lag symbol's return.

    Output (hour, di, past_1h_lead, fwd_1h, past_1h): di = LEAD symbol's
    hourly mean depth imbalance; past_1h_lead = lead's own trailing 1h return;
    fwd_1h / past_1h = LAG symbol's forward / trailing 1h return. fwd_1h (lag)
    is the prediction target.
    """
    lead = build_hourly(depth_lead, klines_lead).sort("hour")
    lead = lead.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("past_1h_lead")
    ).select(["hour", "di", "past_1h_lead"])
    lag = klines_lag.sort("hour").with_columns(
        (pl.col("close").shift(-1) / pl.col("close") - 1).alias("fwd_1h"),
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("past_1h"),
    ).select(["hour", "fwd_1h", "past_1h"])
    return (
        lead.join(lag, on="hour", how="inner")
        .drop_nulls(["di", "fwd_1h"])
        .sort("hour")
    )
```

- [ ] **Step 4: run, verify PASS**, then full scoped suite (expect 36 passed): `./venv/bin/pytest tests/research/microstructure/ -v`

- [ ] **Step 5: commit**
```bash
git add src/research/microstructure/depth_study.py tests/research/microstructure/test_depth_study.py
git commit -m "$(printf 'feat(recon): build_hourly_cross — lead book vs lag return dataset\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: cross_validation driver (dual-control gate)

**Files:** Create `scripts/recon/cross_validation.py`; Test `tests/research/microstructure/test_cross_validation.py`

- [ ] **Step 1: create failing test** `tests/research/microstructure/test_cross_validation.py`:

```python
import datetime as dt
import polars as pl

from scripts.recon.cross_validation import summarize_cross


def test_summarize_cross_dual_controls_and_verdict():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 2000
    di = rng.normal(size=n)
    ds = pl.DataFrame({
        "hour": [dt.datetime(2023, 6, 1) + dt.timedelta(hours=i) for i in range(n)],
        "di": di,
        "past_1h_lead": rng.normal(size=n),
        "past_1h": rng.normal(size=n),
        "fwd_1h": di * 0.001 + rng.normal(scale=0.002, size=n),
    })
    rep = summarize_cross(ds)
    for key in ("ic_all", "ic_train", "ic_test", "ic_momentum_lag",
                "ic_momentum_lead", "gross_bps", "net_std_taker_bps",
                "nw_tstat", "monotone", "verdict", "n_hours"):
        assert key in rep
    assert rep["verdict"] == "REAL-ALPHA candidate"  # synthetic true signal
```

- [ ] **Step 2: run, verify FAIL** (ImportError).

- [ ] **Step 3: create** `scripts/recon/cross_validation.py`:

```python
"""BTC->ETH cross-asset lead-lag validation driver.

Does the LEAD symbol's hourly depth imbalance predict the LAG symbol's
forward 1h return beyond BOTH assets' own price momentum? Same four
pre-committed gates as depth_validation, with a dual momentum control.

Run (network — manual):
    PYTHONPATH=src venv/bin/python -m scripts.recon.cross_validation \
        --lead BTCUSDT --lag ETHUSDT --start 2023-05-16 --end 2024-03-30
"""
from __future__ import annotations

import argparse
import datetime as dt
import zipfile
from pathlib import Path

import polars as pl

from research.microstructure.depth_study import (
    build_hourly_cross, decile_edge_bps, ic, newey_west_tstat, time_split,
)
from research.microstructure.download import (
    build_url, download_zip, extract_zip_to_parquet,
    klines_1h_url, load_book_depth, load_klines_1h,
)


def summarize_cross(ds: pl.DataFrame) -> dict:
    """Pure gate: (hour, di, past_1h_lead, past_1h, fwd_1h) -> verdict report."""
    train, test = time_split(ds, train_frac=0.7)
    edge = decile_edge_bps(ds)
    ic_all, ic_tr, ic_te = ic(ds, "di"), ic(train, "di"), ic(test, "di")
    ic_mom_lag = ic(ds, "past_1h")
    ic_mom_lead = ic(ds, "past_1h_lead")
    d2 = ds.drop_nulls(["di", "fwd_1h"])
    thr = d2["di"].median()
    strat = d2.with_columns(
        pl.when(pl.col("di") > thr).then(pl.col("fwd_1h"))
        .otherwise(-pl.col("fwd_1h")).alias("strat_ret")
    )
    nw = newey_west_tstat(strat["strat_ret"].to_numpy(), lags=5)
    monotone = edge["means"] == sorted(edge["means"])
    beats_controls = abs(ic_all) > max(abs(ic_mom_lag), abs(ic_mom_lead)) + 0.05
    survives_oos = (ic_tr * ic_te > 0) and abs(ic_te) > 0.1
    tradeable = edge["net_std_taker_bps"] > 0
    verdict = (
        "REAL-ALPHA candidate"
        if (survives_oos and beats_controls and tradeable and monotone)
        else "FAILED — " + ", ".join(
            x for x, ok in [
                ("OOS", survives_oos), ("vs-controls", beats_controls),
                ("post-cost", tradeable), ("monotone", monotone),
            ] if not ok
        )
    )
    return {
        "ic_all": ic_all, "ic_train": ic_tr, "ic_test": ic_te,
        "ic_momentum_lag": ic_mom_lag, "ic_momentum_lead": ic_mom_lead,
        "gross_bps": edge["gross_bps"],
        "net_std_taker_bps": edge["net_std_taker_bps"], "nw_tstat": nw,
        "monotone": monotone, "verdict": verdict, "n_hours": ds.height,
    }


def _fetch_depth(symbol: str, d: dt.date, out: Path) -> pl.DataFrame:
    p = out / f"{symbol}-bd-{d}.parquet"
    if not p.exists():
        extract_zip_to_parquet(
            download_zip(build_url(symbol, "bookDepth", d), out / f"{symbol}-bd-{d}.zip"), p,
        )
    return load_book_depth(p)


def _fetch_klines(symbol: str, d: dt.date, out: Path) -> pl.DataFrame:
    c = out / f"{symbol}-kl-{d}.csv"
    if not c.exists():
        z = out / f"{symbol}-kl-{d}.zip"
        download_zip(klines_1h_url(symbol, d), z)
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            zf.extract(name, out)
            (out / name).rename(c)
    return load_klines_1h(c)


def _daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def build_window_cross(
    lead: str, lag: str, start: dt.date, end: dt.date, out: Path
) -> pl.DataFrame:
    parts = []
    for d in _daterange(start, end):
        try:
            day = build_hourly_cross(
                _fetch_depth(lead, d, out),
                _fetch_klines(lead, d, out),
                _fetch_klines(lag, d, out),
            )
            parts.append(day)
        except Exception as e:  # noqa: BLE001
            print(f"  {d} skipped: {e}")
    return pl.concat(parts).sort("hour")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", default="BTCUSDT")
    ap.add_argument("--lag", default="ETHUSDT")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out-dir", default="data/orderbook/_cross")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = build_window_cross(
        args.lead, args.lag,
        dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end), out,
    )
    rep = summarize_cross(ds)
    print(f"\n=== {args.lead} book -> {args.lag} 1h — cross-asset validation ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run, verify PASS**, then full scoped suite (expect 37 passed).

- [ ] **Step 5: commit**
```bash
git add scripts/recon/cross_validation.py tests/research/microstructure/test_cross_validation.py
git commit -m "$(printf 'feat(recon): cross-asset lead-lag driver with dual momentum controls\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## THE GATE (manual, network — BTC depth+klines ~200MB, ETH klines tiny)

```bash
PYTHONPATH=src venv/bin/python -m scripts.recon.cross_validation \
    --lead BTCUSDT --lag ETHUSDT --start 2023-05-16 --end 2024-03-30
```

- **REAL-ALPHA candidate** → BTC-book → ETH is the lead; next: capacity/regime breakdown, then wire as a 1h feature.
- **FAILED** → record the negative result. That closes the last free-data hypothesis; remaining paths are qi maker/HF (different architecture) or a different market.
