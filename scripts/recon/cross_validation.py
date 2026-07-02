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
