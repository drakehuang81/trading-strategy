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
