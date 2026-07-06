"""qi maker Phase A — pessimistic-fill lower bound on FREE archives.

Pre-registration: docs/superpowers/plans/2026-07-06-qi-maker-phaseA-preregistration.md
(PRE_REGISTERED mirrors it; committed before any sample-day download.)

Pessimistic maker cycle (long side; short mirrored):
  signal second t (qi >= +0.6, no open position/order) -> rest bid at
  best_bid(t); FILL only if some aggTrade prints STRICTLY BELOW that price
  within 10s (we take our resting price as the fill price); no fill -> free
  cancel. Exit taker at t_fill + h, hitting the then-current best_bid.
  Net bps = move - 7 (maker 2 + taker 5). Fills whose exit would cross the
  UTC day boundary are dropped (boundary effect, noted).

Run:
    PYTHONPATH=src venv/bin/python -m scripts.qimaker.phase_a download
    PYTHONPATH=src venv/bin/python -m scripts.qimaker.phase_a run
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import io
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests

PRE_REGISTERED = {
    "symbol": "ETHUSDT",
    "sample_days": "5th & 20th of each month, 2023-06 .. 2024-03 (20 days)",
    "qi_threshold": 0.6,          # fixed, symmetric
    "fill_window_s": 10,          # trade-through must occur within 10s
    "primary_horizon_s": 10,
    "desc_horizons_s": [1, 5, 30, 60],
    "fee_bps": 7.0,               # maker 2.0 entry + taker 5.0 exit
    "verdict": "DEAD if mean net/fill <= 0 @10s; WEAK if t<2 or <60% positive days; else PROCEED",
}
BASE = "https://data.binance.vision/data/futures/um/daily"
SECS_PER_DAY = 86_400


def sample_days() -> list[dt.date]:
    out = []
    y, m = 2023, 6
    for _ in range(10):
        out += [dt.date(y, m, 5), dt.date(y, m, 20)]
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# ---------------------------------------------------------------- data layer

def _read_csv_guarded(raw: bytes, keep: dict[str, str]) -> pl.DataFrame:
    """Read a Binance daily CSV keeping/renaming `keep` (header expected;
    recon probe confirmed daily um CSVs carry headers)."""
    df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
    return df.select(
        [pl.col(src).cast(pl.Float64 if dst != "ts_ms" else pl.Int64).alias(dst)
         for src, dst in keep.items()]
    )


def fetch_day(symbol: str, data_type: str, day: dt.date, out_dir: Path,
              keep: dict[str, str], sess: requests.Session) -> Path | None:
    dest = out_dir / f"{symbol}-{data_type}-{day.isoformat()}.parquet"
    if dest.exists():
        return dest
    url = f"{BASE}/{data_type}/{symbol}/{symbol}-{data_type}-{day.isoformat()}.zip"
    resp = sess.get(url, timeout=180)
    if resp.status_code == 404:
        return None                      # missing archive day — recorded upstream
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        df = _read_csv_guarded(zf.read(name), keep)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)               # zip never touches disk
    return dest


QUOTE_KEEP = {
    "transaction_time": "ts_ms",
    "best_bid_price": "bid",
    "best_bid_qty": "bid_qty",
    "best_ask_price": "ask",
    "best_ask_qty": "ask_qty",
}
TRADE_KEEP = {"transact_time": "ts_ms", "price": "price"}


# ------------------------------------------------------------- second grid

def per_second_grid(quotes: pl.DataFrame, day: dt.date) -> dict[str, list[float | None]]:
    """Forward-filled per-UTC-second arrays of the last quote in each second."""
    day_ms = int(dt.datetime(day.year, day.month, day.day,
                             tzinfo=dt.timezone.utc).timestamp() * 1000)
    sec = (
        quotes.with_columns(((pl.col("ts_ms") - day_ms) // 1000).alias("sec"))
        .filter((pl.col("sec") >= 0) & (pl.col("sec") < SECS_PER_DAY))
        .group_by("sec").agg(pl.all().last())
        .sort("sec")
    )
    grid: dict[str, list[float | None]] = {
        k: [None] * SECS_PER_DAY for k in ("bid", "ask", "bid_qty", "ask_qty")
    }
    for row in sec.iter_rows(named=True):
        s = row["sec"]
        for k in grid:
            grid[k][s] = row[k]
    for k in grid:                       # forward fill
        last = None
        col = grid[k]
        for i in range(SECS_PER_DAY):
            if col[i] is None:
                col[i] = last
            else:
                last = col[i]
    return grid


def qi_of(grid: dict[str, list[float | None]], s: int) -> float | None:
    bq, aq = grid["bid_qty"][s], grid["ask_qty"][s]
    if not bq or not aq or bq + aq == 0:
        return None
    return (bq - aq) / (bq + aq)


# ---------------------------------------------------------------- simulation

@dataclass
class Fill:
    side: int                    # +1 long (rest bid), -1 short (rest ask)
    signal_sec: int
    fill_ts_ms: int
    entry: float
    net_bps: dict[int, float]    # horizon -> net after fees
    adverse_bps: float           # mid-at-fill vs entry, signed against us


def simulate_day(grid: dict[str, list[float | None]], trades_ts: list[int],
                 trades_px: list[float], day: dt.date) -> tuple[int, list[Fill]]:
    """(n_signals, fills) under the registered pessimistic rules."""
    th = PRE_REGISTERED["qi_threshold"]
    window_ms = PRE_REGISTERED["fill_window_s"] * 1000
    horizons = sorted(set([PRE_REGISTERED["primary_horizon_s"]]
                          + PRE_REGISTERED["desc_horizons_s"]))
    fee = PRE_REGISTERED["fee_bps"]
    day_ms = int(dt.datetime(day.year, day.month, day.day,
                             tzinfo=dt.timezone.utc).timestamp() * 1000)
    n_signals = 0
    fills: list[Fill] = []
    busy_until_sec = 0
    for s in range(SECS_PER_DAY):
        if s < busy_until_sec:
            continue
        q = qi_of(grid, s)
        if q is None or abs(q) < th:
            continue
        side = 1 if q > 0 else -1
        entry = grid["bid"][s] if side == 1 else grid["ask"][s]
        if entry is None:
            continue
        n_signals += 1
        p0 = day_ms + s * 1000
        lo = bisect.bisect_right(trades_ts, p0)
        hi = bisect.bisect_right(trades_ts, p0 + window_ms)
        fill_ts = None
        for i in range(lo, hi):
            px = trades_px[i]
            if (side == 1 and px < entry) or (side == -1 and px > entry):
                fill_ts = trades_ts[i]
                break
        if fill_ts is None:
            busy_until_sec = s + PRE_REGISTERED["fill_window_s"]  # order rested
            continue
        fill_sec = (fill_ts - day_ms) // 1000
        if fill_sec + max(horizons) >= SECS_PER_DAY:
            busy_until_sec = SECS_PER_DAY   # boundary fill dropped (registered)
            continue
        nets: dict[int, float] = {}
        for h in horizons:
            ex_sec = fill_sec + h
            ex = grid["bid"][ex_sec] if side == 1 else grid["ask"][ex_sec]
            if ex is None:
                nets[h] = float("nan")
                continue
            move = (ex - entry) / entry * 1e4 * side
            nets[h] = move - fee
        mid_bid, mid_ask = grid["bid"][fill_sec], grid["ask"][fill_sec]
        adverse = 0.0
        if mid_bid is not None and mid_ask is not None:
            mid = (mid_bid + mid_ask) / 2
            adverse = (entry - mid) / entry * 1e4 * side
        fills.append(Fill(side, s, fill_ts, entry, nets, adverse))
        busy_until_sec = fill_sec + PRE_REGISTERED["primary_horizon_s"] + 1
    return n_signals, fills


# --------------------------------------------------------------------- main

def cmd_download(out: Path) -> None:
    sess = requests.Session()
    sym = PRE_REGISTERED["symbol"]
    missing = []
    for day in sample_days():
        q = fetch_day(sym, "bookTicker", day, out / "bookTicker", QUOTE_KEEP, sess)
        t = fetch_day(sym, "aggTrades", day, out / "aggTrades", TRADE_KEEP, sess)
        status = "ok" if (q and t) else "MISSING"
        if status == "MISSING":
            missing.append(day.isoformat())
        print(f"{day}: {status}", flush=True)
    print(f"download done; missing days: {missing or 'none'}")


def cmd_run(out: Path) -> None:
    sym = PRE_REGISTERED["symbol"]
    primary = PRE_REGISTERED["primary_horizon_s"]
    day_rows = []
    all_desc: dict[int, list[float]] = {h: [] for h in
                                        set([primary] + PRE_REGISTERED["desc_horizons_s"])}
    for day in sample_days():
        qp = out / "bookTicker" / f"{sym}-bookTicker-{day.isoformat()}.parquet"
        tp = out / "aggTrades" / f"{sym}-aggTrades-{day.isoformat()}.parquet"
        if not (qp.exists() and tp.exists()):
            continue
        grid = per_second_grid(pl.read_parquet(qp), day)
        trades = pl.read_parquet(tp).sort("ts_ms")
        n_signals, fills = simulate_day(
            grid, trades["ts_ms"].to_list(), trades["price"].to_list(), day
        )
        nets = [f.net_bps[primary] for f in fills
                if f.net_bps[primary] == f.net_bps[primary]]  # NaN guard
        for h in all_desc:
            all_desc[h] += [f.net_bps[h] for f in fills if f.net_bps[h] == f.net_bps[h]]
        if nets:
            day_rows.append({
                "day": day.isoformat(), "signals": n_signals, "fills": len(nets),
                "mean_net": statistics.fmean(nets), "sum_net": sum(nets),
                "adverse": statistics.fmean(f.adverse_bps for f in fills),
            })
        print(f"{day}: signals={n_signals} fills={len(nets)} "
              f"mean_net={statistics.fmean(nets) if nets else float('nan'):+.2f}bps",
              flush=True)

    if not day_rows:
        print("no fills anywhere — DEAD (nothing tradeable even pessimistically)")
        return
    means = [r["mean_net"] for r in day_rows]
    overall = statistics.fmean(means)
    tstat = (overall / (statistics.stdev(means) / len(means) ** 0.5)
             if len(means) > 1 and statistics.stdev(means) > 0 else float("nan"))
    pos_share = sum(m > 0 for m in means) / len(means)
    fill_rate = sum(r["fills"] for r in day_rows) / max(1, sum(r["signals"] for r in day_rows))
    print(f"\n=== Phase A verdict inputs (primary h={primary}s, fees 7bps) ===")
    print(f"days with fills: {len(day_rows)}/20  total fills: {sum(r['fills'] for r in day_rows)}")
    print(f"fill rate: {fill_rate:.1%}  mean adverse selection: "
          f"{statistics.fmean(r['adverse'] for r in day_rows):+.2f}bps")
    print(f"mean net per fill: {overall:+.2f}bps  t={tstat:.2f}  positive days: {pos_share:.0%}")
    print("descriptive horizons (mean net bps/fill): "
          + "  ".join(f"{h}s {statistics.fmean(v):+.2f}" for h, v in sorted(all_desc.items()) if v))
    if overall <= 0:
        print("\nVERDICT: DEAD — do NOT buy L2 data; gap to breakeven = "
              f"{-overall:.2f}bps/fill for queue-credit to bridge.")
    elif tstat < 2 or pos_share < 0.6:
        print("\nVERDICT: WEAK — do NOT buy; revisit on self-recorded data.")
    else:
        print("\nVERDICT: PROCEED — draft Phase B purchase proposal for user approval.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "run"])
    ap.add_argument("--out", default="data/qimaker")
    args = ap.parse_args()
    (cmd_download if args.cmd == "download" else cmd_run)(Path(args.out))


if __name__ == "__main__":
    main()
