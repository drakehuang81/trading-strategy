"""Spot-perp basis mean-reversion study — pre-registered, single shot.

Pre-registration: docs/superpowers/plans/2026-07-06-basis-meanreversion-preregistration.md
(constants below are its machine-readable mirror; committed before any
kline data was downloaded — only CSV schemas were probed).

Run:
    PYTHONPATH=src venv/bin/python -m scripts.basis.study --out data/basis download
    PYTHONPATH=src venv/bin/python -m scripts.basis.study --out data/basis run
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests

from scripts.carry.universe import S3_LIST_URL, parse_keys

PRE_REGISTERED = {
    "universe": "top-20 um-perp 24h quoteVolume with >=80% month coverage both markets",
    "n_universe": 20,
    "coverage_min": 0.80,
    "entry_bps": 60.0,        # |basis| >= 60bps triggers an episode
    "exit_bps": 10.0,         # converged
    "timeout_h": 48,
    "side": "positive basis only (long spot + short perp)",
    "rt_cost_bps": 40.0,      # 4 taker legs + slippage, per episode
    "deploy_factor": 1.4,
    "train": ("2022-07-01", "2024-06-30"),
    "test": ("2024-07-01", "2026-06-30"),
    "replication": ("2020-07-01", "2022-06-30"),
    "step0_kill_gross_apr": 0.10,   # top-5 symbols' gross capture, deployed
    "g2_min_episodes": 30,          # per half, portfolio-wide
    "g3_net_apr": 0.05,
    "g4_cost_multiplier": 2.0,
}
TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
FIRST_MONTH, LAST_MONTH = "2020-06", "2026-06"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
MONTH_1H_RE = re.compile(r"-1h-(\d{4}-\d{2})\.zip$")


# ---------------------------------------------------------------- data layer

def read_kline_csv(raw: bytes) -> pl.DataFrame:
    """Monthly 1h kline CSV -> (ts_ms, close). Spot files are headerless,
    um-futures files have a header row (probed 2026-07-06) — guard both."""
    has_header = raw.lstrip()[:9] == b"open_time"
    df = pl.read_csv(io.BytesIO(raw), has_header=has_header, infer_schema_length=0)
    df = df.rename(dict(zip(df.columns, KLINE_COLS[: len(df.columns)])))
    return df.select(
        pl.col("open_time").cast(pl.Int64).alias("ts_ms"),
        pl.col("close").cast(pl.Float64).alias("close"),
    ).drop_nulls()


def market_prefix(market: str, symbol: str) -> str:
    root = "data/spot" if market == "spot" else "data/futures/um"
    return f"{root}/monthly/klines/{symbol}/1h/"


def list_months(market: str, symbol: str, sess: requests.Session) -> set[str]:
    resp = sess.get(f"{S3_LIST_URL}?prefix={market_prefix(market, symbol)}", timeout=30)
    resp.raise_for_status()
    return {
        m.group(1)
        for k in parse_keys(resp.text)
        if (m := MONTH_1H_RE.search(k)) and FIRST_MONTH <= m.group(1) <= LAST_MONTH
    }


def window_months() -> list[str]:
    """Months of the MAIN window (coverage rule per pre-registration §1)."""
    out, cur = [], dt.date(2022, 7, 1)
    while cur <= dt.date(2026, 6, 30):
        out.append(cur.strftime("%Y-%m"))
        cur = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    return out


def pick_universe(sess: requests.Session) -> list[dict]:
    """Top-20 by perp 24h quoteVolume with >=80% main-window coverage in BOTH
    markets (same symbol name). Present-tense selection disclosed in the doc."""
    tickers = sess.get(TICKER_URL, timeout=30).json()
    ranked = sorted(
        (t for t in tickers if t["symbol"].endswith("USDT")),
        key=lambda t: float(t["quoteVolume"]),
        reverse=True,
    )
    need = window_months()
    chosen: list[dict] = []
    for t in ranked:
        sym = t["symbol"]
        cov = {
            mkt: len(list_months(mkt, sym, sess) & set(need)) / len(need)
            for mkt in ("spot", "um")
        }
        if min(cov.values()) >= PRE_REGISTERED["coverage_min"]:
            chosen.append({"symbol": sym, "coverage": cov,
                           "quote_volume_24h": float(t["quoteVolume"])})
            if len(chosen) == PRE_REGISTERED["n_universe"]:
                break
    return chosen


def fetch_series(market: str, symbol: str, out_dir: Path, sess: requests.Session) -> int:
    dest = out_dir / market / f"{symbol}.parquet"
    if dest.exists():
        return -1
    frames = []
    for month in sorted(list_months(market, symbol, sess)):
        url = f"https://data.binance.vision/{market_prefix(market, symbol)}{symbol}-1h-{month}.zip"
        resp = sess.get(url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            frames.append(read_kline_csv(zf.read(name)))
    if not frames:
        return 0
    df = pl.concat(frames).unique(subset=["ts_ms"], keep="last").sort("ts_ms")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    return len(df)


def build_basis(spot: pl.DataFrame, perp: pl.DataFrame) -> pl.DataFrame:
    """Inner-join on the hourly open_time -> (ts_ms, basis)."""
    return (
        perp.rename({"close": "perp_close"})
        .join(spot.rename({"close": "spot_close"}), on="ts_ms", how="inner")
        .with_columns(
            ((pl.col("perp_close") - pl.col("spot_close")) / pl.col("spot_close"))
            .alias("basis")
        )
        .select("ts_ms", "basis")
        .sort("ts_ms")
    )


# ------------------------------------------------------------- episode logic

@dataclass
class Episode:
    symbol: str
    side: int              # +1 = perp premium (tradeable), -1 = discount
    t_entry: int           # ms
    t_exit: int
    capture: float         # |basis_entry| - |basis_exit|, in fraction


def find_episodes(symbol: str, basis: pl.DataFrame) -> list[Episode]:
    """Non-overlapping episodes per the registered 60/10bps + 48h rules."""
    entry = PRE_REGISTERED["entry_bps"] / 1e4
    exit_ = PRE_REGISTERED["exit_bps"] / 1e4
    timeout_ms = PRE_REGISTERED["timeout_h"] * 3_600_000
    episodes: list[Episode] = []
    in_pos = False
    side = 0
    t0 = 0
    b0 = 0.0
    for ts, b in basis.iter_rows():
        if b is None:
            continue
        if not in_pos:
            if abs(b) >= entry:
                in_pos, side, t0, b0 = True, (1 if b > 0 else -1), ts, abs(b)
        else:
            timed_out = ts - t0 >= timeout_ms
            if abs(b) <= exit_ or timed_out:
                episodes.append(Episode(symbol, side, t0, ts, b0 - abs(b)))
                in_pos = False
    return episodes


def half_years(lo: dt.date, hi: dt.date) -> float:
    return ((hi - lo).days + 1) / 365.0


def portfolio_apr(
    episodes: list[Episode], lo: dt.date, hi: dt.date,
    n_symbols: int, cost_mult: float = 0.0,
) -> float:
    """Equal-weight-across-universe net APR on deployed capital.

    cost_mult=0 -> gross; 1 -> registered costs; 2 -> G4.
    Positive-basis side only (the registered tradeable side)."""
    cost = cost_mult * PRE_REGISTERED["rt_cost_bps"] / 1e4
    total = sum(
        e.capture - cost
        for e in episodes
        if e.side > 0 and lo <= dt.date.fromtimestamp(e.t_entry / 1000) <= hi
    )
    return total / n_symbols / half_years(lo, hi) / PRE_REGISTERED["deploy_factor"]


def count_pos(episodes: list[Episode], lo: dt.date, hi: dt.date) -> int:
    return sum(
        1 for e in episodes
        if e.side > 0 and lo <= dt.date.fromtimestamp(e.t_entry / 1000) <= hi
    )


def symbol_gross_apr(
    episodes: list[Episode], symbol: str, lo: dt.date, hi: dt.date
) -> float:
    total = sum(
        e.capture for e in episodes
        if e.symbol == symbol and e.side > 0
        and lo <= dt.date.fromtimestamp(e.t_entry / 1000) <= hi
    )
    return total / half_years(lo, hi) / PRE_REGISTERED["deploy_factor"]


# --------------------------------------------------------------------- main

def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def cmd_download(out: Path) -> None:
    sess = requests.Session()
    universe = pick_universe(sess)
    out.mkdir(parents=True, exist_ok=True)
    (out / "universe_snapshot.json").write_text(json.dumps(universe, indent=1))
    syms = [u["symbol"] for u in universe]
    print(f"universe ({len(syms)}): {syms}")
    jobs = [(mkt, s) for s in syms for mkt in ("spot", "um")]

    def job(args: tuple[str, str]) -> str:
        mkt, s = args
        try:
            n = fetch_series(mkt, s, out, requests.Session())
            return f"{mkt}/{s}: {n}"
        except Exception as e:  # noqa: BLE001
            return f"{mkt}/{s}: ERROR {e!r}"

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, msg in enumerate(ex.map(job, jobs), 1):
            if "ERROR" in msg or i % 10 == 0:
                print(f"  [{i}/{len(jobs)}] {msg}")
    print("download done")


def cmd_run(out: Path) -> None:
    syms = sorted(p.stem for p in (out / "um").glob("*.parquet"))
    episodes: list[Episode] = []
    for s in syms:
        spot = pl.read_parquet(out / "spot" / f"{s}.parquet")
        perp = pl.read_parquet(out / "um" / f"{s}.parquet")
        episodes.extend(find_episodes(s, build_basis(spot, perp)))
    n = len(syms)
    n_neg = sum(1 for e in episodes if e.side < 0)
    print(f"{n} symbols, {len(episodes)} episodes ({n_neg} negative-side, descriptive only)")

    train = tuple(map(d, PRE_REGISTERED["train"]))
    test = tuple(map(d, PRE_REGISTERED["test"]))

    print("\n=== Step 0: gross capture ceiling (positive side, no costs) ===")
    survived = True
    for name, (lo, hi) in (("train", train), ("test", test)):
        per_sym = sorted(
            (symbol_gross_apr(episodes, s, lo, hi) for s in syms), reverse=True
        )
        top5 = sum(per_sym[:5]) / 5
        survived &= top5 >= PRE_REGISTERED["step0_kill_gross_apr"]
        print(f"{name}: top-5 symbols gross APR (deployed) = {top5:+.1%} "
              f"-> {'PASS' if top5 >= PRE_REGISTERED['step0_kill_gross_apr'] else 'KILL'}")
    if not survived:
        print("\nSTEP 0 KILL -> basis mean-reversion on free Binance data CLOSED. "
              "Free-data mechanism space is now fully swept.")
        return

    print("\n=== Phase 1 ===")
    tr = portfolio_apr(episodes, *train, n_symbols=n, cost_mult=1.0)
    te = portfolio_apr(episodes, *test, n_symbols=n, cost_mult=1.0)
    te2x = portfolio_apr(episodes, *test, n_symbols=n,
                         cost_mult=PRE_REGISTERED["g4_cost_multiplier"])
    c_tr, c_te = count_pos(episodes, *train), count_pos(episodes, *test)
    g1 = tr > 0 and te > 0
    g2 = c_tr >= PRE_REGISTERED["g2_min_episodes"] and c_te >= PRE_REGISTERED["g2_min_episodes"]
    g3 = te >= PRE_REGISTERED["g3_net_apr"]
    g4 = te2x > 0
    print(f"episodes train/test = {c_tr}/{c_te}; net APR train {tr:+.1%} "
          f"test {te:+.1%} 2x {te2x:+.1%}")
    for name, ok in (("G1 OOS", g1), ("G2 mass", g2), ("G3 bar", g3), ("G4 cost", g4)):
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if not (g1 and g2 and g3 and g4):
        print("\nPhase 1 FAIL -> basis MR CLOSED; free-data mechanism space fully swept.")
        return

    rep = tuple(map(d, PRE_REGISTERED["replication"]))
    mid = rep[0] + (rep[1] - rep[0]) / 2
    r1 = portfolio_apr(episodes, rep[0], mid, n_symbols=n, cost_mult=1.0)
    r2 = portfolio_apr(episodes, mid + dt.timedelta(days=1), rep[1], n_symbols=n, cost_mult=1.0)
    rfull = portfolio_apr(episodes, rep[0], rep[1], n_symbols=n, cost_mult=1.0)
    r2x = portfolio_apr(episodes, rep[0], rep[1], n_symbols=n, cost_mult=2.0)
    rc = count_pos(episodes, rep[0], rep[1])
    rg = (r1 > 0 and r2 > 0 and rc >= 2 * PRE_REGISTERED["g2_min_episodes"]
          and rfull >= PRE_REGISTERED["g3_net_apr"] and r2x > 0)
    print(f"\n=== Replication === halves {r1:+.1%}/{r2:+.1%} full {rfull:+.1%} "
          f"2x {r2x:+.1%} episodes {rc}")
    print(f"FINAL: {'PASS & REPLICATED' if rg else 'FAIL -> CLOSED; free-data space fully swept.'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "run"])
    ap.add_argument("--out", default="data/basis")
    args = ap.parse_args()
    (cmd_download if args.cmd == "download" else cmd_run)(Path(args.out))


if __name__ == "__main__":
    main()
