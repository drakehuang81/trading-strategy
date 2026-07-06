"""Phase 2 spot-availability audit — pre-registration §6, locked definition.

Every symbol the strategy ACTUALLY held (main + replication windows) must
have a hedgeable Binance USDT spot pair (1000x-prefixed perps hedge via the
unit-converted base). Symbols with no spot hedge are removed from the
universe and the ENTIRE gate chain re-runs; all gates must still pass or
the study verdict downgrades to FAIL.

Present-tense spot check is the locked definition; a perp whose spot pair
no longer exists today is excluded (conservative: its carry history is
dropped from the re-run).

Run:
    PYTHONPATH=src venv/bin/python -m scripts.carry.spot_audit --data data/carry
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import requests

from scripts.carry.study import (
    PRE_REGISTERED,
    lazy_control,
    load_daily,
    simulate,
    step0_gross_ceiling,
)

SPOT_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
THOUSANDS_PREFIX = re.compile(r"^1(?:000)+")  # 1000, 1000000, ...


def spot_bases(exchange_info: dict) -> set[str]:
    """TRADING spot pairs quoted in USDT -> set of base symbols."""
    return {
        s["baseAsset"]
        for s in exchange_info["symbols"]
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
    }


def hedgeable(perp_symbol: str, bases: set[str]) -> bool:
    """Perp has a USDT spot hedge, directly or via 1000x unit conversion."""
    base = perp_symbol.removesuffix("USDT")
    if base in bases:
        return True
    stripped = THOUSANDS_PREFIX.sub("", base)
    return stripped != base and stripped in bases


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/carry")
    args = ap.parse_args()

    daily = load_daily(Path(args.data) / "funding")
    train = tuple(map(d, PRE_REGISTERED["train"]))
    test = tuple(map(d, PRE_REGISTERED["test"]))
    rep = tuple(map(d, PRE_REGISTERED["replication"]))

    # 1) who did the original strategy actually hold?
    sim_main = simulate(daily, train[0], test[1])
    sim_rep = simulate(daily, rep[0], rep[1])
    held = sorted(sim_main.held_ever | sim_rep.held_ever)

    bases = spot_bases(requests.get(SPOT_INFO_URL, timeout=30).json())
    unhedgeable = [s for s in held if not hedgeable(s, bases)]
    print(f"held symbols: {len(held)} across both windows")
    print(f"no spot hedge today: {len(unhedgeable)} -> {unhedgeable}")

    # 2) locked consequence: drop them from the UNIVERSE and re-run everything
    all_syms = daily["symbol"].unique().to_list()
    keep = {s for s in all_syms if hedgeable(s, bases)}
    print(f"universe: {len(all_syms)} -> {len(keep)} after spot filter")
    import polars as pl
    filtered = daily.filter(pl.col("symbol").is_in(sorted(keep)))

    kill = PRE_REGISTERED["step0_kill_gross_apr"]
    ok = True
    for name, (lo, hi) in (("train", train), ("test", test)):
        c = step0_gross_ceiling(filtered, lo, hi)["trail_top5_gross_apr"]
        ok &= c >= kill
        print(f"Step0 {name}: {c:+.1%} (kill {kill:.0%})")

    sim = simulate(filtered, train[0], test[1])
    tr = sim.net_apr_deployed(*train)
    te = sim.net_apr_deployed(*test)
    te2x = sim.net_apr_deployed(*test, cost_mult=PRE_REGISTERED["g4_cost_multiplier"])
    lazy = lazy_control(filtered, *test)
    g1, g2 = tr > 0 and te > 0, te >= lazy + PRE_REGISTERED["g2_lazy_margin"]
    g3, g4 = te >= PRE_REGISTERED["g3_net_apr"], te2x > 0
    print(f"Phase1: train {tr:+.1%} test {te:+.1%} 2x {te2x:+.1%} lazy {lazy:+.1%} "
          f"-> G1 {g1} G2 {g2} G3 {g3} G4 {g4}")

    sim_r = simulate(filtered, rep[0], rep[1])
    mid = rep[0] + (rep[1] - rep[0]) / 2
    r1 = sim_r.net_apr_deployed(rep[0], mid)
    r2 = sim_r.net_apr_deployed(mid + dt.timedelta(days=1), rep[1])
    rfull = sim_r.net_apr_deployed(rep[0], rep[1])
    r2x = sim_r.net_apr_deployed(rep[0], rep[1], cost_mult=2.0)
    rlazy = lazy_control(filtered, rep[0], rep[1])
    rg = r1 > 0 and r2 > 0 and rfull >= rlazy + 0.02 and rfull >= 0.05 and r2x > 0
    print(f"Replication: halves {r1:+.1%}/{r2:+.1%} full {rfull:+.1%} "
          f"2x {r2x:+.1%} lazy {rlazy:+.1%} -> {rg}")

    ok = ok and g1 and g2 and g3 and g4 and rg
    print(f"\nPHASE 2 AUDIT: {'PASS — verdict stands' if ok else 'FAIL — verdict downgrades to FAIL'}")


if __name__ == "__main__":
    main()
