"""v2 study runner: funding carry on the point-in-time hedgeable universe.

Pre-registration: docs/superpowers/plans/2026-07-06-funding-carry-pit-preregistration.md
Identical constants/gates to v1 (study.py::PRE_REGISTERED); the ONLY change
is the eligibility filter (PIT spot hedgeability) plus forced exit on
mid-hold spot delisting (implemented in simulate's ineligibility exit).

Run (after scripts.carry.universe + scripts.carry.spot_history):
    PYTHONPATH=src venv/bin/python -m scripts.carry.study_pit --data data/carry
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from scripts.carry.study import (
    PRE_REGISTERED,
    apply_pit_hedgeability,
    lazy_control,
    load_daily,
    simulate,
    step0_gross_ceiling,
)


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/carry")
    args = ap.parse_args()
    data = Path(args.data)

    perp_months = json.loads((data / "spot_months.json").read_text())
    daily = apply_pit_hedgeability(load_daily(data / "funding"), perp_months)
    n_hedged = sum(1 for m in perp_months.values() if m)
    print(f"universe: {daily['symbol'].n_unique()} perps, "
          f"{n_hedged} with any spot history (PIT filter active)")

    train = tuple(map(d, PRE_REGISTERED["train"]))
    test = tuple(map(d, PRE_REGISTERED["test"]))
    rep = tuple(map(d, PRE_REGISTERED["replication"]))
    kill = PRE_REGISTERED["step0_kill_gross_apr"]

    print("\n=== v2 Step 0: gross ceiling kill-test (PIT universe) ===")
    survived = True
    for name, (lo, hi) in (("train", train), ("test", test)):
        m = step0_gross_ceiling(daily, lo, hi)
        c = m["trail_top5_gross_apr"]
        survived &= c >= kill
        print(f"{name}: trail-top5 {c:+.1%} (oracle top5 "
              f"{m['oracle_top5_gross_apr']:+.1%}) -> {'PASS' if c >= kill else 'KILL'}")
    if not survived:
        print("\nSTEP 0 KILL -> v2 FAIL; carry closed under BOTH evidence bases.")
        return

    print("\n=== v2 Phase 1: four gates ===")
    sim = simulate(daily, train[0], test[1])
    tr = sim.net_apr_deployed(*train)
    te = sim.net_apr_deployed(*test)
    te2x = sim.net_apr_deployed(*test, cost_mult=PRE_REGISTERED["g4_cost_multiplier"])
    lazy = lazy_control(daily, *test)
    print(f"entries={sim.entries} exits={sim.exits} "
          f"avg slots={sum(sim.n_held)/len(sim.n_held):.2f}")
    print(f"train {tr:+.1%}  test {te:+.1%}  2x {te2x:+.1%}  lazy {lazy:+.1%}")
    g1 = tr > 0 and te > 0
    g2 = te >= lazy + PRE_REGISTERED["g2_lazy_margin"]
    g3 = te >= PRE_REGISTERED["g3_net_apr"]
    g4 = te2x > 0
    for name, ok in (("G1 OOS", g1), ("G2 lazy", g2), ("G3 bar", g3), ("G4 cost", g4)):
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if not (g1 and g2 and g3 and g4):
        print("\nv2 FAIL -> carry closed under BOTH evidence bases.")
        return

    print("\n=== v2 Replication ===")
    sim_r = simulate(daily, rep[0], rep[1])
    mid = rep[0] + (rep[1] - rep[0]) / 2
    r1 = sim_r.net_apr_deployed(rep[0], mid)
    r2 = sim_r.net_apr_deployed(mid + dt.timedelta(days=1), rep[1])
    rfull = sim_r.net_apr_deployed(rep[0], rep[1])
    r2x = sim_r.net_apr_deployed(rep[0], rep[1], cost_mult=2.0)
    rlazy = lazy_control(daily, rep[0], rep[1])
    rg = r1 > 0 and r2 > 0 and rfull >= rlazy + PRE_REGISTERED["g2_lazy_margin"] \
        and rfull >= PRE_REGISTERED["g3_net_apr"] and r2x > 0
    print(f"halves {r1:+.1%}/{r2:+.1%}  full {rfull:+.1%}  2x {r2x:+.1%}  lazy {rlazy:+.1%}")
    print(f"\nv2 FINAL: {'PASS & REPLICATED (PIT universe)' if rg else 'FAIL -> carry closed under BOTH evidence bases.'}")


if __name__ == "__main__":
    main()
