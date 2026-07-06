"""Cross-universe funding carry study — Step 0 kill-test + Phase 1 gates.

PRE_REGISTERED below is the machine-readable mirror of
docs/superpowers/plans/2026-07-05-funding-carry-preregistration.md.
All constants were locked (and committed) BEFORE any universe data was
downloaded. Changing them after seeing results voids the study.

Positive-funding side only: long spot + short perp collects funding when
funding > 0. Daily UTC grid; day_funding = sum of that day's rates (handles
8h/4h interval heterogeneity). All APRs on DEPLOYED capital (1.4x notional)
unless labelled gross/notional.

Run (after scripts.carry.universe has populated data/carry/funding/):
    PYTHONPATH=src venv/bin/python -m scripts.carry.study --data data/carry
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

PRE_REGISTERED = {
    "universe": "S3 fundingRate folders, USDT-quoted, incl. delisted",
    "side": "positive funding only (long spot + short perp)",
    "grid": "daily UTC; day_funding = sum(rates in day)",
    "seasoning_days": 30,          # listed >= 30d before eligible
    "trail_days": 3,               # signal = trailing 3d funding sum
    "entry_apr": 0.10,             # enter when trail APR > 10%
    "exit_apr": 0.05,              # exit when trail APR < 5% (hysteresis)
    "slots": 5,                    # top-K equal-weight slots
    "half_rt_cost": 0.0020,        # 20bps per entry AND per exit (RT 40bps)
    "deploy_factor": 1.4,          # spot 1.0 + perp margin 0.4
    "train": ("2022-07-01", "2024-06-30"),
    "test": ("2024-07-01", "2026-06-30"),
    "replication": ("2020-07-01", "2022-06-30"),
    "step0_kill_gross_apr": 0.10,  # top-5 trailing-selected next-day gross
    "g2_lazy_margin": 0.02,        # beat BTC+ETH 50/50 carry by +2pp
    "g3_net_apr": 0.05,
    "g4_cost_multiplier": 2.0,
    # implementation decisions fixed pre-run (documented before first run):
    # - trail3 needs 3 consecutive calendar days present, else signal is null
    # - null signal while held => exit (missing/ended data is not holdable)
    # - fewer than K qualifying candidates => hold fewer; idle weight earns 0
}


def load_daily(funding_dir: Path) -> pl.DataFrame:
    """All symbol parquets -> (symbol, day, fund_day, eligible) daily table.

    Calendar is made contiguous per symbol between its first and last day:
    gap days exist with fund_day=null so trailing windows see the hole
    (recon lesson: silent gaps fabricate signals).
    """
    frames = []
    for p in sorted(funding_dir.glob("*.parquet")):
        df = pl.read_parquet(p)
        if df.is_empty():
            continue
        daily = (
            df.with_columns(
                pl.from_epoch(pl.col("ts_ms"), time_unit="ms")
                .dt.date()
                .alias("day")
            )
            .group_by("day")
            .agg(pl.col("rate").sum().alias("fund_day"))
            .sort("day")
            .upsample(time_column="day", every="1d")  # gap days -> null
            .with_columns(pl.lit(p.stem).alias("symbol"))
        )
        frames.append(daily)
    out = pl.concat(frames)
    first = out.filter(pl.col("fund_day").is_not_null()).group_by("symbol").agg(
        pl.col("day").min().alias("first_day")
    )
    return (
        out.join(first, on="symbol")
        .with_columns(
            (
                pl.col("day")
                >= pl.col("first_day") + pl.duration(days=PRE_REGISTERED["seasoning_days"])
            ).alias("eligible")
        )
        .select("symbol", "day", "fund_day", "eligible")
        .sort("symbol", "day")
    )


def with_trail_apr(daily: pl.DataFrame) -> pl.DataFrame:
    """Add trail_apr: trailing 3d funding sum (D-3..D-1) annualized at D.

    rolling_sum over the contiguous calendar (nulls poison the window, which
    is exactly the pre-registered "3 days complete or no signal" rule).
    """
    n = PRE_REGISTERED["trail_days"]
    return daily.with_columns(
        (
            pl.col("fund_day").rolling_sum(n).shift(1).over("symbol")
            * (365.0 / n)
        ).alias("trail_apr")
    )


def apply_pit_hedgeability(
    daily: pl.DataFrame, perp_months: dict[str, list[str]]
) -> pl.DataFrame:
    """AND point-in-time spot hedgeability into `eligible` (v2 study).

    perp_months: perp symbol -> YYYY-MM months its spot hedge traded
    (from scripts.carry.spot_history). Days outside those months lose
    eligibility, which blocks entries AND force-exits held positions
    (simulate exits on ineligibility).
    """
    pairs = [(s, m) for s, months in perp_months.items() for m in months]
    hedge = pl.DataFrame(
        {"symbol": [p[0] for p in pairs], "month": [p[1] for p in pairs]},
        schema={"symbol": pl.Utf8, "month": pl.Utf8},
    ).with_columns(pl.lit(True).alias("hedge_ok"))
    return (
        daily.with_columns(pl.col("day").dt.strftime("%Y-%m").alias("month"))
        .join(hedge, on=["symbol", "month"], how="left")
        .with_columns(
            (pl.col("eligible") & pl.col("hedge_ok").fill_null(False)).alias("eligible")
        )
        .drop("month", "hedge_ok")
    )


@dataclass
class SimResult:
    """Daily net PnL on NOTIONAL (deploy scaling applied at reporting)."""
    days: list[dt.date] = field(default_factory=list)
    net: list[float] = field(default_factory=list)          # funding - costs
    gross: list[float] = field(default_factory=list)
    n_held: list[int] = field(default_factory=list)
    entries: int = 0
    exits: int = 0
    held_ever: set[str] = field(default_factory=set)        # audit trail

    def net_apr_deployed(self, start: dt.date, end: dt.date, cost_mult: float = 1.0) -> float:
        rows = [
            (g, n) for d, g, n in zip(self.days, self.gross, self.net)
            if start <= d <= end
        ]
        if not rows:
            return float("nan")
        # cost component of each day = net - gross (<= 0); scale it by mult
        daily = [g + cost_mult * (n - g) for g, n in rows]
        apr_notional = sum(daily) / len(daily) * 365.0
        return apr_notional / PRE_REGISTERED["deploy_factor"]


def simulate(daily: pl.DataFrame, start: dt.date, end: dt.date) -> SimResult:
    """Pre-registered top-K trailing-funding rotation with hysteresis exits."""
    k = PRE_REGISTERED["slots"]
    w = 1.0 / k
    cost = PRE_REGISTERED["half_rt_cost"]
    entry_apr = PRE_REGISTERED["entry_apr"]
    exit_apr = PRE_REGISTERED["exit_apr"]

    window = with_trail_apr(daily).filter(
        (pl.col("day") >= start) & (pl.col("day") <= end)
    )
    by_day: dict[dt.date, dict[str, tuple[float | None, float | None, bool]]] = {}
    for sym, day, fund, elig, trail in window.select(
        "symbol", "day", "fund_day", "eligible", "trail_apr"
    ).iter_rows():
        by_day.setdefault(day, {})[sym] = (fund, trail, elig)

    held: set[str] = set()
    res = SimResult()
    for day in sorted(by_day):
        rows = by_day[day]
        day_cost = 0.0
        # exits first: signal below band, data missing/ended, or hedge gone.
        # (Ineligibility exit is a no-op for v1 frames — seasoning-only
        # eligibility is monotone — and implements v2's forced exit when
        # the spot hedge delists mid-hold.)
        for sym in sorted(held):
            _, trail, elig = rows.get(sym, (None, None, False))
            if trail is None or trail < exit_apr or not elig:
                held.discard(sym)
                day_cost += w * cost
                res.exits += 1
        # entries: best trail APR above entry band, eligible, fill free slots
        candidates = sorted(
            (
                (trail, sym)
                for sym, (fund, trail, elig) in rows.items()
                if sym not in held and elig and trail is not None and trail > entry_apr
            ),
            reverse=True,
        )
        for trail, sym in candidates[: max(0, k - len(held))]:
            held.add(sym)
            res.held_ever.add(sym)
            day_cost += w * cost
            res.entries += 1
        # collect today's funding on held names (missing counts as 0)
        day_gross = sum(
            w * (rows.get(sym, (None, None, False))[0] or 0.0) for sym in held
        )
        res.days.append(day)
        res.gross.append(day_gross)
        res.net.append(day_gross - day_cost)
        res.n_held.append(len(held))
    return res


def lazy_control(daily: pl.DataFrame, start: dt.date, end: dt.date) -> float:
    """BTC+ETH 50/50 always-on carry, one entry cost — net APR (deployed)."""
    rows = (
        daily.filter(
            pl.col("symbol").is_in(["BTCUSDT", "ETHUSDT"])
            & (pl.col("day") >= start)
            & (pl.col("day") <= end)
        )
        .group_by("day")
        .agg(pl.col("fund_day").mean().alias("f"))
        .sort("day")
    )
    if rows.is_empty():
        return float("nan")
    total = rows["f"].fill_null(0.0).sum() - PRE_REGISTERED["half_rt_cost"]
    apr_notional = total / len(rows) * 365.0
    return apr_notional / PRE_REGISTERED["deploy_factor"]


def step0_gross_ceiling(daily: pl.DataFrame, start: dt.date, end: dt.date) -> dict[str, float]:
    """Gross (no-cost) APRs on notional: trailing top-5 realized, + oracles."""
    k = PRE_REGISTERED["slots"]
    window = with_trail_apr(daily).filter(
        (pl.col("day") >= start) & (pl.col("day") <= end) & pl.col("eligible")
    )
    trail_sel = (
        window.filter(pl.col("trail_apr").is_not_null())
        .sort(["day", "trail_apr"], descending=[False, True])
        .group_by("day", maintain_order=True)
        .agg(pl.col("fund_day").head(k).fill_null(0.0).mean().alias("g"))
    )
    oracle5 = (
        window.filter(pl.col("fund_day").is_not_null())
        .sort(["day", "fund_day"], descending=[False, True])
        .group_by("day", maintain_order=True)
        .agg(pl.col("fund_day").head(k).mean().alias("g"))
    )
    oracle1 = (
        window.filter(pl.col("fund_day").is_not_null())
        .group_by("day")
        .agg(pl.col("fund_day").max().alias("g"))
    )
    n_days = (end - start).days + 1

    def apr(df: pl.DataFrame) -> float:
        return float(df["g"].sum()) / n_days * 365.0 if not df.is_empty() else float("nan")

    return {
        "trail_top5_gross_apr": apr(trail_sel),
        "oracle_top5_gross_apr": apr(oracle5),
        "oracle_top1_gross_apr": apr(oracle1),
    }


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/carry")
    ap.add_argument("--phase1", action="store_true",
                    help="run Phase 1 gates (only if Step 0 survived)")
    ap.add_argument("--replication", action="store_true",
                    help="run replication window (only after a full PASS)")
    args = ap.parse_args()

    daily = load_daily(Path(args.data) / "funding")
    n_sym = daily["symbol"].n_unique()
    print(f"daily table: {len(daily)} rows, {n_sym} symbols")

    train = tuple(map(d, PRE_REGISTERED["train"]))
    test = tuple(map(d, PRE_REGISTERED["test"]))
    kill = PRE_REGISTERED["step0_kill_gross_apr"]

    print("\n=== Step 0: gross ceiling kill-test (on notional, no costs) ===")
    survived = True
    for name, (lo, hi) in (("train", train), ("test", test)):
        m = step0_gross_ceiling(daily, lo, hi)
        c = m["trail_top5_gross_apr"]
        verdict = "PASS" if c >= kill else "KILL"
        survived &= c >= kill
        print(
            f"{name}: trail-top5 gross APR = {c:+.1%}  "
            f"(oracle top5 {m['oracle_top5_gross_apr']:+.1%}, "
            f"top1 {m['oracle_top1_gross_apr']:+.1%})  -> {verdict}"
        )
    if not survived:
        print("\nSTEP 0 KILL: gross ceiling below pre-registered 10% APR — "
              "funding carry question CLOSED (no Phase 1).")
        return
    print("\nStep 0 survived. Run with --phase1 for the gate backtest.")

    if not args.phase1:
        return

    print("\n=== Phase 1: pre-registered rotation, four gates ===")
    sim = simulate(daily, train[0], test[1])
    tr = sim.net_apr_deployed(*train)
    te = sim.net_apr_deployed(*test)
    te2x = sim.net_apr_deployed(*test, cost_mult=PRE_REGISTERED["g4_cost_multiplier"])
    lazy = lazy_control(daily, *test)
    held_days = [n for n in sim.n_held if n > 0]
    print(f"entries={sim.entries} exits={sim.exits} "
          f"avg slots held={sum(sim.n_held)/len(sim.n_held):.2f} "
          f"days in market={len(held_days)}/{len(sim.n_held)}")
    print(f"train net APR (deployed) = {tr:+.1%}")
    print(f"test  net APR (deployed) = {te:+.1%}")
    print(f"test  net APR @2x costs  = {te2x:+.1%}")
    print(f"lazy BTC+ETH control     = {lazy:+.1%}")

    g1 = tr > 0 and te > 0
    g2 = te >= lazy + PRE_REGISTERED["g2_lazy_margin"]
    g3 = te >= PRE_REGISTERED["g3_net_apr"]
    g4 = te2x > 0
    for name, ok, desc in (
        ("G1 OOS", g1, "net>0 in train AND test"),
        ("G2 lazy", g2, "test beats BTC+ETH control by +2pp"),
        ("G3 bar", g3, "test net APR >= 5% deployed"),
        ("G4 cost", g4, "test net > 0 at doubled costs"),
    ):
        print(f"{name}: {'PASS' if ok else 'FAIL'}  ({desc})")
    overall = g1 and g2 and g3 and g4
    print(f"\nVERDICT: {'PASS — run --replication window' if overall else 'FAIL — question closed per pre-registration'}")

    if overall and args.replication:
        rep = tuple(map(d, PRE_REGISTERED["replication"]))
        mid = rep[0] + (rep[1] - rep[0]) / 2
        sim_r = simulate(daily, rep[0], rep[1])
        r1 = sim_r.net_apr_deployed(rep[0], mid)
        r2 = sim_r.net_apr_deployed(mid + dt.timedelta(days=1), rep[1])
        r2x = sim_r.net_apr_deployed(rep[0], rep[1], cost_mult=2.0)
        rfull = sim_r.net_apr_deployed(rep[0], rep[1])
        rlazy = lazy_control(daily, rep[0], rep[1])
        rg = r1 > 0 and r2 > 0 and rfull >= rlazy + 0.02 and rfull >= 0.05 and r2x > 0
        print(f"\n=== Replication {rep[0]}..{rep[1]} ===")
        print(f"halves {r1:+.1%} / {r2:+.1%}, full {rfull:+.1%}, "
              f"2x costs {r2x:+.1%}, lazy {rlazy:+.1%}")
        print(f"REPLICATION: {'PASS' if rg else 'FAIL'}")


if __name__ == "__main__":
    main()
