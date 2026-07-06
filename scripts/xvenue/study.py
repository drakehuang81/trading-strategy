"""Cross-venue funding spread study — pre-registered, single shot.

Pre-registration: docs/superpowers/plans/2026-07-06-xvenue-funding-spread-preregistration.md
(PRE_REGISTERED below is its machine-readable mirror, committed before any
Bybit data was downloaded).

spread_day(key, D) = Binance day funding sum - Bybit day funding sum.
Position direction = -sign(trail3): short the richer-funding venue, long
the other; collects |spread| while it persists. Both signs tradeable
(perp-perp: no spot, no borrow).

Implementation decisions fixed pre-run (documented before first run):
- trail3 needs 3 consecutive joint days, else no signal (nulls poison)
- while held: exit on |trail| < exit band, trail sign flip, missing day,
  or ineligibility — each exit/entry costs half-RT
- lazy control recomputes its direction daily from trail3 sign; sign
  flips cost half-RT (it is "no selection", not "no execution")

Run (after scripts.xvenue.bybit):
    PYTHONPATH=src venv/bin/python -m scripts.xvenue.study --data data/xvenue
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

PRE_REGISTERED = {
    "venues": "Binance <-> Bybit linear USDT perps",
    "grid": "daily UTC; spread_day = binance_day_sum - bybit_day_sum",
    "seasoning_days": 30,       # joint history >= 30d before eligible
    "trail_days": 3,
    "entry_apr": 0.10,          # |trail3| annualized must exceed
    "exit_apr": 0.05,           # hysteresis band
    "slots": 5,
    "half_rt_cost": 0.0015,     # 15bps per entry AND per exit (RT 30bps)
    "deploy_factor": 1.0,       # 0.5 margin per perp leg incl. buffer
    "train": ("2022-07-01", "2024-06-30"),
    "test": ("2024-07-01", "2026-06-30"),
    "replication": ("2020-07-01", "2022-06-30"),
    "step0_kill_gross_apr": 0.10,
    "g2_lazy_margin": 0.02,     # beat BTC+ETH always-on spread by +2pp
    "g3_net_apr": 0.05,
    "g4_cost_multiplier": 2.0,
}


# ---------------------------------------------------------------- data layer

def _daily_side(path: Path, key: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if df.is_empty():
        return pl.DataFrame(schema={"key": pl.Utf8, "day": pl.Date, "fund": pl.Float64})
    return (
        df.with_columns(
            pl.from_epoch(pl.col("ts_ms"), time_unit="ms").dt.date().alias("day")
        )
        .group_by("day")
        .agg(pl.col("rate").sum().alias("fund"))
        .with_columns(pl.lit(key).alias("key"))
        .select("key", "day", "fund")
    )


def load_spread_daily(data: Path, binance_dir: Path) -> pl.DataFrame:
    """(key, day, spread_day, eligible) with contiguous per-key calendars.

    Join is inner per day (both venues must have data); the calendar is then
    made contiguous between the joint first/last day so gap days exist as
    nulls and poison the trailing window (same discipline as carry).
    """
    mapping = json.loads((data / "mapping.json").read_text())
    frames = []
    for key, names in mapping.items():
        byb_path = data / "bybit" / f"{names['bybit']}.parquet"
        bnb_path = binance_dir / f"{names['binance']}.parquet"
        if not (byb_path.exists() and bnb_path.exists()):
            continue
        joint = (
            _daily_side(bnb_path, key)
            .rename({"fund": "bnb"})
            .join(_daily_side(byb_path, key).rename({"fund": "byb"}),
                  on=["key", "day"], how="inner")
            .with_columns((pl.col("bnb") - pl.col("byb")).alias("spread_day"))
            .select("key", "day", "spread_day")
            .sort("day")
        )
        if joint.is_empty():
            continue
        frames.append(joint.upsample(time_column="day", every="1d")
                      .with_columns(pl.col("key").fill_null(key)))
    out = pl.concat(frames)
    first = (
        out.filter(pl.col("spread_day").is_not_null())
        .group_by("key").agg(pl.col("day").min().alias("first_day"))
    )
    return (
        out.join(first, on="key")
        .with_columns(
            (pl.col("day") >= pl.col("first_day")
             + pl.duration(days=PRE_REGISTERED["seasoning_days"])).alias("eligible")
        )
        .select("key", "day", "spread_day", "eligible")
        .sort("key", "day")
    )


def with_trail(daily: pl.DataFrame) -> pl.DataFrame:
    n = PRE_REGISTERED["trail_days"]
    return daily.with_columns(
        (pl.col("spread_day").rolling_sum(n).shift(1).over("key") * (365.0 / n))
        .alias("trail_apr")
    )


# ------------------------------------------------------------------ sim core

@dataclass
class SimResult:
    days: list[dt.date] = field(default_factory=list)
    net: list[float] = field(default_factory=list)      # on notional
    gross: list[float] = field(default_factory=list)
    n_held: list[int] = field(default_factory=list)
    entries: int = 0
    exits: int = 0

    def net_apr_deployed(self, start: dt.date, end: dt.date, cost_mult: float = 1.0) -> float:
        rows = [(g, n) for d_, g, n in zip(self.days, self.gross, self.net)
                if start <= d_ <= end]
        if not rows:
            return float("nan")
        daily = [g + cost_mult * (n - g) for g, n in rows]
        return sum(daily) / len(daily) * 365.0 / PRE_REGISTERED["deploy_factor"]


def simulate(daily: pl.DataFrame, start: dt.date, end: dt.date) -> SimResult:
    """Registered top-K |trail| rotation, direction = -sign(trail)."""
    k = PRE_REGISTERED["slots"]
    w = 1.0 / k
    cost = PRE_REGISTERED["half_rt_cost"]
    entry_apr = PRE_REGISTERED["entry_apr"]
    exit_apr = PRE_REGISTERED["exit_apr"]

    window = with_trail(daily).filter((pl.col("day") >= start) & (pl.col("day") <= end))
    by_day: dict[dt.date, dict[str, tuple[float | None, float | None, bool]]] = {}
    for key, day, spread, elig, trail in window.select(
        "key", "day", "spread_day", "eligible", "trail_apr"
    ).iter_rows():
        by_day.setdefault(day, {})[key] = (spread, trail, elig)

    held: dict[str, int] = {}          # key -> direction (+1 long-spread)
    res = SimResult()
    for day in sorted(by_day):
        rows = by_day[day]
        day_cost = 0.0
        for key in sorted(held):
            _, trail, elig = rows.get(key, (None, None, False))
            # sign convention: direction=+1 means short-Binance/long-Bybit,
            # whose daily PnL is +spread_day (funding received on the short
            # rich leg minus funding paid on the long cheap leg). So the
            # profitable direction while the sign persists is +sign(trail).
            want = 0 if trail is None else (1 if trail > 0 else -1)
            if (trail is None or abs(trail) < exit_apr or not elig
                    or want != held[key]):
                del held[key]
                day_cost += w * cost
                res.exits += 1
        candidates = sorted(
            ((abs(trail), key, 1 if trail > 0 else -1)
             for key, (spread, trail, elig) in rows.items()
             if key not in held and elig and trail is not None
             and abs(trail) > entry_apr),
            reverse=True,
        )
        for _, key, direction in candidates[: max(0, k - len(held))]:
            held[key] = direction
            day_cost += w * cost
            res.entries += 1
        day_gross = sum(
            w * d_ * (rows.get(key, (None, None, False))[0] or 0.0)
            for key, d_ in held.items()
        )
        res.days.append(day)
        res.gross.append(day_gross)
        res.net.append(day_gross - day_cost)
        res.n_held.append(len(held))
    return res


def lazy_control(daily: pl.DataFrame, start: dt.date, end: dt.date) -> float:
    """BTC+ETH spread always-on, direction from trail3 sign, no selection.

    Sign flips (and the initial entries) cost half-RT per leg-pair — lazy
    means no research, not free execution."""
    cost = PRE_REGISTERED["half_rt_cost"]
    window = (
        with_trail(daily)
        .filter(pl.col("key").is_in(["BTC", "ETH"])
                & (pl.col("day") >= start) & (pl.col("day") <= end))
        .sort(["key", "day"])
    )
    total, n_days = 0.0, 0
    for key in ("BTC", "ETH"):
        sub = window.filter(pl.col("key") == key)
        prev_dir = 0
        for spread, trail in sub.select("spread_day", "trail_apr").iter_rows():
            n_days += 1
            if trail is None:
                continue
            direction = 1 if trail > 0 else -1   # +1 = short rich venue
            if direction != prev_dir:
                total -= 0.5 * cost          # per-leg-pair weight is 1/2
                prev_dir = direction
            total += 0.5 * direction * (spread or 0.0)
    if n_days == 0:
        return float("nan")
    return total / (n_days / 2) * 365.0 / PRE_REGISTERED["deploy_factor"]


def step0(daily: pl.DataFrame, start: dt.date, end: dt.date) -> float:
    """Gross APR of trailing-|spread| top-5, direction applied, no costs."""
    k = PRE_REGISTERED["slots"]
    window = with_trail(daily).filter(
        (pl.col("day") >= start) & (pl.col("day") <= end) & pl.col("eligible")
    )
    sel = (
        window.filter(pl.col("trail_apr").is_not_null())
        .with_columns(
            pl.col("trail_apr").abs().alias("mag"),
            (pl.when(pl.col("trail_apr") > 0).then(1).otherwise(-1)
             * pl.col("spread_day").fill_null(0.0)).alias("realized"),
        )
        .sort(["day", "mag"], descending=[False, True])
        .group_by("day", maintain_order=True)
        .agg(pl.col("realized").head(k).mean().alias("g"))
    )
    n_days = (end - start).days + 1
    if sel.is_empty():
        return float("nan")
    return float(sel["g"].sum()) / n_days * 365.0 / PRE_REGISTERED["deploy_factor"]


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/xvenue")
    ap.add_argument("--binance", default="data/carry/funding")
    args = ap.parse_args()

    daily = load_spread_daily(Path(args.data), Path(args.binance))
    print(f"spread table: {len(daily)} rows, {daily['key'].n_unique()} keys")

    train = tuple(map(d, PRE_REGISTERED["train"]))
    test = tuple(map(d, PRE_REGISTERED["test"]))
    kill = PRE_REGISTERED["step0_kill_gross_apr"]

    print("\n=== Step 0: gross spread-capture ceiling ===")
    survived = True
    for name, (lo, hi) in (("train", train), ("test", test)):
        c = step0(daily, lo, hi)
        survived &= c >= kill
        print(f"{name}: trail-top5 gross APR (deployed) = {c:+.1%} "
              f"-> {'PASS' if c >= kill else 'KILL'}")
    if not survived:
        print("\nSTEP 0 KILL -> cross-venue spread CLOSED; "
              "the free-data research chapter ends here.")
        return

    print("\n=== Phase 1 ===")
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
        print("\nPhase 1 FAIL -> CLOSED; the free-data research chapter ends here.")
        return

    rep = tuple(map(d, PRE_REGISTERED["replication"]))
    mid = rep[0] + (rep[1] - rep[0]) / 2
    sim_r = simulate(daily, rep[0], rep[1])
    r1 = sim_r.net_apr_deployed(rep[0], mid)
    r2 = sim_r.net_apr_deployed(mid + dt.timedelta(days=1), rep[1])
    rfull = sim_r.net_apr_deployed(rep[0], rep[1])
    r2x = sim_r.net_apr_deployed(rep[0], rep[1], cost_mult=2.0)
    rlazy = lazy_control(daily, rep[0], rep[1])
    rg = (r1 > 0 and r2 > 0 and rfull >= rlazy + PRE_REGISTERED["g2_lazy_margin"]
          and rfull >= PRE_REGISTERED["g3_net_apr"] and r2x > 0)
    print(f"\n=== Replication === halves {r1:+.1%}/{r2:+.1%} full {rfull:+.1%} "
          f"2x {r2x:+.1%} lazy {rlazy:+.1%}")
    print(f"FINAL: {'PASS & REPLICATED' if rg else 'FAIL -> CLOSED; the free-data research chapter ends here.'}")


if __name__ == "__main__":
    main()
