"""Unit tests for the carry study's pure logic (synthetic frames only)."""
import datetime as dt

import polars as pl
import pytest

from scripts.carry.study import (
    PRE_REGISTERED,
    SimResult,
    lazy_control,
    load_daily,
    simulate,
    step0_gross_ceiling,
    with_trail_apr,
)

D = dt.date


def make_daily(rows: list[tuple[str, dt.date, float | None, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "day": [r[1] for r in rows],
            "fund_day": pl.Series([r[2] for r in rows], dtype=pl.Float64),
            "eligible": [r[3] for r in rows],
        }
    )


def days_of(sym: str, start: dt.date, funds: list[float | None], eligible: bool = True):
    return [
        (sym, start + dt.timedelta(days=i), f, eligible)
        for i, f in enumerate(funds)
    ]


def test_load_daily_gap_null_and_seasoning(tmp_path):
    def ts(day: int, h: int) -> int:
        return int(dt.datetime(2024, 1, day, h, tzinfo=dt.timezone.utc).timestamp() * 1000)

    # day1 has two rates (sum), day2 missing entirely (gap), day3 present
    pl.DataFrame(
        {"ts_ms": [ts(1, 0), ts(1, 8), ts(3, 0)], "interval_h": [8, 8, 8],
         "rate": [0.001, 0.002, 0.004]}
    ).write_parquet(tmp_path / "AAAUSDT.parquet")
    daily = load_daily(tmp_path)
    assert daily["day"].to_list() == [D(2024, 1, 1), D(2024, 1, 2), D(2024, 1, 3)]
    assert daily["fund_day"].to_list() == [pytest.approx(0.003), None, pytest.approx(0.004)]
    # 3 days since first record < 30d seasoning -> never eligible here
    assert daily["eligible"].to_list() == [False, False, False]


def test_trail_apr_needs_three_complete_days():
    rows = days_of("X", D(2024, 1, 1), [0.001, 0.001, 0.001, None, 0.001, 0.001])
    out = with_trail_apr(make_daily(rows))
    trail = out["trail_apr"].to_list()
    assert trail[0] is None and trail[1] is None and trail[2] is None  # warmup
    assert trail[3] == pytest.approx(0.003 * 365 / 3)                  # days 1-3
    assert trail[4] is None                                            # gap poisons
    assert trail[5] is None


def test_simulate_enters_collects_and_charges_costs():
    # constant 0.001/day (36.5% APR): warmup 3d, enter day4, hold to end
    rows = days_of("X", D(2024, 1, 1), [0.001] * 10)
    res = simulate(make_daily(rows), D(2024, 1, 1), D(2024, 1, 10))
    w = 1.0 / PRE_REGISTERED["slots"]
    assert (res.entries, res.exits) == (1, 0)
    assert res.net[:3] == [0.0, 0.0, 0.0]
    assert res.net[3] == pytest.approx(w * 0.001 - w * PRE_REGISTERED["half_rt_cost"])
    assert res.net[4] == pytest.approx(w * 0.001)
    assert res.n_held[-1] == 1


def test_simulate_exits_when_trail_decays_below_band():
    # 0.001 for 5 days then 0: trail APR falls below 5% at day9 -> exit
    rows = days_of("X", D(2024, 1, 1), [0.001] * 5 + [0.0] * 7)
    res = simulate(make_daily(rows), D(2024, 1, 1), D(2024, 1, 12))
    assert res.entries == 1
    assert res.exits == 1
    w = 1.0 / PRE_REGISTERED["slots"]
    day9 = res.days.index(D(2024, 1, 9))
    assert res.net[day9] == pytest.approx(-w * PRE_REGISTERED["half_rt_cost"])
    assert res.n_held[day9] == 0


def test_simulate_exit_on_data_end_via_filler_calendar():
    # X delists after day6; filler keeps the calendar alive -> forced exit
    rows = days_of("X", D(2024, 1, 1), [0.002] * 6)
    rows += days_of("FILLER", D(2024, 1, 1), [0.0] * 10)
    res = simulate(make_daily(rows), D(2024, 1, 1), D(2024, 1, 10))
    assert res.entries == 1          # X only; FILLER never clears 10% APR
    assert res.exits == 1            # forced when X's rows vanish
    assert max(res.n_held) == 1 and res.n_held[-1] == 0


def test_simulate_respects_slots_and_eligibility():
    start = D(2024, 1, 1)
    rows = []
    for i in range(7):  # 7 rich symbols, only 5 slots
        rows += days_of(f"S{i}USDT", start, [0.002] * 8)
    rows += days_of("NEWUSDT", start, [0.01] * 8, eligible=False)  # unseasoned
    res = simulate(make_daily(rows), start, D(2024, 1, 8))
    assert max(res.n_held) == PRE_REGISTERED["slots"]
    assert res.entries == PRE_REGISTERED["slots"]


def test_net_apr_deployed_cost_multiplier_scales_only_costs():
    res = SimResult(
        days=[D(2024, 1, 1), D(2024, 1, 2)],
        gross=[0.001, 0.001],
        net=[0.0006, 0.001],      # day1 carries 0.0004 of costs
        n_held=[1, 1],
    )
    base = res.net_apr_deployed(D(2024, 1, 1), D(2024, 1, 2))
    twice = res.net_apr_deployed(D(2024, 1, 1), D(2024, 1, 2), cost_mult=2.0)
    f = PRE_REGISTERED["deploy_factor"]
    assert base == pytest.approx((0.0016 / 2) * 365 / f)
    # day1 @2x: 0.001 + 2*(0.0006-0.001) = 0.0002; day2 unchanged
    assert twice == pytest.approx(((0.0002 + 0.001) / 2) * 365 / f)


def test_step0_gross_ceiling_trailing_vs_oracle():
    start = D(2024, 1, 1)
    # HI is rich but its final day spikes; trailing selection still collects
    # realized next-day funding, oracle picks by same-day (perfect foresight)
    rows = days_of("HI", start, [0.002] * 5 + [0.010])
    rows += days_of("LO", start, [0.0001] * 6)
    m = step0_gross_ceiling(make_daily(rows), D(2024, 1, 4), D(2024, 1, 6))
    n_days = 3
    # trailing top-5 (k=5, only 2 names): mean of both names' fund each day
    expect_trail = (0.00105 + 0.00105 + 0.00505) / n_days * 365
    assert m["trail_top5_gross_apr"] == pytest.approx(expect_trail)
    assert m["oracle_top1_gross_apr"] == pytest.approx((0.002 + 0.002 + 0.010) / 3 * 365)


def test_lazy_control_btc_eth_mean_minus_one_entry():
    rows = days_of("BTCUSDT", D(2024, 1, 1), [0.0002] * 4)
    rows += days_of("ETHUSDT", D(2024, 1, 1), [0.0004] * 4)
    rows += days_of("ALTUSDT", D(2024, 1, 1), [0.9] * 4)  # ignored by control
    apr = lazy_control(make_daily(rows), D(2024, 1, 1), D(2024, 1, 4))
    total = 0.0003 * 4 - PRE_REGISTERED["half_rt_cost"]
    assert apr == pytest.approx(total / 4 * 365 / PRE_REGISTERED["deploy_factor"])


def test_simulate_records_held_ever_for_audit():
    rows = days_of("X", D(2024, 1, 1), [0.001] * 10)
    res = simulate(make_daily(rows), D(2024, 1, 1), D(2024, 1, 10))
    assert res.held_ever == {"X"}
