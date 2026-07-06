"""Unit tests for the cross-venue spread study's pure logic."""
import datetime as dt

import polars as pl
import pytest

from scripts.xvenue.study import (
    PRE_REGISTERED,
    lazy_control,
    simulate,
    step0,
    with_trail,
)

D = dt.date


def make_daily(rows: list[tuple[str, dt.date, float | None, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "key": [r[0] for r in rows],
            "day": [r[1] for r in rows],
            "spread_day": pl.Series([r[2] for r in rows], dtype=pl.Float64),
            "eligible": [r[3] for r in rows],
        }
    )


def days_of(key: str, start: dt.date, vals: list[float | None], eligible: bool = True):
    return [(key, start + dt.timedelta(days=i), v, eligible) for i, v in enumerate(vals)]


def test_with_trail_sign_preserved_and_gap_poisons():
    rows = days_of("X", D(2024, 1, 1), [-0.001, -0.001, -0.001, None, -0.001])
    out = with_trail(make_daily(rows))
    trail = out["trail_apr"].to_list()
    assert trail[3] == pytest.approx(-0.003 * 365 / 3)   # negative spread kept
    assert trail[4] is None                              # gap poisons


def test_simulate_shorts_rich_venue_and_collects_abs_spread():
    # Binance persistently richer (+20bps/day): the registered trade is
    # short-Binance/long-Bybit, whose daily PnL = funding received on the
    # short rich leg - funding paid on the long cheap leg = +spread_day.
    # A persistent spread must therefore book POSITIVE gross while held.
    rows = days_of("X", D(2024, 1, 1), [0.002] * 10)
    res = simulate(make_daily(rows), D(2024, 1, 1), D(2024, 1, 10))
    w = 1.0 / PRE_REGISTERED["slots"]
    assert res.entries == 1
    # day4 enters (trail from days1-3), collects the positive spread
    assert res.gross[3] == pytest.approx(w * 0.002)
    assert res.net[3] == pytest.approx(w * 0.002 - w * PRE_REGISTERED["half_rt_cost"])
    # persistent-sign spread keeps collecting, never bleeds
    assert all(g >= 0 for g in res.gross)


def test_simulate_exits_on_sign_flip_with_cost():
    rows = days_of("X", D(2024, 1, 1), [0.002] * 5 + [-0.002] * 6)
    res = simulate(make_daily(rows), D(2024, 1, 1), D(2024, 1, 11))
    # enters day4 short-rich-venue; after flip, trail decays then crosses
    # sign -> exit fires (and possibly re-entry the other way later)
    assert res.exits >= 1
    assert res.entries >= 1


def test_step0_uses_magnitude_ranking_and_direction():
    start = D(2024, 1, 1)
    rows = days_of("POS", start, [0.004] * 6)          # rich positive spread
    rows += days_of("NEG", start, [-0.003] * 6)        # rich negative spread
    rows += days_of("FLAT", start, [0.0001] * 6)
    c = step0(make_daily(rows), D(2024, 1, 4), D(2024, 1, 6))
    # top-5 by |trail| includes all 3; realized = |spread| for POS and NEG
    per_day = (0.004 + 0.003 + 0.0001) / 3
    assert c == pytest.approx(per_day * 365 / PRE_REGISTERED["deploy_factor"])


def test_lazy_control_flips_with_trail_and_charges_costs():
    rows = days_of("BTC", D(2024, 1, 1), [0.001] * 8)
    rows += days_of("ETH", D(2024, 1, 1), [0.001] * 8)
    rows += days_of("ALT", D(2024, 1, 1), [0.9] * 8)   # ignored
    apr = lazy_control(make_daily(rows), D(2024, 1, 1), D(2024, 1, 8))
    # days 4-8 have signal for both keys: collect 0.001 each at weight 0.5,
    # one 0.5*cost sign-entry per key
    total = 2 * (5 * 0.5 * 0.001 - 0.5 * PRE_REGISTERED["half_rt_cost"])
    expect = total / 8 * 365 / PRE_REGISTERED["deploy_factor"]
    assert apr == pytest.approx(expect)
