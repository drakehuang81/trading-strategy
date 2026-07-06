"""Unit tests for qi maker Phase A pure logic (synthetic day fixtures)."""
import datetime as dt

import polars as pl
import pytest

from scripts.qimaker.phase_a import (
    PRE_REGISTERED,
    per_second_grid,
    qi_of,
    sample_days,
    simulate_day,
)

DAY = dt.date(2023, 6, 5)
DAY_MS = int(dt.datetime(2023, 6, 5, tzinfo=dt.timezone.utc).timestamp() * 1000)


def quotes_df(rows: list[tuple[int, float, float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_ms": [DAY_MS + r[0] for r in rows],
            "bid": [r[1] for r in rows],
            "bid_qty": [r[2] for r in rows],
            "ask": [r[3] for r in rows],
            "ask_qty": [r[4] for r in rows],
        }
    )


def test_sample_days_registered_rule():
    days = sample_days()
    assert len(days) == 20
    assert days[0] == dt.date(2023, 6, 5)
    assert days[-1] == dt.date(2024, 3, 20)


def test_per_second_grid_last_tick_and_ffill():
    q = quotes_df([
        (0, 100.0, 5.0, 100.1, 5.0),
        (500, 100.2, 9.0, 100.3, 1.0),   # same second: last wins
        (3500, 100.4, 1.0, 100.5, 9.0),  # second 3
    ])
    grid = per_second_grid(q, DAY)
    assert grid["bid"][0] == 100.2
    assert grid["bid"][1] == 100.2       # forward-filled
    assert grid["bid"][3] == 100.4
    assert qi_of(grid, 0) == pytest.approx((9 - 1) / 10)
    assert qi_of(grid, 3) == pytest.approx((1 - 9) / 10)


def _bullish_grid(bid=100.0, ask=100.02, exit_bid=100.30):
    # qi=+0.8 at second 0; exit-time quotes rise so the long wins
    rows = [(0, bid, 9.0, ask, 1.0)]
    rows += [(11_000, exit_bid, 5.0, exit_bid + 0.02, 5.0)]  # from sec 11 on
    return per_second_grid(quotes_df(rows), DAY)


def test_simulate_fill_requires_strict_trade_through():
    grid = _bullish_grid()
    h = PRE_REGISTERED["primary_horizon_s"]
    # trade AT the bid does not fill; strictly below does.
    # (unfilled order cancels after 10s and the still-active signal re-posts
    # at s=10 before the s=11 quote flattens qi -> 2 signals, 0 fills)
    n, fills = simulate_day(grid, [DAY_MS + 2000], [100.0], DAY)
    assert (n, fills) == (2, [])
    n, fills = simulate_day(grid, [DAY_MS + 2000], [99.99], DAY)
    assert n == 1 and len(fills) == 1
    f = fills[0]
    assert f.side == 1 and f.entry == 100.0
    # exit at bid(fill_sec + h) = 100.30 -> move +30bps - 7 fees
    assert f.net_bps[h] == pytest.approx(30.0 - 7.0, abs=0.01)
    # adverse selection: mid at fill (sec 2 -> ffilled 100.01) vs entry
    assert f.adverse_bps == pytest.approx((100.0 - 100.01) / 100.0 * 1e4, abs=0.01)


def test_simulate_short_mirror():
    # qi=-0.8: rest ask at 100.02; fill needs a print strictly above
    rows = [(0, 100.0, 1.0, 100.02, 9.0), (11_000, 99.70, 5.0, 99.72, 5.0)]
    grid = per_second_grid(quotes_df(rows), DAY)
    n, fills = simulate_day(grid, [DAY_MS + 1500], [100.03], DAY)
    assert len(fills) == 1
    f = fills[0]
    assert f.side == -1
    # exit buys at ask 99.72: move = (100.02-99.72)/100.02 in our favor
    expect = (100.02 - 99.72) / 100.02 * 1e4 - 7.0
    assert f.net_bps[PRE_REGISTERED["primary_horizon_s"]] == pytest.approx(expect, abs=0.01)


def test_simulate_no_overlap_busy_window():
    # persistent signal every second, but one pending order blocks re-entry
    rows = [(0, 100.0, 9.0, 100.02, 1.0)]
    grid = per_second_grid(quotes_df(rows), DAY)
    n, fills = simulate_day(grid, [], [], DAY)   # no trades -> never fills
    # signals only every fill_window_s seconds while unfilled orders rest
    assert fills == []
    assert n == pytest.approx(86_400 / PRE_REGISTERED["fill_window_s"], rel=0.01)
