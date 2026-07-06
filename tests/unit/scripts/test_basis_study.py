"""Unit tests for the basis mean-reversion study's pure logic."""
import datetime as dt

import polars as pl
import pytest

from scripts.basis.study import (
    PRE_REGISTERED,
    Episode,
    build_basis,
    count_pos,
    find_episodes,
    portfolio_apr,
    read_kline_csv,
    window_months,
)

H = 3_600_000
T0 = 1_700_000_000_000  # arbitrary hour-ish anchor


def bases_df(vals: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"ts_ms": [T0 + i * H for i in range(len(vals))],
         "basis": pl.Series(vals, dtype=pl.Float64)}
    )


def test_read_kline_csv_header_and_headerless():
    headered = (
        b"open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        b"taker_buy_volume,taker_buy_quote_volume,ignore\n"
        b"1704067200000,1,2,0.5,1.5,10,1704070799999,15,3,5,7,0\n"
    )
    df = read_kline_csv(headered)
    assert df.columns == ["ts_ms", "close"]
    assert df.row(0) == (1704067200000, 1.5)

    headerless = b"1704067200000,1,2,0.5,2.5,10,1704070799999,15,3,5,7,0\n"
    assert read_kline_csv(headerless).row(0) == (1704067200000, 2.5)


def test_build_basis_inner_join_alignment():
    spot = pl.DataFrame({"ts_ms": [T0, T0 + H, T0 + 2 * H], "close": [100.0, 100.0, 100.0]})
    perp = pl.DataFrame({"ts_ms": [T0, T0 + 2 * H], "close": [100.1, 99.9]})  # gap at T0+H
    out = build_basis(spot, perp)
    assert out["ts_ms"].to_list() == [T0, T0 + 2 * H]
    assert out["basis"].to_list() == [pytest.approx(0.001), pytest.approx(-0.001)]


def test_find_episodes_trigger_converge_and_capture():
    # 80bps premium -> converges to 5bps: capture = 75bps
    eps = find_episodes("X", bases_df([0.0, 0.0080, 0.0040, 0.0005, 0.0]))
    assert len(eps) == 1
    e = eps[0]
    assert e.side == 1
    assert e.capture == pytest.approx(0.0075)
    assert e.t_exit - e.t_entry == 2 * H


def test_find_episodes_timeout_and_negative_side_and_nonoverlap():
    # stays wide 50h -> timeout exit at whatever |basis| is then (capture ~0)
    vals = [-0.0090] * 60
    eps = find_episodes("X", bases_df(vals))
    assert len(eps) == 1
    assert eps[0].side == -1
    assert eps[0].capture == pytest.approx(0.0)
    assert eps[0].t_exit - eps[0].t_entry >= PRE_REGISTERED["timeout_h"] * H
    # non-overlap: second trigger only after first exit
    assert len(find_episodes("X", bases_df([0.007, 0.007, 0.0005, 0.007, 0.0005]))) == 2


def test_portfolio_apr_costs_and_side_filter():
    lo, hi = dt.date(2024, 1, 1), dt.date(2024, 12, 30)  # 365 days
    t = int(dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    eps = [
        Episode("A", 1, t, t + H, 0.0100),   # +100bps captured
        Episode("B", -1, t, t + H, 0.0500),  # negative side -> excluded
    ]
    f = PRE_REGISTERED["deploy_factor"]
    gross = portfolio_apr(eps, lo, hi, n_symbols=2, cost_mult=0.0)
    net = portfolio_apr(eps, lo, hi, n_symbols=2, cost_mult=1.0)
    assert gross == pytest.approx(0.0100 / 2 / 1.0 / f)
    assert net == pytest.approx((0.0100 - 0.0040) / 2 / 1.0 / f)
    assert count_pos(eps, lo, hi) == 1


def test_window_months_span():
    months = window_months()
    assert months[0] == "2022-07" and months[-1] == "2026-06"
    assert len(months) == 48
