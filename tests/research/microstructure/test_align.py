import datetime as dt

import polars as pl

from research.microstructure.align import forward_returns, to_mid_grid


def _book_with_mid():
    # bookTicker rows at irregular seconds; mid = (bid+ask)/2
    ts = [dt.datetime(2026, 1, 1, 0, 0, s) for s in (0, 2, 5)]
    return pl.DataFrame({
        "ts": ts,
        "bid_price": [100.0, 102.0, 104.0],
        "ask_price": [100.0, 102.0, 104.0],
        "bid_qty": [1.0, 1.0, 1.0],
        "ask_qty": [1.0, 1.0, 1.0],
    })


def test_to_mid_grid_backward_asof():
    grid = to_mid_grid(_book_with_mid(), every="1s")
    # at t=1s the latest book <= 1s is the t=0 row -> mid 100
    row1 = grid.filter(pl.col("ts") == dt.datetime(2026, 1, 1, 0, 0, 1))
    assert row1["mid"][0] == 100.0
    # at t=3s the latest book <= 3s is the t=2 row -> mid 102
    row3 = grid.filter(pl.col("ts") == dt.datetime(2026, 1, 1, 0, 0, 3))
    assert row3["mid"][0] == 102.0


def test_forward_returns_uses_future_only():
    grid = pl.DataFrame({
        "ts": [dt.datetime(2026, 1, 1, 0, 0, s) for s in range(4)],
        "mid": [100.0, 110.0, 121.0, 121.0],
    })
    out = forward_returns(grid, horizons_secs=[1])
    # r(t->t+1s) at t0 = 110/100 - 1 = 0.10
    assert abs(out["fwd_1s"][0] - 0.10) < 1e-12
    # last row has no future point -> null
    assert out["fwd_1s"][3] is None
