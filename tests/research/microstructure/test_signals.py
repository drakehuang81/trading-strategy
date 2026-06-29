import datetime as dt

import polars as pl

from research.microstructure.signals import queue_imbalance


def _book(rows):
    return pl.DataFrame(
        {
            "ts": [dt.datetime(2026, 1, 1, 0, 0, i) for i in range(len(rows))],
            "bid_price": [r[0] for r in rows],
            "bid_qty": [r[1] for r in rows],
            "ask_price": [r[2] for r in rows],
            "ask_qty": [r[3] for r in rows],
        }
    )


def test_queue_imbalance_formula():
    # QI = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    book = _book([(100.0, 6.0, 100.5, 2.0)])  # (6-2)/(6+2) = 0.5
    qi = queue_imbalance(book)
    assert abs(qi["qi"][0] - 0.5) < 1e-12


def test_queue_imbalance_point_in_time():
    # QI at row i must depend ONLY on row i (no lookahead): computing on a
    # truncated prefix yields identical values for the rows present.
    book = _book([(100.0, 6.0, 100.5, 2.0), (101.0, 1.0, 101.5, 3.0)])
    full = queue_imbalance(book)
    prefix = queue_imbalance(book.head(1))
    assert abs(full["qi"][0] - prefix["qi"][0]) < 1e-12


def test_queue_imbalance_zero_depth_is_null():
    book = _book([(100.0, 0.0, 100.5, 0.0)])
    qi = queue_imbalance(book)
    assert qi["qi"][0] is None


def _depth(ts_rows):
    # ts_rows: list of (ts, percentage, depth)
    return pl.DataFrame({
        "ts": [r[0] for r in ts_rows],
        "percentage": [r[1] for r in ts_rows],
        "depth": [r[2] for r in ts_rows],
    })


def test_depth_imbalance_per_snapshot():
    from research.microstructure.signals import depth_imbalance
    t = dt.datetime(2026, 1, 1, 0, 0, 0)
    # bid side (neg pct) total depth 30, ask side (pos pct) total depth 10
    book = _depth([
        (t, -1.0, 20.0), (t, -0.2, 10.0), (t, 0.2, 4.0), (t, 1.0, 6.0),
    ])
    di = depth_imbalance(book)
    # (30 - 10) / (30 + 10) = 0.5
    assert di.height == 1
    assert abs(di["depth_imbalance"][0] - 0.5) < 1e-12


def test_depth_imbalance_zero_total_is_null():
    from research.microstructure.signals import depth_imbalance
    t = dt.datetime(2026, 1, 1, 0, 0, 0)
    book = _depth([(t, -1.0, 0.0), (t, 1.0, 0.0)])
    di = depth_imbalance(book)
    assert di["depth_imbalance"][0] is None


def test_book_slope_log_far_near_ratio():
    from research.microstructure.signals import book_slope
    t = dt.datetime(2026, 1, 1, 0, 0, 0)
    # near (|pct| <= 1) total depth = 10+10 = 20; far (|pct| >= 3) = 40+40 = 80
    book = _depth([
        (t, -1.0, 10.0), (t, 1.0, 10.0),
        (t, -3.0, 40.0), (t, 3.0, 40.0),
    ])
    bs = book_slope(book)
    # log(80 / 20) = log(4)
    import math
    assert abs(bs["book_slope"][0] - math.log(4.0)) < 1e-9


def test_ofi_cont_contributions():
    from research.microstructure.signals import ofi
    # rows: (bid_price, bid_qty, ask_price, ask_qty)
    book = _book([
        (100.0, 5.0, 100.5, 5.0),   # t0 baseline
        (100.0, 8.0, 100.5, 5.0),   # bid_qty up, bid_price same -> +3 bid contrib
        (101.0, 2.0, 101.5, 4.0),   # bid_price up -> +bid_qty(2); ask_price up -> +prev ask_qty(5)
    ])
    out = ofi(book, window=10)
    assert "ofi" in out.columns
    assert out["ofi"][0] is None   # first row: no previous update
    # e_1 = +3 (bid up by 3, ask unchanged contributes 0)
    assert abs(out["ofi"][1] - 3.0) < 1e-9


def test_taker_imbalance_rolling():
    from research.microstructure.signals import taker_imbalance
    t0 = dt.datetime(2026, 1, 1, 0, 0, 0)
    trades = pl.DataFrame({
        "ts": [t0 + dt.timedelta(seconds=s) for s in range(3)],
        "taker_buy_qty": [3.0, 0.0, 6.0],
        "taker_sell_qty": [1.0, 4.0, 0.0],
    })
    out = taker_imbalance(trades, window=3)
    # cumulative within window=3 at last row: buy=9, sell=5 -> (9-5)/14
    assert abs(out["taker_imbalance"][2] - (4.0 / 14.0)) < 1e-12
