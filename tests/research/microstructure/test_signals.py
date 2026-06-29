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
