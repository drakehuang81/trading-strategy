import datetime as dt

import polars as pl

from scripts.recon.run_recon import recon_from_book


def test_recon_from_book_end_to_end():
    # 120s of synthetic book where qi predicts next-second up-move
    ts = [dt.datetime(2026, 1, 1, 0, 0, 0) + dt.timedelta(seconds=s) for s in range(120)]
    bid_qty = [5.0 if s % 2 == 0 else 1.0 for s in range(120)]
    ask_qty = [1.0 if s % 2 == 0 else 5.0 for s in range(120)]
    price = [100.0 + (1 if s % 2 == 0 else 0) for s in range(120)]
    book = pl.DataFrame({
        "ts": ts,
        "bid_price": price, "ask_price": price,
        "bid_qty": bid_qty, "ask_qty": ask_qty,
    })
    md, ic = recon_from_book(book, horizons_secs=[1, 5])
    assert "fwd_1s" in ic
    assert isinstance(md, str) and "signal" in md
