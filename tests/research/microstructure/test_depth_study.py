import datetime as dt
import polars as pl

from research.microstructure.depth_study import build_hourly, time_split


def test_build_hourly_aligns_depth_to_kline_forward_return():
    # two hours of depth snapshots + 1h klines
    h0 = dt.datetime(2023, 6, 1, 0, 0, 0)
    h1 = dt.datetime(2023, 6, 1, 1, 0, 0)
    h2 = dt.datetime(2023, 6, 1, 2, 0, 0)
    # depth long-form: hour 0 bid-heavy (di>0), hour 1 ask-heavy (di<0)
    depth = pl.DataFrame({
        "ts": [h0, h0, h1, h1],
        "percentage": [-1.0, 1.0, -1.0, 1.0],
        "depth": [30.0, 10.0, 10.0, 30.0],
    })
    klines = pl.DataFrame({"hour": [h0, h1, h2], "close": [100.0, 110.0, 99.0]})
    ds = build_hourly(depth, klines)
    # hour 0: di = (30-10)/40 = 0.5; fwd_1h = 110/100-1 = 0.10
    row0 = ds.filter(pl.col("hour") == h0)
    assert abs(row0["di"][0] - 0.5) < 1e-9
    assert abs(row0["fwd_1h"][0] - 0.10) < 1e-9
    # last hour has no forward kline -> dropped
    assert ds.filter(pl.col("hour") == h2).height == 0


def test_time_split_70_30_by_date():
    hours = [dt.datetime(2023, 6, d, 0) for d in range(1, 11)]
    ds = pl.DataFrame({"hour": hours, "di": [0.0] * 10, "fwd_1h": [0.0] * 10})
    train, test = time_split(ds, train_frac=0.7)
    assert train.height == 7 and test.height == 3
    assert train["hour"].max() < test["hour"].min()  # strictly time-ordered
