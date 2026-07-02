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


def test_decile_edge_bps_monotone_and_tail():
    import numpy as np
    rng = np.random.default_rng(0)
    di = rng.normal(size=4000)
    fwd = di * 0.001 + rng.normal(scale=0.002, size=4000)  # di predicts fwd
    ds = pl.DataFrame({"di": di, "fwd_1h": fwd})
    from research.microstructure.depth_study import decile_edge_bps
    res = decile_edge_bps(ds, n_buckets=10)
    assert res["means"] == sorted(res["means"])          # full-bucket monotone
    assert res["net_std_taker_bps"] == res["gross_bps"] - 8.0


def test_newey_west_tstat_runs():
    import numpy as np
    rng = np.random.default_rng(1)
    r = rng.normal(loc=0.5, scale=1.0, size=200)  # positive-mean series
    from research.microstructure.depth_study import newey_west_tstat
    t = newey_west_tstat(r, lags=5)
    assert t > 3.0  # clearly positive


def test_momentum_control_ic():
    # depth IC vs momentum(past return) IC — if equal, depth is trend proxy
    import numpy as np
    rng = np.random.default_rng(2)
    di = rng.normal(size=1000)
    past = rng.normal(size=1000)
    fwd = di * 0.001 + rng.normal(scale=0.002, size=1000)  # driven by di, not past
    ds = pl.DataFrame({"di": di, "past_1h": past, "fwd_1h": fwd})
    from research.microstructure.depth_study import ic
    assert abs(ic(ds, "di")) > abs(ic(ds, "past_1h")) + 0.1


def test_build_hourly_cross_lead_book_lag_return():
    from research.microstructure.depth_study import build_hourly_cross
    h0 = dt.datetime(2023, 6, 1, 0, 0, 0)
    h1 = dt.datetime(2023, 6, 1, 1, 0, 0)
    h2 = dt.datetime(2023, 6, 1, 2, 0, 0)
    depth_btc = pl.DataFrame({
        "ts": [h0, h0, h1, h1],
        "percentage": [-1.0, 1.0, -1.0, 1.0],
        "depth": [30.0, 10.0, 10.0, 30.0],
    })
    klines_btc = pl.DataFrame({"hour": [h0, h1, h2], "close": [200.0, 202.0, 201.0]})
    klines_eth = pl.DataFrame({"hour": [h0, h1, h2], "close": [100.0, 105.0, 99.0]})
    ds = build_hourly_cross(depth_btc, klines_btc, klines_eth)
    row0 = ds.filter(pl.col("hour") == h0)
    assert abs(row0["di"][0] - 0.5) < 1e-9            # di from the BTC book
    assert abs(row0["fwd_1h"][0] - 0.05) < 1e-9       # target = ETH 100->105
    row1 = ds.filter(pl.col("hour") == h1)
    assert abs(row1["past_1h"][0] - 0.05) < 1e-9      # ETH's own momentum
    assert abs(row1["past_1h_lead"][0] - 0.01) < 1e-9  # BTC momentum 200->202
