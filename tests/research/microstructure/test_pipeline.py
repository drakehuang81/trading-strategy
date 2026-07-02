import datetime as dt

import polars as pl

from research.microstructure.pipeline import recon_multi


def test_recon_multi_attaches_each_signal_and_computes_ic():
    # grid with mid + one signal series ("qi") already aligned
    ts = [dt.datetime(2026, 1, 1, 0, 0, s) for s in range(60)]
    grid = pl.DataFrame({"ts": ts, "mid": [100.0 + (s % 2) for s in range(60)]})
    qi = pl.DataFrame({"ts": ts, "qi": [0.5 if s % 2 == 0 else -0.5 for s in range(60)]})
    md, ic = recon_multi(grid, {"qi": qi}, horizons_secs=[1, 5])
    assert set(ic.keys()) == {"qi"}
    assert "fwd_1s" in ic["qi"] and "fwd_5s" in ic["qi"]
    assert "signal" in md


def test_recon_multi_derives_column_name_not_dict_key():
    import datetime as dt
    ts = [dt.datetime(2026, 1, 1, 0, 0, s) for s in range(60)]
    grid = pl.DataFrame({"ts": ts, "mid": [100.0 + (s % 2) for s in range(60)]})
    # dict key "sig" differs from the actual column "depth_imbalance"
    sig = pl.DataFrame({"ts": ts, "depth_imbalance": [0.5 if s % 2 == 0 else -0.5 for s in range(60)]})
    md, ic = recon_multi(grid, {"sig": sig}, horizons_secs=[1])
    assert "fwd_1s" in ic["sig"]  # must not KeyError on the column name
