import datetime as dt
import polars as pl

from scripts.recon.cross_validation import summarize_cross


def test_summarize_cross_dual_controls_and_verdict():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 2000
    di = rng.normal(size=n)
    ds = pl.DataFrame({
        "hour": [dt.datetime(2023, 6, 1) + dt.timedelta(hours=i) for i in range(n)],
        "di": di,
        "past_1h_lead": rng.normal(size=n),
        "past_1h": rng.normal(size=n),
        "fwd_1h": di * 0.001 + rng.normal(scale=0.002, size=n),
    })
    rep = summarize_cross(ds)
    for key in ("ic_all", "ic_train", "ic_test", "ic_momentum_lag",
                "ic_momentum_lead", "gross_bps", "net_std_taker_bps",
                "nw_tstat", "monotone", "verdict", "n_hours"):
        assert key in rep
    assert rep["verdict"] == "REAL-ALPHA candidate"  # synthetic true signal
