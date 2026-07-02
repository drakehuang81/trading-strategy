import datetime as dt
import polars as pl

from scripts.recon.depth_validation import summarize


def test_summarize_reports_train_test_and_control():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 2000
    di = rng.normal(size=n)
    ds = pl.DataFrame({
        "hour": [dt.datetime(2023, 6, 1) + dt.timedelta(hours=i) for i in range(n)],
        "di": di,
        "past_1h": rng.normal(size=n),
        "fwd_1h": di * 0.001 + rng.normal(scale=0.002, size=n),
    })
    rep = summarize(ds)
    assert "ic_all" in rep and "ic_train" in rep and "ic_test" in rep
    assert "ic_momentum" in rep and "gross_bps" in rep and "nw_tstat" in rep
    assert isinstance(rep["verdict"], str)
