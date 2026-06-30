import numpy as np
import polars as pl

from research.microstructure.ic import compute_ic, quantile_layering


def test_compute_ic_recovers_known_rank_correlation():
    # signal perfectly rank-correlated with forward return -> IC ~ 1.0
    n = 500
    sig = np.linspace(-1, 1, n)
    fwd = sig * 2.0  # monotone increasing -> Spearman 1.0
    df = pl.DataFrame({"qi": sig, "fwd_1s": fwd, "fwd_5s": -fwd})
    ic = compute_ic(df, signal_col="qi", horizon_cols=["fwd_1s", "fwd_5s"])
    assert abs(ic["fwd_1s"] - 1.0) < 1e-6
    assert abs(ic["fwd_5s"] + 1.0) < 1e-6  # perfectly anti-correlated


def test_quantile_layering_is_monotone_for_real_signal():
    n = 1000
    rng = np.random.default_rng(0)
    sig = rng.normal(size=n)
    fwd = sig * 0.5 + rng.normal(scale=0.1, size=n)  # signal predicts fwd
    df = pl.DataFrame({"qi": sig, "fwd_1s": fwd})
    buckets = quantile_layering(df, signal_col="qi", horizon_col="fwd_1s", n_buckets=5)
    means = buckets["mean_fwd"].to_list()
    assert means == sorted(means)  # monotone increasing


def test_quantile_layering_monotone_at_double_digit_buckets():
    # regression: bucket labels must sort numerically, not lexically, at n >= 10
    # (plain str labels put "10" before "2" -> false non-monotone)
    n = 5000
    rng = np.random.default_rng(1)
    sig = rng.normal(size=n)
    fwd = sig * 0.5 + rng.normal(scale=0.1, size=n)
    df = pl.DataFrame({"qi": sig, "fwd_1s": fwd})
    buckets = quantile_layering(df, signal_col="qi", horizon_col="fwd_1s", n_buckets=15)
    means = buckets["mean_fwd"].to_list()
    assert means == sorted(means)


def test_compute_ic_drops_nan_not_just_null():
    from research.microstructure.ic import compute_ic
    # one NaN in the signal must be excluded, not poison the whole horizon
    sig = [float("nan")] + list(np.linspace(-1, 1, 100))
    fwd = [0.0] + list(np.linspace(-1, 1, 100) * 2.0)
    df = pl.DataFrame({"qi": sig, "fwd_1s": fwd})
    ic = compute_ic(df, signal_col="qi", horizon_cols=["fwd_1s"])
    assert abs(ic["fwd_1s"] - 1.0) < 1e-6  # ~1.0, not nan
