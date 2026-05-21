"""Sharpe + Deflated Sharpe — Plan 5B-3 Task 1."""
from __future__ import annotations

import math

import numpy as np
import pytest

from observability.sharpe import sharpe_ratio, deflated_sharpe_ratio


def test_sharpe_ratio_positive_drift_returns_positive():
    rng = np.random.default_rng(0)
    # Daily returns with mean 0.001, std 0.01 → Sharpe ≈ 0.1 daily, annualised ~1.6
    returns = rng.normal(0.001, 0.01, size=2000)
    sr = sharpe_ratio(returns, periods_per_year=252)
    assert sr > 1.0


def test_sharpe_ratio_zero_returns_yields_nan():
    returns = np.zeros(100)
    sr = sharpe_ratio(returns, periods_per_year=252)
    assert math.isnan(sr)


def test_sharpe_ratio_handles_short_series():
    """N<2 returns NaN, doesn't crash."""
    assert math.isnan(sharpe_ratio(np.array([0.01]), periods_per_year=252))
    assert math.isnan(sharpe_ratio(np.array([]), periods_per_year=252))


def test_deflated_sharpe_below_observed_when_n_trials_high():
    """More trials = larger correction = lower DSR."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=2000)
    dsr_n2 = deflated_sharpe_ratio(returns, n_trials=2, periods_per_year=252)
    dsr_n100 = deflated_sharpe_ratio(returns, n_trials=100, periods_per_year=252)
    assert dsr_n2 > dsr_n100


def test_deflated_sharpe_n_trials_one_is_bounded_probability():
    """With n_trials=1, SR0=0; for zero-drift returns the observed
    SR is ~0, so DSR (probability) is centred near 0.5."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=500)
    dsr = deflated_sharpe_ratio(returns, n_trials=1, periods_per_year=252)
    # Probability bounded in [0, 1].
    assert 0.0 <= dsr <= 1.0
    # For zero-drift returns, expected dsr is near 0.5 ± noise.
    assert 0.2 <= dsr <= 0.8
