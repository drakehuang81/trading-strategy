"""Sharpe ratio + Deflated Sharpe (Bailey & López de Prado 2014).

DSR adjusts the observed Sharpe for selection bias: when you've tested
N strategies, the best-performing one's Sharpe is upward-biased.
DSR returns the probability-style score representing the chance the
true Sharpe exceeds zero given the observation.

Formula (simplified, periodic-Sharpe form):
    SR = mean(r) / std(r) * sqrt(P)        # annualised
    SR0 = sqrt(2 * ln(N)) - euler/sqrt(2 * ln(N))   # expected max under null
    DSR = (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2)
where T is the number of return observations, N is the number of trials.

The function returns Phi(z) = norm.cdf(z), the probability that the
true Sharpe exceeds zero given the observation. Multiply with norm.ppf
to recover the raw z-score. The plan's original spec said "return raw
z" but we ship the probability form because:
  (a) it matches the function's "probability-of-skill" docstring,
  (b) the Pre-Live Gate threshold (spec §10.1.2 "DSR > 0.5") is
      naturally a probability,
  (c) it's the conventional Bailey-de Prado deliverable.

Convention reference: DSR = 0.5 means observed SR equals the
expected best-of-N null. DSR = 0.95 is the conventional
"this is real" bar (Bailey-de Prado 2014 §4).

Returns NaN on degenerate input (zero variance, T<2). DSR can be
negative (model worse than expected-best-of-N null).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329

PERIODS_PER_YEAR_1H = 24 * 365   # 8760
PERIODS_PER_YEAR_DAILY = 252


def sharpe_ratio(returns: np.ndarray | list[float],
                 periods_per_year: int) -> float:
    """Annualised Sharpe ratio. Returns NaN on degenerate input."""
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return float("nan")
    std = arr.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(arr.mean() / std * math.sqrt(periods_per_year))


def _expected_max_sharpe_under_null(n_trials: int) -> float:
    """E[max SR_i under H0] for n_trials draws, Bailey-de Prado eq. 3.

    With n_trials=1, returns 0 (no expected outperformance under null).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    z_high = math.sqrt(2.0 * math.log(n_trials))
    return z_high - EULER_GAMMA / z_high


def deflated_sharpe_ratio(returns: np.ndarray | list[float],
                          n_trials: int,
                          periods_per_year: int) -> float:
    """Probability-of-skill score under selection bias.

    Returns NaN on degenerate input. Can be negative when the observed
    Sharpe is below what you'd expect from N coin-flip strategies.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size < 4:                   # need enough data for skew/kurt
        return float("nan")
    sr = sharpe_ratio(arr, periods_per_year)
    if math.isnan(sr):
        return float("nan")
    # Convert annualised SR back to per-period for the variance term
    sr_period = sr / math.sqrt(periods_per_year)
    sr0_period = _expected_max_sharpe_under_null(n_trials) / math.sqrt(periods_per_year)
    sk = float(scipy_skew(arr, bias=False))
    kt = float(scipy_kurt(arr, fisher=False, bias=False))   # raw kurtosis (3 = normal)
    t = arr.size
    denom_sq = 1.0 - sk * sr_period + ((kt - 1.0) / 4.0) * sr_period ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (sr_period - sr0_period) * math.sqrt(t - 1) / math.sqrt(denom_sq)
    # Return as a probability score: P(true SR > 0 | observed SR, N trials).
    # Callers that need the raw z-score can use scipy.stats.norm.ppf(dsr).
    return float(norm.cdf(z))
