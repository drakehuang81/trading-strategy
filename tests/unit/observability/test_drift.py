"""FeatureDriftMonitor unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from observability.drift import compute_psi, compute_ks, FeatureDriftMonitor


def test_psi_identical_distributions():
    """PSI of identical distributions is 0."""
    ref = np.random.RandomState(42).normal(0, 1, 1000)
    psi = compute_psi(ref, ref, n_bins=10)
    assert psi == pytest.approx(0.0, abs=1e-10)


def test_psi_different_distributions():
    """PSI of shifted distribution exceeds threshold."""
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 1000)
    shifted = rng.normal(3, 1, 1000)  # large shift
    psi = compute_psi(ref, shifted, n_bins=10)
    assert psi > 0.25


def test_ks_identical_distributions():
    ref = np.random.RandomState(42).normal(0, 1, 1000)
    ks = compute_ks(ref, ref)
    assert ks < 0.05


def test_ks_different_distributions():
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 1000)
    shifted = rng.normal(3, 1, 1000)
    ks = compute_ks(ref, shifted)
    assert ks > 0.10


def test_monitor_no_breach():
    """No breach when distributions match."""
    rng = np.random.RandomState(42)
    ref = {"feat_a": rng.normal(0, 1, 1000)}
    test = {"feat_a": rng.normal(0, 1, 1000)}
    monitor = FeatureDriftMonitor(
        reference=ref, psi_threshold=0.25, ks_threshold=0.10,
    )
    breaches = monitor.check(test)
    assert len(breaches) == 0
    assert not monitor.has_breach(test)


def test_monitor_psi_breach():
    """Detects PSI breach on shifted data."""
    rng = np.random.RandomState(42)
    ref = {"feat_a": rng.normal(0, 1, 1000)}
    test = {"feat_a": rng.normal(5, 1, 100)}
    monitor = FeatureDriftMonitor(
        reference=ref, psi_threshold=0.25, ks_threshold=0.10,
    )
    breaches = monitor.check(test)
    assert len(breaches) > 0
    assert any(b["metric"] == "psi" for b in breaches)
    assert monitor.has_breach(test)
