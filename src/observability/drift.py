"""FeatureDriftMonitor — spec §9.5 scenario 5, Q3.

Computes PSI and KS statistics per feature to detect distribution shift.
When any feature breaches its threshold, ML predictor should be auto-disabled
and HALT may be triggered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def compute_psi(reference: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between reference and test arrays.

    PSI = Σ (P_i - Q_i) * ln(P_i / Q_i)
    where P = expected (reference), Q = actual (test).
    """
    eps = 1e-6
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    test_counts = np.histogram(test, bins=breakpoints)[0].astype(float)
    ref_pct = ref_counts / ref_counts.sum() + eps
    test_pct = test_counts / test_counts.sum() + eps
    psi: float = float(np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct)))
    return psi


def compute_ks(reference: np.ndarray, test: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max |F_ref(x) - F_test(x)|."""
    combined = np.sort(np.concatenate([reference, test]))
    n_ref = len(reference)
    n_test = len(test)
    cdf_ref = np.searchsorted(np.sort(reference), combined, side="right") / n_ref
    cdf_test = np.searchsorted(np.sort(test), combined, side="right") / n_test
    ks: float = float(np.max(np.abs(cdf_ref - cdf_test)))
    return ks


@dataclass
class FeatureDriftMonitor:
    """Checks feature distributions for drift against a reference window.

    Produces a list of breach records. Use has_breach() for a boolean gate.
    """
    reference: dict[str, np.ndarray]
    psi_threshold: float = 0.25
    ks_threshold: float = 0.10
    n_bins: int = 10

    def check(self, test: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        """Return list of breach dicts: {feature, metric, value, threshold}."""
        breaches: list[dict[str, Any]] = []
        for name, ref_vals in self.reference.items():
            if name not in test:
                continue
            test_vals = test[name]
            psi = compute_psi(ref_vals, test_vals, n_bins=self.n_bins)
            if psi > self.psi_threshold:
                breaches.append({
                    "feature": name, "metric": "psi",
                    "value": round(psi, 4), "threshold": self.psi_threshold,
                })
            ks = compute_ks(ref_vals, test_vals)
            if ks > self.ks_threshold:
                breaches.append({
                    "feature": name, "metric": "ks",
                    "value": round(ks, 4), "threshold": self.ks_threshold,
                })
        return breaches

    def has_breach(self, test: dict[str, np.ndarray]) -> bool:
        return len(self.check(test)) > 0
