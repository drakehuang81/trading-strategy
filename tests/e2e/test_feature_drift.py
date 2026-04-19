"""E2E: Feature drift — §9.5 scenario 5.

Inject synthetic OOD data → PSI breaches → HALT triggered.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from decision.halt import HaltManager
from observability.drift import FeatureDriftMonitor


@pytest.mark.e2e
def test_feature_drift_triggers_halt(tmp_path: Path):
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    engine = sa.create_engine(f"sqlite:///{db}")

    rng = np.random.RandomState(42)

    # Reference: normal(0, 1)
    reference = {"smc_bias": rng.normal(0, 1, 1000)}

    # Test: severely shifted — OOD
    test_data = {"smc_bias": rng.normal(5, 2, 100)}

    monitor = FeatureDriftMonitor(
        reference=reference, psi_threshold=0.25, ks_threshold=0.10,
    )

    # Verify drift detected
    assert monitor.has_breach(test_data)
    breaches = monitor.check(test_data)
    assert len(breaches) > 0

    # Wire to HaltManager as trigger
    class DriftTrigger:
        name = "feature_drift"
        def __init__(self, mon, data):
            self._mon = mon
            self._data = data
        def is_breached(self) -> bool:
            return self._mon.has_breach(self._data)

    trigger = DriftTrigger(monitor, test_data)
    halt_file = tmp_path / "HALT"
    halt = HaltManager(halt_file=halt_file, engine=engine, triggers=[trigger])

    # Activate HALT due to drift
    halt.activate(source="feature_drift", reason=f"PSI breach: {breaches}")
    assert halt.is_halted()

    # Resume blocked while drift persists
    ok, _ = halt.attempt_resume()
    assert not ok

    # Fix drift (use normal data)
    trigger._data = {"smc_bias": rng.normal(0, 1, 100)}
    ok, _ = halt.attempt_resume()
    assert ok
    assert not halt.is_halted()
