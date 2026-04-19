"""E2E: full orchestrator boot → stop without Telegram / Ollama.

Proves the wiring survives a real boot and shutdown cycle.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import sqlalchemy as sa


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_orchestrator_boots_and_stops(tmp_path: Path):
    from orchestrator import Orchestrator, OrchestratorConfig

    drift_yaml = tmp_path / "drift.yaml"
    drift_yaml.write_text(
        "reference_window: 100\ntest_window: 10\npsi_bins: 5\n"
        "default:\n  psi_threshold: 0.25\n  ks_threshold: 0.10\n"
    )
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml=str(drift_yaml),
        telegram_token="",  # no telegram
        scan_interval_hours=99,  # no scan during the test window
        drift_check_interval_minutes=99,
    )
    orch = Orchestrator(cfg)

    task = asyncio.create_task(orch.run())
    # Wait long enough for boot to finish and heartbeat loop to write one row
    await asyncio.sleep(1.2)

    engine = sa.create_engine(f"sqlite:///{cfg.sqlite_path}")
    with engine.connect() as conn:
        count = conn.execute(sa.text("SELECT COUNT(*) FROM heartbeat")).scalar()
    assert count is not None and count >= 1

    assert orch.ctx is not None
    assert orch.ctx.halt is not None
    assert len(orch.ctx.halt._triggers) == 3

    orch.request_stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert orch.is_stopping()
