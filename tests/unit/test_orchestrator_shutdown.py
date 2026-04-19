"""Graceful shutdown on SIGTERM / explicit stop()."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_stop_event_unwinds_taskgroup(tmp_path: Path):
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
        telegram_token="",  # skip telegram
        drift_check_interval_minutes=60,
    )
    orch = Orchestrator(cfg)

    async def _run_and_stop():
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.2)  # let boot + TaskGroup start
        orch.request_stop()
        await asyncio.wait_for(task, timeout=5.0)

    await _run_and_stop()
    assert orch.is_stopping()
