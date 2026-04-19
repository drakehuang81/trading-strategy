"""Orchestrator wiring tests — verify TaskGroup lifecycle."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from orchestrator import Orchestrator, OrchestratorConfig


def test_boot_creates_db(tmp_path: Path):
    db = tmp_path / "state.db"
    orch = Orchestrator(OrchestratorConfig(
        sqlite_path=str(db),
        halt_file=str(tmp_path / "HALT"),
    ))
    asyncio.run(orch.boot())
    assert db.exists()
    assert orch.engine is not None


def test_boot_aborts_on_halt_file(tmp_path: Path):
    halt = tmp_path / "HALT"
    halt.write_text("test halt")
    orch = Orchestrator(OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(halt),
    ))
    with pytest.raises(SystemExit):
        asyncio.run(orch.boot())


@pytest.mark.asyncio
async def test_boot_scheduler_present(tmp_path: Path):
    """After boot, scheduler should be configured."""
    db = tmp_path / "state.db"
    orch = Orchestrator(OrchestratorConfig(
        sqlite_path=str(db),
        halt_file=str(tmp_path / "HALT"),
    ))
    await orch.boot()
    assert orch._scheduler is not None


@pytest.mark.asyncio
async def test_run_uses_scan_context(tmp_path: Path, monkeypatch):
    """boot() populates self.ctx; _scheduled_scan delegates to scheduled_macro_scan."""
    drift_yaml = tmp_path / "drift.yaml"
    drift_yaml.write_text(
        "reference_window: 100\ntest_window: 10\npsi_bins: 5\n"
        "default:\n  psi_threshold: 0.25\n  ks_threshold: 0.10\n"
    )
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml=str(drift_yaml),
        telegram_token="",
    )
    orch = Orchestrator(cfg)
    await orch.boot()
    assert orch.ctx is not None
    assert orch.ctx.symbols == ["ETHUSDT"]

    fake = AsyncMock()
    monkeypatch.setattr("orchestrator.scheduled_macro_scan", fake)
    await orch._scheduled_scan()
    fake.assert_called_once()
    called_ctx = fake.call_args[0][0]
    assert called_ctx is orch.ctx
