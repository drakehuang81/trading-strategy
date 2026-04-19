"""Tests for Orchestrator boot sequence."""
from pathlib import Path

import pytest

from orchestrator import Orchestrator, OrchestratorConfig


@pytest.mark.asyncio
async def test_halt_file_aborts_boot(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "HALT").write_text("manual")
    orch = Orchestrator(OrchestratorConfig(sqlite_path=str(tmp_path / "state.db")))
    with pytest.raises(SystemExit) as exc_info:
        await orch.boot()
    assert "HALT" in str(exc_info.value)


@pytest.mark.asyncio
async def test_boot_creates_db(tmp_path: Path):
    db_path = tmp_path / "state.db"
    orch = Orchestrator(OrchestratorConfig(
        sqlite_path=str(db_path),
        halt_file=str(tmp_path / "HALT"),  # non-existent → no halt
    ))
    await orch.boot()
    assert db_path.exists()
    assert orch.engine is not None
