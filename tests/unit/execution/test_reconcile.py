"""PaperAutoRepair reconciliation tests."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from execution.base import Position
from execution.reconcile import PaperAutoRepair


def _make_engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


@pytest.mark.asyncio
async def test_no_diff_no_repair(tmp_path: Path):
    engine = _make_engine(tmp_path)
    broker = AsyncMock()
    now = datetime.now(tz=timezone.utc)
    broker.positions = AsyncMock(return_value=[
        Position(symbol="ETHUSDT", qty=1.0, avg_entry=3000.0,
                 opened_at=now, last_update_ts=now),
    ])
    local_positions = {"ETHUSDT": 1.0}

    repair = PaperAutoRepair(engine=engine)
    diffs = await repair.reconcile(broker, local_positions)
    assert len(diffs) == 0


@pytest.mark.asyncio
async def test_missing_local_triggers_repair(tmp_path: Path):
    engine = _make_engine(tmp_path)
    broker = AsyncMock()
    now = datetime.now(tz=timezone.utc)
    broker.positions = AsyncMock(return_value=[
        Position(symbol="ETHUSDT", qty=1.0, avg_entry=3000.0,
                 opened_at=now, last_update_ts=now),
    ])
    local_positions: dict[str, float] = {}  # missing

    repair = PaperAutoRepair(engine=engine)
    diffs = await repair.reconcile(broker, local_positions)
    assert len(diffs) == 1
    assert diffs[0]["symbol"] == "ETHUSDT"

    # verify persisted to reconciliation_diffs
    with engine.connect() as conn:
        row = conn.execute(sa.text("SELECT kind, resolution FROM reconciliation_diffs")).first()
    assert row is not None
    assert row[0] == "position"
    assert row[1] == "auto_repaired"


@pytest.mark.asyncio
async def test_qty_mismatch_triggers_repair(tmp_path: Path):
    engine = _make_engine(tmp_path)
    broker = AsyncMock()
    now = datetime.now(tz=timezone.utc)
    broker.positions = AsyncMock(return_value=[
        Position(symbol="ETHUSDT", qty=1.0, avg_entry=3000.0,
                 opened_at=now, last_update_ts=now),
    ])
    local_positions = {"ETHUSDT": 0.5}  # mismatch

    repair = PaperAutoRepair(engine=engine)
    diffs = await repair.reconcile(broker, local_positions)
    assert len(diffs) == 1
    assert diffs[0]["broker_qty"] == 1.0
    assert diffs[0]["local_qty"] == 0.5
