"""E2E: Broker desync — §9.5 scenario 4.

Inject mismatched startup positions → PaperAutoRepair detects diff,
trusts broker, overwrites local, logs reconciliation_diff.
"""
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


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_broker_desync_auto_repaired(tmp_path: Path):
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    engine = sa.create_engine(f"sqlite:///{db}")

    now = datetime.now(tz=timezone.utc)
    broker = AsyncMock()
    broker.positions = AsyncMock(return_value=[
        Position(symbol="ETHUSDT", qty=1.5, avg_entry=3000.0,
                 opened_at=now, last_update_ts=now),
    ])

    # Local state says qty=1.0, broker says 1.5
    local_positions = {"ETHUSDT": 1.0}

    repair = PaperAutoRepair(engine=engine)
    diffs = await repair.reconcile(broker, local_positions)

    assert len(diffs) == 1
    assert diffs[0]["broker_qty"] == 1.5
    assert diffs[0]["local_qty"] == 1.0
    assert diffs[0]["action"] == "trust_broker"

    # Verify persisted
    with engine.connect() as conn:
        row = conn.execute(sa.text(
            "SELECT kind, resolution FROM reconciliation_diffs"
        )).first()
    assert row is not None
    assert row[0] == "position"
    assert row[1] == "auto_repaired"
