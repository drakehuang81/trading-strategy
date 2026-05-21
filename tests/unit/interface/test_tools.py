"""ToolExecutor unit tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from decision.proposal import TradeProposal
from execution.repositories import ProposalRepo
from interface.tools import ToolExecutor


def _engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


def _proposal(symbol: str = "ETHUSDT", proposal_id: str = "p1") -> TradeProposal:
    return TradeProposal(
        proposal_id=proposal_id, trace_id="t1",
        ts=datetime.now(tz=timezone.utc),
        symbol=symbol, direction="long",
        entry=3000.0, stop_loss=2985.0, take_profit=[3022.5, 3045.0],
        size=1.0, confidence=0.65,
        feature_snapshot={"smc": {"trend": "up"}, "fib": {"level": 0.618}},
        bundle_json="{}", risk_checks=[],
        feature_registry_version="1.0.0",
        ml_model_version="stub-v0", llm_prompt_version="v0",
    )


@pytest.mark.asyncio
async def test_get_feature_snapshot_returns_latest(tmp_path: Path):
    engine = _engine(tmp_path)
    repo = ProposalRepo(engine)
    repo.insert(_proposal(), accepted=True)

    executor = ToolExecutor(engine=engine, broker=MagicMock())
    result = await executor.execute("get_feature_snapshot", {"symbol": "ETHUSDT"})
    data = json.loads(result)
    assert data["symbol"] == "ETHUSDT"
    assert data["features"]["smc"]["trend"] == "up"


@pytest.mark.asyncio
async def test_get_feature_snapshot_missing_returns_error(tmp_path: Path):
    engine = _engine(tmp_path)
    executor = ToolExecutor(engine=engine, broker=MagicMock())
    result = await executor.execute("get_feature_snapshot", {"symbol": "UNKNOWN"})
    data = json.loads(result)
    assert data["error"] == "no_proposal_found"
    assert data["symbol"] == "UNKNOWN"


@pytest.mark.asyncio
async def test_get_feature_snapshot_picks_most_recent(tmp_path: Path):
    """When multiple proposals exist for the same symbol, return the latest by ts."""
    import time
    engine = _engine(tmp_path)
    repo = ProposalRepo(engine)
    old = _proposal(proposal_id="p_old")
    old.feature_snapshot = {"smc": {"trend": "down"}}
    repo.insert(old, accepted=True)

    time.sleep(0.01)  # ensure ts strictly increases
    new = _proposal(proposal_id="p_new")
    new.feature_snapshot = {"smc": {"trend": "up"}}
    new.ts = datetime.now(tz=timezone.utc)
    repo.insert(new, accepted=True)

    executor = ToolExecutor(engine=engine, broker=MagicMock())
    result = await executor.execute("get_feature_snapshot", {"symbol": "ETHUSDT"})
    data = json.loads(result)
    assert data["features"]["smc"]["trend"] == "up"
