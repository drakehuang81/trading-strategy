"""Scan pipeline unit tests with mock stack."""
from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from decision.proposal import PortfolioSnapshot, RiskCheckResult, TradeProposal
from execution.base import Order
from models.base import LLMContextFlags, PredictionBundle
from pipeline import ScanContext, scheduled_macro_scan


def _make_engine(tmp_path: Path) -> sa.Engine:
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    return sa.create_engine(f"sqlite:///{db}")


def _mock_proposal() -> TradeProposal:
    from datetime import datetime, timezone
    return TradeProposal(
        proposal_id="p1", trace_id="t1", ts=datetime.now(tz=timezone.utc),
        symbol="ETHUSDT", direction="long", entry=3000.0, stop_loss=2985.0,
        take_profit=[3022.5, 3045.0], size=1.0, confidence=0.65,
        feature_snapshot={}, bundle_json="{}",
        risk_checks=[], feature_registry_version="1.0.0",
        ml_model_version="stub-v0", llm_prompt_version="mock-v0",
    )


@pytest.mark.asyncio
async def test_scan_skips_when_halted(tmp_path: Path):
    """Scan should no-op when HALT is active."""
    halt = MagicMock()
    halt.is_halted.return_value = True
    ctx = ScanContext(
        symbols=["ETHUSDT"], halt=halt,
        data_source=MagicMock(), registry=MagicMock(),
        ensemble=MagicMock(), policy=MagicMock(),
        risk=MagicMock(), sizer=MagicMock(),
        broker=MagicMock(), proposal_repo=MagicMock(),
        event_repo=MagicMock(), session_repo=MagicMock(),
    )
    await scheduled_macro_scan(ctx, trace_id="t1")
    halt.is_halted.assert_called_once()
    ctx.data_source.fetch_latest.assert_not_called()


@pytest.mark.asyncio
async def test_scan_runs_full_pipeline(tmp_path: Path):
    """Scan executes: fetch → features → predict → propose → risk → size → execute."""
    engine = _make_engine(tmp_path)

    df = pd.read_csv("tests/fixtures/ethusdt_1h_sample.csv",
                     parse_dates=["open_time"]).set_index("open_time")
    data_source = AsyncMock()
    data_source.fetch_latest = AsyncMock(return_value=df)

    from features.registry import build_default_registry
    registry = build_default_registry()

    bundle = PredictionBundle(
        direction="long", prob_up=0.65, horizon_bars=4,
        feature_snapshot_hash="abc", feature_registry_version="1.0.0",
        ml_model_version="stub-v0", llm_prompt_version="mock-v0",
    )
    ensemble = AsyncMock()
    ensemble.predict = AsyncMock(return_value=bundle)

    proposal = _mock_proposal()
    policy = AsyncMock()
    policy.propose = AsyncMock(return_value=proposal)

    risk = MagicMock()
    risk.evaluate.return_value = [RiskCheckResult(name="test", passed=True)]
    risk.is_accepted = MagicMock(return_value=True)

    sizer = MagicMock()
    sizer.size.return_value = 0.5

    broker = AsyncMock()
    broker.submit = AsyncMock(return_value="order-1")

    from execution.repositories import BrokerEventRepo, ProposalRepo, SessionStateRepo
    proposal_repo = ProposalRepo(engine)
    event_repo = BrokerEventRepo(engine)
    session_repo = SessionStateRepo(engine)

    halt = MagicMock()
    halt.is_halted.return_value = False

    ctx = ScanContext(
        symbols=["ETHUSDT"], halt=halt,
        data_source=data_source, registry=registry,
        ensemble=ensemble, policy=policy,
        risk=risk, sizer=sizer,
        broker=broker, proposal_repo=proposal_repo,
        event_repo=event_repo, session_repo=session_repo,
    )
    await scheduled_macro_scan(ctx, trace_id="t1")

    data_source.fetch_latest.assert_called_once()
    ensemble.predict.assert_called_once()
    policy.propose.assert_called_once()
    broker.submit.assert_called_once()
