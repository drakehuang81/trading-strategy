"""E2E smoke: full wire from features → broker → rebuild_positions."""
import random
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd
import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from decision.ensemble import Ensemble
from decision.policy import ThresholdPolicy
from decision.proposal import PortfolioSnapshot
from decision.risk.checks import (
    DailyLossKillSwitch, MandatoryStopLoss, MaxConcurrentPositions, SpreadGate,
)
from decision.risk.pipeline import RiskPipeline
from decision.sizing import FixedFractionalSizer
from execution.base import Order
from execution.paper_broker import PaperBroker, PaperBrokerConfig
from execution.repositories import BrokerEventRepo, ProposalRepo
from execution.replay import rebuild_positions
from features.registry import build_default_registry
from models.base import LLMContextFlags
from models.xgb_predictor import XGBPredictor


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_smoke_pipeline_runs_end_to_end(tmp_path: Path):
    # --- state ---
    db = tmp_path / "state.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    engine = sa.create_engine(f"sqlite:///{db}")
    event_repo = BrokerEventRepo(engine)
    proposal_repo = ProposalRepo(engine)

    # --- data (fixture CSV from Plan 1) ---
    df = pd.read_csv("tests/fixtures/ethusdt_1h_sample.csv",
                     parse_dates=["open_time"]).set_index("open_time")
    as_of = df.index[-5]
    registry = build_default_registry()
    features = registry.compute_all(df, as_of=as_of)

    # --- predictor + ensemble ---
    ml = XGBPredictor.stub(prob_up=0.65, ml_model_version="stub-v0")
    llm = AsyncMock()
    llm.flags = AsyncMock(return_value=LLMContextFlags(
        context_veto=False, veto_reason=None, structural_flags=[]))
    llm.prompt_version = "mock-v0"
    ensemble = Ensemble(ml=ml, llm_ctx=llm)
    bundle = await ensemble.predict(features)

    # --- policy ---
    policy = ThresholdPolicy(
        long_threshold=0.55, short_threshold=0.45,
        symbol="ETHUSDT",
        mid_provider=lambda s: float(df["close"].iloc[-5]),
        atr_provider=lambda s: float(df["close"].iloc[-5]) * 0.005,
    )
    portfolio = PortfolioSnapshot(
        equity_usdt=10_000, open_positions={},
        day_pnl_r=0.0, consecutive_wins=0,
    )
    proposal = await policy.propose(features, bundle, portfolio)
    assert proposal is not None

    # --- risk ---
    risk = RiskPipeline([
        MandatoryStopLoss(),
        SpreadGate(max_bps=20.0, spread_provider=lambda s: 5.0),
        DailyLossKillSwitch(threshold_r=-2.0),
        MaxConcurrentPositions(cap=3),
    ])
    results = risk.evaluate(proposal, portfolio)
    assert RiskPipeline.is_accepted(results)
    proposal = proposal.model_copy(update={"risk_checks": results})
    proposal_repo.insert(proposal, accepted=True)

    # --- size ---
    sized = FixedFractionalSizer(fraction=0.0025).size(
        equity_usdt=10_000, entry=proposal.entry, stop_loss=proposal.stop_loss,
    )
    assert sized > 0

    # --- execute ---
    broker = PaperBroker(
        cfg=PaperBrokerConfig(latency_ms_mean=5, latency_ms_stdev=1,
                              partial_fill_prob=0.0, rejection_prob=0.0),
        rng=random.Random(7),
        mid_provider=lambda s: proposal.entry,
    )
    side = "buy" if proposal.direction == "long" else "sell"
    order_id = await broker.submit(Order(
        client_order_id=proposal.proposal_id,
        symbol=proposal.symbol, side=side, type="market", qty=sized,
    ))

    # --- drain events, persist, rebuild positions ---
    events = []
    async for e in broker.events():
        events.append(e)
        event_repo.insert(e)
        if e.kind == "filled":
            break

    # rebuild from in-memory events (has symbol); repo round-trip tested in unit tests
    snap = rebuild_positions(events)
    assert "ETHUSDT" in snap

    # verify repo persisted all events
    stored = event_repo.all()
    assert len(stored) == len(events)
