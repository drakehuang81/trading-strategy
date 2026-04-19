"""Wiring factory — builds ScanContext + lifecycle bundle from OrchestratorConfig.

Single source of truth for production dependency injection. Orchestrator
calls this once in boot(); tests wire components by hand
(see tests/unit/test_pipeline.py).

Plan-4 note: BinanceKline requires an async Binance client and
`build_scan_context` is synchronous, so we ship a no-op in-process
data source that returns an empty frame on `fetch_latest`. Plan 5
replaces with `await BinanceKline.open(...)` at orchestrator boot.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa

from decision.ensemble import Ensemble
from decision.halt import HaltManager
from decision.policy import ThresholdPolicy
from decision.risk.checks import (
    DailyLossKillSwitch,
    MandatoryStopLoss,
    MaxConcurrentPositions,
    SpreadGate,
)
from decision.risk.pipeline import RiskPipeline
from decision.sizing import FixedFractionalSizer
from decision.triggers import (
    DailyLossTrigger,
    FeatureDriftTrigger,
    HeartbeatTrigger,
)
from execution.paper_broker import PaperBroker, PaperBrokerConfig
from execution.repositories import (
    BrokerEventRepo,
    ProposalRepo,
    SessionStateRepo,
)
from features.registry import build_default_registry
from interface.chat_llm import ChatLLM
from interface.repositories import MessageRepo, ToolCallRepo
from interface.tools import ToolExecutor
from models.llm.gemma_context import GemmaContextProvider
from models.llm.ollama_client import OllamaClient
from models.xgb_predictor import XGBPredictor
from observability.drift import FeatureDriftMonitor
from observability.drift_config import load_drift_config
from orchestrator import OrchestratorConfig
from pipeline import ScanContext


class _StubDataSource:
    """Placeholder data source — Plan 5 replaces with real BinanceKline."""
    name = "stub_data_source"

    def supports(self, symbol: str, timeframe: str) -> bool:
        return True

    async def fetch_latest(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _stub_mid(_symbol: str) -> float:
    # Plan 5 will pull the latest mid from data_source cache.
    return 3000.0


def _stub_atr(_symbol: str) -> float:
    # Plan 5 will compute from recent klines.
    return 15.0


def _stub_spread_bps(_symbol: str) -> float:
    return 0.0


def build_scan_context(
    cfg: OrchestratorConfig,
    engine: sa.Engine,
) -> tuple[ScanContext, dict[str, Any]]:
    """Build a fully wired ScanContext + lifecycle bundle."""
    # — Data
    data_source = _StubDataSource()

    # — Features
    registry = build_default_registry()

    # — Models
    ml = XGBPredictor.stub(prob_up=0.55, ml_model_version="stub-v0")
    ollama = OllamaClient(model=cfg.ollama_model, host=cfg.ollama_host)
    gemma = GemmaContextProvider(client=ollama)
    ensemble = Ensemble(ml=ml, llm_ctx=gemma)

    # — Broker
    rng = random.Random(cfg.paper_broker_seed)
    broker = PaperBroker(
        cfg=PaperBrokerConfig(),
        rng=rng,
        mid_provider=_stub_mid,
    )

    # — Policy (one symbol in watchlist for Plan 4)
    symbol = cfg.watchlist[0]
    policy = ThresholdPolicy(
        long_threshold=cfg.long_threshold,
        short_threshold=cfg.short_threshold,
        symbol=symbol,
        mid_provider=_stub_mid,
        atr_provider=_stub_atr,
    )

    # — Risk + sizing
    risk = RiskPipeline(checks=[
        MandatoryStopLoss(),
        SpreadGate(max_bps=10.0, spread_provider=_stub_spread_bps),
        DailyLossKillSwitch(threshold_r=cfg.daily_loss_max_r),
        MaxConcurrentPositions(cap=3),
    ])
    sizer = FixedFractionalSizer(fraction=0.01)

    # — Repos
    proposal_repo = ProposalRepo(engine)
    event_repo = BrokerEventRepo(engine)
    session_repo = SessionStateRepo(engine)
    message_repo = MessageRepo(engine)
    tool_call_repo = ToolCallRepo(engine)

    # — ChatLLM (read-only tools)
    tool_executor = ToolExecutor(engine=engine, broker=broker)
    chat_llm = ChatLLM(
        client=ollama,
        tool_executor=tool_executor,
        message_repo=message_repo,
        tool_call_repo=tool_call_repo,
    )

    # — Drift monitor (state dict aliased into FeatureDriftTrigger)
    drift_cfg = load_drift_config(cfg.drift_yaml)
    drift_monitor = FeatureDriftMonitor(
        reference={},   # populated by drift-monitor loop on first batch
        psi_threshold=drift_cfg.psi_threshold,
        ks_threshold=drift_cfg.ks_threshold,
        n_bins=drift_cfg.psi_bins,
    )
    drift_state: dict[str, Any] = {"breached": False}

    # — Halt manager with all three triggers populated
    halt = HaltManager(
        halt_file=Path(cfg.halt_file),
        engine=engine,
        triggers=[
            HeartbeatTrigger(engine=engine, max_stale_seconds=cfg.heartbeat_max_stale_seconds),
            DailyLossTrigger(engine=engine, max_loss_r=cfg.daily_loss_max_r),
            FeatureDriftTrigger(state=drift_state),
        ],
    )

    ctx = ScanContext(
        symbols=cfg.watchlist,
        halt=halt,
        data_source=data_source,
        registry=registry,
        ensemble=ensemble,
        policy=policy,
        risk=risk,
        sizer=sizer,
        broker=broker,
        proposal_repo=proposal_repo,
        event_repo=event_repo,
        session_repo=session_repo,
        chat_llm=chat_llm,
        telegram=None,
        notify_chat_id=cfg.notify_chat_id,
    )
    lifecycle: dict[str, Any] = {
        "ollama_client": ollama,
        "drift_state": drift_state,
        "drift_monitor": drift_monitor,
    }
    return ctx, lifecycle
