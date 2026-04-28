"""Wiring factory — async version (Plan 5A Task 4).

Opens the real BinanceKline, builds a RollingKlineCache, seeds it at boot,
and threads cache-backed mid/ATR/spread providers into ScanContext.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
import structlog

log = structlog.get_logger()

from data.binance_kline import BinanceKline
from data.kline_cache import RollingKlineCache
from data.providers import (
    cache_backed_atr_provider,
    cache_backed_mid_provider,
    cache_backed_spread_bps_provider,
)
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


async def build_scan_context(
    cfg: OrchestratorConfig,
    engine: sa.Engine,
) -> tuple[ScanContext, dict[str, Any]]:
    """Async: opens BinanceKline, builds the wired ScanContext."""
    data_source = await BinanceKline.open()

    registry = build_default_registry()

    if cfg.use_trained_model:
        # Plan 5A Task 9 wires the real load path; until then this branch is
        # expected to raise FileNotFoundError on a fresh checkout.
        from models.registry import load_latest_model
        ml = load_latest_model(Path(cfg.model_dir))
    else:
        ml = XGBPredictor.stub(prob_up=0.55, ml_model_version="stub-v0")

    ollama = OllamaClient(model=cfg.ollama_model, host=cfg.ollama_host)
    gemma = GemmaContextProvider(client=ollama)
    ensemble = Ensemble(ml=ml, llm_ctx=gemma)

    symbol = cfg.watchlist[0]
    timeframe = "1h"

    cache = RollingKlineCache(max_bars=200)
    try:
        seed = await data_source.fetch_latest(symbol, timeframe, 200)
        cache.ingest(symbol, timeframe, seed)
    except Exception:
        # Boot must succeed even if the seed fetch fails; refresh loop will
        # try again. Keep providers in fallback mode until then.
        # exc_info=True preserves traceback so a bug (wrong column name,
        # missing kwarg) is distinguishable from a network outage.
        log.warning("kline_seed_failed",
                    symbol=symbol, timeframe=timeframe, exc_info=True)

    mid_provider = cache_backed_mid_provider(cache, timeframe=timeframe, fallback=3000.0)
    atr_provider = cache_backed_atr_provider(cache, timeframe=timeframe, n=14, fallback=15.0)
    spread_provider = cache_backed_spread_bps_provider(cache, fallback=0.0)

    rng = random.Random(cfg.paper_broker_seed)
    if cfg.broker_kind == "paper":
        broker = PaperBroker(
            cfg=PaperBrokerConfig(),
            rng=rng,
            mid_provider=mid_provider,
        )
    elif cfg.broker_kind == "replay":
        from execution.replay_broker import ReplayBroker
        replay_klines = pd.read_parquet(cfg.replay_kline_path)
        if Path(cfg.replay_funding_path).exists():
            replay_funding = pd.read_parquet(cfg.replay_funding_path)
        else:
            log.warning("replay_funding_missing",
                        path=cfg.replay_funding_path,
                        note="backtest PnL will exclude funding cost")
            replay_funding = None
        broker = ReplayBroker(
            cfg=PaperBrokerConfig(),
            klines=replay_klines,
            funding=replay_funding,
            symbol=symbol,
        )
    elif cfg.broker_kind == "live":
        # Plan 5D-1: run all 8 Pre-Live Gates before allowing live mode.
        from execution.live_broker import LiveBroker
        from execution.pre_live_gate import (
            BacktestDSRGate,
            CalibrationBrierGate,
            DriftStabilityGate,
            GateContext,
            HaltDiversityGate,
            NoRepaintingGate,
            PaperRuntimeGate,
            PreLiveGateBlocked,
            ReconciliationGate,
            WatchdogUptimeGate,
            run_all_gates,
        )
        gate_ctx = GateContext(
            sqlite_path=cfg.sqlite_path,
            brier_threshold=0.24,
            watchdog_log_path="data/watchdog_pings.log",
            model_dir=cfg.model_dir,
        )
        gates = [
            NoRepaintingGate(),
            BacktestDSRGate(),
            CalibrationBrierGate(),
            PaperRuntimeGate(),
            ReconciliationGate(),
            DriftStabilityGate(),
            WatchdogUptimeGate(),
            HaltDiversityGate(),
        ]
        results = run_all_gates(gates, gate_ctx)
        failed = [r.name for r in results if not r.passed]
        if failed:
            log.error("pre_live_gate_blocked", failed_gates=failed)
            raise PreLiveGateBlocked(
                f"live mode refused; failed gates: {failed}. "
                f"Run `python -m scripts.pre_live_gate` for full per-gate reasons."
            )
        log.info("pre_live_gate_passed", gates=[r.name for r in results])
        broker = LiveBroker()
    else:
        raise ValueError(f"unknown broker_kind: {cfg.broker_kind!r}")

    policy = ThresholdPolicy(
        long_threshold=cfg.long_threshold,
        short_threshold=cfg.short_threshold,
        symbol=symbol,
        mid_provider=mid_provider,
        atr_provider=atr_provider,
    )

    risk = RiskPipeline(checks=[
        MandatoryStopLoss(),
        SpreadGate(max_bps=10.0, spread_provider=spread_provider),
        DailyLossKillSwitch(threshold_r=cfg.daily_loss_max_r),
        MaxConcurrentPositions(cap=3),
    ])
    sizer = FixedFractionalSizer(fraction=0.01)

    proposal_repo = ProposalRepo(engine)
    event_repo = BrokerEventRepo(engine)
    session_repo = SessionStateRepo(engine)
    message_repo = MessageRepo(engine)
    tool_call_repo = ToolCallRepo(engine)

    tool_executor = ToolExecutor(engine=engine, broker=broker)
    chat_llm = ChatLLM(
        client=ollama,
        tool_executor=tool_executor,
        message_repo=message_repo,
        tool_call_repo=tool_call_repo,
    )

    drift_cfg = load_drift_config(cfg.drift_yaml)
    reference: dict[str, list[float]] = {}
    drift_ref_path = Path(cfg.drift_reference_path)
    if drift_ref_path.exists():
        import json as _json
        reference = _json.loads(drift_ref_path.read_text())

    drift_monitor = FeatureDriftMonitor(
        reference=reference,
        psi_threshold=drift_cfg.psi_threshold,
        ks_threshold=drift_cfg.ks_threshold,
        n_bins=drift_cfg.psi_bins,
    )
    drift_state: dict[str, Any] = {"breached": False}

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
        "binance_kline": data_source,
        "kline_cache": cache,
    }
    return ctx, lifecycle
