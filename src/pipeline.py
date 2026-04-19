"""Scan pipeline — spec §5.1 (macro) and §5.2 (deep).

Pure async functions that compose the full pipeline:
fetch → features → predict → propose → risk → size → execute.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from decision.proposal import PortfolioSnapshot, TradeProposal
from decision.risk.pipeline import RiskPipeline
from execution.base import Broker, Order
from execution.repositories import BrokerEventRepo, ProposalRepo, SessionStateRepo
from features.registry import FeatureRegistry

log = structlog.get_logger()


@dataclass
class ScanContext:
    """All dependencies for a scan pipeline run."""
    symbols: list[str]
    halt: Any                       # HaltManager
    data_source: Any                # BinanceKline
    registry: FeatureRegistry
    ensemble: Any                   # Ensemble
    policy: Any                     # Policy
    risk: RiskPipeline | Any
    sizer: Any                      # FixedFractionalSizer
    broker: Broker | Any
    proposal_repo: ProposalRepo
    event_repo: BrokerEventRepo
    session_repo: SessionStateRepo
    chat_llm: Any | None = None     # ChatLLM — optional for rationale
    telegram: Any | None = None     # TelegramBot — optional for notifications
    notify_chat_id: str | int = ""


async def scheduled_macro_scan(ctx: ScanContext, trace_id: str) -> None:
    """§5.1 — Scheduled 1h macro scan for each symbol."""
    if ctx.halt.is_halted():
        log.warning("halt_active_skipping_scan", trace_id=trace_id)
        return

    for symbol in ctx.symbols:
        try:
            await _scan_symbol(ctx, symbol, trace_id, timeframe="1h", n=200)
        except Exception:
            log.exception("scan_symbol_failed", symbol=symbol, trace_id=trace_id)


async def on_demand_deep_scan(ctx: ScanContext, symbol: str, trace_id: str) -> str:
    """§5.2 — On-demand 15m deep scan for a single symbol.

    Returns a text summary (sent back to Telegram).
    """
    if ctx.halt.is_halted():
        return "HALT is active — scan aborted."

    try:
        result = await _scan_symbol(ctx, symbol, trace_id, timeframe="15m", n=300)
        if result is None:
            return f"{symbol}: no signal on 15m deep scan."
        return (
            f"{symbol} {result.direction.upper()} signal\n"
            f"Entry: {result.entry:.2f}  SL: {result.stop_loss:.2f}\n"
            f"Confidence: {result.confidence:.2f}\n"
            f"Rationale: {result.rationale or 'N/A'}"
        )
    except Exception as e:
        log.exception("deep_scan_failed", symbol=symbol)
        return f"Deep scan failed: {e}"


async def _scan_symbol(
    ctx: ScanContext, symbol: str, trace_id: str,
    timeframe: str, n: int,
) -> TradeProposal | None:
    """Core scan logic for one symbol."""
    # 1. Fetch klines
    df = await ctx.data_source.fetch_latest(symbol, timeframe, n)
    if df.empty:
        log.warning("empty_klines", symbol=symbol, timeframe=timeframe, trace_id=trace_id)
        return None
    as_of = df.index[-1]

    # 2. Compute features
    features = ctx.registry.compute_all(df, as_of=as_of)

    # 3. Predict
    bundle = await ctx.ensemble.predict(features)

    # 4. Build portfolio snapshot
    today = datetime.now(tz=timezone.utc).date().isoformat()
    consecutive_wins, day_pnl_r = ctx.session_repo.get(today)
    positions = await ctx.broker.positions()
    balance = await ctx.broker.balance()
    portfolio = PortfolioSnapshot(
        equity_usdt=balance.equity_usdt,
        open_positions={p.symbol: p.qty for p in positions},
        day_pnl_r=day_pnl_r,
        consecutive_wins=consecutive_wins,
    )

    # 5. Propose
    proposal = await ctx.policy.propose(features, bundle, portfolio)
    if proposal is None:
        log.info("no_signal", symbol=symbol, trace_id=trace_id)
        return None

    proposal = proposal.model_copy(update={"trace_id": trace_id})

    # 6. Risk check
    results = ctx.risk.evaluate(proposal, portfolio)
    proposal = proposal.model_copy(update={"risk_checks": results})

    if not RiskPipeline.is_accepted(results):
        ctx.proposal_repo.insert(proposal, accepted=False)
        log.info("proposal_rejected", symbol=symbol, trace_id=trace_id,
                 failed=[r.name for r in results if not r.passed])
        return None

    # 7. Size
    sized_qty = ctx.sizer.size(
        equity_usdt=portfolio.equity_usdt,
        entry=proposal.entry,
        stop_loss=proposal.stop_loss,
    )
    proposal = proposal.model_copy(update={"size": sized_qty})

    # 8. Generate rationale (optional)
    if ctx.chat_llm:
        try:
            rationale = await ctx.chat_llm.explain(proposal.model_dump(mode="json"))
            proposal = proposal.model_copy(update={"rationale": rationale})
        except Exception:
            log.warning("rationale_generation_failed", symbol=symbol)

    # 9. Persist accepted proposal
    ctx.proposal_repo.insert(proposal, accepted=True)

    # 10. Submit order
    side = "buy" if proposal.direction == "long" else "sell"
    order_id = await ctx.broker.submit(Order(
        client_order_id=proposal.proposal_id,
        symbol=proposal.symbol,
        side=side,
        type="market",
        qty=sized_qty,
    ))
    log.info("order_submitted", symbol=symbol, order_id=order_id, trace_id=trace_id)

    # 11. Telegram notification (optional)
    if ctx.telegram and ctx.notify_chat_id:
        try:
            msg = (
                f"📊 {proposal.direction.upper()} {symbol}\n"
                f"Entry: {proposal.entry:.2f}  SL: {proposal.stop_loss:.2f}\n"
                f"Size: {sized_qty:.4f}  Confidence: {proposal.confidence:.2f}"
            )
            await ctx.telegram.send_message(ctx.notify_chat_id, msg)
        except Exception:
            log.warning("telegram_notify_failed", symbol=symbol)

    return proposal
