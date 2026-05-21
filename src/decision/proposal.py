"""Decision layer payloads — spec §4.4."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class RiskCheckResult(BaseModel):
    name: str
    passed: bool
    detail: str | None = None


class TradeProposal(BaseModel):
    proposal_id: str
    trace_id: str
    ts: datetime
    symbol: str
    direction: Literal["long", "short"]
    entry: float
    stop_loss: float
    take_profit: list[float]
    size: float
    confidence: float
    feature_snapshot: dict[str, Any]
    bundle_json: str                 # serialized PredictionBundle — JSON
    risk_checks: list[RiskCheckResult]
    rationale: str | None = None
    feature_registry_version: str
    ml_model_version: str
    llm_prompt_version: str


class PortfolioSnapshot(BaseModel):
    equity_usdt: float
    open_positions: dict[str, float]  # symbol → signed qty
    day_pnl_r: float
    consecutive_wins: int
