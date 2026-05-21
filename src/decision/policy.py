"""Policy Protocol — spec §4.4."""
from __future__ import annotations

from typing import Any, Protocol

from decision.proposal import PortfolioSnapshot, TradeProposal
from models.base import PredictionBundle


class Policy(Protocol):
    async def propose(
        self,
        features: dict[str, Any],
        bundle: PredictionBundle,
        portfolio: PortfolioSnapshot,
    ) -> TradeProposal | None: ...


# ──────────────────────────────────────────────────────────────────
# ThresholdPolicy — first concrete Policy (spec §4.4)
# ──────────────────────────────────────────────────────────────────
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


@dataclass
class ThresholdPolicy:
    long_threshold: float
    short_threshold: float
    symbol: str
    mid_provider: Callable[[str], float]
    atr_provider: Callable[[str], float]           # stop distance proxy (USDT)
    tp_multiples: tuple[float, ...] = (1.5, 3.0)

    async def propose(
        self,
        features: dict[str, Any],
        bundle: PredictionBundle,
        portfolio: PortfolioSnapshot,
    ) -> TradeProposal | None:
        if bundle.direction == "flat" or bundle.size_multiplier == 0.0:
            return None
        if bundle.direction == "long" and bundle.prob_up < self.long_threshold:
            return None
        if bundle.direction == "short" and bundle.prob_up > (1 - self.short_threshold):
            return None

        mid = self.mid_provider(self.symbol)
        atr = self.atr_provider(self.symbol)
        sign = 1 if bundle.direction == "long" else -1
        entry = mid
        stop_loss = mid - sign * atr
        take_profit = [mid + sign * atr * m for m in self.tp_multiples]

        return TradeProposal(
            proposal_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            ts=datetime.now(tz=timezone.utc),
            symbol=self.symbol,
            direction=bundle.direction,
            entry=entry, stop_loss=stop_loss, take_profit=take_profit,
            size=1.0,                                # sized later by SizingPipeline
            confidence=bundle.prob_up if bundle.direction == "long" else 1 - bundle.prob_up,
            feature_snapshot=features,
            bundle_json=bundle.model_dump_json(),
            risk_checks=[],
            feature_registry_version=bundle.feature_registry_version,
            ml_model_version=bundle.ml_model_version,
            llm_prompt_version=bundle.llm_prompt_version,
        )
