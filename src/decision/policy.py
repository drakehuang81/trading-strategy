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
