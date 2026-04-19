"""RiskCheck Protocol — spec §4.4."""
from __future__ import annotations

from typing import Protocol

from decision.proposal import PortfolioSnapshot, RiskCheckResult, TradeProposal


class RiskCheck(Protocol):
    name: str

    def check(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioSnapshot,
    ) -> RiskCheckResult: ...
