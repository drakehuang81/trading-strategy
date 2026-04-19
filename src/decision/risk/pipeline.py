"""Risk pipeline — evaluate checks in order; any fail → reject."""
from __future__ import annotations

from typing import Iterable

from decision.proposal import PortfolioSnapshot, RiskCheckResult, TradeProposal
from decision.risk.base import RiskCheck


class RiskPipeline:
    def __init__(self, checks: Iterable[RiskCheck]) -> None:
        self._checks = list(checks)

    def evaluate(self, p: TradeProposal, port: PortfolioSnapshot) -> list[RiskCheckResult]:
        results: list[RiskCheckResult] = []
        for c in self._checks:
            r = c.check(p, port)
            results.append(r)
            if not r.passed:
                return results               # short-circuit on first fail
        return results

    @staticmethod
    def is_accepted(results: list[RiskCheckResult]) -> bool:
        return all(r.passed for r in results)
