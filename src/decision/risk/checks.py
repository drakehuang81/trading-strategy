"""Day-1 risk checks — spec §4.4.

Spread is queried from an injected provider so tests don't need a live
L2 book. Production will plug in a MidProvider backed by TickRecorder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from decision.proposal import PortfolioSnapshot, RiskCheckResult, TradeProposal


@dataclass
class MandatoryStopLoss:
    name: str = "MandatoryStopLoss"

    def check(self, p: TradeProposal, port: PortfolioSnapshot) -> RiskCheckResult:
        if p.stop_loss <= 0:
            return RiskCheckResult(name=self.name, passed=False, detail="stop_loss<=0")
        if p.direction == "long" and p.stop_loss >= p.entry:
            return RiskCheckResult(name=self.name, passed=False, detail="long SL>=entry")
        if p.direction == "short" and p.stop_loss <= p.entry:
            return RiskCheckResult(name=self.name, passed=False, detail="short SL<=entry")
        return RiskCheckResult(name=self.name, passed=True)


@dataclass
class SpreadGate:
    max_bps: float
    spread_provider: Callable[[str], float]    # symbol -> current bps
    name: str = "SpreadGate"

    def check(self, p: TradeProposal, port: PortfolioSnapshot) -> RiskCheckResult:
        bps = self.spread_provider(p.symbol)
        if bps > self.max_bps:
            return RiskCheckResult(name=self.name, passed=False, detail=f"{bps:.1f}bps>{self.max_bps}")
        return RiskCheckResult(name=self.name, passed=True)


@dataclass
class DailyLossKillSwitch:
    threshold_r: float     # e.g., -2.0
    name: str = "DailyLossKillSwitch"

    def check(self, p: TradeProposal, port: PortfolioSnapshot) -> RiskCheckResult:
        if port.day_pnl_r <= self.threshold_r:
            return RiskCheckResult(name=self.name, passed=False,
                                   detail=f"day_pnl_r={port.day_pnl_r}<={self.threshold_r}")
        return RiskCheckResult(name=self.name, passed=True)


@dataclass
class MaxConcurrentPositions:
    cap: int
    name: str = "MaxConcurrentPositions"

    def check(self, p: TradeProposal, port: PortfolioSnapshot) -> RiskCheckResult:
        currently_open_distinct = set(port.open_positions.keys())
        would_be = currently_open_distinct | {p.symbol}
        if len(would_be) > self.cap:
            return RiskCheckResult(name=self.name, passed=False,
                                   detail=f"{len(would_be)}>{self.cap}")
        return RiskCheckResult(name=self.name, passed=True)
