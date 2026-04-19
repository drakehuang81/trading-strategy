"""Sizing pipeline — spec §4.4.

FixedFractionalSizer converts (equity, entry, SL) → contract units.
SizingPipeline applies post-sizing modifiers; Day-1 ships identity only
because win-streak tapering has no basis for calibrated models (§7.6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass
class FixedFractionalSizer:
    fraction: float       # e.g., 0.0025 = 0.25% risk per trade

    def size(self, equity_usdt: float, entry: float, stop_loss: float) -> float:
        risk_budget = equity_usdt * self.fraction
        distance = abs(entry - stop_loss)
        if distance <= 0:
            raise ValueError("entry == stop_loss; cannot size")
        return risk_budget / distance


class SizingModifier(Protocol):
    name: str

    def apply(self, size: float, consecutive_wins: int, day_pnl_r: float) -> float: ...


@dataclass
class IdentityModifier:
    name: str = "IdentityModifier"

    def apply(self, size: float, consecutive_wins: int, day_pnl_r: float) -> float:
        return size


class SizingPipeline:
    def __init__(self, modifiers: Iterable[SizingModifier]) -> None:
        self._modifiers = list(modifiers)

    def apply(self, size: float, *, consecutive_wins: int, day_pnl_r: float) -> float:
        for m in self._modifiers:
            size = m.apply(size, consecutive_wins, day_pnl_r)
        return size
