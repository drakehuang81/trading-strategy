# src/execution/cost_model.py
"""Shared slippage + fee math — Plan 5B-2 Task 1.

Single source of truth so PaperBroker (live paper) and ReplayBroker
(backtest) produce IDENTICAL fill prices for the same (mid, side, qty,
config) tuple. Spec §7.2: "PaperBroker's cost model IS backtest's
cost model — single source of truth."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SlippageConfig:
    slippage_bps_base: float
    slippage_bps_per_adv_unit: float
    adv_stub: float


def slippage_fill_price(*, mid: float, side: Literal["buy", "sell"],
                        qty: float, cfg: SlippageConfig) -> float:
    """Apply linear slippage: base + per-unit-of-ADV.

    Buy fills at higher price (worse), sell at lower price (worse).
    """
    sign = 1 if side == "buy" else -1
    slip_bps = cfg.slippage_bps_base + cfg.slippage_bps_per_adv_unit * (qty / cfg.adv_stub)
    return mid * (1 + sign * slip_bps / 10_000)


def taker_or_maker_fee(*, price: float, qty: float,
                       order_type: Literal["market", "limit"],
                       taker_bps: float, maker_bps: float) -> float:
    """Absolute fee in quote currency. Uses abs(qty) so short fills
    (qty < 0) charge symmetric fees."""
    fee_bps = taker_bps if order_type == "market" else maker_bps
    return price * abs(qty) * fee_bps / 10_000.0
