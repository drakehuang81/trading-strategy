"""Pure cost-model functions — Plan 5B-2 Task 1."""
from __future__ import annotations

import math

import pytest

from execution.cost_model import (
    slippage_fill_price,
    taker_or_maker_fee,
    SlippageConfig,
)


def test_buy_slippage_pushes_price_up():
    cfg = SlippageConfig(slippage_bps_base=1.0, slippage_bps_per_adv_unit=20.0, adv_stub=1000.0)
    # qty 100, base+20*0.1 = 3 bps total -> mid * 1.0003
    fill = slippage_fill_price(mid=2000.0, side="buy", qty=100.0, cfg=cfg)
    assert math.isclose(fill, 2000.0 * 1.0003, rel_tol=1e-12)


def test_sell_slippage_pushes_price_down():
    cfg = SlippageConfig(slippage_bps_base=1.0, slippage_bps_per_adv_unit=20.0, adv_stub=1000.0)
    fill = slippage_fill_price(mid=2000.0, side="sell", qty=100.0, cfg=cfg)
    assert math.isclose(fill, 2000.0 * 0.9997, rel_tol=1e-12)


def test_zero_qty_gives_only_base_slippage():
    cfg = SlippageConfig(slippage_bps_base=2.0, slippage_bps_per_adv_unit=10.0, adv_stub=1000.0)
    fill = slippage_fill_price(mid=2000.0, side="buy", qty=0.0, cfg=cfg)
    assert math.isclose(fill, 2000.0 * 1.0002, rel_tol=1e-12)


def test_taker_fee_for_market_order():
    fee = taker_or_maker_fee(price=2000.0, qty=0.5, order_type="market",
                             taker_bps=5.0, maker_bps=2.0)
    # 2000 * 0.5 * 5/10000 = 0.5
    assert math.isclose(fee, 0.5, rel_tol=1e-12)


def test_maker_fee_for_limit_order():
    fee = taker_or_maker_fee(price=2000.0, qty=0.5, order_type="limit",
                             taker_bps=5.0, maker_bps=2.0)
    # 2000 * 0.5 * 2/10000 = 0.2
    assert math.isclose(fee, 0.2, rel_tol=1e-12)


def test_fee_uses_abs_qty_for_short_fills():
    fee = taker_or_maker_fee(price=2000.0, qty=-0.5, order_type="market",
                             taker_bps=5.0, maker_bps=2.0)
    assert math.isclose(fee, 0.5, rel_tol=1e-12)
