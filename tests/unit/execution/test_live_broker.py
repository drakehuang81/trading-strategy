"""LiveBroker stub — refuses orders; Plan 5D will replace."""
from __future__ import annotations

import asyncio

import pytest

from execution.base import Order
from execution.live_broker import LiveBroker, LiveBrokerNotImplemented


@pytest.mark.asyncio
async def test_submit_raises_not_implemented():
    lb = LiveBroker()
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    with pytest.raises(LiveBrokerNotImplemented):
        await lb.submit(order)


@pytest.mark.asyncio
async def test_positions_returns_empty():
    lb = LiveBroker()
    assert await lb.positions() == []


@pytest.mark.asyncio
async def test_balance_returns_zero_equity():
    lb = LiveBroker()
    bal = await lb.balance()
    assert bal.equity_usdt == 0.0
    assert bal.free_usdt == 0.0


@pytest.mark.asyncio
async def test_cancel_raises_not_implemented():
    lb = LiveBroker()
    with pytest.raises(LiveBrokerNotImplemented):
        await lb.cancel("any_id")
