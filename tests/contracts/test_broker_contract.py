"""Shared Broker contract — every implementation must pass this suite.

Lives under tests/contracts/ per spec §9.4. A concrete Broker adds a
pytest fixture named `broker` and a `mid_provider` fixture; the contract
tests run automatically.
"""
from __future__ import annotations

import asyncio
import random

import pytest

from execution.base import Order
from execution.paper_broker import PaperBroker, PaperBrokerConfig


@pytest.fixture
def paper_broker():
    return PaperBroker(
        cfg=PaperBrokerConfig(
            latency_ms_mean=5.0, latency_ms_stdev=1.0,
            partial_fill_prob=0.0, rejection_prob=0.0,
        ),
        rng=random.Random(1),
        mid_provider=lambda s: 2000.0,
    )


class BrokerContractTests:
    broker_fixture = "paper_broker"

    @pytest.mark.asyncio
    async def test_submit_yields_submitted_then_filled(self, request):
        broker = request.getfixturevalue(self.broker_fixture)
        o = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
        await broker.submit(o)
        seen = []
        async for e in broker.events():
            seen.append(e.kind)
            if e.kind in ("filled", "rejected"):
                break
        assert seen[0] == "submitted"
        assert seen[-1] == "filled"

    @pytest.mark.asyncio
    async def test_event_ids_are_unique(self, request):
        broker = request.getfixturevalue(self.broker_fixture)
        await broker.submit(Order(client_order_id="c1", symbol="ETHUSDT",
                                  side="buy", type="market", qty=0.1))
        await broker.submit(Order(client_order_id="c2", symbol="ETHUSDT",
                                  side="sell", type="market", qty=0.05))
        ids = set()
        for _ in range(4):
            e = await asyncio.wait_for(broker._queue.get(), timeout=1.0)
            assert e.event_id not in ids
            ids.add(e.event_id)


class TestPaperBrokerContract(BrokerContractTests):
    broker_fixture = "paper_broker"
