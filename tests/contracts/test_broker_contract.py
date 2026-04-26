"""Shared Broker contract — every implementation must pass this suite.

Lives under tests/contracts/ per spec §9.4. A concrete Broker adds a
pytest fixture named `broker` and a `mid_provider` fixture; the contract
tests run automatically.
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from execution.base import Order
from execution.cost_model import SlippageConfig
from execution.paper_broker import PaperBroker, PaperBrokerConfig
from execution.replay_broker import ReplayBroker


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


@pytest.fixture
def replay_broker():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    klines = pd.DataFrame({
        "open":   [3000.0, 3010.0, 3020.0],
        "high":   [3015.0, 3025.0, 3035.0],
        "low":    [2985.0, 2995.0, 3005.0],
        "close":  [3005.0, 3015.0, 3025.0],
        "volume": [1.0, 1.0, 1.0],
    }, index=pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(3)], name="open_time",
    ))
    rb = ReplayBroker(
        cfg=PaperBrokerConfig(
            taker_bps=5.0, maker_bps=2.0,
            slippage=SlippageConfig(
                slippage_bps_base=1.0, slippage_bps_per_adv_unit=0.0, adv_stub=1000.0,
            ),
        ),
        klines=klines,
    )
    rb.set_time(klines.index[1])
    return rb


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


class TestReplayBrokerContract(BrokerContractTests):
    broker_fixture = "replay_broker"
