import asyncio
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from execution.base import Order, Position
from execution.paper_broker import PaperBroker, PaperBrokerConfig


@pytest.fixture
def cfg():
    return PaperBrokerConfig(
        taker_bps=5.0,
        maker_bps=2.0,
        latency_ms_mean=10.0,
        latency_ms_stdev=1.0,
        partial_fill_prob=0.0,
        rejection_prob=0.0,
    )


@pytest.mark.asyncio
async def test_market_order_fills_deterministically(cfg):
    rng = random.Random(42)
    broker = PaperBroker(cfg, rng=rng, mid_provider=lambda sym: 2000.0)
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    order_id = await broker.submit(order)
    events = []
    async for e in broker.events():
        events.append(e)
        if e.kind == "filled":
            break
    assert events[0].kind == "submitted"
    assert events[-1].kind == "filled"
    assert abs(events[-1].fill_qty - 0.1) < 1e-9
    # deterministic: same seed → same fill_price
    broker2 = PaperBroker(cfg, rng=random.Random(42), mid_provider=lambda s: 2000.0)
    oid2 = await broker2.submit(order)
    events2 = [e async for e in _take_one_fill(broker2)]
    assert events2[-1].fill_price == events[-1].fill_price


async def _take_one_fill(broker):
    async for e in broker.events():
        yield e
        if e.kind == "filled":
            return


@pytest.mark.asyncio
async def test_slippage_and_fee_reflect_config(cfg):
    rng = random.Random(0)
    broker = PaperBroker(cfg, rng=rng, mid_provider=lambda sym: 2000.0)
    order = Order(client_order_id="c2", symbol="ETHUSDT", side="buy", type="market", qty=0.1)
    await broker.submit(order)
    async for e in broker.events():
        if e.kind == "filled":
            expected_fee = e.fill_price * 0.1 * cfg.taker_bps / 10000
            assert abs(e.fee - expected_fee) < 1e-6
            break


@pytest.mark.asyncio
async def test_funding_tick_emits_charged_event(tmp_path: Path, cfg):
    funding_df = pd.DataFrame(
        {"funding_rate": [0.0001]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-04-18T00:00", tz="UTC")]),
    )
    funding_df.to_parquet(tmp_path / "ETHUSDT.parquet")

    broker = PaperBroker(cfg, rng=random.Random(0), mid_provider=lambda s: 2000.0,
                        funding_dir=tmp_path)
    broker._positions["ETHUSDT"] = Position(
        symbol="ETHUSDT", qty=0.1, avg_entry=2000.0,
        opened_at=datetime(2026, 4, 17, tzinfo=timezone.utc),
        last_update_ts=datetime(2026, 4, 17, tzinfo=timezone.utc),
    )

    await broker.tick_funding(pd.Timestamp("2026-04-18T00:00", tz="UTC"))
    evt = await asyncio.wait_for(broker._queue.get(), timeout=0.1)
    assert evt.kind == "funding_charged"
    expected = 0.1 * 2000.0 * 0.0001
    assert abs(evt.fee - expected) < 1e-8
