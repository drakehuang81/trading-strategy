"""ReplayBroker — deterministic discrete-time broker for backtest."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from execution.base import Order
from execution.paper_broker import PaperBrokerConfig
from execution.replay_broker import ReplayBroker


def _klines() -> pd.DataFrame:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(5)], name="open_time",
    )
    return pd.DataFrame({
        "open":   [3000.0, 3010.0, 3020.0, 3030.0, 3040.0],
        "high":   [3015.0, 3025.0, 3035.0, 3045.0, 3055.0],
        "low":    [2985.0, 2995.0, 3005.0, 3015.0, 3025.0],
        "close":  [3005.0, 3015.0, 3025.0, 3035.0, 3045.0],
        "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
    }, index=idx)


@pytest.fixture
def cfg():
    from execution.cost_model import SlippageConfig
    return PaperBrokerConfig(
        taker_bps=5.0, maker_bps=2.0,
        slippage=SlippageConfig(
            slippage_bps_base=1.0, slippage_bps_per_adv_unit=0.0, adv_stub=1000.0,
        ),
    )


@pytest.mark.asyncio
async def test_set_time_required_before_submit(cfg):
    rb = ReplayBroker(cfg=cfg, klines=_klines())
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    with pytest.raises(RuntimeError, match="set_time"):
        await rb.submit(order)


@pytest.mark.asyncio
async def test_submit_fills_at_current_bar_close_with_slippage(cfg):
    klines = _klines()
    rb = ReplayBroker(cfg=cfg, klines=klines)
    rb.set_time(klines.index[2])         # 3rd bar, close = 3025.0
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    await rb.submit(order)

    seen = []
    for _ in range(2):
        e = await asyncio.wait_for(rb._queue.get(), timeout=0.5)
        seen.append(e)
    assert seen[0].kind == "submitted"
    assert seen[1].kind == "filled"
    # 1 bps slippage on a buy: 3025 * 1.0001 = 3025.3025
    assert abs(seen[1].fill_price - 3025.3025) < 1e-9
    assert seen[1].fill_qty == 0.1   # buy is positive
    # fee = 3025.3025 * 0.1 * 5/10000 = 0.15126512...
    assert abs(seen[1].fee - 3025.3025 * 0.1 * 5.0 / 10_000) < 1e-9


@pytest.mark.asyncio
async def test_sell_signs_fill_qty_negative(cfg):
    klines = _klines()
    rb = ReplayBroker(cfg=cfg, klines=klines)
    rb.set_time(klines.index[0])
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="sell",
                  type="market", qty=0.05)
    await rb.submit(order)

    seen = []
    for _ in range(2):
        seen.append(await asyncio.wait_for(rb._queue.get(), timeout=0.5))
    assert seen[1].fill_qty == -0.05


@pytest.mark.asyncio
async def test_set_time_advances_funding_for_open_positions(cfg, tmp_path):
    klines = _klines()
    base = klines.index[0]
    # Funding tick at base + 1h (one tick that should fire when set_time crosses it)
    funding_ts = base + timedelta(hours=1)
    funding = pd.DataFrame(
        {"funding_rate": [0.0001]},
        index=pd.DatetimeIndex([funding_ts], name="ts"),
    )
    rb = ReplayBroker(cfg=cfg, klines=klines, funding=funding)
    rb.set_time(base)  # before the tick
    # Open a position so funding has something to charge
    order = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
                  type="market", qty=0.1)
    await rb.submit(order)
    # Drain submit/fill events
    for _ in range(2):
        await asyncio.wait_for(rb._queue.get(), timeout=0.5)
    # Now advance past the funding tick
    rb.set_time(base + timedelta(hours=2))
    funding_event = await asyncio.wait_for(rb._queue.get(), timeout=0.5)
    assert funding_event.kind == "funding_charged"
    assert funding_event.symbol == "ETHUSDT"


@pytest.mark.asyncio
async def test_positions_aggregate_fills(cfg):
    klines = _klines()
    rb = ReplayBroker(cfg=cfg, klines=klines)
    rb.set_time(klines.index[0])
    o1 = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
               type="market", qty=0.1)
    o2 = Order(client_order_id="c2", symbol="ETHUSDT", side="buy",
               type="market", qty=0.05)
    await rb.submit(o1)
    await rb.submit(o2)
    # Drain 4 events
    for _ in range(4):
        await asyncio.wait_for(rb._queue.get(), timeout=0.5)
    pos = await rb.positions()
    assert len(pos) == 1
    assert abs(pos[0].qty - 0.15) < 1e-9


@pytest.mark.asyncio
async def test_balance_returns_initial_equity(cfg):
    rb = ReplayBroker(cfg=cfg, klines=_klines(), initial_equity_usdt=5_000.0)
    bal = await rb.balance()
    assert bal.equity_usdt == 5_000.0
