import asyncio
from datetime import datetime, timezone

import pytest

from decision.policy import ThresholdPolicy
from decision.proposal import PortfolioSnapshot
from models.base import PredictionBundle


def _bundle(direction="long", prob_up=0.70) -> PredictionBundle:
    return PredictionBundle(
        direction=direction, prob_up=prob_up, horizon_bars=4,
        size_multiplier=1.0, feature_snapshot_hash="h",
        feature_registry_version="1.0.0",
        ml_model_version="stub", llm_prompt_version="stub",
        predictions_detail={},
    )


@pytest.mark.asyncio
async def test_policy_emits_long_on_high_prob():
    pol = ThresholdPolicy(long_threshold=0.55, short_threshold=0.45,
                          symbol="ETHUSDT", mid_provider=lambda s: 2000.0,
                          atr_provider=lambda s: 10.0)
    features = {"smc": {}, "fib": {}, "liquidity": {}, "divergence": {},
                "funding": {}, "confidence": {}}
    port = PortfolioSnapshot(equity_usdt=10_000, open_positions={}, day_pnl_r=0.0, consecutive_wins=0)
    proposal = await pol.propose(features, _bundle(), port)
    assert proposal is not None
    assert proposal.direction == "long"
    assert proposal.stop_loss < proposal.entry


@pytest.mark.asyncio
async def test_policy_returns_none_on_flat():
    pol = ThresholdPolicy(long_threshold=0.55, short_threshold=0.45,
                          symbol="ETHUSDT", mid_provider=lambda s: 2000.0,
                          atr_provider=lambda s: 10.0)
    flat = _bundle(direction="flat", prob_up=0.50)
    port = PortfolioSnapshot(equity_usdt=10_000, open_positions={}, day_pnl_r=0.0, consecutive_wins=0)
    assert await pol.propose({}, flat, port) is None
