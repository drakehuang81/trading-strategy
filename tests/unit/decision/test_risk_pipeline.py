from datetime import datetime, timezone

import pytest

from decision.proposal import PortfolioSnapshot, TradeProposal
from decision.risk.pipeline import RiskPipeline
from decision.risk.checks import (
    DailyLossKillSwitch, MandatoryStopLoss, MaxConcurrentPositions, SpreadGate,
)


def _prop(**kw) -> TradeProposal:
    base = dict(
        proposal_id="p1", trace_id="t1",
        ts=datetime(2026, 4, 18, tzinfo=timezone.utc),
        symbol="ETHUSDT", direction="long",
        entry=2000.0, stop_loss=1980.0, take_profit=[2020.0],
        size=0.1, confidence=0.65,
        feature_snapshot={}, bundle_json="{}", risk_checks=[],
        feature_registry_version="1.0.0",
        ml_model_version="stub", llm_prompt_version="stub",
    )
    base.update(kw)
    return TradeProposal(**base)


def _port(**kw) -> PortfolioSnapshot:
    base = dict(equity_usdt=10_000, open_positions={}, day_pnl_r=0.0, consecutive_wins=0)
    base.update(kw)
    return PortfolioSnapshot(**base)


def test_mandatory_sl_rejects_when_missing():
    c = MandatoryStopLoss()
    p = _prop(stop_loss=0.0)
    assert c.check(p, _port()).passed is False


def test_spread_gate_rejects_wide_spread():
    c = SpreadGate(max_bps=20.0, spread_provider=lambda sym: 25.0)
    assert c.check(_prop(), _port()).passed is False


def test_daily_loss_kill_switch_rejects_below_threshold():
    c = DailyLossKillSwitch(threshold_r=-2.0)
    assert c.check(_prop(), _port(day_pnl_r=-2.5)).passed is False


def test_max_concurrent_rejects_at_cap():
    c = MaxConcurrentPositions(cap=3)
    full = _port(open_positions={"BTC": 0.1, "SOL": 0.2, "LINK": 1.0})
    assert c.check(_prop(), full).passed is False


def test_pipeline_rejects_on_first_fail_and_records_all_names():
    p = RiskPipeline([
        MandatoryStopLoss(),
        SpreadGate(max_bps=20.0, spread_provider=lambda s: 50.0),
        DailyLossKillSwitch(threshold_r=-2.0),
    ])
    results = p.evaluate(_prop(), _port())
    assert any(r.name == "SpreadGate" and not r.passed for r in results)
    assert p.is_accepted(results) is False
