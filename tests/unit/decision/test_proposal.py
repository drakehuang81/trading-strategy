from datetime import datetime, timezone
from decision.proposal import TradeProposal

def test_trade_proposal_round_trip():
    p = TradeProposal(
        proposal_id="p1", trace_id="t1",
        ts=datetime(2026, 4, 18, tzinfo=timezone.utc),
        symbol="ETHUSDT", direction="long",
        entry=2000.0, stop_loss=1980.0, take_profit=[2020.0, 2040.0],
        size=0.1, confidence=0.65,
        feature_snapshot={"smc": {}},
        bundle_json="{}",
        risk_checks=[],
        feature_registry_version="1.0.0",
        ml_model_version="stub",
        llm_prompt_version="stub",
    )
    assert p.model_dump()["direction"] == "long"
