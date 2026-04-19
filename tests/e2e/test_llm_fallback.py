"""E2E: LLM down — §9.5 scenario 3.

When Ollama times out, Ensemble falls through to ML-only prediction.
Pipeline continues uninterrupted.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from decision.ensemble import LLM_UNAVAILABLE_MARKER, Ensemble
from models.base import PredictionBundle
from models.xgb_predictor import XGBPredictor


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_ensemble_falls_back_on_llm_timeout():
    """When LLM raises, Ensemble returns ML-only prediction with llm_prompt_version='llm_unavailable'."""
    ml = XGBPredictor.stub(prob_up=0.65, ml_model_version="stub-v0")

    llm = AsyncMock()
    llm.flags = AsyncMock(side_effect=TimeoutError("Ollama connection refused"))
    llm.prompt_version = "timeout-v0"

    ensemble = Ensemble(ml=ml, llm_ctx=llm)

    features = {"smc": {}, "fib": {}, "liquidity": {},
                "divergence": {}, "funding_rate": {}, "confidence": {}}
    bundle = await ensemble.predict(features)

    # ML prediction still works
    assert bundle.prob_up == pytest.approx(0.65)
    assert bundle.direction == "long"
    assert bundle.ml_model_version == "stub-v0"

    # LLM fallback marker
    assert bundle.llm_prompt_version == LLM_UNAVAILABLE_MARKER

    # No veto applied (LLM was unavailable)
    assert bundle.size_multiplier == 1.0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_existing_ensemble_veto_still_works():
    """Verify the normal LLM veto path is not broken by the fallback logic."""
    from models.base import LLMContextFlags
    ml = XGBPredictor.stub(prob_up=0.65, ml_model_version="stub-v0")
    llm = AsyncMock()
    llm.flags = AsyncMock(return_value=LLMContextFlags(
        context_veto=True, veto_reason="structural_concern"))
    llm.prompt_version = "v1"

    ensemble = Ensemble(ml=ml, llm_ctx=llm)
    bundle = await ensemble.predict({"smc": {}})
    assert bundle.size_multiplier == 0.0
    assert bundle.veto_reason == "structural_concern"
    assert bundle.llm_prompt_version == "v1"
