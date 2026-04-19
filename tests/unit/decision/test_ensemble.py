import pytest
from unittest.mock import AsyncMock

from decision.ensemble import Ensemble
from models.base import LLMContextFlags, PredictionBundle


def _bundle(prob_up=0.7):
    return PredictionBundle(
        direction="long", prob_up=prob_up, horizon_bars=4, size_multiplier=1.0,
        feature_snapshot_hash="h", feature_registry_version="1.0.0",
        ml_model_version="m", llm_prompt_version="none", predictions_detail={},
    )


@pytest.mark.asyncio
async def test_no_veto_passes_through():
    ml = AsyncMock(); ml.predict = AsyncMock(return_value=_bundle(0.7))
    llm = AsyncMock(); llm.flags = AsyncMock(return_value=LLMContextFlags(
        context_veto=False, veto_reason=None, structural_flags=[]))
    llm.prompt_version = "v1"
    ens = Ensemble(ml=ml, llm_ctx=llm)
    out = await ens.predict({})
    assert out.prob_up == 0.7 and out.size_multiplier == 1.0
    assert out.llm_prompt_version == "v1"


@pytest.mark.asyncio
async def test_veto_zeros_size_multiplier_but_keeps_prob():
    ml = AsyncMock(); ml.predict = AsyncMock(return_value=_bundle(0.7))
    llm = AsyncMock(); llm.flags = AsyncMock(return_value=LLMContextFlags(
        context_veto=True, veto_reason="regime_mismatch", structural_flags=[]))
    llm.prompt_version = "v1"
    ens = Ensemble(ml=ml, llm_ctx=llm)
    out = await ens.predict({})
    assert out.size_multiplier == 0.0
    assert out.veto_reason == "regime_mismatch"
    assert out.prob_up == 0.7              # prob untouched
