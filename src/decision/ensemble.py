"""Ensemble — spec §4.3. ML prob + LLM flags → PredictionBundle.

LLM never writes a probability (§7.1). On veto, we set size_multiplier=0
but leave prob_up intact so audit logs show what the ML model said.

On LLM failure (§9.5 scenario 3): fall through to ML-only prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from models.base import LLMContextProvider, PredictionBundle, Predictor

log = structlog.get_logger()

LLM_UNAVAILABLE_MARKER = "llm_unavailable"


@dataclass
class Ensemble:
    ml: Predictor
    llm_ctx: LLMContextProvider

    async def predict(self, features: dict[str, Any]) -> PredictionBundle:
        ml_pred = await self.ml.predict(features)
        update: dict[str, Any] = {}
        try:
            flags = await self.llm_ctx.flags(features)
            update["llm_prompt_version"] = self.llm_ctx.prompt_version
            if flags.context_veto:
                update["size_multiplier"] = 0.0
                update["veto_reason"] = flags.veto_reason
        except Exception as exc:
            log.warning("llm_context_failed_falling_back_to_ml_only", error=str(exc))
            update["llm_prompt_version"] = LLM_UNAVAILABLE_MARKER
        return ml_pred.model_copy(update=update)
