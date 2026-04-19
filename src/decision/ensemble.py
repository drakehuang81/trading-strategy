"""Ensemble — spec §4.3. ML prob + LLM flags → PredictionBundle.

LLM never writes a probability (§7.1). On veto, we set size_multiplier=0
but leave prob_up intact so audit logs show what the ML model said.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.base import LLMContextProvider, PredictionBundle, Predictor


@dataclass
class Ensemble:
    ml: Predictor
    llm_ctx: LLMContextProvider

    async def predict(self, features: dict[str, Any]) -> PredictionBundle:
        ml_pred = await self.ml.predict(features)
        flags = await self.llm_ctx.flags(features)
        update: dict[str, Any] = {"llm_prompt_version": self.llm_ctx.prompt_version}
        if flags.context_veto:
            update["size_multiplier"] = 0.0
            update["veto_reason"] = flags.veto_reason
        return ml_pred.model_copy(update=update)
