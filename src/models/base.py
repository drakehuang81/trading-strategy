"""Model layer Protocol and PredictionBundle — spec §4.3.

PredictionBundle is the single object the Decision layer consumes.
prob_up comes only from the calibrated ML predictor; LLMContextProvider
contributes boolean/categorical flags via Ensemble (Plan 2)."""
from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel


class PredictionBundle(BaseModel):
    direction: Literal["long", "short", "flat"]
    prob_up: float
    horizon_bars: int
    size_multiplier: float = 1.0
    veto_reason: str | None = None
    feature_snapshot_hash: str
    feature_registry_version: str
    ml_model_version: str
    llm_prompt_version: str
    predictions_detail: dict[str, Any] = {}


class Predictor(Protocol):
    async def predict(self, features: dict[str, Any]) -> PredictionBundle: ...


class LLMContextFlags(BaseModel):
    context_veto: bool
    veto_reason: str | None = None
    structural_flags: list[str] = []


class LLMContextProvider(Protocol):
    """Distinct Protocol — not a Predictor. Emits boolean/categorical
    flags only; never outputs prob_up."""

    prompt_version: str

    async def flags(self, features: dict[str, Any]) -> LLMContextFlags: ...
