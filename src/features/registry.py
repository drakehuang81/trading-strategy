"""Feature registry + canonical hash — spec §4.2."""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from hashlib import sha256
from typing import Any, Iterable

import pandas as pd

from features.base import Feature

FEATURE_REGISTRY_VERSION = "1.0.0"   # bump on registry composition change


def _canonical_default(o: Any) -> Any:
    """JSON default that renders floats via repr() (preserves bit-exact
    representation) and datetimes as ISO-8601 UTC strings."""
    if isinstance(o, float):
        if math.isnan(o):
            return "NaN"
        if math.isinf(o):
            return "Infinity" if o > 0 else "-Infinity"
        return repr(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, pd.Timestamp):
        return o.to_pydatetime().isoformat()
    if hasattr(o, "tolist"):   # numpy scalars / arrays
        return o.tolist()
    raise TypeError(f"Cannot canonicalise object of type {type(o).__name__}")


def canonical_hash(features: dict[str, Any]) -> str:
    """Deterministic content-addressable hash for a feature vector.

    Stable across: dict key insertion order, platform, Python patch
    versions. Unstable across: FEATURE_REGISTRY_VERSION changes (by
    design — a new version is a new hash space)."""
    payload = json.dumps(
        features,
        sort_keys=True,
        default=_canonical_default,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return sha256(
        f"{FEATURE_REGISTRY_VERSION}|{payload}".encode("ascii")
    ).hexdigest()


class FeatureRegistry:
    """Holds the live set of Feature implementations and provides
    point-in-time composition for the model layer."""

    def __init__(self, features: Iterable[Feature]) -> None:
        self._features = list(features)

    @property
    def features(self) -> list[Feature]:
        return list(self._features)

    def compute_all(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in self._features:
            out[f.name] = f.compute(df, as_of)
        return out


def build_default_registry(
    *,
    symbol: str = "ETHUSDT",
    confidence_direction: str = "long",
) -> FeatureRegistry:
    """Canonical Plan-1 feature set. Order is stable and part of the
    canonical hash — do not reshuffle without bumping
    FEATURE_REGISTRY_VERSION."""
    from features.confidence import ConfidenceFeature
    from features.divergence import DivergenceFeature
    from features.fibonacci import FibFeature
    from features.funding_rate import FundingFeature
    from features.liquidity import LiquidityFeature
    from features.smc import SMCFeature

    return FeatureRegistry([
        SMCFeature(),
        FibFeature(),
        LiquidityFeature(),
        DivergenceFeature(),
        FundingFeature(symbol=symbol),
        ConfidenceFeature(direction=confidence_direction),
    ])
