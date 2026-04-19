from __future__ import annotations

from features.registry import (
    FEATURE_REGISTRY_VERSION,
    build_default_registry,
    canonical_hash,
)


def test_default_registry_has_six_features(eth_1h_df) -> None:
    registry = build_default_registry()
    names = [f.name for f in registry.features]
    assert names == ["smc", "fib", "liquidity", "divergence", "funding", "confidence"]


def test_compute_all_hashes_stably(eth_1h_df) -> None:
    registry = build_default_registry()
    as_of = eth_1h_df.index[-1]
    features1 = registry.compute_all(eth_1h_df, as_of)
    features2 = registry.compute_all(eth_1h_df, as_of)
    assert canonical_hash(features1) == canonical_hash(features2)


def test_registry_version_is_present() -> None:
    assert FEATURE_REGISTRY_VERSION
