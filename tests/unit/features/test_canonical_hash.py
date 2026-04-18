"""Canonical hash reproducibility — spec §4.2."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from features.registry import FEATURE_REGISTRY_VERSION, canonical_hash


def test_version_is_stringy() -> None:
    assert isinstance(FEATURE_REGISTRY_VERSION, str)
    assert FEATURE_REGISTRY_VERSION.count(".") == 2


def test_hash_is_order_independent() -> None:
    a = {"smc": {"bias": "bull", "score": 3}, "fib": [0.618, 0.705]}
    b = {"fib": [0.618, 0.705], "smc": {"score": 3, "bias": "bull"}}
    assert canonical_hash(a) == canonical_hash(b)


def test_hash_changes_when_value_changes() -> None:
    a = canonical_hash({"x": 1.0})
    b = canonical_hash({"x": 1.0000001})
    assert a != b


def test_nan_is_stable() -> None:
    a = canonical_hash({"x": math.nan})
    b = canonical_hash({"x": math.nan})
    assert a == b


def test_datetime_serialises() -> None:
    ts = datetime(2026, 4, 18, tzinfo=timezone.utc)
    h = canonical_hash({"ts": ts})
    assert len(h) == 64


def test_version_tag_is_part_of_hash() -> None:
    """Same content, different version → different hash."""
    from features import registry
    original = registry.FEATURE_REGISTRY_VERSION
    content = {"x": 1}
    h1 = canonical_hash(content)
    registry.FEATURE_REGISTRY_VERSION = "9.9.9"
    try:
        h2 = canonical_hash(content)
    finally:
        registry.FEATURE_REGISTRY_VERSION = original
    assert h1 != h2
