from __future__ import annotations

import math

from tests.helpers.feature_equality import features_equal


def test_scalar_equals() -> None:
    assert features_equal(1.0, 1.0)


def test_float_tolerance() -> None:
    assert features_equal(1.0, 1.0 + 1e-15)
    assert not features_equal(1.0, 1.0 + 1e-3)


def test_nan_equals_nan() -> None:
    assert features_equal({"x": math.nan}, {"x": math.nan})


def test_nested() -> None:
    a = {"a": [1.0, 2.0, {"b": math.nan}]}
    b = {"a": [1.0 + 1e-15, 2.0, {"b": math.nan}]}
    assert features_equal(a, b)


def test_structural_mismatch() -> None:
    assert not features_equal({"a": 1}, {"b": 1})
    assert not features_equal([1, 2], [1, 2, 3])
