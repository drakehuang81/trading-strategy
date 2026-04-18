"""Recursive feature-dict comparator — spec §9.2.

Why not `==`: EMA / rolling-window features produce floats that differ
by ~1e-15 between full-df and truncated-df passes due to order of
operations. `math.isclose` absorbs that. NaN is treated as equal to
NaN (unlike IEEE-754 default), matching pandas semantics."""
from __future__ import annotations

import math
from typing import Any


def features_equal(
    a: Any,
    b: Any,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    if type(a) is not type(b):
        # int vs float OK if numerically close
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return _numbers_close(a, b, rel_tol, abs_tol)
        return False
    if isinstance(a, dict):
        assert isinstance(b, dict)
        if a.keys() != b.keys():
            return False
        return all(features_equal(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol) for k in a)
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple))
        if len(a) != len(b):
            return False
        return all(features_equal(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b))
    if isinstance(a, float):
        return _numbers_close(a, b, rel_tol, abs_tol)
    return a == b


def _numbers_close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
