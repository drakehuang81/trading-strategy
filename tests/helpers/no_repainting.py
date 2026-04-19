"""Parametrised no-repainting test (spec §9.2).

Any feature module can import `assert_no_repainting` and plug it into
its own parametrised test. Uses multiple seeds so a single lucky run
doesn't mask repainting."""
from __future__ import annotations

import random
from datetime import datetime
from typing import Protocol

import pandas as pd

from tests.helpers.feature_equality import features_equal


class _Computable(Protocol):
    required_lookback: int

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict: ...


def assert_no_repainting(
    feature: _Computable,
    df: pd.DataFrame,
    *,
    seed: int,
    n_samples: int = 50,
) -> None:
    rng = random.Random(seed)
    # Only sample timestamps where the feature has enough lookback.
    eligible = df.index[feature.required_lookback :]
    if len(eligible) == 0:
        raise AssertionError("DataFrame has no eligible as_of timestamps; check required_lookback")
    sample_size = min(n_samples, len(eligible))
    sampled = rng.sample(list(eligible), sample_size)
    for ts in sampled:
        truncated = df[df.index <= ts]
        full_result = feature.compute(df, as_of=ts)
        truncated_result = feature.compute(truncated, as_of=ts)
        assert features_equal(full_result, truncated_result), (
            f"Repainting detected in {type(feature).__name__} at as_of={ts}:\n"
            f"  full:      {full_result}\n"
            f"  truncated: {truncated_result}"
        )
