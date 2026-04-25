from __future__ import annotations

import json
import numpy as np
import pandas as pd

from scripts.train_xgb import write_drift_reference


def test_write_drift_reference_writes_one_entry_per_column(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a.x": rng.normal(size=500),
        "a.y": rng.normal(size=500),
    })
    out = tmp_path / "drift_reference.json"
    write_drift_reference(X, out)
    blob = json.loads(out.read_text())
    assert set(blob.keys()) == {"a.x", "a.y"}
    # Each entry is a sample of values (capped) we can use as PSI/KS reference.
    for vals in blob.values():
        assert isinstance(vals, list)
        assert 0 < len(vals) <= 5000
