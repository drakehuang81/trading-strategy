from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.train_xgb import (
    walk_forward_calibration_choice,
    train_walk_forward,
)


def _synthetic_xy(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Two informative features + one noise.
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    noise = rng.normal(size=n)
    logits = 0.6 * f1 - 0.4 * f2 + 0.1 * noise
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    X = pd.DataFrame({"a.f1": f1, "a.f2": f2, "b.noise": noise})
    return X, pd.Series(y)


def test_walk_forward_calibration_choice_returns_one_of_two_methods():
    X, y = _synthetic_xy()
    choice = walk_forward_calibration_choice(X, y, n_splits=3)
    assert choice.method in {"isotonic", "platt"}
    assert 0.0 <= choice.brier_isotonic <= 0.5
    assert 0.0 <= choice.brier_platt <= 0.5
    # Picked method matches the lower OOS Brier.
    assert (choice.brier_isotonic <= choice.brier_platt) is (choice.method == "isotonic")


def test_train_walk_forward_writes_bundle(tmp_path):
    X, y = _synthetic_xy()
    bundle_meta = train_walk_forward(
        X=X, y=y,
        out_dir=tmp_path,
        training_window_start="2026-01-01",
        training_window_end="2026-04-01",
        n_splits=3,
    )
    # File artefacts present.
    assert (tmp_path / f"xgb_{bundle_meta.model_version}.json").exists()
    assert (tmp_path / f"calib_{bundle_meta.model_version}.pkl").exists()
    assert (tmp_path / f"meta_{bundle_meta.model_version}.json").exists()
    # Method is in {"isotonic", "platt"}; recorded in meta JSON.
    import json
    meta = json.loads((tmp_path / f"meta_{bundle_meta.model_version}.json").read_text())
    assert meta["calibration_method"] in {"isotonic", "platt"}
    assert meta["feature_order"] == list(X.columns)


def test_register_writes_model_versions_row(tmp_path):
    """Verifies _register inserts a row matching the bundle meta."""
    import sqlalchemy as sa
    import alembic.command
    import alembic.config
    from datetime import datetime, timezone
    from scripts.train_xgb import _register, BundleMeta

    db_path = tmp_path / "test.db"
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    alembic.command.upgrade(ac, "head")

    bundle = BundleMeta(
        model_version="testver12345",
        calibration_method="isotonic",
        brier_isotonic=0.20,
        brier_platt=0.21,
        feature_order=["a", "b"],
    )
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 1, tzinfo=timezone.utc)
    _register(bundle, out_dir=tmp_path,
              window_start=start, window_end=end,
              sqlite_path=str(db_path))

    engine = sa.create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        rows = conn.execute(sa.text(
            "SELECT ml_model_version, calibration_method, "
            "training_window_start, training_window_end "
            "FROM model_versions WHERE ml_model_version = :v"
        ), {"v": "testver12345"}).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "testver12345"
    assert rows[0][1] == "isotonic"
