"""Pre-Live Gate driver — Plan 5B-4 Task 1."""
from __future__ import annotations

from execution.pre_live_gate import (
    GateResult,
    Gate,
    GateContext,
    run_all_gates,
)


class _FixedGate:
    """Test gate that returns a pre-set result."""

    def __init__(self, name: str, passed: bool, reason: str = "") -> None:
        self.name = name
        self._passed = passed
        self._reason = reason

    def evaluate(self, ctx: GateContext) -> GateResult:
        return GateResult(name=self.name, passed=self._passed, reason=self._reason)


def test_run_all_gates_returns_result_per_gate(tmp_path):
    ctx = GateContext(sqlite_path=str(tmp_path / "state.db"),
                       brier_threshold=0.24,
                       watchdog_log_path=str(tmp_path / "watchdog_pings.log"),
                       model_dir=str(tmp_path / "models"))
    gates = [
        _FixedGate("a", passed=True),
        _FixedGate("b", passed=False, reason="boom"),
    ]
    results = run_all_gates(gates, ctx)
    assert len(results) == 2
    assert results[0].name == "a" and results[0].passed
    assert results[1].name == "b" and not results[1].passed and results[1].reason == "boom"


def test_run_all_gates_does_not_short_circuit_on_first_failure(tmp_path):
    """All gates run even after one fails — operators want full picture."""
    ctx = GateContext(sqlite_path=str(tmp_path / "state.db"),
                       brier_threshold=0.24,
                       watchdog_log_path=str(tmp_path / "watchdog_pings.log"),
                       model_dir=str(tmp_path / "models"))
    gates = [
        _FixedGate("a", passed=False, reason="first"),
        _FixedGate("b", passed=False, reason="second"),
        _FixedGate("c", passed=True),
    ]
    results = run_all_gates(gates, ctx)
    assert [r.name for r in results] == ["a", "b", "c"]
    assert sum(r.passed for r in results) == 1


import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import sqlalchemy as sa


def _bootstrap_db(sqlite_path: str) -> sa.Engine:
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{sqlite_path}")
    alembic.command.upgrade(ac, "head")
    return sa.create_engine(f"sqlite:///{sqlite_path}")


def _seed_model_meta(model_dir: Path, version: str, brier_iso: float, brier_platt: float) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"xgb_{version}.json").write_text("{}")
    import pickle
    with open(model_dir / f"calib_{version}.pkl", "wb") as fh:
        pickle.dump({"calibrator": object(), "feature_order": []}, fh)
    (model_dir / f"meta_{version}.json").write_text(json.dumps({
        "model_version": version,
        "calibration_method": "platt" if brier_platt < brier_iso else "isotonic",
        "brier_isotonic": brier_iso,
        "brier_platt": brier_platt,
        "feature_order": [],
    }))


# ── Gate 1: no_repainting tests pass ───────────────────────────

def test_gate_no_repainting_passes_when_pytest_exits_zero(tmp_path):
    from execution.pre_live_gate import GateContext, NoRepaintingGate
    ctx = GateContext(sqlite_path=str(tmp_path / "x.db"),
                       brier_threshold=0.24,
                       watchdog_log_path=str(tmp_path / "w.log"),
                       model_dir=str(tmp_path / "m"))
    with patch("execution.pre_live_gate._run_pytest_repainting", return_value=0):
        result = NoRepaintingGate().evaluate(ctx)
    assert result.passed
    assert result.name == "no_repainting"


def test_gate_no_repainting_fails_when_pytest_exits_nonzero(tmp_path):
    from execution.pre_live_gate import GateContext, NoRepaintingGate
    ctx = GateContext(sqlite_path=str(tmp_path / "x.db"),
                       brier_threshold=0.24,
                       watchdog_log_path=str(tmp_path / "w.log"),
                       model_dir=str(tmp_path / "m"))
    with patch("execution.pre_live_gate._run_pytest_repainting", return_value=1):
        result = NoRepaintingGate().evaluate(ctx)
    assert not result.passed
    assert "exit code 1" in result.reason or "1" in result.reason


# ── Gate 2: backtest DSR > 0.5 ──────────────────────────────────

def test_gate_dsr_passes_when_latest_run_above_threshold(tmp_path):
    from execution.pre_live_gate import GateContext, BacktestDSRGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO backtest_runs (run_id, started_at, deflated_sharpe, "
            "cost_model_version, summary_json) "
            "VALUES (:rid, :ts, :dsr, :cmv, :sj)"
        ), {"rid": "r1", "ts": datetime.now(tz=timezone.utc),
            "dsr": 0.6, "cmv": "v1", "sj": "{}"})
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = BacktestDSRGate().evaluate(ctx)
    assert result.passed


def test_gate_dsr_fails_when_no_backtest_rows(tmp_path):
    from execution.pre_live_gate import GateContext, BacktestDSRGate
    db = tmp_path / "state.db"
    _bootstrap_db(str(db))
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = BacktestDSRGate().evaluate(ctx)
    assert not result.passed
    assert "no backtest_runs" in result.reason.lower()


def test_gate_dsr_fails_when_latest_below_threshold(tmp_path):
    from execution.pre_live_gate import GateContext, BacktestDSRGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO backtest_runs (run_id, started_at, deflated_sharpe, "
            "cost_model_version, summary_json) "
            "VALUES ('r1', :ts, 0.4, 'v1', '{}')"
        ), {"ts": datetime.now(tz=timezone.utc)})
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = BacktestDSRGate().evaluate(ctx)
    assert not result.passed
    assert "0.4" in result.reason


# ── Gate 3: Brier below threshold ───────────────────────────────

def test_gate_brier_passes_when_chosen_calibrator_brier_below_threshold(tmp_path):
    from execution.pre_live_gate import GateContext, CalibrationBrierGate
    model_dir = tmp_path / "models"
    _seed_model_meta(model_dir, "v1", brier_iso=0.26, brier_platt=0.22)
    ctx = GateContext(sqlite_path="", brier_threshold=0.24,
                       watchdog_log_path="", model_dir=str(model_dir))
    result = CalibrationBrierGate().evaluate(ctx)
    assert result.passed   # Platt 0.22 < 0.24


def test_gate_brier_fails_when_chosen_calibrator_brier_above_threshold(tmp_path):
    from execution.pre_live_gate import GateContext, CalibrationBrierGate
    model_dir = tmp_path / "models"
    _seed_model_meta(model_dir, "v1", brier_iso=0.27, brier_platt=0.26)
    ctx = GateContext(sqlite_path="", brier_threshold=0.24,
                       watchdog_log_path="", model_dir=str(model_dir))
    result = CalibrationBrierGate().evaluate(ctx)
    assert not result.passed
    assert "0.26" in result.reason


def test_gate_brier_fails_when_no_model_meta(tmp_path):
    from execution.pre_live_gate import GateContext, CalibrationBrierGate
    ctx = GateContext(sqlite_path="", brier_threshold=0.24,
                       watchdog_log_path="", model_dir=str(tmp_path / "empty"))
    result = CalibrationBrierGate().evaluate(ctx)
    assert not result.passed
    assert "no model" in result.reason.lower()
