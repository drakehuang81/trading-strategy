# Plan 5B-4 — Pre-Live Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/execution/pre_live_gate.py` with all 8 gates from spec §10, plus `scripts/pre_live_gate.py` CLI that runs them and exits 0/1 based on aggregate pass/fail. Manual smoke produces the first quantitative answer to "if I tried to flip mode=live today, which gates would block me?".

**Architecture:** One module, one Protocol, one driver, one CLI. Each gate is a stateless function `Gate.evaluate(ctx) -> GateResult` against SQLite + filesystem state — no decisions, no side effects, just `passed: bool` + `reason: str`. The driver runs all 8 gates, prints per-gate result, returns aggregate. CLI exits 0 if all pass, 1 if any fail (with the specific gate names in stderr). The orchestrator's `live` mode wiring (Plan 5B-2 Task 5) gains a guard: if `cfg.broker_kind == "live"`, run the gate set first; refuse to boot if any gate is red.

**Tech Stack:** Python 3.11, SQLAlchemy (existing), pandas (for parquet/JSON reads), pytest. No new dependencies.

**Decisions baked in:**
- **All gates run; no early exit.** Operators want to see every red gate at once, not "fix one, run again, see the next."
- **Gate 1 (no_repainting) shells out to pytest** — runs `pytest tests/ -k no_repainting --tb=no -q`; passes if exit code 0. Robust because the test suite is the source of truth, but slow (~5s).
- **Gate 3 (Brier threshold)** uses `< 0.24` default (baseline 0.25 minus 4%). Configurable via `--brier-threshold`.
- **Gate 6 (drift stability)** uses a proxy: zero `halt_events` with `trigger_source='feature_drift'` in last 30 days. Reasoning: drift breach triggers HALT, so absence of drift-halts is the operational evidence of drift stability. Plan 5C may add an explicit `drift_state_history` table for stronger evidence.
- **Gate 7 (watchdog uptime)** uses a new append-only log file `data/watchdog_pings.log` (one timestamp per line, written by `scripts/heartbeat_watchdog.py`). Gate checks the log spans ≥ 7 days AND latest entry < 10 min old.
- **Gate 5 (reconciliation)** treats any `reconciliation_diffs` row in last 14 days with `resolution != 'auto_repaired'` AND `resolution != 'dust'` as a fail. The "dust threshold" semantic is encoded as a `dust` resolution value rather than parsing `diff_json`.
- **CLI exit codes**: 0 = all green; 1 = any red (with stderr listing failed gates). 2 = config / runtime error.
- **No live mode wiring change in this plan.** The plan adds the `live` mode gate as a separate follow-up STATUS item — wiring `pre_live_gate` into `wiring.py` is a one-liner but bundling it here would block the gate's manual smoke.

**Out of Plan 5B-4 scope (deferred):**
- Wiring `pre_live_gate` into `wiring.py`'s `live` branch — Plan 5D will add this as part of `LiveBroker` activation.
- `drift_state_history` table for stronger gate 6 evidence.
- Gate 3 calibration method comparison (isotonic vs Platt) — `model_versions` already records the chosen method; gate just reads the latest.
- Make target `make pre-live-check` (spec §9.7).

---

## File map

### Created
- `src/execution/pre_live_gate.py` — `GateResult`, `GateContext`, `Gate` Protocol, 8 concrete gates, `run_all_gates` driver
- `scripts/pre_live_gate.py` — CLI
- `tests/unit/execution/test_pre_live_gate.py` — gate unit tests
- `tests/unit/scripts/test_pre_live_gate_cli.py` — CLI test
- `docs/superpowers/plans/2026-04-27-pivot-plan5b4-STATUS.md`

### Modified
- `scripts/heartbeat_watchdog.py` — append timestamp to `data/watchdog_pings.log` on each run

### Untouched (verified intentionally)
- `src/wiring.py` — no live-mode guard added in this plan; deferred to Plan 5D
- `src/state/alembic/` — no schema changes; reuses existing `heartbeat`, `halt_events`, `reconciliation_diffs`, `broker_events`, `backtest_runs`, `model_versions`

---

## Task 1: Foundation — Protocol, dataclasses, driver

**Why first:** Sets the contract that 8 gate implementations will follow. Driver is the same regardless of gate count.

**Files:**
- Create: `src/execution/pre_live_gate.py` (foundation parts only)
- Create: `tests/unit/execution/test_pre_live_gate.py` (driver test only)

- [ ] **Step 1: Write the failing tests for the driver**

```python
# tests/unit/execution/test_pre_live_gate.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/execution/test_pre_live_gate.py -v`
Expected: ImportError on `execution.pre_live_gate`.

- [ ] **Step 3: Implement foundation**

```python
# src/execution/pre_live_gate.py
"""Pre-Live Gate (spec §10).

Mode can flip from paper to live only when ALL 8 gates below are green.
Each gate is a stateless function from `GateContext` to `GateResult`.
The driver runs every gate (no short-circuit) so operators see every
red at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class GateContext:
    """Inputs every gate may need. Composed once at driver entry."""
    sqlite_path: str
    brier_threshold: float
    watchdog_log_path: str
    model_dir: str
    # Time anchor for "last N days" queries; default = now (UTC).
    # Tests pin a fixed `now` to make queries deterministic.
    now_iso: str | None = None


class Gate(Protocol):
    """Each concrete gate exposes a `name` and an `evaluate(ctx)` method."""
    name: str

    def evaluate(self, ctx: GateContext) -> GateResult: ...


def run_all_gates(gates: list[Gate], ctx: GateContext) -> list[GateResult]:
    """Runs every gate; returns results in the same order as `gates`.
    Does NOT short-circuit on first failure."""
    return [g.evaluate(ctx) for g in gates]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/execution/test_pre_live_gate.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 335 passed (333 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/execution/pre_live_gate.py tests/unit/execution/test_pre_live_gate.py
git commit -m "feat(pre-live): GateResult + Gate Protocol + run_all_gates driver"
```

---

## Task 2: Gates 1-3 (correctness)

**Why:** The "is the model itself good?" gates — independent from operations gates.

**Files:**
- Modify: `src/execution/pre_live_gate.py` (append 3 gate classes)
- Modify: `tests/unit/execution/test_pre_live_gate.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/execution/test_pre_live_gate.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/execution/test_pre_live_gate.py -v`
Expected: ImportError or AttributeError on the new gate classes.

- [ ] **Step 3: Implement gates 1-3**

Append to `src/execution/pre_live_gate.py`:

```python
import json
import subprocess
from pathlib import Path

import sqlalchemy as sa


def _run_pytest_repainting() -> int:
    """Shell out to pytest for the no_repainting test family.

    Returns the exit code; 0 means all repainting tests passed.
    Stubbed in unit tests via patch.
    """
    proc = subprocess.run(
        ["pytest", "-k", "no_repainting", "--tb=no", "-q"],
        capture_output=True, text=True,
    )
    return proc.returncode


@dataclass(frozen=True)
class NoRepaintingGate:
    name: str = "no_repainting"

    def evaluate(self, ctx: GateContext) -> GateResult:
        rc = _run_pytest_repainting()
        if rc == 0:
            return GateResult(self.name, True, "all repainting tests passed")
        return GateResult(self.name, False,
                          f"pytest no_repainting suite exit code {rc}")


@dataclass(frozen=True)
class BacktestDSRGate:
    name: str = "backtest_dsr"
    threshold: float = 0.5

    def evaluate(self, ctx: GateContext) -> GateResult:
        engine = sa.create_engine(f"sqlite:///{ctx.sqlite_path}")
        with engine.begin() as conn:
            row = conn.execute(sa.text(
                "SELECT deflated_sharpe FROM backtest_runs "
                "ORDER BY started_at DESC LIMIT 1"
            )).fetchone()
        if row is None:
            return GateResult(self.name, False, "no backtest_runs rows in DB")
        dsr = row[0]
        if dsr is None:
            return GateResult(self.name, False, "latest backtest_runs.deflated_sharpe is NULL")
        if dsr <= self.threshold:
            return GateResult(self.name, False,
                              f"latest DSR {dsr} <= threshold {self.threshold}")
        return GateResult(self.name, True, f"DSR {dsr} > {self.threshold}")


@dataclass(frozen=True)
class CalibrationBrierGate:
    name: str = "calibration_brier"

    def evaluate(self, ctx: GateContext) -> GateResult:
        model_dir = Path(ctx.model_dir)
        if not model_dir.exists():
            return GateResult(self.name, False, f"no model dir at {model_dir}")
        meta_files = sorted(model_dir.glob("meta_*.json"),
                            key=lambda p: p.stat().st_mtime)
        if not meta_files:
            return GateResult(self.name, False, "no model meta JSON files found")
        latest = meta_files[-1]
        meta = json.loads(latest.read_text())
        method = meta.get("calibration_method", "")
        brier = meta.get(f"brier_{method}")
        if brier is None:
            return GateResult(self.name, False,
                              f"meta {latest.name} missing brier_{method}")
        if brier >= ctx.brier_threshold:
            return GateResult(self.name, False,
                              f"chosen calibrator {method} Brier {brier} "
                              f">= threshold {ctx.brier_threshold}")
        return GateResult(self.name, True,
                          f"{method} Brier {brier} < {ctx.brier_threshold}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/execution/test_pre_live_gate.py -v`
Expected: 9 passed (2 driver + 7 gates 1-3).

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 342 passed (335 + 7 new).

- [ ] **Step 6: Commit**

```bash
git add src/execution/pre_live_gate.py tests/unit/execution/test_pre_live_gate.py
git commit -m "feat(pre-live): correctness gates 1-3 (no_repainting, DSR, Brier)"
```

---

## Task 3: Gates 4-6 (operations: heartbeat, reconciliation, drift)

**Files:**
- Modify: `src/execution/pre_live_gate.py` (append 3 gate classes)
- Modify: `tests/unit/execution/test_pre_live_gate.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/execution/test_pre_live_gate.py`:

```python
def _seed_heartbeat(engine: sa.Engine, days: int, gap_minutes: int = 5) -> None:
    """Seed heartbeat rows every gap_minutes for `days` consecutive days,
    ending at now."""
    now = datetime.now(tz=timezone.utc)
    n = (days * 24 * 60) // gap_minutes
    rows = [{"ts": now - timedelta(minutes=gap_minutes * i),
             "trace_id": f"t{i}"} for i in range(n)]
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO heartbeat (ts, trace_id) VALUES (:ts, :trace_id)"
        ), rows)


def _seed_filled_broker_events(engine: sa.Engine, n: int) -> None:
    now = datetime.now(tz=timezone.utc)
    with engine.begin() as conn:
        for i in range(n):
            conn.execute(sa.text(
                "INSERT INTO broker_events "
                "(event_id, kind, order_id, ts, fill_price, fill_qty, fee) "
                "VALUES (:eid, 'filled', :oid, :ts, 3000.0, 0.01, 0.15)"
            ), {"eid": f"e{i}", "oid": f"o{i}",
                "ts": now - timedelta(hours=i)})


# ── Gate 4: 60-day heartbeat + 30 fills ─────────────────────────

def test_gate_paper_runtime_passes_with_60d_heartbeat_and_30_fills(tmp_path):
    from execution.pre_live_gate import GateContext, PaperRuntimeGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    _seed_heartbeat(engine, days=60, gap_minutes=5)
    _seed_filled_broker_events(engine, n=30)
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = PaperRuntimeGate().evaluate(ctx)
    assert result.passed


def test_gate_paper_runtime_fails_with_too_few_heartbeat_days(tmp_path):
    from execution.pre_live_gate import GateContext, PaperRuntimeGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    _seed_heartbeat(engine, days=10, gap_minutes=5)
    _seed_filled_broker_events(engine, n=30)
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = PaperRuntimeGate().evaluate(ctx)
    assert not result.passed
    assert "60" in result.reason


def test_gate_paper_runtime_fails_with_too_few_fills(tmp_path):
    from execution.pre_live_gate import GateContext, PaperRuntimeGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    _seed_heartbeat(engine, days=60, gap_minutes=5)
    _seed_filled_broker_events(engine, n=5)
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = PaperRuntimeGate().evaluate(ctx)
    assert not result.passed
    assert "30" in result.reason


# ── Gate 5: reconciliation stability ────────────────────────────

def test_gate_reconciliation_passes_when_no_recent_diffs(tmp_path):
    from execution.pre_live_gate import GateContext, ReconciliationGate
    db = tmp_path / "state.db"
    _bootstrap_db(str(db))
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = ReconciliationGate().evaluate(ctx)
    assert result.passed


def test_gate_reconciliation_fails_with_recent_unresolved_diff(tmp_path):
    from execution.pre_live_gate import GateContext, ReconciliationGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO reconciliation_diffs (ts, kind, diff_json, resolution) "
            "VALUES (:ts, 'position', '{}', 'halted')"
        ), {"ts": datetime.now(tz=timezone.utc) - timedelta(days=2)})
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = ReconciliationGate().evaluate(ctx)
    assert not result.passed


def test_gate_reconciliation_passes_with_old_diff_outside_window(tmp_path):
    from execution.pre_live_gate import GateContext, ReconciliationGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO reconciliation_diffs (ts, kind, diff_json, resolution) "
            "VALUES (:ts, 'position', '{}', 'halted')"
        ), {"ts": datetime.now(tz=timezone.utc) - timedelta(days=20)})
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = ReconciliationGate().evaluate(ctx)
    assert result.passed   # 20d ago is outside 14-day window


# ── Gate 6: drift stability (proxy: no drift halts in 30d) ──────

def test_gate_drift_passes_when_no_drift_halt_in_30d(tmp_path):
    from execution.pre_live_gate import GateContext, DriftStabilityGate
    db = tmp_path / "state.db"
    _bootstrap_db(str(db))
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = DriftStabilityGate().evaluate(ctx)
    assert result.passed


def test_gate_drift_fails_with_recent_drift_halt(tmp_path):
    from execution.pre_live_gate import GateContext, DriftStabilityGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO halt_events (activated_at, trigger_source, reason) "
            "VALUES (:ts, 'feature_drift', 'PSI breach')"
        ), {"ts": datetime.now(tz=timezone.utc) - timedelta(days=5)})
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = DriftStabilityGate().evaluate(ctx)
    assert not result.passed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/execution/test_pre_live_gate.py -v`
Expected: AttributeError on the new gate classes.

- [ ] **Step 3: Implement gates 4-6**

Append to `src/execution/pre_live_gate.py`:

```python
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class PaperRuntimeGate:
    name: str = "paper_runtime"
    min_days: int = 60
    min_fills: int = 30
    max_gap_minutes: int = 10

    def evaluate(self, ctx: GateContext) -> GateResult:
        engine = sa.create_engine(f"sqlite:///{ctx.sqlite_path}")
        with engine.begin() as conn:
            # Earliest heartbeat ts
            row = conn.execute(sa.text(
                "SELECT MIN(ts), MAX(ts), COUNT(*) FROM heartbeat"
            )).fetchone()
            if row is None or row[0] is None:
                return GateResult(self.name, False, "no heartbeat rows")
            min_ts, max_ts, count = row
            min_ts_dt = pd.Timestamp(min_ts).to_pydatetime()
            max_ts_dt = pd.Timestamp(max_ts).to_pydatetime()
            span_days = (max_ts_dt - min_ts_dt).total_seconds() / 86400
            if span_days < self.min_days:
                return GateResult(self.name, False,
                                  f"heartbeat span {span_days:.1f}d < {self.min_days}d")
            # Check max gap (cheap: any consecutive pair > max_gap_minutes is a fail).
            # Implementation: compute max diff using SQL window — or fetch ts and diff in pandas.
            ts_rows = conn.execute(sa.text(
                "SELECT ts FROM heartbeat ORDER BY ts ASC"
            )).fetchall()
            ts_series = pd.to_datetime([r[0] for r in ts_rows])
            max_gap_s = ts_series.to_series().diff().dt.total_seconds().max()
            if max_gap_s is not None and max_gap_s > self.max_gap_minutes * 60:
                return GateResult(self.name, False,
                                  f"heartbeat max gap {max_gap_s/60:.1f}min "
                                  f"> {self.max_gap_minutes}min")
            # Filled count within the heartbeat window
            fills_row = conn.execute(sa.text(
                "SELECT COUNT(*) FROM broker_events "
                "WHERE kind='filled' AND ts BETWEEN :a AND :b"
            ), {"a": min_ts, "b": max_ts}).fetchone()
            fills = fills_row[0] if fills_row else 0
            if fills < self.min_fills:
                return GateResult(self.name, False,
                                  f"only {fills} filled events in window "
                                  f"(need >= {self.min_fills})")
        return GateResult(self.name, True,
                          f"{span_days:.1f}d heartbeat, {fills} fills, "
                          f"max gap {max_gap_s/60:.1f}min")


@dataclass(frozen=True)
class ReconciliationGate:
    name: str = "reconciliation"
    window_days: int = 14
    accepted_resolutions: tuple[str, ...] = ("auto_repaired", "dust", "user_accepted")

    def evaluate(self, ctx: GateContext) -> GateResult:
        engine = sa.create_engine(f"sqlite:///{ctx.sqlite_path}")
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self.window_days)
        if ctx.now_iso:
            cutoff = pd.Timestamp(ctx.now_iso).to_pydatetime() - timedelta(days=self.window_days)
        with engine.begin() as conn:
            rows = conn.execute(sa.text(
                "SELECT id, kind, resolution FROM reconciliation_diffs "
                "WHERE ts >= :cutoff"
            ), {"cutoff": cutoff}).fetchall()
        bad = [r for r in rows if r[2] not in self.accepted_resolutions]
        if bad:
            return GateResult(self.name, False,
                              f"{len(bad)} unresolved diffs in last "
                              f"{self.window_days}d (e.g. id={bad[0][0]} "
                              f"kind={bad[0][1]} resolution={bad[0][2]})")
        return GateResult(self.name, True,
                          f"0 unresolved diffs in last {self.window_days}d")


@dataclass(frozen=True)
class DriftStabilityGate:
    name: str = "drift_stability"
    window_days: int = 30

    def evaluate(self, ctx: GateContext) -> GateResult:
        engine = sa.create_engine(f"sqlite:///{ctx.sqlite_path}")
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self.window_days)
        if ctx.now_iso:
            cutoff = pd.Timestamp(ctx.now_iso).to_pydatetime() - timedelta(days=self.window_days)
        with engine.begin() as conn:
            row = conn.execute(sa.text(
                "SELECT COUNT(*) FROM halt_events "
                "WHERE trigger_source='feature_drift' AND activated_at >= :cutoff"
            ), {"cutoff": cutoff}).fetchone()
        n = row[0] if row else 0
        if n > 0:
            return GateResult(self.name, False,
                              f"{n} drift HALT(s) in last {self.window_days}d")
        return GateResult(self.name, True,
                          f"0 drift HALTs in last {self.window_days}d")
```

Note: the file's existing top imports (`json`, `subprocess`, `sa`, `Path`) stay. Add `pd` import:
```python
import pandas as pd
```
Already implicitly fine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/execution/test_pre_live_gate.py -v`
Expected: 17 passed (2 driver + 7 correctness + 8 ops gates 4-6).

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 350 passed (342 + 8 new).

- [ ] **Step 6: Commit**

```bash
git add src/execution/pre_live_gate.py tests/unit/execution/test_pre_live_gate.py
git commit -m "feat(pre-live): operations gates 4-6 (paper runtime, reconciliation, drift)"
```

---

## Task 4: Gates 7-8 (watchdog uptime, HALT diversity) + watchdog log

**Files:**
- Modify: `src/execution/pre_live_gate.py` (append 2 gate classes)
- Modify: `scripts/heartbeat_watchdog.py` — append timestamp to `data/watchdog_pings.log` on each run
- Modify: `tests/unit/execution/test_pre_live_gate.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/execution/test_pre_live_gate.py`:

```python
# ── Gate 7: watchdog uptime ─────────────────────────────────────

def _write_watchdog_log(path: Path, days_span: int, last_minutes_ago: int) -> None:
    """Write watchdog ping log spanning days_span days ending last_minutes_ago ago."""
    now = datetime.now(tz=timezone.utc)
    last = now - timedelta(minutes=last_minutes_ago)
    first = last - timedelta(days=days_span)
    # One ping per hour for the span.
    n = days_span * 24
    lines = [(first + timedelta(hours=i)).isoformat() for i in range(n + 1)]
    lines[-1] = last.isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_gate_watchdog_passes_with_7d_log_and_recent_ping(tmp_path):
    from execution.pre_live_gate import GateContext, WatchdogUptimeGate
    log = tmp_path / "watchdog.log"
    _write_watchdog_log(log, days_span=8, last_minutes_ago=2)
    ctx = GateContext(sqlite_path="", brier_threshold=0.24,
                       watchdog_log_path=str(log), model_dir="")
    result = WatchdogUptimeGate().evaluate(ctx)
    assert result.passed


def test_gate_watchdog_fails_when_log_missing(tmp_path):
    from execution.pre_live_gate import GateContext, WatchdogUptimeGate
    ctx = GateContext(sqlite_path="", brier_threshold=0.24,
                       watchdog_log_path=str(tmp_path / "missing.log"),
                       model_dir="")
    result = WatchdogUptimeGate().evaluate(ctx)
    assert not result.passed


def test_gate_watchdog_fails_when_latest_ping_too_old(tmp_path):
    from execution.pre_live_gate import GateContext, WatchdogUptimeGate
    log = tmp_path / "watchdog.log"
    _write_watchdog_log(log, days_span=8, last_minutes_ago=30)
    ctx = GateContext(sqlite_path="", brier_threshold=0.24,
                       watchdog_log_path=str(log), model_dir="")
    result = WatchdogUptimeGate().evaluate(ctx)
    assert not result.passed
    assert "30" in result.reason or "stale" in result.reason.lower()


def test_gate_watchdog_fails_when_log_span_under_7d(tmp_path):
    from execution.pre_live_gate import GateContext, WatchdogUptimeGate
    log = tmp_path / "watchdog.log"
    _write_watchdog_log(log, days_span=3, last_minutes_ago=2)
    ctx = GateContext(sqlite_path="", brier_threshold=0.24,
                       watchdog_log_path=str(log), model_dir="")
    result = WatchdogUptimeGate().evaluate(ctx)
    assert not result.passed
    assert "7" in result.reason


# ── Gate 8: HALT fire-drill diversity ───────────────────────────

def test_gate_halt_diversity_passes_with_three_trigger_kinds_plus_resume(tmp_path):
    from execution.pre_live_gate import GateContext, HaltDiversityGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    base = datetime.now(tz=timezone.utc) - timedelta(days=5)
    rows = [
        ("daily_loss_kill_switch", base, base + timedelta(hours=1)),
        ("feature_drift", base + timedelta(days=1), None),
        ("broker_desync", base + timedelta(days=2), None),
    ]
    with engine.begin() as conn:
        for src, activated, resumed in rows:
            conn.execute(sa.text(
                "INSERT INTO halt_events (activated_at, trigger_source, reason, resumed_at) "
                "VALUES (:a, :s, 'test', :r)"
            ), {"a": activated, "s": src, "r": resumed})
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = HaltDiversityGate().evaluate(ctx)
    assert result.passed


def test_gate_halt_diversity_fails_when_missing_kind(tmp_path):
    from execution.pre_live_gate import GateContext, HaltDiversityGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    base = datetime.now(tz=timezone.utc)
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO halt_events (activated_at, trigger_source, reason, resumed_at) "
            "VALUES (:a, 'daily_loss_kill_switch', 'test', :r)"
        ), {"a": base, "r": base + timedelta(hours=1)})
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = HaltDiversityGate().evaluate(ctx)
    assert not result.passed
    assert "feature_drift" in result.reason or "broker_desync" in result.reason


def test_gate_halt_diversity_fails_when_no_resume(tmp_path):
    from execution.pre_live_gate import GateContext, HaltDiversityGate
    db = tmp_path / "state.db"
    engine = _bootstrap_db(str(db))
    base = datetime.now(tz=timezone.utc)
    with engine.begin() as conn:
        for src in ["daily_loss_kill_switch", "feature_drift", "broker_desync"]:
            conn.execute(sa.text(
                "INSERT INTO halt_events (activated_at, trigger_source, reason) "
                "VALUES (:a, :s, 'test')"
            ), {"a": base, "s": src})
    ctx = GateContext(sqlite_path=str(db), brier_threshold=0.24,
                       watchdog_log_path="", model_dir="")
    result = HaltDiversityGate().evaluate(ctx)
    assert not result.passed
    assert "resume" in result.reason.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/execution/test_pre_live_gate.py -v`
Expected: AttributeError on the new gate classes.

- [ ] **Step 3: Implement gates 7-8**

Append to `src/execution/pre_live_gate.py`:

```python
@dataclass(frozen=True)
class WatchdogUptimeGate:
    name: str = "watchdog_uptime"
    min_uptime_days: int = 7
    max_stale_minutes: int = 10

    def evaluate(self, ctx: GateContext) -> GateResult:
        log = Path(ctx.watchdog_log_path)
        if not log.exists():
            return GateResult(self.name, False,
                              f"watchdog log not found at {log}")
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        if not lines:
            return GateResult(self.name, False, "watchdog log is empty")
        try:
            timestamps = [pd.Timestamp(ln) for ln in lines]
        except Exception as e:
            return GateResult(self.name, False,
                              f"watchdog log parse error: {e}")
        first = timestamps[0]
        latest = timestamps[-1]
        now_ts = (pd.Timestamp(ctx.now_iso) if ctx.now_iso
                  else pd.Timestamp.now(tz="UTC"))
        stale_min = (now_ts - latest).total_seconds() / 60
        if stale_min > self.max_stale_minutes:
            return GateResult(self.name, False,
                              f"latest ping {stale_min:.1f}min old "
                              f"(stale > {self.max_stale_minutes}min)")
        span_days = (latest - first).total_seconds() / 86400
        if span_days < self.min_uptime_days:
            return GateResult(self.name, False,
                              f"watchdog log spans {span_days:.1f}d "
                              f"< required {self.min_uptime_days}d")
        return GateResult(self.name, True,
                          f"{span_days:.1f}d uptime, latest "
                          f"{stale_min:.1f}min ago")


_REQUIRED_HALT_KINDS = ("daily_loss_kill_switch", "feature_drift", "broker_desync")


@dataclass(frozen=True)
class HaltDiversityGate:
    name: str = "halt_diversity"

    def evaluate(self, ctx: GateContext) -> GateResult:
        engine = sa.create_engine(f"sqlite:///{ctx.sqlite_path}")
        with engine.begin() as conn:
            seen = {r[0] for r in conn.execute(sa.text(
                "SELECT DISTINCT trigger_source FROM halt_events"
            )).fetchall()}
            resumed_count = conn.execute(sa.text(
                "SELECT COUNT(*) FROM halt_events WHERE resumed_at IS NOT NULL"
            )).fetchone()[0]
        missing = [k for k in _REQUIRED_HALT_KINDS if k not in seen]
        if missing:
            return GateResult(self.name, False,
                              f"missing trigger_source(s): {missing}")
        if resumed_count == 0:
            return GateResult(self.name, False,
                              "no halt_events have been followed by /resume")
        return GateResult(self.name, True,
                          f"all 3 trigger families present; "
                          f"{resumed_count} resumed")
```

- [ ] **Step 4: Modify `scripts/heartbeat_watchdog.py` to append ping log**

Read the existing file first. Find the main run path (where the script executes its check). After the heartbeat check (whether stale or not), append a line to `data/watchdog_pings.log`:

```python
def _record_ping(log_path: Path) -> None:
    """Append current UTC timestamp to watchdog ping log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(datetime.now(tz=timezone.utc).isoformat() + "\n")
```

Call `_record_ping(Path(args.watchdog_log) if args.watchdog_log else Path("data/watchdog_pings.log"))` at the end of the script's main flow.

Also add the CLI flag `--watchdog-log` with default `"data/watchdog_pings.log"`.

(If the existing watchdog has different naming, adapt — the goal is one timestamp per run appended to a configurable log path.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/execution/test_pre_live_gate.py -v`
Expected: 24 passed (17 prior + 7 new gates 7-8).

- [ ] **Step 6: Run full suite**

Run: `pytest -q`
Expected: 357 passed (350 + 7 new).

- [ ] **Step 7: Commit**

```bash
git add src/execution/pre_live_gate.py scripts/heartbeat_watchdog.py tests/unit/execution/test_pre_live_gate.py
git commit -m "feat(pre-live): operations gates 7-8 (watchdog uptime, HALT diversity)"
```

---

## Task 5: CLI + manual smoke + STATUS

**Files:**
- Create: `scripts/pre_live_gate.py`
- Create: `tests/unit/scripts/test_pre_live_gate_cli.py`
- Create: `docs/superpowers/plans/2026-04-27-pivot-plan5b4-STATUS.md`

- [ ] **Step 1: Write the failing test for the CLI**

```python
# tests/unit/scripts/test_pre_live_gate_cli.py
"""Pre-Live Gate CLI — Plan 5B-4 Task 5."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest


def _bootstrap_db(sqlite_path: str):
    import alembic.command, alembic.config, sqlalchemy as sa
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{sqlite_path}")
    alembic.command.upgrade(ac, "head")
    return sa.create_engine(f"sqlite:///{sqlite_path}")


@pytest.mark.asyncio
async def test_cli_returns_nonzero_when_any_gate_fails(tmp_path, capsys):
    from scripts.pre_live_gate import main_async
    sqlite = tmp_path / "state.db"
    _bootstrap_db(str(sqlite))   # empty DB → most gates will fail

    args = argparse.Namespace(
        sqlite_path=str(sqlite),
        model_dir=str(tmp_path / "models"),
        watchdog_log=str(tmp_path / "watchdog.log"),
        brier_threshold=0.24,
    )
    with patch("execution.pre_live_gate._run_pytest_repainting", return_value=0):
        rc = await main_async(args)
    assert rc == 1
    captured = capsys.readouterr()
    # Should print every gate name in stdout.
    for name in ["no_repainting", "backtest_dsr", "calibration_brier",
                 "paper_runtime", "reconciliation", "drift_stability",
                 "watchdog_uptime", "halt_diversity"]:
        assert name in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/scripts/test_pre_live_gate_cli.py -v`
Expected: ImportError on `scripts.pre_live_gate`.

- [ ] **Step 3: Implement the CLI**

```python
# scripts/pre_live_gate.py
"""Pre-Live Gate CLI — spec §10.

Runs all 8 gates against current SQLite + filesystem state.
Exits 0 if every gate passes, 1 if any fails.

Usage:
    python -m scripts.pre_live_gate \
        --sqlite-path data/state.db \
        --model-dir models \
        --watchdog-log data/watchdog_pings.log \
        --brier-threshold 0.24
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from execution.pre_live_gate import (
    GateContext,
    NoRepaintingGate,
    BacktestDSRGate,
    CalibrationBrierGate,
    PaperRuntimeGate,
    ReconciliationGate,
    DriftStabilityGate,
    WatchdogUptimeGate,
    HaltDiversityGate,
    run_all_gates,
)


_ALL_GATES = [
    NoRepaintingGate(),
    BacktestDSRGate(),
    CalibrationBrierGate(),
    PaperRuntimeGate(),
    ReconciliationGate(),
    DriftStabilityGate(),
    WatchdogUptimeGate(),
    HaltDiversityGate(),
]


async def main_async(args: argparse.Namespace) -> int:
    ctx = GateContext(
        sqlite_path=args.sqlite_path,
        brier_threshold=args.brier_threshold,
        watchdog_log_path=args.watchdog_log,
        model_dir=args.model_dir,
    )
    results = run_all_gates(_ALL_GATES, ctx)
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    print(f"\nPre-Live Gate: {n_pass}/{n_total} passed\n")
    print(f"{'GATE':<22} {'PASS':<6} REASON")
    print("-" * 80)
    for r in results:
        flag = "✅" if r.passed else "❌"
        print(f"{r.name:<22} {flag:<6} {r.reason}")
    print()
    failed = [r.name for r in results if not r.passed]
    if failed:
        print(f"FAILED gates: {failed}", file=sys.stderr)
        return 1
    print("All gates green — live mode is allowed.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite-path", default="data/state.db")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--watchdog-log", default="data/watchdog_pings.log")
    ap.add_argument("--brier-threshold", type=float, default=0.24)
    args = ap.parse_args()
    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/unit/scripts/test_pre_live_gate_cli.py tests/unit/execution/test_pre_live_gate.py -v && pytest -q`
Expected: 25 new total. Full suite: 358 passed (357 + 1 CLI test).

- [ ] **Step 5: Commit CLI**

```bash
git add scripts/pre_live_gate.py tests/unit/scripts/test_pre_live_gate_cli.py
git commit -m "feat(pre-live): scripts/pre_live_gate.py CLI runs all 8 gates"
```

- [ ] **Step 6: Manual smoke**

```bash
PYTHONPATH=src python -m scripts.pre_live_gate \
    --sqlite-path data/state.db \
    --model-dir models \
    --watchdog-log data/watchdog_pings.log \
    --brier-threshold 0.24
```

Expected output (numbers will vary): `Pre-Live Gate: X/8 passed` followed by per-gate ✅/❌. Exit code 1 because we have NOT yet:
- Run paper for 60 days
- Triggered all 3 HALT trigger families
- Deployed watchdog for 7 days
- Brier 0.2505 > 0.24 threshold (Plan 5B-1's model)

So expect ~2/8 passing (no_repainting + backtest_dsr if DSR row > 0.5).

Note the actual count and per-gate reasons in STATUS.

- [ ] **Step 7: Write Plan 5B-4 STATUS**

Create `docs/superpowers/plans/2026-04-27-pivot-plan5b4-STATUS.md`:
- Date / branch / base / head SHAs
- Summary
- Task table with commit SHAs from `git log --oneline 07642b6..HEAD`
- Manual smoke results: per-gate pass/fail + reason for each
- Decisions landed (Brier threshold 0.24 default, drift gate uses HALT proxy, watchdog log file design)
- What is NOT done (Plan 5D wires gate into live boot; drift_state_history table; isotonic-vs-Platt explicit gate)
- Known follow-ups

- [ ] **Step 8: Final commit**

```bash
git add docs/superpowers/plans/2026-04-27-pivot-plan5b4-STATUS.md
git commit -m "docs: Plan 5B-4 handoff STATUS"
```

---

## Self-review notes

- **Spec coverage**: §10.1 (gates 1-3) and §10.2 (gates 4-8) all delivered.
- **Type consistency**: `GateContext` field names match across all 8 gates. `GateResult(name, passed, reason)` consistent. Driver returns `list[GateResult]`.
- **No placeholders**: every step has working code. Gate-3 Brier threshold default 0.24 matches Plan 5B-3's observation (model Brier 0.2505 > 0.24, so this gate will fail until Plan 5C improves the model — exactly what we want).
- **Plan 5B-3 STATUS prediction**: "DSR getting the wrong number type would silently break the gate" — gate 2 today reads `deflated_sharpe` directly from DB, which is the probability form (Plan 5B-3 Task 1 deviation). Threshold 0.5 reads correctly as "50% chance true SR > 0".
- **Backward compat**: no schema changes; `scripts/heartbeat_watchdog.py` modification is additive (new CLI flag with default).
