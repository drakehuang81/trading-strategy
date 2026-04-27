"""Pre-Live Gate (spec §10).

Mode can flip from paper to live only when ALL 8 gates below are green.
Each gate is a stateless function from `GateContext` to `GateResult`.
The driver runs every gate (no short-circuit) so operators see every
red at once.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import sqlalchemy as sa


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
