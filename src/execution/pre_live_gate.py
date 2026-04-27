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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd
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


@dataclass(frozen=True)
class PaperRuntimeGate:
    name: str = "paper_runtime"
    min_days: int = 60
    min_fills: int = 30
    max_gap_minutes: int = 10

    def evaluate(self, ctx: GateContext) -> GateResult:
        engine = sa.create_engine(f"sqlite:///{ctx.sqlite_path}")
        with engine.begin() as conn:
            row = conn.execute(sa.text(
                "SELECT MIN(ts), MAX(ts), COUNT(*) FROM heartbeat"
            )).fetchone()
            if row is None or row[0] is None:
                return GateResult(self.name, False, "no heartbeat rows")
            min_ts, max_ts, count = row
            min_ts_dt = pd.Timestamp(min_ts).to_pydatetime()
            max_ts_dt = pd.Timestamp(max_ts).to_pydatetime()
            span_days = (max_ts_dt - min_ts_dt).total_seconds() / 86400
            # Allow up to 1 gap-interval of tolerance to handle integer row counts
            tolerance_days = self.max_gap_minutes / (24 * 60)
            if span_days < self.min_days - tolerance_days:
                return GateResult(self.name, False,
                                  f"heartbeat span {span_days:.1f}d < {self.min_days}d")
            ts_rows = conn.execute(sa.text(
                "SELECT ts FROM heartbeat ORDER BY ts ASC"
            )).fetchall()
            ts_series = pd.to_datetime([r[0] for r in ts_rows])
            max_gap_s = ts_series.to_series().diff().dt.total_seconds().max()
            if max_gap_s is not None and max_gap_s > self.max_gap_minutes * 60:
                return GateResult(self.name, False,
                                  f"heartbeat max gap {max_gap_s/60:.1f}min "
                                  f"> {self.max_gap_minutes}min")
            fills_row = conn.execute(sa.text(
                "SELECT COUNT(*) FROM broker_events "
                "WHERE kind='filled' AND ts >= :a"
            ), {"a": min_ts}).fetchone()
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
