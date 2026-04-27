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
