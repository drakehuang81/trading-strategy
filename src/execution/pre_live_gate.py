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
