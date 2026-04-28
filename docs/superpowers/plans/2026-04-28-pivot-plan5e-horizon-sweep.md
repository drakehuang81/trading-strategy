# Plan 5E — Horizon Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sweep XGBoost training across horizons `H ∈ {4, 24, 48, 96}` to test the hypothesis "4-bar horizon is too short to capture signal in our feature stack." For each horizon: build labels → retrain → record Brier. Then backtest each model. Compare results to determine whether longer horizons recover edge OR conclude that single-asset / single-timeframe ETHUSDT 1h has no edge regardless (spec §12 red flag confirmed).

**Architecture:** One automation script (`scripts/sweep_horizons.py`) that loops over horizons, shells out to existing `build_labels.py` and `train_xgb.py`, and prints a comparison table. Per-horizon backtests stay manual (`scripts/backtest.py`). No production code changes — pure experiment scaffolding.

**Tech Stack:** Python 3.11, subprocess, pandas. No new dependencies. Reuses Plan 5A `build_training_set.py` features parquet (independent of horizon — only labels and the trained model differ).

**Decisions baked in:**
- **Forward-up labels only** for the sweep — Plan 5C's TB labels gave ≈ same Brier (0.2510 vs 0.2505), so adding TB to the matrix doubles compute for no diagnostic value. If 5E shows a horizon sweet spot, Plan 5F can re-test TB at that horizon.
- **Horizons `[4, 24, 48, 96]`** — 4 is current baseline; 24 = 1 day; 48 = 2 days; 96 = 4 days. Spread across log-scale to find the inflection if any.
- **Per-horizon model bundles** under `models/sweep_h<N>/` subdirs — keeps the sweep separate from the production `models/` directory which Pre-Live Gate reads. After sweep, the operator manually copies the winner (if any) to `models/`.
- **No Pre-Live Gate run** in this plan. The gate's `calibration_brier` threshold (0.24) is the diagnostic we're checking against per-horizon, but no model gets promoted automatically.
- **Backtests use loose threshold 0.51** since Plan 5C confirmed all models give 0 trades at 0.58. We're comparing *relative* model quality, not chasing the spec's production threshold.
- **No code changes to `train_xgb.py` or `build_labels.py`** — they already accept `--horizon` and `--out` flags. The sweep script just iterates.
- **Sweep is reproducible**: same random seed inside `train_xgb.py`, same features parquet, only labels parquet changes. Re-running the sweep on identical inputs produces identical model_versions.

**Out of Plan 5E scope (deferred):**
- Triple-barrier × horizon matrix (Plan 5F if 5E shows a sweet spot).
- Different feature families (volume profile, on-chain) — Plan 5G.
- Multi-symbol — Plan 5H.
- Walk-forward Brier comparison (this plan uses single-fold from train_xgb's last walk-forward fold).
- Model promotion automation — operator manually picks winner.
- `XGBPredictor.horizon_bars` field plumbing — currently hardcoded to 4; if 5E picks a non-4 winner, Plan 5F adds the metadata flow.

---

## File map

### Created
- `scripts/sweep_horizons.py` — sweep automation
- `tests/unit/scripts/test_sweep_horizons.py`
- `docs/superpowers/plans/2026-04-28-pivot-plan5e-STATUS.md` (handoff)

### Modified
- None.

### Untouched (verified intentionally)
- `scripts/build_labels.py` — already accepts `--horizon` (Plan 5A Task 7 + Plan 5C extension).
- `scripts/train_xgb.py` — auto-detects label column from parquet.
- `src/labels/triple_barrier.py` — not used in this sweep.
- `models/` production dir — sweep writes to `models/sweep_h<N>/` subdirs, never overwrites.
- `data/state.db` — sweep doesn't insert backtest_runs rows; per-horizon backtests do (manual).

---

## Task 1: `scripts/sweep_horizons.py`

**Why:** Automates the boilerplate of "build labels at H, train, parse Brier, repeat" so the operator runs one command to get the comparison table.

**Files:**
- Create: `scripts/sweep_horizons.py`
- Create: `tests/unit/scripts/test_sweep_horizons.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/scripts/test_sweep_horizons.py
"""Horizon sweep automation — Plan 5E Task 1."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.sweep_horizons import (
    SweepResult,
    parse_brier_from_meta,
    run_sweep,
)


def test_parse_brier_picks_chosen_calibration_method(tmp_path):
    """parse_brier_from_meta returns the brier of whichever calibrator
    train_xgb chose (not always isotonic)."""
    meta = tmp_path / "meta_v1.json"
    meta.write_text(json.dumps({
        "model_version": "v1",
        "calibration_method": "platt",
        "brier_isotonic": 0.27,
        "brier_platt": 0.23,
    }))
    chosen, version, calib = parse_brier_from_meta(meta)
    assert chosen == 0.23
    assert version == "v1"
    assert calib == "platt"


def test_parse_brier_handles_isotonic_choice(tmp_path):
    meta = tmp_path / "meta_v2.json"
    meta.write_text(json.dumps({
        "model_version": "v2",
        "calibration_method": "isotonic",
        "brier_isotonic": 0.21,
        "brier_platt": 0.25,
    }))
    chosen, version, calib = parse_brier_from_meta(meta)
    assert chosen == 0.21
    assert calib == "isotonic"


def test_run_sweep_invokes_build_labels_and_train_per_horizon(tmp_path, monkeypatch):
    """For each horizon, run_sweep calls build_labels then train_xgb,
    collects the resulting meta JSON's chosen Brier."""
    kline = tmp_path / "kline.parquet"
    features = tmp_path / "features.parquet"
    out_root = tmp_path / "sweep_models"
    sqlite = tmp_path / "state.db"
    kline.touch()
    features.touch()

    def fake_run(cmd, **kwargs):
        # Identify whether this was a build_labels or train_xgb call.
        if "build_labels.py" in " ".join(cmd):
            # find horizon from cmd
            h = int(cmd[cmd.index("--horizon") + 1])
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"")  # placeholder
        elif "train_xgb.py" in " ".join(cmd):
            out_dir = Path(cmd[cmd.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "meta_fake.json").write_text(json.dumps({
                "model_version": "fake",
                "calibration_method": "platt",
                "brier_isotonic": 0.27,
                "brier_platt": 0.24,
            }))
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.sweep_horizons.subprocess.run", fake_run)

    results = run_sweep(
        horizons=[4, 24],
        kline=kline,
        features=features,
        out_root=out_root,
        sqlite_path=sqlite,
    )
    assert len(results) == 2
    assert results[0].horizon == 4
    assert results[1].horizon == 24
    assert all(isinstance(r, SweepResult) for r in results)
    assert all(r.brier == 0.24 for r in results)
    assert all(r.calibration_method == "platt" for r in results)


def test_run_sweep_continues_on_train_failure(tmp_path, monkeypatch):
    """If train fails for one horizon, continue with the rest; mark that result with brier=None."""
    kline = tmp_path / "kline.parquet"
    features = tmp_path / "features.parquet"
    out_root = tmp_path / "sweep_models"
    sqlite = tmp_path / "state.db"
    kline.touch()
    features.touch()

    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        from types import SimpleNamespace
        if "train_xgb.py" in " ".join(cmd):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First training call fails (e.g. for horizon=4).
                return SimpleNamespace(returncode=1, stdout="", stderr="boom")
            # Second call succeeds.
            out_dir = Path(cmd[cmd.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "meta_ok.json").write_text(json.dumps({
                "model_version": "ok",
                "calibration_method": "isotonic",
                "brier_isotonic": 0.22,
                "brier_platt": 0.25,
            }))
        else:
            # build_labels: write a placeholder file.
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.sweep_horizons.subprocess.run", fake_run)

    results = run_sweep(
        horizons=[4, 24],
        kline=kline,
        features=features,
        out_root=out_root,
        sqlite_path=sqlite,
    )
    assert results[0].brier is None
    assert "train" in results[0].error.lower()
    assert results[1].brier == 0.22
    assert results[1].error is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/scripts/test_sweep_horizons.py -v`
Expected: ImportError on `scripts.sweep_horizons`.

- [ ] **Step 3: Implement `sweep_horizons.py`**

```python
# scripts/sweep_horizons.py
"""Horizon sweep — Plan 5E Task 1.

For each horizon H in --horizons:
  1. build_labels.py --horizon H --out <out>/labels_h<H>.parquet
  2. train_xgb.py --labels <out>/labels_h<H>.parquet --out <out>/h<H>/
  3. parse the trained meta_*.json for the chosen calibrator's Brier
Print a comparison table at the end.

Usage:
    python -m scripts.sweep_horizons \
        --kline data/history/ETHUSDT_1h.parquet \
        --features data/training/ETHUSDT_1h_features.parquet \
        --horizons 4,24,48,96 \
        --out-root models/sweep \
        --sqlite-path data/state.db
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SweepResult:
    horizon: int
    model_version: str | None
    calibration_method: str | None
    brier: float | None
    error: str | None = None


def parse_brier_from_meta(meta_path: Path) -> tuple[float, str, str]:
    """Returns (chosen_brier, model_version, calibration_method)."""
    meta = json.loads(meta_path.read_text())
    method = meta["calibration_method"]
    brier = float(meta[f"brier_{method}"])
    return brier, meta["model_version"], method


def run_sweep(
    *,
    horizons: list[int],
    kline: Path,
    features: Path,
    out_root: Path,
    sqlite_path: Path,
) -> list[SweepResult]:
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[SweepResult] = []
    for h in horizons:
        labels_path = out_root / f"labels_h{h}.parquet"
        model_dir = out_root / f"h{h}"
        # Step 1: build labels.
        rc = subprocess.run([
            "python", "scripts/build_labels.py",
            "--kline", str(kline),
            "--out", str(labels_path),
            "--horizon", str(h),
        ], capture_output=True, text=True)
        if rc.returncode != 0:
            results.append(SweepResult(
                horizon=h, model_version=None, calibration_method=None,
                brier=None, error=f"build_labels failed: {rc.stderr.strip()}",
            ))
            continue

        # Step 2: train.
        rc = subprocess.run([
            "python", "scripts/train_xgb.py",
            "--features", str(features),
            "--labels", str(labels_path),
            "--out", str(model_dir),
            "--sqlite-path", str(sqlite_path),
        ], capture_output=True, text=True)
        if rc.returncode != 0:
            results.append(SweepResult(
                horizon=h, model_version=None, calibration_method=None,
                brier=None, error=f"train_xgb failed: {rc.stderr.strip()}",
            ))
            continue

        # Step 3: parse latest meta JSON in model_dir.
        meta_files = sorted(model_dir.glob("meta_*.json"),
                            key=lambda p: p.stat().st_mtime)
        if not meta_files:
            results.append(SweepResult(
                horizon=h, model_version=None, calibration_method=None,
                brier=None, error="no meta JSON written by train_xgb",
            ))
            continue
        brier, version, method = parse_brier_from_meta(meta_files[-1])
        results.append(SweepResult(
            horizon=h, model_version=version,
            calibration_method=method, brier=brier, error=None,
        ))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--horizons", required=True, type=str,
                    help="comma-separated horizons, e.g. 4,24,48,96")
    ap.add_argument("--out-root", default=Path("models/sweep"), type=Path)
    ap.add_argument("--sqlite-path", default="data/state.db")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    results = run_sweep(
        horizons=horizons,
        kline=args.kline,
        features=args.features,
        out_root=args.out_root,
        sqlite_path=Path(args.sqlite_path),
    )

    print(f"\nHorizon sweep — {len(results)} runs\n")
    print(f"{'H':<6} {'BRIER':<8} {'CALIB':<10} {'MODEL':<14} ERROR")
    print("-" * 80)
    for r in results:
        b = f"{r.brier:.4f}" if r.brier is not None else "—"
        v = r.model_version or "—"
        c = r.calibration_method or "—"
        e = r.error or ""
        print(f"{r.horizon:<6} {b:<8} {c:<10} {v:<14} {e}")
    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/scripts/test_sweep_horizons.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 373 passed (369 + 4 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/sweep_horizons.py tests/unit/scripts/test_sweep_horizons.py
git commit -m "feat(scripts): sweep_horizons.py — automate label-build + train across N horizons"
```

---

## Task 2: Manual smoke — run the horizon sweep + per-horizon backtests

**Files:** No code changes; only commands and observation.

- [ ] **Step 1: Run the sweep (~20 minutes for 4 horizons)**

```bash
PYTHONPATH=src python -m scripts.sweep_horizons \
    --kline data/history/ETHUSDT_1h.parquet \
    --features data/training/ETHUSDT_1h_features.parquet \
    --horizons 4,24,48,96 \
    --out-root models/sweep \
    --sqlite-path data/state.db
```

Expected stdout: a table with 4 rows showing `H | Brier | calib | model_version`. Each train_xgb writes a model_versions row to SQLite as a side-effect (harmless — Plan 5C has historical rows already).

Note the table contents in STATUS.

- [ ] **Step 2: Backtest each horizon's model**

For each horizon directory under `models/sweep/h<N>/`, run a backtest at the loose 0.51 threshold (so we get non-zero trades regardless of model strength):

```bash
for h in 4 24 48 96; do
  echo "=== horizon=$h ==="
  PYTHONPATH=src python -m scripts.backtest \
    --kline data/history/ETHUSDT_1h.parquet \
    --features data/training/ETHUSDT_1h_features.parquet \
    --funding data/funding/ETHUSDT.parquet \
    --model-dir models/sweep/h$h \
    --sqlite-path data/state.db \
    --oos-fraction 0.2 \
    --long-threshold 0.51 --short-threshold 0.51 \
    --n-trials 4
done
```

`--n-trials 4` reflects the actual count of model variants tried in this experiment (4 horizons), not the default 2. This deflates DSR appropriately.

Note each horizon's `sharpe / dsr / n_trades` in STATUS.

- [ ] **Step 3: Inspect `backtest_runs` for the 4 new rows**

```bash
source venv/bin/activate
python -c "
import sqlalchemy as sa, json
engine = sa.create_engine('sqlite:///data/state.db')
with engine.begin() as conn:
    rows = conn.execute(sa.text(
        'SELECT run_id, deflated_sharpe, summary_json FROM backtest_runs '
        'ORDER BY started_at DESC LIMIT 4'
    )).fetchall()
for r in rows:
    s = json.loads(r[2])
    print(f\"run_id={r[0]} dsr={r[1]:.4f} sharpe={s.get('sharpe'):.4f if s.get('sharpe') else 'None'} \"
          f\"trades={s['n_trades']} model={s['ml_model_version']}\")
"
```

Cross-reference each model_version with the sweep table to know which horizon each row represents.

---

## Task 3: STATUS handoff

**Files:**
- Create: `docs/superpowers/plans/2026-04-28-pivot-plan5e-STATUS.md`

- [ ] **Step 1: Write Plan 5E STATUS**

Create `docs/superpowers/plans/2026-04-28-pivot-plan5e-STATUS.md`. Sections:
- Date / branch / base / head SHAs (from `git log --oneline 5f3241c..HEAD`)
- Summary (lead with: did any horizon get Brier below 0.24? did any backtest DSR jump?)
- Task table (3 rows)
- Manual smoke results — full sweep table:
  - Per horizon: Brier, chosen calibrator, model_version
  - Per horizon backtest: Sharpe, DSR, n_trades
- Verdict: pick one of the three outcomes (write only the matching one):
  - **(A) Found a sweet spot**: e.g., H=24 has Brier 0.23 + Sharpe 1.5+ at default threshold. Plan 5F should harden that horizon (TB labels, gap=horizon CV, multi-symbol).
  - **(B) Marginal improvement**: Brier dropped slightly for some H but never below 0.24. Plan 5F still worth trying with TB + the best H.
  - **(C) No sweet spot**: All horizons give Brier ≈ 0.25, all backtest Sharpes within noise. Spec §12 red flag confirmed. **Recommend Plan 5G feature engineering OR pivot strategy concept.**
- Decisions landed (n_trials=4 in backtest, loose threshold 0.51, no production model promotion)
- What is NOT done (TB × horizon matrix, multi-symbol, new features) — Plan 5F+
- Known follow-ups (`XGBPredictor.horizon_bars` field plumbing, walk-forward Brier across folds rather than single fold)

- [ ] **Step 2: Final commit**

```bash
git add docs/superpowers/plans/2026-04-28-pivot-plan5e-STATUS.md
git commit -m "docs: Plan 5E handoff STATUS — horizon sweep results"
```

---

## Self-review notes

- **Spec coverage**: spec doesn't prescribe horizon sweep specifically. Plan addresses the diagnostic gap left by Plan 5C ("is it features or label noise?") by isolating horizon as the variable.
- **Type consistency**: `SweepResult` fields stable across the test cases. `parse_brier_from_meta` returns a 3-tuple (chosen_brier, version, method) — same shape across both tests.
- **No placeholders**: every step has working code; manual smoke commands are concrete with expected outputs.
- **Backward compat**: production `models/` dir is untouched. `models/sweep/h<N>/` is new sub-tree. The existing 4 model bundles (Plan 5A, 5B-1, 5C, plus the test residue) stay intact in `models/`.
- **Sweep idempotency**: re-running `sweep_horizons.py` on identical inputs writes deterministic model_versions to deterministic paths. Re-runs overwrite the previous run's parquets/models in place (safe; sweep dir is dedicated).
- **Compute budget**: 4 horizons × ~5 minutes per training = ~20 minutes total CPU. Acceptable.
