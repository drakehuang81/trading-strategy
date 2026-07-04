# Order Book Recon — Phase 2c (Pre-Registered Multi-Symbol Sweep) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The last cheap lottery ticket: run the two surviving hypothesis families (own-book depth@1h; BTC book → alt) through the existing four-gate harness across a **pre-registered** list of Binance alt perps, with staged replication on a disjoint window and a pre-committed closure rule.

**Architecture:** One new script (`sweep_symbols.py`) that loops the pre-registered symbols and reuses `depth_validation.build_window/summarize` and `cross_validation.build_window_cross/summarize_cross` verbatim — no new statistics, no new gates. Staged: replication window is only downloaded/run for discovery-window passers.

**Tech Stack:** unchanged. Same worktree (`recon-phase2b1`) / venv / data caches.

---

## PRE-REGISTRATION (committed before any sweep run — do not amend after results)

- **Symbols (12, fixed):** SOLUSDT, XRPUSDT, DOGEUSDT, BNBUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, LTCUSDT, DOTUSDT, MATICUSDT, ATOMUSDT, NEARUSDT. No additions mid-sweep. (MATICUSDT was rebranded ~2024-09; its replication window may be partial — per-day skips handle it, and `MIN_HOURS` guards the verdict.)
- **Hypotheses per symbol (2, fixed):** `own` = the alt's own hourly depth imbalance vs its forward 1h return; `cross` = BTCUSDT's hourly depth imbalance vs the alt's forward 1h return. 24 tests total in discovery.
- **Windows:** discovery `2023-05-16 → 2024-03-30` (comparable to the ETH baseline); replication `2024-04-01 → 2025-03-31` (fully disjoint).
- **Gates:** the existing four, unchanged (OOS same-sign & |ic_test|>0.1; beats momentum control(s) +0.05; net after 8 bps taker > 0; full-bucket monotone).
- **Pass rule:** a (symbol × hypothesis) counts ONLY if verdict == "REAL-ALPHA candidate" on **both** windows.
- **Closure commitment:** if nothing passes both windows, the "directional edge from free Binance data" question is **closed permanently** — no revisiting with new symbol lists or windows.
- **Insufficient data:** if a window yields < 1000 hours for a symbol, report INSUFFICIENT DATA (no verdict) — it neither passes nor triggers closure exceptions.

---

### Task 1: sweep runner

**Files:**
- Create: `scripts/recon/sweep_symbols.py`
- Test: `tests/research/microstructure/test_sweep_symbols.py`

- [ ] **Step 1 — create failing test** `tests/research/microstructure/test_sweep_symbols.py`:

```python
from scripts.recon.sweep_symbols import render_sweep_markdown


def test_render_sweep_markdown_flags_passes_and_insufficient():
    results = [
        {"symbol": "SOLUSDT",
         "own": {"verdict": "REAL-ALPHA candidate", "ic_test": 0.15},
         "cross": {"verdict": "FAILED — OOS", "ic_test": 0.02}},
        {"symbol": "XRPUSDT",
         "own": {"verdict": "INSUFFICIENT DATA (500h)"},
         "cross": {"verdict": "FAILED — OOS, post-cost", "ic_test": -0.01}},
    ]
    md = render_sweep_markdown(results)
    assert "SOLUSDT:own" in md                                  # flagged as pass
    assert "| SOLUSDT | REAL-ALPHA candidate | 0.150 |" in md
    assert "—" in md                                            # no ic_test cell
    assert "NONE" not in md
```

- [ ] **Step 2 — run, verify FAIL** (ImportError):
`./venv/bin/pytest tests/research/microstructure/test_sweep_symbols.py -v`

- [ ] **Step 3 — create** `scripts/recon/sweep_symbols.py`:

```python
"""Phase 2c: pre-registered multi-symbol sweep — the last cheap lottery ticket.

For each PRE-REGISTERED alt perp, run both remaining hypotheses through the
existing four-gate harness on a window:
  own   — the alt's own hourly depth imbalance vs its forward 1h return
  cross — BTCUSDT's hourly depth imbalance vs the alt's forward 1h return
Staged replication: only discovery-window passers get the disjoint replication
window; a hypothesis counts ONLY if it passes both. Closure is pre-committed
in the plan doc (docs/superpowers/plans/2026-06-29-...-phase2c-symbol-sweep.md).

Run (network — manual; discovery first, replication only for passers):
    PYTHONPATH=src venv/bin/python -m scripts.recon.sweep_symbols \
        --start 2023-05-16 --end 2024-03-30
    PYTHONPATH=src venv/bin/python -m scripts.recon.sweep_symbols \
        --start 2024-04-01 --end 2025-03-31 --symbols SOLUSDT,XRPUSDT
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from scripts.recon.cross_validation import build_window_cross, summarize_cross
from scripts.recon.depth_validation import build_window, summarize

PRE_REGISTERED = [
    "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT",
]
MIN_HOURS = 1000  # below this a window gets INSUFFICIENT DATA, not a verdict


def sweep_symbol(
    symbol: str, start: dt.date, end: dt.date, out_root: Path, cross_dir: Path
) -> dict:
    """Run both hypotheses for one symbol on one window. Never raises."""
    result: dict = {"symbol": symbol}
    own_dir = out_root / symbol
    own_dir.mkdir(parents=True, exist_ok=True)
    try:
        ds = build_window(symbol, start, end, own_dir)
        result["own"] = (
            summarize(ds) if ds.height >= MIN_HOURS
            else {"verdict": f"INSUFFICIENT DATA ({ds.height}h)"}
        )
    except Exception as e:  # noqa: BLE001 — sweep must survive any one symbol
        result["own"] = {"verdict": f"ERROR: {e}"}
    try:
        ds = build_window_cross("BTCUSDT", symbol, start, end, cross_dir)
        result["cross"] = (
            summarize_cross(ds) if ds.height >= MIN_HOURS
            else {"verdict": f"INSUFFICIENT DATA ({ds.height}h)"}
        )
    except Exception as e:  # noqa: BLE001
        result["cross"] = {"verdict": f"ERROR: {e}"}
    return result


def _ic_cell(rep: dict) -> str:
    return f"{rep['ic_test']:.3f}" if "ic_test" in rep else "—"


def render_sweep_markdown(results: list[dict]) -> str:
    lines = [
        "| symbol | own verdict | own ic_test | cross verdict | cross ic_test |",
        "|---|---|---|---|---|",
    ]
    passes: list[str] = []
    for r in results:
        own, cross = r["own"], r["cross"]
        lines.append(
            f"| {r['symbol']} | {own['verdict']} | {_ic_cell(own)} "
            f"| {cross['verdict']} | {_ic_cell(cross)} |"
        )
        for kind in ("own", "cross"):
            if r[kind]["verdict"] == "REAL-ALPHA candidate":
                passes.append(f"{r['symbol']}:{kind}")
    lines.append("")
    lines.append(
        f"**PASSES (must replicate on the disjoint window): {passes if passes else 'NONE'}**"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbols", default=",".join(PRE_REGISTERED),
                    help="comma list; replication runs pass a subset")
    ap.add_argument("--out-dir", default="data/orderbook/_sweep")
    ap.add_argument("--cross-dir", default="data/orderbook/_cross",
                    help="shared dir for BTC depth + prefixed klines (reuses 2b-2 cache)")
    args = ap.parse_args()
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    out_root, cross_dir = Path(args.out_dir), Path(args.cross_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cross_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        print(f"\n##### {sym} #####", flush=True)
        r = sweep_symbol(sym, start, end, out_root, cross_dir)
        for kind in ("own", "cross"):
            print(f"  {kind}: {r[kind]}", flush=True)
        results.append(r)

    print(f"\n=== SWEEP {args.start} -> {args.end} ===")
    print(render_sweep_markdown(results))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 — run, verify PASS**, then full scoped suite (expect 38 passed):
`./venv/bin/pytest tests/research/microstructure/ -v`

- [ ] **Step 5 — commit:**
```bash
git add scripts/recon/sweep_symbols.py tests/research/microstructure/test_sweep_symbols.py
git commit -m "$(printf 'feat(recon): pre-registered multi-symbol sweep runner (Phase 2c)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## THE SWEEP (manual, network — discovery ≈ 12 symbols × ~320 days depth+klines ≈ 2 GB, hours; restart-safe thanks to per-day caches)

```bash
# Stage 1 — discovery (all 12)
PYTHONPATH=src venv/bin/python -m scripts.recon.sweep_symbols \
    --start 2023-05-16 --end 2024-03-30

# Stage 2 — replication (ONLY discovery passers; skip if PASSES == NONE)
PYTHONPATH=src venv/bin/python -m scripts.recon.sweep_symbols \
    --start 2024-04-01 --end 2025-03-31 --symbols <passers>
```

- Passers on BOTH windows → a real lead: next step is capacity/regime analysis, then wiring into the 1h architecture.
- **NONE on both → the pre-committed closure fires**: record the final negative STATUS; the free-Binance-data directional question is closed permanently. Remaining paths are §7.2/§7.3 of the handoff (paper ops; qi maker option-preservation).
