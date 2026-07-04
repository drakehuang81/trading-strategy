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
