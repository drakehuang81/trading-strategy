"""Point-in-time spot availability from spot monthly kline archives (v2).

Pre-registration: docs/superpowers/plans/2026-07-06-funding-carry-pit-preregistration.md
A spot pair traded in month M iff data/spot/monthly/klines/<SPOT>/1d/ holds
that month's zip. Missing folder = never spot-tradeable. Perp -> spot
candidates: direct BASE+USDT plus the 1000x unit-stripped base (both can
exist — 1000SATSUSDT is a real spot listing — so months are unioned).

Run (after scripts.carry.universe):
    PYTHONPATH=src venv/bin/python -m scripts.carry.spot_history --data data/carry
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from scripts.carry.spot_audit import THOUSANDS_PREFIX
from scripts.carry.universe import S3_LIST_URL, parse_keys

MONTH_FILE_RE = re.compile(r"-1d-(\d{4}-\d{2})\.zip$")


def spot_candidates(perp_symbol: str) -> list[str]:
    """Spot pairs that would hedge this perp (direct + unit-stripped)."""
    base = perp_symbol.removesuffix("USDT")
    cands = [base + "USDT"]
    stripped = THOUSANDS_PREFIX.sub("", base)
    if stripped != base:
        cands.append(stripped + "USDT")
    return cands


def parse_month_files(keys: list[str]) -> set[str]:
    """S3 keys of a spot 1d klines folder -> set of YYYY-MM traded months."""
    return {m.group(1) for k in keys if (m := MONTH_FILE_RE.search(k))}


def months_for_spot(spot_symbol: str, sess: requests.Session) -> set[str]:
    resp = sess.get(
        f"{S3_LIST_URL}?prefix=data/spot/monthly/klines/{spot_symbol}/1d/",
        timeout=30,
    )
    resp.raise_for_status()
    return parse_month_files(parse_keys(resp.text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/carry")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    data = Path(args.data)
    perps = sorted(p.stem for p in (data / "funding").glob("*.parquet"))
    print(f"resolving spot months for {len(perps)} perps")

    def job(perp: str) -> tuple[str, list[str]]:
        sess = requests.Session()
        months: set[str] = set()
        for cand in spot_candidates(perp):
            try:
                months |= months_for_spot(cand, sess)
            except Exception as e:  # noqa: BLE001 — record and continue
                print(f"  {perp}/{cand}: ERROR {e!r}")
        return perp, sorted(months)

    out: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (perp, months) in enumerate(ex.map(job, perps), 1):
            out[perp] = months
            if i % 100 == 0:
                print(f"  {i}/{len(perps)}")

    dest = data / "spot_months.json"
    dest.write_text(json.dumps(out, indent=0))
    hedged = sum(1 for m in out.values() if m)
    print(f"done -> {dest}: {hedged}/{len(perps)} perps have some spot history")


if __name__ == "__main__":
    main()
