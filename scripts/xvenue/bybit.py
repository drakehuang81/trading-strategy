"""Bybit v5 funding-history downloader for the cross-venue spread study.

Pre-registration: docs/superpowers/plans/2026-07-06-xvenue-funding-spread-preregistration.md
Universe = canonical-key intersection of local Binance funding parquets
(survivorship-free) and Bybit's CURRENT linear-USDT perp list (Bybit-side
survivorship bias disclosed in the doc — no free fix).

Canonical key: strip 1000-style multiplier groups from either end of the
base (Binance 1000SHIBUSDT <-> Bybit SHIB1000USDT -> SHIB). Funding rates
are percentages, so unit multipliers don't affect the spread.

Run (after carry data exists):
    PYTHONPATH=src venv/bin/python -m scripts.xvenue.bybit --out data/xvenue
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import polars as pl
import requests

BYBIT = "https://api.bybit.com"
SINCE_MS = 1_590_969_600_000  # 2020-06-01 UTC — warmup before replication window
_LEAD = re.compile(r"^1(?:0+)")
_TAIL = re.compile(r"1(?:0+)$")


def canonical_key(symbol: str) -> str:
    """1000-multiplier-agnostic base key for a USDT perp symbol.

    1000SHIBUSDT -> SHIB; SHIB1000USDT -> SHIB; 1INCHUSDT -> 1INCH
    (a lone '1' with no zeros is not a multiplier); PORT3USDT -> PORT3.
    """
    base = symbol.removesuffix("USDT")
    return _TAIL.sub("", _LEAD.sub("", base))


def unique_key_map(symbols: list[str]) -> dict[str, str]:
    """canonical key -> symbol; keys colliding within the venue are dropped."""
    seen: dict[str, list[str]] = {}
    for s in symbols:
        seen.setdefault(canonical_key(s), []).append(s)
    return {k: v[0] for k, v in seen.items() if len(v) == 1}


def parse_funding_rows(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_ms": [int(r["fundingRateTimestamp"]) for r in rows],
            "rate": [float(r["fundingRate"]) for r in rows],
        },
        schema={"ts_ms": pl.Int64, "rate": pl.Float64},
    )


def list_linear_usdt(sess: requests.Session) -> list[str]:
    """All current Bybit linear-perpetual USDT symbols (paginated)."""
    out: list[str] = []
    cursor = ""
    while True:
        resp = sess.get(
            f"{BYBIT}/v5/market/instruments-info",
            params={"category": "linear", "limit": 1000, "cursor": cursor or None},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        out.extend(
            i["symbol"]
            for i in result["list"]
            if i.get("quoteCoin") == "USDT"
            and i.get("contractType") == "LinearPerpetual"
        )
        cursor = result.get("nextPageCursor") or ""
        if not cursor:
            return sorted(set(out))


def fetch_funding(symbol: str, sess: requests.Session, since_ms: int = SINCE_MS) -> pl.DataFrame:
    """Backward-paginate funding history (desc pages of <=200) down to since_ms."""
    frames = []
    end = int(time.time() * 1000)
    while True:
        resp = sess.get(
            f"{BYBIT}/v5/market/funding/history",
            params={"category": "linear", "symbol": symbol,
                    "startTime": since_ms, "endTime": end, "limit": 200},
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()["result"]["list"]
        if not rows:
            break
        frames.append(parse_funding_rows(rows))
        oldest = min(int(r["fundingRateTimestamp"]) for r in rows)
        if oldest <= since_ms or len(rows) < 200:
            break
        end = oldest - 1
    if not frames:
        return pl.DataFrame(schema={"ts_ms": pl.Int64, "rate": pl.Float64})
    return pl.concat(frames).unique(subset=["ts_ms"], keep="last").sort("ts_ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/xvenue")
    ap.add_argument("--binance", default="data/carry/funding")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "bybit").mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    bnb_map = unique_key_map(sorted(p.stem for p in Path(args.binance).glob("*.parquet")))
    byb_map = unique_key_map(list_linear_usdt(sess))
    common = sorted(set(bnb_map) & set(byb_map))
    mapping = {k: {"binance": bnb_map[k], "bybit": byb_map[k]} for k in common}
    (out / "mapping.json").write_text(json.dumps(mapping, indent=0))
    print(f"binance keys {len(bnb_map)}, bybit keys {len(byb_map)}, common {len(common)}")

    done = 0
    for i, key in enumerate(common, 1):
        dest = out / "bybit" / f"{byb_map[key]}.parquet"
        if dest.exists():
            continue
        try:
            df = fetch_funding(byb_map[key], sess)
            if len(df):
                df.write_parquet(dest)
                done += 1
        except Exception as e:  # noqa: BLE001 — record and continue
            print(f"  {byb_map[key]}: ERROR {e!r}")
        if i % 25 == 0:
            print(f"  {i}/{len(common)}")
        time.sleep(0.12)  # stay well under public rate limits
    print(f"done: {done} new symbol files")


if __name__ == "__main__":
    main()
