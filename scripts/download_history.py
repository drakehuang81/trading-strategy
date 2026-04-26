"""Downloads 2 years of ETHUSDT 1h klines + funding history into Parquet
(Plan 5A Task 5).

Usage:
    python scripts/download_history.py \
        --symbol ETHUSDT --timeframe 1h --years 2 \
        --out-dir data/history --funding-out-dir data/funding

Idempotent: re-running upserts new bars onto the existing Parquet.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data.binance_kline import BinanceKline
from data.funding import FundingRateWriter

_TF_TO_TIMEDELTA = {
    "1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15), "1h": timedelta(hours=1),
    "4h": timedelta(hours=4), "1d": timedelta(days=1),
}


async def fetch_klines_paginated(
    source: BinanceKline,
    symbol: str,
    timeframe: str,
    since: datetime,
    until: datetime,
) -> pd.DataFrame:
    step = _TF_TO_TIMEDELTA[timeframe]
    cursor = since
    parts: list[pd.DataFrame] = []
    while cursor < until:
        chunk = await source.fetch(symbol, timeframe, cursor, until)
        if chunk.empty:
            break
        parts.append(chunk)
        last_ts = chunk.index.max()
        next_cursor = (last_ts + step).to_pydatetime()
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def upsert_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = df
    combined.to_parquet(path)


async def main_async(args: argparse.Namespace) -> None:
    until = datetime.now(tz=timezone.utc)
    since = until - timedelta(days=365 * args.years)

    source = await BinanceKline.open()
    try:
        klines = await fetch_klines_paginated(
            source, args.symbol, args.timeframe, since, until,
        )
        kline_path = Path(args.out_dir) / f"{args.symbol}_{args.timeframe}.parquet"
        upsert_parquet(kline_path, klines)
        print(f"klines: {len(klines)} bars -> {kline_path}")

        funding_dir = Path(args.funding_out_dir)
        funding_writer = FundingRateWriter(client=source.client, out_dir=funding_dir)
        funding_path = funding_dir / f"{args.symbol}.parquet"

        # Decide: backfill if parquet is missing OR doesn't reach `since`.
        needs_backfill = True
        if funding_path.exists():
            from data.funding import load_funding
            existing = load_funding(funding_path)
            if not existing.empty and existing.index.min().to_pydatetime() <= since:
                needs_backfill = False

        if needs_backfill:
            backfilled = await funding_writer.backfill(args.symbol, since=since)
            print(f"funding backfill: {backfilled} rows fetched")

        added = await funding_writer.update(args.symbol)
        print(f"funding update: {added} new rows -> {funding_path}")
    finally:
        await source.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--timeframe", default="1h", choices=list(_TF_TO_TIMEDELTA))
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--out-dir", default="data/history")
    ap.add_argument("--funding-out-dir", default="data/funding")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
