"""Universe enumeration + funding history download for the carry study.

Pre-registration: docs/superpowers/plans/2026-07-05-funding-carry-preregistration.md
Universe = S3 monthly fundingRate folder listing (survivorship-FREE: the
bucket keeps folders for delisted contracts), filtered to USDT-quoted.
Data = monthly zips -> one parquet per symbol: (ts_ms, interval_h, rate).

Run:
    PYTHONPATH=src venv/bin/python -m scripts.carry.universe --out data/carry
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import requests

S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FUNDING_PREFIX = "data/futures/um/monthly/fundingRate/"
DOWNLOAD_BASE = "https://data.binance.vision/"
# full archive range (archives begin 2020-01); trailing/seasoning warm up
# inside the data, so download everything — the whole universe is only ~MBs.
FIRST_MONTH, LAST_MONTH = "2020-01", "2026-06"
MONTH_RE = re.compile(r"-fundingRate-(\d{4}-\d{2})\.zip$")
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def parse_prefixes(xml: str) -> list[str]:
    return re.findall(r"<Prefix>([^<]+)</Prefix>", xml)


def parse_keys(xml: str) -> list[str]:
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def next_marker(xml: str) -> str | None:
    """S3 v1 pagination: NextMarker when truncated (present with delimiter)."""
    if "<IsTruncated>true</IsTruncated>" not in xml:
        return None
    m = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
    return m.group(1) if m else None


def usdt_symbols_from_prefixes(prefixes: list[str]) -> list[str]:
    """Folder prefixes -> USDT-quoted symbol names (drops USDC/BUSD legs)."""
    out = []
    for p in prefixes:
        if p == FUNDING_PREFIX:
            continue
        sym = p.rstrip("/").rsplit("/", 1)[-1]
        if sym.endswith("USDT"):
            out.append(sym)
    return out


def list_symbols(sess: requests.Session) -> list[str]:
    """Enumerate every fundingRate symbol folder (paginated), USDT only."""
    symbols: list[str] = []
    marker = ""
    while True:
        url = f"{S3_LIST_URL}?delimiter=/&prefix={FUNDING_PREFIX}"
        if marker:
            url += f"&marker={marker}"
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()
        xml = resp.text
        symbols.extend(usdt_symbols_from_prefixes(parse_prefixes(xml)))
        nm = next_marker(xml)
        if nm is None:
            break
        marker = nm
    return sorted(set(symbols))


def month_zip_keys(symbol: str, sess: requests.Session) -> list[str]:
    """Zip keys for one symbol folder, filtered to the archive window."""
    resp = sess.get(f"{S3_LIST_URL}?prefix={FUNDING_PREFIX}{symbol}/", timeout=30)
    resp.raise_for_status()
    keys = []
    for k in parse_keys(resp.text):
        m = MONTH_RE.search(k)
        if m and FIRST_MONTH <= m.group(1) <= LAST_MONTH:
            keys.append(k)
    return sorted(keys)


def read_funding_csv(raw: bytes) -> pl.DataFrame:
    """One monthly funding CSV -> (ts_ms, interval_h, rate).

    Sampled files have a header row; guard the headerless case anyway
    (recon lesson: never assume archive header conventions are uniform).
    """
    has_header = raw.lstrip()[:9] == b"calc_time"
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=has_header,
        infer_schema_length=0,  # read as str; cast explicitly below
    )
    if len(df.columns) == 2:  # tolerate an interval-less variant
        df = df.rename(dict(zip(df.columns, ["calc_time", "last_funding_rate"])))
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("funding_interval_hours"))
    elif not has_header:
        df = df.rename(dict(zip(df.columns, FUNDING_COLS)))
    return (
        df.select(
            pl.col("calc_time").cast(pl.Int64).alias("ts_ms"),
            pl.col("funding_interval_hours").cast(pl.Int64, strict=False).alias("interval_h"),
            pl.col("last_funding_rate").cast(pl.Float64).alias("rate"),
        )
        .drop_nulls(subset=["ts_ms", "rate"])
    )


def fetch_symbol(symbol: str, out_dir: Path, sess: requests.Session) -> int:
    """Download all monthly zips for one symbol -> <out>/<SYMBOL>.parquet."""
    dest = out_dir / f"{symbol}.parquet"
    if dest.exists():
        return -1  # cached from a previous run
    frames = []
    for key in month_zip_keys(symbol, sess):
        resp = sess.get(DOWNLOAD_BASE + key, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"expected 1 csv in {key}, found {names}")
            frames.append(read_funding_csv(zf.read(names[0])))
    if not frames:
        return 0
    df = (
        pl.concat(frames)
        .unique(subset=["ts_ms"], keep="last")
        .sort("ts_ms")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/carry")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    funding_dir = out / "funding"

    sess = requests.Session()
    symbols = list_symbols(sess)
    print(f"universe: {len(symbols)} USDT-quoted fundingRate folders")

    def job(sym: str) -> tuple[str, int | str]:
        try:
            # one Session per call is thread-safe enough at this scale
            return sym, fetch_symbol(sym, funding_dir, requests.Session())
        except Exception as e:  # noqa: BLE001 — record and keep sweeping
            return sym, f"ERROR: {e!r}"

    results: dict[str, int | str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (sym, res) in enumerate(ex.map(job, symbols), 1):
            results[sym] = res
            if i % 50 == 0:
                print(f"  {i}/{len(symbols)} done")

    errors = {s: r for s, r in results.items() if isinstance(r, str)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "universe_snapshot.json").write_text(
        json.dumps({"symbols": symbols, "rows": results}, indent=1, default=str)
    )
    print(f"done: {len(symbols) - len(errors)} ok, {len(errors)} errors")
    for s, r in sorted(errors.items()):
        print(f"  {s}: {r}")


if __name__ == "__main__":
    main()
