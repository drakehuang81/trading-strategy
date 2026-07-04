"""Record Binance USD-M book/trade streams to daily JSONL — maker-research fuel.

Binance stopped publishing historical bookTicker archives in 2024-03; live
recording is the ONLY way the qi maker/HF hypothesis ever becomes testable
(see docs/handoff/current/2026-06-29-post-recon-operations.md §1b).

Binance sharded the futures websocket hosts (verified empirically 2026-07):
book streams (bookTicker / depth*) are served ONLY by the legacy endpoint
(category=None), while aggTrade is served ONLY by the "market" shard. So the
recorder runs one multiplex socket per shard, each with its own reconnect
loop, appending raw payloads to
    <out>/<kind>/<SYMBOL>/<YYYY-MM-DD>.jsonl   (UTC rollover)

Run (long-lived; Ctrl-C to stop):
    PYTHONPATH=src venv/bin/python -m scripts.record_book
    # 24/7 on a laptop: also run `caffeinate -dis &` (see docs/SANDBOX_OPS.md)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from binance import AsyncClient, BinanceSocketManager

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_KINDS = ["bookTicker", "aggTrade", "depth5@500ms"]
RECONNECT_BACKOFF_SECS = 5.0


def route_message(msg: dict) -> tuple[str, str, int]:
    """(multiplex msg) -> (kind dir, SYMBOL, event ts ms).

    kind = stream suffix with "@" made path-safe ("depth5@500ms" ->
    "depth5_500ms"). Symbol from payload "s" when present, else the stream
    prefix. Timestamp from payload event time E (fallback T, else wall clock).
    """
    prefix, suffix = msg["stream"].split("@", 1)
    kind = suffix.replace("@", "_")
    data = msg["data"]
    symbol = data.get("s") or prefix.upper()
    ts = int(data.get("E") or data.get("T") or time.time() * 1000)
    return kind, symbol, ts


def append_jsonl(root: Path, kind: str, symbol: str, ts_ms: int, payload: dict) -> Path:
    """Append one payload line to <root>/<kind>/<SYMBOL>/<date>.jsonl (UTC)."""
    day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()
    path = root / kind / symbol / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
    return path


def split_streams(symbols: list[str], kinds: list[str]) -> dict[str | None, list[str]]:
    """Group stream names by the futures ws shard that actually serves them.

    aggTrade -> "market" shard; bookTicker/depth* -> legacy endpoint (None).
    """
    groups: dict[str | None, list[str]] = {}
    for kind in kinds:
        category = "market" if kind == "aggTrade" else None
        groups.setdefault(category, []).extend(
            f"{s.lower()}@{kind}" for s in symbols
        )
    return groups


async def _record_stream_group(
    streams: list[str], category: str | None, root: Path
) -> None:
    """Reconnecting record loop over one multiplex socket on one shard."""
    while True:
        client = await AsyncClient.create()
        try:
            sock = BinanceSocketManager(client).futures_multiplex_socket(
                streams, category=category,
            )
            async with sock as stream:
                print(f"recording {streams} (category={category!r}) -> {root}")
                while True:
                    msg = await stream.recv()
                    if not isinstance(msg, dict) or "stream" not in msg:
                        continue  # error/control frames — socket layer handles retry
                    kind, symbol, ts = route_message(msg)
                    append_jsonl(root, kind, symbol, ts, msg["data"])
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — recorder must outlive transient errors
            print(f"stream error ({e!r}); reconnecting in {RECONNECT_BACKOFF_SECS}s")
            await asyncio.sleep(RECONNECT_BACKOFF_SECS)
        finally:
            # bound the close: an unresponsive ws must not wedge shutdown
            try:
                await asyncio.wait_for(client.close_connection(), timeout=5)
            except Exception:  # noqa: BLE001
                pass


async def record(symbols: list[str], kinds: list[str], root: Path) -> None:
    """Run one recording loop per required shard, concurrently."""
    groups = split_streams(symbols, kinds)
    await asyncio.gather(*(
        _record_stream_group(streams, category, root)
        for category, streams in groups.items()
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    ap.add_argument("--out", default="data/ticks")
    args = ap.parse_args()
    asyncio.run(record(args.symbols.split(","), args.kinds.split(","), Path(args.out)))


if __name__ == "__main__":
    main()
