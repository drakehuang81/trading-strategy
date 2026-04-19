"""TickRecorder — spec §4.9. Fuel for ReplayBroker.

Subscribes to Binance trades WS, appends raw ticks to
data/ticks/<symbol>/<YYYY-MM-DD>.jsonl. Rollover at UTC midnight.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable


class TickRecorder:
    def __init__(
        self,
        symbol: str,
        out_dir: Path,
        stream_factory: Callable[[str], AsyncIterator[dict[str, Any]]],
    ) -> None:
        self.symbol = symbol
        self.out_dir = out_dir / symbol
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._stream_factory = stream_factory

    async def run(self) -> None:
        """Run forever. Orchestrator owns this task's lifecycle."""
        stream = self._stream_factory(self.symbol)
        async for tick in stream:
            self._append(tick)

    async def record_once_for_test(self) -> None:
        stream = self._stream_factory(self.symbol)
        async for tick in stream:
            self._append(tick)

    def _append(self, tick: dict[str, Any]) -> None:
        ts = datetime.fromtimestamp(tick["t"] / 1000, tz=timezone.utc)
        path = self.out_dir / f"{ts.date().isoformat()}.jsonl"
        with path.open("a") as fh:
            fh.write(json.dumps(tick, separators=(",", ":")) + "\n")
