import asyncio
import json
from pathlib import Path

import pytest

from execution.tick_recorder import TickRecorder


class FakeTradeStream:
    def __init__(self, trades):
        self._trades = trades

    async def __aiter__(self):
        for t in self._trades:
            yield t
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_records_to_daily_jsonl(tmp_path: Path):
    stream = FakeTradeStream([
        {"t": 1700000000000, "p": "2000", "q": "0.1", "m": False},
        {"t": 1700000001000, "p": "2001", "q": "0.05", "m": True},
    ])
    rec = TickRecorder(symbol="ETHUSDT", out_dir=tmp_path, stream_factory=lambda sym: stream)
    await rec.record_once_for_test()

    day_file = tmp_path / "ETHUSDT" / "2023-11-14.jsonl"
    assert day_file.exists()
    lines = day_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["p"] == "2000"
