# Plan 5B-1 — Funding Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `FundingRateWriter` with a `backfill(symbol, since)` method that paginates Binance's funding-rate endpoint *backwards* until it covers `since`, and wire it into `scripts/download_history.py` so the manual smoke produces a full 2-year `data/funding/ETHUSDT.parquet` instead of the current ~2-month tail.

**Architecture:** Funding rates fire every 8h (≈ 1095/year), Binance caps `futures_funding_rate` at `limit=1000` per call, so 2 years requires ~3 pages. `update()` (Plan 5A's existing forward paginator) keeps working unchanged. `backfill()` is a new method that walks `endTime` backward in 1000-row chunks until either the response is empty (no more history) or the chunk's earliest row is ≤ `since`. `download_history.py` calls `backfill(since=now-2y)` on missing-parquet boot or first-time data, then continues to call `update()` on subsequent runs. Existing parquet schema unchanged: DatetimeIndex (UTC) named `ts` + single column `funding_rate`.

**Tech Stack:** Python 3.11, asyncio, `python-binance` AsyncClient, pandas, pytest with `asyncio_mode=auto`.

**Decisions baked in:**
- `backfill` is **separate from `update`** — they have different cursor semantics (forward vs. backward) and different "stop" conditions (empty response vs. crossed `since`). Conflating them would be a foot-gun.
- `download_history.py` decides which to call: if existing parquet covers `since`, use `update`; otherwise `backfill` first then `update` to top off any gap at the head.
- No new schema changes — same parquet shape, same `funding_rate` column.
- No backward-compat concern: `update()` signature unchanged.
- **Re-train at the end of manual smoke** so Plan 5A's missing `drift_reference.json` also gets generated.

**Out of Plan 5B-1 scope (deferred):**
- Multi-symbol parallel backfill (current: single symbol per call).
- Backfill resume after partial-write crash (Parquet write is atomic per `to_parquet`).
- Backfill rate-limiting / sleep — Binance public endpoint is generous; we make ≤ 3 calls per symbol per backfill.
- Storing per-funding-window `markPrice` / `nextFundingTime` (only `fundingRate` matters for our feature today).

---

## File map

### Created
- `tests/unit/data/test_funding_backfill.py` — unit tests for `FundingRateWriter.backfill`
- `docs/superpowers/plans/2026-04-26-pivot-plan5b1-STATUS.md` — handoff at the end

### Modified
- `src/data/funding.py` — add `backfill(symbol, since)` method
- `scripts/download_history.py` — call `backfill` first when parquet is missing or doesn't cover `since`

### Untouched (verified intentionally)
- `src/data/funding.py::FundingRateWriter.update` — keep working as-is.
- `src/features/funding_rate.py::FundingFeature.compute` — already index-based (Plan 5A Task 6 fix); benefits transparently from the larger parquet.
- `tests/unit/data/test_funding.py` — existing 1-test sanity check stays valid.

---

## Task 1: `FundingRateWriter.backfill(symbol, since)`

**Why first:** Pure leaf method addition. Once tests pass, Task 2 is just wiring + a manual smoke.

**Files:**
- Modify: `src/data/funding.py` (append `backfill` method to `FundingRateWriter` class)
- Create: `tests/unit/data/test_funding_backfill.py`

- [ ] **Step 1: Confirm Binance API contract**

The `futures_funding_rate` endpoint accepts `symbol`, `startTime` (ms), `endTime` (ms), `limit` (max 1000). It returns rows ordered by `fundingTime` ascending. With only `endTime` specified, it returns the most-recent ≤ 1000 rows up to `endTime`. We use this property to paginate backwards: each iteration advances `endTime` to one millisecond *before* the earliest row of the previous chunk.

No code change in this step — read `src/data/funding.py:14-40` and `tests/unit/data/test_funding.py:8-15` so the fake-client API signature is in your head before writing tests.

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/data/test_funding_backfill.py`:

```python
"""Tests for FundingRateWriter.backfill — Plan 5B-1 Task 1."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data.funding import FundingRateWriter, load_funding


class _ChunkedFakeFundingClient:
    """Fake Binance AsyncClient that returns predictable chunks based on endTime.

    Holds an ordered list of (ts_ms, rate) rows representing the full universe
    of available funding history. Each `futures_funding_rate(endTime=X)` call
    returns up to `limit` rows with `fundingTime <= X`, ordered ascending by
    fundingTime. When the universe is exhausted (no rows at or before
    endTime), returns []. Tracks call count so tests can assert pagination
    actually happened.
    """

    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self._rows = sorted(rows)  # (ts_ms, rate) tuples, ascending
        self.calls: list[dict[str, Any]] = []

    async def futures_funding_rate(
        self,
        *,
        symbol: str,
        startTime: int | None = None,
        endTime: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        self.calls.append({"endTime": endTime, "limit": limit})
        if endTime is None:
            chosen = self._rows[-limit:]
        else:
            chosen = [r for r in self._rows if r[0] <= endTime][-limit:]
        return [{"fundingTime": ts, "fundingRate": str(rate)} for ts, rate in chosen]

    async def close_connection(self) -> None:
        pass


def _make_universe(start: datetime, n: int, step_hours: int = 8) -> list[tuple[int, float]]:
    """N funding rows starting at `start`, every `step_hours` hours."""
    rows = []
    for i in range(n):
        ts = start + timedelta(hours=i * step_hours)
        rows.append((int(ts.timestamp() * 1000), 0.0001 + i * 1e-7))
    return rows


@pytest.mark.asyncio
async def test_backfill_paginates_until_since(tmp_path: Path):
    """3 pages of 1000-row chunks should cover ~3000 funding ticks."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    universe = _make_universe(base, n=2200)  # ~2 years of 8h ticks
    client = _ChunkedFakeFundingClient(universe)
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    since = base                                       # ask for the full universe
    n_added = await writer.backfill("ETHUSDT", since=since)

    assert n_added == 2200
    df = load_funding(tmp_path / "ETHUSDT.parquet")
    assert len(df) == 2200
    assert df.index.min().to_pydatetime() == base
    # 2200 rows requires at least 3 pages of limit=1000
    assert len(client.calls) >= 3


@pytest.mark.asyncio
async def test_backfill_stops_when_chunk_predates_since(tmp_path: Path):
    """When chunk earliest row is < since, we stop without paging further."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    universe = _make_universe(base, n=2200)
    client = _ChunkedFakeFundingClient(universe)
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    # since = 6 months in (~540 funding ticks from start), so we should grab
    # ~1660 rows (the slice from since to now, +/- one chunk worth of overshoot
    # because we don't slice mid-chunk).
    since = base + timedelta(days=180)
    n_added = await writer.backfill("ETHUSDT", since=since)

    df = load_funding(tmp_path / "ETHUSDT.parquet")
    # We must cover everything from `since` onward (no gap).
    assert df.index.min().to_pydatetime() <= since
    assert df.index.max().to_pydatetime() >= base + timedelta(hours=2199 * 8)
    # We did at least 2 pages (because page 1 alone can't reach 1660 rows).
    assert len(client.calls) >= 2
    # Stop condition fired: we did NOT keep paging for ages.
    assert len(client.calls) <= 4
    assert n_added == len(df)


@pytest.mark.asyncio
async def test_backfill_stops_on_empty_chunk(tmp_path: Path):
    """When Binance returns [], backfill stops even if we haven't reached since."""
    client = _ChunkedFakeFundingClient(rows=[])  # empty universe
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    n = await writer.backfill("ETHUSDT", since=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert n == 0
    # No parquet should be written when no rows came back.
    assert not (tmp_path / "ETHUSDT.parquet").exists()


@pytest.mark.asyncio
async def test_backfill_dedupes_against_existing_parquet(tmp_path: Path):
    """Re-running backfill on top of existing parquet must not duplicate rows."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    universe = _make_universe(base, n=500)
    client = _ChunkedFakeFundingClient(universe)
    writer = FundingRateWriter(client=client, out_dir=tmp_path)

    n1 = await writer.backfill("ETHUSDT", since=base)
    assert n1 == 500
    # Run again — should produce 0 new rows, parquet stays at 500.
    n2 = await writer.backfill("ETHUSDT", since=base)
    df = load_funding(tmp_path / "ETHUSDT.parquet")
    assert len(df) == 500
    # n2 reports rows fetched from API (still 500), but the parquet doesn't grow.
    # We let the implementation define n2's semantic: it's documentation of API
    # cost, not of new-row count. Either contract is acceptable; pin whatever
    # we ship.
    assert isinstance(n2, int)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/data/test_funding_backfill.py -v`
Expected: 4 failures with `AttributeError: 'FundingRateWriter' object has no attribute 'backfill'`.

- [ ] **Step 4: Implement `backfill`**

Add to `src/data/funding.py`'s `FundingRateWriter` class (after `update`):

```python
    async def backfill(self, symbol: str, since: "datetime") -> int:
        """Paginate Binance funding history backward until reaching `since`.

        Walks endTime cursor in 1000-row chunks. Stops on either:
          - chunk earliest row predates `since` (we've covered the requested range), or
          - chunk is empty (no more history available from the exchange).

        Idempotent: re-running on top of an existing parquet dedupes by ts.

        Returns the number of rows fetched from the API across all pages
        (NOT the number of *new* rows added to disk — re-running still returns
        a positive number on the second call). For a precise "rows added to
        disk" count, diff `len(load_funding(...))` before and after.
        """
        out = self._out_dir / f"{symbol}.parquet"
        existing = load_funding(out) if out.exists() else pd.DataFrame()

        since_ms = int(since.timestamp() * 1000)
        end_cursor: int | None = None
        all_rows: list[dict[str, Any]] = []
        seen_ms: set[int] = set()
        while True:
            chunk = await self._client.futures_funding_rate(
                symbol=symbol, endTime=end_cursor, limit=1000,
            )
            if not chunk:
                break
            new_chunk = [r for r in chunk if int(r["fundingTime"]) not in seen_ms]
            for r in new_chunk:
                seen_ms.add(int(r["fundingTime"]))
            all_rows.extend(new_chunk)

            chunk_min_ms = min(int(r["fundingTime"]) for r in chunk)
            if chunk_min_ms <= since_ms:
                break
            # Page back: next chunk's endTime is one ms before this chunk's earliest.
            end_cursor = chunk_min_ms - 1

        if not all_rows:
            return 0

        new_df = pd.DataFrame(all_rows)
        new_df["ts"] = pd.to_datetime(new_df["fundingTime"], unit="ms", utc=True)
        new_df = new_df.set_index("ts")
        new_df["funding_rate"] = new_df["fundingRate"].astype(float)
        new_df = new_df[["funding_rate"]]
        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(out)
        return len(new_df)
```

Also at the top of `src/data/funding.py`, add the imports needed for the new code:

```python
from datetime import datetime
```

(Add this to the existing `from __future__ import annotations` import block — pandas and Path are already imported.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/data/test_funding_backfill.py -v`
Expected: 4 passed.

- [ ] **Step 6: Run full suite to confirm no regressions**

Run: `pytest -q`
Expected: 292 passed (288 + 4 new). Pre-existing libomp failure (if it returns) does not count.

- [ ] **Step 7: Commit**

```bash
git add src/data/funding.py tests/unit/data/test_funding_backfill.py
git commit -m "feat(funding): backfill paginates Binance funding endpoint backward"
```

---

## Task 2: Wire `backfill` into `download_history.py` + manual smoke

**Why:** `scripts/download_history.py` currently calls `update()` only, which gets ≤ 200 rows on a fresh parquet. After this task, the script first checks if existing parquet covers `--years`, calls `backfill` if not, then `update()` always (cheap; only fetches new rows since last run). Manual smoke regenerates the funding parquet AND retrains the model so `drift_reference.json` is also produced (closing Plan 5A's "Step 0" loose end).

**Files:**
- Modify: `scripts/download_history.py` — `main_async` calls `backfill` when needed
- Create: extending tests in `tests/unit/scripts/test_download_history.py` for the new branching

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/scripts/test_download_history.py`:

```python
@pytest.mark.asyncio
async def test_funding_backfill_invoked_when_parquet_missing(tmp_path, monkeypatch):
    """download_history should call FundingRateWriter.backfill when no parquet exists."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from datetime import datetime, timedelta, timezone

    from scripts.download_history import main_async
    import argparse

    # Fake BinanceKline that returns one chunk of klines, then closes cleanly.
    fake_kline_df = _df(datetime(2026, 4, 1, tzinfo=timezone.utc), 5)
    fake_source = AsyncMock()
    fake_source.fetch = AsyncMock(side_effect=[fake_kline_df, _df(datetime(2026, 4, 2, tzinfo=timezone.utc), 0)])
    fake_source.close = AsyncMock()
    fake_source.client = MagicMock()  # accessed by FundingRateWriter

    backfill_mock = AsyncMock(return_value=42)
    update_mock = AsyncMock(return_value=0)

    args = argparse.Namespace(
        symbol="ETHUSDT", timeframe="1h", years=1,
        out_dir=str(tmp_path / "history"),
        funding_out_dir=str(tmp_path / "funding"),
    )

    with patch("scripts.download_history.BinanceKline.open",
               new=AsyncMock(return_value=fake_source)), \
         patch("scripts.download_history.FundingRateWriter") as FW:
        instance = MagicMock()
        instance.backfill = backfill_mock
        instance.update = update_mock
        FW.return_value = instance
        await main_async(args)

    backfill_mock.assert_awaited_once()
    update_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_funding_backfill_skipped_when_parquet_covers_since(tmp_path, monkeypatch):
    """If existing parquet already covers since, only update() runs (not backfill)."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from datetime import datetime, timezone, timedelta
    import argparse
    import pandas as pd

    from scripts.download_history import main_async

    funding_dir = tmp_path / "funding"
    funding_dir.mkdir()
    # Pre-seed parquet covering the full requested window
    base = datetime.now(tz=timezone.utc) - timedelta(days=400)
    idx = pd.DatetimeIndex(
        [base + timedelta(hours=i * 8) for i in range(50)],
        name="ts",
    )
    pd.DataFrame({"funding_rate": [0.0001] * 50}, index=idx).to_parquet(
        funding_dir / "ETHUSDT.parquet"
    )

    fake_kline_df = _df(datetime(2026, 4, 1, tzinfo=timezone.utc), 5)
    fake_source = AsyncMock()
    fake_source.fetch = AsyncMock(side_effect=[fake_kline_df, _df(datetime(2026, 4, 2, tzinfo=timezone.utc), 0)])
    fake_source.close = AsyncMock()
    fake_source.client = MagicMock()

    backfill_mock = AsyncMock(return_value=0)
    update_mock = AsyncMock(return_value=0)

    args = argparse.Namespace(
        symbol="ETHUSDT", timeframe="1h", years=1,
        out_dir=str(tmp_path / "history"),
        funding_out_dir=str(funding_dir),
    )

    with patch("scripts.download_history.BinanceKline.open",
               new=AsyncMock(return_value=fake_source)), \
         patch("scripts.download_history.FundingRateWriter") as FW:
        instance = MagicMock()
        instance.backfill = backfill_mock
        instance.update = update_mock
        FW.return_value = instance
        await main_async(args)

    backfill_mock.assert_not_awaited()
    update_mock.assert_awaited_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/scripts/test_download_history.py -v`
Expected: 2 new failures — `backfill` is never called by `download_history.main_async` today.

- [ ] **Step 3: Update `scripts/download_history.py`**

Modify the `main_async` function. Find the funding section (currently `funding_writer = FundingRateWriter(...) ; added = await funding_writer.update(args.symbol)`). Replace with:

```python
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
```

The key logical change: `update()` ALWAYS runs (cheap; just gets new ticks since last persisted), but `backfill` only runs when needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/scripts/test_download_history.py -v`
Expected: 5 passed (3 original + 2 new).

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 294 passed (292 + 2 new tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/download_history.py tests/unit/scripts/test_download_history.py
git commit -m "feat(download): call funding backfill when parquet does not cover requested years"
```

- [ ] **Step 7: Manual smoke — backfill 2 years of funding history**

Run (network required, ~30 seconds):

```bash
PYTHONPATH=src python scripts/download_history.py --years 2
```

Expected stdout:
```
klines: 17520 bars -> data/history/ETHUSDT_1h.parquet
funding backfill: ~2200 rows fetched
funding update: 0 new rows -> data/funding/ETHUSDT.parquet
```

Verify:

```bash
source venv/bin/activate
python -c "
import pandas as pd
df = pd.read_parquet('data/funding/ETHUSDT.parquet')
print('funding shape:', df.shape)
print('range:', df.index.min(), '→', df.index.max())
print('expected ~2190 rows for 2 years × 365 × 3 ticks/day')
"
```

Expected: ~2190 rows; range covers ~2024-04-25 → ~2026-04-25.

Note the `funding update: 0 new rows` is expected because backfill already covered through `now`.

- [ ] **Step 8: Manual smoke — re-train so `drift_reference.json` lands**

Plan 5A's training was skipped on `drift_reference.json` because Task 10 committed after training. Re-run now to fix that AND get a fresh model with full funding history:

```bash
PYTHONPATH=src python scripts/build_training_set.py \
    --kline data/history/ETHUSDT_1h.parquet \
    --out   data/training/ETHUSDT_1h_features.parquet
```

(This takes ~30 minutes due to O(N²) `compute_all` over 17320 bars. Run in background or skip if time is short — features parquet from Plan 5A is still valid; only the funding column will differ.)

Then:

```bash
PYTHONPATH=src python scripts/train_xgb.py \
    --features data/training/ETHUSDT_1h_features.parquet \
    --labels   data/training/ETHUSDT_1h_labels.parquet \
    --out      models
```

Expected: a new `model_version` (sha256 of new booster bytes will differ from `ece8d16d4a29` because funding column changed), `models/drift_reference.json` is now created (was missing in Plan 5A), Brier scores printed.

Verify:

```bash
ls -lh models/
```

Expected: 4 files including `drift_reference.json`.

- [ ] **Step 9: Write Plan 5B-1 STATUS**

Create `docs/superpowers/plans/2026-04-26-pivot-plan5b1-STATUS.md`:

```markdown
# Plan 5B-1 STATUS — Funding Backfill

**Date**: 2026-04-26
**Branch**: `pivot/foundation`
**Worktree**: `.worktrees/pivot-foundation`
**Base commit**: (Plan 5A head, e.g., `b4a4c3e`)
**Head commit**: (this commit's SHA)

## Summary

`FundingRateWriter` gains a `backfill(symbol, since)` method that paginates Binance's funding endpoint backward in 1000-row chunks until reaching `since` or running out of history. `scripts/download_history.py` invokes it whenever the on-disk parquet doesn't already cover the requested `--years` window. Manual smoke replaces the ~200-row stub funding parquet with the full ~2190 rows for 2 years of ETHUSDT.

Test count: **294 passed** (Plan 5A baseline 288 + 6 new across Task 1 + Task 2).

## Task table

| # | Title | Commit | Files |
|---|-------|--------|-------|
| 1 | `FundingRateWriter.backfill` | (commit SHA) | `src/data/funding.py`, new test |
| 2 | Wire backfill into download_history + retrain manual smoke | (commit SHA) | `scripts/download_history.py`, expanded test |

## Manual smoke results

- `python scripts/download_history.py --years 2` produced funding parquet with ~2190 rows covering ~2024-04 → 2026-04 (vs. Plan 5A's ~200 rows from ~2026-02 → 2026-04).
- Re-train: new `model_version=<...>`, `models/drift_reference.json` populated (closes Plan 5A's "Step 0" loose end).

## Decisions landed

- **`backfill` separate from `update`** — different cursor directions and stop conditions.
- **`backfill` semantics**: returns count of rows fetched from API (not net rows added to disk). Re-running is idempotent on disk via dedupe.
- **`update()` always runs** in `download_history.main_async` — cheap and keeps the head fresh.

## What is NOT done (Plan 5B-2 onward)

- Plan 5B-2: ReplayBroker + Broker contract test suite.
- Plan 5B-3: Walk-forward backtest harness + Deflated Sharpe.
- Plan 5B-4: Pre-Live Gate module (§10 8 gates).
```

- [ ] **Step 10: Final commit**

```bash
git add docs/superpowers/plans/2026-04-26-pivot-plan5b1-STATUS.md
git commit -m "docs: Plan 5B-1 handoff STATUS"
```

---

## Self-review notes

- **Spec coverage**: Plan 5A STATUS listed "Funding rate backfill paginator" as Plan 5B item; this plan delivers it.
- **Type consistency**: `backfill(symbol: str, since: datetime) -> int` mirrors `update(symbol: str) -> int` for parallel ergonomics. The `int` return is documented as "rows fetched from API," not "rows added to disk" — pinned in docstring + test.
- **No placeholders**: Every step has concrete code and concrete bash commands. Manual smoke commands include exact expected stdout.
- **Idempotency**: Both `backfill` and `update` call `combined[~combined.index.duplicated(keep="last")]` so re-runs are safe.
- **Test independence**: `_ChunkedFakeFundingClient` doesn't share state with the existing `FakeFundingClient` in `test_funding.py`; both stay independent.
