# Plan 5A — Real Data + Real Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stub data source and stub `XGBPredictor` with a real Binance kline pipeline + a calibrated XGBoost model trained on 2 years of ETHUSDT 1h history; end-to-end one real proposal lands in `proposals` table.

**Architecture:** Three phases — (A) flip `build_scan_context` to async and wire `BinanceKline` + a `RollingKlineCache` so `mid_provider/atr_provider/spread_provider` return real values; (B) build the training pipeline (download → features → labels → walk-forward XGBoost with isotonic vs Platt A/B → model registry); (C) flip a config flag to load the trained model and run an E2E smoke. The Ensemble's existing LLM fallback path (`LLM_UNAVAILABLE_MARKER`) absorbs the lack of Ollama, so Plan 5A intentionally does NOT require Ollama installation — that lands in Plan 5B.

**Tech Stack:** Python 3.11, asyncio, `python-binance` AsyncClient, pandas, XGBoost 2.x, scikit-learn (`IsotonicRegression`, `LogisticRegression` for Platt, `TimeSeriesSplit`), SQLAlchemy, pytest with `asyncio_mode=auto`.

**Decisions baked in:**
- Calibration: train **both** isotonic and Platt; pick winner by **OOS Brier score** on the last walk-forward fold; persist choice to `model_versions.calibration_method`.
- Label: binary `close[t+4] / close[t] - 1 > 0` (4-bar forward up). Triple-barrier deferred.
- Training data: **ETHUSDT 1h, 2 years** (~17,500 bars).
- Wiring: `build_scan_context` becomes async (cleaner than the inject-post-boot workaround Plan 4 STATUS noted).
- LLM: stays stubbed via Ensemble's existing exception fallback. Plan 5A's E2E does NOT require Ollama.
- Funding feature: requires `data/funding/ETHUSDT.parquet` populated by `FundingRateWriter` — Plan 5A includes a one-shot job in Task 5.

**Out of Plan 5A scope (deferred to Plan 5B):**
- ReplayBroker / LiveBroker / Broker contract test suite.
- Walk-forward Deflated Sharpe report (Pre-Live Gate §10.1.2).
- Pre-Live Gate module (§10).
- Ollama / Gemma activation.
- mypy `--strict` global pass.

---

## File map

### Created
- `src/data/kline_cache.py` — `RollingKlineCache` (in-process N-bar buffer + last-tick spread)
- `src/data/providers.py` — concrete `mid_provider / atr_provider / spread_provider` callables backed by the cache
- `src/models/registry.py` — model-bundle save/load + `model_versions` SQL insert helpers
- `scripts/download_history.py` — fetches ETHUSDT 1h klines + funding history into Parquet
- `scripts/build_training_set.py` — runs `compute_all` over every bar → `data/training/<symbol>_<tf>_<cutoff>.parquet`
- `scripts/build_labels.py` — adds `y_4bar_up` column to a training-set parquet
- `tests/unit/data/test_kline_cache.py`
- `tests/unit/data/test_providers.py`
- `tests/unit/models/test_registry.py`
- `tests/unit/scripts/__init__.py` (empty marker)
- `tests/unit/scripts/test_download_history.py`
- `tests/unit/scripts/test_build_training_set.py`
- `tests/unit/scripts/test_build_labels.py`
- `tests/unit/scripts/test_train_xgb.py` (covers the new walk-forward + AB calibration logic)
- `tests/unit/scripts/test_drift_reference.py`
- `tests/e2e/test_real_data_smoke.py`
- `data/history/.gitkeep`, `data/training/.gitkeep`, `data/funding/.gitkeep`, `models/.gitkeep`
- `docs/superpowers/plans/2026-04-25-pivot-plan5a-STATUS.md` (handoff at end)

### Modified
- `src/wiring.py` — `build_scan_context` becomes `async def`; opens `BinanceKline`; reads `cfg.use_trained_model` to switch stub→`XGBPredictor.load`; wires real providers; populates `FeatureDriftMonitor.reference` from training baseline.
- `src/orchestrator.py` — `boot()` awaits `build_scan_context`; tracks the `BinanceKline` instance in `_lifecycle` so `run()` closes it on shutdown; new config fields.
- `scripts/train_xgb.py` — rewrite: loads training-set parquet (not raw CSV), walk-forward CV, isotonic vs Platt A/B by OOS Brier, persists drift reference alongside the model bundle.
- `tests/e2e/test_orchestrator_boot.py` — adapt to async `build_scan_context` (rename of internal stubs only).
- `.gitignore` — ensure `data/history/`, `data/training/`, `data/funding/`, `models/` ignored except their `.gitkeep`.

### Untouched (verified intentionally)
- `src/decision/ensemble.py` — existing LLM fallback handles missing Ollama.
- `src/features/*` — feature implementations stable; canonical hash invariant.
- `src/state/alembic/` — `model_versions` table already in baseline schema (no migration).

---

## Task 1: Async-aware `build_scan_context`

**Why first:** Every later task either uses async data fetching (downloader, RollingKlineCache, BinanceKline) or wants to inject a real `XGBPredictor.load()` result. Flipping the wiring shape now avoids a second rewrite later. The Plan 4 STATUS doc explicitly flagged this as Plan 5's first move.

**Files:**
- Modify: `src/wiring.py:81-184`
- Modify: `src/orchestrator.py:64-72` (the `boot()` body that currently calls `build_scan_context` synchronously)
- Modify: `tests/e2e/test_orchestrator_boot.py` (only if it instantiates `build_scan_context` directly — use grep first)
- Create: `tests/unit/test_wiring_async.py`

- [ ] **Step 1: Confirm callers of `build_scan_context`**

Run: `grep -rn "build_scan_context" src tests scripts`

Expected: hits in `src/orchestrator.py` (one call inside `boot()`) and possibly tests. If `tests/unit/test_pipeline.py` or any other test imports it, list them — they all need to migrate.

- [ ] **Step 2: Write the failing async test**

Create `tests/unit/test_wiring_async.py`:

```python
"""Verifies build_scan_context is async and returns a (ScanContext, lifecycle) pair
even when BinanceKline open() succeeds via an injected fake client."""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest
import sqlalchemy as sa

from orchestrator import OrchestratorConfig
from wiring import build_scan_context


@pytest.mark.asyncio
async def test_build_scan_context_is_async(tmp_path):
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        use_trained_model=False,                 # stay on stub for this unit test
    )
    # Build a real engine + run migrations so repo wiring works.
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{cfg.sqlite_path}")
    alembic.command.upgrade(ac, "head")
    engine = sa.create_engine(f"sqlite:///{cfg.sqlite_path}")

    # Patch BinanceKline.open so we never touch the network.
    fake_kline = AsyncMock()
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        assert inspect.iscoroutinefunction(build_scan_context)
        ctx, lifecycle = await build_scan_context(cfg, engine)

    assert ctx.symbols == ["ETHUSDT"]
    assert "binance_kline" in lifecycle
    assert lifecycle["binance_kline"] is fake_kline
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_wiring_async.py -v`

Expected: FAIL — `build_scan_context` is currently `def`, not `async def`; `inspect.iscoroutinefunction` returns False; or `TypeError` because the call missed `await`.

- [ ] **Step 4: Add `use_trained_model` to `OrchestratorConfig`**

Modify `src/orchestrator.py:27-44` `OrchestratorConfig` dataclass — add three fields right after `paper_broker_seed`:

```python
    use_trained_model: bool = False
    model_dir: str = "models"
    drift_reference_path: str = "models/drift_reference.json"
```

- [ ] **Step 5: Convert `build_scan_context` to async + open BinanceKline**

Rewrite `src/wiring.py` (replace the entire file body below the imports). Key changes:
- Drop `_StubDataSource`.
- Open `BinanceKline` via `BinanceKline.open()`.
- Track it in `lifecycle["binance_kline"]` so the orchestrator can close it.
- Keep stub mid/atr/spread providers in this task — Tasks 2-4 replace them.

```python
"""Wiring factory — async version (Plan 5A Task 1).

Opens the real BinanceKline and threads it into ScanContext. Mid / ATR /
spread providers stay stubbed in this task; Tasks 2-4 wire the
RollingKlineCache-backed providers.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from data.binance_kline import BinanceKline
from decision.ensemble import Ensemble
from decision.halt import HaltManager
from decision.policy import ThresholdPolicy
from decision.risk.checks import (
    DailyLossKillSwitch,
    MandatoryStopLoss,
    MaxConcurrentPositions,
    SpreadGate,
)
from decision.risk.pipeline import RiskPipeline
from decision.sizing import FixedFractionalSizer
from decision.triggers import (
    DailyLossTrigger,
    FeatureDriftTrigger,
    HeartbeatTrigger,
)
from execution.paper_broker import PaperBroker, PaperBrokerConfig
from execution.repositories import (
    BrokerEventRepo,
    ProposalRepo,
    SessionStateRepo,
)
from features.registry import build_default_registry
from interface.chat_llm import ChatLLM
from interface.repositories import MessageRepo, ToolCallRepo
from interface.tools import ToolExecutor
from models.llm.gemma_context import GemmaContextProvider
from models.llm.ollama_client import OllamaClient
from models.xgb_predictor import XGBPredictor
from observability.drift import FeatureDriftMonitor
from observability.drift_config import load_drift_config
from orchestrator import OrchestratorConfig
from pipeline import ScanContext


def _stub_mid(_symbol: str) -> float:
    return 3000.0


def _stub_atr(_symbol: str) -> float:
    return 15.0


def _stub_spread_bps(_symbol: str) -> float:
    return 0.0


async def build_scan_context(
    cfg: OrchestratorConfig,
    engine: sa.Engine,
) -> tuple[ScanContext, dict[str, Any]]:
    """Async: opens BinanceKline, builds the wired ScanContext."""
    data_source = await BinanceKline.open()

    registry = build_default_registry()

    if cfg.use_trained_model:
        # Plan 5A Task 9 wires the real load path; until then this branch is
        # expected to raise FileNotFoundError on a fresh checkout.
        from models.registry import load_latest_model
        ml = load_latest_model(Path(cfg.model_dir))
    else:
        ml = XGBPredictor.stub(prob_up=0.55, ml_model_version="stub-v0")

    ollama = OllamaClient(model=cfg.ollama_model, host=cfg.ollama_host)
    gemma = GemmaContextProvider(client=ollama)
    ensemble = Ensemble(ml=ml, llm_ctx=gemma)

    rng = random.Random(cfg.paper_broker_seed)
    broker = PaperBroker(
        cfg=PaperBrokerConfig(),
        rng=rng,
        mid_provider=_stub_mid,
    )

    symbol = cfg.watchlist[0]
    policy = ThresholdPolicy(
        long_threshold=cfg.long_threshold,
        short_threshold=cfg.short_threshold,
        symbol=symbol,
        mid_provider=_stub_mid,
        atr_provider=_stub_atr,
    )

    risk = RiskPipeline(checks=[
        MandatoryStopLoss(),
        SpreadGate(max_bps=10.0, spread_provider=_stub_spread_bps),
        DailyLossKillSwitch(threshold_r=cfg.daily_loss_max_r),
        MaxConcurrentPositions(cap=3),
    ])
    sizer = FixedFractionalSizer(fraction=0.01)

    proposal_repo = ProposalRepo(engine)
    event_repo = BrokerEventRepo(engine)
    session_repo = SessionStateRepo(engine)
    message_repo = MessageRepo(engine)
    tool_call_repo = ToolCallRepo(engine)

    tool_executor = ToolExecutor(engine=engine, broker=broker)
    chat_llm = ChatLLM(
        client=ollama,
        tool_executor=tool_executor,
        message_repo=message_repo,
        tool_call_repo=tool_call_repo,
    )

    drift_cfg = load_drift_config(cfg.drift_yaml)
    drift_monitor = FeatureDriftMonitor(
        reference={},
        psi_threshold=drift_cfg.psi_threshold,
        ks_threshold=drift_cfg.ks_threshold,
        n_bins=drift_cfg.psi_bins,
    )
    drift_state: dict[str, Any] = {"breached": False}

    halt = HaltManager(
        halt_file=Path(cfg.halt_file),
        engine=engine,
        triggers=[
            HeartbeatTrigger(engine=engine, max_stale_seconds=cfg.heartbeat_max_stale_seconds),
            DailyLossTrigger(engine=engine, max_loss_r=cfg.daily_loss_max_r),
            FeatureDriftTrigger(state=drift_state),
        ],
    )

    ctx = ScanContext(
        symbols=cfg.watchlist,
        halt=halt,
        data_source=data_source,
        registry=registry,
        ensemble=ensemble,
        policy=policy,
        risk=risk,
        sizer=sizer,
        broker=broker,
        proposal_repo=proposal_repo,
        event_repo=event_repo,
        session_repo=session_repo,
        chat_llm=chat_llm,
        telegram=None,
        notify_chat_id=cfg.notify_chat_id,
    )
    lifecycle: dict[str, Any] = {
        "ollama_client": ollama,
        "drift_state": drift_state,
        "drift_monitor": drift_monitor,
        "binance_kline": data_source,
    }
    return ctx, lifecycle
```

- [ ] **Step 6: Adapt `Orchestrator.boot()` to await it + close on shutdown**

In `src/orchestrator.py:64-72`, change the import + call:

```python
        # Lazy import — wiring imports OrchestratorConfig from this module.
        from wiring import build_scan_context
        self.ctx, self._lifecycle = await build_scan_context(self.cfg, self.engine)
```

Then in `Orchestrator.run()`'s shutdown / finally block, add a kline close. Locate the existing `try/finally` that closes `ollama_client` (search for `"ollama_client"` in `src/orchestrator.py`) and append:

```python
            kline = self._lifecycle.get("binance_kline")
            if kline is not None:
                try:
                    await kline.close()
                except Exception:
                    log.warning("binance_kline_close_failed")
```

- [ ] **Step 7: Run the new test to verify pass**

Run: `pytest tests/unit/test_wiring_async.py -v`

Expected: PASS.

- [ ] **Step 8: Run full suite to confirm no regressions**

Run: `pytest -q`

Expected: 253 → may temporarily drop to 252 if `tests/e2e/test_orchestrator_boot.py` patches `_StubDataSource` — fix any failures by patching `BinanceKline.open` instead.

- [ ] **Step 9: Commit**

```bash
git add src/wiring.py src/orchestrator.py tests/unit/test_wiring_async.py tests/e2e/test_orchestrator_boot.py
git commit -m "feat(wiring): build_scan_context is now async; opens BinanceKline + close on shutdown"
```

---

## Task 2: `RollingKlineCache`

**Why:** `mid_provider/atr_provider` need a cheap, in-process source of recent prices that does not hit Binance every microsecond. A 200-bar deque keyed by `(symbol, timeframe)` matches the 200-bar `fetch_latest` already used by `_scan_symbol`.

**Files:**
- Create: `src/data/kline_cache.py`
- Create: `tests/unit/data/test_kline_cache.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/data/test_kline_cache.py
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from data.kline_cache import RollingKlineCache


def _row(ts: datetime, close: float, hi: float | None = None, lo: float | None = None) -> dict:
    return {
        "open": close, "high": hi or close, "low": lo or close,
        "close": close, "volume": 1.0,
    }


def _df(rows: list[tuple[datetime, dict]]) -> pd.DataFrame:
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
    df.index.name = "open_time"
    return df


def test_returns_none_when_empty():
    cache = RollingKlineCache(max_bars=200)
    assert cache.last_close("ETHUSDT", "1h") is None
    assert cache.atr("ETHUSDT", "1h", n=14) is None


def test_ingest_keeps_last_max_bars():
    cache = RollingKlineCache(max_bars=3)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [(base + timedelta(hours=i), _row(base + timedelta(hours=i), 100.0 + i)) for i in range(5)]
    cache.ingest("ETHUSDT", "1h", _df(rows))
    snap = cache.snapshot("ETHUSDT", "1h")
    assert len(snap) == 3
    assert snap["close"].iloc[-1] == 104.0


def test_ingest_dedupes_overlapping_timestamps():
    cache = RollingKlineCache(max_bars=10)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = _df([(base + timedelta(hours=i), _row(base + timedelta(hours=i), 100.0 + i)) for i in range(3)])
    overlap = _df([(base + timedelta(hours=2), _row(base + timedelta(hours=2), 999.0))])
    cache.ingest("ETHUSDT", "1h", first)
    cache.ingest("ETHUSDT", "1h", overlap)
    snap = cache.snapshot("ETHUSDT", "1h")
    assert len(snap) == 3
    # latest write wins for the duplicated bar
    assert snap["close"].iloc[-1] == 999.0


def test_atr_simple_average_of_high_low_range():
    cache = RollingKlineCache(max_bars=20)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(5):
        rows.append((base + timedelta(hours=i),
                     _row(base + timedelta(hours=i), close=100.0,
                          hi=110.0 + i, lo=90.0 + i)))
    cache.ingest("ETHUSDT", "1h", _df(rows))
    # Range = 20.0 every bar -> ATR over n=5 = 20.0
    assert cache.atr("ETHUSDT", "1h", n=5) == pytest.approx(20.0)


def test_spread_bps_recorded_and_returned():
    cache = RollingKlineCache(max_bars=10)
    cache.record_spread("ETHUSDT", bid=2999.0, ask=3001.0)
    # mid = 3000, spread = 2 -> 2/3000*1e4 = 6.6667 bps
    assert cache.spread_bps("ETHUSDT") == pytest.approx(2.0 / 3000.0 * 10_000)


def test_unsupported_symbol_returns_none_for_spread():
    cache = RollingKlineCache(max_bars=10)
    assert cache.spread_bps("BTCUSDT") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/data/test_kline_cache.py -v`

Expected: ImportError — module not yet created.

- [ ] **Step 3: Implement `RollingKlineCache`**

```python
# src/data/kline_cache.py
"""In-process rolling kline + spread cache (Plan 5A Task 2).

Holds the last N bars per (symbol, timeframe). `mid_provider` and
`atr_provider` (Task 3) read from here so we do not hit Binance on every
risk-check call. Updated by a refresh loop owned by the orchestrator
(Task 4) and seeded by `BinanceKline.fetch_latest` at boot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RollingKlineCache:
    max_bars: int = 200
    _frames: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    _spreads_bps: dict[str, float] = field(default_factory=dict)

    def ingest(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        key = (symbol, timeframe)
        existing = self._frames.get(key)
        if existing is None or existing.empty:
            combined = df
        else:
            combined = pd.concat([existing, df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        self._frames[key] = combined.iloc[-self.max_bars:]

    def snapshot(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._frames.get((symbol, timeframe), pd.DataFrame()).copy()

    def last_close(self, symbol: str, timeframe: str) -> float | None:
        snap = self._frames.get((symbol, timeframe))
        if snap is None or snap.empty:
            return None
        return float(snap["close"].iloc[-1])

    def atr(self, symbol: str, timeframe: str, n: int = 14) -> float | None:
        snap = self._frames.get((symbol, timeframe))
        if snap is None or len(snap) < n:
            return None
        rng = (snap["high"] - snap["low"]).iloc[-n:]
        return float(rng.mean())

    def record_spread(self, symbol: str, bid: float, ask: float) -> None:
        if bid <= 0 or ask <= 0 or ask < bid:
            return
        mid = 0.5 * (bid + ask)
        if mid <= 0:
            return
        self._spreads_bps[symbol] = (ask - bid) / mid * 10_000.0

    def spread_bps(self, symbol: str) -> float | None:
        return self._spreads_bps.get(symbol)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/data/test_kline_cache.py -v`

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/kline_cache.py tests/unit/data/test_kline_cache.py
git commit -m "feat(data): RollingKlineCache — in-process N-bar buffer + spread record"
```

---

## Task 3: Real `mid_provider / atr_provider / spread_provider`

**Why:** `wiring.py` and `ThresholdPolicy / SpreadGate / PaperBroker` take callables `(symbol) -> float`. With the cache from Task 2 we can build real ones; if the cache is cold we keep returning the safe stub values, so the orchestrator can boot before the first refresh.

**Files:**
- Create: `src/data/providers.py`
- Create: `tests/unit/data/test_providers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/data/test_providers.py
import pytest

from data.kline_cache import RollingKlineCache
from data.providers import (
    cache_backed_atr_provider,
    cache_backed_mid_provider,
    cache_backed_spread_bps_provider,
)
from datetime import datetime, timezone, timedelta
import pandas as pd


def _seed_cache(cache: RollingKlineCache, symbol: str, n: int) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    idx = []
    for i in range(n):
        idx.append(base + timedelta(hours=i))
        rows.append({"open": 3000, "high": 3010 + i, "low": 2990 - i,
                     "close": 3000 + i, "volume": 1.0})
    df = pd.DataFrame(rows, index=idx)
    cache.ingest(symbol, "1h", df)


def test_mid_provider_uses_last_close_when_warm():
    cache = RollingKlineCache(max_bars=200)
    _seed_cache(cache, "ETHUSDT", 5)
    mid = cache_backed_mid_provider(cache, timeframe="1h", fallback=3000.0)
    assert mid("ETHUSDT") == 3004.0   # last close = 3000 + 4


def test_mid_provider_returns_fallback_when_cold():
    cache = RollingKlineCache(max_bars=200)
    mid = cache_backed_mid_provider(cache, timeframe="1h", fallback=2500.0)
    assert mid("ETHUSDT") == 2500.0


def test_atr_provider_uses_cache_atr_when_warm():
    cache = RollingKlineCache(max_bars=200)
    _seed_cache(cache, "ETHUSDT", 30)
    atr = cache_backed_atr_provider(cache, timeframe="1h", n=14, fallback=15.0)
    assert atr("ETHUSDT") > 0.0
    assert atr("ETHUSDT") != 15.0


def test_atr_provider_returns_fallback_when_cold():
    cache = RollingKlineCache(max_bars=200)
    atr = cache_backed_atr_provider(cache, timeframe="1h", n=14, fallback=15.0)
    assert atr("ETHUSDT") == 15.0


def test_spread_provider_uses_recorded_spread_when_warm():
    cache = RollingKlineCache(max_bars=200)
    cache.record_spread("ETHUSDT", bid=2999.0, ask=3001.0)
    sp = cache_backed_spread_bps_provider(cache, fallback=0.0)
    assert sp("ETHUSDT") == pytest.approx(2.0 / 3000.0 * 10_000)


def test_spread_provider_returns_fallback_when_no_record():
    cache = RollingKlineCache(max_bars=200)
    sp = cache_backed_spread_bps_provider(cache, fallback=0.0)
    assert sp("ETHUSDT") == 0.0
```

- [ ] **Step 2: Run tests to verify fail**

Run: `pytest tests/unit/data/test_providers.py -v`

Expected: ImportError — module not yet created.

- [ ] **Step 3: Implement providers**

```python
# src/data/providers.py
"""Cache-backed callables for mid / ATR / spread (Plan 5A Task 3).

Returned functions match the signature `(symbol: str) -> float` consumed
by ThresholdPolicy, PaperBroker, and SpreadGate. Each returns the
configured `fallback` value when the cache has no data, so the
orchestrator can boot and run unit tests before the first refresh.
"""
from __future__ import annotations

from typing import Callable

from data.kline_cache import RollingKlineCache


def cache_backed_mid_provider(
    cache: RollingKlineCache,
    timeframe: str,
    fallback: float,
) -> Callable[[str], float]:
    def _mid(symbol: str) -> float:
        v = cache.last_close(symbol, timeframe)
        return v if v is not None else fallback
    return _mid


def cache_backed_atr_provider(
    cache: RollingKlineCache,
    timeframe: str,
    n: int,
    fallback: float,
) -> Callable[[str], float]:
    def _atr(symbol: str) -> float:
        v = cache.atr(symbol, timeframe, n=n)
        return v if v is not None else fallback
    return _atr


def cache_backed_spread_bps_provider(
    cache: RollingKlineCache,
    fallback: float,
) -> Callable[[str], float]:
    def _spread(symbol: str) -> float:
        v = cache.spread_bps(symbol)
        return v if v is not None else fallback
    return _spread
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/data/test_providers.py -v`

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/providers.py tests/unit/data/test_providers.py
git commit -m "feat(data): cache-backed mid/atr/spread providers with cold-start fallbacks"
```

---

## Task 4: Wire cache + providers into `build_scan_context` + refresh loop

**Why:** Now that the cache and providers exist, swap them into wiring and add a periodic refresh inside the orchestrator so the cache stays warm. We also seed the cache once at boot via `fetch_latest(n=200)` so the first scheduled scan has data.

**Files:**
- Modify: `src/wiring.py` (replace `_stub_*` callables with cache-backed versions)
- Modify: `src/orchestrator.py` — add `_kline_refresh_loop` started inside the TaskGroup
- Create: `tests/unit/test_wiring_real_providers.py`

- [ ] **Step 1: Failing test for wiring change**

```python
# tests/unit/test_wiring_real_providers.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import sqlalchemy as sa

from orchestrator import OrchestratorConfig
from wiring import build_scan_context


def _fake_klines() -> pd.DataFrame:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    idx = [base + timedelta(hours=i) for i in range(50)]
    return pd.DataFrame({
        "open": [3000.0 + i for i in range(50)],
        "high": [3010.0 + i for i in range(50)],
        "low":  [2990.0 + i for i in range(50)],
        "close": [3005.0 + i for i in range(50)],
        "volume": [1.0] * 50,
    }, index=pd.DatetimeIndex(idx, name="open_time"))


@pytest.mark.asyncio
async def test_wiring_uses_cache_backed_mid_after_seed(tmp_path):
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        use_trained_model=False,
    )
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{cfg.sqlite_path}")
    alembic.command.upgrade(ac, "head")
    engine = sa.create_engine(f"sqlite:///{cfg.sqlite_path}")

    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=_fake_klines())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        ctx, lifecycle = await build_scan_context(cfg, engine)

    cache = lifecycle["kline_cache"]
    # Seed already happened during build_scan_context.
    assert cache.last_close("ETHUSDT", "1h") == 3054.0
    # The mid provider held by the policy must read the same value.
    assert ctx.policy.mid_provider("ETHUSDT") == 3054.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_wiring_real_providers.py -v`

Expected: KeyError `kline_cache` not in lifecycle, or mid is still 3000.0 (the stub).

- [ ] **Step 3: Update `build_scan_context`**

In `src/wiring.py`:
- Add `from data.kline_cache import RollingKlineCache` and `from data.providers import (cache_backed_mid_provider, cache_backed_atr_provider, cache_backed_spread_bps_provider)`.
- After `data_source = await BinanceKline.open()`, build a `RollingKlineCache(max_bars=200)`, fetch a seed, and ingest it:

```python
    cache = RollingKlineCache(max_bars=200)
    symbol = cfg.watchlist[0]
    timeframe = "1h"
    try:
        seed = await data_source.fetch_latest(symbol, timeframe, 200)
        cache.ingest(symbol, timeframe, seed)
    except Exception:
        # Boot must succeed even if the seed fetch fails; refresh loop will
        # try again. Keep providers in fallback mode until then.
        pass

    mid_provider = cache_backed_mid_provider(cache, timeframe=timeframe, fallback=3000.0)
    atr_provider = cache_backed_atr_provider(cache, timeframe=timeframe, n=14, fallback=15.0)
    spread_provider = cache_backed_spread_bps_provider(cache, fallback=0.0)
```

Then thread `mid_provider / atr_provider / spread_provider` into:
- `PaperBroker(cfg=..., rng=..., mid_provider=mid_provider)`
- `ThresholdPolicy(..., mid_provider=mid_provider, atr_provider=atr_provider)`
- `SpreadGate(max_bps=10.0, spread_provider=spread_provider)`

Add to lifecycle:

```python
    lifecycle["kline_cache"] = cache
```

- [ ] **Step 4: Add a refresh loop to the orchestrator**

In `src/orchestrator.py`, inside the TaskGroup body in `run()`, add:

```python
            tg.create_task(self._kline_refresh_loop(), name="kline_refresh")
```

And define the method:

```python
    async def _kline_refresh_loop(self) -> None:
        """Refreshes RollingKlineCache every minute. Failures are logged
        and swallowed; cache providers fall back to last known values."""
        if self.ctx is None or "kline_cache" not in self._lifecycle:
            return
        cache = self._lifecycle["kline_cache"]
        symbol = self.ctx.symbols[0]
        timeframe = "1h"
        while not self.is_stopping():
            try:
                df = await self.ctx.data_source.fetch_latest(symbol, timeframe, 200)
                cache.ingest(symbol, timeframe, df)
            except Exception:
                log.warning("kline_refresh_failed", symbol=symbol)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=60.0)  # type: ignore[arg-type]
            except asyncio.TimeoutError:
                continue
            return
```

- [ ] **Step 5: Run new test + full suite**

Run: `pytest tests/unit/test_wiring_real_providers.py -v && pytest -q`

Expected: new test passes; full suite green (any tests that referenced `_stub_mid` should already be on the patched-builder path).

- [ ] **Step 6: Commit**

```bash
git add src/wiring.py src/orchestrator.py tests/unit/test_wiring_real_providers.py
git commit -m "feat(wiring): cache-backed mid/atr/spread + minute refresh loop"
```

---

## Task 5: `scripts/download_history.py` — historical kline + funding fetch

**Why:** Training needs ~17,500 ETHUSDT 1h bars (2 years). The `BinanceKline.fetch` API caps at 1000 bars per call, so we need a paginating loop. Funding rate parquet is also empty today; the same script populates `data/funding/ETHUSDT.parquet` so `FundingFeature.compute` returns real values.

**Files:**
- Create: `scripts/download_history.py`
- Create: `tests/unit/scripts/__init__.py` (empty)
- Create: `tests/unit/scripts/test_download_history.py`
- Create: `data/history/.gitkeep`, `data/funding/.gitkeep`
- Modify: `.gitignore` (ignore `data/history/*.parquet` and `data/funding/*.parquet`)

- [ ] **Step 1: Failing tests for the paginator**

```python
# tests/unit/scripts/test_download_history.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from scripts.download_history import (
    fetch_klines_paginated,
    upsert_parquet,
)


def _df(start: datetime, n: int, base_close: float = 3000.0) -> pd.DataFrame:
    idx = [start + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame({
        "open": [base_close] * n, "high": [base_close + 5] * n,
        "low": [base_close - 5] * n, "close": [base_close + i for i in range(n)],
        "volume": [1.0] * n,
    }, index=pd.DatetimeIndex(idx, name="open_time"))


@pytest.mark.asyncio
async def test_paginator_walks_in_1000_bar_chunks_until_until():
    chunks = [_df(datetime(2026, 1, 1, tzinfo=timezone.utc), 1000),
              _df(datetime(2026, 2, 11, 16, tzinfo=timezone.utc), 500),
              _df(datetime(2026, 1, 1, tzinfo=timezone.utc), 0)]  # empty -> stop
    fake = AsyncMock()
    fake.fetch = AsyncMock(side_effect=chunks)
    df = await fetch_klines_paginated(
        fake,
        symbol="ETHUSDT", timeframe="1h",
        since=datetime(2026, 1, 1, tzinfo=timezone.utc),
        until=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    assert len(df) == 1500
    assert df.index.is_monotonic_increasing
    # No duplicate indices.
    assert df.index.is_unique


@pytest.mark.asyncio
async def test_paginator_stops_on_empty_chunk():
    fake = AsyncMock()
    fake.fetch = AsyncMock(side_effect=[
        _df(datetime(2026, 1, 1, tzinfo=timezone.utc), 0),
    ])
    df = await fetch_klines_paginated(
        fake,
        symbol="ETHUSDT", timeframe="1h",
        since=datetime(2026, 1, 1, tzinfo=timezone.utc),
        until=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    assert df.empty


def test_upsert_parquet_dedupes_overlap(tmp_path):
    p = tmp_path / "ETHUSDT_1h.parquet"
    a = _df(datetime(2026, 1, 1, tzinfo=timezone.utc), 5, base_close=3000.0)
    b = _df(datetime(2026, 1, 1, 3, tzinfo=timezone.utc), 5, base_close=4000.0)
    upsert_parquet(p, a)
    upsert_parquet(p, b)
    out = pd.read_parquet(p)
    # 5 + 5 with 2 overlapping timestamps -> 8 unique
    assert len(out) == 8
    # Latest write wins on the overlap.
    assert out.loc[datetime(2026, 1, 1, 4, tzinfo=timezone.utc), "close"] == 4001
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/unit/scripts/test_download_history.py -v`

Expected: ImportError — module doesn't exist.

- [ ] **Step 3: Implement the script**

```python
# scripts/download_history.py
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

        # Funding rate (used by FundingFeature). FundingRateWriter expects an
        # async client object exposing futures_funding_rate(...).
        funding_dir = Path(args.funding_out_dir)
        funding_writer = FundingRateWriter(client=source._client, out_dir=funding_dir)
        added = await funding_writer.update(args.symbol)
        print(f"funding: {added} new rows -> {funding_dir / (args.symbol + '.parquet')}")
    finally:
        await source.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--out-dir", default="data/history")
    ap.add_argument("--funding-out-dir", default="data/funding")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run new tests + full suite**

Run: `pytest tests/unit/scripts/test_download_history.py -v && pytest -q`

Expected: 3 new pass, full suite still green.

- [ ] **Step 5: Add `.gitignore` entries + `.gitkeep` files**

Modify `.gitignore` — add:

```
# Plan 5A: training artefacts
data/history/*.parquet
data/funding/*.parquet
data/training/*.parquet
models/xgb_*.json
models/calib_*.pkl
models/drift_reference.json
```

Then create empty marker files so the directories exist on a fresh clone:

```bash
mkdir -p data/history data/funding data/training models
touch data/history/.gitkeep data/funding/.gitkeep data/training/.gitkeep models/.gitkeep
```

- [ ] **Step 6: Commit**

```bash
git add scripts/download_history.py tests/unit/scripts/__init__.py tests/unit/scripts/test_download_history.py .gitignore data/history/.gitkeep data/funding/.gitkeep data/training/.gitkeep models/.gitkeep
git commit -m "feat(scripts): download_history paginates klines + funding into parquet"
```

- [ ] **Step 7: Manual smoke (network required)**

Run (one time, takes ~3-5 minutes):

```bash
python scripts/download_history.py --years 2
ls -lh data/history data/funding
```

Expected: `data/history/ETHUSDT_1h.parquet` ≈ 17500 rows, `data/funding/ETHUSDT.parquet` ≈ 2200 rows. Note the row counts in the next commit message if you want a paper trail.

---

## Task 6: `scripts/build_training_set.py` — features per bar

**Why:** Training data is `compute_all` over every bar. Doing this offline once is much faster than recomputing during walk-forward CV. Output is a Parquet of flat features keyed by `as_of` timestamp.

**Files:**
- Create: `scripts/build_training_set.py`
- Create: `tests/unit/scripts/test_build_training_set.py`

- [ ] **Step 1: Failing test**

```python
# tests/unit/scripts/test_build_training_set.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.build_training_set import build_training_set
from features.registry import build_default_registry


def _kline_df(n: int) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex([base + timedelta(hours=i) for i in range(n)],
                           name="open_time")
    return pd.DataFrame({
        "open":   [3000.0 + i for i in range(n)],
        "high":   [3010.0 + i for i in range(n)],
        "low":    [2990.0 + i for i in range(n)],
        "close":  [3005.0 + i for i in range(n)],
        "volume": [1.0] * n,
    }, index=idx)


def test_build_training_set_skips_lookback_warmup():
    df = _kline_df(250)
    out = build_training_set(df, registry=build_default_registry(),
                             warmup_bars=200)
    # 250 - 200 = 50 rows
    assert len(out) == 50
    # First row's `as_of` is bar #200 (zero-indexed) = base + 200h
    assert out.index[0] == df.index[200]
    # Index name preserved.
    assert out.index.name == "as_of"


def test_build_training_set_emits_one_column_per_flat_feature():
    df = _kline_df(220)
    out = build_training_set(df, registry=build_default_registry(),
                             warmup_bars=200)
    # smc, fib, liquidity, divergence, funding, confidence each contribute >=1 column
    # (some may flatten to multiple sub-keys; we only require >0).
    assert len(out.columns) > 0
    # Column names are dot-prefixed by feature name.
    assert any(c.startswith("smc.") for c in out.columns)
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/unit/scripts/test_build_training_set.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement the builder**

```python
# scripts/build_training_set.py
"""Computes flat features for every bar of a kline parquet (Plan 5A Task 6).

Reuses XGBPredictor._flatten so column names match what XGBPredictor.load
expects at inference time.

Usage:
    python scripts/build_training_set.py \
        --kline data/history/ETHUSDT_1h.parquet \
        --out   data/training/ETHUSDT_1h_features.parquet \
        --warmup 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features.registry import FeatureRegistry, build_default_registry
from models.xgb_predictor import _flatten


def build_training_set(
    df: pd.DataFrame,
    registry: FeatureRegistry,
    warmup_bars: int = 200,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    index: list = []
    for i, ts in enumerate(df.index):
        if i < warmup_bars:
            continue
        feats = registry.compute_all(df, as_of=ts)
        flat = _flatten(feats)
        rows.append(flat)
        index.append(ts)
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="as_of"))
    return out.fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--symbol", default="ETHUSDT")
    args = ap.parse_args()

    df = pd.read_parquet(args.kline)
    df = df.sort_index()
    reg = build_default_registry(symbol=args.symbol)
    out = build_training_set(df, reg, warmup_bars=args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"training set: {len(out)} rows x {len(out.columns)} cols -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + full suite**

Run: `pytest tests/unit/scripts/test_build_training_set.py -v && pytest -q`

Expected: 2 new pass, full suite green.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_training_set.py tests/unit/scripts/test_build_training_set.py
git commit -m "feat(scripts): build_training_set materialises features per bar"
```

- [ ] **Step 6: Manual smoke (after Task 5 download)**

Run:

```bash
python scripts/build_training_set.py \
    --kline data/history/ETHUSDT_1h.parquet \
    --out   data/training/ETHUSDT_1h_features.parquet
```

Expected: ≈ 17,300 rows × tens of columns. Takes a few minutes (`compute_all` per bar).

---

## Task 7: `scripts/build_labels.py` — 4-bar forward up

**Why:** Labels live in a separate parquet so we can iterate on label rules (triple barrier, varying horizons) without re-running feature computation. Today's label is the spec's baseline: `close[t+4] / close[t] - 1 > 0`.

**Files:**
- Create: `scripts/build_labels.py`
- Create: `tests/unit/scripts/test_build_labels.py`

- [ ] **Step 1: Failing test**

```python
# tests/unit/scripts/test_build_labels.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.build_labels import compute_forward_up_labels


def _df(closes: list[float]) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex([base + timedelta(hours=i) for i in range(len(closes))],
                           name="open_time")
    return pd.DataFrame({"close": closes}, index=idx)


def test_label_is_one_when_future_close_higher():
    df = _df([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    out = compute_forward_up_labels(df, horizon=4)
    # only index 0 is valid (needs t+4)
    assert int(out.iloc[0]) == 1
    # index 1 still in range -> 102 -> 105 +ve
    assert int(out.iloc[1]) == 1
    # last 4 rows have no future close -> NaN
    assert out.iloc[-4:].isna().all()


def test_label_is_zero_when_future_close_equal_or_lower():
    df = _df([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    out = compute_forward_up_labels(df, horizon=4)
    # equal does NOT count as up
    assert int(out.iloc[0]) == 0


def test_label_handles_negative_returns():
    df = _df([100.0, 99.0, 98.0, 97.0, 96.0])
    out = compute_forward_up_labels(df, horizon=4)
    assert int(out.iloc[0]) == 0
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/unit/scripts/test_build_labels.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement labels**

```python
# scripts/build_labels.py
"""Builds binary forward-return labels (Plan 5A Task 7).

Label rule: y_<H>bar_up = 1 if close[t+H] > close[t] else 0.
Horizon defaults to 4 (spec PredictionBundle.horizon_bars).

Usage:
    python scripts/build_labels.py \
        --kline data/history/ETHUSDT_1h.parquet \
        --out   data/training/ETHUSDT_1h_labels.parquet \
        --horizon 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_forward_up_labels(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    future = df["close"].shift(-horizon)
    label = (future > df["close"]).astype("float")
    label[future.isna()] = np.nan
    label.name = f"y_{horizon}bar_up"
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--horizon", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_parquet(args.kline).sort_index()
    y = compute_forward_up_labels(df, horizon=args.horizon)
    out = y.dropna().to_frame()
    out.index.name = "as_of"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"labels: {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + full suite**

Run: `pytest tests/unit/scripts/test_build_labels.py -v && pytest -q`

Expected: 3 new pass, full suite green.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_labels.py tests/unit/scripts/test_build_labels.py
git commit -m "feat(scripts): build_labels — 4-bar forward up binary"
```

- [ ] **Step 6: Manual smoke**

```bash
python scripts/build_labels.py \
    --kline data/history/ETHUSDT_1h.parquet \
    --out   data/training/ETHUSDT_1h_labels.parquet
```

---

## Task 8: Walk-forward XGBoost + isotonic vs Platt A/B

**Why:** This is the meat. We replace the existing `scripts/train_xgb.py` so training does walk-forward CV (5 folds), trains both an `IsotonicRegression` and a `LogisticRegression`-Platt calibrator, picks the winner by **OOS Brier on the last fold**, and writes the bundle plus a small JSON metadata blob describing the comparison. The chosen calibrator is stored in `meta["calibrator"]` so `XGBPredictor.load` works without code changes.

**Files:**
- Modify: `scripts/train_xgb.py` (rewrite, see Step 3)
- Modify: `src/models/xgb_predictor.py` — `load()` reads `meta["calibrator"]` (preferred) and falls back to `meta["isotonic"]` for backward compat with the old skeleton bundles
- Create: `tests/unit/scripts/test_train_xgb.py`
- Create: `tests/unit/models/test_xgb_predictor_load.py`

- [ ] **Step 1: Failing tests for the new training entry point**

```python
# tests/unit/scripts/test_train_xgb.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.train_xgb import (
    walk_forward_calibration_choice,
    train_walk_forward,
)


def _synthetic_xy(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Two informative features + one noise.
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    noise = rng.normal(size=n)
    logits = 0.6 * f1 - 0.4 * f2 + 0.1 * noise
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    X = pd.DataFrame({"a.f1": f1, "a.f2": f2, "b.noise": noise})
    return X, pd.Series(y)


def test_walk_forward_calibration_choice_returns_one_of_two_methods():
    X, y = _synthetic_xy()
    choice = walk_forward_calibration_choice(X, y, n_splits=3)
    assert choice.method in {"isotonic", "platt"}
    assert 0.0 <= choice.brier_isotonic <= 0.5
    assert 0.0 <= choice.brier_platt <= 0.5
    # Picked method matches the lower OOS Brier.
    assert (choice.brier_isotonic <= choice.brier_platt) is (choice.method == "isotonic")


def test_train_walk_forward_writes_bundle(tmp_path):
    X, y = _synthetic_xy()
    bundle_meta = train_walk_forward(
        X=X, y=y,
        out_dir=tmp_path,
        training_window_start="2026-01-01",
        training_window_end="2026-04-01",
        n_splits=3,
    )
    # File artefacts present.
    assert (tmp_path / f"xgb_{bundle_meta.model_version}.json").exists()
    assert (tmp_path / f"calib_{bundle_meta.model_version}.pkl").exists()
    assert (tmp_path / f"meta_{bundle_meta.model_version}.json").exists()
    # Method is in {"isotonic", "platt"}; recorded in meta JSON.
    import json
    meta = json.loads((tmp_path / f"meta_{bundle_meta.model_version}.json").read_text())
    assert meta["calibration_method"] in {"isotonic", "platt"}
    assert meta["feature_order"] == list(X.columns)
```

```python
# tests/unit/models/test_xgb_predictor_load.py
import pickle
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from models.xgb_predictor import XGBPredictor


def _train_tiny_booster() -> xgb.XGBClassifier:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=200) > 0).astype(int)
    booster = xgb.XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss")
    booster.fit(X, y)
    return booster


@pytest.mark.asyncio
async def test_load_uses_calibrator_key(tmp_path):
    booster = _train_tiny_booster()
    booster_path = tmp_path / "xgb_test.json"
    booster.save_model(str(booster_path))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    meta_path = tmp_path / "calib_test.pkl"
    with open(meta_path, "wb") as fh:
        pickle.dump({
            "calibrator": iso,
            "feature_order": ["x.a", "x.b", "x.c"],
        }, fh)

    pred = XGBPredictor.load(str(booster_path), str(meta_path))
    bundle = await pred.predict({"x": {"a": 0.5, "b": 0.1, "c": -0.2}})
    assert 0.0 <= bundle.prob_up <= 1.0
    assert pred.ml_model_version == "test"


@pytest.mark.asyncio
async def test_load_falls_back_to_isotonic_key(tmp_path):
    booster = _train_tiny_booster()
    booster_path = tmp_path / "xgb_legacy.json"
    booster.save_model(str(booster_path))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    meta_path = tmp_path / "calib_legacy.pkl"
    with open(meta_path, "wb") as fh:
        pickle.dump({
            "isotonic": iso,
            "feature_order": ["x.a", "x.b", "x.c"],
        }, fh)

    pred = XGBPredictor.load(str(booster_path), str(meta_path))
    bundle = await pred.predict({"x": {"a": 0.5, "b": 0.1, "c": -0.2}})
    assert 0.0 <= bundle.prob_up <= 1.0
```

- [ ] **Step 2: Run tests to verify fail**

Run: `pytest tests/unit/scripts/test_train_xgb.py tests/unit/models/test_xgb_predictor_load.py -v`

Expected: ImportError on `walk_forward_calibration_choice` / `train_walk_forward`; legacy bundle test passes only if you have not modified `XGBPredictor.load` yet (it currently reads `meta["isotonic"]`), the calibrator-key test fails.

- [ ] **Step 3: Update `XGBPredictor.load` to accept either key**

Edit `src/models/xgb_predictor.py:43-57` to:

```python
    @classmethod
    def load(cls, model_path: str, calib_path: str) -> "XGBPredictor":
        import pickle
        import xgboost as xgb
        booster = xgb.XGBClassifier()
        booster.load_model(model_path)
        with open(calib_path, "rb") as fh:
            meta = pickle.load(fh)
        # Prefer the new "calibrator" key (Plan 5A Task 8); fall back to the
        # original "isotonic" key for bundles produced by the pre-rewrite
        # script.
        calibrator = meta.get("calibrator", meta.get("isotonic"))
        if calibrator is None:
            raise ValueError(f"meta at {calib_path} missing calibrator")
        version = Path(model_path).stem.removeprefix("xgb_")
        return cls(
            _model=booster,
            _calibrator=calibrator,
            _feature_order=tuple(meta["feature_order"]),
            ml_model_version=version,
        )
```

The transform call already uses `self._calibrator.transform(...)` which works for `IsotonicRegression`. For Platt we need to call `predict_proba` instead. Update `_run_model` to dispatch by attribute presence:

```python
    def _run_model(self, features: dict[str, Any]) -> float:
        flat = _flatten(features)
        row = [flat.get(k, 0.0) for k in self._feature_order]
        raw = self._model.predict_proba([row])[0, 1]
        # Isotonic: .transform([raw]) -> array.  Platt (LogisticRegression):
        # .predict_proba([[raw]])[:, 1].
        if hasattr(self._calibrator, "transform"):
            calibrated = self._calibrator.transform([raw])[0]
        else:
            calibrated = self._calibrator.predict_proba([[raw]])[0, 1]
        return float(calibrated)
```

- [ ] **Step 4: Rewrite `scripts/train_xgb.py`**

```python
"""Trains XGBoost on a precomputed features+labels parquet pair, runs
walk-forward CV, picks isotonic vs Platt by OOS Brier (Plan 5A Task 8).

Writes:
    <out_dir>/xgb_<model_version>.json   (booster)
    <out_dir>/calib_<model_version>.pkl  (calibrator + feature_order)
    <out_dir>/meta_<model_version>.json  (training window, calib comparison)
    <out_dir>/drift_reference.json       (overwritten by Task 10)

Inserts a row into model_versions.

Usage:
    python scripts/train_xgb.py \
        --features data/training/ETHUSDT_1h_features.parquet \
        --labels   data/training/ETHUSDT_1h_labels.parquet \
        --out      models
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class CalibrationChoice:
    method: str               # "isotonic" or "platt"
    brier_isotonic: float
    brier_platt: float
    calibrator: object        # the chosen, fit calibrator (last fold)


@dataclass
class BundleMeta:
    model_version: str
    calibration_method: str
    brier_isotonic: float
    brier_platt: float
    feature_order: list[str]


def _fit_booster(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    booster = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", tree_method="hist",
    )
    booster.fit(X_train, y_train)
    return booster


def _fit_isotonic(raw: np.ndarray, y_calib: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y_calib)
    return iso


def _fit_platt(raw: np.ndarray, y_calib: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression()
    lr.fit(raw.reshape(-1, 1), y_calib)
    return lr


def walk_forward_calibration_choice(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> CalibrationChoice:
    """Average OOS Brier across folds for isotonic vs Platt; pick the
    lower one. Returns the chosen calibrator fit on the LAST fold's
    calibration set so it sees the most recent regime."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    bs_iso: list[float] = []
    bs_platt: list[float] = []
    last_iso: IsotonicRegression | None = None
    last_platt: LogisticRegression | None = None
    splits = list(tscv.split(X))
    for fit_idx, calib_idx in splits:
        # Within a fold, split fit_idx in two: first 80% train booster,
        # last 20% fit calibrator. calib_idx is the OOS chunk we score on.
        cut = int(len(fit_idx) * 0.8)
        train_idx, cal_idx = fit_idx[:cut], fit_idx[cut:]
        booster = _fit_booster(X.iloc[train_idx], y.iloc[train_idx])
        raw_cal = booster.predict_proba(X.iloc[cal_idx])[:, 1]
        iso = _fit_isotonic(raw_cal, y.iloc[cal_idx].to_numpy())
        platt = _fit_platt(raw_cal, y.iloc[cal_idx].to_numpy())

        raw_oos = booster.predict_proba(X.iloc[calib_idx])[:, 1]
        p_iso = iso.transform(raw_oos)
        p_platt = platt.predict_proba(raw_oos.reshape(-1, 1))[:, 1]
        bs_iso.append(brier_score_loss(y.iloc[calib_idx], p_iso))
        bs_platt.append(brier_score_loss(y.iloc[calib_idx], p_platt))
        last_iso, last_platt = iso, platt

    avg_iso = float(np.mean(bs_iso))
    avg_platt = float(np.mean(bs_platt))
    if avg_iso <= avg_platt:
        return CalibrationChoice("isotonic", avg_iso, avg_platt, last_iso)
    return CalibrationChoice("platt", avg_iso, avg_platt, last_platt)


def train_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    training_window_start: str,
    training_window_end: str,
    n_splits: int = 5,
) -> BundleMeta:
    choice = walk_forward_calibration_choice(X, y, n_splits=n_splits)

    # Final booster fit on ALL training rows.
    final_booster = _fit_booster(X, y)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_version = hashlib.sha256(
        final_booster.get_booster().save_raw()
    ).hexdigest()[:12]

    booster_path = out_dir / f"xgb_{model_version}.json"
    final_booster.save_model(str(booster_path))
    calib_path = out_dir / f"calib_{model_version}.pkl"
    with open(calib_path, "wb") as fh:
        pickle.dump({
            "calibrator": choice.calibrator,
            "feature_order": list(X.columns),
        }, fh)

    meta = BundleMeta(
        model_version=model_version,
        calibration_method=choice.method,
        brier_isotonic=choice.brier_isotonic,
        brier_platt=choice.brier_platt,
        feature_order=list(X.columns),
    )
    meta_path = out_dir / f"meta_{model_version}.json"
    meta_path.write_text(json.dumps({
        "model_version": meta.model_version,
        "calibration_method": meta.calibration_method,
        "brier_isotonic": meta.brier_isotonic,
        "brier_platt": meta.brier_platt,
        "training_window_start": training_window_start,
        "training_window_end": training_window_end,
        "feature_order": meta.feature_order,
    }, indent=2))

    return meta


def _register(meta: BundleMeta, out_dir: Path,
              window_start: str, window_end: str,
              sqlite_path: str) -> None:
    engine = sa.create_engine(f"sqlite:///{sqlite_path}")
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT OR REPLACE INTO model_versions "
            "(ml_model_version, path, training_window_start, training_window_end, "
            " calibration_method, deployed_at) "
            "VALUES (:mv, :path, :s, :e, :cm, :ts)"
        ), {
            "mv": meta.model_version,
            "path": str(out_dir / f"xgb_{meta.model_version}.json"),
            "s": window_start, "e": window_end, "cm": meta.calibration_method,
            "ts": datetime.now(tz=timezone.utc),
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", default=Path("models"), type=Path)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--sqlite-path", default="data/state.db")
    args = ap.parse_args()

    X = pd.read_parquet(args.features)
    y_df = pd.read_parquet(args.labels)
    # Inner join on as_of -> drop rows where label is NaN.
    joined = X.join(y_df, how="inner")
    label_col = y_df.columns[0]
    X_aligned = joined.drop(columns=[label_col])
    y_aligned = joined[label_col].astype(int)

    meta = train_walk_forward(
        X=X_aligned, y=y_aligned,
        out_dir=args.out,
        training_window_start=str(X_aligned.index.min()),
        training_window_end=str(X_aligned.index.max()),
        n_splits=args.n_splits,
    )
    _register(meta, args.out,
              window_start=str(X_aligned.index.min()),
              window_end=str(X_aligned.index.max()),
              sqlite_path=args.sqlite_path)
    print(f"trained {meta.model_version}; calib={meta.calibration_method} "
          f"brier_iso={meta.brier_isotonic:.4f} brier_platt={meta.brier_platt:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run new tests + full suite**

Run: `pytest tests/unit/scripts/test_train_xgb.py tests/unit/models/test_xgb_predictor_load.py -v && pytest -q`

Expected: 5 new pass, full suite still green. The pre-existing `xgboost`/`libomp` env failure may bite — if it does, install libomp (`brew install libomp`) and re-run.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_xgb.py src/models/xgb_predictor.py tests/unit/scripts/test_train_xgb.py tests/unit/models/test_xgb_predictor_load.py
git commit -m "feat(train): walk-forward XGBoost + isotonic vs Platt by OOS Brier"
```

- [ ] **Step 7: Manual smoke (after Tasks 5-7 produced parquets)**

```bash
python scripts/train_xgb.py \
    --features data/training/ETHUSDT_1h_features.parquet \
    --labels   data/training/ETHUSDT_1h_labels.parquet \
    --out      models
```

Expected stdout: `trained <12-char-hex>; calib=isotonic|platt brier_iso=... brier_platt=...`. A row also lands in `model_versions`.

---

## Task 9: `src/models/registry.py` — `load_latest_model`

**Why:** `wiring.py` Task 1 already references `from models.registry import load_latest_model`. We delivered the bundle layout in Task 8; now the helper that picks the most recently registered version and returns a configured `XGBPredictor`.

**Files:**
- Create: `src/models/registry.py`
- Create: `tests/unit/models/test_registry.py`

- [ ] **Step 1: Failing tests**

```python
# tests/unit/models/test_registry.py
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from models.registry import load_latest_model, list_bundles


def _write_bundle(model_dir: Path, version: str, calibration_method: str = "isotonic") -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)
    booster = xgb.XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss")
    booster.fit(X, y)
    booster.save_model(str(model_dir / f"xgb_{version}.json"))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    with open(model_dir / f"calib_{version}.pkl", "wb") as fh:
        pickle.dump({"calibrator": iso, "feature_order": ["a.x", "a.y"]}, fh)
    (model_dir / f"meta_{version}.json").write_text(json.dumps({
        "model_version": version,
        "calibration_method": calibration_method,
        "feature_order": ["a.x", "a.y"],
    }))


def test_list_bundles_returns_versions_in_order(tmp_path):
    _write_bundle(tmp_path, "aaaa00000001")
    _write_bundle(tmp_path, "bbbb00000002")
    bundles = list_bundles(tmp_path)
    assert [b.version for b in bundles] == ["aaaa00000001", "bbbb00000002"]


def test_load_latest_model_returns_xgb_predictor(tmp_path):
    _write_bundle(tmp_path, "aaaa00000001")
    _write_bundle(tmp_path, "bbbb00000002")
    pred = load_latest_model(tmp_path)
    assert pred.ml_model_version == "bbbb00000002"


def test_load_latest_model_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_latest_model(tmp_path)
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/unit/models/test_registry.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement registry**

```python
# src/models/registry.py
"""Model bundle discovery + load helpers (Plan 5A Task 9).

Bundle layout (written by scripts/train_xgb.py):
    <model_dir>/xgb_<version>.json
    <model_dir>/calib_<version>.pkl
    <model_dir>/meta_<version>.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from models.xgb_predictor import XGBPredictor


@dataclass(frozen=True)
class BundleHandle:
    version: str
    booster_path: Path
    calib_path: Path
    meta_path: Path
    mtime: float


def list_bundles(model_dir: Path) -> list[BundleHandle]:
    """Returns bundle handles sorted by booster mtime (oldest first)."""
    out: list[BundleHandle] = []
    for booster in sorted(model_dir.glob("xgb_*.json")):
        version = booster.stem.removeprefix("xgb_")
        calib = model_dir / f"calib_{version}.pkl"
        meta = model_dir / f"meta_{version}.json"
        if not calib.exists() or not meta.exists():
            continue
        out.append(BundleHandle(
            version=version,
            booster_path=booster,
            calib_path=calib,
            meta_path=meta,
            mtime=booster.stat().st_mtime,
        ))
    out.sort(key=lambda b: b.mtime)
    return out


def load_latest_model(model_dir: Path) -> XGBPredictor:
    bundles = list_bundles(model_dir)
    if not bundles:
        raise FileNotFoundError(f"No model bundles in {model_dir}")
    latest = bundles[-1]
    return XGBPredictor.load(str(latest.booster_path), str(latest.calib_path))


def load_meta(bundle: BundleHandle) -> dict:
    return json.loads(bundle.meta_path.read_text())
```

- [ ] **Step 4: Run tests + full suite**

Run: `pytest tests/unit/models/test_registry.py -v && pytest -q`

Expected: 3 new pass, full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/models/registry.py tests/unit/models/test_registry.py
git commit -m "feat(models): registry — list_bundles + load_latest_model"
```

---

## Task 10: Drift reference baseline

**Why:** `FeatureDriftMonitor.reference` is `{}` today, so `has_breach` is a no-op. We compute the reference from the same training feature parquet (= the distribution the model was calibrated on) and persist it as JSON next to the model bundle. `wiring.py` then loads it into the monitor.

**Files:**
- Modify: `scripts/train_xgb.py` — also write `models/drift_reference.json` (one entry per feature column) at the end of `train_walk_forward`
- Modify: `src/wiring.py` — load `cfg.drift_reference_path` if present, populate `FeatureDriftMonitor.reference`
- Create: `tests/unit/scripts/test_drift_reference.py`
- Create: `tests/unit/test_wiring_drift_reference.py`

- [ ] **Step 1: Failing test for the writer side**

```python
# tests/unit/scripts/test_drift_reference.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from scripts.train_xgb import write_drift_reference


def test_write_drift_reference_writes_one_entry_per_column(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a.x": rng.normal(size=500),
        "a.y": rng.normal(size=500),
    })
    out = tmp_path / "drift_reference.json"
    write_drift_reference(X, out)
    blob = json.loads(out.read_text())
    assert set(blob.keys()) == {"a.x", "a.y"}
    # Each entry is a sample of values (capped) we can use as PSI/KS reference.
    for vals in blob.values():
        assert isinstance(vals, list)
        assert 0 < len(vals) <= 5000
```

- [ ] **Step 2: Failing test for the wiring side**

```python
# tests/unit/test_wiring_drift_reference.py
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import sqlalchemy as sa

from orchestrator import OrchestratorConfig
from wiring import build_scan_context


def _fake_klines() -> pd.DataFrame:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    idx = [base + timedelta(hours=i) for i in range(50)]
    return pd.DataFrame({
        "open": [3000.0] * 50, "high": [3010.0] * 50,
        "low": [2990.0] * 50, "close": [3005.0] * 50, "volume": [1.0] * 50,
    }, index=pd.DatetimeIndex(idx, name="open_time"))


@pytest.mark.asyncio
async def test_drift_reference_loaded_from_json(tmp_path):
    drift_path = tmp_path / "drift.json"
    drift_path.write_text(json.dumps({"a.x": [0.1, 0.2, 0.3]}))
    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        drift_reference_path=str(drift_path),
        use_trained_model=False,
    )
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{cfg.sqlite_path}")
    alembic.command.upgrade(ac, "head")
    engine = sa.create_engine(f"sqlite:///{cfg.sqlite_path}")

    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=_fake_klines())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        _, lifecycle = await build_scan_context(cfg, engine)

    monitor = lifecycle["drift_monitor"]
    assert "a.x" in monitor.reference
```

- [ ] **Step 3: Implement `write_drift_reference` in `scripts/train_xgb.py`**

Add to `scripts/train_xgb.py` (above `train_walk_forward`):

```python
def write_drift_reference(X: pd.DataFrame, out_path: Path,
                          max_samples: int = 5000) -> None:
    """Persists per-column samples for use as FeatureDriftMonitor reference."""
    rng = np.random.default_rng(0)
    blob: dict[str, list[float]] = {}
    for col in X.columns:
        vals = X[col].dropna().to_numpy()
        if len(vals) > max_samples:
            idx = rng.choice(len(vals), size=max_samples, replace=False)
            vals = vals[idx]
        blob[col] = [float(v) for v in vals]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob))
```

Wire it into `train_walk_forward`'s tail:

```python
    write_drift_reference(X, out_dir / "drift_reference.json")
    return meta
```

- [ ] **Step 4: Implement loader side in `src/wiring.py`**

Replace the `FeatureDriftMonitor(reference={}, ...)` block with:

```python
    reference: dict[str, list[float]] = {}
    drift_ref_path = Path(cfg.drift_reference_path)
    if drift_ref_path.exists():
        import json as _json
        reference = _json.loads(drift_ref_path.read_text())

    drift_monitor = FeatureDriftMonitor(
        reference=reference,
        psi_threshold=drift_cfg.psi_threshold,
        ks_threshold=drift_cfg.ks_threshold,
        n_bins=drift_cfg.psi_bins,
    )
```

- [ ] **Step 5: Run tests + full suite**

Run: `pytest tests/unit/scripts/test_drift_reference.py tests/unit/test_wiring_drift_reference.py -v && pytest -q`

Expected: 2 new pass, full suite green.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_xgb.py src/wiring.py tests/unit/scripts/test_drift_reference.py tests/unit/test_wiring_drift_reference.py
git commit -m "feat(drift): persist training distribution as drift reference + wire into monitor"
```

---

## Task 11: End-to-end smoke with real data + real model

**Why:** Final acceptance — boot the orchestrator with `use_trained_model=True` against a fake `BinanceKline` that returns synthetic but plausible klines, run one scan tick, verify a row appears in `proposals` (even if rejected — what matters is that `Ensemble.predict` ran the trained model and `_scan_symbol` reached the `proposal_repo.insert` line).

**Files:**
- Create: `tests/e2e/test_real_data_smoke.py`
- Create: `docs/superpowers/plans/2026-04-25-pivot-plan5a-STATUS.md` (handoff)

- [ ] **Step 1: Write the smoke test**

```python
# tests/e2e/test_real_data_smoke.py
"""Plan 5A end-to-end smoke: trained model + real-shaped klines + one scan."""
from __future__ import annotations

import asyncio
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from orchestrator import Orchestrator, OrchestratorConfig
from pipeline import scheduled_macro_scan


def _fake_klines() -> pd.DataFrame:
    # 250 bars of plausible ETHUSDT 1h.
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    closes = 3000.0 + np.cumsum(rng.normal(0, 5.0, size=250))
    return pd.DataFrame({
        "open":   closes,
        "high":   closes + 5,
        "low":    closes - 5,
        "close":  closes,
        "volume": np.full(250, 1.0),
    }, index=pd.DatetimeIndex([base + timedelta(hours=i) for i in range(250)],
                               name="open_time"))


def _seed_model_dir(model_dir: Path, feature_order: list[str]) -> str:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(feature_order)))
    y = (X[:, 0] > 0).astype(int)
    booster = xgb.XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss")
    booster.fit(X, y)
    version = "smoke00000001"
    booster.save_model(str(model_dir / f"xgb_{version}.json"))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    with open(model_dir / f"calib_{version}.pkl", "wb") as fh:
        pickle.dump({"calibrator": iso, "feature_order": feature_order}, fh)
    (model_dir / f"meta_{version}.json").write_text(json.dumps({
        "model_version": version,
        "calibration_method": "isotonic",
        "feature_order": feature_order,
    }))
    return version


@pytest.mark.e2e
async def test_real_data_smoke_inserts_proposal_row(tmp_path):
    # Discover the actual feature_order our registry would emit for these
    # synthetic klines (avoids hardcoding the column list).
    from features.registry import build_default_registry
    from models.xgb_predictor import _flatten
    feats = build_default_registry().compute_all(_fake_klines(),
                                                  as_of=_fake_klines().index[-1])
    feature_order = sorted(_flatten(feats).keys())
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _seed_model_dir(model_dir, feature_order)

    cfg = OrchestratorConfig(
        sqlite_path=str(tmp_path / "state.db"),
        halt_file=str(tmp_path / "HALT"),
        drift_yaml="config/drift.yaml",
        use_trained_model=True,
        model_dir=str(model_dir),
        drift_reference_path=str(tmp_path / "missing.json"),  # absent on purpose
        ollama_host="http://127.0.0.1:0",  # no Ollama; Ensemble fallback path
        long_threshold=0.0,                # accept any prob_up so we get a proposal
        short_threshold=0.0,
    )

    orch = Orchestrator(cfg)
    fake_kline = AsyncMock()
    fake_kline.fetch_latest = AsyncMock(return_value=_fake_klines())
    fake_kline.close = AsyncMock()
    with patch("data.binance_kline.BinanceKline.open",
               new=AsyncMock(return_value=fake_kline)):
        await orch.boot()
    assert orch.ctx is not None

    await scheduled_macro_scan(orch.ctx, trace_id="smoke")

    engine = sa.create_engine(f"sqlite:///{cfg.sqlite_path}")
    with engine.begin() as conn:
        row_count = conn.execute(sa.text("SELECT COUNT(*) FROM proposals")).scalar()
    assert row_count >= 1, "no proposal rows after scheduled_macro_scan"

    await fake_kline.close()
```

- [ ] **Step 2: Run test to verify fail**

Run: `pytest tests/e2e/test_real_data_smoke.py -v`

Expected: depending on what's wired, either:
- the test passes already (Tasks 1-10 covered everything) — proceed,
- or fails because `LLM_UNAVAILABLE_MARKER` triggers but RiskPipeline rejects the proposal so no row gets persisted.

The pipeline currently inserts even rejected proposals (`pipeline.py:123`), so the row should land. If it doesn't, debug at the proposal step.

- [ ] **Step 3: If test fails, narrow down**

Add `pytest --capture=no` to see structlog output. Common fixes:
- `feature_order` mismatch: the trained model has an empty intersection with current registry features. Re-derive `feature_order` from `build_default_registry().compute_all(...)` (the test does this; if it still drifts, the registry has changed and `_flatten` ordering must be deterministic).
- `OllamaClient` hangs: confirm `ollama.start()` is NOT called in this test (we never invoked `orch.run()`).

- [ ] **Step 4: Run full suite**

Run: `pytest -q`

Expected: 254+ tests pass (depending on how many net-new tests Tasks 1-10 added). The pre-existing `xgboost`/`libomp` env failure may still show — that's not Plan 5A's regression.

- [ ] **Step 5: Commit smoke**

```bash
git add tests/e2e/test_real_data_smoke.py
git commit -m "test(e2e): trained-model + fake-binance smoke -> proposal persisted"
```

- [ ] **Step 6: Write Plan 5A STATUS**

Create `docs/superpowers/plans/2026-04-25-pivot-plan5a-STATUS.md`. Sections to include:
- Summary (1 paragraph)
- Task table (11 rows; commit hashes)
- Verification (test counts; manual smoke results: kline parquet rows, training rows, model_version, brier scores)
- Decisions landed (calibration winner, drift reference baseline path)
- What's NOT done (Plan 5B scope reminders)
- Known follow-ups

- [ ] **Step 7: Final commit**

```bash
git add docs/superpowers/plans/2026-04-25-pivot-plan5a-STATUS.md
git commit -m "docs: Plan 5A handoff STATUS"
```

---

## Out-of-band manual smoke (after Tasks 5-8 land)

Once code is committed, the operator runs the actual training (one shot, ~10 min) and confirms the trained model loads cleanly. This is the moment when "real predictions" stops being theoretical:

```bash
# 1. Download history (network; ~3-5 min)
python scripts/download_history.py --years 2

# 2. Build features (CPU bound; ~2-3 min)
python scripts/build_training_set.py \
    --kline data/history/ETHUSDT_1h.parquet \
    --out   data/training/ETHUSDT_1h_features.parquet

# 3. Build labels (~5s)
python scripts/build_labels.py \
    --kline data/history/ETHUSDT_1h.parquet \
    --out   data/training/ETHUSDT_1h_labels.parquet

# 4. Train (~3-5 min)
python scripts/train_xgb.py \
    --features data/training/ETHUSDT_1h_features.parquet \
    --labels   data/training/ETHUSDT_1h_labels.parquet \
    --out      models

# 5. Boot the orchestrator with use_trained_model=True
#    (set via env/config; for first poke use python -c)
python -c "
import asyncio
from orchestrator import Orchestrator, OrchestratorConfig
cfg = OrchestratorConfig(use_trained_model=True)
orch = Orchestrator(cfg)
async def main():
    await orch.boot()
    from pipeline import scheduled_macro_scan
    await scheduled_macro_scan(orch.ctx, trace_id='manual')
asyncio.run(main())
"

# 6. Inspect
sqlite3 data/state.db "SELECT proposal_id, direction, confidence, ml_model_version FROM proposals ORDER BY ts DESC LIMIT 1;"
```

If the row shows a non-`stub-v0` `ml_model_version` and a confidence != 0.55, you have made your first real prediction.

---

## Self-review notes

- **Spec coverage**: every Plan 5A goal in the prior STATUS doc maps to a task here (real mid/atr/spread → Tasks 2-4; BinanceKline async → Task 1; drift reference → Task 10; trained model → Tasks 5-9). Spec §4.3 calibration A/B (Q1) is Task 8. Spec §10.1 gate 3 needs the Brier-score artefact, also Task 8. ReplayBroker / LiveBroker / Pre-Live Gate / 60-day paper / DSR are explicitly out of scope (Plan 5B).
- **Type consistency**: `XGBPredictor.load`'s `meta` accepts both `"calibrator"` (new) and `"isotonic"` (legacy). `BundleHandle` is the only new public dataclass; matches `list_bundles → load_latest_model` chain. `OrchestratorConfig` gains exactly three fields (`use_trained_model`, `model_dir`, `drift_reference_path`); all consumers updated.
- **No placeholders**: every script has full implementation; every test has assertions; no "implement later" markers; the only deferred items are explicitly tagged Plan 5B in the doc header and STATUS.
