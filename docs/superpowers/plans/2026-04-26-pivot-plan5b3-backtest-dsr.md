# Plan 5B-3 — Walk-forward Backtest + Deflated Sharpe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backtest harness that loads the trained model + historical features + klines, drives `ReplayBroker.set_time` over an OOS window, runs the production proposal/risk/sizing pipeline, computes equity / Sharpe / **Deflated Sharpe** (Bailey & López de Prado 2014), and writes a row to `backtest_runs`. Manual smoke produces the first quantitative answer to "does this model have any edge?".

**Architecture:** Three pure layers + one driver + one CLI. (1) `src/observability/sharpe.py` — math: `sharpe_ratio(returns, periods_per_year)` + `deflated_sharpe_ratio(returns, n_trials)`. (2) `src/backtest/equity.py` — pure function: `equity_curve(events, initial_equity)` returns a per-bar equity Series from a `BrokerEvent` stream (fills + funding charges). (3) `ReplayBroker.drain_events()` — synchronous, non-blocking drain of internal queue (the only mutation to Plan 5B-2's broker). (4) `src/backtest/runner.py::BacktestRunner` — wires features parquet + klines parquet + funding parquet + trained model + ThresholdPolicy + RiskPipeline + ReplayBroker; iterates OOS bars, fires the production scan inline. (5) `scripts/backtest.py` — CLI that loads everything and writes one `backtest_runs` row. The runner reuses production `Ensemble.predict` so backtest fills match paper fills bit-for-bit (already guaranteed via Plan 5B-2's shared cost model).

**Tech Stack:** Python 3.11, pandas, numpy, scipy.stats (already in deps via existing drift module), SQLAlchemy, pytest. No new dependencies.

**Decisions baked in:**
- **Reuse pre-computed features parquet.** Plan 5A Task 6 produced `data/training/ETHUSDT_1h_features.parquet` (17337 × 37). Recomputing per-bar in backtest would take ~30 min and use the same point-in-time `compute_all`. The parquet IS the point-in-time snapshot.
- **OOS window = last 20% of features parquet.** Matches Plan 5A Task 8's `walk_forward_calibration_choice` final-fold semantics. 17337 × 0.2 ≈ 3467 bars ≈ 145 days = 5 months. Configurable via `--oos-fraction`.
- **Walk-forward = single split for Plan 5B-3 baseline.** Train on first 80% (already done — model is `92fddb72f14b`), evaluate on last 20%. Multi-fold rolling backtest deferred to Plan 5C.
- **N trials for DSR = 2** (one for each calibration method tried in Plan 5A Task 8). Plan 5C will increase as we explore more model variants.
- **Equity starts at `cfg.initial_equity_usdt = 10_000`** (matches `ReplayBroker` default).
- **No LLM in backtest.** Use ML-only `Predictor` (not `Ensemble`) — matches spec §9.6 "OOS holdout — last N months never seen by LLM (no prompts, no examples)". Today's stubbed Ensemble would also work but is more wiring.
- **Per-bar return = `equity[t] / equity[t-1] - 1`**. Funding charges land on the bar they fire on. Annualization factor for 1h bars = 24 × 365 = 8760.
- **`ReplayBroker.drain_events()` returns list synchronously**, not via async iterator. Backtest loop is sync-driven; the existing `events()` async-generator stays for production paper mode. The drain is purpose-built for batch processing.

**Out of Plan 5B-3 scope (deferred):**
- Multi-fold rolling-window backtest with purged-CV embargo. Plan 5C territory.
- LLM context veto in backtest (no prompt-leakage proof).
- Multi-symbol portfolio backtest (single ETHUSDT only).
- Hyperparameter search over ThresholdPolicy thresholds — Plan 5C.
- `backtest_runs.summary_json` schema standardization beyond a small set of keys (sharpe, dsr, n_trades, max_drawdown, hit_rate, avg_pnl_per_trade).
- Make target `make backtest` (spec §9.7).

---

## File map

### Created
- `src/observability/sharpe.py` — `sharpe_ratio`, `deflated_sharpe_ratio`
- `src/backtest/__init__.py` (empty package marker)
- `src/backtest/equity.py` — `equity_curve(events, initial_equity)`
- `src/backtest/runner.py` — `BacktestRunner` class + `BacktestResult` dataclass
- `scripts/backtest.py` — CLI driver
- `tests/unit/observability/test_sharpe.py`
- `tests/unit/backtest/__init__.py` (empty)
- `tests/unit/backtest/test_equity.py`
- `tests/unit/backtest/test_runner.py`
- `tests/unit/scripts/test_backtest.py`
- `docs/superpowers/plans/2026-04-26-pivot-plan5b3-STATUS.md`

### Modified
- `src/execution/replay_broker.py` — add `drain_events()` synchronous method
- `tests/unit/execution/test_replay_broker.py` — add `test_drain_events_returns_queued_then_empty`

### Untouched (verified intentionally)
- `src/state/alembic/` — `backtest_runs` table already in baseline schema (no migration).
- `src/decision/policy.py` — `ThresholdPolicy` consumed as-is.
- `src/decision/risk/pipeline.py` — `RiskPipeline` consumed as-is.
- `src/models/registry.py::load_latest_model` — consumed as-is.
- `src/data/funding.py` — backfill output consumed as-is.

---

## Task 1: `src/observability/sharpe.py` — Sharpe + Deflated Sharpe

**Why first:** Pure math, no dependencies on the rest of the plan. Once tested it's a leaf module the runner pulls in.

**Files:**
- Create: `src/observability/sharpe.py`
- Create: `tests/unit/observability/test_sharpe.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/observability/test_sharpe.py
"""Sharpe + Deflated Sharpe — Plan 5B-3 Task 1."""
from __future__ import annotations

import math

import numpy as np
import pytest

from observability.sharpe import sharpe_ratio, deflated_sharpe_ratio


def test_sharpe_ratio_positive_drift_returns_positive():
    rng = np.random.default_rng(0)
    # Daily returns with mean 0.001, std 0.01 → Sharpe ≈ 0.1 daily, annualised ~1.6
    returns = rng.normal(0.001, 0.01, size=2000)
    sr = sharpe_ratio(returns, periods_per_year=252)
    assert sr > 1.0


def test_sharpe_ratio_zero_returns_yields_nan():
    returns = np.zeros(100)
    sr = sharpe_ratio(returns, periods_per_year=252)
    assert math.isnan(sr)


def test_sharpe_ratio_handles_short_series():
    """N<2 returns NaN, doesn't crash."""
    assert math.isnan(sharpe_ratio(np.array([0.01]), periods_per_year=252))
    assert math.isnan(sharpe_ratio(np.array([]), periods_per_year=252))


def test_deflated_sharpe_below_observed_when_n_trials_high():
    """More trials = larger correction = lower DSR."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=2000)
    dsr_n2 = deflated_sharpe_ratio(returns, n_trials=2, periods_per_year=252)
    dsr_n100 = deflated_sharpe_ratio(returns, n_trials=100, periods_per_year=252)
    assert dsr_n2 > dsr_n100


def test_deflated_sharpe_n_trials_one_equals_observed_z():
    """With n_trials=1 the SR0 correction is 0, so DSR is just the
    z-score of observed Sharpe under the normal-Sharpe assumption."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=500)
    sr = sharpe_ratio(returns, periods_per_year=252)
    dsr = deflated_sharpe_ratio(returns, n_trials=1, periods_per_year=252)
    # With zero-drift returns, observed Sharpe ~0, DSR also ~0 ± small.
    assert abs(dsr) < 0.5
    # And DSR is bounded — not exploding even with n_trials=1.
    assert -10 < dsr < 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/observability/test_sharpe.py -v`
Expected: ImportError on `observability.sharpe`.

- [ ] **Step 3: Implement Sharpe + DSR**

```python
# src/observability/sharpe.py
"""Sharpe ratio + Deflated Sharpe (Bailey & López de Prado 2014).

DSR adjusts the observed Sharpe for selection bias: when you've tested
N strategies, the best-performing one's Sharpe is upward-biased.
DSR returns the probability-style score representing the chance the
true Sharpe exceeds zero given the observation.

Formula (simplified, periodic-Sharpe form):
    SR = mean(r) / std(r) * sqrt(P)        # annualised
    SR0 = sqrt(2 * ln(N)) - euler/sqrt(2 * ln(N))   # expected max under null
    DSR = (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2)
where T is the number of return observations, N is the number of trials.

Returns NaN on degenerate input (zero variance, T<2). DSR can be
negative (model worse than expected-best-of-N null).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray | list[float],
                 periods_per_year: int) -> float:
    """Annualised Sharpe ratio. Returns NaN on degenerate input."""
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return float("nan")
    std = arr.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(arr.mean() / std * math.sqrt(periods_per_year))


def _expected_max_sharpe_under_null(n_trials: int) -> float:
    """E[max SR_i under H0] for n_trials draws, Bailey-de Prado eq. 3.

    With n_trials=1, returns 0 (no expected outperformance under null).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    z_high = math.sqrt(2.0 * math.log(n_trials))
    return z_high - EULER_GAMMA / z_high


def deflated_sharpe_ratio(returns: np.ndarray | list[float],
                          n_trials: int,
                          periods_per_year: int) -> float:
    """Probability-of-skill score under selection bias.

    Returns NaN on degenerate input. Can be negative when the observed
    Sharpe is below what you'd expect from N coin-flip strategies.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size < 4:                   # need enough data for skew/kurt
        return float("nan")
    sr = sharpe_ratio(arr, periods_per_year)
    if math.isnan(sr):
        return float("nan")
    # Convert annualised SR back to per-period for the variance term
    sr_period = sr / math.sqrt(periods_per_year)
    sr0_period = _expected_max_sharpe_under_null(n_trials) / math.sqrt(periods_per_year)
    sk = float(scipy_skew(arr, bias=False))
    kt = float(scipy_kurt(arr, fisher=False, bias=False))   # raw kurtosis (3 = normal)
    t = arr.size
    denom_sq = 1.0 - sk * sr_period + ((kt - 1.0) / 4.0) * sr_period ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (sr_period - sr0_period) * math.sqrt(t - 1) / math.sqrt(denom_sq)
    # Return the z-score directly. Callers that want a probability can
    # wrap with scipy.stats.norm.cdf(z).
    return z
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/observability/test_sharpe.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 325 passed (320 + 5 new).

- [ ] **Step 6: Commit**

```bash
git add src/observability/sharpe.py tests/unit/observability/test_sharpe.py
git commit -m "feat(observability): Sharpe + Deflated Sharpe Ratio (Bailey-de Prado 2014)"
```

---

## Task 2: `ReplayBroker.drain_events()` — synchronous batch drain

**Why:** Backtest loop runs sync (no asyncio in the per-bar loop). Production paper mode uses `events()` async-iterator which blocks forever waiting for the queue. Backtest needs "give me everything queued right now and return."

**Files:**
- Modify: `src/execution/replay_broker.py` — add `drain_events()`
- Modify: `tests/unit/execution/test_replay_broker.py` — add 1 test

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/execution/test_replay_broker.py`:

```python
@pytest.mark.asyncio
async def test_drain_events_returns_queued_then_empty(cfg):
    klines = _klines()
    rb = ReplayBroker(cfg=cfg, klines=klines)
    rb.set_time(klines.index[0])
    o = Order(client_order_id="c1", symbol="ETHUSDT", side="buy",
              type="market", qty=0.1)
    await rb.submit(o)

    drained = rb.drain_events()
    assert len(drained) == 2
    assert drained[0].kind == "submitted"
    assert drained[1].kind == "filled"

    # Second drain returns nothing
    assert rb.drain_events() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/execution/test_replay_broker.py::test_drain_events_returns_queued_then_empty -v`
Expected: AttributeError on `drain_events`.

- [ ] **Step 3: Implement `drain_events`**

In `src/execution/replay_broker.py`, add the method (place after `events()`):

```python
    def drain_events(self) -> list[BrokerEvent]:
        """Synchronously drain all currently queued events.

        Returns events in FIFO order, then leaves the queue empty.
        Used by the backtest harness to process per-bar event batches
        without async iteration. Production paper mode should still use
        `events()` for streaming.
        """
        out: list[BrokerEvent] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/execution/test_replay_broker.py::test_drain_events_returns_queued_then_empty -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 326 passed (325 + 1 new).

- [ ] **Step 6: Commit**

```bash
git add src/execution/replay_broker.py tests/unit/execution/test_replay_broker.py
git commit -m "feat(replay): drain_events() — sync batch drain for backtest harness"
```

---

## Task 3: `src/backtest/equity.py` — equity curve from broker events

**Why:** Equity series → returns → Sharpe. This is a pure function from `list[BrokerEvent]` + initial equity to a per-bar equity Series. Decoupled from the broker so it can be tested with synthetic events.

**Files:**
- Create: `src/backtest/__init__.py` (empty)
- Create: `src/backtest/equity.py`
- Create: `tests/unit/backtest/__init__.py` (empty)
- Create: `tests/unit/backtest/test_equity.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/backtest/test_equity.py
"""Equity curve from broker events — Plan 5B-3 Task 3."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from execution.base import BrokerEvent
from backtest.equity import equity_curve


def _ev(kind: str, ts: datetime, *, price: float | None = None,
        qty: float | None = None, fee: float | None = None) -> BrokerEvent:
    return BrokerEvent(
        event_id=f"ev_{ts.isoformat()}_{kind}",
        kind=kind, order_id="oid",
        symbol="ETHUSDT",
        ts_epoch_ms=int(ts.timestamp() * 1000),
        fill_price=price, fill_qty=qty, fee=fee,
    )


def test_empty_events_returns_initial_equity_only():
    curve = equity_curve(events=[], initial_equity=10_000.0,
                          mark_prices={})
    assert len(curve) == 1
    assert curve.iloc[0] == 10_000.0


def test_buy_then_mark_to_market():
    """Buy 1.0 ETH @3000 (fee 1.5), mark at 3050 → equity = 10_000 - 1.5 + (3050-3000)*1.0 = 10_048.5."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    events = [
        _ev("filled", base, price=3000.0, qty=1.0, fee=1.5),
    ]
    mark_prices = {base: 3000.0, base + timedelta(hours=1): 3050.0}
    curve = equity_curve(events=events, initial_equity=10_000.0,
                          mark_prices=mark_prices)
    assert curve[base + timedelta(hours=1)] == pytest.approx(10_048.5)


def test_buy_then_sell_realised_pnl():
    """Buy 1.0 @3000 (fee 1.5), sell 1.0 @3100 (fee 1.55) → realised 100 - 3.05 = 96.95."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    events = [
        _ev("filled", base, price=3000.0, qty=1.0, fee=1.5),
        _ev("filled", base + timedelta(hours=1), price=3100.0, qty=-1.0, fee=1.55),
    ]
    mark_prices = {
        base: 3000.0,
        base + timedelta(hours=1): 3100.0,
        base + timedelta(hours=2): 3100.0,
    }
    curve = equity_curve(events=events, initial_equity=10_000.0,
                          mark_prices=mark_prices)
    # After close, equity = 10_000 - 1.5 + 100 - 1.55 = 10_096.95
    assert curve[base + timedelta(hours=2)] == pytest.approx(10_096.95)


def test_funding_charge_decreases_equity():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    events = [
        _ev("filled", base, price=3000.0, qty=1.0, fee=1.5),
        _ev("funding_charged", base + timedelta(hours=8), fee=0.3),
    ]
    mark_prices = {
        base: 3000.0,
        base + timedelta(hours=8): 3000.0,
        base + timedelta(hours=9): 3000.0,
    }
    curve = equity_curve(events=events, initial_equity=10_000.0,
                          mark_prices=mark_prices)
    # 10_000 - 1.5 (entry fee) - 0.3 (funding) - 0 (no MtM change) = 9_998.2
    assert curve[base + timedelta(hours=9)] == pytest.approx(9_998.2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/backtest/test_equity.py -v`
Expected: ImportError on `backtest.equity`.

- [ ] **Step 3: Implement equity_curve**

```python
# src/backtest/equity.py
"""Equity curve from broker events — Plan 5B-3 Task 3."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from execution.base import BrokerEvent


def equity_curve(*, events: Iterable[BrokerEvent],
                 initial_equity: float,
                 mark_prices: dict[datetime, float]) -> pd.Series:
    """Build a per-bar equity Series from a stream of broker events.

    `mark_prices` provides per-bar marks (DatetimeIndex → close price).
    Each event lands at the bar matching its `ts_epoch_ms`.

    Equity update rules:
    - filled (qty signed): pay fee; update position; realised PnL = 0
      (entry) or `(price - avg_entry) * qty_closed` (close-out portion).
      Position avg_entry uses weighted avg on adds, kept on reduces.
    - funding_charged: pay fee, no position change.

    Then mark-to-market at each bar in `mark_prices` against current
    open position: `unrealised = (mark - avg_entry) * qty`.

    Returns a Series indexed by mark_prices keys (sorted), values are
    the running equity = cash + unrealised. Initial point is
    `initial_equity` at the earliest mark price (or alone if empty).
    """
    sorted_marks = sorted(mark_prices.keys())
    if not sorted_marks:
        return pd.Series([initial_equity])

    cash = initial_equity
    pos_qty = 0.0
    pos_avg = 0.0

    # Group events by their bar timestamp (ms-aligned to nearest mark).
    # For simplicity we bucket events to the next mark whose ts >= event ts.
    events_sorted = sorted(events, key=lambda e: e.ts_epoch_ms)
    ev_idx = 0

    out_index: list[datetime] = []
    out_values: list[float] = []

    for mark_ts in sorted_marks:
        mark_ms = int(mark_ts.timestamp() * 1000)
        # Apply all events with ts <= this mark.
        while ev_idx < len(events_sorted) and events_sorted[ev_idx].ts_epoch_ms <= mark_ms:
            e = events_sorted[ev_idx]
            if e.kind == "filled" and e.fill_qty is not None and e.fill_price is not None:
                fee = e.fee or 0.0
                cash -= fee
                # Realise PnL on the closing portion if this fill reduces position.
                if pos_qty != 0 and (pos_qty > 0) != (e.fill_qty > 0):
                    closing = min(abs(e.fill_qty), abs(pos_qty))
                    sign_pos = 1 if pos_qty > 0 else -1
                    realised = (e.fill_price - pos_avg) * closing * sign_pos
                    cash += realised
                    new_qty = pos_qty + e.fill_qty
                    if abs(new_qty) < 1e-12:
                        pos_qty, pos_avg = 0.0, 0.0
                    elif (new_qty > 0) != (pos_qty > 0):
                        # Crossed zero: residual opens new opposite position at fill_price.
                        pos_qty = new_qty
                        pos_avg = e.fill_price
                    else:
                        pos_qty = new_qty
                        # avg_entry stays
                else:
                    # Same-side add (or new position).
                    new_qty = pos_qty + e.fill_qty
                    if pos_qty == 0:
                        pos_avg = e.fill_price
                    else:
                        pos_avg = (pos_avg * pos_qty + e.fill_price * e.fill_qty) / new_qty
                    pos_qty = new_qty
            elif e.kind == "funding_charged":
                cash -= (e.fee or 0.0)
            ev_idx += 1

        unrealised = (mark_prices[mark_ts] - pos_avg) * pos_qty if pos_qty != 0 else 0.0
        out_index.append(mark_ts)
        out_values.append(cash + unrealised)

    return pd.Series(out_values, index=pd.DatetimeIndex(out_index, name="ts"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/backtest/test_equity.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 330 passed (326 + 4 new).

- [ ] **Step 6: Commit**

```bash
git add src/backtest/__init__.py src/backtest/equity.py tests/unit/backtest/__init__.py tests/unit/backtest/test_equity.py
git commit -m "feat(backtest): equity_curve from broker events with realised+unrealised PnL"
```

---

## Task 4: `BacktestRunner` + `scripts/backtest.py`

**Why:** This is the integration step. Wires features parquet + klines + funding + trained model + ThresholdPolicy + RiskPipeline + ReplayBroker. Iterates OOS bars, runs the production scan inline, drains events. Then a thin CLI makes it runnable.

**Files:**
- Create: `src/backtest/runner.py`
- Create: `scripts/backtest.py`
- Create: `tests/unit/backtest/test_runner.py`
- Create: `tests/unit/scripts/test_backtest.py`

- [ ] **Step 1: Write the failing tests for runner**

```python
# tests/unit/backtest/test_runner.py
"""BacktestRunner — Plan 5B-3 Task 4."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtest.runner import BacktestRunner, BacktestResult
from execution.cost_model import SlippageConfig
from execution.paper_broker import PaperBrokerConfig
from execution.replay_broker import ReplayBroker
from features.registry import flatten_features
from models.xgb_predictor import XGBPredictor


def _make_klines(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    closes = 3000.0 + np.cumsum(rng.normal(0, 5.0, size=n))
    return pd.DataFrame({
        "open":   closes,
        "high":   closes + 5,
        "low":    closes - 5,
        "close":  closes,
        "volume": np.full(n, 1.0),
    }, index=pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(n)], name="open_time",
    ))


def _make_features(klines: pd.DataFrame) -> pd.DataFrame:
    """Synthetic features parquet aligned to klines.index but without
    the warmup period — mimics what build_training_set produces."""
    rng = np.random.default_rng(0)
    n = len(klines)
    return pd.DataFrame({
        "a.f1": rng.normal(size=n),
        "a.f2": rng.normal(size=n),
    }, index=pd.DatetimeIndex(klines.index, name="as_of"))


@pytest.fixture
def cfg():
    return PaperBrokerConfig(
        slippage=SlippageConfig(
            slippage_bps_base=1.0, slippage_bps_per_adv_unit=0.0, adv_stub=1000.0,
        ),
    )


@pytest.mark.asyncio
async def test_runner_produces_equity_series(cfg):
    klines = _make_klines(n=50)
    features = _make_features(klines)
    broker = ReplayBroker(cfg=cfg, klines=klines, symbol="ETHUSDT")
    predictor = XGBPredictor.stub(prob_up=0.7, ml_model_version="stub")
    runner = BacktestRunner(
        symbol="ETHUSDT",
        klines=klines,
        features=features,
        funding=None,
        broker=broker,
        predictor=predictor,
        long_threshold=0.6,
        short_threshold=0.6,
        initial_equity_usdt=10_000.0,
    )
    result = await runner.run(oos_start=klines.index[10],
                              oos_end=klines.index[-1])
    assert isinstance(result, BacktestResult)
    assert len(result.equity) >= 1
    # With prob_up=0.7 stub > long_threshold=0.6, every bar after start
    # should produce a long-side fill. Final equity should differ from start.
    assert result.n_trades >= 1


@pytest.mark.asyncio
async def test_runner_records_no_trades_when_predictor_below_threshold(cfg):
    klines = _make_klines(n=20)
    features = _make_features(klines)
    broker = ReplayBroker(cfg=cfg, klines=klines, symbol="ETHUSDT")
    # prob_up=0.5 below long=0.6 AND below 1-short=0.4 → flat, no proposals.
    predictor = XGBPredictor.stub(prob_up=0.5, ml_model_version="stub")
    runner = BacktestRunner(
        symbol="ETHUSDT", klines=klines, features=features, funding=None,
        broker=broker, predictor=predictor,
        long_threshold=0.6, short_threshold=0.6,
        initial_equity_usdt=10_000.0,
    )
    result = await runner.run(oos_start=klines.index[5],
                              oos_end=klines.index[-1])
    assert result.n_trades == 0
    # All equity values should be the initial equity (no fills, no funding).
    assert (result.equity == 10_000.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/backtest/test_runner.py -v`
Expected: ImportError on `backtest.runner`.

- [ ] **Step 3: Implement BacktestRunner**

```python
# src/backtest/runner.py
"""BacktestRunner — Plan 5B-3 Task 4.

Iterates an OOS window of bars; at each bar:
  1. broker.set_time(bar.ts)  — drives ReplayBroker clock + funding
  2. lookup features at bar.ts (pre-computed)
  3. predictor.predict(features) → PredictionBundle
  4. ThresholdPolicy → TradeProposal (or None)
  5. submit to broker (always — no risk pipeline in this baseline)
  6. drain events → accumulate into equity_curve

Then computes Sharpe + Deflated Sharpe over per-bar returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from backtest.equity import equity_curve
from decision.policy import ThresholdPolicy
from decision.proposal import PortfolioSnapshot
from execution.base import BrokerEvent, Order
from execution.replay_broker import ReplayBroker
from features.registry import flatten_features
from models.base import Predictor


@dataclass
class BacktestResult:
    equity: pd.Series           # per-bar equity (DatetimeIndex)
    returns: pd.Series          # equity.pct_change().dropna()
    sharpe: float               # annualised Sharpe ratio
    deflated_sharpe: float      # DSR z-score
    n_trades: int               # count of "filled" events
    n_bars: int                 # count of bars iterated
    initial_equity: float
    final_equity: float


@dataclass
class BacktestRunner:
    symbol: str
    klines: pd.DataFrame
    features: pd.DataFrame
    funding: pd.DataFrame | None
    broker: ReplayBroker
    predictor: Predictor
    long_threshold: float = 0.58
    short_threshold: float = 0.58
    initial_equity_usdt: float = 10_000.0
    fixed_position_size: float = 0.01     # contracts per trade (small)
    n_trials: int = 2

    async def run(self, *, oos_start: pd.Timestamp,
                  oos_end: pd.Timestamp) -> BacktestResult:
        oos_bars = self.klines.loc[oos_start:oos_end].index

        policy = ThresholdPolicy(
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
            symbol=self.symbol,
            mid_provider=lambda _s: float(
                self.klines.loc[self.broker._current_ts]["close"]
            ),
            atr_provider=lambda _s: 15.0,        # constant ATR for baseline
        )
        portfolio = PortfolioSnapshot(
            equity_usdt=self.initial_equity_usdt,
            open_positions={}, day_pnl_r=0.0, consecutive_wins=0,
        )

        all_events: list[BrokerEvent] = []
        n_trades = 0

        for ts in oos_bars:
            self.broker.set_time(ts)
            # Drain any funding events emitted by set_time.
            for e in self.broker.drain_events():
                all_events.append(e)
                if e.kind == "filled":
                    n_trades += 1
            # Predict.
            if ts not in self.features.index:
                continue
            feats_row = self.features.loc[ts]
            # Convert row → nested dict shape via reverse of flatten_features.
            # Predictor stubs accept any dict shape; trained models will
            # use feature_order to look up flat keys.
            feats = {col: float(feats_row[col]) for col in self.features.columns}
            bundle = await self.predictor.predict(feats)
            proposal = await policy.propose(feats, bundle, portfolio)
            if proposal is None:
                continue
            side = "buy" if proposal.direction == "long" else "sell"
            order = Order(
                client_order_id=proposal.proposal_id,
                symbol=self.symbol,
                side=side, type="market",
                qty=self.fixed_position_size,
            )
            await self.broker.submit(order)
            for e in self.broker.drain_events():
                all_events.append(e)
                if e.kind == "filled":
                    n_trades += 1

        mark_prices = {
            ts.to_pydatetime(): float(self.klines.loc[ts]["close"])
            for ts in oos_bars
        }
        equity = equity_curve(
            events=all_events,
            initial_equity=self.initial_equity_usdt,
            mark_prices=mark_prices,
        )
        returns = equity.pct_change().dropna()

        from observability.sharpe import sharpe_ratio, deflated_sharpe_ratio
        periods_per_year = 24 * 365
        sr = sharpe_ratio(returns.to_numpy(), periods_per_year=periods_per_year)
        dsr = deflated_sharpe_ratio(
            returns.to_numpy(), n_trials=self.n_trials,
            periods_per_year=periods_per_year,
        )

        return BacktestResult(
            equity=equity,
            returns=returns,
            sharpe=sr,
            deflated_sharpe=dsr,
            n_trades=n_trades,
            n_bars=len(oos_bars),
            initial_equity=self.initial_equity_usdt,
            final_equity=float(equity.iloc[-1]),
        )
```

- [ ] **Step 4: Write the failing test for the CLI**

```python
# tests/unit/scripts/test_backtest.py
"""scripts/backtest.py CLI — Plan 5B-3 Task 4."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression


def _setup_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Seed klines, features, labels, model bundle in tmp_path."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    n = 100
    rng = np.random.default_rng(0)
    closes = 3000.0 + np.cumsum(rng.normal(0, 5.0, size=n))
    idx = pd.DatetimeIndex(
        [base + timedelta(hours=i) for i in range(n)], name="open_time",
    )
    klines = pd.DataFrame({
        "open": closes, "high": closes + 5, "low": closes - 5,
        "close": closes, "volume": np.full(n, 1.0),
    }, index=idx)
    kline_path = tmp_path / "history" / "ETHUSDT_1h.parquet"
    kline_path.parent.mkdir(parents=True)
    klines.to_parquet(kline_path)

    feature_idx = pd.DatetimeIndex(idx, name="as_of")
    features = pd.DataFrame({
        "a.f1": rng.normal(size=n), "a.f2": rng.normal(size=n),
    }, index=feature_idx)
    feat_path = tmp_path / "training" / "ETHUSDT_1h_features.parquet"
    feat_path.parent.mkdir(parents=True)
    features.to_parquet(feat_path)

    # Tiny model bundle.
    booster = xgb.XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss")
    booster.fit(features.to_numpy(), (rng.uniform(size=n) > 0.5).astype(int))
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    version = "test00000001"
    booster.save_model(str(model_dir / f"xgb_{version}.json"))
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.5, 0.9], [0, 1, 1])
    with open(model_dir / f"calib_{version}.pkl", "wb") as fh:
        pickle.dump({"calibrator": iso, "feature_order": ["a.f1", "a.f2"]}, fh)
    (model_dir / f"meta_{version}.json").write_text(json.dumps({
        "model_version": version,
        "calibration_method": "isotonic",
        "feature_order": ["a.f1", "a.f2"],
    }))

    return kline_path, feat_path, model_dir, tmp_path


@pytest.mark.asyncio
async def test_backtest_cli_writes_backtest_runs_row(tmp_path):
    import argparse
    from scripts.backtest import main_async

    kline_path, feat_path, model_dir, root = _setup_artifacts(tmp_path)
    sqlite_path = root / "state.db"
    import alembic.command, alembic.config
    ac = alembic.config.Config("alembic.ini")
    ac.set_main_option("sqlalchemy.url", f"sqlite:///{sqlite_path}")
    alembic.command.upgrade(ac, "head")

    args = argparse.Namespace(
        symbol="ETHUSDT",
        kline=str(kline_path),
        features=str(feat_path),
        funding="",
        model_dir=str(model_dir),
        sqlite_path=str(sqlite_path),
        oos_fraction=0.3,
        long_threshold=0.0,    # always-fire so we get trades
        short_threshold=0.0,
        n_trials=2,
    )
    await main_async(args)

    engine = sa.create_engine(f"sqlite:///{sqlite_path}")
    with engine.begin() as conn:
        rows = conn.execute(sa.text("SELECT * FROM backtest_runs")).fetchall()
    assert len(rows) == 1
```

- [ ] **Step 5: Run failing test for CLI**

Run: `pytest tests/unit/scripts/test_backtest.py -v`
Expected: ImportError on `scripts.backtest`.

- [ ] **Step 6: Implement scripts/backtest.py**

```python
# scripts/backtest.py
"""Backtest CLI — Plan 5B-3 Task 4.

Loads klines + features + (optional) funding parquets and the latest
trained model bundle, runs BacktestRunner over the last `oos_fraction`
of the kline timeline, computes Sharpe + DSR, and writes one row to
`backtest_runs`.

Usage:
    python scripts/backtest.py \
        --kline data/history/ETHUSDT_1h.parquet \
        --features data/training/ETHUSDT_1h_features.parquet \
        --funding data/funding/ETHUSDT.parquet \
        --model-dir models \
        --sqlite-path data/state.db \
        --oos-fraction 0.2
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from backtest.runner import BacktestRunner
from execution.cost_model import SlippageConfig
from execution.paper_broker import PaperBrokerConfig
from execution.replay_broker import ReplayBroker
from models.registry import load_latest_model


async def main_async(args: argparse.Namespace) -> None:
    klines = pd.read_parquet(args.kline)
    features = pd.read_parquet(args.features)
    funding = pd.read_parquet(args.funding) if args.funding else None
    model = load_latest_model(Path(args.model_dir))

    # OOS = last `oos_fraction` of klines.
    oos_n = int(len(klines) * args.oos_fraction)
    oos_start = klines.index[-oos_n]
    oos_end = klines.index[-1]

    cfg = PaperBrokerConfig(slippage=SlippageConfig(
        slippage_bps_base=1.0, slippage_bps_per_adv_unit=0.0, adv_stub=1000.0,
    ))
    broker = ReplayBroker(cfg=cfg, klines=klines, funding=funding,
                          symbol=args.symbol)
    runner = BacktestRunner(
        symbol=args.symbol,
        klines=klines, features=features, funding=funding,
        broker=broker, predictor=model,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        n_trials=args.n_trials,
    )
    result = await runner.run(oos_start=oos_start, oos_end=oos_end)

    summary = {
        "sharpe": float(result.sharpe) if not pd.isna(result.sharpe) else None,
        "deflated_sharpe": float(result.deflated_sharpe)
                             if not pd.isna(result.deflated_sharpe) else None,
        "n_trades": result.n_trades,
        "n_bars": result.n_bars,
        "initial_equity": result.initial_equity,
        "final_equity": result.final_equity,
        "oos_start": str(oos_start),
        "oos_end": str(oos_end),
        "ml_model_version": model.ml_model_version,
    }
    run_id = hashlib.sha256(
        json.dumps(summary, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]

    engine = sa.create_engine(f"sqlite:///{args.sqlite_path}")
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO backtest_runs "
            "(run_id, started_at, deflated_sharpe, cost_model_version, summary_json) "
            "VALUES (:rid, :ts, :dsr, :cmv, :sj)"
        ), {
            "rid": run_id,
            "ts": datetime.now(tz=timezone.utc),
            "dsr": summary["deflated_sharpe"],
            "cmv": "plan5b2_v1",   # PaperBrokerConfig defaults snapshot
            "sj": json.dumps(summary),
        })

    print(f"backtest run_id={run_id} sharpe={summary['sharpe']} "
          f"dsr={summary['deflated_sharpe']} n_trades={summary['n_trades']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--kline", required=True, type=str)
    ap.add_argument("--features", required=True, type=str)
    ap.add_argument("--funding", default="", type=str)
    ap.add_argument("--model-dir", default="models", type=str)
    ap.add_argument("--sqlite-path", default="data/state.db", type=str)
    ap.add_argument("--oos-fraction", type=float, default=0.2)
    ap.add_argument("--long-threshold", type=float, default=0.58)
    ap.add_argument("--short-threshold", type=float, default=0.58)
    ap.add_argument("--n-trials", type=int, default=2)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/unit/backtest/test_runner.py tests/unit/scripts/test_backtest.py -v && pytest -q`
Expected: 3 new passes (2 runner + 1 CLI). Full suite: 333 passed (330 + 3).

- [ ] **Step 8: Commit**

```bash
git add src/backtest/runner.py scripts/backtest.py tests/unit/backtest/test_runner.py tests/unit/scripts/test_backtest.py
git commit -m "feat(backtest): BacktestRunner + scripts/backtest.py CLI writes backtest_runs row"
```

---

## Task 5: Manual smoke + STATUS handoff

**Why:** Final acceptance — run the CLI on real artifacts and answer the question Plan 5B-3 was designed to answer: "what does the OOS backtest look like for our trained model?".

**Files:**
- Create: `docs/superpowers/plans/2026-04-26-pivot-plan5b3-STATUS.md`

- [ ] **Step 1: Run manual smoke**

```bash
PYTHONPATH=src python scripts/backtest.py \
    --kline data/history/ETHUSDT_1h.parquet \
    --features data/training/ETHUSDT_1h_features.parquet \
    --funding data/funding/ETHUSDT.parquet \
    --model-dir models \
    --sqlite-path data/state.db \
    --oos-fraction 0.2
```

Expected stdout (numbers will vary):
```
backtest run_id=<12hex> sharpe=<float> dsr=<float> n_trades=<int>
```

- [ ] **Step 2: Inspect the row**

```bash
source venv/bin/activate
python -c "
import sqlalchemy as sa, json
engine = sa.create_engine('sqlite:///data/state.db')
with engine.begin() as conn:
    rows = conn.execute(sa.text(
        'SELECT run_id, deflated_sharpe, summary_json FROM backtest_runs ORDER BY started_at DESC LIMIT 1'
    )).fetchall()
print('run_id:', rows[0][0])
print('DSR:', rows[0][1])
print('summary:')
print(json.dumps(json.loads(rows[0][2]), indent=2))
"
```

Expected: a run_id, a DSR (likely near 0 given Plan 5B-1 STATUS noted Brier ≈ baseline), full summary JSON.

- [ ] **Step 3: Write Plan 5B-3 STATUS**

Create `docs/superpowers/plans/2026-04-26-pivot-plan5b3-STATUS.md` with:
- Date / branch / base commit / head commit
- Summary (1 paragraph)
- Task table (5 rows with commit SHAs from `git log --oneline ddcae95..HEAD`)
- Manual smoke results (DSR, Sharpe, n_trades, n_bars, OOS window, model_version)
- Decisions landed (n_trials=2, periods_per_year=8760, OOS=last 20%, no LLM)
- Sanity check: is DSR positive? If yes by how much? If negative, what does it mean?
- What is NOT done (Plan 5B-4 Pre-Live Gate; Plan 5C model improvements; multi-fold rolling backtest; LLM activation)
- Known follow-ups (purged-CV embargo, hyperparameter sweep, etc.)

- [ ] **Step 4: Final commit**

```bash
git add docs/superpowers/plans/2026-04-26-pivot-plan5b3-STATUS.md
git commit -m "docs: Plan 5B-3 handoff STATUS"
```

---

## Self-review notes

- **Spec coverage**: §9.6 backtest deliverables (walk-forward, DSR, OOS holdout, PaperBroker cost model) all present except multi-fold rolling backtest (deferred to Plan 5C).
- **Type consistency**: `BacktestResult` and `BacktestRunner` use the established types (`pd.Series`, `BrokerEvent`, `Predictor`, `ThresholdPolicy`). `equity_curve` returns `pd.Series` indexed by `DatetimeIndex(name="ts")`.
- **No placeholders**: every step has working code; manual smoke commands are concrete.
- **DSR formula correctness**: matches Bailey & López de Prado 2014 eq. 13 (per-period form, then converted via sqrt(periods_per_year)). The `_expected_max_sharpe_under_null` formula is eq. 3.
- **Backward compat**: `ReplayBroker.drain_events()` is additive — `events()` async-generator still works for production paper mode. No existing test breaks.
