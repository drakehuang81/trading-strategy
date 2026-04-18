# Pivot Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the existing rule-based ETHUSDT bot into the spec's 6-layer architecture foundation: Python 3.11 + Alembic-managed SQLite baseline + Protocol-driven `src/` skeleton + all 6 strategy modules migrated as `Feature` implementations with `test_no_repainting` passing.

**Architecture:** Physical layout follows spec §3.1. Source code lives under `src/` (set as `pythonpath` in `pyproject.toml`) so imports are flat (`from features.smc import SMCFeature`). SQLite schema is created by Alembic baseline migration (`render_as_batch=True`) covering all §8.1 tables. Strategy modules move wholesale from `strategy/` to `src/features/` and gain a `Feature` Protocol wrapper; the legacy function API is preserved so the existing 116 tests move over unchanged and stay green.

**Tech Stack:** Python 3.11, Pydantic v2, SQLAlchemy Core + Alembic, pytest, mypy, ruff, structlog, pandas, numpy, `ta`. No external services in Plan 1 — everything is local.

**Phase:** 1 of 3 (Foundation). Subsequent plans: Plan 2 Model + Decision + End-to-End Scaffold; Plan 3 Interface + Ops + Pre-Live Gate.

**Spec:** [2026-04-18-personal-trading-assistant-design.md](../specs/2026-04-18-personal-trading-assistant-design.md) — authoritative. Read §3 (Architecture), §4.1–4.3 (Data / Feature / Model contracts), §8 (State), §9.2 (No-Repainting test), §13 (Migration Plan) before starting.

---

## Task 1: Create pre-pivot checkpoint

**Files:**
- Create: git tag `pre-pivot` (no file; annotated tag)

Rationale: Spec §13 step 0 — the old rule-based bot must always be recoverable via `git reset --hard pre-pivot`.

- [ ] **Step 1: Confirm working tree is clean**

Run: `git status`
Expected: `nothing to commit, working tree clean` (or only untracked `.claude/`, `.superpowers/`, which are in `.gitignore`). If dirty, commit or stash before proceeding.

- [ ] **Step 2: Create annotated tag at current HEAD**

Run:
```bash
git tag -a pre-pivot -m "Checkpoint: last green state of rule-based bot before pivot to LLM-augmented assistant (spec 2026-04-18)"
```

- [ ] **Step 3: Verify tag exists and points to current HEAD**

Run: `git tag -l --format='%(refname:short) %(objectname:short) %(subject)' pre-pivot && git rev-parse HEAD`
Expected: both SHAs match. Tag subject contains "Checkpoint".

- [ ] **Step 4: Commit note — nothing to commit (tag only)**

This task creates no tracked files. Do NOT push the tag yet; Plan 1 final task pushes it along with other commits.

---

## Task 2: Upgrade to Python 3.11 and rebuild venv

**Files:**
- Modify: `requirements.txt`
- Delete: existing `venv/`
- Create: new `venv/` using Python 3.11

Rationale: Spec §7.9 — 3.11 is a **prerequisite** for `asyncio.TaskGroup`, Pydantic v2, `tomllib`. Current venv is 3.9.6.

- [ ] **Step 1: Install Python 3.11 via Homebrew (if not present)**

Run:
```bash
python3.11 --version || brew install python@3.11
python3.11 --version
```
Expected: `Python 3.11.x`.

- [ ] **Step 2: Remove old venv**

Run: `rm -rf venv`

- [ ] **Step 3: Create new 3.11 venv**

Run: `python3.11 -m venv venv && venv/bin/python --version`
Expected: `Python 3.11.x`.

- [ ] **Step 4: Rewrite `requirements.txt` with pinned 3.11-compatible versions**

Replace entire file contents with:
```
# Core runtime
python-binance>=1.0.29
pandas>=2.2,<3
numpy>=1.26,<2
python-dotenv>=1.0
ta>=0.11

# Data model / validation
pydantic>=2.6,<3

# State
SQLAlchemy>=2.0,<3
alembic>=1.13,<2

# Observability
structlog>=24.1

# Testing
pytest>=8.1
pytest-asyncio>=0.23

# Dev tooling
mypy>=1.10
ruff>=0.4
```

- [ ] **Step 5: Install dependencies**

Run: `venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt`
Expected: all install without conflicts.

- [ ] **Step 6: Smoke-test existing test suite on 3.11**

Run: `venv/bin/python -m pytest tests -q 2>&1 | tail -5`
Expected: `116 passed` (with warnings permitted). If any test fails, fix only 3.11-compatibility issues (e.g., `pkg_resources` deprecation); don't change test logic.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt
git commit -m "chore: upgrade runtime to Python 3.11, pin Pydantic v2 + SQLAlchemy + structlog"
```

---

## Task 3: Configure tooling via `pyproject.toml`

**Files:**
- Create: `pyproject.toml`

Rationale: Central place for `pythonpath`, `pytest`, `mypy`, `ruff`. No setuptools build config yet — Plan 2 may add `pip install -e .` when the orchestrator entry point lands.

- [ ] **Step 1: Create `pyproject.toml`**

Write file contents:
```toml
[tool.pytest.ini_options]
# Both rootdir (".") and "src" on path: lets tests import `tests.helpers.X`
# as well as src-layer modules (`features.smc`, etc.).
pythonpath = [".", "src"]
testpaths = ["tests"]
addopts = "-q --strict-markers"
asyncio_mode = "auto"
markers = [
    "contract: cross-implementation contract tests",
    "e2e: end-to-end scenario tests",
]

[tool.mypy]
python_version = "3.11"
strict = true
mypy_path = "src"
namespace_packages = true
explicit_package_bases = true
# Targeted relaxations for third-party libs without stubs
[[tool.mypy.overrides]]
module = ["binance.*", "ta.*"]
ignore_missing_imports = true

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["E501"]  # line-length handled by formatter
```

- [ ] **Step 2: Run pytest to confirm config is picked up**

Run: `venv/bin/python -m pytest tests -q 2>&1 | tail -3`
Expected: `116 passed`. (pythonpath for `src` has no effect yet since `src/` doesn't exist.)

- [ ] **Step 3: Run mypy on current tree to establish a baseline**

Run: `venv/bin/python -m mypy tests strategy 2>&1 | tail -10`
Expected: errors allowed — we care that mypy runs, not that legacy code is clean. Note the count for reference.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject.toml with pytest pythonpath=src, mypy strict, ruff"
```

---

## Task 4: Create `src/` skeleton with layer directories

**Files:**
- Create: `src/__init__.py` — **do NOT** create this; `src/` is a namespace, not a package (per pyproject `namespace_packages = true`).
- Create: `src/data/__init__.py`, `src/features/__init__.py`, `src/models/__init__.py`, `src/models/ml/__init__.py`, `src/models/llm/__init__.py`, `src/decision/__init__.py`, `src/execution/__init__.py`, `src/interface/__init__.py`, `src/state/__init__.py`, `src/observability/__init__.py`
- Create: `src/state/alembic/__init__.py` (placeholder; Alembic env arrives in Task 7)

Rationale: Spec §3.1 repo layout. Each layer is a package; `src/` itself is not (flat imports via pyproject `pythonpath = ["src"]`).

- [ ] **Step 1: Create all layer directories with empty `__init__.py`**

Run:
```bash
mkdir -p src/data src/features src/models/ml src/models/llm src/decision src/execution src/interface src/state/alembic src/observability
for d in src/data src/features src/models src/models/ml src/models/llm src/decision src/execution src/interface src/state src/state/alembic src/observability; do
    touch "$d/__init__.py"
done
ls -la src/
```
Expected: eight top-level package dirs under `src/`.

- [ ] **Step 2: Create config directory scaffolding**

Run:
```bash
mkdir -p config/prompts
touch config/settings.yaml config/prompts/.gitkeep
```

- [ ] **Step 3: Populate `config/settings.yaml` with Plan 1 defaults**

Write:
```yaml
mode: paper            # paper | live — live blocked by pre_live_gate (§10)
watchlist: ["ETHUSDT"]
risk:
  net_directional_cap:
    enabled: false     # §7.6 stub
    max_net_long: 0.5
    max_net_short: 0.5
  daily_loss_r_multiple: -2.0
  max_trades_per_day: 10
  max_concurrent_positions: 3
  spread_bps_max: 20
  fixed_fractional_equity_risk: 0.0025
  dust_balance_threshold: 0.01
fees:
  maker_bps: 2
  taker_bps: 5
database:
  sqlite_path: "data/state.db"
```

- [ ] **Step 4: Verify pytest still collects 116 tests with new dirs present**

Run: `venv/bin/python -m pytest tests -q 2>&1 | tail -3`
Expected: `116 passed`.

- [ ] **Step 5: Commit**

```bash
git add src config
git commit -m "feat: add src/ skeleton with six layer packages and config/settings.yaml"
```

---

## Task 5: Define layer Protocols (Data + Feature + Model)

**Files:**
- Create: `src/data/base.py`
- Create: `src/features/base.py`
- Create: `src/models/base.py`
- Create: `tests/unit/data/__init__.py`, `tests/unit/features/__init__.py`, `tests/unit/models/__init__.py`
- Create: `tests/unit/__init__.py`, `tests/contracts/__init__.py`, `tests/e2e/__init__.py`

Rationale: Spec §4.1–§4.3. Plan 1 only requires Data / Feature / Model; Decision + Execution + Interface protocols land in Plan 2. Defining these three now locks the interfaces the feature migration (Tasks 8–13) must satisfy.

- [ ] **Step 1: Create `src/data/base.py`**

```python
"""Data layer Protocol — spec §4.1."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    """Historical + latest candles abstraction. Every implementation
    must be async-safe and return a DataFrame indexed by UTC timestamp."""

    name: str

    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame: ...

    async def fetch_latest(
        self,
        symbol: str,
        timeframe: str,
        n: int,
    ) -> pd.DataFrame: ...

    def supports(self, symbol: str, timeframe: str) -> bool: ...
```

- [ ] **Step 2: Create `src/features/base.py`**

```python
"""Feature layer Protocol — spec §4.2.

Every Feature MUST only use df[df.index <= as_of]. The no-repainting
test (§9.2) enforces this at CI time."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

import pandas as pd


class Feature(Protocol):
    """Point-in-time feature computation."""

    name: str
    version: str           # bump on logic change
    required_lookback: int

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]: ...
```

- [ ] **Step 3: Create `src/models/base.py`**

```python
"""Model layer Protocol and PredictionBundle — spec §4.3.

PredictionBundle is the single object the Decision layer consumes.
prob_up comes only from the calibrated ML predictor; LLMContextProvider
contributes boolean/categorical flags via Ensemble (Plan 2)."""
from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel


class PredictionBundle(BaseModel):
    direction: Literal["long", "short", "flat"]
    prob_up: float
    horizon_bars: int
    size_multiplier: float = 1.0
    veto_reason: str | None = None
    feature_snapshot_hash: str
    feature_registry_version: str
    ml_model_version: str
    llm_prompt_version: str
    predictions_detail: dict[str, Any] = {}


class Predictor(Protocol):
    async def predict(self, features: dict[str, Any]) -> PredictionBundle: ...


class LLMContextFlags(BaseModel):
    context_veto: bool
    veto_reason: str | None = None
    structural_flags: list[str] = []


class LLMContextProvider(Protocol):
    """Distinct Protocol — not a Predictor. Emits boolean/categorical
    flags only; never outputs prob_up."""

    prompt_version: str

    async def flags(self, features: dict[str, Any]) -> LLMContextFlags: ...
```

- [ ] **Step 4: Create test directory scaffolding**

Run:
```bash
mkdir -p tests/unit/data tests/unit/features tests/unit/models tests/contracts tests/e2e tests/helpers tests/fixtures
for d in tests/unit tests/unit/data tests/unit/features tests/unit/models tests/contracts tests/e2e tests/helpers; do
    touch "$d/__init__.py"
done
```

- [ ] **Step 5: Write a tiny import-smoke test to lock the Protocol surface**

Create `tests/unit/test_protocol_imports.py`:
```python
"""Import smoke test — catches broken Protocol signatures early."""


def test_data_protocol_importable():
    from data.base import DataSource
    assert DataSource.__name__ == "DataSource"


def test_feature_protocol_importable():
    from features.base import Feature
    assert Feature.__name__ == "Feature"


def test_model_protocols_importable():
    from models.base import (
        LLMContextFlags,
        LLMContextProvider,
        PredictionBundle,
        Predictor,
    )
    bundle = PredictionBundle(
        direction="flat",
        prob_up=0.5,
        horizon_bars=1,
        feature_snapshot_hash="x",
        feature_registry_version="0.0.0",
        ml_model_version="stub",
        llm_prompt_version="stub",
    )
    assert bundle.direction == "flat"
    flags = LLMContextFlags(context_veto=False)
    assert flags.context_veto is False
```

- [ ] **Step 6: Run the smoke test**

Run: `venv/bin/python -m pytest tests/unit/test_protocol_imports.py -v 2>&1 | tail -10`
Expected: 3 passed.

- [ ] **Step 7: Run mypy on new src modules**

Run: `venv/bin/python -m mypy src/data src/features src/models 2>&1 | tail -5`
Expected: `Success: no issues found`.

- [ ] **Step 8: Commit**

```bash
git add src/data src/features src/models tests/unit tests/contracts tests/e2e tests/helpers tests/fixtures
git commit -m "feat: define Data / Feature / Model Protocols and Pydantic bundles (spec §4.1–§4.3)"
```

---

## Task 6: Set up Alembic with SQLite + `render_as_batch=True`

**Files:**
- Create: `alembic.ini`
- Create: `src/state/alembic/env.py`
- Create: `src/state/alembic/script.py.mako`
- Create: `src/state/alembic/versions/` (directory)
- Create: `data/` (directory for sqlite_path)

Rationale: Spec §8.1 + §13 step 2. Alembic must be set up **before** any code writes to SQLite so every schema change is captured from day one.

- [ ] **Step 1: Install alembic if missing, then scaffold in-place**

We write the files manually instead of `alembic init` so the layout matches spec §3.1 (`src/state/alembic/`).

- [ ] **Step 2: Create `alembic.ini`**

```ini
[alembic]
script_location = src/state/alembic
prepend_sys_path = src
sqlalchemy.url = sqlite:///data/state.db
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s
timezone = UTC

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

- [ ] **Step 3: Create `src/state/alembic/env.py`**

```python
"""Alembic environment — SQLite with render_as_batch=True (spec §8.1).

Batch mode is required because SQLite ALTER TABLE is limited; every
column change must be wrapped in a batch_alter_table block."""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None  # schemas are defined in migration files directly


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Create `src/state/alembic/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 5: Ensure versions directory exists and data directory exists**

Run:
```bash
mkdir -p src/state/alembic/versions data
touch src/state/alembic/versions/.gitkeep
echo "data/state.db" >> .gitignore
echo "data/ticks/" >> .gitignore
echo "data/feature_cache/" >> .gitignore
```

- [ ] **Step 6: Confirm `alembic current` runs (no revisions yet)**

Run: `venv/bin/alembic current 2>&1 | tail -3`
Expected: no error; blank current revision.

- [ ] **Step 7: Commit**

```bash
git add alembic.ini src/state/alembic .gitignore
git commit -m "feat: scaffold Alembic with render_as_batch=True for SQLite (spec §8.1)"
```

---

## Task 7: Write Alembic baseline migration (all §8.1 tables)

**Files:**
- Create: `src/state/alembic/versions/<stamp>_baseline_schema.py` (generated by alembic)

Rationale: Spec §8.1 lists 16 tables. The baseline is one revision so schema history starts with the real design.

- [ ] **Step 1: Generate an empty baseline revision**

Run: `venv/bin/alembic revision -m "baseline_schema"`
Expected: new file under `src/state/alembic/versions/` is printed.

Capture its path for the next step (e.g., `src/state/alembic/versions/20260418_1230_abc123_baseline_schema.py`).

- [ ] **Step 2: Replace the generated revision file's `upgrade()` and `downgrade()` with the baseline schema**

Find the newly created revision file and replace its body (keeping the auto-generated `revision`, `down_revision`, `branch_labels`, `depends_on` lines) with:

```python
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# (keep the auto-generated revision/down_revision/branch_labels/depends_on above)


def upgrade() -> None:
    # ── Core event log ──────────────────────────────────────────────
    op.create_table(
        "proposals",
        sa.Column("proposal_id", sa.String, primary_key=True),
        sa.Column("trace_id", sa.String, nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("symbol", sa.String, nullable=False, index=True),
        sa.Column("direction", sa.String, nullable=False),
        sa.Column("entry", sa.Float, nullable=False),
        sa.Column("stop_loss", sa.Float, nullable=False),
        sa.Column("take_profit_json", sa.Text, nullable=False),
        sa.Column("size", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("feature_snapshot_json", sa.Text, nullable=False),
        sa.Column("bundle_json", sa.Text, nullable=False),
        sa.Column("risk_checks_json", sa.Text, nullable=False),
        sa.Column("accepted", sa.Boolean, nullable=False),
        sa.Column("rationale", sa.Text),
        sa.Column("feature_registry_version", sa.String, nullable=False),
        sa.Column("ml_model_version", sa.String, nullable=False),
        sa.Column("llm_prompt_version", sa.String, nullable=False),
    )

    op.create_table(
        "broker_events",
        sa.Column("event_id", sa.String, primary_key=True),   # idempotency key §8.3
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("order_id", sa.String, nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("fill_price", sa.Float),
        sa.Column("fill_qty", sa.Float),
        sa.Column("fee", sa.Float),
        sa.Column("reason", sa.Text),
        sa.Column("ml_model_version", sa.String),
        sa.Column("llm_prompt_version", sa.String),
    )

    op.create_table(
        "positions",
        sa.Column("symbol", sa.String, primary_key=True),
        sa.Column("qty", sa.Float, nullable=False),
        sa.Column("avg_entry", sa.Float, nullable=False),
        sa.Column("opened_at", sa.DateTime, nullable=False),
        sa.Column("last_update_ts", sa.DateTime, nullable=False),
    )

    op.create_table(
        "fills",
        sa.Column("fill_id", sa.String, primary_key=True),
        sa.Column("event_id", sa.String, nullable=False, index=True),
        sa.Column("order_id", sa.String, nullable=False, index=True),
        sa.Column("symbol", sa.String, nullable=False),
        sa.Column("ts", sa.DateTime, nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("qty", sa.Float, nullable=False),
        sa.Column("fee", sa.Float, nullable=False),
    )

    # ── Prediction / reconciliation / conversations ─────────────────
    op.create_table(
        "prediction_disagreements",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("trace_id", sa.String, nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False),
        sa.Column("ml_direction", sa.String, nullable=False),
        sa.Column("llm_veto", sa.Boolean, nullable=False),
        sa.Column("detail_json", sa.Text, nullable=False),
    )

    op.create_table(
        "reconciliation_diffs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("kind", sa.String, nullable=False),  # position | balance
        sa.Column("diff_json", sa.Text, nullable=False),
        sa.Column("resolution", sa.String, nullable=False),  # auto_repaired | user_accepted | halted
    )

    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.String, primary_key=True),
        sa.Column("started_at", sa.DateTime, nullable=False),
        sa.Column("telegram_chat_id", sa.String, nullable=False, index=True),
    )
    op.create_table(
        "messages",
        sa.Column("message_id", sa.String, primary_key=True),
        sa.Column("conversation_id", sa.String, nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False),
        sa.Column("role", sa.String, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
    )
    op.create_table(
        "tool_calls",
        sa.Column("tool_call_id", sa.String, primary_key=True),
        sa.Column("message_id", sa.String, nullable=False, index=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("args_json", sa.Text, nullable=False),
        sa.Column("result_json", sa.Text, nullable=False),
    )

    # ── Ops / observability ─────────────────────────────────────────
    op.create_table(
        "heartbeat",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("trace_id", sa.String, nullable=False),
    )

    op.create_table(
        "halt_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("activated_at", sa.DateTime, nullable=False, index=True),
        sa.Column("trigger_source", sa.String, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("resumed_at", sa.DateTime),
    )

    op.create_table(
        "log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("level", sa.String, nullable=False),
        sa.Column("trace_id", sa.String, nullable=False, index=True),
        sa.Column("event", sa.String, nullable=False),
        sa.Column("payload_json", sa.Text, nullable=False),
    )

    op.create_table(
        "backtest_runs",
        sa.Column("run_id", sa.String, primary_key=True),
        sa.Column("started_at", sa.DateTime, nullable=False),
        sa.Column("deflated_sharpe", sa.Float),
        sa.Column("cost_model_version", sa.String, nullable=False),
        sa.Column("summary_json", sa.Text, nullable=False),
    )

    op.create_table(
        "session_state",
        sa.Column("date", sa.Date, primary_key=True),
        sa.Column("consecutive_wins", sa.Integer, nullable=False),
        sa.Column("day_pnl_r", sa.Float, nullable=False),
        sa.Column("last_update_ts", sa.DateTime, nullable=False),
    )

    op.create_table(
        "dead_letter",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("source", sa.String, nullable=False),
        sa.Column("payload_json", sa.Text, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
    )

    op.create_table(
        "model_versions",
        sa.Column("ml_model_version", sa.String, primary_key=True),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("training_window_start", sa.DateTime, nullable=False),
        sa.Column("training_window_end", sa.DateTime, nullable=False),
        sa.Column("calibration_method", sa.String, nullable=False),
        sa.Column("deployed_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "feature_cache_manifest",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String, nullable=False, index=True),
        sa.Column("timeframe", sa.String, nullable=False),
        sa.Column("feature_name", sa.String, nullable=False, index=True),
        sa.Column("feature_version", sa.String, nullable=False),
        sa.Column("as_of_start", sa.DateTime, nullable=False),
        sa.Column("as_of_end", sa.DateTime, nullable=False),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("row_count", sa.Integer, nullable=False),
    )


def downgrade() -> None:
    for table in [
        "feature_cache_manifest",
        "model_versions",
        "dead_letter",
        "session_state",
        "backtest_runs",
        "log",
        "halt_events",
        "heartbeat",
        "tool_calls",
        "messages",
        "conversations",
        "reconciliation_diffs",
        "prediction_disagreements",
        "fills",
        "positions",
        "broker_events",
        "proposals",
    ]:
        op.drop_table(table)
```

- [ ] **Step 3: Apply the migration**

Run: `venv/bin/alembic upgrade head 2>&1 | tail -5`
Expected: `Running upgrade  -> <rev>, baseline_schema`.

- [ ] **Step 4: Verify the schema with sqlite3**

Run: `venv/bin/python -c "import sqlite3; c=sqlite3.connect('data/state.db'); print(sorted(r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'alembic_%'\")))"`
Expected (alphabetised):
```
['backtest_runs', 'broker_events', 'conversations', 'dead_letter', 'fills', 'feature_cache_manifest', 'halt_events', 'heartbeat', 'log', 'messages', 'model_versions', 'positions', 'prediction_disagreements', 'proposals', 'reconciliation_diffs', 'session_state', 'tool_calls']`
```
Count: 17 tables.

- [ ] **Step 5: Confirm round-trip downgrade works**

Run: `venv/bin/alembic downgrade base && venv/bin/alembic upgrade head`
Expected: both succeed without error.

- [ ] **Step 6: Delete the dev DB so CI / fresh-clone tests rebuild it from migrations**

Run: `rm data/state.db`

- [ ] **Step 7: Commit**

```bash
git add src/state/alembic/versions
git commit -m "feat: Alembic baseline migration for all §8.1 tables (17 tables)"
```

---

## Task 8: Write Alembic smoke test

**Files:**
- Create: `tests/unit/state/__init__.py`
- Create: `tests/unit/state/test_alembic_baseline.py`

Rationale: CI must verify every fresh checkout can build the schema from zero. The smoke test is also the regression guard against future migrations that break `downgrade → upgrade`.

- [ ] **Step 1: Create the test file**

```python
"""Alembic baseline smoke test — spec §8.1 + §8.3.

Verifies: (a) fresh upgrade from base creates all expected tables;
(b) broker_events.event_id has a UNIQUE constraint (idempotency, §8.3);
(c) round-trip downgrade → upgrade is clean."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

EXPECTED_TABLES = {
    "backtest_runs",
    "broker_events",
    "conversations",
    "dead_letter",
    "fills",
    "feature_cache_manifest",
    "halt_events",
    "heartbeat",
    "log",
    "messages",
    "model_versions",
    "positions",
    "prediction_disagreements",
    "proposals",
    "reconciliation_diffs",
    "session_state",
    "tool_calls",
}


@pytest.fixture
def alembic_config(tmp_path: Path) -> Config:
    db_path = tmp_path / "smoke.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _tables(db_url: str) -> set[str]:
    path = db_url.removeprefix("sqlite:///")
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'alembic_%'"
        ).fetchall()
    return {r[0] for r in rows}


def test_baseline_creates_all_expected_tables(alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    db_url = alembic_config.get_main_option("sqlalchemy.url")
    assert _tables(db_url) == EXPECTED_TABLES


def test_broker_events_event_id_is_primary_key(alembic_config: Config) -> None:
    """Spec §8.3 idempotency contract."""
    command.upgrade(alembic_config, "head")
    db_url = alembic_config.get_main_option("sqlalchemy.url")
    path = db_url.removeprefix("sqlite:///")
    with sqlite3.connect(path) as conn:
        info = conn.execute("PRAGMA table_info(broker_events)").fetchall()
    pk_cols = [row[1] for row in info if row[5] > 0]
    assert pk_cols == ["event_id"], f"event_id must be sole PK; got {pk_cols}"


def test_round_trip_downgrade_upgrade(alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")
    db_url = alembic_config.get_main_option("sqlalchemy.url")
    assert _tables(db_url) == set()
    command.upgrade(alembic_config, "head")
    assert _tables(db_url) == EXPECTED_TABLES
```

- [ ] **Step 2: Run the smoke test**

Run: `venv/bin/python -m pytest tests/unit/state/test_alembic_baseline.py -v 2>&1 | tail -10`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/state
git commit -m "test: Alembic baseline smoke tests (schema, PK on event_id, round-trip)"
```

---

## Task 9: Create `FEATURE_REGISTRY_VERSION`, `canonical_hash`, registry skeleton

**Files:**
- Create: `src/features/registry.py`
- Create: `tests/unit/features/test_canonical_hash.py`

Rationale: Spec §4.2 reproducibility contract. `canonical_hash` must be bit-exact so `feature_snapshot_hash` joins work across restarts and replays.

- [ ] **Step 1: Create `src/features/registry.py`**

```python
"""Feature registry + canonical hash — spec §4.2."""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from hashlib import sha256
from typing import Any, Iterable

import pandas as pd

from features.base import Feature

FEATURE_REGISTRY_VERSION = "1.0.0"   # bump on registry composition change


def _canonical_default(o: Any) -> Any:
    """JSON default that renders floats via repr() (preserves bit-exact
    representation) and datetimes as ISO-8601 UTC strings."""
    if isinstance(o, float):
        if math.isnan(o):
            return "NaN"
        if math.isinf(o):
            return "Infinity" if o > 0 else "-Infinity"
        return repr(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, pd.Timestamp):
        return o.to_pydatetime().isoformat()
    if hasattr(o, "tolist"):   # numpy scalars / arrays
        return o.tolist()
    raise TypeError(f"Cannot canonicalise object of type {type(o).__name__}")


def canonical_hash(features: dict[str, Any]) -> str:
    """Deterministic content-addressable hash for a feature vector.

    Stable across: dict key insertion order, platform, Python patch
    versions. Unstable across: FEATURE_REGISTRY_VERSION changes (by
    design — a new version is a new hash space)."""
    payload = json.dumps(
        features,
        sort_keys=True,
        default=_canonical_default,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return sha256(
        f"{FEATURE_REGISTRY_VERSION}|{payload}".encode("ascii")
    ).hexdigest()


class FeatureRegistry:
    """Holds the live set of Feature implementations and provides
    point-in-time composition for the model layer."""

    def __init__(self, features: Iterable[Feature]) -> None:
        self._features = list(features)

    @property
    def features(self) -> list[Feature]:
        return list(self._features)

    def compute_all(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in self._features:
            out[f.name] = f.compute(df, as_of)
        return out
```

- [ ] **Step 2: Create `tests/unit/features/test_canonical_hash.py`**

```python
"""Canonical hash reproducibility — spec §4.2."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from features.registry import FEATURE_REGISTRY_VERSION, canonical_hash


def test_version_is_stringy() -> None:
    assert isinstance(FEATURE_REGISTRY_VERSION, str)
    assert FEATURE_REGISTRY_VERSION.count(".") == 2


def test_hash_is_order_independent() -> None:
    a = {"smc": {"bias": "bull", "score": 3}, "fib": [0.618, 0.705]}
    b = {"fib": [0.618, 0.705], "smc": {"score": 3, "bias": "bull"}}
    assert canonical_hash(a) == canonical_hash(b)


def test_hash_changes_when_value_changes() -> None:
    a = canonical_hash({"x": 1.0})
    b = canonical_hash({"x": 1.0000001})
    assert a != b


def test_nan_is_stable() -> None:
    a = canonical_hash({"x": math.nan})
    b = canonical_hash({"x": math.nan})
    assert a == b


def test_datetime_serialises() -> None:
    ts = datetime(2026, 4, 18, tzinfo=timezone.utc)
    h = canonical_hash({"ts": ts})
    assert len(h) == 64


def test_version_tag_is_part_of_hash() -> None:
    """Same content, different version → different hash."""
    from features import registry
    original = registry.FEATURE_REGISTRY_VERSION
    content = {"x": 1}
    h1 = canonical_hash(content)
    registry.FEATURE_REGISTRY_VERSION = "9.9.9"
    try:
        h2 = canonical_hash(content)
    finally:
        registry.FEATURE_REGISTRY_VERSION = original
    assert h1 != h2
```

- [ ] **Step 3: Run the tests**

Run: `venv/bin/python -m pytest tests/unit/features/test_canonical_hash.py -v 2>&1 | tail -15`
Expected: 6 passed.

- [ ] **Step 4: Commit**

```bash
git add src/features/registry.py tests/unit/features/test_canonical_hash.py
git commit -m "feat: FeatureRegistry + canonical_hash (spec §4.2 reproducibility contract)"
```

---

## Task 10: Build no-repainting test helper

**Files:**
- Create: `tests/helpers/feature_equality.py`
- Create: `tests/helpers/no_repainting.py`
- Create: `tests/helpers/fixtures.py`
- Create: `tests/fixtures/ethusdt_1h_sample.csv`

Rationale: Spec §9.2. Every feature migrated in Tasks 11–16 reuses this helper, so build it once, first.

- [ ] **Step 1: Create `tests/helpers/feature_equality.py`**

```python
"""Recursive feature-dict comparator — spec §9.2.

Why not `==`: EMA / rolling-window features produce floats that differ
by ~1e-15 between full-df and truncated-df passes due to order of
operations. `math.isclose` absorbs that. NaN is treated as equal to
NaN (unlike IEEE-754 default), matching pandas semantics."""
from __future__ import annotations

import math
from typing import Any


def features_equal(
    a: Any,
    b: Any,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    if type(a) is not type(b):
        # int vs float OK if numerically close
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return _numbers_close(a, b, rel_tol, abs_tol)
        return False
    if isinstance(a, dict):
        assert isinstance(b, dict)
        if a.keys() != b.keys():
            return False
        return all(features_equal(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol) for k in a)
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple))
        if len(a) != len(b):
            return False
        return all(features_equal(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b))
    if isinstance(a, float):
        return _numbers_close(a, b, rel_tol, abs_tol)
    return a == b


def _numbers_close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
```

- [ ] **Step 2: Create `tests/helpers/no_repainting.py`**

```python
"""Parametrised no-repainting test (spec §9.2).

Any feature module can import `assert_no_repainting` and plug it into
its own parametrised test. Uses multiple seeds so a single lucky run
doesn't mask repainting."""
from __future__ import annotations

import random
from typing import Protocol

import pandas as pd

from tests.helpers.feature_equality import features_equal


class _Computable(Protocol):
    required_lookback: int

    def compute(self, df: pd.DataFrame, as_of: pd.Timestamp) -> dict: ...


def assert_no_repainting(
    feature: _Computable,
    df: pd.DataFrame,
    *,
    seed: int,
    n_samples: int = 50,
) -> None:
    rng = random.Random(seed)
    # Only sample timestamps where the feature has enough lookback.
    eligible = df.index[feature.required_lookback :]
    if len(eligible) == 0:
        raise AssertionError("DataFrame has no eligible as_of timestamps; check required_lookback")
    sample_size = min(n_samples, len(eligible))
    sampled = rng.sample(list(eligible), sample_size)
    for ts in sampled:
        truncated = df[df.index <= ts]
        full_result = feature.compute(df, as_of=ts)
        truncated_result = feature.compute(truncated, as_of=ts)
        assert features_equal(full_result, truncated_result), (
            f"Repainting detected in {type(feature).__name__} at as_of={ts}:\n"
            f"  full:      {full_result}\n"
            f"  truncated: {truncated_result}"
        )
```

- [ ] **Step 3: Create `tests/helpers/fixtures.py`**

```python
"""Shared test fixtures — ETH OHLCV sample data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def eth_1h_df() -> pd.DataFrame:
    """Small (~400 rows) ETHUSDT 1h sample covering enough history for
    any single feature's required_lookback."""
    df = pd.read_csv(FIXTURES_DIR / "ethusdt_1h_sample.csv", parse_dates=["open_time"])
    df = df.set_index("open_time").sort_index()
    return df
```

- [ ] **Step 4: Generate the sample fixture from existing `data/ETHUSDT_15m_20260120.csv`**

Run:
```bash
venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
src = pd.read_csv("data/ETHUSDT_15m_20260120.csv")
# Inspect columns so we know how to rename
print(list(src.columns)[:10])
PY
```

Based on the printed column names, write a converter script. Typical column names are `open_time,open,high,low,close,volume,...`. If different, adjust accordingly.

```bash
venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
src = pd.read_csv("data/ETHUSDT_15m_20260120.csv")
# Normalise column case
src.columns = [c.lower() for c in src.columns]
# Pick timestamp col
ts_col = next(c for c in src.columns if "time" in c or "timestamp" in c)
src[ts_col] = pd.to_datetime(src[ts_col], utc=True)
src = src.rename(columns={ts_col: "open_time"}).set_index("open_time").sort_index()
# Resample to 1h from 15m for a compact fixture
hourly = src.resample("1h").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}).dropna()
hourly = hourly.tail(400).reset_index()
out = Path("tests/fixtures/ethusdt_1h_sample.csv")
out.parent.mkdir(parents=True, exist_ok=True)
hourly.to_csv(out, index=False)
print(f"Wrote {len(hourly)} rows to {out}")
PY
```
Expected: prints ~400 rows written.

- [ ] **Step 5: Write a trivial helper self-test**

Create `tests/unit/test_feature_equality.py`:
```python
from __future__ import annotations

import math

from tests.helpers.feature_equality import features_equal


def test_scalar_equals() -> None:
    assert features_equal(1.0, 1.0)


def test_float_tolerance() -> None:
    assert features_equal(1.0, 1.0 + 1e-15)
    assert not features_equal(1.0, 1.0 + 1e-3)


def test_nan_equals_nan() -> None:
    assert features_equal({"x": math.nan}, {"x": math.nan})


def test_nested() -> None:
    a = {"a": [1.0, 2.0, {"b": math.nan}]}
    b = {"a": [1.0 + 1e-15, 2.0, {"b": math.nan}]}
    assert features_equal(a, b)


def test_structural_mismatch() -> None:
    assert not features_equal({"a": 1}, {"b": 1})
    assert not features_equal([1, 2], [1, 2, 3])
```

- [ ] **Step 6: Run helper tests**

Run: `venv/bin/python -m pytest tests/unit/test_feature_equality.py -v 2>&1 | tail -10`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add tests/helpers tests/fixtures tests/unit/test_feature_equality.py
git commit -m "test: recursive feature equality + no-repainting helper (spec §9.2)"
```

---

## Task 11: Migrate `strategy/smc.py` → `src/features/smc.py`

**Files:**
- Create: `src/features/smc.py` — legacy functions copied verbatim + new `SMCFeature` wrapper class
- Create: `tests/unit/features/test_smc.py` — copy of `tests/test_strategy_smc.py` with imports updated + no-repainting test added
- Delete (at end of plan, Task 17): `strategy/smc.py`, `tests/test_strategy_smc.py`

Rationale: Spec §4.2 + §13 step 4. Each feature gets a `Feature` Protocol wrapper without changing the legacy function signatures, so existing tests move over unchanged.

- [ ] **Step 1: Copy `strategy/smc.py` body into `src/features/smc.py` verbatim**

Run:
```bash
cp strategy/smc.py src/features/smc.py
```

- [ ] **Step 2: Add `SMCFeature` class wrapper at the bottom of `src/features/smc.py`**

Append these lines to `src/features/smc.py` (do NOT remove existing functions):
```python
# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2
# (from __future__ imports intentionally omitted: appended blocks
# cannot contain __future__ imports. Use modern `X | None` syntax
# only where the file has no pre-existing `from __future__` line.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class SMCFeature:
    """Packages SMC structure (swings, BOS/CHoCH, OB, FVG) as a single
    point-in-time Feature. Only reads df[df.index <= as_of]."""

    name: str = "smc"
    version: str = "1.0.0"
    required_lookback: int = 100

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {
                "swing_highs": [],
                "swing_lows": [],
                "structure": None,
                "order_blocks": [],
                "fvgs": [],
            }
        swing_highs, swing_lows = find_swing_points(slice_df)
        structure = identify_structure(slice_df, swing_highs, swing_lows)
        order_blocks = find_order_blocks(slice_df, structure)
        fvgs = find_fvg(slice_df)
        return {
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "structure": structure,
            "order_blocks": order_blocks,
            "fvgs": fvgs,
        }
```

Note: if the existing `find_swing_points` / `identify_structure` / `find_order_blocks` / `find_fvg` signatures differ from what's assumed above, adjust the `compute` body — do NOT change the functions themselves. Verify with `grep -n "^def " src/features/smc.py`.

- [ ] **Step 3: Move the SMC test file**

Run:
```bash
cp tests/test_strategy_smc.py tests/unit/features/test_smc.py
```

Then edit `tests/unit/features/test_smc.py`: replace `from strategy.smc import` with `from features.smc import` (single occurrence).

- [ ] **Step 4: Append a no-repainting test to `tests/unit/features/test_smc.py`**

At the end of the file:
```python
import pytest

from features.smc import SMCFeature
from tests.helpers.fixtures import eth_1h_df
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_smc_no_repainting(eth_1h_df, seed: int) -> None:
    assert_no_repainting(SMCFeature(), eth_1h_df, seed=seed)
```

- [ ] **Step 5: Run the new test file**

Run: `venv/bin/python -m pytest tests/unit/features/test_smc.py -v 2>&1 | tail -15`
Expected: all original SMC tests pass **plus** 3 new `test_smc_no_repainting` cases pass.

If `test_smc_no_repainting` fails, the feature has a repainting bug. Fix the `SMCFeature.compute` body (most commonly: some downstream helper peeks ahead via iloc; restrict its inputs to `slice_df`). Don't fudge tolerances — repainting is real.

- [ ] **Step 6: Run the entire suite to confirm no regression**

Run: `venv/bin/python -m pytest -q 2>&1 | tail -5`
Expected: all prior-green tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/features/smc.py tests/unit/features/test_smc.py
git commit -m "feat(features): migrate SMC to Feature Protocol + no-repainting test"
```

---

## Task 12: Migrate `strategy/fibonacci.py` → `src/features/fibonacci.py`

**Files:**
- Create: `src/features/fibonacci.py`
- Create: `tests/unit/features/test_fibonacci.py`

Same pattern as Task 11.

- [ ] **Step 1: Copy**

```bash
cp strategy/fibonacci.py src/features/fibonacci.py
```

- [ ] **Step 2: Append wrapper**

Append to `src/features/fibonacci.py`:
```python
# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2
# (from __future__ imports intentionally omitted: appended blocks
# cannot contain __future__ imports. Use modern `X | None` syntax
# only where the file has no pre-existing `from __future__` line.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FibFeature:
    """Fibonacci levels + OTE zone + confluence as a point-in-time Feature."""

    name: str = "fib"
    version: str = "1.0.0"
    required_lookback: int = 50

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {"levels": {}, "ote": None, "pd_zone": None}
        levels = calculate_fib_from_df(slice_df)
        if not levels:
            return {"levels": {}, "ote": None, "pd_zone": None}
        ote = get_ote_zone(levels)
        current_price = float(slice_df["close"].iloc[-1])
        pd_zone = get_premium_discount_zone(current_price, levels)
        return {"levels": levels, "ote": ote, "pd_zone": pd_zone}
```

Verify `calculate_fib_from_df`, `get_ote_zone`, `get_premium_discount_zone` signatures at `src/features/fibonacci.py` match; if different, adjust. (Based on `strategy/fibonacci.py:184`, `:78`, `:94` they exist.)

- [ ] **Step 3: Move test**

```bash
cp tests/test_strategy_fibonacci.py tests/unit/features/test_fibonacci.py
```
Edit: replace `from strategy.fibonacci` with `from features.fibonacci`.

- [ ] **Step 4: Append no-repainting test**

```python
import pytest

from features.fibonacci import FibFeature
from tests.helpers.fixtures import eth_1h_df
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fib_no_repainting(eth_1h_df, seed: int) -> None:
    assert_no_repainting(FibFeature(), eth_1h_df, seed=seed)
```

- [ ] **Step 5: Run**

Run: `venv/bin/python -m pytest tests/unit/features/test_fibonacci.py -v 2>&1 | tail -15`
Expected: all original + 3 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/features/fibonacci.py tests/unit/features/test_fibonacci.py
git commit -m "feat(features): migrate Fibonacci to Feature Protocol + no-repainting test"
```

---

## Task 13: Migrate `strategy/liquidity.py` → `src/features/liquidity.py`

**Files:**
- Create: `src/features/liquidity.py`
- Create: `tests/unit/features/test_liquidity.py`

- [ ] **Step 1: Copy**

```bash
cp strategy/liquidity.py src/features/liquidity.py
```

- [ ] **Step 1b: Fix the stale cross-module import**

`strategy/liquidity.py` contains `from strategy.smc import find_swing_points` (line 4). After Task 17 deletes `strategy/`, that import breaks. Update it now:

Replace `from strategy.smc import find_swing_points` with `from features.smc import find_swing_points` in `src/features/liquidity.py`.

- [ ] **Step 2: Append wrapper**

Append to `src/features/liquidity.py`:
```python
# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2
# (from __future__ imports intentionally omitted: appended blocks
# cannot contain __future__ imports. Use modern `X | None` syntax
# only where the file has no pre-existing `from __future__` line.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class LiquidityFeature:
    """Equal highs/lows + sweep detection as point-in-time feature."""

    name: str = "liquidity"
    version: str = "1.0.0"
    required_lookback: int = 100

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {"zones": [], "sweep": None, "nearest_target": None}
        zones = get_liquidity_zones(slice_df)
        sweep = check_liquidity_sweep(slice_df, zones)
        nearest = get_nearest_liquidity_target(slice_df, zones)
        return {"zones": zones, "sweep": sweep, "nearest_target": nearest}
```

Verify signatures: `get_liquidity_zones`, `check_liquidity_sweep`, `get_nearest_liquidity_target` exist (see strategy/liquidity.py:167, :189, :279). Adjust call sites to the real signatures.

- [ ] **Step 3: Move test**

```bash
cp tests/test_strategy_liquidity.py tests/unit/features/test_liquidity.py
```
Edit: `from strategy.liquidity` → `from features.liquidity`.

- [ ] **Step 4: Append no-repainting test**

```python
import pytest

from features.liquidity import LiquidityFeature
from tests.helpers.fixtures import eth_1h_df
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_liquidity_no_repainting(eth_1h_df, seed: int) -> None:
    assert_no_repainting(LiquidityFeature(), eth_1h_df, seed=seed)
```

- [ ] **Step 5: Run**

Run: `venv/bin/python -m pytest tests/unit/features/test_liquidity.py -v 2>&1 | tail -15`
Expected: all original + 3 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/features/liquidity.py tests/unit/features/test_liquidity.py
git commit -m "feat(features): migrate Liquidity to Feature Protocol + no-repainting test"
```

---

## Task 14: Migrate `strategy/divergence.py` → `src/features/divergence.py`

**Files:**
- Create: `src/features/divergence.py`
- Create: `tests/unit/features/test_divergence.py`

- [ ] **Step 1: Copy**

```bash
cp strategy/divergence.py src/features/divergence.py
```

- [ ] **Step 2: Append wrapper**

Append to `src/features/divergence.py`:
```python
# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2
# (from __future__ imports intentionally omitted: appended blocks
# cannot contain __future__ imports. Use modern `X | None` syntax
# only where the file has no pre-existing `from __future__` line.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DivergenceFeature:
    """RSI + MACD divergence summary at as_of."""

    name: str = "divergence"
    version: str = "1.0.0"
    required_lookback: int = 60

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {"rsi": None, "macd": None, "summary": {}}
        rsi_div = detect_rsi_divergence(slice_df)
        macd_div = detect_macd_divergence(slice_df)
        summary = get_divergence_summary(slice_df)
        return {"rsi": rsi_div, "macd": macd_div, "summary": summary}
```

- [ ] **Step 3: Move test**

```bash
cp tests/test_strategy_divergence.py tests/unit/features/test_divergence.py
```
Edit: `from strategy.divergence` → `from features.divergence`.

- [ ] **Step 4: Append no-repainting test**

```python
import pytest

from features.divergence import DivergenceFeature
from tests.helpers.fixtures import eth_1h_df
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_divergence_no_repainting(eth_1h_df, seed: int) -> None:
    assert_no_repainting(DivergenceFeature(), eth_1h_df, seed=seed)
```

- [ ] **Step 5: Run**

Run: `venv/bin/python -m pytest tests/unit/features/test_divergence.py -v 2>&1 | tail -15`
Expected: all original + 3 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/features/divergence.py tests/unit/features/test_divergence.py
git commit -m "feat(features): migrate Divergence to Feature Protocol + no-repainting test"
```

---

## Task 15: Migrate `strategy/funding_rate.py` → `src/features/funding_rate.py`

**Files:**
- Create: `src/features/funding_rate.py`
- Create: `tests/unit/features/test_funding_rate.py`

Rationale: Spec §4.5 — funding rate historical data will live at `data/funding/<symbol>.parquet` (Plan 2 creates it). For Plan 1, the `FundingFeature.compute` returns `{}` unless that parquet exists — it's a structural shell now, concrete wiring comes later.

- [ ] **Step 1: Copy**

```bash
cp strategy/funding_rate.py src/features/funding_rate.py
```

- [ ] **Step 2: Append wrapper that reads from local parquet (graceful if missing)**

Append to `src/features/funding_rate.py`:
```python
# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2 + §4.5 (historical parquet)
# (no `from __future__` — appended blocks cannot contain it.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_FUNDING_DIR = Path("data/funding")


@dataclass
class FundingFeature:
    """Reads funding rate from data/funding/<symbol>.parquet (historical,
    no network). Returns empty dict if no parquet exists — Plan 2 wires
    the ingestion job. Network path lives in FundingRateDataSource (also
    Plan 2); this Feature is point-in-time only."""

    symbol: str = "ETHUSDT"
    name: str = "funding"
    version: str = "1.0.0"
    required_lookback: int = 1

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        path = _FUNDING_DIR / f"{self.symbol}.parquet"
        if not path.exists():
            return {"rate": None, "evaluation": None, "position_adj": None}
        funding = pd.read_parquet(path)
        funding = funding[funding["ts"] <= as_of]
        if funding.empty:
            return {"rate": None, "evaluation": None, "position_adj": None}
        rate = float(funding.iloc[-1]["rate"])
        evaluation = evaluate_funding_rate(rate)
        position_adj = get_position_adjustment(rate, direction="long")
        return {"rate": rate, "evaluation": evaluation, "position_adj": position_adj}
```

- [ ] **Step 3: Move test**

```bash
cp tests/test_strategy_funding_rate.py tests/unit/features/test_funding_rate.py
```
Edit: `from strategy.funding_rate` → `from features.funding_rate`.

- [ ] **Step 4: Append no-repainting test (no parquet present → returns empty dict, trivially stable)**

```python
import pytest

from features.funding_rate import FundingFeature
from tests.helpers.fixtures import eth_1h_df
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_funding_no_repainting(eth_1h_df, seed: int) -> None:
    """Without data/funding/ETHUSDT.parquet present, compute returns a
    fixed empty result — trivially non-repainting. When parquet exists
    (Plan 2 populates it), this test must still pass because parquet is
    read point-in-time (ts <= as_of)."""
    assert_no_repainting(FundingFeature(symbol="ETHUSDT"), eth_1h_df, seed=seed)
```

- [ ] **Step 5: Run**

Run: `venv/bin/python -m pytest tests/unit/features/test_funding_rate.py -v 2>&1 | tail -15`
Expected: all original + 3 new tests pass.

Any pre-existing test that made network calls to Binance for funding rate should be mocked — if it was previously passing on CI it was either mocked already or skipped. Don't un-skip here.

- [ ] **Step 6: Commit**

```bash
git add src/features/funding_rate.py tests/unit/features/test_funding_rate.py
git commit -m "feat(features): migrate FundingRate to Feature Protocol + parquet-backed compute"
```

---

## Task 16: Migrate `strategy/confidence.py` → `src/features/confidence.py`

**Files:**
- Create: `src/features/confidence.py`
- Create: `tests/unit/features/test_confidence.py`

Rationale: Spec §12 red flag — the 8-factor confidence score is uncalibrated. Migrate as-is for now; Plan 2/3 addresses calibration. Feature must still be point-in-time.

- [ ] **Step 1: Copy**

```bash
cp strategy/confidence.py src/features/confidence.py
```

- [ ] **Step 2: Append wrapper**

Append to `src/features/confidence.py`:
```python
# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2 (un-calibrated, see §12)
# (no `from __future__` — appended blocks cannot contain it.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import pandas as pd


@dataclass
class ConfidenceFeature:
    """Packages the 8-factor confidence score. WARNING (spec §12):
    this score is not calibrated; do not use as a probability."""

    direction: str = "long"   # "long" | "short" — caller supplies
    name: str = "confidence"
    version: str = "1.0.0"
    required_lookback: int = 100

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {"score": 0, "breakdown": {}, "recommendation": None}
        # Delegates to the legacy aggregator — only uses slice_df.
        # calculate_confidence_score expects precomputed sub-features;
        # Plan 2 wires it to FeatureRegistry. For Plan 1 we just verify
        # it is callable on a slice and returns a stable dict shape.
        result = calculate_confidence_score(
            htf_bias="Neutral",
            direction=self.direction,
            poi=None,
            fib_analysis=None,
            liquidity_sweep=None,
            rsi=float(slice_df["close"].pct_change().rolling(14).mean().iloc[-1] or 0.5) * 100,
            divergence=None,
            macd_hist=0.0,
            prev_macd_hist=0.0,
            asia_range=None,
            current_price=float(slice_df["close"].iloc[-1]),
        )
        return result
```

Verify `calculate_confidence_score` signature at `strategy/confidence.py:164` — if the keyword-arg list above doesn't match, copy the real signature verbatim and pass harmless defaults. The point is to exercise point-in-time flow without crashing.

- [ ] **Step 3: Move test**

```bash
cp tests/test_strategy_confidence.py tests/unit/features/test_confidence.py
```
Edit: `from strategy.confidence` → `from features.confidence`.

- [ ] **Step 4: Append no-repainting test**

```python
import pytest

from features.confidence import ConfidenceFeature
from tests.helpers.fixtures import eth_1h_df
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_confidence_no_repainting(eth_1h_df, seed: int) -> None:
    assert_no_repainting(ConfidenceFeature(direction="long"), eth_1h_df, seed=seed)
```

- [ ] **Step 5: Run**

Run: `venv/bin/python -m pytest tests/unit/features/test_confidence.py -v 2>&1 | tail -15`
Expected: all original + 3 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/features/confidence.py tests/unit/features/test_confidence.py
git commit -m "feat(features): migrate Confidence to Feature Protocol + no-repainting test"
```

---

## Task 17: Retire legacy `strategy/` and `auto_bot.py`

**Files:**
- Delete: `strategy/` (entire directory)
- Delete: `auto_bot.py`
- Delete: `tests/test_strategy_*.py` (7 files — replaced by copies under `tests/unit/features/`)

Rationale: Spec says existing modules move wholesale — leaving both the old and new copies around would invite drift. The `pre-pivot` tag (Task 1) preserves the old code if rollback is ever needed.

`strategy/trade_setup.py` is **not** a Feature — it's Decision-layer concern (sizing + SL/TP). We move it separately in Plan 2 when the Decision layer lands. For Plan 1, it stays as a standalone shim until Plan 2.

- [ ] **Step 1: Move `strategy/trade_setup.py` to a holding place (not yet a Feature)**

```bash
mkdir -p src/decision/_legacy
git mv strategy/trade_setup.py src/decision/_legacy/trade_setup.py
git mv tests/test_strategy_trade_setup.py tests/unit/test_trade_setup_legacy.py
```

Edit `tests/unit/test_trade_setup_legacy.py`: replace `from strategy.trade_setup` with `from decision._legacy.trade_setup`.

- [ ] **Step 2: Delete legacy strategy and auto_bot**

```bash
git rm -r strategy
git rm auto_bot.py
git rm tests/test_strategy_smc.py tests/test_strategy_fibonacci.py tests/test_strategy_liquidity.py tests/test_strategy_divergence.py tests/test_strategy_funding_rate.py tests/test_strategy_confidence.py
```

Note: `data_ingestion/` and `fetch_latest.py` also still depend on `from strategy.funding_rate import ...`. Those will be replaced in Plan 2 by `BinanceKline` and `FundingRateDataSource`. For Plan 1 they become orphaned. That's intentional — `pre-pivot` tag preserves the working copy.

- [ ] **Step 3: Run full test suite**

Run: `venv/bin/python -m pytest -q 2>&1 | tail -5`
Expected: all green — 116 (original) + 18 (6 features × 3 seeds) new no-repainting + 6 hash + 5 equality helper + 3 protocol + 3 alembic + legacy trade_setup tests. Exact total ≈ 150+.

If any test fails, the cause is usually a stray `from strategy....` import in a migrated test file. Grep for leftovers:
```bash
grep -rn "from strategy" tests src 2>/dev/null || echo "clean"
```

- [ ] **Step 4: Run mypy on the full src/ tree**

Run: `venv/bin/python -m mypy src 2>&1 | tail -10`
Expected: `Success: no issues found in N source files` — note: if the legacy wrappers' type annotations are loose, you may see some errors. Acceptable at Plan 1; Plan 2 tightens.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: retire strategy/ and auto_bot.py; move trade_setup to decision/_legacy (Plan 2 will re-home)"
```

---

## Task 18: Wire FeatureRegistry with all 6 features

**Files:**
- Modify: `src/features/registry.py` — add `build_default_registry()` helper
- Create: `tests/unit/features/test_registry_composition.py`

Rationale: Single place where the 6 Features are enumerated. Plan 2 consumers (Predictors, Orchestrator) import `build_default_registry()`; never hand-assemble.

- [ ] **Step 1: Add registry builder to `src/features/registry.py`**

Append to `src/features/registry.py`:
```python
def build_default_registry(
    *,
    symbol: str = "ETHUSDT",
    confidence_direction: str = "long",
) -> FeatureRegistry:
    """Canonical Plan-1 feature set. Order is stable and part of the
    canonical hash — do not reshuffle without bumping
    FEATURE_REGISTRY_VERSION."""
    from features.confidence import ConfidenceFeature
    from features.divergence import DivergenceFeature
    from features.fibonacci import FibFeature
    from features.funding_rate import FundingFeature
    from features.liquidity import LiquidityFeature
    from features.smc import SMCFeature

    return FeatureRegistry([
        SMCFeature(),
        FibFeature(),
        LiquidityFeature(),
        DivergenceFeature(),
        FundingFeature(symbol=symbol),
        ConfidenceFeature(direction=confidence_direction),
    ])
```

- [ ] **Step 2: Create `tests/unit/features/test_registry_composition.py`**

```python
from __future__ import annotations

from features.registry import (
    FEATURE_REGISTRY_VERSION,
    build_default_registry,
    canonical_hash,
)
from tests.helpers.fixtures import eth_1h_df


def test_default_registry_has_six_features(eth_1h_df) -> None:
    registry = build_default_registry()
    names = [f.name for f in registry.features]
    assert names == ["smc", "fib", "liquidity", "divergence", "funding", "confidence"]


def test_compute_all_hashes_stably(eth_1h_df) -> None:
    registry = build_default_registry()
    as_of = eth_1h_df.index[-1]
    features1 = registry.compute_all(eth_1h_df, as_of)
    features2 = registry.compute_all(eth_1h_df, as_of)
    assert canonical_hash(features1) == canonical_hash(features2)


def test_registry_version_is_present() -> None:
    assert FEATURE_REGISTRY_VERSION
```

- [ ] **Step 3: Run**

Run: `venv/bin/python -m pytest tests/unit/features/test_registry_composition.py -v 2>&1 | tail -10`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add src/features/registry.py tests/unit/features/test_registry_composition.py
git commit -m "feat(features): build_default_registry() composes the six Plan-1 Features"
```

---

## Task 19: Final verification + plan handoff

**Files:**
- Create: `docs/superpowers/plans/2026-04-18-pivot-foundation.md` (this plan — already exists; add a "Completion checklist" section note if revised during execution)

- [ ] **Step 1: Full suite + mypy**

Run both:
```bash
venv/bin/python -m pytest -q 2>&1 | tail -5
venv/bin/python -m mypy src 2>&1 | tail -5
```
Expected: all tests pass; mypy clean.

- [ ] **Step 2: Confirm Alembic still up-to-date**

Run: `venv/bin/alembic current 2>&1 | tail -2`
Expected: prints baseline revision SHA followed by `(head)`.

- [ ] **Step 3: Confirm directory layout matches spec §3.1**

Run: `find src -type d -not -path '*/alembic/*' | sort`
Expected:
```
src
src/data
src/decision
src/decision/_legacy
src/execution
src/features
src/interface
src/models
src/models/llm
src/models/ml
src/observability
src/state
```

- [ ] **Step 4: Push `pre-pivot` tag and Plan 1 commits**

```bash
git push origin main
git push origin pre-pivot
```

(Confirm with user first if remote tag push policy is strict.)

- [ ] **Step 5: Commit this plan document**

```bash
git add docs/superpowers/plans/2026-04-18-pivot-foundation.md
git commit -m "docs: Plan 1 (Pivot Foundation) implementation plan"
```

- [ ] **Step 6: Report handoff to user**

Draft a one-screen summary naming:
- What is green now: test count, mypy clean, Alembic head applied
- What is NOT done yet (intentional — Plan 2 scope): XGBPredictor + Ensemble + LLMContextProvider, Broker/PaperBroker, Policy + RiskPipeline, SizingPipeline, orchestrator.py, Telegram bot, ChatLLM, TickRecorder, FeatureDriftMonitor, backtest infra, pre-live gate module
- Propose starting Plan 2 (Model + Decision + End-to-End Scaffold) or pause

---

## Completion Criteria (checked at Task 19)

At the end of Plan 1, the following must all be true:

- [ ] `pre-pivot` git tag exists at the state preceding this plan.
- [ ] `venv/bin/python --version` reports Python 3.11.x.
- [ ] `venv/bin/python -m pytest -q` reports **≥ 150 passed**, **0 failed** (116 original + 6 × 3 no-repainting + helper/registry/alembic/protocol tests).
- [ ] `venv/bin/python -m mypy src` reports `Success: no issues found`.
- [ ] `venv/bin/alembic current` reports the baseline revision is applied to `data/state.db` (or a fresh-build smoke test confirms it).
- [ ] No file under the repository contains `from strategy.` except in the plan / spec docs.
- [ ] `src/features/registry.py::build_default_registry()` returns 6 features, and `compute_all` on the 1h fixture is hash-stable across two calls.

Any un-ticked box blocks starting Plan 2.
