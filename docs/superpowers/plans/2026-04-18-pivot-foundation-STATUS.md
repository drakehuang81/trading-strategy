# Pivot Foundation — 階段性狀態 (2026-04-18)

> 下次接手時先讀這份。快速掌握目前進度、下一步、與重要決策。

## 1. 專案 pivot 背景

從**規則式 auto-bot** 轉向 **LLM 強化的個人交易助理**：
- 硬體：MacBook M1 Pro / 16GB
- 模型：Gemma 4 E4B via Ollama (context provider / veto)、XGBoost (ML prob_up)
- 介面：Telegram
- 核心：Ensemble — ML 給機率、LLM 給 boolean veto flags，**永不混合 prob**

參考文件：
- Spec: [docs/superpowers/specs/2026-04-18-personal-trading-assistant-design.md](../specs/2026-04-18-personal-trading-assistant-design.md)
- Plan 1: [docs/superpowers/plans/2026-04-18-pivot-foundation.md](2026-04-18-pivot-foundation.md)

## 2. 目前狀態

### Plan 1 (Pivot Foundation) — 18/19 tasks 完成

| # | Task | 狀態 | Commit |
|---|------|------|--------|
| 1 | Pre-pivot checkpoint (tag) | done | tag `pre-pivot` |
| 2 | Python 3.11 + venv rebuild | done | `395a3ef` |
| 3 | pyproject.toml tooling | done | `63e6320` |
| 4 | src/ skeleton + config | done | `471cb06` + `9ab61b2` (gitignore fix) |
| 5 | Data / Feature / Model Protocols | done | `d7e8850` |
| 6 | Alembic (render_as_batch) | done | `a8caf9b` |
| 7 | Alembic baseline (17 tables) | done | `2861693` |
| 8 | Alembic smoke tests | done | `de20d87` |
| 9 | FEATURE_REGISTRY_VERSION + canonical_hash | done | `d5f0820` |
| 10 | No-repainting helpers + fixture | done | `423776a` |
| 11 | SMC feature | done | `aa91348` |
| 12 | Fibonacci feature | done | `12fc58f` |
| 13 | Liquidity feature | done | `ac3b166` |
| 14 | Divergence feature | done | `e7a6878` |
| 15 | Funding rate feature | done | `33eeb33` |
| 16 | Confidence feature | done | `ff1d203` |
| 17 | Retire legacy strategy/ + auto_bot | done | `d7e3e70` |
| 18 | build_default_registry (6 features) | done | `f6bf4d6` |
| 19 | Final verification + push tag | **in progress** (驗證跑完，push 待確認) | — |

### 驗證結果（已跑過）

- `pytest`: **154 passed** / 0 failed
- `mypy src`: 71 errors（**全部**是 migrated legacy feature 內既有錯誤，Plan 2 會收緊）
- `alembic heads`: `15fdbaffd2bf` ✅
- `build_default_registry()`: 回傳 6 features，順序 `smc → fib → liquidity → divergence → funding → confidence` ✅

### 分支 / Worktree

- **Worktree**: `/Users/drakehuang/SideProject/Trading/quant-trading-project/.worktrees/pivot-foundation`
- **Branch**: `pivot/foundation` (27 commits ahead of main)
- **Tag**: `pre-pivot` 已建立 **本地**，尚未 push
- **main**: 停留在 `98bf16d` (只有 .worktrees/ gitignore + plan 文件)

## 3. 下一步

### A. Task 19 剩餘步驟（需用戶確認）

1. **是否先 merge `pivot/foundation` 到 main**，再 push？
   - 選項 A：開 PR → review → squash merge
   - 選項 B：直接 `git push origin pivot/foundation` 保留 feature branch
   - 選項 C：local fast-forward merge 再 push main
2. **Push `pre-pivot` tag** 到 origin（檢查點保險）
3. **Final code review subagent** 掃整個 Plan 1 實作（純讀取，無破壞性）

### B. Plan 2 (Model + Decision + End-to-End Scaffold)

**已撰寫 plan 文件：** [2026-04-18-pivot-plan2-model-decision-scaffold.md](2026-04-18-pivot-plan2-model-decision-scaffold.md)

19 個 task 涵蓋：Decision/Execution Protocols → SQLite repos + `rebuild_positions` → PaperBroker 含 funding → Broker contract → BinanceKline → FundingWriter → TickRecorder → RiskPipeline → Sizing → trade_setup re-home → ThresholdPolicy → XGBPredictor (stub + isotonic calibration) → GemmaContextProvider → Ensemble → Orchestrator + CLI → E2E smoke。

依賴 Plan 1 的：
- `src/features/registry.py` (canonical_hash 契約)
- `src/models/base.py` (PredictionBundle, Predictor, LLMContextProvider)
- `src/state/alembic/` (proposals / broker_events schema)

### C. Plan 3 (Interface + Ops + Pre-Live Gate)

尚未撰寫。預計涵蓋：apscheduler 1h 排程、Telegram bot + ChatLLM、FeatureDriftMonitor、ReplayBroker/LiveBroker contract、walk-forward backtest + Deflated Sharpe、Pre-Live Gate module、HALT fire-drill、外部 heartbeat watchdog。

## 4. 重要決策 / 約定（避免走冤枉路）

### 架構決策

- **Ensemble 原則**：ML 出機率、LLM 出 boolean flags，**絕不**把 LLM 機率拿來加權 ML。
- **Feature Protocol**: `compute(df, as_of) -> dict[str, Any]`，內部以 `df[df.index <= as_of]` 保證 point-in-time。
- **Canonical hash**: `FEATURE_REGISTRY_VERSION="1.0.0"` + sorted JSON sha256；版本 bump 就是新 hash space。
- **No-repainting 測試**: multi-seed `[0, 1, 2]` + `math.isclose(rel_tol=1e-9, abs_tol=1e-12)` + NaN==NaN。
- **SQLite + Alembic**: 一律 `render_as_batch=True`（SQLite ALTER TABLE 限制）。
- **broker_events.event_id**: 主鍵＝idempotency key（spec §8.3）。

### 踩過的坑

1. `.gitignore` `data/` 若未錨定（`/data/`）會連 `src/data/` 一起忽略 → 用 `/data/` 錨根目錄。
2. Bash CWD 跨呼叫**有時**會持續、有時不會 → 重要路徑一律用絕對路徑或 `cd x && cmd`。
3. Python 3.11 需 `brew install python@3.11` 後重建 venv；pandas-stubs 是 mypy strict 必裝。
4. 當年 legacy features 的函式簽名跟 plan 寫的不一致（SMC / Fib / Liquidity / Funding / Confidence 都有小 bug）— subagent 已在 wrapper 層修正，**從未改 legacy 函式**。
5. Subagent 修改範圍必須用白名單硬限制；越界視為 BLOCKED（遵守 user CLAUDE.md 規則）。

### 工作流程

- 跑 Plan 用 `superpowers:subagent-driven-development`。
- 每 task 兩階段 review：**spec compliance → code quality**。
- 成對 dispatch sonnet subagent（Task 9+10, 11+12, ...）比單派效率高。
- 簡單任務（scaffolding / 單檔）直接主 session 做，不派 subagent（遵守 user CLAUDE.md）。

## 5. 檔案地圖（Plan 1 產出）

```
src/
├── data/base.py                   # DataSource Protocol
├── decision/_legacy/trade_setup.py  # 暫放區（Plan 2 會 re-home）
├── features/
│   ├── base.py                    # Feature Protocol
│   ├── registry.py                # canonical_hash + build_default_registry
│   ├── smc.py / fibonacci.py / liquidity.py
│   ├── divergence.py / funding_rate.py / confidence.py
├── models/base.py                 # PredictionBundle / Predictor / LLMContextProvider
├── state/
│   ├── alembic/
│   │   ├── env.py                 # render_as_batch=True
│   │   ├── script.py.mako
│   │   └── versions/20260418_1302_15fdbaffd2bf_baseline_schema.py (17 tables)

tests/
├── fixtures/ethusdt_1h_sample.csv  # 400 rows resampled
├── helpers/
│   ├── feature_equality.py        # recursive NaN==NaN
│   ├── no_repainting.py           # multi-seed truncation test
│   └── fixtures.py                # eth_1h_df fixture
├── unit/features/                 # 6 feature test files + registry composition
└── unit/state/test_alembic_baseline.py

alembic.ini (script_location=src/state/alembic)
pyproject.toml (pytest pythonpath=["src"], mypy strict, ruff)
config/settings.yaml (paper mode, ETHUSDT, risk config)
```

## 6. 快速接手指令

```bash
# 進 worktree
cd /Users/drakehuang/SideProject/Trading/quant-trading-project/.worktrees/pivot-foundation
source venv/bin/activate

# 檢查狀態
git log --oneline -5
pytest -q
alembic current
alembic heads
```

---

**下次可以直接問我：「繼續 Plan 1 Task 19 push / 開始 Plan 2」**。
