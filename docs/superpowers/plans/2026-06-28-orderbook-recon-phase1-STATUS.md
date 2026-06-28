# Order Book Recon — Phase 1 STATUS (2026-06-28)

> 下次接手先讀這份。Phase 1(端到端薄管線 + Step 0 工具)已完成。

## 完成狀態

8 tasks 全部完成,subagent-driven 執行 + controller spot-check + 一輪 final review。

- **Branch**: `worktree-recon-phase1`(從 `origin/main` = `429932f` 長出)
- **Worktree**: `.claude/worktrees/recon-phase1`
- **Tests**: 14 passed (`venv/bin/pytest tests/research/microstructure/`)
- **venv**: 精簡 3.11(polars 1.42 / pandas / numpy / scipy / pyarrow / requests / pytest)。**非完整 requirements**——research-only 不需生產依賴;全 suite 跑會有 ~50 個既有 collection error(缺 pydantic 等),屬預期。

## 產出(commits `a72ef5d` → `da80c48`)

```
src/research/microstructure/   download.py schema_probe.py signals.py(QI) align.py ic.py report.py
scripts/recon/                 probe_schema.py(Step 0 CLI)  run_recon.py(薄管線 CLI)
tests/research/microstructure/ test_{download,schema_probe,signals,align,ic,report,run_recon}.py
```

完全隔離(無生產層 import)。薄管線:book → queue_imbalance → mid grid(backward as-of)→ forward returns → Spearman IC → markdown。

## Final review findings

- ✅ **#1 (Critical) FIXED** `da80c48`:`quantile_layering` 的 bucket label 改 zero-pad,n_buckets≥10 才會正確 numeric sort(原本字典序 "10"<"2" 會讓 monotone check 失效)+ n=15 回歸測試。Phase 1 default n=5 原本就不觸發。
- ⏳ **#2 (Important) → Phase 2**:`render_ic_markdown` 對「異質 horizon keys」或「空 signal dict」會 KeyError/StopIteration。Phase 1 單一 qi signal 不觸發;Phase 2 多 signal 進來前要改成 union horizons + `ic.get(h, nan)` + empty guard,並加 test。
- ⏳ **#3 (Minor) → Phase 2**:`compute_ic` 的 `drop_nulls()` 不會濾 `NaN`(一個 NaN 會毒化整個 horizon 的 IC)。Phase 1 的 `queue_imbalance` 用 null-guard 不產生 NaN,故安全。Phase 2 約定:**signal 函數永遠 null-guard,不讓 NaN 外漏**;或在 compute_ic 加 `is_finite` 過濾。
- 📝 **#4 (Minor)**:polars `qcut` 上游標記 unstable;`requirements.txt` 只寫 `polars>=1.0`。Phase 2 可考慮收緊版本範圍。
- 📝 **#5 (Minor)**:plan 的 File Structure 表列了 `scripts/recon/download_orderbook.py`,但從未有 task 建立它(probe CLI 直接呼叫 `download.py` 的函數)。純 stale 文件項,無實作缺口。

## ⚠️ 下一步關鍵 GATE:Step 0(手動,需網路 — 尚未跑)

這是 spec §3.1 的 de-risk gate,**Phase 2 動工前必跑**:

```bash
cd .claude/worktrees/recon-phase1
PYTHONPATH=src venv/bin/python -m scripts.recon.probe_schema --symbol ETHUSDT --date 2026-06-01
```

確認:`bookDepth` 實際 schema(**percentage-distance depth vs raw L2 levels** — 決定 depth imbalance / book slope 怎麼定義)、cadence、各 data type 每日檔案大小。若 `bookTicker` 原始欄位與 `download.py` 的 `BOOK_TICKER_MAP`/`BOOK_TICKER_TS_COL` 不符,只需改那兩個常數(signal/align/ic 不動)。若 `bookDepth` 不可得 → 走 spec 的 L1 + aggTrades fallback。

## Phase 2 範圍(待 Step 0 結果後再用 writing-plans 寫 plan)

OFI(Cont 2014)、depth imbalance、book slope(定義依 Step 0)、taker imbalance(aggTrades)、BTC→ETH 跨資產 lead-lag;多日 chunked / lazy 處理;OOS holdout;Newey-West / block-bootstrap 顯著性;粗略成本敏感度;plotly/notebook 視覺報告;spec §10 decision-rule 判定。並一併修 #2、#3。

## 待整合 / 待辦

- **worktree branch → main**:finishing-a-development-branch 處理中。
- **main 的 spec + plan commit**(`038ae50`、`62faa2f`)尚未 push origin——`git push origin main` 被 auto classifier 擋下,需使用者明確授權。
