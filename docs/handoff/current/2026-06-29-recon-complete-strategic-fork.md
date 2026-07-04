# 交接:Recon Program 完結 → 戰略分岔點(2026-06-29)

> **TL;DR**:order book edge 偵察計畫已完整跑完——六個免費數據方向假設,六個否證(全部通過事先承諾的四道 gate 檢驗)。程式碼與紀錄全部在 `main`(`572c60b`)並已同步遠端。專案現在站在一個**戰略選擇點**(§7;§6 是走到這裡的決策紀錄),不是技術卡點。接手者的第一件事是和 Drake 確認走哪條路,而不是寫 code。

## 1. 專案是什麼

個人加密貨幣交易助理,從規則式 bot pivot 成 **本地 LLM 強化的 1h 助理**:M1 Pro 16GB、$0 雲端、Telegram 介面、XGBoost 出機率 + Gemma(Ollama)只出 boolean veto、paper 預設、8 道 Pre-Live Gate 才能實盤。完整設計見 [specs/2026-04-18-personal-trading-assistant-design.md](../../superpowers/specs/2026-04-18-personal-trading-assistant-design.md)。

**基礎設施(Plan 1–5E)早已完成且合併**:六層架構、PaperBroker/ReplayBroker、walk-forward backtest + Deflated Sharpe、drift monitor、Telegram + ChatLLM,~373 個生產測試。缺的從來不是管線,是 **edge**。

## 2. 走到這裡的路(時間線)

| 階段 | 結論 |
|---|---|
| Plan 5A→5E(2026-04) | TA 特徵對 ETHUSDT 1h 方向無預測力(Brier ≥ 0.25 擲硬幣基線;跨 2 種校準、2 種標籤、4 種 horizon) |
| 策略探索(2026-06 歸檔) | BTC/ETH 配對交易(ratio 是趨勢,half-life 151 天)、低波動突破(OOS 崩潰)皆死 → `scripts/btc_eth_ratio_analysis.py` 等三檔,結論寫死在 docstring |
| **Recon Phase 1**(2026-06-28) | order book 偵察薄管線:downloader / L1 queue imbalance / mid-grid / Spearman IC / report,端到端打通 |
| **Step 0 偵察** | 三個決定性數據事實(見 §4) |
| **Phase 2a** | bookDepth/aggTrades loaders + 4 信號(OFI、depth imbalance、book slope、taker imbalance)+ 多信號 pipeline |
| **單日 integration + cost check** | qi 秒級 IC 0.37 但 edge 0.2–0.9 bps << taker fee → **maker-only**;depth@1h 單日 IC 0.50、扣 taker 淨 +14 bps → 成為唯一 lead |
| **Phase 2b-1**(depth 驗證) | **FAILED**:全窗口 IC 0.046(單日 0.497 是 regime artifact)、連動能控制都贏它、淨 −7.1 bps、不 monotone |
| **Phase 2b-2**(cross-asset) | **FAILED**:BTC book → ETH IC 0.042,雙動能控制(ETH \|0.067\| / BTC \|0.081\|)都比信號強、淨 −7.1 bps |

**最終計分板:六個假設、六個否證。**「用免費公開數據做 ETH 方向預測」已被系統性回答:**沒有 edge**。詳細數字見 [2b-1 STATUS](../../superpowers/plans/2026-06-29-orderbook-recon-phase2b1-STATUS.md)、[2b-2 STATUS](../../superpowers/plans/2026-06-29-orderbook-recon-phase2b2-STATUS.md)。

## 3. 現在擁有的資產

- **通用信號驗證 harness**(`src/research/microstructure/` + `scripts/recon/`,37 tests):任何(信號 × 資產 × 窗口)一個指令 → 四道 gate verdict。四道 gate(事先承諾、防 p-hacking):
  1. OOS:train/test IC 同號且 |ic_test| > 0.1
  2. vs-controls:|IC| > 動能控制 + 0.05(cross 版是雙控制取 max)
  3. post-cost:decile edge − 8 bps taker > 0
  4. 全 bucket monotone
- **數據管線**:bookTicker/bookDepth/aggTrades/klines 下載+標準化 loaders(header 容錯)。
- **本地數據快取**(gitignored,在 worktree `recon-phase2b1` 的 `data/orderbook/` 下):`_fw/`(ETH depth+klines 320 天)、`_cross/`(BTC depth + 雙邊 klines 320 天)、`_integ/`(2023-06-01 單日三數據源)。**刪掉 worktree 前先搬走,重下載要 ~30 分鐘。**
- 生產 1h 助理基礎設施(Plan 1–5E),原封不動等一個真有 edge 的模型。

## 4. 關鍵事實與坑(hard-won,別重踩)

1. **bookTicker daily 只存在 2023-05-16 → 2024-03-30**(Binance 之後停更 USD-M bookTicker)。bookDepth 2023-01→今、aggTrades 2019→今。三源重疊窗口被釘死在那 10.5 個月。
2. **bookDepth 是 percentage-distance depth,不是 raw L2**:12 個對稱 level(±0.2/1/2/3/4/5%)、~33s/snapshot、`timestamp` 是**字串**。
3. **um-futures daily kline CSV 帶 header 列**(320/320 檔都有)——`load_klines_1h` 已容錯(全字串讀入→過濾數字列→轉型)。老「headerless」傳說只適用舊 spot 檔。
4. **成本的 binding constraint 是 taker fee(round-trip 3.6–8 bps),不是 spread(ETH perp 中位數 0.054 bps)**。任何秒級~分鐘級信號先過這關再談。
5. **單日/單段的漂亮數字不可信**——depth 單日 IC 0.50 → 全窗口 0.046;Plan 5E 的 Sharpe 1.61 → buy-and-hold beta。永遠跑全窗口 + OOS + 動能控制。
6. **環境**:主 repo 的 `venv/` 是**壞掉的舊 3.9**,不要用。可用的 3.11 venv 在 `.claude/worktrees/recon-phase2b1/venv`(polars 1.42 / scipy / statsmodels / pytest;**精簡集,無 pydantic 等生產依賴**——research tests 一律 scoped 跑 `tests/research/microstructure/`,全 suite 會有 ~50 個既有 collection error,是預期)。
7. **`pythonpath=[".","src"]` 是 pytest-only**——bare `python` 跑 script 要 `PYTHONPATH=src`。
8. polars:`rolling_sum` 用 `min_samples=`(不是 `min_periods=`);`qcut` label 要 zero-pad 否則 n≥10 排序錯(已修);`drop_nulls()` 不濾 NaN(compute_ic 已加 `is_finite`)。
9. worktree 殘留:`.claude/worktrees/{recon-phase1,recon-phase2a,recon-phase2b1}` 內容全已 merge,無害;清理需手動(`ExitWorktree` 只認建立它的 session)。**注意 §3 的數據快取在 phase2b1 裡。**
10. 根目錄 `README.md` 還停在 pivot 前(Streamlit dashboard 時代),已過時。

## 5. 快速接手指令

```bash
cd /Users/drakehuang/SideProject/Trading/quant-trading-project/.claude/worktrees/recon-phase2b1

# research tests(37 個,~2s)
venv/bin/pytest tests/research/microstructure/ -q

# 重跑 depth 驗證 gate(數據已快取,~1 分鐘)
PYTHONPATH=src venv/bin/python -m scripts.recon.depth_validation \
    --symbol ETHUSDT --start 2023-05-16 --end 2024-03-30

# 重跑 cross-asset gate(同樣已快取)
PYTHONPATH=src venv/bin/python -m scripts.recon.cross_validation \
    --lead BTCUSDT --lag ETHUSDT --start 2023-05-16 --end 2024-03-30
```

## 6. 本輪的戰略決策紀錄(為什麼走到這裡)

recon program 的每個分岔點、當時的選項、Drake 的決定與結果——接手者請延續同一個決策原則:**每一步都選「最便宜的可證偽下一步」**(整個 program 因此只花了 2026-06-28 → 06-29 兩天就得到確定性答案)。

| 時點 | 面臨的選項 | 決定 | 理由 | 結果 |
|---|---|---|---|---|
| Plan 5E 終局後 | 認真找 edge / 當研究平台 / 先評估值不值得 | **認真找 edge** | 目標是真能實盤的策略,接受引入新數據源 | 開啟 recon program |
| 選 edge 來源 | 多資產動能 / **order book 微結構** / on-chain | **order book** | 理論 alpha 最強(OFI 文獻),接受數據門檻最高 | Step 0 證實歷史數據可得 |
| 頻率架構衝突 | 直接建分鐘級子系統 / 聚合進 1h 架構 / **先做 edge 偵察** | **先偵察再選架構** | 便宜、快,用 IC-vs-horizon 數據決定,避免賭錯架構 | 事後證明兩個「直接建」方向都會白做 |
| 偵察範圍 | 精實 / **完整** / 超精實試水 | **完整偵察** | 寧可多測,不因範圍太窄漏掉 edge | 五個信號全數實測 |
| 單日 cost-check 後 | (數據自動收斂,無需人工選) | 聚焦驗證 depth@1h(→ 2b-1) | 唯一扣 taker fee 後淨正的信號,且 horizon 契合 1h 架構 | **FAILED**(單日 IC 0.50 是 regime artifact) |
| 2b-1 失敗後的分岔 | qi maker 路線 / **cross-asset lead-lag** / 收手 | **cross-asset(→ 2b-2)** | 最後一個沒測過的免費數據假設;harness 現成、成本最低;maker 路線是整個新專案 | **FAILED**(雙動能控制都比信號強) |
| **現在(2b-2 失敗後)** | 見 §7 三條路 | **未決** — 等 Drake 拍板 | — | — |

注意:§7 的三條路裡**不再包含 cross-asset**——它已在 2b-2 被選過、測過、否證,別重走。

## 7. 未決事項:戰略分岔(接手後第一件事)

三條路都是**結構性不同的方向**,由 Drake 拍板,不是預設繼續哪條:

1. **qi maker/HF 做市路線** — 唯一有真實資訊量的信號(秒級 IC 0.37)。但需要 maker-fill 模擬、queue position、庫存風險——是**新專案**,從 brainstorming/spec 開始,不是在現有 harness 上加 task。
2. **換市場** — harness 現成(`--lead/--lag/--symbol` 參數化),但 data.binance.vision 只有 Binance;其他 venue 要新 loaders。先想清楚「哪個市場的費用/波動比更友善」再動手。
3. **收手/守成** — 讓 1h 助理以 paper 模式跑著(需完整 venv + Ollama,見 spec §4.8 boot 流程),把六輪否證當作「這套流程可信」的證明。

## 8. 文件地圖

- 設計 spec:`docs/superpowers/specs/2026-04-18-personal-trading-assistant-design.md`(主系統)、`2026-06-28-orderbook-microstructure-recon-design.md`(recon)
- Recon plans + STATUS:`docs/superpowers/plans/2026-06-28-orderbook-recon-phase1.md`、`...-phase2a.md`、`2026-06-29-...-phase2b1-depth-validation.md`(含 STATUS)、`...-phase2b2-cross-asset.md`(含 STATUS)
- 單日發現(含 cost-check UPDATE):`docs/superpowers/plans/2026-06-29-orderbook-recon-phase2a-integration-findings.md`
- 舊 pivot 交接:`docs/superpowers/plans/2026-04-28-pivot-plan5e-STATUS.md`(TA 線的終局)

---
*建立:2026-06-29。此文件屬 `handoff/current/`;戰略方向拍板並完成(或放棄)後,移入 `handoff/archive/`。*
