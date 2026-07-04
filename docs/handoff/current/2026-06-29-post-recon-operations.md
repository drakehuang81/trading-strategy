# 交接:Recon 完結後的營運狀態(2026-06-29)

> **TL;DR**:研究線已**永久關閉**——六個假設家族 + 24 測試事先登記掃描,全數在四道 gate 下否證(完整脈絡見 [archive/2026-06-29-recon-complete-strategic-fork.md](../archive/2026-06-29-recon-complete-strategic-fork.md))。**目前沒有任何進行中的研究**。活著的只有兩個營運項目(§1),都便宜、都還沒開始。接手者不需要讀完整個 recon 歷史才能動手——本文件自足。

## 1. 兩個活著的項目(2026-06-29 拍板的框架中,尚未執行的部分)

### 1a. Paper 助理跑起來(營運,零研究成本)

讓 1h 助理以 paper 模式長跑,累積 Pre-Live Gate 要求的營運履歷(60 天 heartbeat、HALT 演練等——見 spec §10)。就算沒有 edge 模型,營運履歷與紀律本身就是產出。

前置(都是 ops,不是研究):
1. 主 repo 重建**完整** 3.11 venv(現有 `venv/` 是壞的 3.9;`pip install -r requirements.txt` 全裝,含 pydantic/sqlalchemy/ollama 等)
2. 安裝/啟動 Ollama + Gemma 模型(boot 會 ping,失敗則 LLM-disabled 模式繼續)
3. `python -m orchestrator`(boot 流程見 spec §4.8)+ Telegram token(`.env`,有 `.env.example`)
4. 驗證:完整 test suite(~373+38 個)應全綠

### 1b. TickRecorder 錄 book stream(保留 qi maker 選擇權)

qi(L1 imbalance)是唯一有真實資訊量的信號(秒級 IC 0.37),但 maker/HF 路線**無法回測**——公開歷史沒有逐筆 L2。唯一讓它未來可測的方法:**現在開始錄**。
- 現有 `src/execution/tick_recorder.py` 只錄 trades WS;需擴充 stream factory 訂 bookTicker/depth stream
- 錄 2–3 個月後,maker 假設才第一次可證偽;在那之前不投入任何 maker 研究

## 2. 環境與資產(接手必讀)

- **可用的 research venv**:`.claude/worktrees/recon-phase2b1/venv`(3.11,精簡集)。research tests:`venv/bin/pytest tests/research/microstructure/ -q`(38 個,scoped 跑)。
- **⚠️ 數據快取 ~7GB 在同一個 worktree**:`data/orderbook/{_fw,_cross,_sweep,_integ}/`——12 alt + BTC + ETH 的 depth/klines。**刪 worktree 前先搬走**,重下載要數小時。
- 主 repo `venv/` 壞(3.9),別用。根目錄 README 過時(pivot 前),有待辦 chip。
- 通用信號驗證 harness(`scripts/recon/{depth,cross}_validation.py`、`sweep_symbols.py`)可對任何 signal × symbol × window 出四道 gate verdict——**但依終局條款,不再用於免費 Binance 方向假設**;留給未來新數據源(如自錄 L2)或新市場。

## 3. 完整歷史指路

- Recon 全史 + 決策紀錄:`docs/handoff/archive/2026-06-29-recon-complete-strategic-fork.md`
- 終局:`docs/superpowers/plans/2026-06-29-orderbook-recon-phase2c-STATUS.md`
- 主系統設計:`docs/superpowers/specs/2026-04-18-personal-trading-assistant-design.md`

---
*建立:2026-06-29。1a/1b 完成或放棄後,本文件移入 archive。*
