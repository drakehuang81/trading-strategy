# 交接:Recon 完結後的營運狀態(2026-06-29)

> **TL;DR**:**方向性**研究線已永久關閉——六個假設家族 + 24 測試事先登記掃描全滅(脈絡見 [archive/2026-06-29-recon-complete-strategic-fork.md](../archive/2026-06-29-recon-complete-strategic-fork.md))。但 **2026-07-05 非方向性 funding carry 研究首次全 gate PASS & REPLICATED**(§1c)——這不觸犯終局條款(它關的是方向 edge)。營運面:兩個長跑 2026-07-05 起在背景跑(§1a/1b),接手先 `pgrep -fl 'src.cli|record_book'` 確認活著。本文件自足。

## 1. 兩個活著的項目(2026-06-29 拍板的框架中,尚未執行的部分)

### 1a. Paper 助理跑起來(營運,零研究成本)

讓 1h 助理以 paper 模式長跑,累積 Pre-Live Gate 要求的營運履歷(60 天 heartbeat、HALT 演練等——見 spec §10)。就算沒有 edge 模型,營運履歷與紀律本身就是產出。

**✅ 2026-06-29 已完成(可開機)**:
1. ✅ 主 repo 3.11 venv 重建完成(注意:`instructor` 需 `<1.15.2`,已釘進 requirements——1.15.4 會 hard-reject ollama client)
2. ✅ `.env`(Telegram token)已從舊 pivot worktree 復用
3. ✅ boot 煙霧通過:`PYTHONPATH=src venv/bin/python -m src.cli` → `boot_complete` + `telegram_bot_started`,SIGINT 乾淨關機
4. ✅ 完整 suite 綠(production + research,~416 tests)

**✅ 2026-07-05 長跑已啟動**:`nohup env PYTHONPATH=src venv/bin/python -m src.cli >> ~/orchestrator.log 2>&1 &`(主 repo 根目錄執行;boot_complete + telegram_bot_started 已驗證)。停止:`pgrep -f 'src.cli'` → `kill -INT <PID>`。**同時只能有一個實例**(Telegram getUpdates 衝突)。

**尚未做(要用時再做)**:
- 24/7 需 `caffeinate -dis &` 或 always-on 機器(laptop 休眠時暫停,醒來自動 catch up,見 `docs/SANDBOX_OPS.md`)
- Ollama + Gemma(**可選**——free-text chat 才需要,沒裝 bot 不死;`brew install ollama` + pull 模型 ~10 分鐘)

### 1b. TickRecorder 錄 book stream(保留 qi maker 選擇權)

qi(L1 imbalance)是唯一有真實資訊量的信號(秒級 IC 0.37),但 maker/HF 路線**無法回測**——公開歷史沒有逐筆 L2。唯一讓它未來可測的方法:**現在開始錄**。

**✅ 2026-06-29 錄製器已完成並實測**:`scripts/record_book.py`(雙 shard socket + 自動重連 → `data/ticks/<kind>/<SYMBOL>/<date>.jsonl`)。

**✅ 2026-07-05 長跑已啟動 + 精簡錄製決策**:
- 實測揭穿原估計:bookTicker ~250 msg/s/symbol ≈ **11GB/天**(不是 400-600MB),磁碟只剩 28GB → 2.5 天就滿。
- **決策(使用者拍板)**:預設丟 bookTicker,只錄 **depth5@500ms + aggTrade**(~0.5GB/天原始),錄製器內建**每小時 gzip 過往日檔**(~8-15x → **~50MB/天**,3 個月 ~5GB)。
- **理由**:qi(L1 imbalance)從 depth5 頂層即可算,500ms 解析度對秒級假設夠用;bookTicker 的增量價值只在 tick 級 maker 排隊模擬,要用 `--kinds` 明確加回。已錄到的 ~70MB bookTicker 樣本壓縮保留在 `data/ticks/bookTicker/`。
- 啟動:`nohup env PYTHONPATH=src venv/bin/python -m scripts.record_book >> ~/record_book.log 2>&1 &`(主 repo 根目錄)。驗證:`stat -f "%z %N" data/ticks/*/*/*.jsonl` 間隔 10 秒比對有增長。

錄滿 2–3 個月前不投入任何 maker 研究。laptop 休眠會斷錄(24/7 要 `caffeinate`);中斷可接受——資料是 append-only 日檔,重啟即續。

### 1c. Funding carry 研究(2026-07-05 新開,首個全 PASS)

**問題**:跨全 universe(791 USDT 永續,含已下市)的 cash-and-carry(long spot + short perp 收正 funding)成本後能否過四道 gate?**非方向性,不觸犯 2026-06-29 終局條款。**

**狀態:PASS & REPLICATED**(專案史上第一個)——事先登記 → Step 0 毛上界存活(train +23.8%/test +71.4% notional)→ 四 gate 全過(deployed 淨 APR:train +7.4%/test +23.8%,lazy 對照 +3.4%,2× 成本 +23.0%)→ 複製窗 2020-2022 再全過(+19.8%,**G2 只贏 0.6pp**)。NW t=7.3/4.2。

**必讀警語**:Sharpe 15/maxDD 0.5% 是**記帳模型假象**(basis MTM 未建模);真實風險=基差波動下的保證金壓力+交易所風險。test 窗 +23.8% 是牛市 regime;乏味 regime 合理預期是 train 的 +7.4%。

**下一步(事先登記,未跑)**:Phase 2 spot 可得性稽核——持倉名單逐一核對 USDT spot 對,剔除無對沖者後重跑,全 gate 須仍過,否則降級 FAIL。**在 Phase 2 過之前,不寫任何執行/部署程式碼。**

文件:`docs/superpowers/plans/2026-07-05-funding-carry-preregistration.md`(§6 verdict);程式 `scripts/carry/{universe,study}.py`;數據 `data/carry/`(~17MB,可由 universe.py 重建)。

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
*建立:2026-06-29;2026-07-05 更新(兩長跑啟動 + 精簡錄製決策)。錄滿 2–3 個月(或放棄)後,本文件移入 archive。*
