# 交接:Recon 完結後的營運狀態(2026-06-29)

> **TL;DR**:**兩條研究線都已永久關閉**——方向性:六假設家族+24 測試全滅(2026-06-29,見 [archive](../archive/2026-06-29-recon-complete-strategic-fork.md));非方向性 funding carry:**v1(現在式稽核,差 1.0pp)與 v2(point-in-time 重開,差 0.2pp)先後死在同一道複製窗 G2**,2026-07-06 起兩個證據基礎永久關閉、無 v3(§1c)。**目前沒有進行中的研究**;唯一活的研究選擇權是 qi maker,gate 在錄滿 2-3 個月(§1b,約 2026-09/10)。營運面:兩個長跑在背景跑(§1a/1b),接手先 `pgrep -fl 'src.cli|record_book'` 確認活著。本文件自足。

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

**已知斷口與修復**:2026-07-05 03:47 UTC 錄製器整個 process 死掉(不是單純斷線)——根因:`AsyncClient.create()` 在 reconnect 迴圈的 `try` 外,休眠喚醒斷網時拋出、單一 shard task 死亡連帶殺掉整個 gather。已修(create 移入 try + client None 防護),2026-07-06 12:49 UTC 重啟。**資料斷口 ~33h(07-05 03:47 → 07-06 12:49 UTC)**,對「錄滿 2-3 個月」的目標無實質影響。若再發現 process 消失:`tail ~/record_book.log` + 用 §1b 的 nohup 指令重啟即可。

### 1c. Funding carry 研究(v1+v2 雙終局:FAIL,2026-07-06 起永久關閉)

**v2 補記(2026-07-06)**:使用者批准後以 point-in-time 證據基礎重開(spot 月度 kline 目錄 = 歷史上市狀態;466/791 可對沖,gate 一字不改)——**Step 0 存活、四 gate 全過(test +5.9%)、複製窗 G2 差 0.2pp 再死**。v1 差 1.0pp、v2 差 0.2pp,同一 gate 陣亡互為複製:**「輪動勝 lazy majors +2pp」在 2020-2022 是 regime 上不成立**,且誠實可對沖 universe 的 carry 只有 +5.9%(v1 的 +23.8% 大半來自不可對沖名字)。cash-and-carry 於兩個證據基礎下**永久關閉,無 v3**(詳見 `2026-07-06-funding-carry-pit-preregistration.md` §4)。

以下為 v1 記錄:

**問題**:跨全 universe(791 USDT 永續,含已下市)的 cash-and-carry(long spot + short perp 收正 funding)成本後能否過四道 gate?非方向性,不觸犯 2026-06-29 方向性終局條款。

**完整判定鏈(全程事先登記,參數零改動)**:Step 0 存活(毛上界 train +23.8%/test +71.4%)→ Phase 1 四 gate 全過(deployed 淨 test +23.8%,NW t=7.3)→ 複製窗全過(+19.8%,G2 只贏 0.6pp)→ **Phase 2 spot 稽核:99 個持倉名字 47 個今日無 spot 對沖,universe 791→381 重跑——主窗仍全過(test +7.3%),但複製窗 G2 差 1.0pp(+18.2% vs 需 19.2%)→ 依鎖定定義降級 FAIL,終局條款生效**。

**永久關閉**:「免費 Binance funding 數據 cash-and-carry」家族(含正/負側、任何參數/窗口變體)不重訪。留下的真相:spot 可對沖 universe 上的正 carry 現象在 2022-2026 主窗是真的(最嚴苛過濾下四 gate 全過),死的是「協議下可複製」這個 claim;複製窗敗因是 2020-2022 majors funding 本身就肥,輪動增量打不出 +2pp。方法論教訓:稽核 gate 要用 point-in-time 定義,現在式代理會誤殺歷史名字(MATIC/FTM)——教訓成立,verdict 不變。若未來取得實質不同證據基礎(point-in-time spot 上市史+借貸成本),構成**新問題**,須使用者明示批准另立登記。

文件:`docs/superpowers/plans/2026-07-05-funding-carry-preregistration.md`(§6 首跑/§7 終局);程式 `scripts/carry/{universe,study,spot_audit}.py`(15 tests);數據 `data/carry/`(~17MB,可重建)。

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
