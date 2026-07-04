# Sandbox Operations Handoff

> **下次坐下來先讀這份。** 5 分鐘掌握目前狀態 + 開關方式 + 常見問題。

## 我現在能做什麼

這個 codebase **不是** profitable trading bot，是**個人 quant sandbox** — 跑得起來、所有元件可互動，但**沒 trading edge**（6 個獨立策略嘗試全 OOS 失敗，spec §12 紅旗成真，詳見 `docs/superpowers/plans/2026-04-28-pivot-plan5e-STATUS.md`）。

可以做的：
- 看 Telegram bot 回應 (`/status` `/positions` `/halt` `/resume` `/analyze`)
- 觀察 hourly scan 跑（永遠不發單，因 stub model `prob_up=0.55 < threshold=0.58`）
- 跑歷史 backtest (`scripts/backtest.py`) 試新 idea
- 跑 horizon sweep (`scripts/sweep_horizons.py`)
- 跑 funding harvest analysis (`scripts/btc_eth_ratio_analysis.py` / `vol_breakout_analysis.py`)

不能做的：
- 真錢交易（`broker_kind=live` 會被 8-gate Pre-Live Gate 擋下）
- 對 sandbox 期待真 alpha（model 沒 edge）

## 開機

> 2026-06-29 起主 repo 即為運行環境(3.11 venv 已重建,pivot 內容早已 merge)。
> 不要再從 `.worktrees/pivot-foundation` 跑。

```bash
cd /Users/drakehuang/SideProject/Trading/quant-trading-project
PYTHONPATH=src venv/bin/python -m src.cli
```

預期看到（JSON log）：
```
{"event": "boot_complete", ...}
{"event": "telegram_bot_started", ...}
```

之後 console 安靜（log 走 structlog JSON，每小時 scan + 30 秒 heartbeat 才有新行）。

**讓它前景跑 → 你按 Ctrl-C 才結束**。要背景跑就在另一個 terminal：

```bash
nohup python -m src.cli >> ~/orchestrator.log 2>&1 &
```

不過 sandbox 階段建議前景跑，方便看 log。

## 關機

**前景**：`Ctrl-C` → orchestrator graceful shutdown 5 秒內結束（會關 Telegram bot、寫 final heartbeat）。

**背景**：
```bash
# 找 PID 並 kill
pgrep -fl 'python -m src.cli' | head
kill -INT <PID>   # 等同 Ctrl-C，graceful shutdown
# 如果 30s 後還沒死才用 kill -9
```

## Telegram 設定

`.env` 已建好，含 token。**`.env` 在 `.gitignore`，不會 commit**。

| 變數 | 來源 |
|------|------|
| `TELEGRAM_BOT_TOKEN` | `@BotFather` `/newbot` |
| `TELEGRAM_CHAT_ID` | `@userinfobot` 隨便發訊息他回 |

驗證 wiring：

```bash
PYTHONPATH=src python scripts/telegram_smoke.py
```

## Telegram 指令對照

| Command | 不需要 Ollama | 做什麼 |
|---------|--------------|--------|
| `/status` | ✅ | HALT + heartbeat + positions |
| `/positions` | ✅ | open positions（現在永遠 0）|
| `/halt` | ✅ | 手動 HALT |
| `/resume` | ✅ | 跑 trigger 檢查再放行 |
| `/analyze ETHUSDT` | ✅ | on-demand deep scan |
| 任意 free-text | ❌ | 走 ChatLLM → Ollama，**沒裝 Ollama 會 silently log error 但 bot 不死** |

## 休眠行為

電腦休眠時 Python + Ollama daemon 全 SIGSTOP 暫停。Telegram 訊息 Telegram server 端**保留 24h**，醒來自動 catch up。

- **sandbox 模式**（laptop 平常用）：休眠 OK，醒來 bot 補處理
- **想 24/7 跑**：另一個 terminal `caffeinate -dis &`，不睡。要關掉 `pkill caffeinate`

未來真要 production 必須上 always-on 機器（VM / Pi），laptop 不適合。

## Pre-Live Gate（live mode 安全網）

切 `cfg.broker_kind="live"` 啟動會自動跑 8 個 gate；任何一個 ❌ → `PreLiveGateBlocked` 拒絕 boot。

手動 check：
```bash
PYTHONPATH=src python -m scripts.pre_live_gate
```

目前狀態 4/8 passed（`calibration_brier`、`paper_runtime`、`watchdog_uptime`、`halt_diversity` 都 ❌）。**所以即使你誤切 live，1 秒內就會被擋**。

## Book stream 錄製器(qi maker 選擇權保留)

Binance 2024-03 起不再發布歷史 bookTicker;要讓 maker/HF 假設未來可測,唯一方法是自己錄。長跑:

```bash
PYTHONPATH=src venv/bin/python -m scripts.record_book
# 24/7:另開 terminal `caffeinate -dis &`
```

錄 BTC/ETH × aggTrade/depth5@500ms → `data/ticks/<kind>/<SYMBOL>/<date>.jsonl`(UTC 日切),過往日檔每小時自動 gzip(~8-15x)。
注意:Binance futures ws 已分流——book 類只在 legacy 端點、aggTrade 只在 market shard,錄製器自動開雙 socket。
量級(2026-07-05 實測):預設兩路 ~0.5GB/天原始、gzip 後 ~50MB/天。**bookTicker 已從預設拿掉**——實測 ~250 msg/s/symbol ≈ 11GB/天(佔 96%),laptop 磁碟撐不住;qi(L1 imbalance)從 depth5 頂層即可算(500ms 解析度,秒級假設夠用)。要 tick 級 L1 用 `--kinds` 明確加回。

## 目前 git 狀態(2026-06-29 更新)

- 一切在 `main`,與 origin 同步;pivot / recon 歷史全數 merge。
- 專案現況與交接:**先讀 `docs/handoff/current/`**(recon program 已終局關閉,詳見 archive)。
- requirements 注意:`instructor` 已釘 `<1.15.2`(1.15.4 會 hard-reject ollama client)。

## Plan 5 軌道全部文件位置

```
docs/superpowers/specs/2026-04-18-personal-trading-assistant-design.md  ← 整個系統的 spec
docs/superpowers/plans/2026-04-18-pivot-foundation-STATUS.md           ← Plan 1（架構）
docs/superpowers/plans/2026-04-18-pivot-plan2-STATUS.md                ← Plan 2（model）
docs/superpowers/plans/2026-04-19-pivot-plan3-STATUS.md                ← Plan 3（interface + ops）
docs/superpowers/plans/2026-04-19-pivot-plan4-STATUS.md                ← Plan 4（orchestrator）
docs/superpowers/plans/2026-04-25-pivot-plan5a-STATUS.md               ← Plan 5A（real model）
docs/superpowers/plans/2026-04-26-pivot-plan5b1-STATUS.md              ← Plan 5B-1（funding backfill）
docs/superpowers/plans/2026-04-26-pivot-plan5b2-STATUS.md              ← Plan 5B-2（ReplayBroker）
docs/superpowers/plans/2026-04-26-pivot-plan5b3-STATUS.md              ← Plan 5B-3（backtest + DSR）
docs/superpowers/plans/2026-04-27-pivot-plan5b4-STATUS.md              ← Plan 5B-4（Pre-Live Gate）
docs/superpowers/plans/2026-04-27-pivot-plan5c-STATUS.md               ← Plan 5C（triple-barrier — negative）
docs/superpowers/plans/2026-04-28-pivot-plan5e-STATUS.md               ← Plan 5E（horizon sweep — negative）
```

5G/5D-1/5D-2 進行中沒寫 STATUS — 直接看 git log。

## 下一步可選

| 方向 | 工作量 | 期待 |
|------|--------|------|
| 裝 Ollama → 啟用 free-text chat | 10 分鐘 | sandbox 體驗升級 |
| 完成 Plan 5D 剩下（HALT escalation、real LiveBroker stub→真 Binance） | 中 | 為未來真 model 鋪路 |
| **找新 strategy concept** | 大 | 唯一可能讓系統有 edge 的方向（5G 試了 3 個都失敗，需要更深思熟慮） |
| 接受現狀，**這就是收尾** | 0 | sandbox 在這狀態已經是價值，未來想到新 setup 直接套 |

## 我下次回來時會忘記什麼

- 你之前討論到「換策略」3 種候選都試完了：funding harvest（邊際）、BTC/ETH ratio（死）、vol-breakout（OOS 死）
- 結論是 venue（ETHUSDT 1h, retail）本身難找 edge，不是 codebase 問題
- 所以「再來一個策略」要先**仔細想 venue/setup 變不變**，不要急著寫 code
