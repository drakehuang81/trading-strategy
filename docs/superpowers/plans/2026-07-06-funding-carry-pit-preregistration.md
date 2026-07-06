# 事先登記 v2:point-in-time 可對沖 universe 上的 funding carry

**日期**:2026-07-06
**狀態**:PRE-REGISTERED(本文件 commit 後才允許下載 spot 上市史數據)
**開題授權**:v1 終局條款規定新證據基礎須使用者明示批准——**已於 2026-07-06 批准**(選項「批准:point-in-time carry」)。

## 0. 與 v1 的關係(2026-07-05-funding-carry-preregistration.md,FINAL FAIL)

v1 死因:Phase 2 稽核用**現在式** spot 檢查(當時以為歷史上市狀態不可得),誤殺歷史上可對沖的名字(MATIC/FTM/EOS...),過濾後複製窗 G2 差 1.0pp。v1 verdict 不變、不翻案;其關閉範圍是「免費 Binance funding 數據 + 現在式/無 spot 驗證」這條證據基礎。

v2 是**新問題**:改用 `data.binance.vision` 的 **spot 月度 kline 目錄**作為 point-in-time spot 上市史(某 symbol 某月有 1d kline 月檔 = 該月 spot 可交易)。這是 v1 §7 預留的「實質不同證據基礎」第一例。

**先驗知識完整揭露(誠實聲明)**:登記人已看過 v1 全部結果——主窗在最嚴苛過濾下全過、複製窗 G2 差 1.0pp;且可預期 PIT universe 在 2020-2022 比現在式 381 個**更大**(含 MATIC 時代名字),對複製窗 G2 大概率有利。v2 的正當性不在「換到會過的定義」,而在:(a) PIT 是**經濟上正確**的定義(量測交易當時的可對沖性);(b) 該定義在 v1 §7 postmortem 中、v2 構想之前就被指名為正確做法;(c) **gate 與參數與 v1 一字不改**,不存在調參空間。若 v2 仍 FAIL,carry 方向在兩個證據基礎下都死,永久關閉。

## 1. 相對 v1 的唯一變更(其餘全部沿用 v1 §1–§4,一字不改)

| 項目 | v1 | v2 |
|------|----|----|
| 合格性 | 上市 ≥30 天 | 上市 ≥30 天 **且** 當月 spot 可對沖(PIT) |
| spot 可對沖定義 | (Phase 2 才檢查,現在式) | `data/spot/monthly/klines/<BASE>USDT/1d/` 存在該月檔案;1000x 前綴按單位換算映射 BASE |
| 持有中 spot 下市 | (未定義) | **強制出場並計成本**(當月起不可對沖 → 該日視同數據終止) |
| 月粒度誤差 | — | 上市/下市月的 ±半月邊界噪聲,接受不修 |
| Phase 2 稽核 | 現在式 spot 檢查 | **不需要**——PIT 過濾已內建於 universe,無殘餘稽核 |

參數(trail3、10%/5% 帶、K=5、40bps RT、1.4× deployed、窗口、Step 0 kill 10%、G1–G4、複製窗)**全部與 v1 相同**,機器可讀鏡像仍為 `scripts/carry/study.py::PRE_REGISTERED`。

## 2. 判定與終局條款

- Step 0(PIT universe)任一 half < 10% 毛 APR → FAIL。
- Phase 1 四道 gate(G1 OOS / G2 lazy+2pp / G3 ≥5% / G4 2× 成本)須全過。
- 複製窗 2020-07→2022-06 四道須再全過。
- **任何一關 FAIL → carry 方向在 Binance 免費數據的兩個證據基礎下永久關閉**,不再有 v3;唯一例外是未來出現「非 Binance 數據」的全新 venue 問題,依慣例另行使用者批准。
- 全 PASS → 產出物是「可信的歷史期望值」,**不是部署授權**——部署仍受 Pre-Live Gate 與後續營運決策管轄。

## 3. 產出物

- `scripts/carry/spot_history.py` — 791 個 perp 對應 spot 對的月度上市史(S3 目錄列表 → `data/carry/spot_months.json`)
- `scripts/carry/study_pit.py` — PIT 過濾 + 沿用 study.py 全部函數重跑全鏈
- `simulate()` 增加「不可對沖 → 強制出場」語義(對 v1 輸入可證明為 no-op:v1 的 eligible 對時間單調不減)
- verdict 寫回本文件 + handoff current
