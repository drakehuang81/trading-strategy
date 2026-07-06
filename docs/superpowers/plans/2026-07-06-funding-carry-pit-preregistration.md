# 事先登記 v2:point-in-time 可對沖 universe 上的 funding carry

**日期**:2026-07-06
**狀態**:**FINAL — FAIL(複製窗 G2 差 0.2pp);carry 於兩個證據基礎下永久關閉,無 v3**(§4)
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

## 4. FINAL VERDICT(2026-07-06 首跑,參數零改動)

**FAIL——複製窗 G2 差 0.2pp;§2 終局條款生效:cash-and-carry 於免費 Binance 數據的兩個證據基礎(現在式/point-in-time)下永久關閉,不再有 v3。**

PIT universe:466/791 個 perp 有 spot 上市史(比現在式 381 多 85 個歷史名字)。

| 判定 | 數字 | 結果 |
|------|------|------|
| Step 0 | train +16.9% / test +18.9%(kill 10%) | 存活 |
| Phase 1 | train +6.9% / test **+5.9%** / 2× +3.8% / lazy +3.4% | **四道全過**(G2 剩 0.5pp、G3 剩 0.9pp) |
| Replication | halves +33.0%/+5.1%,full **+19.0%**,2× +16.1%,lazy +17.2% | **G2 需 ≥19.2%,差 0.2pp → FAIL** |

**結構性結論(兩個獨立證據基礎、同一 gate 陣亡,這是 regime 事實不是資料工件)**:

1. **正 carry 期望值存在但薄**:誠實 PIT universe 上 test 淨 APR 只有 +5.9%——v1 未過濾版的 +23.8% 大半來自**不可對沖名字**的 funding。可實際執行的 carry,扣掉未建模的 basis MTM 與交易所風險後,與穩定幣/短債收益的差距不具吸引力(G3 只剩 0.9pp 餘裕)。
2. **「輪動勝 lazy majors +2pp」在 2020-2022 不成立**:當年 majors funding 本身 +17.2%,alt 輪動的增量打不進 2pp。v1(差 1.0pp)與 v2(差 0.2pp)在同一處死亡,互為複製。
3. 方法論教訓已兌現仍不夠:把 v1 的定義缺陷修掉(PIT)確實讓複製窗變好(18.2%→19.0%),但差距的主體是 regime,不是量測誤差。

**關閉範圍**:免費 Binance 數據 cash-and-carry,全部證據基礎、全部參數/窗口變體,**永久**。未來唯一例外通道是「非 Binance 數據的全新 venue 問題」,依慣例須使用者明示批准另立登記。
