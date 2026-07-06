# 事先登記:跨所 funding spread(Binance↔Bybit perp-perp)

**日期**:2026-07-06
**狀態**:**FINAL — FAIL(複製窗 2× 成本 gate −0.5%);終局賭注生效:免費數據研究篇章完結**(§6)
**開題授權**:免費 Binance 數據空間已全數掃畢(方向/carry/basis 三終局);新 venue 問題依終局條款例外通道,**使用者已於 2026-07-06 明示批准**。
**終局賭注**:本研究任一關 FAIL → 跨所 funding spread 於免費 CEX 數據永久關閉,且**整個免費數據研究篇章從此完結**(使用者批准選項之原文)。

## 0. 機制與先驗

兩所同名永續的 funding rate 持續分歧時:**short 高 funding 所 + long 低 funding 所**,delta-neutral 收差額。無 spot 腿、無借券,**正負兩側皆可做**;4 條 perp taker 腿成本低於 cash-and-carry。這是專業玩家實際在做的策略,分歧的存在性有公開文獻;問題是 **retail taker 成本後、事先登記協議下是否倖存**。

**先驗誠實揭露**:(a) Binance 側 funding 數據已在本地(carry 研究遺產),登記人知其水位分布;但 spread=兩所之**差**,是未曾觀察的新維度;Bybit 側數據零接觸。(b) 先驗:Step 0 存活機率中等(~50%)——majors spread 通常 <5% APR 被 arb 壓緊,alt spread 事件性放大;複製窗(2020-2022)Bybit alt 覆蓋薄,是已知風險。(c) 部署前提是 2+ 交易所開戶+資金分置——即使 PASS 也只是歷史期望值,不是部署授權。

## 1. Universe 與數據

| 項目 | 規則 |
|------|------|
| Venue 對 | **僅 Binance↔Bybit**(OKX/Hyperliquid 為未來另立問題;Hyperliquid 2023 中才上線且 1h funding,窗口不可比) |
| Binance 側 | 本地 `data/carry/funding/`(791 symbols 含已下市,S3 archives) |
| Bybit 側 | v5 API `GET /v5/market/funding/history`(category=linear,免費分頁);universe 由 `instruments-info` 現行清單枚舉——**Bybit 側倖存者偏差存在且無免費解**,如實揭露 |
| 名稱正規化 | 抽離 base 中的 `1000` 級數群組取 canonical key(Binance `1000SHIBUSDT` ↔ Bybit `SHIB1000USDT` 同 key);funding 率為百分比,單位換算不影響 spread;key 碰撞則棄該 symbol |
| 日網格 | UTC 日:`spread_day(s,D)` = Binance 日 funding 和 − Bybit 日 funding 和(自然處理 8h/4h/1h 異質) |
| 合格性 | 兩所皆有 ≥30 天歷史(風化期,沿用 carry 規則);兩所當日皆有數據 |

## 2. 窗口(沿用 carry,便於比較)

train 2022-07-01→2024-06-30;test 2024-07-01→2026-06-30;replication 2020-07-01→2022-06-30(Bybit 覆蓋薄為已知;gate 不因此放寬)。

## 3. 策略(鎖定)

- 訊號:`trail3(s,D)` = D-3..D-1 三日 spread 和(需齊全),年化 ×365/3;**方向 = −sign(trail3)**(short 高 funding 所)。
- 進場:每日 00:00 UTC,|trail3 年化| > **10%** 者按 |trail3| 由高到低補進,**K=5** 槽等權。
- 出場:|trail3 年化| < **5%**、或 **sign 翻轉**、或任一所數據終止(強制出場計成本)。
- 收益:持有日收 `direction × spread_day(s,D)`(選股只用 D-1 前數據,無 look-ahead)。
- 成本:每次進出場各 **15bps**(RT 30bps:Binance taker 5×2 + Bybit taker 5.5×2 ≈21 + 滑價 9);G4 加倍 60bps。
- 資本:deployed = **1.0×** notional(兩腿各 0.5 保證金含壓力緩衝);APR 以 deployed 計。
- 未建模並揭露:兩所價差 MTM、資金在兩所間的再平衡摩擦、ADL/清算引擎差異、venue 對手風險 ×2。

## 4. 判定

**Step 0 kill**:以 |trail3| 選前 5(無門檻無成本)收下一日 |實現 spread|……不——收 `direction×spread_day`(可實作毛利);任一 half 年化 < **10%** → FAIL,不進 Phase 1。

**Phase 1 gates**:
- G1 OOS:淨 APR(deployed)> 0,train 且 test。
- G2 懶惰對照:test 淨 APR ≥ **BTC+ETH spread 常駐**(方向依 trail3,不輪動不選股)淨 APR **+2pp**。
- G3:test 淨 APR ≥ **5%**。
- G4:2× 成本後 test 淨 > 0。

**複製窗**:全 gate 須再過。**任一關 FAIL → 依終局賭注關閉一切。** 全 PASS → REPLICATED,產出為歷史期望值 + 部署前提清單(雙所開戶/資金/實時 spread 監控),部署另議。

## 5. 產出物

`scripts/xvenue/bybit.py`(下載)、`scripts/xvenue/study.py`(spread 表 + Step 0 + Phase 1 + 複製)、`tests/unit/scripts/test_xvenue_study.py`、verdict 回寫本文件 + handoff。

## 6. FINAL VERDICT(2026-07-06 首跑,參數零改動)

**FAIL——複製窗 2× 成本 gate 翻負(−0.5%);依終局賭注,跨所 spread 永久關閉,且整個免費數據研究篇章從此完結。**

數據:563/563 Bybit symbols 下載零錯誤;559 個共同 canonical keys,374,406 日行。

| 判定 | 數字 | 結果 |
|------|------|------|
| Step 0 | train **+32.8%** / test **+62.9%**(kill 10%) | 存活 |
| Phase 1 | train **+11.7%** / test **+29.0%** / 2× **+11.8%** / lazy **−7.6%**;1123 進 1118 出 | **四道全過,全程最大幅度** |
| Replication 1× 成本 | halves +12.7%/+6.8%,full **+9.7%**(≥5% ✓;lazy −4.1%,G2 ✓) | 過 |
| Replication 2× 成本 | **−0.5%** | **FAIL** |

**結構性解讀(記錄,不作翻案)**:(a) 這是整個 program 最強的候選——主窗淨 +29% 且對成本加倍穩健;死因是**高換手**(sign-flip 出場,~85 cycle/年)撞上**早期 spread 薄**:2020-2022 複製窗成本拖累 ≈10pp/年,加倍即滅頂。(b) lazy 對照兩窗皆負(−7.6%/−4.1%)——majors spread 太小、翻向成本吃光,證明對照設計有效。(c) Bybit 側 2020-2021 alt 覆蓋薄 + 倖存者偏差(已登記揭露)使複製窗先天不利;這是本研究證據基礎的已知極限,不構成重跑理由。

**Program 終局聲明**:依使用者批准之條款,免費數據(Binance + 跨所 CEX)的獲利機制研究**全部完結**——方向性、funding carry(×2 證據基礎)、basis 均值回歸、跨所 funding spread,五案皆有事先登記終局。存活選項僅餘 **qi maker**(自錄 tick,約 2026-09/10 可開題)。任何重啟均須使用者明示批准且須有實質新證據基礎。
