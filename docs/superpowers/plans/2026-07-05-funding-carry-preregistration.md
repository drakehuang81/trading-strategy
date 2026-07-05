# 事先登記:跨 universe funding carry(cash-and-carry)研究

**日期**:2026-07-05
**狀態**:**PASS & REPLICATED**(首跑 verdict 見 §6;Phase 2 spot 稽核未跑)
**類型**:非方向性 carry 研究——**不觸犯 2026-06-29 終局條款**(該條款關閉的是「免費 Binance 數據找*方向性* edge」;本研究不預測價格方向,收的是 funding 現金流)

## 0. 為什麼這個問題值得一發子彈

- 先前「funding harvest(邊際)」的 verdict 來自**對話層級的粗估,且只看 majors**(BTC/ETH funding ~5-11% APR vs 成本)。repo 內從無跨 universe、成本感知、事先登記的 carry 研究(5E STATUS 將其列為未試的 P2 選項)。
- Alt 永續的 funding 分布與 majors 是不同母體:牛市階段常見 30-100%+ 年化。跨截面輪動是否能在成本後存活,是**真空白**。
- 機制上這不是預測,是收保險費:多頭付 funding 給空頭時,short perp + long spot 是 delta-neutral 收款方。可證偽、可實作、免費數據可測。
- 已抽驗:`data.binance.vision` 月度 fundingRate 檔案涵蓋 2020-01→2026-06,S3 目錄**含已下市合約**(無倖存者偏差),schema = `calc_time, funding_interval_hours, last_funding_rate`。

**參數選定聲明**:以下所有參數在只看過 BTCUSDT 2026-06 單月檔(schema 抽驗)的情況下,由成本結構與第一性原理推導鎖定。任何跑完後的參數修改都構成 p-hacking,verdict 不得更改。

## 1. Universe 與數據

| 項目 | 規則 |
|------|------|
| 枚舉來源 | S3 `data/futures/um/monthly/fundingRate/` 目錄列表(含已下市) |
| 過濾 | symbol 以 `USDT` 結尾(排除 USDC/BUSD 本位) |
| 數據來源 | 月度 zip → per-symbol parquet |
| 日切網格 | UTC 日:`day_funding(s, D)` = 當日所有 funding rate 之和(自然處理 8h/4h 異質間隔) |
| 上市風化期 | symbol 首筆 funding 後 **30 天**才進入合格池(避開上市初期異常) |
| 下市處理 | funding 記錄終止 → 強制出場並計成本 |

## 2. 時間窗

| 窗 | 範圍 | 用途 |
|----|------|------|
| train | 2022-07-01 → 2024-06-30 | G1 |
| test | 2024-07-01 → 2026-06-30 | G1–G4 主判定 |
| replication | 2020-07-01 → 2022-06-30 | 僅在主窗全 PASS 時跑,須再過 G1/G2/G3/G4 |

## 3. 策略(Phase 1,參數鎖定)

僅做**正 funding 側**(long spot + short perp;負側需借券做空 spot,借貸成本與可得性不可控,列為 out of scope)。

- **訊號**:`trail3(s, D)` = D-3 到 D-1 三日 funding 和,年化 = ×365/3。需三日數據齊全。
- **進場**:每日 00:00 UTC,合格且 trail3 年化 > **10%** 者,依 trail3 由高到低補進空槽,最多 **K=5** 槽,每槽等權(1/5 notional)。
- **出場**:trail3 年化 < **5%**(遲滯帶避免抖動)或下市。
- **收益**:持有日收 `day_funding(s, D)`(進場日即收當日,選股只用 D-1 前數據,無 look-ahead)。
- **成本**:每次進場 **20bps**、出場 **20bps**(RT 40bps = spot taker 10×2 + perp taker 5×2 + 滑價/spread 額度 10)。
- **資本佔用**:deployed = **1.4×** notional(spot 1.0 + perp 保證金 0.4)。**所有 APR 以 deployed 計**。
- 空槽閒置資金報酬 0%。

**明列的簡化與其風險方向**:basis 損益假設均值零(壓力時 spot-perp 基差擴張的 MTM 痛不建模)——G4 成本加倍部分覆蓋;spot 腿可得性(如 1000X 前綴需換算)不逐一驗證——若 PASS,部署前須過 spot 可得性稽核,稽核失敗視同 FAIL。

## 4. 判定

### Step 0 — 毛利上界 kill-test(先跑,最便宜的死法)

`C_half` = 每日以 trail3 選前 5 名(無門檻、無成本)、收下一日毛 funding 的年化(on notional)。

**KILL 規則:任一 half 的 C < 10% 年化 → 直接 FAIL,不進 Phase 1。**
(G3 要 deployed 5% ⇒ notional 7%,加成本緩衝 ⇒ 毛 10% 是及格底線;上界都不到,細節無意義。)
另報 oracle ceiling(完美後見 top-1/top-5)僅供描述,不是 gate。

### Phase 1 — 四道 gate(Step 0 存活才跑)

| Gate | 條件 |
|------|------|
| G1 OOS | 淨 APR(deployed)> 0,train **且** test |
| G2 懶惰對照 | test 淨 APR ≥ BTC+ETH 50/50 常駐 carry 淨 APR **+ 2pp**(打不贏不用研究的做法就沒價值) |
| G3 絕對門檻 | test 淨 APR ≥ **5%**(低於此,穩定幣/短債風險調整後更好) |
| G4 成本穩健 | RT 成本加倍(80bps)後 test 淨 APR 仍 > 0 |

**PASS = 四道全過 → 跑 replication 窗,四道須再全過才算 REPLICATED。**

### 終局條款(事先承諾)

Step 0 kill、或任一 gate FAIL、或 replication FAIL →「**免費 Binance funding 數據的 cash-and-carry**」問題**永久關閉**:不改參數重跑、不換窗口、不重訪。與 2026-06-29 方向性終局同等效力。

## 5. 產出物

- `scripts/carry/universe.py` — S3 枚舉 + 下載 + parquet
- `scripts/carry/study.py` — 日表構建 + Step 0 + Phase 1 模擬 + gates(常數區塊 = 本文件的機器可讀鏡像)
- `tests/unit/scripts/test_carry_*.py` — 純函數單元測試(合成數據)
- verdict 寫回本文件 + handoff current

## 6. VERDICT(2026-07-05 首跑,參數零改動)

**PASS & REPLICATED**。universe 實際取得 791 個 USDT 永續(含已下市;快照在 `data/carry/universe_snapshot.json`),日表 563,926 rows。

| 判定 | 數字 | 結果 |
|------|------|------|
| Step 0 毛上界(notional) | train **+23.8%** / test **+71.4%**(kill 線 10%;oracle top5 +31.3%/+117.6%) | 存活 |
| G1 OOS(deployed 淨 APR) | train **+7.4%**,test **+23.8%** | PASS |
| G2 懶惰對照 | lazy BTC+ETH **+3.4%**,策略超出 +20.4pp(門檻 +2pp) | PASS |
| G3 絕對門檻 | test +23.8% ≥ 5% | PASS |
| G4 成本加倍 | test **+23.0%**(80bps RT) | PASS |
| Replication 2020-07→2022-06 | halves **+32.9%/+6.7%**,full **+19.8%**(lazy +17.2%,門檻 19.2%,**只贏 0.6pp**),2× 成本 +16.8% | PASS |

成交統計:主窗 4 年 70 進 / 65 出,平均滿倉 5 槽——低換手,成本敏感度低(G4 只掉 0.8pp)。

**描述性診斷(不影響 verdict)**:日淨損益 NW t-stat(lag 30)= **7.30**(main)/ **4.18**(repl)。記帳模型 maxDD 0.55%/0.78%、年化 Sharpe 15/11——**這是平滑性假象**:本研究只記 funding 現金流,basis MTM 波動未建模;真實日損益會被 spot-perp 基差波動放大數倍,壓力期(如連環清算)單日 -1%~-3% 屬常態,保證金壓力才是實際主風險。**APR 期望值可信;Sharpe/DD 數字不是真實風險輪廓。**

誠實註記:(a) test 窗 +23.8% 是 2024-2026 牛市 regime 的產物,train 窗 +7.4% 才是乏味 regime 的合理預期;(b) replication 的 G2 只贏 0.6pp——2021 牛市 majors funding 本身就肥,輪動相對 lazy 的增量在那種 regime 很薄;(c) 未建模:交易所對手風險、下市結算細節、alt spot 深度。

### Phase 2(事先登記的部署前 gate,尚未跑)

**Spot 可得性稽核**,操作化定義(在跑稽核前於此鎖定):取主窗+複製窗全部實際持倉 symbol 清單,逐一核對 Binance 是否存在可對沖的 USDT spot 對(1000x 前綴按單位換算視為可對沖);把**無 spot 對沖**的 symbol 從 universe 剔除後重跑 Phase 1 + replication,**全部 gate 須仍過**,否則整體 verdict 降級 FAIL。此定義一經寫下不再放寬。
