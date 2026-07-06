# 事先登記:Binance spot-perp basis 均值回歸(5E P2 清單最後一個未測機制)

**日期**:2026-07-06
**狀態**:PRE-REGISTERED(本文件 commit 後才允許下載 kline 數據)
**合法性**:非方向性(做 spread 收斂,不預測價格)、非 funding carry 家族(賺的是基差收斂不是 funding 現金流)、不在 2026-06-29 方向性終局與 2026-07-05/06 carry 雙終局的關閉範圍內。5E STATUS P2 清單列名("Mean-reversion on basis spreads")但從未被測。測完它,免費 Binance 數據上的全部 P2 機制家族即**掃描完畢**。

**先驗誠實聲明**:登記人先驗認為 Step 0 大概率(~80%)殺掉本問題——流動性 universe 的 basis 被套利者壓得極緊,4 條 taker 腿 ~40bps 成本難以跨越。仍值一發子彈的理由:(a) 它是最後一個未測的合法機制;(b) 2022/2024 的清算級聯期 basis 確實出現過 >100bps 的episodes,量級是否足夠是實證問題;(c) 否定答案完成整個免費數據 program 的完備性。

**Schema 已探明(僅此,無其他數據接觸)**:spot 月度 1h kline **無 header**;um-futures 月度 1h kline **有 header**;皆 12 欄同佈局。

## 1. Universe(規則先於名單)

今日 um-perp 24h quoteVolume 前 20 名中,spot 與 um 月度 1h 檔案皆覆蓋主窗 ≥80% 月份者。**現在式選樣揭露**:方向偏保守——今日流動的名字歷史上 basis 更緊、episodes 更少,偏向 kill 而非偏向 pass;流動性篩選是可執行性要求(稀薄 alt 的 1h close「basis」摻雜 stale-print 假象,不可交易)。快照存 `data/basis/universe_snapshot.json`。

## 2. 窗口(與 carry 研究一致,便於比較)

train 2022-07-01→2024-06-30;test 2024-07-01→2026-06-30;replication 2020-07-01→2022-06-30(僅主窗全 PASS 時跑,全 gate 須再過)。

## 3. 機制與參數(鎖定)

- `basis_h` =(perp_close − spot_close)/ spot_close,同 open_time 對齊,1h 網格。
- **Episode(每 symbol 非重疊)**:|basis| ≥ **60bps** 觸發進場;|basis| ≤ **10bps** 收斂出場,或 **48h** 強制出場;capture = |basis_entry| − |basis_exit|。
- **只做正 basis 側**(perp 溢價:long spot + short perp,無需借券)——與 carry 同樣的 scope 理由;負側只做描述性報告。
- 持有期間 funding 流**不計入**(正 basis 期 short perp 通常收 funding,排除它是保守方向)。
- 成本:**40bps RT**/episode(spot taker 10×2 + perp taker 5×2 + 滑價 10);G4 加倍 80bps。
- 資本:deployed = 1.4× notional;APR 以 deployed 計。每 symbol 1 槽,組合 = 全部 universe 等權。

## 4. 判定

**Step 0 kill**:正 basis 側、無成本毛 capture,取 per-symbol 年化後**前 5 名平均**(deployed 計)< **10%** APR → FAIL,不進 Phase 1。

**Phase 1 gates**:
- G1 OOS:net APR(deployed)> 0,train 且 test。
- G2(質量 gate,取代無意義的 lazy 對照):train 與 test **各 ≥30 個** episodes(不足 = 統計上不可依賴,FAIL)。
- G3:test net APR ≥ **5%**。
- G4:2× 成本後 test net > 0。

**終局條款**:任一關 FAIL → basis MR 於免費 Binance 數據**永久關閉**;屆時免費 Binance 數據的機制空間(方向、carry、basis MR)全數掃畢,唯餘 qi maker(時間 gate)。全 PASS → 複製窗全 gate 再驗,通過才算 REPLICATED;產出是歷史期望值,非部署授權。

## 5. 產出物

`scripts/basis/study.py`(下載 + 對齊 + Step 0 + Phase 1)、`tests/unit/scripts/test_basis_study.py`、verdict 回寫本文件 + handoff。
