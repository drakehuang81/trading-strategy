# Handoff 交接資料夾

Session 之間(或人之間)的工作交接。規則:

- **`current/`** — 進行中的交接文件。每份必須**自足**:不看對話紀錄也能接手——含目標、已完成、關鍵事實/坑、未決事項、快速接手指令、文件地圖。
- **`archive/`** — 該工作流(workstream)**結束後**,把 current 的文件原封移入,留作歷史紀錄。移入時不改內容,只搬位置。

命名:`YYYY-MM-DD-<topic>.md`(日期 = 建立日;內容更新不改檔名)。

一份 current 文件對應一個 workstream。新 session 接手時:先讀 `current/` 全部文件,再動手。
