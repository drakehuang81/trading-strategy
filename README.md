# Quant Trading Project

一個量化交易系統的開發專案。

## 專案簡介 (Overview)
本專案旨在建構一個用於 **加密貨幣** 的自動化交易與回測系統，整合 **Smart Money Concepts (SMC)** 策略與技術指標分析。

## 功能規劃 (Features)
- **數據獲取 (Data Ingestion)**: 透過 `python-binance` 從幣安 API 獲取歷史與即時 K 線數據。
- **Web 視覺化儀表板**: 使用 Streamlit 建立互動式看盤介面，支援即時 K 線圖、技術指標與交易訊號。
- **策略回測 (Backtesting)**: 模擬歷史數據進行策略驗證，產出績效報告（Sharpe Ratio, Max Drawdown）。
- **實盤交易 (Live Trading)**: 連接交易所 API 進行自動化下單。
- **風險控管 (Risk Management)**: 設定止盈止損與倉位管理規則。

---

## 快速開始 (Quick Start)

### 1. 環境設定

```bash
# Clone the repository
git clone https://github.com/drakehuang81/quant-trading-project.git

# Enter directory
cd quant-trading-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. 抓取幣安數據

```bash
# 執行數據抓取腳本 (預設: ETH/USDT 15分鐘K線，過去一個月)
python fetch_binance_data.py
```

數據會儲存到 `data/` 資料夾中，格式為 CSV。

---

## 🌐 Web 視覺化儀表板

本專案提供一個基於 **Streamlit** 的互動式 Web 儀表板，讓你可以即時監控加密貨幣的價格走勢與交易訊號。

### 啟動儀表板

```bash
# 確保虛擬環境已啟動
source venv/bin/activate

# 啟動 Streamlit 應用
streamlit run app.py
```

啟動後，在瀏覽器中打開：
- **本地網址**: http://localhost:8501
- **區域網路**: http://[你的IP]:8501 (可用手機連線)

### 儀表板功能

| 功能 | 說明 |
|------|------|
| **K 線圖** | 互動式 Plotly 蠟燭圖，支援縮放與平移 |
| **RSI 指標** | 14 週期相對強弱指數，標示超買/超賣區 |
| **MACD 指標** | 動能指標，顯示柱狀圖與訊號線 |
| **斐波那契回撤** | 自動計算並繪製 Fib 0.382, 0.5, 0.618, 0.786 |
| **AMD 交易時段** | 標示亞洲盤區間 (ASH/ASL)，輔助 Sweep 判斷 |
| **交易訊號警報** | 當 RSI/MACD 出現共振時，顯示做多/做空建議 |
| **信心分數 (Confidence Score)** | 基於多因子共振，給出 1-6 星評級 |

### 畫面預覽

```
┌────────────────────────────────────────────────────────┐
│  🦅 加密貨幣剝頭皮儀表板: ETHUSDT                       │
├────────────────────────────────────────────────────────┤
│  當前價格: $3,110.62   RSI: 44.08   MACD: 3.68        │
│                                    [倫敦盤 操縱階段]   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │         K 線圖 (含 Fib + AMD 區間)               │  │
│  │    ════ Fib 0.618 ════                          │  │
│  │    ▓▓▓▓ 亞洲盤區間 ▓▓▓▓                         │  │
│  │                                                 │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  ┌───────────────┐  ┌───────────────────────────┐     │
│  │   RSI (14)    │  │     MACD                  │     │
│  │   ════ 70 ════│  │    ▃▃▃ Histogram         │     │
│  │   ════ 30 ════│  │    ─── Signal Line       │     │
│  └───────────────┘  └───────────────────────────┘     │
│                                                        │
│  🚨 交易訊號與警報                                     │
│  ⚪ 目前無明確訊號。市場處於中性區間。                  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 專案結構 (Directory Structure)

```
quant-trading-project/
├── app.py                      # Streamlit Web 儀表板主程式
├── fetch_binance_data.py       # 幣安數據抓取腳本
├── requirements.txt            # Python 依賴套件
├── data/                       # 存放 CSV 數據檔
│   └── ETHUSDT_15m_*.csv
├── .gemini/
│   └── skills/
│       └── crypto_trading_skill/
│           └── SKILL.md        # AI 交易策略 Skill 定義
├── GEMINI.md                   # 專案上下文
└── README.md                   # 本說明文件
```

---

## 技術棧 (Tech Stack)

| 類別 | 工具 |
|------|------|
| **語言** | Python 3.9+ |
| **數據獲取** | python-binance |
| **數據處理** | Pandas, NumPy |
| **技術指標** | ta (Technical Analysis Library) |
| **Web 框架** | Streamlit |
| **圖表** | Plotly |
| **環境管理** | venv, pip |

---

## 待辦事項 (To-Do)

- [x] 專案初始化
- [x] 連接幣安 API 抓取數據
- [x] 建立 Web 視覺化儀表板
- [x] 整合 RSI, MACD, Fibonacci, AMD 指標
- [x] 建立 AI 交易策略 Skill (crypto_trading_skill)
- [ ] 加入回測功能
- [ ] 串接 Telegram 推播通知
- [ ] 部署到 Streamlit Cloud

---

## 授權 (License)

本專案僅供教育與研究用途，不構成任何投資建議。請務必做好風險管理。
