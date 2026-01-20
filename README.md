# Quant Trading Project

一個量化交易系統的開發專案。

## 專案簡介 (Overview)
本專案旨在建構一個用於 [加密貨幣/股票/期貨] 的自動化交易與回測系統。

## 功能規劃 (Features)
- **數據獲取 (Data Ingestion)**: 支援從各大交易所 API 獲取歷史與即時 K 線數據。
- **策略回測 (Backtesting)**: 模擬歷史數據進行策略驗證，產出績效報告（Sharpe Ratio, Max Drawdown）。
- **實盤交易 (Live Trading)**: 連接交易所 API 進行自動化下單。
- **風險控管 (Risk Management)**: 設定止盈止損與倉位管理規則。

## 環境設定 (Setup)

```bash
# Clone the repository
git clone https://github.com/drakehuang81/quant-trading-project.git

# Enter directory
cd quant-trading-project

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
# pip install -r requirements.txt
```

## 專案結構 (Directory Structure)
```
/
├── data/           # 存放原始與處理後的數據
├── strategies/     # 交易策略邏輯
├── backtest/       # 回測引擎與績效分析
├── execution/      # 實盤下單與交易所串接
└── notebooks/      # 研究與數據分析 (Jupyter Notebooks)
```

## 待辦事項 (To-Do)
- [ ] 專案初始化
- [ ] 決定使用的語言與框架 (e.g., Python, Pandas, ccxt, Backtrader)
- [ ] 連接範例數據源
