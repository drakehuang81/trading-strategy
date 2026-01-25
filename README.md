# Quant Trading Project

A quantitative trading system development project.

## Overview
This project aims to build an automated trading and backtesting system for **cryptocurrency**, integrating **Smart Money Concepts (SMC)** strategy and technical indicator analysis.

## Features
- **Data Ingestion**: Fetch historical and real-time K-line data from Binance API using `python-binance`.
- **Web Visualization Dashboard**: Interactive trading interface built with Streamlit, supporting real-time K-line charts, technical indicators, and trading signals.
- **Backtesting**: Simulate historical data to validate strategies and generate performance reports (Sharpe Ratio, Max Drawdown).
- **Live Trading**: Connect to exchange APIs for automated order execution.
- **Risk Management**: Configure rules for take-profit, stop-loss, and position sizing.

---

## Quick Start

### 1. Environment Setup

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

### 2. Fetch Binance Data

```bash
# Run data fetching script (Default: ETH/USDT 15m K-lines, past 30 days)
python data_ingestion/fetch_history.py
```

Data will be saved to the `data/` directory in CSV format.

---

## 🌐 Web Visualization Dashboard

This project provides an interactive Web Dashboard based on **Streamlit**, allowing you to monitor cryptocurrency price trends and trading signals in real-time.

### Launch Dashboard

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Start Streamlit application
streamlit run archive/app/app.py
```

After launching, open in your browser:
- **Local URL**: http://localhost:8501
- **Network URL**: http://[YOUR_IP]:8501 (Accessible via mobile on the same network)

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **K-Line Chart** | Interactive Plotly candlestick chart, supports zooming and panning |
| **RSI Indicator** | 14-period Relative Strength Index, marking overbought/oversold zones |
| **MACD Indicator** | Momentum indicator showing histogram and signal lines |
| **Fibonacci Retracement** | Automatically calculates and draws Fib 0.382, 0.5, 0.618, 0.786 |
| **AMD Trading Sessions** | Marks Asia session range (ASH/ASL) to assist in Sweep identification |
| **Trading Signal Alerts** | Displays Long/Short suggestions when RSI/MACD confluence is detected |
| **Confidence Score** | 1-6 star rating based on multi-factor confluence |

### Dashboard Preview

```
┌────────────────────────────────────────────────────────┐
│  🦅 Crypto Scalping Dashboard: ETHUSDT                  │
├────────────────────────────────────────────────────────┤
│  Current Price: $3,110.62   RSI: 44.08   MACD: 3.68   │
│                                    [London Manipulation]│
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │         K-Line Chart (w/ Fib + AMD Range)       │  │
│  │    ════ Fib 0.618 ════                          │  │
│  │    ▓▓▓▓ Asia Session Range ▓▓▓▓                 │  │
│  │                                                 │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  ┌───────────────┐  ┌───────────────────────────┐     │
│  │   RSI (14)    │  │     MACD                  │     │
│  │   ════ 70 ════│  │    ▃▃▃ Histogram         │     │
│  │   ════ 30 ════│  │    ─── Signal Line       │     │
│  └───────────────┘  └───────────────────────────┘     │
│                                                        │
│  🚨 Trading Signals & Alerts                           │
│  ⚪ No clear signal. Market is neutral.                 │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
quant-trading-project/
├── archive/
│   └── app/                    # Streamlit Dashboard (Deprecated)
├── data_ingestion/             # Data Fetching Scripts
│   ├── fetch_history.py        # Fetch historical klines to CSV
│   └── fetch_latest.py         # Fetch latest data for AI analysis
├── requirements.txt            # Python Dependencies
├── data/                       # CSV Data Files Directory
│   └── ETHUSDT_15m_*.csv
├── .gemini/
│   └── skills/
│       └── crypto_trading_skill/
│           └── SKILL.md        # AI Trading Strategy Skill Definition
├── GEMINI.md                   # Project Context
└── README.md                   # This Documentation
```

---

## Tech Stack

| Category | Tool |
|----------|------|
| **Language** | Python 3.9+ |
| **Data Ingestion** | python-binance |
| **Data Processing** | Pandas, NumPy |
| **Technical Indicators** | ta (Technical Analysis Library) |
| **Web Framework** | Streamlit |
| **Charting** | Plotly |
| **Environment Management** | venv, pip |

---

## To-Do

- [x] Project Initialization
- [x] Connect Binance API for Data Fetching
- [x] Build Web Visualization Dashboard
- [x] Integrate RSI, MACD, Fibonacci, AMD Indicators
- [x] Create AI Trading Strategy Skill (crypto_trading_skill)
- [ ] Add Backtesting Functionality
- [ ] Integration with Telegram Notifications
- [ ] Deploy to Streamlit Cloud

---

## License

This project is for educational and research purposes only and does not constitute investment advice. Please manage your risk accordingly.
