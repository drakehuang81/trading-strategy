# Project Context: Quant Trading Project

## Project Overview
This project is a quantitative trading system designed for automated trading and backtesting across markets (Crypto/Stocks/Futures). It aims to provide modules for data ingestion, strategy development, backtesting, and live execution with risk management.

## Technology Stack (Planned)
Based on the initialization plans and setup instructions:
*   **Language:** Python
*   **Key Libraries (Potential):** Pandas, ccxt (exchange connectivity), Backtrader (backtesting logic).
*   **Environment:** Virtual environment (`venv`) recommended.

## Project Structure
*Note: The project is currently in the initialization phase. The following structure is planned:*

*   `data/`: Storage for raw and processed market data.
*   `strategies/`: Implementation of trading strategy logic.
*   `backtest/`: Backtesting engine and performance analysis tools (Sharpe Ratio, Max Drawdown).
*   `execution/`: Live trading execution and exchange API connectors.
*   `notebooks/`: Jupyter Notebooks for research and data analysis.

## Setup & Usage
1.  **Clone & Enter:**
    ```bash
    git clone https://github.com/drakehuang81/quant-trading-project.git
    cd quant-trading-project
    ```
2.  **Environment Setup:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    # pip install -r requirements.txt (File to be created)
    ```

## Development Status
*   **Current State:** Project Initialization.
*   **Immediate Goals:**
    *   Initialize project structure.
    *   Finalize technology choices (Language/Frameworks).
    *   Connect to sample data sources.
