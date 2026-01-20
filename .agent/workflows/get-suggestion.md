---
description: Get crypto trading suggestions based on current market data and skill strategy
---

1. Fetch the latest market data for BTC and ETH.
   Run the python script to get formatted data:
   `source venv/bin/activate && python get_latest_market_data.py`

2. Read the Crypto Trading Skill.
   Read the content of `.gemini/skills/crypto_trading_skill/SKILL.md` to understand the strategy rules.

3. Generate Analysis Report.
   Based on the output from Step 1 and the rules from Step 2, generate a detailed trading analysis report.
   
   The analysis MUST include:
   - **Market Context**: Current price, trend, and session status (AMD model).
   - **Indicator Analysis**: Status of RSI and MACD.
   - **Confidence Score**: Calculate the score (0-8) based on the checklist in the Skill.
   - **Trade Setup**: Suggest a LONG or SHORT setup (or WAIT) with specific Entry, TP, and SL prices if a setup exists.
   - **Risk Warning**: Remind about risk management.
   
   Format the output as a clean Markdown report.
