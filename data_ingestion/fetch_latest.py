"""
Multi-Timeframe Market Data Fetcher

Fetches data across multiple timeframes (4H, 1H, 15m) to support 
the crypto trading skill's top-down analysis approach.

- 4H: HTF Bias confirmation (Bullish/Bearish/Neutral)
- 1H: Trend confirmation + BOS/CHoCH identification
- 15m: Execution timeframe + POI identification
- Asia Session: ASH/ASL for AMD model analysis
- Funding Rate: Risk filter for position sizing
"""

import os
import sys
import pandas as pd
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from strategy.funding_rate import (
        fetch_funding_rate,
        evaluate_funding_rate,
        format_funding_rate_filter
    )
    FUNDING_RATE_AVAILABLE = True
except ImportError:
    FUNDING_RATE_AVAILABLE = False

# Timeframe configurations
TIMEFRAMES = {
    "4h": {
        "interval": Client.KLINE_INTERVAL_4HOUR,
        "lookback": "7 days ago UTC",  # ~42 candles
        "purpose": "HTF Bias"
    },
    "1h": {
        "interval": Client.KLINE_INTERVAL_1HOUR,
        "lookback": "3 days ago UTC",  # ~72 candles
        "purpose": "Trend Confirmation"
    },
    "15m": {
        "interval": Client.KLINE_INTERVAL_15MINUTE,
        "lookback": "2 days ago UTC",  # ~192 candles
        "purpose": "Execution"
    }
}


def get_binance_client():
    """Initialize Binance client (public API if no keys provided)."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if api_key and api_secret:
        return Client(api_key, api_secret)
    return Client()


def fetch_timeframe_data(client, symbol, interval, lookback):
    """
    Fetches OHLCV data and calculates RSI/MACD indicators.
    Returns a DataFrame with indicators.
    """
    try:
        klines = client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)

        # Calculate RSI (14)
        rsi_indicator = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi_indicator.rsi()

        # Calculate MACD (12, 26, 9)
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        return df
    except Exception as e:
        print(f"Error fetching {interval} data for {symbol}: {e}")
        return None


def fetch_multi_timeframe_data(symbol="BTCUSDT"):
    """
    Fetches data for all configured timeframes.
    Returns a dictionary of DataFrames keyed by timeframe.
    """
    client = get_binance_client()
    data = {}

    for tf_name, tf_config in TIMEFRAMES.items():
        df = fetch_timeframe_data(
            client,
            symbol,
            tf_config["interval"],
            tf_config["lookback"]
        )
        if df is not None:
            data[tf_name] = df

    return data


def determine_htf_bias(df_4h, df_1h):
    """
    Determines the Higher Timeframe (HTF) bias based on 4H and 1H data.
    
    Criteria:
    - Bullish: Price making higher highs/higher lows, RSI > 50, MACD > 0
    - Bearish: Price making lower highs/lower lows, RSI < 50, MACD < 0
    - Neutral: Mixed signals or ranging
    """
    if df_4h is None or df_1h is None:
        return "UNKNOWN", {}

    # Get recent candles for structure analysis
    recent_4h = df_4h.tail(10)
    recent_1h = df_1h.tail(24)

    # 4H Analysis
    rsi_4h = df_4h['rsi'].iloc[-1]
    macd_4h = df_4h['macd'].iloc[-1]
    macd_hist_4h = df_4h['macd_hist'].iloc[-1]
    
    # Check for higher highs / higher lows (bullish structure)
    highs_4h = recent_4h['high'].values
    lows_4h = recent_4h['low'].values
    
    # Simple structure check: compare first half vs second half
    mid = len(highs_4h) // 2
    higher_highs = highs_4h[mid:].max() > highs_4h[:mid].max()
    higher_lows = lows_4h[mid:].min() > lows_4h[:mid].min()
    lower_highs = highs_4h[mid:].max() < highs_4h[:mid].max()
    lower_lows = lows_4h[mid:].min() < lows_4h[:mid].min()

    # 1H Analysis
    rsi_1h = df_1h['rsi'].iloc[-1]
    macd_1h = df_1h['macd'].iloc[-1]
    macd_hist_1h = df_1h['macd_hist'].iloc[-1]

    # Determine bias
    bullish_score = 0
    bearish_score = 0

    # Structure scoring
    if higher_highs and higher_lows:
        bullish_score += 2
    elif lower_highs and lower_lows:
        bearish_score += 2
    
    # RSI scoring
    if rsi_4h > 55:
        bullish_score += 1
    elif rsi_4h < 45:
        bearish_score += 1
    
    if rsi_1h > 55:
        bullish_score += 1
    elif rsi_1h < 45:
        bearish_score += 1

    # MACD scoring
    if macd_4h > 0 and macd_hist_4h > 0:
        bullish_score += 1
    elif macd_4h < 0 and macd_hist_4h < 0:
        bearish_score += 1

    if macd_1h > 0 and macd_hist_1h > 0:
        bullish_score += 1
    elif macd_1h < 0 and macd_hist_1h < 0:
        bearish_score += 1

    # Determine final bias
    if bullish_score >= 4 and bullish_score > bearish_score + 2:
        bias = "BULLISH"
    elif bearish_score >= 4 and bearish_score > bullish_score + 2:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    details = {
        "4h_rsi": rsi_4h,
        "4h_macd": macd_4h,
        "4h_macd_hist": macd_hist_4h,
        "4h_structure": "HH/HL" if (higher_highs and higher_lows) else ("LH/LL" if (lower_highs and lower_lows) else "RANGE"),
        "1h_rsi": rsi_1h,
        "1h_macd": macd_1h,
        "1h_macd_hist": macd_hist_1h,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score
    }

    return bias, details


def calculate_asia_range(df_15m):
    """
    Calculates the Asia Session High (ASH) and Asia Session Low (ASL).
    
    Asia Session: 08:00 - 15:00 UTC+8 (00:00 - 07:00 UTC)
    """
    if df_15m is None or df_15m.empty:
        return None, None, None, None

    # Convert to UTC+8 timezone
    taipei_tz = ZoneInfo("Asia/Taipei")
    df = df_15m.copy()
    df.index = df.index.tz_localize('UTC').tz_convert(taipei_tz)

    # Get today's and yesterday's date in Taipei timezone
    now_taipei = datetime.now(taipei_tz)
    
    # Determine which Asia session to use
    # If current time is before 15:00, use today's Asia session (if available) or yesterday's
    # If current time is after 15:00, use today's Asia session
    if now_taipei.hour < 15:
        # Use yesterday's completed Asia session
        target_date = now_taipei.date() - timedelta(days=1)
    else:
        # Use today's Asia session
        target_date = now_taipei.date()

    asia_start = datetime(target_date.year, target_date.month, target_date.day, 8, 0, tzinfo=taipei_tz)
    asia_end = datetime(target_date.year, target_date.month, target_date.day, 15, 0, tzinfo=taipei_tz)

    # Filter for Asia session
    asia_data = df[(df.index >= asia_start) & (df.index < asia_end)]

    if asia_data.empty:
        # Try previous day if no data
        target_date = target_date - timedelta(days=1)
        asia_start = datetime(target_date.year, target_date.month, target_date.day, 8, 0, tzinfo=taipei_tz)
        asia_end = datetime(target_date.year, target_date.month, target_date.day, 15, 0, tzinfo=taipei_tz)
        asia_data = df[(df.index >= asia_start) & (df.index < asia_end)]

    if asia_data.empty:
        return None, None, None, None

    ash = asia_data['high'].max()  # Asia Session High
    asl = asia_data['low'].min()   # Asia Session Low
    asia_date = target_date

    return ash, asl, asia_date, asia_data


def get_current_session(now=None):
    """
    Determines the current trading session based on UTC+8 time.
    
    Sessions:
    - Asia: 08:00 - 15:00 UTC+8
    - London: 15:00 - 00:00 UTC+8
    - New York: 21:00 - 06:00 UTC+8
    - London/NY Overlap: 21:00 - 00:00 UTC+8
    """
    taipei_tz = ZoneInfo("Asia/Taipei")
    if now is None:
        now = datetime.now(taipei_tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=taipei_tz)
    
    hour = now.hour

    if 8 <= hour < 15:
        return "ASIA", "Accumulation - 低波動，觀察區間形成"
    elif 15 <= hour < 21:
        return "LONDON", "Manipulation - 高波動，等待流動性掃蕩"
    elif 21 <= hour < 24:
        return "LONDON_NY_OVERLAP", "最高波動 - 最佳交易窗口"
    elif 0 <= hour < 6:
        return "NEW_YORK", "Distribution - 趨勢延續"
    else:
        return "LOW_VOLUME", "低流動性時段"


def print_comprehensive_summary(symbol, data):
    """
    Prints a comprehensive multi-timeframe analysis summary.
    """
    if not data:
        print(f"No data available for {symbol}")
        return

    df_4h = data.get("4h")
    df_1h = data.get("1h")
    df_15m = data.get("15m")

    print(f"\n{'='*60}")
    print(f"📊 MULTI-TIMEFRAME ANALYSIS: {symbol}")
    print(f"{'='*60}")

    # Current Session
    session, session_desc = get_current_session()
    print(f"\n🕐 Current Session: {session}")
    print(f"   {session_desc}")

    # HTF Bias
    bias, bias_details = determine_htf_bias(df_4h, df_1h)
    bias_emoji = "🟢" if bias == "BULLISH" else ("🔴" if bias == "BEARISH" else "⚪")
    print(f"\n📈 HTF BIAS: {bias_emoji} {bias}")
    print(f"   Structure (4H): {bias_details.get('4h_structure', 'N/A')}")
    print(f"   Bullish Score: {bias_details.get('bullish_score', 0)}/6")
    print(f"   Bearish Score: {bias_details.get('bearish_score', 0)}/6")

    # 4H Timeframe
    if df_4h is not None:
        print(f"\n⏰ 4H TIMEFRAME (HTF Bias):")
        print(f"   Price:     ${df_4h['close'].iloc[-1]:,.2f}")
        print(f"   RSI (14):  {df_4h['rsi'].iloc[-1]:.2f}")
        print(f"   MACD:      {df_4h['macd'].iloc[-1]:.4f}")
        print(f"   MACD Hist: {df_4h['macd_hist'].iloc[-1]:.4f}")
        print(f"   7D High:   ${df_4h['high'].max():,.2f}")
        print(f"   7D Low:    ${df_4h['low'].min():,.2f}")

    # 1H Timeframe
    if df_1h is not None:
        print(f"\n⏰ 1H TIMEFRAME (Trend Confirmation):")
        print(f"   Price:     ${df_1h['close'].iloc[-1]:,.2f}")
        print(f"   RSI (14):  {df_1h['rsi'].iloc[-1]:.2f}")
        print(f"   MACD:      {df_1h['macd'].iloc[-1]:.4f}")
        print(f"   MACD Hist: {df_1h['macd_hist'].iloc[-1]:.4f}")

    # 15M Timeframe
    if df_15m is not None:
        print(f"\n⏰ 15M TIMEFRAME (Execution):")
        print(f"   Price:     ${df_15m['close'].iloc[-1]:,.2f}")
        print(f"   RSI (14):  {df_15m['rsi'].iloc[-1]:.2f}")
        print(f"   MACD:      {df_15m['macd'].iloc[-1]:.4f}")
        print(f"   MACD Hist: {df_15m['macd_hist'].iloc[-1]:.4f}")

        # Recent 5 candles
        print(f"\n   Recent 5 Candles (15m):")
        recent = df_15m.tail(5)
        for idx, row in recent.iterrows():
            print(f"   {idx}: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}, RSI={row['rsi']:.2f}")

    # Asia Range (AMD Model)
    ash, asl, asia_date, _ = calculate_asia_range(df_15m)
    if ash is not None and asl is not None:
        current_price = df_15m['close'].iloc[-1] if df_15m is not None else 0
        print(f"\n📍 ASIA SESSION RANGE (AMD Model):")
        print(f"   Date: {asia_date}")
        print(f"   ASH (Asia Session High): ${ash:,.2f}")
        print(f"   ASL (Asia Session Low):  ${asl:,.2f}")
        print(f"   Range Size: ${(ash - asl):,.2f} ({((ash - asl) / asl * 100):.2f}%)")
        
        # Position relative to Asia Range
        if current_price > ash:
            position = f"Above ASH (+${(current_price - ash):,.2f})"
        elif current_price < asl:
            position = f"Below ASL (-${(asl - current_price):,.2f})"
        else:
            position = f"Inside Range"
        print(f"   Current Position: {position}")

    # Quick Summary
    print(f"\n{'='*60}")
    print(f"📋 QUICK SUMMARY")
    print(f"{'='*60}")
    
    if df_15m is not None:
        current_price = df_15m['close'].iloc[-1]
        rsi_15m = df_15m['rsi'].iloc[-1]
        
        # RSI Status
        if rsi_15m < 30:
            rsi_status = "🟢 OVERSOLD (Potential Long)"
        elif rsi_15m > 70:
            rsi_status = "🔴 OVERBOUGHT (Potential Short)"
        elif rsi_15m < 45:
            rsi_status = "⚪ Neutral-Bearish"
        elif rsi_15m > 55:
            rsi_status = "⚪ Neutral-Bullish"
        else:
            rsi_status = "⚪ Neutral"
        
        print(f"   Current Price: ${current_price:,.2f}")
        print(f"   HTF Bias: {bias_emoji} {bias}")
        print(f"   RSI (15m): {rsi_15m:.2f} - {rsi_status}")
        print(f"   Session: {session}")
        
        # Actionable recommendation
        print(f"\n💡 RECOMMENDATION:")
        if bias == "BULLISH" and rsi_15m < 40:
            print("   Consider LONG entries at support/demand zones")
        elif bias == "BEARISH" and rsi_15m > 60:
            print("   Consider SHORT entries at resistance/supply zones")
        elif session == "ASIA":
            print("   WAIT - Mark Asia Range (ASH/ASL), don't trade during accumulation")
        elif session in ["LONDON", "LONDON_NY_OVERLAP"]:
            print("   ACTIVE - Watch for liquidity sweeps at ASH/ASL")
        else:
            print("   Monitor for confluence setups")

    # Funding Rate Filter
    if FUNDING_RATE_AVAILABLE:
        try:
            funding_data = fetch_funding_rate(symbol)
            if funding_data.get('available'):
                evaluation = evaluate_funding_rate(funding_data['rate'])
                print(format_funding_rate_filter(funding_data, evaluation))
            else:
                print("\n🚦 RISK FILTER: Funding Rate")
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print("   Status: ❓ N/A (Futures data unavailable)")
        except Exception as e:
            print(f"\n🚦 Funding Rate: Error fetching data - {e}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print("Fetching multi-timeframe market data...")
    print(f"Time: {datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")

    # Fetch and analyze BTC
    btc_data = fetch_multi_timeframe_data("BTCUSDT")
    print_comprehensive_summary("BTCUSDT", btc_data)

    # Fetch and analyze ETH
    eth_data = fetch_multi_timeframe_data("ETHUSDT")
    print_comprehensive_summary("ETHUSDT", eth_data)
