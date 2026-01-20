import os
import pandas as pd
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD
from datetime import datetime

def get_market_analysis(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_15MINUTE):
    """
    Fetches data and prints a summary for AI analysis.
    """
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = Client(api_key, api_secret)
    
    # Fetch slightly more data to ensure stable indicators
    klines = client.get_historical_klines(symbol, interval, "3 days ago UTC")
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Indicators
    rsi_indicator = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi_indicator.rsi()
    
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # Get last 5 candles for context
    recent = df.tail(5)
    
    print(f"\n--- DATA FOR {symbol} ({interval}) ---")
    print(f"Current Price: {df['close'].iloc[-1]:.2f}")
    print(f"RSI (14): {df['rsi'].iloc[-1]:.2f}")
    print(f"MACD Hist: {df['macd_hist'].iloc[-1]:.4f}")
    print(f"MACD Line: {df['macd'].iloc[-1]:.4f}")
    
    # Fibonacci Context (High/Low of last 3 days)
    high_3d = df['high'].max()
    low_3d = df['low'].min()
    print(f"3-Day High: {high_3d:.2f}")
    print(f"3-Day Low: {low_3d:.2f}")
    
    print("\nRecent 5 Candles (Open, High, Low, Close, RSI, MACD_Hist):")
    for index, row in recent.iterrows():
        print(f"{index}: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}, RSI={row['rsi']:.2f}, MACD_Hist={row['macd_hist']:.4f}")
    print("------------------------------------------\n")

if __name__ == "__main__":
    print("fetching market data...")
    get_market_analysis("BTCUSDT")
    get_market_analysis("ETHUSDT")
