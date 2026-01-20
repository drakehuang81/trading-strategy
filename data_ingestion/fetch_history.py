import os
import pandas as pd
from binance.client import Client
from datetime import datetime

def fetch_data(symbol="ETHUSDT", interval=Client.KLINE_INTERVAL_15MINUTE, start_str="1 week ago UTC"):
    """
    Fetches historical kline data from Binance and saves it to a CSV file.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTCUSDT', 'ETHUSDT').
        interval (str): Candle interval (e.g., Client.KLINE_INTERVAL_15MINUTE).
        start_str (str): Start date string (e.g., '1 Jan, 2024', '1 week ago UTC').
    """
    print(f"Fetching {symbol} data ({interval}) starting from {start_str}...")
    
    # Initialize Client (API keys are optional for public data, but recommended for higher limits)
    # You can set BINANCE_API_KEY and BINANCE_API_SECRET in your environment variables or .env file
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    client = Client(api_key, api_secret)

    # Fetch data
    klines = client.get_historical_klines(symbol, interval, start_str)
    
    print(f"Fetched {len(klines)} candles.")

    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Data Cleaning and Formatting
    # Convert timestamp to readable date
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Convert numeric columns to float (they come as strings)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Keep only relevant columns for analysis
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Save to CSV
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    filename = f"data/{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename)
    
    print(f"Data saved to {filename}")
    print("\nRecent 5 candles:")
    print(df.tail())

if __name__ == "__main__":
    # You can change these parameters
    target_symbol = "ETHUSDT"
    target_interval = Client.KLINE_INTERVAL_15MINUTE # 15m for our scalping strategy
    start_time = "1 month ago UTC" # Get 1 month of data
    
    fetch_data(target_symbol, target_interval, start_time)
