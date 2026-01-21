import time
import os
import pandas as pd
from datetime import datetime
from data_ingestion.fetch_latest import fetch_market_data

# --- Strategy Logic (Ported from Dashboard) ---

def calculate_fibonacci(df):
    """根据近期高低点计算斐波那契回撤位"""
    swing_high = df['high'].max()
    swing_low = df['low'].min()
    
    diff = swing_high - swing_low
    
    levels = {
        '0.5': swing_high - 0.5 * diff,
    }
    
    return levels, swing_high, swing_low

def get_asia_session_range(df):
    """取得亞洲盤的高低點 (UTC 00:00 - 07:00)"""
    try:
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            return None, None
            
        asia_mask = (df_copy.index.hour >= 0) & (df_copy.index.hour < 7)
        asia_data = df_copy[asia_mask]
        
        if len(asia_data) >= 4:
            asia_data_sorted = asia_data.sort_index()
            recent_date = asia_data_sorted.index[-1].date()
            today_asia = asia_data_sorted[asia_data_sorted.index.date == recent_date]
            
            if len(today_asia) > 0:
                asia_high = today_asia['high'].max()
                asia_low = today_asia['low'].min()
                return asia_high, asia_low
        
        return None, None
    except Exception:
        return None, None

def check_amd_session():
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    
    session = "未知"
    
    if 0 <= hour < 7:
        session = "亞洲盤 (Accumulation)"
    elif 7 <= hour < 13:
        session = "倫敦盤 (Manipulation)"
    elif 13 <= hour < 16:
        session = "倫敦/紐約重疊 (High Volatility)"
    elif 16 <= hour < 22:
        session = "紐約盤 (Distribution)"
    else:
        session = "紐約盤後"
    
    return session

def calculate_confidence_score(df, is_long=True):
    """計算信心分數，基於多個共振因子 (Simplified for CLI)"""
    score = 0
    factors = []
    
    last_rsi = df['rsi'].iloc[-1]
    last_hist = df['macd_hist'].iloc[-1]
    prev_hist = df['macd_hist'].iloc[-2] if len(df) > 1 else 0
    
    fib_levels, _, _ = calculate_fibonacci(df)
    current_price = df['close'].iloc[-1]
    equilibrium = fib_levels['0.5']
    
    asia_high, asia_low = get_asia_session_range(df)
    
    if is_long:
        # 1. 大週期動能 (Simple 5-candle trend)
        if df['close'].iloc[-1] > df['close'].iloc[-5]:
            score += 1
            factors.append('✅ 大週期多頭動能')
        
        # 2. RSI 共振
        if last_rsi < 30:
            score += 1
            factors.append(f'✅ RSI 極度超賣 ({last_rsi:.1f})')
        elif last_rsi < 40:
            score += 0.5 # Partial score
            factors.append(f'✅ RSI 超賣 ({last_rsi:.1f})')
        
        # 3. MACD 共振
        if last_hist > prev_hist:
            score += 1
            factors.append('✅ MACD 動能增強')
            
        # 4. Fib 區域
        if current_price < equilibrium:
            score += 1
            factors.append('✅ 位於折扣區 (Discount)')
        
        # 5. Asia Range Sweep (Approximation)
        if asia_low and abs(current_price - asia_low) / asia_low < 0.005:
            score += 1
            factors.append('✅ 接近亞洲低點 (可能 Sweep)')

    else: # Short
        if df['close'].iloc[-1] < df['close'].iloc[-5]:
            score += 1
            factors.append('✅ 大週期空頭動能')
        
        if last_rsi > 70:
            score += 1
            factors.append(f'✅ RSI 極度超買 ({last_rsi:.1f})')
        elif last_rsi > 60:
            score += 0.5
            factors.append(f'✅ RSI 超買 ({last_rsi:.1f})')
            
        if last_hist < prev_hist:
            score += 1
            factors.append('✅ MACD 動能減弱')
            
        if current_price > equilibrium:
            score += 1
            factors.append('✅ 位於溢價區 (Premium)')
            
        if asia_high and abs(current_price - asia_high) / asia_high < 0.005:
            score += 1
            factors.append('✅ 接近亞洲高點 (可能 Sweep)')
            
    # Session Check
    session = check_amd_session()
    if '倫敦' in session or '紐約' in session:
        score += 1
        factors.append(f'✅ {session}')
    
    return score, factors

# --- Main Bot Loop ---

def analyze_symbol(symbol):
    print(f"Analyzing {symbol}...")
    df = fetch_market_data(symbol)
    
    if df is None:
        print(f"Failed to fetch data for {symbol}")
        return

    # Check Long
    long_score, long_factors = calculate_confidence_score(df, is_long=True)
    
    # Check Short
    short_score, short_factors = calculate_confidence_score(df, is_long=False)
    
    current_price = df['close'].iloc[-1]
    
    print(f"Price: {current_price:.2f} | RSI: {df['rsi'].iloc[-1]:.2f}")
    
    # Alert Logic
    if long_score >= 4:
        print(f"\n🟢 [ALERT] {symbol} LONG Opportunity (Score: {long_score}/6)")
        for f in long_factors:
            print(f"  - {f}")
    
    if short_score >= 4:
        print(f"\n🔴 [ALERT] {symbol} SHORT Opportunity (Score: {short_score}/6)")
        for f in short_factors:
            print(f"  - {f}")
            
    if long_score < 4 and short_score < 4:
        print(f"⚪ No strong setup. (Long: {long_score}, Short: {short_score})")
    
    print("-" * 30)

def main():
    print("🤖 Crypto Auto Bot Initialized...")
    print("Time interval: 60 minutes")
    
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Scan...")
        
        try:
            analyze_symbol("BTCUSDT")
            analyze_symbol("ETHUSDT")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        print("Scan complete. Sleeping for 60 minutes...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
