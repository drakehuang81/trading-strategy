import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD
from datetime import datetime, timedelta
import os
import numpy as np

# --- 設定 ---
st.set_page_config(page_title="量化交易儀表板", layout="wide", page_icon="📈")

# --- 自訂 CSS 樣式 ---
st.markdown("""
<style>
.alert-box-long {
    padding: 20px;
    background-color: #1a472a;
    border-left: 6px solid #00ff00;
    margin: 10px 0;
    border-radius: 5px;
}
.alert-box-short {
    padding: 20px;
    background-color: #4a1a1a;
    border-left: 6px solid #ff0000;
    margin: 10px 0;
    border-radius: 5px;
}
.confidence-score {
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- 側邊欄 ---
st.sidebar.title("⚙️ 設定")
symbol = st.sidebar.text_input("交易對", value="ETHUSDT")
interval = st.sidebar.selectbox("K線週期", options=[
    Client.KLINE_INTERVAL_15MINUTE,
    Client.KLINE_INTERVAL_1HOUR,
    Client.KLINE_INTERVAL_4HOUR
], format_func=lambda x: {"15m": "15分鐘", "1h": "1小時", "4h": "4小時"}.get(x, x))

lookback = st.sidebar.selectbox("回看時間", options=[
    "1 day ago UTC", "3 days ago UTC", "1 week ago UTC", "2 weeks ago UTC"
], format_func=lambda x: {"1 day ago UTC": "過去 1 天", "3 days ago UTC": "過去 3 天", 
                          "1 week ago UTC": "過去 1 週", "2 weeks ago UTC": "過去 2 週"}.get(x, x), index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 功能開關")
show_fib = st.sidebar.checkbox("顯示斐波那契回撤位", value=True)
show_amd = st.sidebar.checkbox("顯示 AMD 交易時段", value=True)
enable_alerts = st.sidebar.checkbox("啟用交易警報", value=True)

# --- 函式 ---
@st.cache_data(ttl=60)
def get_data(symbol, interval, lookback):
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = Client(api_key, api_secret)
    
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
    
    return df

def calculate_indicators(df):
    # RSI
    rsi_indicator = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi_indicator.rsi()
    
    # MACD
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    return df

def calculate_fibonacci(df):
    """根據近期高低點計算斐波那契回撤位"""
    swing_high = df['high'].max()
    swing_low = df['low'].min()
    
    diff = swing_high - swing_low
    
    levels = {
        '0.0 (波段高點)': swing_high,
        '0.236': swing_high - 0.236 * diff,
        '0.382': swing_high - 0.382 * diff,
        '0.5 (平衡點)': swing_high - 0.5 * diff,
        '0.618 (黃金分割)': swing_high - 0.618 * diff,
        '0.786': swing_high - 0.786 * diff,
        '1.0 (波段低點)': swing_low
    }
    
    return levels, swing_high, swing_low

def get_asia_session_range(df):
    """取得亞洲盤的高低點 (UTC 00:00 - 07:00)"""
    try:
        df_copy = df.copy()
        # 確保索引是 DatetimeIndex
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            return None, None
            
        # 使用 UTC 時間，亞洲盤 = 00:00 - 07:00 UTC
        # 過濾出亞洲時段的數據
        asia_mask = (df_copy.index.hour >= 0) & (df_copy.index.hour < 7)
        asia_data = df_copy[asia_mask]
        
        if len(asia_data) >= 4:  # 至少要有幾根K線
            # 取最近一個交易日的亞洲盤
            asia_data_sorted = asia_data.sort_index()
            recent_date = asia_data_sorted.index[-1].date()
            today_asia = asia_data_sorted[asia_data_sorted.index.date == recent_date]
            
            if len(today_asia) > 0:
                asia_high = today_asia['high'].max()
                asia_low = today_asia['low'].min()
                return asia_high, asia_low
        
        return None, None
    except Exception as e:
        # 如果出錯，返回 None
        return None, None

def check_amd_session():
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    
    session = "未知"
    color = "gray"
    emoji = "🕐"
    
    if 0 <= hour < 7:
        session = "亞洲盤 (累積階段)"
        color = "#FFD700"
        emoji = "🌏"
    elif 7 <= hour < 13:
        session = "倫敦盤 (操縱階段)"
        color = "#FF6B35"
        emoji = "🇬🇧"
    elif 13 <= hour < 16:
        session = "倫敦/紐約重疊 (高波動)"
        color = "#FF0000"
        emoji = "🔥"
    elif 16 <= hour < 22:
        session = "紐約盤 (派發階段)"
        color = "#4169E1"
        emoji = "🇺🇸"
    else:
        session = "紐約盤後 / 亞洲盤前"
        color = "#808080"
        emoji = "🌙"
    
    return session, color, emoji

def calculate_confidence_score(df, is_long=True):
    """計算信心分數，基於多個共振因子"""
    score = 0
    factors = {}
    
    last_rsi = df['rsi'].iloc[-1]
    last_hist = df['macd_hist'].iloc[-1]
    prev_hist = df['macd_hist'].iloc[-2] if len(df) > 1 else 0
    
    fib_levels, swing_high, swing_low = calculate_fibonacci(df)
    current_price = df['close'].iloc[-1]
    equilibrium = fib_levels['0.5 (平衡點)']
    
    asia_high, asia_low = get_asia_session_range(df)
    
    if is_long:
        # 1. 大週期動能
        if df['close'].iloc[-1] > df['close'].iloc[-5]:
            score += 1
            factors['大週期動能'] = '✅ 多頭動能'
        else:
            factors['大週期動能'] = '❌ 空頭動能'
        
        # 2. RSI 共振
        if last_rsi < 40:
            score += 1
            factors['RSI 指標'] = f'✅ 超賣區 ({last_rsi:.1f})'
        elif last_rsi < 30:
            score += 1
            factors['RSI 指標'] = f'✅✅ 極度超賣 ({last_rsi:.1f})'
        else:
            factors['RSI 指標'] = f'⚠️ 中性 ({last_rsi:.1f})'
        
        # 3. MACD 共振
        if last_hist > prev_hist:
            score += 1
            factors['MACD 指標'] = '✅ 動能增強'
        else:
            factors['MACD 指標'] = '❌ 動能減弱'
            
        # 4. 斐波那契區域 (折扣區)
        if current_price < equilibrium:
            score += 1
            factors['Fib 區域'] = '✅ 位於折扣區 (Discount)'
        else:
            factors['Fib 區域'] = '❌ 位於溢價區 (Premium)'
        
        # 5. 接近亞洲低點 (潛在 Sweep)
        if asia_low and abs(current_price - asia_low) / asia_low < 0.01:
            score += 1
            factors['亞洲區間'] = '✅ 接近亞洲低點 (ASL)'
        else:
            factors['亞洲區間'] = '⚠️ 遠離亞洲低點'
            
    else:  # Short
        if df['close'].iloc[-1] < df['close'].iloc[-5]:
            score += 1
            factors['大週期動能'] = '✅ 空頭動能'
        else:
            factors['大週期動能'] = '❌ 多頭動能'
        
        if last_rsi > 60:
            score += 1
            factors['RSI 指標'] = f'✅ 超買區 ({last_rsi:.1f})'
        elif last_rsi > 70:
            score += 1
            factors['RSI 指標'] = f'✅✅ 極度超買 ({last_rsi:.1f})'
        else:
            factors['RSI 指標'] = f'⚠️ 中性 ({last_rsi:.1f})'
        
        if last_hist < prev_hist:
            score += 1
            factors['MACD 指標'] = '✅ 動能減弱'
        else:
            factors['MACD 指標'] = '❌ 動能增強'
            
        if current_price > equilibrium:
            score += 1
            factors['Fib 區域'] = '✅ 位於溢價區 (Premium)'
        else:
            factors['Fib 區域'] = '❌ 位於折扣區 (Discount)'
        
        if asia_high and abs(current_price - asia_high) / asia_high < 0.01:
            score += 1
            factors['亞洲區間'] = '✅ 接近亞洲高點 (ASH)'
        else:
            factors['亞洲區間'] = '⚠️ 遠離亞洲高點'
    
    # 6. 交易時段檢查
    session, _, _ = check_amd_session()
    if '倫敦' in session or '紐約' in session:
        score += 1
        factors['交易時段'] = f'✅ {session}'
    else:
        factors['交易時段'] = f'⚠️ {session} (低波動)'
    
    return score, factors

def get_rating(score):
    if score >= 5:
        return "⭐⭐⭐⭐⭐ A+ 級設置", "#00ff00"
    elif score >= 4:
        return "⭐⭐⭐⭐ A 級設置", "#7fff00"
    elif score >= 3:
        return "⭐⭐⭐ B 級設置", "#ffff00"
    elif score >= 2:
        return "⭐⭐ C 級設置", "#ffa500"
    else:
        return "⭐ D/F 級設置 (建議跳過)", "#ff0000"

# --- 主程式 ---
st.title(f"🦅 加密貨幣剝頭皮儀表板: {symbol}")

with st.spinner('正在從幣安獲取數據...'):
    try:
        df = get_data(symbol, interval, lookback)
        df = calculate_indicators(df)
        
        current_price = df['close'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['open'].iloc[0]
        price_pct = (price_change / df['open'].iloc[0]) * 100
        
        # 指標列
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("當前價格", f"${current_price:.2f}", f"{price_change:.2f} ({price_pct:.2f}%)")
        col2.metric("RSI (14)", f"{df['rsi'].iloc[-1]:.2f}")
        col3.metric("MACD 柱狀圖", f"{df['macd_hist'].iloc[-1]:.4f}")
        
        session_name, session_color, session_emoji = check_amd_session()
        col4.markdown(f"""
        <div style='background-color: {session_color}; padding: 10px; border-radius: 5px; text-align: center;'>
            <span style='font-size: 20px;'>{session_emoji}</span><br>
            <b>{session_name}</b>
        </div>
        """, unsafe_allow_html=True)
        
        # --- 圖表 ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=(f"{symbol} K線圖", "RSI (14)", "MACD"))

        # K線圖
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'], name='K線'), row=1, col=1)

        # --- 斐波那契線 ---
        if show_fib:
            fib_levels, swing_high, swing_low = calculate_fibonacci(df)
            fib_colors = {
                '0.0 (波段高點)': 'rgba(255,0,0,0.8)',
                '0.236': 'rgba(255,165,0,0.6)',
                '0.382': 'rgba(255,255,0,0.6)',
                '0.5 (平衡點)': 'rgba(0,255,0,0.8)',
                '0.618 (黃金分割)': 'rgba(0,255,255,0.8)',
                '0.786': 'rgba(138,43,226,0.6)',
                '1.0 (波段低點)': 'rgba(0,0,255,0.8)'
            }
            
            for level_name, price in fib_levels.items():
                fig.add_hline(y=price, line_dash="dot", 
                             line_color=fib_colors.get(level_name, 'gray'),
                             annotation_text=f"Fib {level_name}: ${price:.2f}",
                             annotation_position="right",
                             row=1, col=1)

        # --- AMD 交易時段區域 ---
        if show_amd:
            asia_high, asia_low = get_asia_session_range(df)
            if asia_high and asia_low:
                fig.add_hline(y=asia_high, line_dash="dash", line_color="gold", line_width=2,
                             annotation_text=f"亞洲高點 (ASH): ${asia_high:.2f}",
                             annotation_position="left", row=1, col=1)
                fig.add_hline(y=asia_low, line_dash="dash", line_color="gold", line_width=2,
                             annotation_text=f"亞洲低點 (ASL): ${asia_low:.2f}",
                             annotation_position="left", row=1, col=1)
                             
                fig.add_hrect(y0=asia_low, y1=asia_high, 
                             fillcolor="rgba(255, 215, 0, 0.1)", 
                             line_width=0, row=1, col=1,
                             annotation_text="亞洲盤區間", annotation_position="top left")

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="超買 70")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="超賣 30")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

        # MACD
        colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='MACD 柱狀圖', marker_color=colors), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD 線', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='訊號線', line=dict(color='orange')), row=3, col=1)

        fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_dark",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 交易警報區塊 ---
        if enable_alerts:
            st.markdown("---")
            st.subheader("🚨 交易訊號與警報")
            
            last_rsi = df['rsi'].iloc[-1]
            last_hist = df['macd_hist'].iloc[-1]
            prev_hist = df['macd_hist'].iloc[-2] if len(df) > 1 else 0
            
            long_signal = last_rsi < 35 and last_hist > prev_hist
            short_signal = last_rsi > 65 and last_hist < prev_hist
            
            if long_signal:
                score, factors = calculate_confidence_score(df, is_long=True)
                rating, rating_color = get_rating(score)
                
                st.markdown(f"""
                <div class="alert-box-long">
                    <h2 style='color: #00ff00; margin:0;'>🟢 偵測到潛在做多訊號！</h2>
                    <p class="confidence-score" style='color: {rating_color};'>信心分數: {score}/6 - {rating}</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📋 查看共振因子", expanded=True):
                    for factor, status in factors.items():
                        st.write(f"**{factor}**: {status}")
                
                st.warning("⚠️ 進場前請確認 SMC 結構 (Order Block / 流動性獵殺)！")
                
            elif short_signal:
                score, factors = calculate_confidence_score(df, is_long=False)
                rating, rating_color = get_rating(score)
                
                st.markdown(f"""
                <div class="alert-box-short">
                    <h2 style='color: #ff0000; margin:0;'>🔴 偵測到潛在做空訊號！</h2>
                    <p class="confidence-score" style='color: {rating_color};'>信心分數: {score}/6 - {rating}</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📋 查看共振因子", expanded=True):
                    for factor, status in factors.items():
                        st.write(f"**{factor}**: {status}")
                
                st.warning("⚠️ 進場前請確認 SMC 結構 (Supply Zone / 流動性獵殺)！")
                
            else:
                st.info("⚪ **目前無明確訊號。** 市場處於中性區間。")
                
                with st.expander("📊 當前市場分析"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**做多分析:**")
                        score_long, factors_long = calculate_confidence_score(df, is_long=True)
                        rating_long, _ = get_rating(score_long)
                        st.write(f"分數: {score_long}/6 - {rating_long}")
                        for f, s in factors_long.items():
                            st.write(f"- {f}: {s}")
                    
                    with col2:
                        st.markdown("**做空分析:**")
                        score_short, factors_short = calculate_confidence_score(df, is_long=False)
                        rating_short, _ = get_rating(score_short)
                        st.write(f"分數: {score_short}/6 - {rating_short}")
                        for f, s in factors_short.items():
                            st.write(f"- {f}: {s}")

        # --- 關鍵價位摘要 ---
        st.markdown("---")
        st.subheader("📍 關鍵價位")
        
        level_col1, level_col2, level_col3 = st.columns(3)
        
        with level_col1:
            st.markdown("**斐波那契回撤位:**")
            if show_fib:
                fib_levels, _, _ = calculate_fibonacci(df)
                for name, price in fib_levels.items():
                    st.write(f"- {name}: ${price:.2f}")
        
        with level_col2:
            st.markdown("**AMD 交易時段:**")
            asia_high, asia_low = get_asia_session_range(df)
            if asia_high and asia_low:
                st.write(f"- 亞洲高點 (ASH): ${asia_high:.2f}")
                st.write(f"- 亞洲低點 (ASL): ${asia_low:.2f}")
                st.write(f"- 區間幅度: ${asia_high - asia_low:.2f} ({((asia_high - asia_low)/asia_low)*100:.2f}%)")
            else:
                st.write("亞洲盤數據不足")
        
        with level_col3:
            st.markdown("**當前位置:**")
            fib_levels, swing_high, swing_low = calculate_fibonacci(df)
            equilibrium = fib_levels['0.5 (平衡點)']
            if current_price > equilibrium:
                zone = "溢價區 (Premium Zone) - 做空區"
                zone_color = "red"
            else:
                zone = "折扣區 (Discount Zone) - 做多區"
                zone_color = "green"
            st.markdown(f"<span style='color:{zone_color}; font-weight:bold;'>{zone}</span>", unsafe_allow_html=True)
            st.write(f"距離 0.618: ${abs(current_price - fib_levels['0.618 (黃金分割)']):.2f}")

    except Exception as e:
        st.error(f"獲取數據時發生錯誤: {e}")
        st.exception(e)

st.write("---")
st.markdown("*數據來源：幣安。本儀表板僅供教育參考，不構成投資建議。請務必做好風險管理。*")
st.markdown("*使用 Streamlit、Plotly 和 python-binance 構建 ❤️*")
