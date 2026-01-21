"""
Crypto Auto Trading Bot

Automated market analysis using multi-timeframe data and
SMC-based strategy from crypto_trading_skill.

Features:
- Multi-timeframe analysis (4H/1H/15m)
- SMC structure identification (BOS/CHoCH)
- Liquidity pool detection (EQH/EQL)
- Fibonacci OTE zone analysis
- RSI/MACD divergence detection
- 8-factor confidence scoring
- Complete trade setup (Entry/SL/TP)
"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from data_ingestion.fetch_latest import (
    fetch_multi_timeframe_data,
    determine_htf_bias,
    calculate_asia_range,
    get_current_session
)

from strategy.fibonacci import calculate_fib_from_df, identify_fib_confluence
from strategy.smc import identify_structure, find_swing_points, get_nearest_poi
from strategy.liquidity import get_liquidity_zones, check_liquidity_sweep
from strategy.divergence import get_divergence_summary
from strategy.trade_setup import calculate_trade_setup, format_trade_setup
from strategy.confidence import calculate_confidence_score, format_confidence_score


def analyze_symbol(symbol: str, verbose: bool = True):
    """
    Perform complete multi-timeframe analysis on a symbol.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING: {symbol}")
        print(f"{'='*60}")
    
    # 1. Fetch multi-timeframe data
    data = fetch_multi_timeframe_data(symbol)
    if not data or "15m" not in data:
        print(f"❌ Failed to fetch data for {symbol}")
        return None
    
    df_4h = data.get("4h")
    df_1h = data.get("1h")
    df_15m = data.get("15m")
    
    current_price = df_15m['close'].iloc[-1]
    rsi_15m = df_15m['rsi'].iloc[-1]
    macd_hist = df_15m['macd_hist'].iloc[-1]
    prev_macd_hist = df_15m['macd_hist'].iloc[-2] if len(df_15m) > 1 else 0
    
    # 2. Determine HTF Bias
    htf_bias, bias_details = determine_htf_bias(df_4h, df_1h)
    
    # 3. Get current session and Asia range
    session, session_desc = get_current_session()
    ash, asl, asia_date, _ = calculate_asia_range(df_15m)
    
    # 4. SMC Structure Analysis
    structure = identify_structure(df_15m)
    swing_highs, swing_lows = find_swing_points(df_15m)
    
    # 5. Liquidity Analysis
    liquidity_zones = get_liquidity_zones(df_15m)
    liquidity_sweep = check_liquidity_sweep(df_15m, liquidity_zones)
    
    # 6. Fibonacci Analysis
    fib_analysis = calculate_fib_from_df(df_15m, lookback=96)
    
    # 7. Divergence Analysis
    divergence = get_divergence_summary(df_15m)
    
    if verbose:
        print(f"\n📊 MARKET OVERVIEW")
        print(f"   Price: ${current_price:,.2f}")
        print(f"   RSI: {rsi_15m:.2f}")
        print(f"   MACD Hist: {macd_hist:.4f}")
        print(f"   Session: {session}")
        
        print(f"\n📈 HTF BIAS: {'🟢' if htf_bias == 'BULLISH' else '🔴' if htf_bias == 'BEARISH' else '⚪'} {htf_bias}")
        print(f"   Structure: {structure['structure']}")
        print(f"   BOS: {structure['last_bos']}")
        
        print(f"\n💧 LIQUIDITY")
        print(f"   EQH Zones: {len(liquidity_zones['eqh'])}")
        print(f"   EQL Zones: {len(liquidity_zones['eql'])}")
        print(f"   EQH Swept: {liquidity_sweep['eqh_swept']}")
        print(f"   EQL Swept: {liquidity_sweep['eql_swept']}")
        
        print(f"\n📐 FIBONACCI")
        print(f"   Zone: {fib_analysis['zone']}")
        print(f"   In OTE: {fib_analysis['confluence']['is_ote']}")
        
        if divergence['has_divergence']:
            print(f"\n⚡ DIVERGENCE DETECTED")
            if divergence['rsi_divergence']:
                print(f"   RSI: {divergence['rsi_divergence']['type']}")
            if divergence['macd_divergence']:
                print(f"   MACD: {divergence['macd_divergence']['type']}")
    
    # 8. Evaluate both LONG and SHORT setups
    results = {'symbol': symbol, 'price': current_price, 'setups': []}
    
    for direction in ["LONG", "SHORT"]:
        poi = get_nearest_poi(df_15m, direction)
        
        # Get appropriate swing points for SL calculation
        recent_swing_low = swing_lows[-1]['price'] if swing_lows else current_price * 0.98
        recent_swing_high = swing_highs[-1]['price'] if swing_highs else current_price * 1.02
        
        # Calculate confidence score
        score_result = calculate_confidence_score(
            direction=direction,
            htf_bias=htf_bias,
            poi=poi,
            fib_analysis=fib_analysis,
            liquidity_sweep=liquidity_sweep,
            rsi=rsi_15m,
            macd_hist=macd_hist,
            prev_macd_hist=prev_macd_hist,
            ash=ash,
            asl=asl,
            current_price=current_price,
            rsi_divergence=divergence.get('rsi_divergence'),
            macd_divergence=divergence.get('macd_divergence')
        )
        
        results['setups'].append({
            'direction': direction,
            'score': score_result['score'],
            'rating': score_result['rating'],
            'poi': poi,
            'confidence': score_result
        })
        
        # Only show high confidence setups (score >= 4)
        if score_result['score'] >= 4:
            if verbose:
                print(f"\n{'🟢' if direction == 'LONG' else '🔴'} {direction} OPPORTUNITY")
                print(format_confidence_score(score_result))
            
            # Generate trade setup
            entry_price = poi['mid_price'] if poi else current_price
            trade = calculate_trade_setup(
                direction=direction,
                entry_price=entry_price,
                swing_low=recent_swing_low,
                swing_high=recent_swing_high
            )
            
            if verbose:
                print(f"\n{format_trade_setup(trade, symbol)}")
            
            results['setups'][-1]['trade'] = trade
    
    # Summary
    if verbose:
        long_score = next((s['score'] for s in results['setups'] if s['direction'] == 'LONG'), 0)
        short_score = next((s['score'] for s in results['setups'] if s['direction'] == 'SHORT'), 0)
        
        print(f"\n{'='*60}")
        print(f"📋 SUMMARY: {symbol}")
        print(f"   LONG Score:  {long_score}/8")
        print(f"   SHORT Score: {short_score}/8")
        
        if max(long_score, short_score) < 4:
            print(f"\n   ⚪ No strong setup. WAIT for better confluence.")
        print(f"{'='*60}")
    
    return results


def main():
    """Main bot loop."""
    taipei_tz = ZoneInfo("Asia/Taipei")
    
    print("🤖 Crypto Auto Bot v2.0 - Multi-Timeframe Analysis")
    print("=" * 60)
    print(f"Strategy: SMC + RSI/MACD Confluence")
    print(f"Timeframes: 4H (Bias) → 1H (Confirm) → 15m (Execute)")
    print(f"Scan Interval: 60 minutes")
    print("=" * 60)
    
    while True:
        now = datetime.now(taipei_tz)
        print(f"\n⏰ [{now.strftime('%Y-%m-%d %H:%M:%S')}] Starting Scan...")
        
        try:
            analyze_symbol("BTCUSDT")
            analyze_symbol("ETHUSDT")
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n💤 Scan complete. Sleeping for 60 minutes...")
        time.sleep(3600)


if __name__ == "__main__":
    main()
