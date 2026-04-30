"""
Smart Money Concepts (SMC) Module

Implements SMC structure identification including:
- Swing Points (Swing High / Swing Low)
- BOS (Break of Structure)
- CHoCH (Change of Character)
- Order Blocks (OB)
- Fair Value Gaps (FVG)
"""

from typing import Any, List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


def find_swing_points(
    df: pd.DataFrame,
    window: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identify Swing Highs and Swing Lows in price data.
    
    A Swing High is a high that is higher than the highs of 'window' candles
    on both sides.
    
    A Swing Low is a low that is lower than the lows of 'window' candles
    on both sides.
    
    Args:
        df: DataFrame with 'high' and 'low' columns
        window: Number of candles to look on each side
        
    Returns:
        Tuple of (swing_highs, swing_lows) where each is a list of dicts:
        [{'index': datetime, 'price': float, 'idx': int}, ...]
    """
    swing_highs = []
    swing_lows = []
    
    highs = df['high'].values
    lows = df['low'].values
    indices = df.index.tolist()
    
    for i in range(window, len(df) - window):
        # Check for Swing High
        is_swing_high = True
        for j in range(1, window + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append({
                'index': indices[i],
                'price': highs[i],
                'idx': i
            })
        
        # Check for Swing Low
        is_swing_low = True
        for j in range(1, window + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append({
                'index': indices[i],
                'price': lows[i],
                'idx': i
            })
    
    return swing_highs, swing_lows


def identify_structure(
    df: pd.DataFrame,
    swing_highs: Optional[List[Dict[str, Any]]] = None,
    swing_lows: Optional[List[Dict[str, Any]]] = None,
    window: int = 5
) -> Dict[str, Any]:
    """
    Identify market structure including BOS and CHoCH.
    
    Structure Types:
    - BULLISH: Higher Highs (HH) and Higher Lows (HL)
    - BEARISH: Lower Highs (LH) and Lower Lows (LL)
    - RANGING: No clear trend
    
    BOS (Break of Structure):
    - Bullish BOS: Price breaks above previous Swing High
    - Bearish BOS: Price breaks below previous Swing Low
    
    CHoCH (Change of Character):
    - First break against the prevailing trend, signaling potential reversal
    
    Args:
        df: DataFrame with OHLC data
        swing_highs: Pre-calculated swing highs (optional)
        swing_lows: Pre-calculated swing lows (optional)
        window: Window for swing point detection
        
    Returns:
        Dictionary with structure analysis:
        {
            'structure': 'BULLISH' | 'BEARISH' | 'RANGING',
            'swing_highs': [...],
            'swing_lows': [...],
            'last_bos': {'type': 'BULLISH'|'BEARISH', 'price': float, 'index': datetime},
            'last_choch': {...} or None,
            'hh_hl_count': int,
            'lh_ll_count': int
        }
    """
    if swing_highs is None or swing_lows is None:
        swing_highs, swing_lows = find_swing_points(df, window)
    
    result: Dict[str, Any] = {
        'structure': 'RANGING',
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'last_bos': None,
        'last_choch': None,
        'hh_hl_count': 0,
        'lh_ll_count': 0,
        'pattern_sequence': []
    }
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return result
    
    # Analyze swing high patterns (HH or LH)
    high_pattern = []
    for i in range(1, len(swing_highs)):
        if swing_highs[i]['price'] > swing_highs[i-1]['price']:
            high_pattern.append('HH')
        else:
            high_pattern.append('LH')
    
    # Analyze swing low patterns (HL or LL)
    low_pattern = []
    for i in range(1, len(swing_lows)):
        if swing_lows[i]['price'] > swing_lows[i-1]['price']:
            low_pattern.append('HL')
        else:
            low_pattern.append('LL')
    
    # Count patterns in recent swings (last 3-4)
    recent_highs = high_pattern[-3:] if len(high_pattern) >= 3 else high_pattern
    recent_lows = low_pattern[-3:] if len(low_pattern) >= 3 else low_pattern
    
    hh_count = recent_highs.count('HH')
    lh_count = recent_highs.count('LH')
    hl_count = recent_lows.count('HL')
    ll_count = recent_lows.count('LL')
    
    result['hh_hl_count'] = hh_count + hl_count
    result['lh_ll_count'] = lh_count + ll_count
    
    # Determine structure
    if hh_count >= 2 and hl_count >= 2:
        result['structure'] = 'BULLISH'
    elif lh_count >= 2 and ll_count >= 2:
        result['structure'] = 'BEARISH'
    else:
        result['structure'] = 'RANGING'
    
    # Find last BOS
    current_price = df['close'].iloc[-1]
    
    # Check for bullish BOS (break above recent swing high)
    if len(swing_highs) >= 1:
        last_sh = swing_highs[-1]
        if current_price > last_sh['price']:
            result['last_bos'] = {
                'type': 'BULLISH',
                'price': last_sh['price'],
                'index': last_sh['index']
            }
    
    # Check for bearish BOS (break below recent swing low)
    if len(swing_lows) >= 1:
        last_sl = swing_lows[-1]
        if current_price < last_sl['price']:
            result['last_bos'] = {
                'type': 'BEARISH',
                'price': last_sl['price'],
                'index': last_sl['index']
            }
    
    # Detect CHoCH (first break against the trend)
    if result['structure'] == 'BULLISH' and len(low_pattern) >= 2:
        # In bullish structure, CHoCH is breaking below a swing low
        if low_pattern[-1] == 'LL':
            result['last_choch'] = {
                'type': 'BEARISH_CHOCH',
                'price': swing_lows[-1]['price'],
                'index': swing_lows[-1]['index'],
                'warning': 'Potential trend reversal to bearish'
            }
    
    if result['structure'] == 'BEARISH' and len(high_pattern) >= 2:
        # In bearish structure, CHoCH is breaking above a swing high
        if high_pattern[-1] == 'HH':
            result['last_choch'] = {
                'type': 'BULLISH_CHOCH',
                'price': swing_highs[-1]['price'],
                'index': swing_highs[-1]['index'],
                'warning': 'Potential trend reversal to bullish'
            }
    
    # Create pattern sequence for display
    combined = []
    for sh in swing_highs:
        combined.append(('SH', sh))
    for sl in swing_lows:
        combined.append(('SL', sl))
    combined.sort(key=lambda x: x[1]['idx'])
    result['pattern_sequence'] = combined[-8:]  # Last 8 swing points
    
    return result


def find_order_blocks(
    df: pd.DataFrame,
    lookback: int = 50,
    min_move_pct: float = 0.5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify Order Blocks (OB) in price data.
    
    Bullish Order Block:
    - The last bearish (red) candle before a strong bullish move
    - Represents institutional buying zone
    
    Bearish Order Block:
    - The last bullish (green) candle before a strong bearish move
    - Represents institutional selling zone
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of candles to analyze
        min_move_pct: Minimum percentage move to qualify as "strong"
        
    Returns:
        Dictionary with 'bullish' and 'bearish' order blocks
    """
    recent = df.tail(lookback).copy()
    
    bullish_obs = []
    bearish_obs = []
    
    opens = recent['open'].values
    closes = recent['close'].values
    highs = recent['high'].values
    lows = recent['low'].values
    indices = recent.index.tolist()
    
    for i in range(1, len(recent) - 3):
        # Check for Bullish OB (bearish candle before bullish move)
        if closes[i] < opens[i]:  # Bearish candle
            # Check if followed by strong bullish move
            move_up = (closes[i + 3] - closes[i]) / closes[i] * 100
            if move_up >= min_move_pct:
                bullish_obs.append({
                    'index': indices[i],
                    'high': highs[i],
                    'low': lows[i],
                    'open': opens[i],
                    'close': closes[i],
                    'zone': (lows[i], highs[i]),
                    'strength': move_up
                })
        
        # Check for Bearish OB (bullish candle before bearish move)
        if closes[i] > opens[i]:  # Bullish candle
            # Check if followed by strong bearish move
            move_down = (closes[i] - closes[i + 3]) / closes[i] * 100
            if move_down >= min_move_pct:
                bearish_obs.append({
                    'index': indices[i],
                    'high': highs[i],
                    'low': lows[i],
                    'open': opens[i],
                    'close': closes[i],
                    'zone': (lows[i], highs[i]),
                    'strength': move_down
                })
    
    # Sort by recency and strength, keep top 3
    bullish_obs.sort(key=lambda x: x['strength'], reverse=True)
    bearish_obs.sort(key=lambda x: x['strength'], reverse=True)
    
    return {
        'bullish': bullish_obs[:3],
        'bearish': bearish_obs[:3]
    }


def find_fvg(
    df: pd.DataFrame,
    lookback: int = 50,
    min_gap_pct: float = 0.1
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify Fair Value Gaps (FVG) / Imbalances in price data.
    
    FVG is a 3-candle pattern where:
    - The wicks of candle 1 and candle 3 do not overlap
    - This leaves an "imbalance" or gap that price often returns to fill
    
    Bullish FVG:
    - Candle 3's low > Candle 1's high
    - Indicates buying imbalance
    
    Bearish FVG:
    - Candle 3's high < Candle 1's low
    - Indicates selling imbalance
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of candles to analyze
        min_gap_pct: Minimum gap size as percentage
        
    Returns:
        Dictionary with 'bullish' and 'bearish' FVGs
    """
    recent = df.tail(lookback).copy()
    
    bullish_fvgs = []
    bearish_fvgs = []
    
    highs = recent['high'].values
    lows = recent['low'].values
    indices = recent.index.tolist()
    
    for i in range(2, len(recent)):
        # Bullish FVG: low of candle 3 > high of candle 1
        if lows[i] > highs[i - 2]:
            gap_size = lows[i] - highs[i - 2]
            gap_pct = gap_size / highs[i - 2] * 100
            
            if gap_pct >= min_gap_pct:
                bullish_fvgs.append({
                    'index': indices[i - 1],  # Middle candle
                    'gap_high': lows[i],
                    'gap_low': highs[i - 2],
                    'zone': (highs[i - 2], lows[i]),
                    'gap_pct': gap_pct,
                    'filled': False
                })
        
        # Bearish FVG: high of candle 3 < low of candle 1
        if highs[i] < lows[i - 2]:
            gap_size = lows[i - 2] - highs[i]
            gap_pct = gap_size / lows[i - 2] * 100
            
            if gap_pct >= min_gap_pct:
                bearish_fvgs.append({
                    'index': indices[i - 1],  # Middle candle
                    'gap_high': lows[i - 2],
                    'gap_low': highs[i],
                    'zone': (highs[i], lows[i - 2]),
                    'gap_pct': gap_pct,
                    'filled': False
                })
    
    # Check if FVGs have been filled
    current_price = df['close'].iloc[-1]
    
    for fvg in bullish_fvgs:
        if current_price <= fvg['gap_high'] and current_price >= fvg['gap_low']:
            fvg['filled'] = True
    
    for fvg in bearish_fvgs:
        if current_price <= fvg['gap_high'] and current_price >= fvg['gap_low']:
            fvg['filled'] = True
    
    return {
        'bullish': [f for f in bullish_fvgs if not f['filled']][-3:],
        'bearish': [f for f in bearish_fvgs if not f['filled']][-3:]
    }


def get_nearest_poi(
    df: pd.DataFrame,
    direction: str,
    window: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Get the nearest Point of Interest (POI) for entry.
    
    POI can be an Order Block, FVG, or significant swing level.
    
    Args:
        df: DataFrame with OHLC data
        direction: "LONG" or "SHORT"
        window: Swing point detection window
        
    Returns:
        Nearest POI or None if not found
    """
    current_price = df['close'].iloc[-1]
    
    # Get all POIs
    order_blocks = find_order_blocks(df)
    fvgs = find_fvg(df)
    swing_highs, swing_lows = find_swing_points(df, window)
    
    candidates = []
    
    if direction.upper() == "LONG":
        # For longs, look for bullish OBs and bullish FVGs below current price
        for ob in order_blocks['bullish']:
            if ob['zone'][1] < current_price:
                candidates.append({
                    'type': 'ORDER_BLOCK',
                    'zone': ob['zone'],
                    'mid_price': (ob['zone'][0] + ob['zone'][1]) / 2,
                    'distance': current_price - ob['zone'][1],
                    'strength': ob['strength']
                })
        
        for fvg in fvgs['bullish']:
            if fvg['zone'][1] < current_price:
                candidates.append({
                    'type': 'FVG',
                    'zone': fvg['zone'],
                    'mid_price': (fvg['zone'][0] + fvg['zone'][1]) / 2,
                    'distance': current_price - fvg['zone'][1],
                    'strength': fvg['gap_pct']
                })
        
        # Add recent swing lows as potential demand
        for sl in swing_lows[-3:]:
            if sl['price'] < current_price:
                candidates.append({
                    'type': 'SWING_LOW',
                    'zone': (sl['price'] * 0.998, sl['price'] * 1.002),
                    'mid_price': sl['price'],
                    'distance': current_price - sl['price'],
                    'strength': 1.0
                })
    
    else:  # SHORT
        # For shorts, look for bearish OBs and bearish FVGs above current price
        for ob in order_blocks['bearish']:
            if ob['zone'][0] > current_price:
                candidates.append({
                    'type': 'ORDER_BLOCK',
                    'zone': ob['zone'],
                    'mid_price': (ob['zone'][0] + ob['zone'][1]) / 2,
                    'distance': ob['zone'][0] - current_price,
                    'strength': ob['strength']
                })
        
        for fvg in fvgs['bearish']:
            if fvg['zone'][0] > current_price:
                candidates.append({
                    'type': 'FVG',
                    'zone': fvg['zone'],
                    'mid_price': (fvg['zone'][0] + fvg['zone'][1]) / 2,
                    'distance': fvg['zone'][0] - current_price,
                    'strength': fvg['gap_pct']
                })
        
        # Add recent swing highs as potential supply
        for sh in swing_highs[-3:]:
            if sh['price'] > current_price:
                candidates.append({
                    'type': 'SWING_HIGH',
                    'zone': (sh['price'] * 0.998, sh['price'] * 1.002),
                    'mid_price': sh['price'],
                    'distance': sh['price'] - current_price,
                    'strength': 1.0
                })
    
    if not candidates:
        return None
    
    # Sort by distance (nearest first) then by strength
    candidates.sort(key=lambda x: (x['distance'], -x['strength']))

    return candidates[0]


# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2
# (from __future__ imports intentionally omitted: appended blocks
# cannot contain __future__ imports. Use modern `X | None` syntax
# only where the file has no pre-existing `from __future__` line.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class SMCFeature:
    """Packages SMC structure (swings, BOS/CHoCH, OB, FVG) as a single
    point-in-time Feature. Only reads df[df.index <= as_of]."""

    name: str = "smc"
    version: str = "1.0.0"
    required_lookback: int = 100

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {
                "swing_highs": [],
                "swing_lows": [],
                "structure": None,
                "order_blocks": [],
                "fvgs": [],
            }
        swing_highs, swing_lows = find_swing_points(slice_df)
        structure = identify_structure(slice_df, swing_highs, swing_lows)
        order_blocks = find_order_blocks(slice_df)
        fvgs = find_fvg(slice_df)
        return {
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "structure": structure,
            "order_blocks": order_blocks,
            "fvgs": fvgs,
        }
