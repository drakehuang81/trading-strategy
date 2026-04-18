"""
Liquidity Pool Detection Module

Identifies liquidity pools where stop losses are clustered:
- EQH (Equal Highs): Multiple swing highs at similar prices = sell-side liquidity
- EQL (Equal Lows): Multiple swing lows at similar prices = buy-side liquidity

Smart money hunts these liquidity pools before making real moves.
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from features.smc import find_swing_points


def find_equal_highs(
    df: pd.DataFrame,
    tolerance: float = 0.002,
    min_count: int = 2,
    window: int = 5
) -> List[Dict]:
    """
    Identify Equal Highs (EQH) - clusters of similar swing highs.
    
    EQH represents sell-side liquidity (stop losses from shorts,
    and buy orders from breakout traders).
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance as percentage (0.002 = 0.2%)
        min_count: Minimum number of highs to form an EQH
        window: Window for swing point detection
        
    Returns:
        List of EQH zones, each containing:
        {
            'price': average price of the cluster,
            'zone': (low, high) of the cluster,
            'count': number of highs in cluster,
            'indices': list of indices where highs occurred
        }
    """
    swing_highs, _ = find_swing_points(df, window)
    
    if len(swing_highs) < min_count:
        return []
    
    eqh_zones = []
    used_indices = set()
    
    # Group nearby swing highs
    prices = [(sh['price'], sh['index'], sh['idx']) for sh in swing_highs]
    prices.sort(key=lambda x: x[0])
    
    i = 0
    while i < len(prices):
        if i in used_indices:
            i += 1
            continue
        
        cluster = [prices[i]]
        base_price = prices[i][0]
        
        # Find all highs within tolerance
        for j in range(i + 1, len(prices)):
            if j in used_indices:
                continue
            if abs(prices[j][0] - base_price) / base_price <= tolerance:
                cluster.append(prices[j])
                used_indices.add(j)
        
        if len(cluster) >= min_count:
            cluster_prices = [c[0] for c in cluster]
            avg_price = sum(cluster_prices) / len(cluster_prices)
            
            eqh_zones.append({
                'price': avg_price,
                'zone': (min(cluster_prices), max(cluster_prices)),
                'count': len(cluster),
                'indices': [c[1] for c in cluster],
                'type': 'EQH',
                'liquidity_type': 'SELL_SIDE'
            })
        
        used_indices.add(i)
        i += 1
    
    # Sort by recency (most recent first)
    eqh_zones.sort(key=lambda x: max(x['indices']), reverse=True)
    
    return eqh_zones


def find_equal_lows(
    df: pd.DataFrame,
    tolerance: float = 0.002,
    min_count: int = 2,
    window: int = 5
) -> List[Dict]:
    """
    Identify Equal Lows (EQL) - clusters of similar swing lows.
    
    EQL represents buy-side liquidity (stop losses from longs,
    and sell orders from breakdown traders).
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance as percentage (0.002 = 0.2%)
        min_count: Minimum number of lows to form an EQL
        window: Window for swing point detection
        
    Returns:
        List of EQL zones (same format as EQH)
    """
    _, swing_lows = find_swing_points(df, window)
    
    if len(swing_lows) < min_count:
        return []
    
    eql_zones = []
    used_indices = set()
    
    # Group nearby swing lows
    prices = [(sl['price'], sl['index'], sl['idx']) for sl in swing_lows]
    prices.sort(key=lambda x: x[0])
    
    i = 0
    while i < len(prices):
        if i in used_indices:
            i += 1
            continue
        
        cluster = [prices[i]]
        base_price = prices[i][0]
        
        # Find all lows within tolerance
        for j in range(i + 1, len(prices)):
            if j in used_indices:
                continue
            if abs(prices[j][0] - base_price) / base_price <= tolerance:
                cluster.append(prices[j])
                used_indices.add(j)
        
        if len(cluster) >= min_count:
            cluster_prices = [c[0] for c in cluster]
            avg_price = sum(cluster_prices) / len(cluster_prices)
            
            eql_zones.append({
                'price': avg_price,
                'zone': (min(cluster_prices), max(cluster_prices)),
                'count': len(cluster),
                'indices': [c[1] for c in cluster],
                'type': 'EQL',
                'liquidity_type': 'BUY_SIDE'
            })
        
        used_indices.add(i)
        i += 1
    
    # Sort by recency (most recent first)
    eql_zones.sort(key=lambda x: max(x['indices']), reverse=True)
    
    return eql_zones


def get_liquidity_zones(
    df: pd.DataFrame,
    tolerance: float = 0.002,
    window: int = 5
) -> Dict[str, List[Dict]]:
    """
    Get all liquidity zones (both EQH and EQL).
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance for clustering
        window: Swing point detection window
        
    Returns:
        Dictionary with 'eqh' and 'eql' lists
    """
    return {
        'eqh': find_equal_highs(df, tolerance, min_count=2, window=window),
        'eql': find_equal_lows(df, tolerance, min_count=2, window=window)
    }


def check_liquidity_sweep(
    df: pd.DataFrame,
    liquidity_zones: Dict[str, List[Dict]] = None,
    lookback: int = 10
) -> Dict:
    """
    Check if a liquidity zone has been swept (taken out then reclaimed).
    
    A sweep occurs when:
    1. Price briefly breaks through the liquidity zone
    2. Price quickly reverses back
    
    This is a classic "stop hunt" pattern used by smart money.
    
    Args:
        df: DataFrame with OHLC data
        liquidity_zones: Pre-calculated liquidity zones (optional)
        lookback: Number of candles to check for sweep
        
    Returns:
        Dictionary with sweep status for EQH and EQL
    """
    if liquidity_zones is None:
        liquidity_zones = get_liquidity_zones(df)
    
    recent = df.tail(lookback)
    current_price = df['close'].iloc[-1]
    recent_high = recent['high'].max()
    recent_low = recent['low'].min()
    
    result = {
        'eqh_swept': False,
        'eql_swept': False,
        'eqh_sweep_detail': None,
        'eql_sweep_detail': None,
        'bullish_setup': False,  # EQL swept and price above
        'bearish_setup': False   # EQH swept and price below
    }
    
    # Check EQH sweeps
    for eqh in liquidity_zones['eqh']:
        # Check if recent high exceeded EQH zone
        if recent_high > eqh['zone'][1]:
            # Check if price is now below the zone (reclaimed)
            if current_price < eqh['zone'][0]:
                result['eqh_swept'] = True
                result['bearish_setup'] = True
                result['eqh_sweep_detail'] = {
                    'level': eqh['price'],
                    'sweep_high': recent_high,
                    'current_price': current_price,
                    'reclaimed': True
                }
                break
            elif current_price < eqh['zone'][1]:
                result['eqh_swept'] = True
                result['eqh_sweep_detail'] = {
                    'level': eqh['price'],
                    'sweep_high': recent_high,
                    'current_price': current_price,
                    'reclaimed': False  # Still above, watching for reversal
                }
    
    # Check EQL sweeps
    for eql in liquidity_zones['eql']:
        # Check if recent low exceeded EQL zone
        if recent_low < eql['zone'][0]:
            # Check if price is now above the zone (reclaimed)
            if current_price > eql['zone'][1]:
                result['eql_swept'] = True
                result['bullish_setup'] = True
                result['eql_sweep_detail'] = {
                    'level': eql['price'],
                    'sweep_low': recent_low,
                    'current_price': current_price,
                    'reclaimed': True
                }
                break
            elif current_price > eql['zone'][0]:
                result['eql_swept'] = True
                result['eql_sweep_detail'] = {
                    'level': eql['price'],
                    'sweep_low': recent_low,
                    'current_price': current_price,
                    'reclaimed': False  # Still below, watching for reversal
                }
    
    return result


def get_nearest_liquidity_target(
    df: pd.DataFrame,
    direction: str
) -> Optional[Dict]:
    """
    Get the nearest liquidity target for a trade.
    
    For LONG trades: Look for EQH above current price (target)
    For SHORT trades: Look for EQL below current price (target)
    
    Args:
        df: DataFrame with OHLC data
        direction: "LONG" or "SHORT"
        
    Returns:
        Nearest liquidity target or None
    """
    current_price = df['close'].iloc[-1]
    liquidity_zones = get_liquidity_zones(df)
    
    if direction.upper() == "LONG":
        # For longs, EQH above is a target
        targets = [
            eqh for eqh in liquidity_zones['eqh']
            if eqh['price'] > current_price
        ]
        if targets:
            # Return nearest
            targets.sort(key=lambda x: x['price'])
            return targets[0]
    
    else:  # SHORT
        # For shorts, EQL below is a target
        targets = [
            eql for eql in liquidity_zones['eql']
            if eql['price'] < current_price
        ]
        if targets:
            # Return nearest
            targets.sort(key=lambda x: -x['price'])
            return targets[0]
    
    return None


# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2
# (from __future__ imports intentionally omitted: appended blocks
# cannot contain __future__ imports. Use modern `X | None` syntax
# only where the file has no pre-existing `from __future__` line.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class LiquidityFeature:
    """Equal highs/lows + sweep detection as point-in-time feature."""

    name: str = "liquidity"
    version: str = "1.0.0"
    required_lookback: int = 100

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {"zones": [], "sweep": None, "nearest_target": None}
        zones = get_liquidity_zones(slice_df)
        sweep = check_liquidity_sweep(slice_df, zones)
        # get_nearest_liquidity_target(df, direction) — fetch both directions
        nearest_long = get_nearest_liquidity_target(slice_df, "LONG")
        nearest_short = get_nearest_liquidity_target(slice_df, "SHORT")
        nearest = nearest_long or nearest_short
        return {"zones": zones, "sweep": sweep, "nearest_target": nearest}
