"""
Fibonacci Retracement Module

Calculates Fibonacci retracement levels and identifies OTE (Optimal Trade Entry)
zones based on the crypto_trading_skill strategy.

Key Levels:
- 0.236: Shallow retracement
- 0.382: Moderate retracement
- 0.5:   Equilibrium (Premium/Discount divider)
- 0.618: Golden Ratio
- 0.705: ICT OTE level
- 0.786: Deep retracement
"""

from typing import Dict, Tuple, Optional
import pandas as pd

# Fibonacci level definitions
FIB_LEVELS = {
    '0.236': 0.236,
    '0.382': 0.382,
    '0.5': 0.5,
    '0.618': 0.618,
    '0.705': 0.705,
    '0.786': 0.786,
}

# OTE Zone boundaries
OTE_UPPER = 0.618
OTE_LOWER = 0.786


def calculate_fib_levels(
    swing_high: float,
    swing_low: float,
    direction: str = "bullish"
) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels.
    
    For Bullish setup (looking for longs):
        - Draw from swing_low (1.0) to swing_high (0.0)
        - Discount zone is below 0.5
        
    For Bearish setup (looking for shorts):
        - Draw from swing_high (1.0) to swing_low (0.0)
        - Premium zone is above 0.5
    
    Args:
        swing_high: The swing high price
        swing_low: The swing low price
        direction: "bullish" or "bearish"
        
    Returns:
        Dictionary of Fibonacci levels with their prices
    """
    diff = swing_high - swing_low
    levels = {}
    
    for name, ratio in FIB_LEVELS.items():
        if direction.lower() == "bullish":
            # For bullish, retracement goes down from high
            levels[name] = swing_high - (diff * ratio)
        else:
            # For bearish, retracement goes up from low
            levels[name] = swing_low + (diff * ratio)
    
    # Add swing points as reference
    levels['0.0'] = swing_high if direction.lower() == "bullish" else swing_low
    levels['1.0'] = swing_low if direction.lower() == "bullish" else swing_high
    levels['swing_high'] = swing_high
    levels['swing_low'] = swing_low
    
    return levels


def get_ote_zone(levels: Dict[str, float]) -> Tuple[float, float]:
    """
    Get the Optimal Trade Entry (OTE) zone.
    
    The OTE zone is between 0.618 and 0.786 Fibonacci levels.
    This is where ICT (Inner Circle Trader) recommends looking for entries.
    
    Args:
        levels: Dictionary of Fibonacci levels
        
    Returns:
        Tuple of (lower_bound, upper_bound) for OTE zone
    """
    return (levels['0.786'], levels['0.618'])


def get_premium_discount_zone(
    current_price: float,
    levels: Dict[str, float]
) -> Tuple[str, float]:
    """
    Determine if price is in Premium or Discount zone.
    
    - Discount Zone (0.5 - 1.0): Below equilibrium, good for longs
    - Premium Zone (0.0 - 0.5): Above equilibrium, good for shorts
    
    Args:
        current_price: Current market price
        levels: Dictionary of Fibonacci levels
        
    Returns:
        Tuple of (zone_name, distance_from_equilibrium_pct)
    """
    equilibrium = levels['0.5']
    swing_high = levels['swing_high']
    swing_low = levels['swing_low']
    
    # Calculate position relative to equilibrium
    if current_price < equilibrium:
        zone = "DISCOUNT"
        # How deep in discount (closer to 1.0 = deeper discount)
        distance_pct = (equilibrium - current_price) / (equilibrium - swing_low) * 100
    else:
        zone = "PREMIUM"
        # How high in premium (closer to 0.0 = higher premium)
        distance_pct = (current_price - equilibrium) / (swing_high - equilibrium) * 100
    
    return zone, min(distance_pct, 100.0)


def identify_fib_confluence(
    current_price: float,
    levels: Dict[str, float],
    tolerance: float = 0.003
) -> Dict:
    """
    Check if current price is at a significant Fibonacci level.
    
    Args:
        current_price: Current market price
        levels: Dictionary of Fibonacci levels
        tolerance: Price tolerance as percentage (0.003 = 0.3%)
        
    Returns:
        Dictionary with:
        - is_at_fib: Boolean
        - level: The Fibonacci level name (if any)
        - distance_pct: Distance from the level as percentage
        - is_ote: Boolean if price is in OTE zone
    """
    result = {
        'is_at_fib': False,
        'level': None,
        'level_price': None,
        'distance_pct': None,
        'is_ote': False,
        'zone': None
    }
    
    # Check each Fib level
    for level_name, level_price in levels.items():
        if level_name in ['swing_high', 'swing_low', '0.0', '1.0']:
            continue
            
        distance = abs(current_price - level_price) / level_price
        
        if distance <= tolerance:
            result['is_at_fib'] = True
            result['level'] = level_name
            result['level_price'] = level_price
            result['distance_pct'] = distance * 100
            break
    
    # Check if in OTE zone
    ote_lower, ote_upper = get_ote_zone(levels)
    if min(ote_lower, ote_upper) <= current_price <= max(ote_lower, ote_upper):
        result['is_ote'] = True
    
    # Get zone
    zone, zone_depth = get_premium_discount_zone(current_price, levels)
    result['zone'] = zone
    result['zone_depth'] = zone_depth
    
    return result


def calculate_fib_from_df(
    df: pd.DataFrame,
    lookback: int = 50,
    direction: str = "auto"
) -> Dict:
    """
    Calculate Fibonacci levels from a DataFrame.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        lookback: Number of candles to consider for swing points
        direction: "bullish", "bearish", or "auto" (detect from trend)
        
    Returns:
        Dictionary containing all Fibonacci analysis
    """
    recent = df.tail(lookback)
    
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()
    current_price = df['close'].iloc[-1]
    
    # Auto-detect direction based on price position
    if direction == "auto":
        mid_point = (swing_high + swing_low) / 2
        direction = "bullish" if current_price < mid_point else "bearish"
    
    levels = calculate_fib_levels(swing_high, swing_low, direction)
    ote_zone = get_ote_zone(levels)
    zone, zone_depth = get_premium_discount_zone(current_price, levels)
    confluence = identify_fib_confluence(current_price, levels)
    
    return {
        'direction': direction,
        'levels': levels,
        'ote_zone': ote_zone,
        'zone': zone,
        'zone_depth': zone_depth,
        'current_price': current_price,
        'confluence': confluence
    }


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
class FibFeature:
    """Fibonacci levels + OTE zone + confluence as a point-in-time Feature."""

    name: str = "fib"
    version: str = "1.0.0"
    required_lookback: int = 50

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {"levels": {}, "ote": None, "pd_zone": None}
        fib_result = calculate_fib_from_df(slice_df)
        if not fib_result or not fib_result.get("levels"):
            return {"levels": {}, "ote": None, "pd_zone": None}
        levels = fib_result["levels"]
        ote = get_ote_zone(levels)
        current_price = float(slice_df["close"].iloc[-1])
        pd_zone = get_premium_discount_zone(current_price, levels)
        return {"levels": levels, "ote": ote, "pd_zone": pd_zone}
