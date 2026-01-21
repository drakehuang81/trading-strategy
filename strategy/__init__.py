"""
Strategy Analysis Package

This package contains modules for technical analysis based on the
crypto_trading_skill strategy rules.

Modules:
- fibonacci: Fibonacci retracement levels and OTE zone
- smc: Smart Money Concepts (BOS, CHoCH, OB, FVG)
- liquidity: EQH/EQL liquidity pool identification
- divergence: RSI/MACD divergence detection
- trade_setup: Entry/SL/TP calculation
- confidence: 8-factor confidence scoring system
"""

from strategy.fibonacci import (
    calculate_fib_levels,
    get_ote_zone,
    get_premium_discount_zone,
    identify_fib_confluence
)

from strategy.smc import (
    find_swing_points,
    identify_structure,
    find_order_blocks,
    find_fvg
)

from strategy.liquidity import (
    find_equal_highs,
    find_equal_lows,
    get_liquidity_zones,
    check_liquidity_sweep
)

from strategy.divergence import (
    detect_rsi_divergence,
    detect_macd_divergence
)

from strategy.trade_setup import (
    calculate_trade_setup,
    format_trade_setup
)

from strategy.confidence import (
    calculate_confidence_score,
    get_position_recommendation
)

__all__ = [
    # Fibonacci
    'calculate_fib_levels',
    'get_ote_zone',
    'get_premium_discount_zone',
    'identify_fib_confluence',
    # SMC
    'find_swing_points',
    'identify_structure',
    'find_order_blocks',
    'find_fvg',
    # Liquidity
    'find_equal_highs',
    'find_equal_lows',
    'get_liquidity_zones',
    'check_liquidity_sweep',
    # Divergence
    'detect_rsi_divergence',
    'detect_macd_divergence',
    # Trade Setup
    'calculate_trade_setup',
    'format_trade_setup',
    # Confidence
    'calculate_confidence_score',
    'get_position_recommendation',
]
