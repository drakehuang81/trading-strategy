"""
Trade Setup Calculation Module

Calculates Entry, Stop Loss, and Take Profit levels.
"""

from typing import Dict, Tuple, Optional


def calculate_trade_setup(
    direction: str,
    entry_price: float,
    swing_low: float,
    swing_high: float,
    account_balance: float = 10000,
    risk_percent: float = 0.01,
    leverage: int = 5
) -> Dict:
    """
    Calculate complete trade setup with Entry, SL, and TPs.
    
    Args:
        direction: "LONG" or "SHORT"
        entry_price: Entry price
        swing_low: Recent swing low (for Long SL)
        swing_high: Recent swing high (for Short SL)
        account_balance: Account balance in USD
        risk_percent: Risk per trade (0.01 = 1%)
        leverage: Leverage to use
    """
    if direction.upper() == "LONG":
        sl = swing_low * 0.998  # SL below swing low
        sl_distance = entry_price - sl
        tp1 = entry_price + sl_distance * 2   # R:R 1:2
        tp2 = entry_price + sl_distance * 3   # R:R 1:3
        tp3 = entry_price + sl_distance * 5   # R:R 1:5
    else:
        sl = swing_high * 1.002  # SL above swing high
        sl_distance = sl - entry_price
        tp1 = entry_price - sl_distance * 2
        tp2 = entry_price - sl_distance * 3
        tp3 = entry_price - sl_distance * 5
    
    risk_usd = account_balance * risk_percent
    position_size_usd = risk_usd / (sl_distance / entry_price) if sl_distance > 0 else 0
    position_size_coin = position_size_usd / entry_price if entry_price > 0 else 0
    
    sl_pct = abs(sl - entry_price) / entry_price * 100
    tp1_pct = abs(tp1 - entry_price) / entry_price * 100
    tp2_pct = abs(tp2 - entry_price) / entry_price * 100
    tp3_pct = abs(tp3 - entry_price) / entry_price * 100
    
    return {
        'direction': direction.upper(),
        'entry': entry_price,
        'stop_loss': sl,
        'stop_loss_pct': sl_pct,
        'tp1': {'price': tp1, 'rr': 2.0, 'pct': tp1_pct},
        'tp2': {'price': tp2, 'rr': 3.0, 'pct': tp2_pct},
        'tp3': {'price': tp3, 'rr': 5.0, 'pct': tp3_pct},
        'position_size_usd': position_size_usd,
        'position_size_coin': position_size_coin,
        'risk_usd': risk_usd,
        'leverage': leverage
    }


def format_trade_setup(setup: Dict, symbol: str = "COIN/USDT") -> str:
    """Format trade setup for display."""
    direction = setup['direction']
    emoji = "🟢" if direction == "LONG" else "🔴"
    
    lines = [
        f"{emoji} {direction} SETUP: {symbol}",
        "━" * 35,
        f"📍 Entry:        ${setup['entry']:,.2f}",
        f"🛑 Stop Loss:    ${setup['stop_loss']:,.2f} (-{setup['stop_loss_pct']:.2f}%)",
        f"🎯 TP1:          ${setup['tp1']['price']:,.2f} (+{setup['tp1']['pct']:.2f}%) | R:R 1:{setup['tp1']['rr']}",
        f"🎯 TP2:          ${setup['tp2']['price']:,.2f} (+{setup['tp2']['pct']:.2f}%) | R:R 1:{setup['tp2']['rr']}",
        f"🎯 TP3:          ${setup['tp3']['price']:,.2f} (+{setup['tp3']['pct']:.2f}%) | R:R 1:{setup['tp3']['rr']}",
        "━" * 35,
        f"💰 Position:     ${setup['position_size_usd']:,.2f}",
        f"📊 Risk:         ${setup['risk_usd']:,.2f}",
        f"⚡ Leverage:     {setup['leverage']}x",
        "━" * 35
    ]
    return "\n".join(lines)
