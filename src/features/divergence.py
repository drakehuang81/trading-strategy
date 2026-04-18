"""
RSI/MACD Divergence Detection Module

Detects divergences between price and indicators.
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


def find_local_extrema(series: pd.Series, window: int = 5):
    """Find local maxima and minima in a series."""
    values = series.values
    indices = series.index.tolist()
    maxima, minima = [], []
    
    for i in range(window, len(values) - window):
        is_max = all(values[i] > values[i-j] and values[i] > values[i+j] for j in range(1, window+1))
        is_min = all(values[i] < values[i-j] and values[i] < values[i+j] for j in range(1, window+1))
        
        if is_max:
            maxima.append({'index': indices[i], 'value': values[i], 'idx': i})
        if is_min:
            minima.append({'index': indices[i], 'value': values[i], 'idx': i})
    
    return maxima, minima


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 30, window: int = 5) -> Optional[Dict]:
    """Detect RSI divergence."""
    if 'rsi' not in df.columns:
        return None
    
    recent = df.tail(lookback)
    price_maxima, price_minima = find_local_extrema(recent['close'], window)
    rsi_maxima, rsi_minima = find_local_extrema(recent['rsi'], window)
    
    # Bullish divergence check
    if len(price_minima) >= 2 and len(rsi_minima) >= 2:
        p1, p2 = price_minima[-2], price_minima[-1]
        r1 = next((r for r in rsi_minima if abs(r['idx'] - p1['idx']) <= window), None)
        r2 = next((r for r in rsi_minima if abs(r['idx'] - p2['idx']) <= window), None)
        
        if r1 and r2:
            if p2['value'] < p1['value'] and r2['value'] > r1['value']:
                return {'type': 'BULLISH_REGULAR', 'signal': 'REVERSAL', 'direction': 'LONG',
                        'description': f"Price LL + RSI HL"}
            if p2['value'] > p1['value'] and r2['value'] < r1['value']:
                return {'type': 'BULLISH_HIDDEN', 'signal': 'CONTINUATION', 'direction': 'LONG',
                        'description': f"Price HL + RSI LL"}
    
    # Bearish divergence check
    if len(price_maxima) >= 2 and len(rsi_maxima) >= 2:
        p1, p2 = price_maxima[-2], price_maxima[-1]
        r1 = next((r for r in rsi_maxima if abs(r['idx'] - p1['idx']) <= window), None)
        r2 = next((r for r in rsi_maxima if abs(r['idx'] - p2['idx']) <= window), None)
        
        if r1 and r2:
            if p2['value'] > p1['value'] and r2['value'] < r1['value']:
                return {'type': 'BEARISH_REGULAR', 'signal': 'REVERSAL', 'direction': 'SHORT',
                        'description': f"Price HH + RSI LH"}
            if p2['value'] < p1['value'] and r2['value'] > r1['value']:
                return {'type': 'BEARISH_HIDDEN', 'signal': 'CONTINUATION', 'direction': 'SHORT',
                        'description': f"Price LH + RSI HH"}
    
    return None


def detect_macd_divergence(df: pd.DataFrame, lookback: int = 30, window: int = 5) -> Optional[Dict]:
    """Detect MACD histogram divergence."""
    if 'macd_hist' not in df.columns:
        return None
    
    recent = df.tail(lookback)
    price_maxima, price_minima = find_local_extrema(recent['close'], window)
    macd_maxima, macd_minima = find_local_extrema(recent['macd_hist'], window)
    
    # Bullish check
    if len(price_minima) >= 2 and len(macd_minima) >= 2:
        p1, p2 = price_minima[-2], price_minima[-1]
        m1 = next((m for m in macd_minima if abs(m['idx'] - p1['idx']) <= window), None)
        m2 = next((m for m in macd_minima if abs(m['idx'] - p2['idx']) <= window), None)
        
        if m1 and m2:
            if p2['value'] < p1['value'] and m2['value'] > m1['value']:
                return {'type': 'BULLISH_REGULAR', 'signal': 'REVERSAL', 'direction': 'LONG'}
            if p2['value'] > p1['value'] and m2['value'] < m1['value']:
                return {'type': 'BULLISH_HIDDEN', 'signal': 'CONTINUATION', 'direction': 'LONG'}
    
    # Bearish check
    if len(price_maxima) >= 2 and len(macd_maxima) >= 2:
        p1, p2 = price_maxima[-2], price_maxima[-1]
        m1 = next((m for m in macd_maxima if abs(m['idx'] - p1['idx']) <= window), None)
        m2 = next((m for m in macd_maxima if abs(m['idx'] - p2['idx']) <= window), None)
        
        if m1 and m2:
            if p2['value'] > p1['value'] and m2['value'] < m1['value']:
                return {'type': 'BEARISH_REGULAR', 'signal': 'REVERSAL', 'direction': 'SHORT'}
            if p2['value'] < p1['value'] and m2['value'] > m1['value']:
                return {'type': 'BEARISH_HIDDEN', 'signal': 'CONTINUATION', 'direction': 'SHORT'}
    
    return None


def get_divergence_summary(df: pd.DataFrame) -> Dict:
    """Get summary of all divergences."""
    rsi_div = detect_rsi_divergence(df)
    macd_div = detect_macd_divergence(df)
    
    result = {
        'has_divergence': bool(rsi_div or macd_div),
        'rsi_divergence': rsi_div,
        'macd_divergence': macd_div,
        'consensus_direction': None
    }
    
    if rsi_div and macd_div and rsi_div['direction'] == macd_div['direction']:
        result['consensus_direction'] = rsi_div['direction']
    elif rsi_div:
        result['consensus_direction'] = rsi_div['direction']
    elif macd_div:
        result['consensus_direction'] = macd_div['direction']

    return result


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
class DivergenceFeature:
    """RSI + MACD divergence summary at as_of."""

    name: str = "divergence"
    version: str = "1.0.0"
    required_lookback: int = 60

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {"rsi": None, "macd": None, "summary": {}}
        rsi_div = detect_rsi_divergence(slice_df)
        macd_div = detect_macd_divergence(slice_df)
        summary = get_divergence_summary(slice_df)
        return {"rsi": rsi_div, "macd": macd_div, "summary": summary}
