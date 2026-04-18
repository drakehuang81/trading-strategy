"""
Funding Rate Filter

Independent risk filter for adjusting position sizing based on market sentiment.
NOT part of the 9-factor confidence scoring system.

Funding Rate thresholds:
- NORMAL:   -0.05% to +0.05%  → No adjustment
- HIGH:     ±0.05% to ±0.1%   → Reduce position (0.5% risk)
- EXTREME:  Beyond ±0.1%      → Skip trade
"""

from typing import Dict, Optional
from binance.client import Client
import os


# Threshold definitions (as percentages, e.g., 0.05 = 0.05%)
FUNDING_RATE_THRESHOLDS = {
    'NORMAL_LOW': -0.05,
    'NORMAL_HIGH': 0.05,
    'HIGH_LOW': -0.1,
    'HIGH_HIGH': 0.1,
}

FUNDING_RATE_STATUS = {
    'NORMAL': {'emoji': '✅', 'action': 'PROCEED', 'risk_multiplier': 1.0},
    'HIGH': {'emoji': '⚠️', 'action': 'REDUCE', 'risk_multiplier': 0.5},
    'EXTREME': {'emoji': '🛑', 'action': 'SKIP', 'risk_multiplier': 0.0},
}


def get_binance_futures_client() -> Client:
    """Initialize Binance client for futures API."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if api_key and api_secret:
        return Client(api_key, api_secret)
    return Client()


def fetch_funding_rate(symbol: str = "BTCUSDT") -> Dict:
    """
    Fetch current funding rate from Binance Futures API.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
    
    Returns:
        Dict with current rate, next funding time, and status
    """
    try:
        client = get_binance_futures_client()
        
        # Get funding rate info
        funding_info = client.futures_funding_rate(symbol=symbol, limit=1)
        
        if funding_info and len(funding_info) > 0:
            current_rate = float(funding_info[0]['fundingRate']) * 100  # Convert to percentage
            funding_time = funding_info[0]['fundingTime']
            
            return {
                'symbol': symbol,
                'rate': current_rate,
                'rate_display': f"{current_rate:+.4f}%",
                'funding_time': funding_time,
                'available': True
            }
        
        return {'symbol': symbol, 'rate': None, 'available': False}
        
    except Exception as e:
        return {
            'symbol': symbol,
            'rate': None,
            'available': False,
            'error': str(e)
        }


def evaluate_funding_rate(rate: Optional[float]) -> Dict:
    """
    Evaluate funding rate and return risk classification.
    
    Args:
        rate: Funding rate as percentage (e.g., 0.05 for 0.05%)
    
    Returns:
        Dict with level, status info, and recommendation
    """
    if rate is None:
        return {
            'level': 'UNKNOWN',
            'emoji': '❓',
            'action': 'PROCEED',
            'risk_multiplier': 1.0,
            'reason': 'Funding rate data unavailable'
        }
    
    # Determine level
    if FUNDING_RATE_THRESHOLDS['NORMAL_LOW'] <= rate <= FUNDING_RATE_THRESHOLDS['NORMAL_HIGH']:
        level = 'NORMAL'
        reason = '資金費率正常，市場情緒中性'
    elif rate > FUNDING_RATE_THRESHOLDS['HIGH_HIGH']:
        level = 'EXTREME'
        reason = f'資金費率極高 ({rate:+.4f}%)，多頭過度擁擠，小心軋空'
    elif rate < FUNDING_RATE_THRESHOLDS['HIGH_LOW']:
        level = 'EXTREME'
        reason = f'資金費率極低 ({rate:+.4f}%)，空頭過度擁擠，小心軋多'
    elif rate > FUNDING_RATE_THRESHOLDS['NORMAL_HIGH']:
        level = 'HIGH'
        reason = f'資金費率偏高 ({rate:+.4f}%)，多頭偏多，做多需謹慎'
    else:  # rate < FUNDING_RATE_THRESHOLDS['NORMAL_LOW']
        level = 'HIGH'
        reason = f'資金費率偏低 ({rate:+.4f}%)，空頭偏多，做空需謹慎'
    
    status = FUNDING_RATE_STATUS[level]
    
    return {
        'level': level,
        'emoji': status['emoji'],
        'action': status['action'],
        'risk_multiplier': status['risk_multiplier'],
        'reason': reason
    }


def get_position_adjustment(rate: Optional[float], direction: str) -> Dict:
    """
    Get position sizing adjustment based on funding rate and trade direction.
    
    Args:
        rate: Funding rate as percentage
        direction: 'LONG' or 'SHORT'
    
    Returns:
        Dict with adjustment recommendation
    """
    if rate is None:
        return {
            'adjust': False,
            'risk_multiplier': 1.0,
            'warning': None
        }
    
    evaluation = evaluate_funding_rate(rate)
    
    # Check for direction-specific warnings
    warning = None
    risk_multiplier = evaluation['risk_multiplier']
    
    if direction == 'LONG' and rate > FUNDING_RATE_THRESHOLDS['NORMAL_HIGH']:
        warning = f'⚠️ 做多警告：多頭擁擠 (FR: {rate:+.4f}%)，建議減半倉位'
        if risk_multiplier > 0.5:
            risk_multiplier = 0.5
    elif direction == 'SHORT' and rate < FUNDING_RATE_THRESHOLDS['NORMAL_LOW']:
        warning = f'⚠️ 做空警告：空頭擁擠 (FR: {rate:+.4f}%)，建議減半倉位'
        if risk_multiplier > 0.5:
            risk_multiplier = 0.5
    
    # If direction aligns with crowded side in extreme case, skip
    if evaluation['level'] == 'EXTREME':
        if (direction == 'LONG' and rate > 0) or (direction == 'SHORT' and rate < 0):
            warning = f'🛑 停止交易：{direction} 方向與極端擁擠方向一致'
            risk_multiplier = 0.0
    
    return {
        'adjust': risk_multiplier < 1.0,
        'risk_multiplier': risk_multiplier,
        'warning': warning,
        'evaluation': evaluation
    }


def format_funding_rate_filter(funding_data: Dict, evaluation: Dict = None) -> str:
    """
    Format funding rate filter for display in report.
    
    Args:
        funding_data: Result from fetch_funding_rate()
        evaluation: Optional pre-computed evaluation
    
    Returns:
        Formatted string for display
    """
    if not funding_data.get('available'):
        return """
🚦 RISK FILTER: Funding Rate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Status:          ❓ N/A (Futures data unavailable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    if evaluation is None:
        evaluation = evaluate_funding_rate(funding_data['rate'])
    
    return f"""
🚦 RISK FILTER: Funding Rate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Current Rate:    {funding_data['rate_display']}
   Status:          {evaluation['emoji']} {evaluation['level']}
   Action:          {evaluation['action']}
   Risk Multiplier: {evaluation['risk_multiplier']:.1f}x
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   {evaluation['reason']}
"""


if __name__ == "__main__":
    # Test fetching
    print("Testing Funding Rate module...")
    
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        data = fetch_funding_rate(symbol)
        if data['available']:
            print(f"\n{symbol}: {data['rate_display']}")
            evaluation = evaluate_funding_rate(data['rate'])
            print(format_funding_rate_filter(data, evaluation))
        else:
            print(f"\n{symbol}: Data unavailable")


# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2 + §4.5 (historical parquet)
# (no `from __future__` — appended blocks cannot contain it.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_FUNDING_DIR = Path("data/funding")


@dataclass
class FundingFeature:
    """Reads funding rate from data/funding/<symbol>.parquet (historical,
    no network). Returns empty dict if no parquet exists — Plan 2 wires
    the ingestion job. Network path lives in FundingRateDataSource (also
    Plan 2); this Feature is point-in-time only."""

    symbol: str = "ETHUSDT"
    name: str = "funding"
    version: str = "1.0.0"
    required_lookback: int = 1

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        path = _FUNDING_DIR / f"{self.symbol}.parquet"
        if not path.exists():
            return {"rate": None, "evaluation": None, "position_adj": None}
        funding = pd.read_parquet(path)
        funding = funding[funding["ts"] <= as_of]
        if funding.empty:
            return {"rate": None, "evaluation": None, "position_adj": None}
        rate = float(funding.iloc[-1]["rate"])
        evaluation = evaluate_funding_rate(rate)
        position_adj = get_position_adjustment(rate, direction="LONG")
        return {"rate": rate, "evaluation": evaluation, "position_adj": position_adj}
