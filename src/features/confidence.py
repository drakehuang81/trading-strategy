"""
8-Factor Confidence Scoring System

Based on crypto_trading_skill SKILL.md Line 600-637.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo


CONFLUENCE_FACTORS = [
    "HTF_BIAS_ALIGNED",
    "SMC_POI_PRESENT",
    "FIB_CONFLUENCE",
    "LIQUIDITY_SWEPT",
    "RSI_CONFLUENCE",
    "MACD_CONFLUENCE",
    "AMD_SESSION_ALIGNED",
    "ASIA_RANGE_SWEEP"
]

SCORE_RATINGS = {
    8: ("A+", "Maximum confluence. Full position."),
    7: ("A", "Excellent. Full position."),
    6: ("B+", "Strong. Standard position (1% risk)."),
    5: ("B", "Decent. Standard position."),
    4: ("C", "Marginal. Reduce position (0.5% risk)."),
    3: ("D", "Weak. Consider skipping."),
}


def check_htf_bias(htf_bias: str, direction: str) -> Dict[str, Any]:
    """Check if HTF bias aligns with trade direction."""
    aligned = (
        (htf_bias == "BULLISH" and direction == "LONG") or
        (htf_bias == "BEARISH" and direction == "SHORT")
    )
    return {
        'factor': 'HTF_BIAS_ALIGNED',
        'status': aligned,
        'score': 1 if aligned else 0,
        'detail': f"HTF: {htf_bias}, Direction: {direction}"
    }


def check_smc_poi(poi: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Check if a valid SMC POI exists."""
    has_poi = poi is not None and poi.get('type') in ['ORDER_BLOCK', 'FVG', 'SWING_LOW', 'SWING_HIGH']
    return {
        'factor': 'SMC_POI_PRESENT',
        'status': has_poi,
        'score': 1 if has_poi else 0,
        'detail': poi.get('type', 'None') if poi else 'None'
    }


def check_fib_confluence(fib_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Check if price is in OTE zone or at Fib level."""
    is_ote = fib_analysis.get('confluence', {}).get('is_ote', False)
    is_at_fib = fib_analysis.get('confluence', {}).get('is_at_fib', False)
    has_confluence = is_ote or is_at_fib
    return {
        'factor': 'FIB_CONFLUENCE',
        'status': has_confluence,
        'score': 1 if has_confluence else 0,
        'detail': f"OTE: {is_ote}, At Fib: {is_at_fib}"
    }


def check_liquidity_swept(liquidity_sweep: Dict[str, Any], direction: str) -> Dict[str, Any]:
    """Check if liquidity has been swept."""
    if direction == "LONG":
        swept = liquidity_sweep.get('eql_swept', False)
    else:
        swept = liquidity_sweep.get('eqh_swept', False)
    return {
        'factor': 'LIQUIDITY_SWEPT',
        'status': swept,
        'score': 1 if swept else 0,
        'detail': f"EQL Swept: {liquidity_sweep.get('eql_swept')}, EQH Swept: {liquidity_sweep.get('eqh_swept')}"
    }


def check_rsi_confluence(rsi: float, direction: str, divergence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check RSI conditions."""
    has_confluence = False
    detail = f"RSI: {rsi:.1f}"

    if direction == "LONG":
        if rsi < 40 or (divergence and 'BULLISH' in divergence.get('type', '')):
            has_confluence = True
            detail += " (Oversold/Divergence)"
    else:
        if rsi > 60 or (divergence and 'BEARISH' in divergence.get('type', '')):
            has_confluence = True
            detail += " (Overbought/Divergence)"

    return {
        'factor': 'RSI_CONFLUENCE',
        'status': has_confluence,
        'score': 1 if has_confluence else 0,
        'detail': detail
    }


def check_macd_confluence(macd_hist: float, prev_macd_hist: float, direction: str, divergence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check MACD conditions."""
    has_confluence = False

    if direction == "LONG":
        if macd_hist > prev_macd_hist or (divergence and 'BULLISH' in divergence.get('type', '')):
            has_confluence = True
    else:
        if macd_hist < prev_macd_hist or (divergence and 'BEARISH' in divergence.get('type', '')):
            has_confluence = True

    return {
        'factor': 'MACD_CONFLUENCE',
        'status': has_confluence,
        'score': 1 if has_confluence else 0,
        'detail': f"Hist: {macd_hist:.4f}, Prev: {prev_macd_hist:.4f}"
    }


def check_amd_session() -> Dict[str, Any]:
    """Check if in optimal trading session."""
    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.now(taipei_tz)
    hour = now.hour

    # London (15-00) or NY (21-06) or Overlap (21-00)
    in_killzone = (15 <= hour < 24) or (0 <= hour < 6)
    session = "London" if 15 <= hour < 21 else ("NY" if 21 <= hour or hour < 6 else "Asia")

    return {
        'factor': 'AMD_SESSION_ALIGNED',
        'status': in_killzone,
        'score': 1 if in_killzone else 0,
        'detail': f"Session: {session}, Hour: {hour}"
    }


def check_asia_range_sweep(current_price: float, ash: float, asl: float, direction: str) -> Dict[str, Any]:
    """Check if Asia range has been swept."""
    swept = False

    if ash and asl:
        if direction == "LONG" and current_price > asl:
            # Price was below ASL and came back up
            swept = True
        elif direction == "SHORT" and current_price < ash:
            # Price was above ASH and came back down
            swept = True

    return {
        'factor': 'ASIA_RANGE_SWEEP',
        'status': swept,
        'score': 1 if swept else 0,
        'detail': f"ASH: {ash}, ASL: {asl}, Price: {current_price}"
    }


def calculate_confidence_score(
    direction: str,
    htf_bias: str,
    poi: Optional[Dict[str, Any]],
    fib_analysis: Dict[str, Any],
    liquidity_sweep: Dict[str, Any],
    rsi: float,
    macd_hist: float,
    prev_macd_hist: float,
    ash: Optional[float] = None,
    asl: Optional[float] = None,
    current_price: Optional[float] = None,
    rsi_divergence: Optional[Dict[str, Any]] = None,
    macd_divergence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate the 8-factor confidence score."""
    factors = []

    factors.append(check_htf_bias(htf_bias, direction))
    factors.append(check_smc_poi(poi))
    factors.append(check_fib_confluence(fib_analysis))
    factors.append(check_liquidity_swept(liquidity_sweep, direction))
    factors.append(check_rsi_confluence(rsi, direction, rsi_divergence))
    factors.append(check_macd_confluence(macd_hist, prev_macd_hist, direction, macd_divergence))
    factors.append(check_amd_session())
    factors.append(check_asia_range_sweep(current_price or 0, ash or 0, asl or 0, direction))

    total_score = sum(f['score'] for f in factors)
    rating, description = SCORE_RATINGS.get(total_score, ("F", "Insufficient confluence. DO NOT TRADE."))

    return {
        'score': total_score,
        'max_score': 8,
        'rating': rating,
        'description': description,
        'factors': factors,
        'recommendation': get_position_recommendation(total_score)
    }


def get_position_recommendation(score: int) -> Dict[str, Any]:
    """Get position sizing recommendation based on score."""
    if score >= 7:
        return {'action': 'TRADE', 'risk': '1%', 'leverage': 10}
    elif score >= 5:
        return {'action': 'TRADE', 'risk': '1%', 'leverage': 7}
    elif score >= 4:
        return {'action': 'REDUCE', 'risk': '0.5%', 'leverage': 5}
    else:
        return {'action': 'SKIP', 'risk': '0%', 'leverage': 0}


def format_confidence_score(result: Dict[str, Any]) -> str:
    """Format confidence score for display."""
    stars = "⭐" * min(result['score'], 5)
    lines = [
        f"Confidence Score: {result['score']}/{result['max_score']} {stars} ({result['rating']})",
        f"{result['description']}",
        "",
        "Factors:"
    ]
    
    for f in result['factors']:
        status = "✅" if f['status'] else "❌"
        lines.append(f"  {status} {f['factor']}: {f['detail']}")
    
    rec = result['recommendation']
    lines.append(f"\nRecommendation: {rec['action']} | Risk: {rec['risk']} | Max Leverage: {rec['leverage']}x")
    
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Feature Protocol wrapper — spec §4.2 (un-calibrated, see §12)
# (no `from __future__` — appended blocks cannot contain it.)
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


def _safe_rsi(slice_df: pd.DataFrame) -> float:
    """Approximate RSI with NaN-safe fallback (50.0 = neutral)."""
    val = slice_df["close"].pct_change().rolling(14).mean().iloc[-1]
    return float(val) * 100 if pd.notna(val) else 50.0


@dataclass
class ConfidenceFeature:
    """Packages the 8-factor confidence score. WARNING (spec §12):
    this score is not calibrated; do not use as a probability."""

    direction: str = "long"   # "long" | "short" — caller supplies
    name: str = "confidence"
    version: str = "1.0.0"
    required_lookback: int = 100

    def compute(self, df: pd.DataFrame, as_of: datetime) -> dict[str, Any]:
        slice_df = df[df.index <= as_of]
        if len(slice_df) < self.required_lookback:
            return {"score": 0, "breakdown": {}, "recommendation": None}
        result = calculate_confidence_score(
            direction=self.direction.upper(),
            htf_bias="NEUTRAL",
            poi=None,
            fib_analysis={"confluence": {"is_ote": False, "is_at_fib": False}},
            liquidity_sweep={"eql_swept": False, "eqh_swept": False},
            rsi=_safe_rsi(slice_df),
            macd_hist=0.0,
            prev_macd_hist=0.0,
            ash=None,
            asl=None,
            current_price=float(slice_df["close"].iloc[-1]),
            rsi_divergence=None,
            macd_divergence=None,
        )
        return result
