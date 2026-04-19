"""
Unit tests for strategy/funding_rate.py
"""
import pytest
from unittest.mock import patch, MagicMock
from features.funding_rate import (
    evaluate_funding_rate,
    get_position_adjustment,
    format_funding_rate_filter,
    FUNDING_RATE_THRESHOLDS,
    FUNDING_RATE_STATUS
)


class TestEvaluateFundingRate:
    """Tests for evaluate_funding_rate function."""
    
    def test_normal_positive(self):
        """Test normal positive funding rate (0.03%)."""
        result = evaluate_funding_rate(0.03)
        assert result['level'] == 'NORMAL'
        assert result['action'] == 'PROCEED'
        assert result['risk_multiplier'] == 1.0
    
    def test_normal_negative(self):
        """Test normal negative funding rate (-0.02%)."""
        result = evaluate_funding_rate(-0.02)
        assert result['level'] == 'NORMAL'
        assert result['action'] == 'PROCEED'
        assert result['risk_multiplier'] == 1.0
    
    def test_normal_zero(self):
        """Test zero funding rate."""
        result = evaluate_funding_rate(0.0)
        assert result['level'] == 'NORMAL'
        assert result['action'] == 'PROCEED'
    
    def test_normal_boundary_high(self):
        """Test boundary at +0.05% (should be NORMAL)."""
        result = evaluate_funding_rate(0.05)
        assert result['level'] == 'NORMAL'
    
    def test_normal_boundary_low(self):
        """Test boundary at -0.05% (should be NORMAL)."""
        result = evaluate_funding_rate(-0.05)
        assert result['level'] == 'NORMAL'
    
    def test_high_positive(self):
        """Test high positive funding rate (0.07%)."""
        result = evaluate_funding_rate(0.07)
        assert result['level'] == 'HIGH'
        assert result['action'] == 'REDUCE'
        assert result['risk_multiplier'] == 0.5
    
    def test_high_negative(self):
        """Test high negative funding rate (-0.08%)."""
        result = evaluate_funding_rate(-0.08)
        assert result['level'] == 'HIGH'
        assert result['action'] == 'REDUCE'
        assert result['risk_multiplier'] == 0.5
    
    def test_extreme_positive(self):
        """Test extreme positive funding rate (0.15%)."""
        result = evaluate_funding_rate(0.15)
        assert result['level'] == 'EXTREME'
        assert result['action'] == 'SKIP'
        assert result['risk_multiplier'] == 0.0
    
    def test_extreme_negative(self):
        """Test extreme negative funding rate (-0.12%)."""
        result = evaluate_funding_rate(-0.12)
        assert result['level'] == 'EXTREME'
        assert result['action'] == 'SKIP'
        assert result['risk_multiplier'] == 0.0
    
    def test_none_input(self):
        """Test None input returns UNKNOWN."""
        result = evaluate_funding_rate(None)
        assert result['level'] == 'UNKNOWN'
        assert result['action'] == 'PROCEED'
        assert result['risk_multiplier'] == 1.0


class TestGetPositionAdjustment:
    """Tests for get_position_adjustment function."""
    
    def test_normal_no_adjustment_long(self):
        """Test normal rate with LONG direction."""
        result = get_position_adjustment(0.02, 'LONG')
        assert result['adjust'] is False
        assert result['risk_multiplier'] == 1.0
        assert result['warning'] is None
    
    def test_normal_no_adjustment_short(self):
        """Test normal rate with SHORT direction."""
        result = get_position_adjustment(-0.02, 'SHORT')
        assert result['adjust'] is False
        assert result['risk_multiplier'] == 1.0
        assert result['warning'] is None
    
    def test_high_positive_long_warning(self):
        """Test high positive rate with LONG should warn."""
        result = get_position_adjustment(0.07, 'LONG')
        assert result['adjust'] is True
        assert result['risk_multiplier'] == 0.5
        assert result['warning'] is not None
        assert '做多警告' in result['warning']
    
    def test_high_positive_short_no_warning(self):
        """Test high positive rate with SHORT should not warn extra."""
        result = get_position_adjustment(0.07, 'SHORT')
        # HIGH level already reduces to 0.5, but no direction-specific warning
        assert result['risk_multiplier'] == 0.5
    
    def test_high_negative_short_warning(self):
        """Test high negative rate with SHORT should warn."""
        result = get_position_adjustment(-0.08, 'SHORT')
        assert result['adjust'] is True
        assert result['risk_multiplier'] == 0.5
        assert result['warning'] is not None
        assert '做空警告' in result['warning']
    
    def test_extreme_positive_long_skip(self):
        """Test extreme positive with LONG should skip."""
        result = get_position_adjustment(0.15, 'LONG')
        assert result['risk_multiplier'] == 0.0
        assert result['warning'] is not None
        assert '停止交易' in result['warning']
    
    def test_extreme_negative_short_skip(self):
        """Test extreme negative with SHORT should skip."""
        result = get_position_adjustment(-0.15, 'SHORT')
        assert result['risk_multiplier'] == 0.0
        assert result['warning'] is not None
        assert '停止交易' in result['warning']
    
    def test_extreme_opposite_direction_ok(self):
        """Test extreme rate but opposite direction may proceed."""
        # Extreme positive but going SHORT (against the crowd)
        result = get_position_adjustment(0.15, 'SHORT')
        # Should still be 0 because EXTREME level
        assert result['risk_multiplier'] == 0.0
    
    def test_none_rate(self):
        """Test None rate returns no adjustment."""
        result = get_position_adjustment(None, 'LONG')
        assert result['adjust'] is False
        assert result['risk_multiplier'] == 1.0
        assert result['warning'] is None


class TestFormatFundingRateFilter:
    """Tests for format_funding_rate_filter function."""
    
    def test_available_normal(self):
        """Test formatting with available normal data."""
        data = {
            'symbol': 'BTCUSDT',
            'rate': 0.01,
            'rate_display': '+0.0100%',
            'available': True
        }
        evaluation = evaluate_funding_rate(data['rate'])
        result = format_funding_rate_filter(data, evaluation)
        
        assert 'RISK FILTER' in result
        assert '+0.0100%' in result
        assert 'NORMAL' in result
        assert 'PROCEED' in result
    
    def test_available_high(self):
        """Test formatting with high funding rate."""
        data = {
            'symbol': 'BTCUSDT',
            'rate': 0.08,
            'rate_display': '+0.0800%',
            'available': True
        }
        evaluation = evaluate_funding_rate(data['rate'])
        result = format_funding_rate_filter(data, evaluation)
        
        assert 'HIGH' in result
        assert 'REDUCE' in result
        assert '0.5x' in result
    
    def test_unavailable(self):
        """Test formatting when data unavailable."""
        data = {
            'symbol': 'BTCUSDT',
            'available': False
        }
        result = format_funding_rate_filter(data)
        
        assert 'N/A' in result
        assert 'unavailable' in result


class TestThresholds:
    """Tests for threshold constants."""
    
    def test_thresholds_exist(self):
        """Test all required thresholds exist."""
        assert 'NORMAL_LOW' in FUNDING_RATE_THRESHOLDS
        assert 'NORMAL_HIGH' in FUNDING_RATE_THRESHOLDS
        assert 'HIGH_LOW' in FUNDING_RATE_THRESHOLDS
        assert 'HIGH_HIGH' in FUNDING_RATE_THRESHOLDS
    
    def test_thresholds_order(self):
        """Test thresholds are in correct order."""
        assert FUNDING_RATE_THRESHOLDS['HIGH_LOW'] < FUNDING_RATE_THRESHOLDS['NORMAL_LOW']
        assert FUNDING_RATE_THRESHOLDS['NORMAL_LOW'] < FUNDING_RATE_THRESHOLDS['NORMAL_HIGH']
        assert FUNDING_RATE_THRESHOLDS['NORMAL_HIGH'] < FUNDING_RATE_THRESHOLDS['HIGH_HIGH']
    
    def test_status_levels_exist(self):
        """Test all status levels exist."""
        assert 'NORMAL' in FUNDING_RATE_STATUS
        assert 'HIGH' in FUNDING_RATE_STATUS
        assert 'EXTREME' in FUNDING_RATE_STATUS


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_very_small_positive(self):
        """Test very small positive rate."""
        result = evaluate_funding_rate(0.001)
        assert result['level'] == 'NORMAL'
    
    def test_very_small_negative(self):
        """Test very small negative rate."""
        result = evaluate_funding_rate(-0.001)
        assert result['level'] == 'NORMAL'
    
    def test_large_extreme_positive(self):
        """Test very large positive rate (1%)."""
        result = evaluate_funding_rate(1.0)
        assert result['level'] == 'EXTREME'
    
    def test_large_extreme_negative(self):
        """Test very large negative rate (-0.5%)."""
        result = evaluate_funding_rate(-0.5)
        assert result['level'] == 'EXTREME'


import pytest

from features.funding_rate import FundingFeature
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_funding_no_repainting(eth_1h_df, seed: int) -> None:
    """Without data/funding/ETHUSDT.parquet present, compute returns a
    fixed empty result — trivially non-repainting."""
    assert_no_repainting(FundingFeature(symbol="ETHUSDT"), eth_1h_df, seed=seed)
