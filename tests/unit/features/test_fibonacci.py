"""
Unit tests for strategy/fibonacci.py
"""
import pytest
import pandas as pd
from features.fibonacci import (
    calculate_fib_levels,
    get_ote_zone,
    get_premium_discount_zone,
    identify_fib_confluence,
    calculate_fib_from_df
)


class TestCalculateFibLevels:
    def test_bullish_direction(self):
        """Test Fibonacci calculation for bullish setup."""
        levels = calculate_fib_levels(100.0, 0.0, direction="bullish")
        
        assert levels['0.5'] == 50.0
        assert levels['0.618'] == pytest.approx(38.2, rel=0.01)
        assert levels['0.786'] == pytest.approx(21.4, rel=0.01)
        assert levels['swing_high'] == 100.0
        assert levels['swing_low'] == 0.0

    def test_bearish_direction(self):
        """Test Fibonacci calculation for bearish setup."""
        levels = calculate_fib_levels(100.0, 0.0, direction="bearish")
        
        assert levels['0.5'] == 50.0
        assert levels['0.618'] == pytest.approx(61.8, rel=0.01)
        assert levels['0.786'] == pytest.approx(78.6, rel=0.01)

    def test_real_price_range(self):
        """Test with realistic BTC price range."""
        levels = calculate_fib_levels(100000.0, 90000.0, direction="bullish")
        
        # 0.5 level should be at $95,000
        assert levels['0.5'] == 95000.0
        # 0.618 level should be at $93,820
        assert levels['0.618'] == pytest.approx(93820.0, rel=0.01)
        # OTE zone should be between 0.618 and 0.786
        assert levels['0.618'] > levels['0.786']

    def test_contains_all_levels(self):
        """Test that all expected levels are present."""
        levels = calculate_fib_levels(100.0, 50.0)
        
        expected_levels = ['0.236', '0.382', '0.5', '0.618', '0.705', '0.786']
        for level in expected_levels:
            assert level in levels


class TestGetOteZone:
    def test_returns_tuple(self):
        """Test OTE zone returns correct tuple."""
        levels = {'0.618': 40.0, '0.786': 20.0}
        lower, upper = get_ote_zone(levels)
        
        assert lower == 20.0
        assert upper == 40.0

    def test_ote_zone_values(self):
        """Test OTE zone with real values."""
        levels = calculate_fib_levels(100000.0, 90000.0, direction="bullish")
        lower, upper = get_ote_zone(levels)
        
        # OTE should be between 0.618 and 0.786 of the range
        assert lower < upper
        assert lower > 90000  # Above swing low
        assert upper < 100000  # Below swing high


class TestGetPremiumDiscountZone:
    def test_discount_zone(self):
        """Test price in discount zone."""
        levels = {
            '0.5': 50.0,
            'swing_high': 100.0,
            'swing_low': 0.0
        }
        
        zone, depth = get_premium_discount_zone(30.0, levels)
        assert zone == "DISCOUNT"
        assert depth > 0

    def test_premium_zone(self):
        """Test price in premium zone."""
        levels = {
            '0.5': 50.0,
            'swing_high': 100.0,
            'swing_low': 0.0
        }
        
        zone, depth = get_premium_discount_zone(70.0, levels)
        assert zone == "PREMIUM"
        assert depth > 0

    def test_at_equilibrium(self):
        """Test price at equilibrium (0.5 level)."""
        levels = {
            '0.5': 50.0,
            'swing_high': 100.0,
            'swing_low': 0.0
        }
        
        zone, depth = get_premium_discount_zone(50.0, levels)
        # At equilibrium, depth should be 0 or very small
        assert depth == pytest.approx(0.0, abs=0.01)


class TestIdentifyFibConfluence:
    def test_price_at_fib_level(self):
        """Test detection when price is at a Fib level."""
        levels = calculate_fib_levels(100.0, 0.0)
        # Price at 0.618 level (38.2)
        result = identify_fib_confluence(38.2, levels, tolerance=0.01)
        
        assert result['is_at_fib'] is True
        assert result['level'] == '0.618'

    def test_price_in_ote(self):
        """Test detection when price is in OTE zone."""
        levels = calculate_fib_levels(100.0, 0.0)
        # Price at 30 (between 0.618=38.2 and 0.786=21.4)
        result = identify_fib_confluence(30.0, levels, tolerance=0.003)
        
        assert result['is_ote'] is True

    def test_price_not_at_fib(self):
        """Test detection when price is not at any Fib level."""
        levels = calculate_fib_levels(100.0, 0.0)
        # Price at 75 (not near any Fib level)
        result = identify_fib_confluence(75.0, levels, tolerance=0.003)
        
        assert result['is_at_fib'] is False

    def test_returns_zone(self):
        """Test that zone is returned."""
        levels = calculate_fib_levels(100.0, 0.0)
        result = identify_fib_confluence(30.0, levels)
        
        assert 'zone' in result
        assert result['zone'] in ['PREMIUM', 'DISCOUNT']


class TestCalculateFibFromDf:
    def test_with_dataframe(self):
        """Test Fib calculation from DataFrame."""
        dates = pd.date_range(start='2026-01-01', periods=100, freq='15min')
        df = pd.DataFrame({
            'high': [100 + i * 0.5 for i in range(100)],
            'low': [90 + i * 0.5 for i in range(100)],
            'close': [95 + i * 0.5 for i in range(100)],
            'open': [95 + i * 0.5 for i in range(100)]
        }, index=dates)
        
        result = calculate_fib_from_df(df, lookback=50)
        
        assert 'levels' in result
        assert 'ote_zone' in result
        assert 'zone' in result
        assert 'confluence' in result

    def test_auto_direction(self):
        """Test auto direction detection."""
        dates = pd.date_range(start='2026-01-01', periods=50, freq='15min')
        # Uptrending data - close at the high end
        df = pd.DataFrame({
            'high': [100 + i for i in range(50)],
            'low': [90 + i for i in range(50)],
            'close': [140 + i for i in range(50)],  # Close above midpoint
            'open': [95 + i for i in range(50)]
        }, index=dates)
        
        result = calculate_fib_from_df(df, direction="auto")
        assert result['direction'] in ['bullish', 'bearish']


import pytest

from features.fibonacci import FibFeature
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fib_no_repainting(eth_1h_df, seed: int) -> None:
    assert_no_repainting(FibFeature(), eth_1h_df, seed=seed)
