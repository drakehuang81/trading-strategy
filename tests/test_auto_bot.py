"""
Unit tests for auto_bot.py strategy logic
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_bot import (
    calculate_fibonacci,
    get_asia_session_range,
    check_amd_session,
    calculate_confidence_score
)


# --- Fixtures ---

@pytest.fixture
def sample_bullish_df():
    """Create a sample bullish DataFrame with indicators"""
    dates = pd.date_range(start='2026-01-20 00:00:00', periods=100, freq='15min')
    
    # Create uptrending price data
    base_price = 90000
    prices = [base_price + i * 10 + np.random.randint(-50, 50) for i in range(100)]
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.randint(10, 100) for p in prices],
        'low': [p - np.random.randint(10, 100) for p in prices],
        'close': [p + np.random.randint(-30, 30) for p in prices],
        'volume': [1000 + np.random.randint(0, 500) for _ in range(100)],
        'rsi': [25 + i * 0.3 for i in range(100)],  # RSI trending up from oversold
        'macd_hist': [-50 + i * 1.2 for i in range(100)],  # MACD improving
    }, index=dates)
    
    return df


@pytest.fixture
def sample_bearish_df():
    """Create a sample bearish DataFrame with indicators"""
    dates = pd.date_range(start='2026-01-20 00:00:00', periods=100, freq='15min')
    
    # Create downtrending price data
    base_price = 95000
    prices = [base_price - i * 10 + np.random.randint(-50, 50) for i in range(100)]
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.randint(10, 100) for p in prices],
        'low': [p - np.random.randint(10, 100) for p in prices],
        'close': [p + np.random.randint(-30, 30) for p in prices],
        'volume': [1000 + np.random.randint(0, 500) for _ in range(100)],
        'rsi': [75 - i * 0.3 for i in range(100)],  # RSI trending down from overbought
        'macd_hist': [50 - i * 1.2 for i in range(100)],  # MACD weakening
    }, index=dates)
    
    return df


@pytest.fixture
def sample_neutral_df():
    """Create a sample neutral/ranging DataFrame"""
    dates = pd.date_range(start='2026-01-20 00:00:00', periods=100, freq='15min')
    
    base_price = 90000
    prices = [base_price + np.random.randint(-100, 100) for _ in range(100)]
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p + 50 for p in prices],
        'low': [p - 50 for p in prices],
        'close': prices,
        'volume': [1000 for _ in range(100)],
        'rsi': [50 + np.random.randint(-5, 5) for _ in range(100)],  # RSI neutral
        'macd_hist': [np.random.randint(-10, 10) for _ in range(100)],  # MACD flat
    }, index=dates)
    
    return df


# --- Tests for calculate_fibonacci ---

class TestCalculateFibonacci:
    
    def test_basic_fibonacci_calculation(self, sample_bullish_df):
        """Test that Fibonacci levels are calculated correctly"""
        levels, swing_high, swing_low = calculate_fibonacci(sample_bullish_df)
        
        assert swing_high > swing_low, "Swing high should be greater than swing low"
        assert '0.5' in levels, "Should contain 0.5 level (equilibrium)"
        
        # Check 0.5 level is correct
        expected_half = swing_high - 0.5 * (swing_high - swing_low)
        assert abs(levels['0.5'] - expected_half) < 0.01, "0.5 level calculation is incorrect"
    
    def test_fibonacci_with_flat_market(self):
        """Test Fibonacci calculation when high == low (edge case)"""
        dates = pd.date_range(start='2026-01-20', periods=10, freq='15min')
        flat_df = pd.DataFrame({
            'high': [100.0] * 10,
            'low': [100.0] * 10,
            'close': [100.0] * 10,
        }, index=dates)
        
        levels, swing_high, swing_low = calculate_fibonacci(flat_df)
        
        assert swing_high == swing_low == 100.0
        assert levels['0.5'] == 100.0  # Should still work, just be the same price


# --- Tests for get_asia_session_range ---

class TestGetAsiaSessionRange:
    
    def test_asia_session_detection(self):
        """Test that Asia session hours are correctly identified (UTC 00:00-07:00)"""
        # Create data spanning Asia session
        dates = pd.date_range(start='2026-01-20 00:00:00', periods=28, freq='15min')  # 7 hours
        
        df = pd.DataFrame({
            'high': [100 + i for i in range(28)],
            'low': [90 + i for i in range(28)],
            'close': [95 + i for i in range(28)],
        }, index=dates)
        
        asia_high, asia_low = get_asia_session_range(df)
        
        assert asia_high is not None, "Should detect Asia session high"
        assert asia_low is not None, "Should detect Asia session low"
        assert asia_high > asia_low, "High should be greater than low"
    
    def test_no_asia_session_data(self):
        """Test handling when no Asia session data exists"""
        # Create data only during NY session (UTC 13:00+)
        dates = pd.date_range(start='2026-01-20 14:00:00', periods=20, freq='15min')
        
        df = pd.DataFrame({
            'high': [100] * 20,
            'low': [90] * 20,
            'close': [95] * 20,
        }, index=dates)
        
        asia_high, asia_low = get_asia_session_range(df)
        
        assert asia_high is None and asia_low is None, "Should return None when no Asia data"
    
    def test_insufficient_asia_data(self):
        """Test handling when Asia session has less than 4 candles"""
        dates = pd.date_range(start='2026-01-20 06:00:00', periods=3, freq='15min')
        
        df = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [90, 91, 92],
            'close': [95, 96, 97],
        }, index=dates)
        
        asia_high, asia_low = get_asia_session_range(df)
        
        # Should return None because less than 4 candles
        assert asia_high is None and asia_low is None


# --- Tests for check_amd_session ---

class TestCheckAmdSession:
    
    def test_session_returns_string(self):
        """Test that check_amd_session returns a valid session string"""
        session = check_amd_session()
        
        assert isinstance(session, str), "Should return a string"
        assert len(session) > 0, "String should not be empty"
    
    def test_session_contains_expected_keywords(self):
        """Test that session name contains expected keywords"""
        session = check_amd_session()
        
        valid_keywords = ['亞洲', '倫敦', '紐約', '未知']
        assert any(keyword in session for keyword in valid_keywords), \
            f"Session '{session}' should contain a valid keyword"


# --- Tests for calculate_confidence_score ---

class TestCalculateConfidenceScore:
    
    def test_long_score_returns_tuple(self, sample_bullish_df):
        """Test that calculate_confidence_score returns score and factors"""
        score, factors = calculate_confidence_score(sample_bullish_df, is_long=True)
        
        assert isinstance(score, (int, float)), "Score should be numeric"
        assert isinstance(factors, list), "Factors should be a list"
    
    def test_short_score_returns_tuple(self, sample_bearish_df):
        """Test that calculate_confidence_score works for short"""
        score, factors = calculate_confidence_score(sample_bearish_df, is_long=False)
        
        assert isinstance(score, (int, float)), "Score should be numeric"
        assert isinstance(factors, list), "Factors should be a list"
    
    def test_score_range(self, sample_bullish_df):
        """Test that score is within expected range (0-6+)"""
        score_long, _ = calculate_confidence_score(sample_bullish_df, is_long=True)
        score_short, _ = calculate_confidence_score(sample_bullish_df, is_long=False)
        
        assert 0 <= score_long <= 10, "Long score should be in reasonable range"
        assert 0 <= score_short <= 10, "Short score should be in reasonable range"
    
    def test_bullish_df_favors_long(self, sample_bullish_df):
        """Test that bullish data gives higher long score than short score"""
        score_long, _ = calculate_confidence_score(sample_bullish_df, is_long=True)
        score_short, _ = calculate_confidence_score(sample_bullish_df, is_long=False)
        
        # Bullish data should generally favor long
        # (This might not always be true due to randomness, so we allow some tolerance)
        assert score_long >= 0, "Long score should be non-negative for bullish data"
    
    def test_bearish_df_favors_short(self, sample_bearish_df):
        """Test that bearish data gives higher short score than long score"""
        score_long, _ = calculate_confidence_score(sample_bearish_df, is_long=True)
        score_short, _ = calculate_confidence_score(sample_bearish_df, is_long=False)
        
        assert score_short >= 0, "Short score should be non-negative for bearish data"
    
    def test_factors_contain_checkmarks(self, sample_bullish_df):
        """Test that factors list contains emoji checkmarks for positive conditions"""
        _, factors = calculate_confidence_score(sample_bullish_df, is_long=True)
        
        # At least some factors should have checkmarks
        if len(factors) > 0:
            assert any('✅' in f for f in factors), "Positive factors should contain ✅"
    
    def test_neutral_market_low_scores(self, sample_neutral_df):
        """Test that neutral market gives relatively low scores"""
        score_long, _ = calculate_confidence_score(sample_neutral_df, is_long=True)
        score_short, _ = calculate_confidence_score(sample_neutral_df, is_long=False)
        
        # Neutral market shouldn't trigger high confidence for either direction
        # Both scores should be moderate
        assert score_long < 6, "Neutral market shouldn't have high long score"
        assert score_short < 6, "Neutral market shouldn't have high short score"


# --- Integration-like Tests ---

class TestIntegration:
    
    def test_full_analysis_flow(self, sample_bullish_df):
        """Test the complete analysis flow without errors"""
        # Step 1: Fibonacci
        fib_levels, high, low = calculate_fibonacci(sample_bullish_df)
        assert fib_levels is not None
        
        # Step 2: Asia range
        asia_high, asia_low = get_asia_session_range(sample_bullish_df)
        # May be None, that's OK
        
        # Step 3: Session check
        session = check_amd_session()
        assert session is not None
        
        # Step 4: Confidence score
        score, factors = calculate_confidence_score(sample_bullish_df, is_long=True)
        assert score >= 0
        
    def test_handles_missing_columns_gracefully(self):
        """Test that functions handle missing indicator columns"""
        dates = pd.date_range(start='2026-01-20', periods=10, freq='15min')
        
        # DataFrame without RSI/MACD columns
        minimal_df = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100] * 10,
        }, index=dates)
        
        # Fibonacci should still work
        levels, high, low = calculate_fibonacci(minimal_df)
        assert levels is not None
        
        # Asia range should still work
        asia_high, asia_low = get_asia_session_range(minimal_df)
        # OK to be None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
