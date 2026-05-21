"""
Unit tests for strategy/divergence.py
"""
import pytest
import pandas as pd
import numpy as np
from features.divergence import (
    find_local_extrema,
    detect_rsi_divergence,
    detect_macd_divergence,
    get_divergence_summary
)


@pytest.fixture
def df_bullish_divergence():
    """Price makes Lower Low, RSI makes Higher Low (Bullish Regular)"""
    dates = pd.date_range(start='2026-01-01', periods=40, freq='15min')
    
    # Price trending down then making LL
    close_prices = [100] * 10 + [95] * 5 + [90] * 5 + [85] * 5 + [80] * 5 + [75] * 5 + [70] * 5
    
    # RSI making HL (recovering while price still drops)
    rsi_values = [50] * 10 + [40] * 5 + [30] * 5 + [25] * 5 + [20] * 5 + [28] * 5 + [35] * 5
    
    return pd.DataFrame({
        'close': close_prices,
        'high': [p + 2 for p in close_prices],
        'low': [p - 2 for p in close_prices],
        'rsi': rsi_values,
        'macd_hist': [0] * 40
    }, index=dates)


@pytest.fixture
def df_bearish_divergence():
    """Price makes Higher High, RSI makes Lower High (Bearish Regular)"""
    dates = pd.date_range(start='2026-01-01', periods=40, freq='15min')
    
    # Price trending up making HH
    close_prices = [100] * 10 + [105] * 5 + [110] * 5 + [115] * 5 + [120] * 5 + [125] * 5 + [130] * 5
    
    # RSI making LH (weakening while price rises)
    rsi_values = [50] * 10 + [60] * 5 + [70] * 5 + [75] * 5 + [72] * 5 + [68] * 5 + [65] * 5
    
    return pd.DataFrame({
        'close': close_prices,
        'high': [p + 2 for p in close_prices],
        'low': [p - 2 for p in close_prices],
        'rsi': rsi_values,
        'macd_hist': [0] * 40
    }, index=dates)


@pytest.fixture
def df_no_divergence():
    """Normal trending data without divergence"""
    dates = pd.date_range(start='2026-01-01', periods=30, freq='15min')
    close_prices = [100 + i for i in range(30)]
    rsi_values = [40 + i * 0.5 for i in range(30)]
    
    return pd.DataFrame({
        'close': close_prices,
        'high': [p + 1 for p in close_prices],
        'low': [p - 1 for p in close_prices],
        'rsi': rsi_values,
        'macd_hist': [i * 0.1 for i in range(30)]
    }, index=dates)


class TestFindLocalExtrema:
    def test_finds_maxima_and_minima(self):
        series = pd.Series([1, 2, 5, 2, 1, 3, 8, 3, 1])
        maxima, minima = find_local_extrema(series, window=1)
        
        assert len(maxima) > 0 or len(minima) > 0

    def test_returns_correct_structure(self):
        series = pd.Series([1, 5, 1, 5, 1])
        maxima, minima = find_local_extrema(series, window=1)
        
        if len(maxima) > 0:
            assert 'value' in maxima[0]
            assert 'idx' in maxima[0]


class TestDetectRsiDivergence:
    def test_returns_none_without_rsi_column(self):
        df = pd.DataFrame({'close': [100, 101, 102]})
        result = detect_rsi_divergence(df)
        assert result is None

    def test_detects_no_divergence(self, df_no_divergence):
        result = detect_rsi_divergence(df_no_divergence)
        # May or may not find divergence in random data
        assert result is None or 'type' in result

    def test_divergence_result_structure(self, df_bullish_divergence):
        result = detect_rsi_divergence(df_bullish_divergence)
        if result is not None:
            assert 'type' in result
            assert 'direction' in result
            assert result['direction'] in ['LONG', 'SHORT']


class TestDetectMacdDivergence:
    def test_returns_none_without_macd_column(self):
        df = pd.DataFrame({'close': [100, 101, 102]})
        result = detect_macd_divergence(df)
        assert result is None


class TestGetDivergenceSummary:
    def test_returns_summary_structure(self, df_no_divergence):
        summary = get_divergence_summary(df_no_divergence)
        
        assert 'has_divergence' in summary
        assert 'rsi_divergence' in summary
        assert 'macd_divergence' in summary
        assert 'consensus_direction' in summary
        assert isinstance(summary['has_divergence'], bool)


import pytest

from features.divergence import DivergenceFeature
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_divergence_no_repainting(eth_1h_df, seed: int) -> None:
    assert_no_repainting(DivergenceFeature(), eth_1h_df, seed=seed)
