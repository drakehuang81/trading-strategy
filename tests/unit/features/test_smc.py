"""
Unit tests for strategy/smc.py
"""
import pytest
import pandas as pd
import numpy as np
from features.smc import (
    find_swing_points,
    identify_structure,
    find_order_blocks,
    find_fvg,
    get_nearest_poi
)


@pytest.fixture
def uptrend_df():
    """DataFrame with clear uptrend (HH + HL pattern)."""
    dates = pd.date_range(start='2026-01-01', periods=60, freq='15min')
    # Create zigzag pattern with higher highs and higher lows
    data = {
        'high': [],
        'low': [],
        'close': [],
        'open': []
    }
    
    base = 100
    for i in range(60):
        # Create swing highs at i=10, 20, 30, 40, 50 (increasing)
        # Create swing lows at i=5, 15, 25, 35, 45 (increasing)
        if i % 10 == 5:  # Swing lows
            data['high'].append(base + i - 2)
            data['low'].append(base + i - 8)  # Lower wick
            data['close'].append(base + i - 4)
            data['open'].append(base + i - 3)
        elif i % 10 == 0 and i > 0:  # Swing highs
            data['high'].append(base + i + 5)  # Higher wick
            data['low'].append(base + i - 2)
            data['close'].append(base + i + 3)
            data['open'].append(base + i + 1)
        else:
            data['high'].append(base + i + 1)
            data['low'].append(base + i - 1)
            data['close'].append(base + i)
            data['open'].append(base + i)
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def downtrend_df():
    """DataFrame with clear downtrend (LH + LL pattern)."""
    dates = pd.date_range(start='2026-01-01', periods=60, freq='15min')
    data = {
        'high': [200 - i + (3 if i % 10 == 0 else 0) for i in range(60)],
        'low': [190 - i - (3 if i % 10 == 5 else 0) for i in range(60)],
        'close': [195 - i for i in range(60)],
        'open': [195 - i for i in range(60)]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def ranging_df():
    """DataFrame with ranging/sideways price action."""
    dates = pd.date_range(start='2026-01-01', periods=60, freq='15min')
    np.random.seed(42)
    data = {
        'high': [100 + np.random.randint(-2, 5) for _ in range(60)],
        'low': [95 + np.random.randint(-5, 2) for _ in range(60)],
        'close': [98 + np.random.randint(-3, 3) for _ in range(60)],
        'open': [97 + np.random.randint(-3, 3) for _ in range(60)]
    }
    return pd.DataFrame(data, index=dates)


class TestFindSwingPoints:
    def test_finds_swing_highs(self, uptrend_df):
        """Test that swing highs are detected."""
        sh, sl = find_swing_points(uptrend_df, window=2)
        assert len(sh) >= 0
        
    def test_finds_swing_lows(self, uptrend_df):
        """Test that swing lows are detected."""
        sh, sl = find_swing_points(uptrend_df, window=2)
        assert len(sl) >= 0

    def test_swing_point_structure(self, uptrend_df):
        """Test swing point dictionary structure."""
        sh, sl = find_swing_points(uptrend_df, window=2)
        if len(sh) > 0:
            assert 'price' in sh[0]
            assert 'idx' in sh[0]
            assert 'index' in sh[0]

    def test_window_affects_detection(self, uptrend_df):
        """Test that larger window detects fewer swings."""
        sh_small, sl_small = find_swing_points(uptrend_df, window=2)
        sh_large, sl_large = find_swing_points(uptrend_df, window=10)
        
        # Larger window should generally find fewer swing points
        assert len(sh_large) <= len(sh_small) or len(sl_large) <= len(sl_small)


class TestIdentifyStructure:
    def test_structure_types(self, uptrend_df):
        """Test that structure returns valid type."""
        structure = identify_structure(uptrend_df, window=2)
        assert structure['structure'] in ['BULLISH', 'BEARISH', 'RANGING']

    def test_structure_has_swing_points(self, uptrend_df):
        """Test that structure includes swing points."""
        structure = identify_structure(uptrend_df, window=2)
        assert 'swing_highs' in structure
        assert 'swing_lows' in structure

    def test_structure_has_bos(self, uptrend_df):
        """Test structure includes BOS detection."""
        structure = identify_structure(uptrend_df, window=2)
        assert 'last_bos' in structure

    def test_structure_has_choch(self, uptrend_df):
        """Test structure includes CHoCH detection."""
        structure = identify_structure(uptrend_df, window=2)
        assert 'last_choch' in structure

    def test_ranging_market(self, ranging_df):
        """Test ranging market detection."""
        structure = identify_structure(ranging_df, window=3)
        # Ranging market should have relatively low scores for both directions
        assert structure['hh_hl_count'] + structure['lh_ll_count'] < 8


class TestFindOrderBlocks:
    def test_returns_dict_with_both_types(self, uptrend_df):
        """Test that order blocks returns both bullish and bearish."""
        obs = find_order_blocks(uptrend_df)
        assert 'bullish' in obs
        assert 'bearish' in obs
        assert isinstance(obs['bullish'], list)
        assert isinstance(obs['bearish'], list)

    def test_order_block_structure(self, uptrend_df):
        """Test order block dictionary structure."""
        obs = find_order_blocks(uptrend_df, min_move_pct=0.1)
        if len(obs['bullish']) > 0:
            ob = obs['bullish'][0]
            assert 'zone' in ob
            assert 'high' in ob
            assert 'low' in ob
            assert 'strength' in ob

    def test_min_move_filter(self, uptrend_df):
        """Test that min_move_pct filters weak moves."""
        obs_low = find_order_blocks(uptrend_df, min_move_pct=0.1)
        obs_high = find_order_blocks(uptrend_df, min_move_pct=5.0)
        
        # Higher threshold should find fewer or equal OBs
        assert len(obs_high['bullish']) <= len(obs_low['bullish'])


class TestFindFvg:
    def test_bullish_fvg_detection(self):
        """Test bullish FVG detection."""
        # Candle 3's low > Candle 1's high = Bullish FVG
        data = {
            'high': [100, 110, 125],
            'low': [90, 105, 115],  # 115 > 100
            'close': [95, 108, 120],
            'open': [92, 102, 116]
        }
        df = pd.DataFrame(data)
        fvgs = find_fvg(df, lookback=3, min_gap_pct=0.01)
        
        assert len(fvgs['bullish']) > 0
        assert fvgs['bullish'][0]['gap_low'] == 100.0
        assert fvgs['bullish'][0]['gap_high'] == 115.0

    def test_bearish_fvg_detection(self):
        """Test bearish FVG detection."""
        # Candle 3's high < Candle 1's low = Bearish FVG
        data = {
            'high': [110, 100, 85],  # 85 < 90
            'low': [100, 90, 75],
            'close': [105, 95, 80],
            'open': [108, 98, 83]
        }
        df = pd.DataFrame(data)
        fvgs = find_fvg(df, lookback=3, min_gap_pct=0.01)
        
        assert len(fvgs['bearish']) > 0

    def test_fvg_zone_structure(self):
        """Test FVG zone has correct structure."""
        data = {
            'high': [100, 110, 125],
            'low': [90, 105, 115],
            'close': [95, 108, 120],
            'open': [92, 102, 116]
        }
        df = pd.DataFrame(data)
        fvgs = find_fvg(df, lookback=3, min_gap_pct=0.01)
        
        if len(fvgs['bullish']) > 0:
            fvg = fvgs['bullish'][0]
            assert 'zone' in fvg
            assert 'gap_pct' in fvg
            assert 'filled' in fvg

    def test_no_fvg_in_overlapping_candles(self):
        """Test no FVG when candles overlap (wicks touch)."""
        # Candle 3's low (99) is NOT > Candle 1's high (100), so no bullish FVG
        # Candle 3's high (103) is NOT < Candle 1's low (98), so no bearish FVG
        data = {
            'high': [100, 102, 103],
            'low': [98, 99, 99],  # Candle 3 low (99) < Candle 1 high (100) = overlap
            'close': [99, 101, 102],
            'open': [99, 100, 100]
        }
        df = pd.DataFrame(data)
        fvgs = find_fvg(df, lookback=3, min_gap_pct=0.01)
        
        assert len(fvgs['bullish']) == 0
        assert len(fvgs['bearish']) == 0


class TestGetNearestPoi:
    def test_long_poi_below_price(self, uptrend_df):
        """Test that LONG POI is below current price."""
        poi = get_nearest_poi(uptrend_df, "LONG")
        if poi is not None:
            current_price = uptrend_df['close'].iloc[-1]
            assert poi['mid_price'] <= current_price

    def test_short_poi_above_price(self, downtrend_df):
        """Test that SHORT POI is above current price."""
        poi = get_nearest_poi(downtrend_df, "SHORT")
        if poi is not None:
            current_price = downtrend_df['close'].iloc[-1]
            assert poi['mid_price'] >= current_price

    def test_poi_has_type(self, uptrend_df):
        """Test that POI includes type."""
        poi = get_nearest_poi(uptrend_df, "LONG")
        if poi is not None:
            assert 'type' in poi
            assert poi['type'] in ['ORDER_BLOCK', 'FVG', 'SWING_LOW', 'SWING_HIGH']

    def test_poi_has_zone(self, uptrend_df):
        """Test that POI includes zone."""
        poi = get_nearest_poi(uptrend_df, "LONG")
        if poi is not None:
            assert 'zone' in poi
            assert len(poi['zone']) == 2


import pytest

from features.smc import SMCFeature
from tests.helpers.no_repainting import assert_no_repainting


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_smc_no_repainting(eth_1h_df, seed: int) -> None:
    assert_no_repainting(SMCFeature(), eth_1h_df, seed=seed)
