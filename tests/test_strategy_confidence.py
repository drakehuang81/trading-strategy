"""
Unit tests for strategy/confidence.py
"""
import pytest
from strategy.confidence import (
    check_htf_bias,
    check_smc_poi,
    check_fib_confluence,
    check_liquidity_swept,
    check_rsi_confluence,
    check_macd_confluence,
    check_amd_session,
    check_asia_range_sweep,
    calculate_confidence_score,
    get_position_recommendation,
    format_confidence_score
)


class TestCheckHtfBias:
    def test_aligned_bullish(self):
        result = check_htf_bias("BULLISH", "LONG")
        assert result['status'] is True
        assert result['score'] == 1

    def test_aligned_bearish(self):
        result = check_htf_bias("BEARISH", "SHORT")
        assert result['status'] is True
        assert result['score'] == 1

    def test_not_aligned(self):
        result = check_htf_bias("BULLISH", "SHORT")
        assert result['status'] is False
        assert result['score'] == 0

    def test_neutral_not_aligned(self):
        result = check_htf_bias("NEUTRAL", "LONG")
        assert result['status'] is False


class TestCheckSmcPoi:
    def test_valid_poi(self):
        result = check_smc_poi({'type': 'ORDER_BLOCK'})
        assert result['status'] is True
        assert result['score'] == 1

    def test_fvg_poi(self):
        result = check_smc_poi({'type': 'FVG'})
        assert result['status'] is True

    def test_no_poi(self):
        result = check_smc_poi(None)
        assert result['status'] is False
        assert result['score'] == 0


class TestCheckFibConfluence:
    def test_in_ote(self):
        result = check_fib_confluence({
            'confluence': {'is_ote': True, 'is_at_fib': False}
        })
        assert result['status'] is True

    def test_at_fib_level(self):
        result = check_fib_confluence({
            'confluence': {'is_ote': False, 'is_at_fib': True}
        })
        assert result['status'] is True

    def test_no_confluence(self):
        result = check_fib_confluence({
            'confluence': {'is_ote': False, 'is_at_fib': False}
        })
        assert result['status'] is False


class TestCheckLiquiditySwept:
    def test_eql_swept_for_long(self):
        result = check_liquidity_swept(
            {'eql_swept': True, 'eqh_swept': False},
            "LONG"
        )
        assert result['status'] is True

    def test_eqh_swept_for_short(self):
        result = check_liquidity_swept(
            {'eql_swept': False, 'eqh_swept': True},
            "SHORT"
        )
        assert result['status'] is True

    def test_wrong_sweep_for_direction(self):
        result = check_liquidity_swept(
            {'eql_swept': False, 'eqh_swept': True},
            "LONG"
        )
        assert result['status'] is False


class TestCheckRsiConfluence:
    def test_oversold_for_long(self):
        result = check_rsi_confluence(25.0, "LONG")
        assert result['status'] is True

    def test_overbought_for_short(self):
        result = check_rsi_confluence(75.0, "SHORT")
        assert result['status'] is True

    def test_neutral_no_confluence(self):
        result = check_rsi_confluence(50.0, "LONG")
        assert result['status'] is False

    def test_divergence_adds_confluence(self):
        result = check_rsi_confluence(
            50.0, "LONG",
            divergence={'type': 'BULLISH_REGULAR'}
        )
        assert result['status'] is True


class TestCheckMacdConfluence:
    def test_increasing_for_long(self):
        result = check_macd_confluence(100.0, 50.0, "LONG")
        assert result['status'] is True

    def test_decreasing_for_short(self):
        result = check_macd_confluence(50.0, 100.0, "SHORT")
        assert result['status'] is True

    def test_wrong_direction(self):
        result = check_macd_confluence(50.0, 100.0, "LONG")
        assert result['status'] is False


class TestCheckAmdSession:
    def test_returns_structure(self):
        result = check_amd_session()
        assert 'factor' in result
        assert 'status' in result
        assert 'score' in result
        assert isinstance(result['status'], bool)


class TestCheckAsiaRangeSweep:
    def test_long_sweep(self):
        result = check_asia_range_sweep(
            current_price=100.0,
            ash=110.0,
            asl=90.0,
            direction="LONG"
        )
        assert result['status'] is True  # Price above ASL

    def test_short_sweep(self):
        result = check_asia_range_sweep(
            current_price=100.0,
            ash=110.0,
            asl=90.0,
            direction="SHORT"
        )
        assert result['status'] is True  # Price below ASH


class TestCalculateConfidenceScore:
    def test_perfect_score(self):
        result = calculate_confidence_score(
            direction="LONG",
            htf_bias="BULLISH",
            poi={'type': 'ORDER_BLOCK'},
            fib_analysis={'confluence': {'is_ote': True, 'is_at_fib': True}},
            liquidity_sweep={'eql_swept': True, 'eqh_swept': False},
            rsi=30.0,
            macd_hist=100.0,
            prev_macd_hist=50.0,
            ash=100.0,
            asl=90.0,
            current_price=95.0
        )
        
        assert result['score'] == 8
        assert result['rating'] == 'A+'
        assert len(result['factors']) == 8

    def test_low_score(self):
        result = calculate_confidence_score(
            direction="LONG",
            htf_bias="BEARISH",
            poi=None,
            fib_analysis={'confluence': {'is_ote': False, 'is_at_fib': False}},
            liquidity_sweep={'eql_swept': False, 'eqh_swept': False},
            rsi=50.0,
            macd_hist=50.0,
            prev_macd_hist=100.0,
            ash=None,
            asl=None,
            current_price=100.0
        )
        
        assert result['score'] < 4

    def test_has_recommendation(self):
        result = calculate_confidence_score(
            direction="LONG",
            htf_bias="BULLISH",
            poi={'type': 'ORDER_BLOCK'},
            fib_analysis={'confluence': {'is_ote': True, 'is_at_fib': False}},
            liquidity_sweep={'eql_swept': True, 'eqh_swept': False},
            rsi=35.0,
            macd_hist=100.0,
            prev_macd_hist=50.0,
            ash=100.0,
            asl=90.0,
            current_price=95.0
        )
        
        assert 'recommendation' in result
        assert 'action' in result['recommendation']


class TestGetPositionRecommendation:
    def test_high_score_full_position(self):
        rec = get_position_recommendation(8)
        assert rec['action'] == 'TRADE'
        assert rec['risk'] == '1%'
        assert rec['leverage'] == 10

    def test_medium_score(self):
        rec = get_position_recommendation(5)
        assert rec['action'] == 'TRADE'
        assert rec['leverage'] == 7

    def test_low_score_skip(self):
        rec = get_position_recommendation(2)
        assert rec['action'] == 'SKIP'
        assert rec['leverage'] == 0


class TestFormatConfidenceScore:
    def test_format_output(self):
        result = calculate_confidence_score(
            direction="LONG",
            htf_bias="BULLISH",
            poi={'type': 'ORDER_BLOCK'},
            fib_analysis={'confluence': {'is_ote': True, 'is_at_fib': True}},
            liquidity_sweep={'eql_swept': True, 'eqh_swept': False},
            rsi=30.0,
            macd_hist=100.0,
            prev_macd_hist=50.0,
            ash=100.0,
            asl=90.0,
            current_price=95.0
        )
        
        formatted = format_confidence_score(result)
        
        assert 'Confidence Score' in formatted
        assert '8/8' in formatted
        assert '✅' in formatted
        assert 'Recommendation' in formatted
