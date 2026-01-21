"""
Unit tests for strategy/trade_setup.py
"""
import pytest
from strategy.trade_setup import calculate_trade_setup, format_trade_setup


class TestCalculateTradeSetup:
    def test_long_setup_structure(self):
        setup = calculate_trade_setup(
            direction="LONG",
            entry_price=100.0,
            swing_low=95.0,
            swing_high=110.0,
            account_balance=10000,
            risk_percent=0.01,
            leverage=5
        )
        
        assert setup['direction'] == 'LONG'
        assert setup['entry'] == 100.0
        assert 'stop_loss' in setup
        assert 'tp1' in setup
        assert 'tp2' in setup
        assert 'tp3' in setup
        assert 'position_size_usd' in setup
        assert 'risk_usd' in setup

    def test_short_setup_structure(self):
        setup = calculate_trade_setup(
            direction="SHORT",
            entry_price=100.0,
            swing_low=90.0,
            swing_high=105.0,
            account_balance=10000,
            risk_percent=0.01,
            leverage=5
        )
        
        assert setup['direction'] == 'SHORT'
        assert setup['stop_loss'] > setup['entry']  # SL above entry for short

    def test_long_sl_below_entry(self):
        setup = calculate_trade_setup(
            direction="LONG",
            entry_price=100.0,
            swing_low=95.0,
            swing_high=110.0
        )
        
        assert setup['stop_loss'] < setup['entry']

    def test_long_tp_above_entry(self):
        setup = calculate_trade_setup(
            direction="LONG",
            entry_price=100.0,
            swing_low=95.0,
            swing_high=110.0
        )
        
        assert setup['tp1']['price'] > setup['entry']
        assert setup['tp2']['price'] > setup['tp1']['price']
        assert setup['tp3']['price'] > setup['tp2']['price']

    def test_short_tp_below_entry(self):
        setup = calculate_trade_setup(
            direction="SHORT",
            entry_price=100.0,
            swing_low=90.0,
            swing_high=105.0
        )
        
        assert setup['tp1']['price'] < setup['entry']
        assert setup['tp2']['price'] < setup['tp1']['price']
        assert setup['tp3']['price'] < setup['tp2']['price']

    def test_rr_ratios(self):
        setup = calculate_trade_setup(
            direction="LONG",
            entry_price=100.0,
            swing_low=95.0,
            swing_high=110.0
        )
        
        assert setup['tp1']['rr'] == 2.0
        assert setup['tp2']['rr'] == 3.0
        assert setup['tp3']['rr'] == 5.0

    def test_risk_calculation(self):
        setup = calculate_trade_setup(
            direction="LONG",
            entry_price=100.0,
            swing_low=95.0,
            swing_high=110.0,
            account_balance=10000,
            risk_percent=0.01
        )
        
        assert setup['risk_usd'] == 100.0  # 1% of 10000


class TestFormatTradeSetup:
    def test_format_long_setup(self):
        setup = calculate_trade_setup(
            direction="LONG",
            entry_price=50000.0,
            swing_low=49000.0,
            swing_high=52000.0
        )
        
        formatted = format_trade_setup(setup, "BTCUSDT")
        
        assert "🟢" in formatted
        assert "LONG" in formatted
        assert "BTCUSDT" in formatted
        assert "Entry" in formatted
        assert "Stop Loss" in formatted
        assert "TP1" in formatted

    def test_format_short_setup(self):
        setup = calculate_trade_setup(
            direction="SHORT",
            entry_price=50000.0,
            swing_low=48000.0,
            swing_high=51000.0
        )
        
        formatted = format_trade_setup(setup, "BTCUSDT")
        
        assert "🔴" in formatted
        assert "SHORT" in formatted
