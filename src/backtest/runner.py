"""BacktestRunner — Plan 5B-3 Task 4.

Iterates an OOS window of bars; at each bar:
  1. broker.set_time(bar.ts)  — drives ReplayBroker clock + funding
  2. lookup features at bar.ts (pre-computed)
  3. predictor.predict(features) → PredictionBundle
  4. ThresholdPolicy → TradeProposal (or None)
  5. submit to broker (always — no risk pipeline in this baseline)
  6. drain events → accumulate into equity_curve

Then computes Sharpe + Deflated Sharpe over per-bar returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from backtest.equity import equity_curve
from decision.policy import ThresholdPolicy
from decision.proposal import PortfolioSnapshot
from execution.base import BrokerEvent, Order
from execution.replay_broker import ReplayBroker
from features.registry import flatten_features
from models.base import Predictor


@dataclass
class BacktestResult:
    equity: pd.Series           # per-bar equity (DatetimeIndex)
    returns: pd.Series          # equity.pct_change().dropna()
    sharpe: float               # annualised Sharpe ratio
    deflated_sharpe: float      # DSR z-score
    n_trades: int               # count of "filled" events
    n_bars: int                 # count of bars iterated
    initial_equity: float
    final_equity: float


@dataclass
class BacktestRunner:
    symbol: str
    klines: pd.DataFrame
    features: pd.DataFrame
    funding: pd.DataFrame | None
    broker: ReplayBroker
    predictor: Any              # Predictor protocol (XGBPredictor or stub)
    long_threshold: float = 0.58
    short_threshold: float = 0.58
    initial_equity_usdt: float = 10_000.0
    fixed_position_size: float = 0.01     # contracts per trade (small)
    n_trials: int = 2

    async def run(self, *, oos_start: pd.Timestamp,
                  oos_end: pd.Timestamp) -> BacktestResult:
        oos_bars = self.klines.loc[oos_start:oos_end].index

        policy = ThresholdPolicy(
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
            symbol=self.symbol,
            mid_provider=lambda _s: float(
                self.klines.loc[self.broker._current_ts]["close"]
            ),
            atr_provider=lambda _s: 15.0,        # constant ATR for baseline
        )
        portfolio = PortfolioSnapshot(
            equity_usdt=self.initial_equity_usdt,
            open_positions={}, day_pnl_r=0.0, consecutive_wins=0,
        )

        all_events: list[BrokerEvent] = []
        n_trades = 0

        for ts in oos_bars:
            self.broker.set_time(ts)
            # Drain any funding events emitted by set_time.
            for e in self.broker.drain_events():
                all_events.append(e)
                if e.kind == "filled":
                    n_trades += 1
            # Predict.
            if ts not in self.features.index:
                continue
            feats_row = self.features.loc[ts]
            # Convert row → flat dict; predictor stubs accept any shape;
            # trained models look up flat keys via feature_order.
            feats = {col: float(feats_row[col]) for col in self.features.columns}
            bundle = await self.predictor.predict(feats)
            proposal = await policy.propose(feats, bundle, portfolio)
            if proposal is None:
                continue
            side = "buy" if proposal.direction == "long" else "sell"
            order = Order(
                client_order_id=proposal.proposal_id,
                symbol=self.symbol,
                side=side, type="market",
                qty=self.fixed_position_size,
            )
            await self.broker.submit(order)
            for e in self.broker.drain_events():
                all_events.append(e)
                if e.kind == "filled":
                    n_trades += 1

        mark_prices = {
            ts.to_pydatetime(): float(self.klines.loc[ts]["close"])
            for ts in oos_bars
        }
        equity = equity_curve(
            events=all_events,
            initial_equity=self.initial_equity_usdt,
            mark_prices=mark_prices,
        )
        returns = equity.pct_change().dropna()

        from observability.sharpe import sharpe_ratio, deflated_sharpe_ratio, PERIODS_PER_YEAR_1H
        sr = sharpe_ratio(returns.to_numpy(), periods_per_year=PERIODS_PER_YEAR_1H)
        dsr = deflated_sharpe_ratio(
            returns.to_numpy(), n_trials=self.n_trials,
            periods_per_year=PERIODS_PER_YEAR_1H,
        )

        return BacktestResult(
            equity=equity,
            returns=returns,
            sharpe=sr,
            deflated_sharpe=dsr,
            n_trades=n_trades,
            n_bars=len(oos_bars),
            initial_equity=self.initial_equity_usdt,
            final_equity=float(equity.iloc[-1]),
        )
