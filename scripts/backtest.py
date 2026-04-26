"""Backtest CLI — Plan 5B-3 Task 4.

Loads klines + features + (optional) funding parquets and the latest
trained model bundle, runs BacktestRunner over the last `oos_fraction`
of the kline timeline, computes Sharpe + DSR, and writes one row to
`backtest_runs`.

Usage:
    python scripts/backtest.py \
        --kline data/history/ETHUSDT_1h.parquet \
        --features data/training/ETHUSDT_1h_features.parquet \
        --funding data/funding/ETHUSDT.parquet \
        --model-dir models \
        --sqlite-path data/state.db \
        --oos-fraction 0.2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from backtest.runner import BacktestRunner
from execution.cost_model import SlippageConfig
from execution.paper_broker import PaperBrokerConfig
from execution.replay_broker import ReplayBroker
from models.registry import load_latest_model


async def main_async(args: argparse.Namespace) -> None:
    klines = pd.read_parquet(args.kline)
    features = pd.read_parquet(args.features)
    funding = pd.read_parquet(args.funding) if args.funding else None
    model = load_latest_model(Path(args.model_dir))

    # OOS = last `oos_fraction` of klines.
    oos_n = int(len(klines) * args.oos_fraction)
    oos_start = klines.index[-oos_n]
    oos_end = klines.index[-1]

    cfg = PaperBrokerConfig(slippage=SlippageConfig(
        slippage_bps_base=1.0, slippage_bps_per_adv_unit=0.0, adv_stub=1000.0,
    ))
    broker = ReplayBroker(cfg=cfg, klines=klines, funding=funding,
                          symbol=args.symbol)
    runner = BacktestRunner(
        symbol=args.symbol,
        klines=klines, features=features, funding=funding,
        broker=broker, predictor=model,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        n_trials=args.n_trials,
    )
    result = await runner.run(oos_start=oos_start, oos_end=oos_end)

    summary = {
        "sharpe": float(result.sharpe) if not pd.isna(result.sharpe) else None,
        "deflated_sharpe": float(result.deflated_sharpe)
                             if not pd.isna(result.deflated_sharpe) else None,
        "n_trades": result.n_trades,
        "n_bars": result.n_bars,
        "initial_equity": result.initial_equity,
        "final_equity": result.final_equity,
        "oos_start": str(oos_start),
        "oos_end": str(oos_end),
        "ml_model_version": model.ml_model_version,
    }
    run_id = uuid.uuid4().hex[:12]

    engine = sa.create_engine(f"sqlite:///{args.sqlite_path}")
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO backtest_runs "
            "(run_id, started_at, deflated_sharpe, cost_model_version, summary_json) "
            "VALUES (:rid, :ts, :dsr, :cmv, :sj)"
        ), {
            "rid": run_id,
            "ts": datetime.now(tz=timezone.utc),
            "dsr": summary["deflated_sharpe"],
            "cmv": "plan5b2_v1",   # PaperBrokerConfig defaults snapshot
            "sj": json.dumps(summary),
        })

    print(f"backtest run_id={run_id} sharpe={summary['sharpe']} "
          f"dsr={summary['deflated_sharpe']} n_trades={summary['n_trades']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--kline", required=True, type=str)
    ap.add_argument("--features", required=True, type=str)
    ap.add_argument("--funding", default="", type=str)
    ap.add_argument("--model-dir", default="models", type=str)
    ap.add_argument("--sqlite-path", default="data/state.db", type=str)
    ap.add_argument("--oos-fraction", type=float, default=0.2)
    ap.add_argument("--long-threshold", type=float, default=0.58)
    ap.add_argument("--short-threshold", type=float, default=0.58)
    ap.add_argument("--n-trials", type=int, default=2)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
