"""Contract: threshold sweep row dict includes ops columns (no ML load)."""
from __future__ import annotations

import numpy as np

from scripts.threshold_sweep_latest_tmax_60d_fast import (
    _mean_return_and_sharpe,
    equity_step_metrics,
)
from src.backtest.engine import Trade, dedupe_trades_round_trips


def _build_row_like_sweep(res: dict) -> dict:
    """Mirror scripts/threshold_sweep_latest_tmax_60d_fast.py loop body (subset)."""
    equity_curve = res.get("equity_curve") or []
    mean_return, sharpe = _mean_return_and_sharpe(res.get("equity_curve"))
    if isinstance(equity_curve, list) and len(equity_curve) >= 1:
        cost_on = float(equity_curve[-1] - 1.0)
    else:
        cost_on = float(res.get("total_return", 0.0) or 0.0)
    equity_step_status, equity_step_count, equity_step_mean_return = equity_step_metrics(
        equity_curve if isinstance(equity_curve, list) else []
    )
    trades_list = list(res.get("trades") or [])
    deduped = dedupe_trades_round_trips(trades_list)
    unique_round_trips = int(res.get("unique_round_trips", len(deduped)))
    duplicate_trade_rows = int(res.get("duplicate_trade_rows", len(trades_list) - unique_round_trips))
    profits_u = [float(t["profit"]) for t in deduped if t.get("profit") is not None]
    mean_profit_roundtrip = float(np.mean(profits_u)) if profits_u else None
    return {
        "mean_return": mean_return,
        "sharpe": sharpe,
        "cost_on": cost_on,
        "equity_step_status": equity_step_status,
        "equity_step_count": equity_step_count,
        "equity_step_mean_return": equity_step_mean_return,
        "unique_round_trips": unique_round_trips,
        "duplicate_trade_rows": duplicate_trade_rows,
        "mean_profit_roundtrip": mean_profit_roundtrip,
    }


def test_sweep_row_has_ops_columns() -> None:
    t = Trade(
        entry_time="a",
        exit_time="b",
        entry_price=100.0,
        exit_price=101.0,
        direction="LONG",
        profit=0.01,
    )
    res = {
        "equity_curve": [1.0, 1.01],
        "trades": [t],
        "total_return": 0.01,
        "unique_round_trips": 1,
        "duplicate_trade_rows": 0,
    }
    row = _build_row_like_sweep(res)
    assert row["equity_step_status"] == "ok"
    assert row["equity_step_count"] == 1
    assert row["equity_step_mean_return"] is not None
    assert row["unique_round_trips"] == 1
    assert row["duplicate_trade_rows"] == 0
    assert row["mean_profit_roundtrip"] is not None


def test_insufficient_steps_row() -> None:
    t = Trade(
        entry_time="a",
        exit_time="b",
        entry_price=100.0,
        exit_price=101.0,
        direction="LONG",
        profit=0.01,
    )
    res = {
        "equity_curve": [1.02],
        "trades": [t],
        "total_return": 0.02,
        "unique_round_trips": 1,
        "duplicate_trade_rows": 0,
    }
    row = _build_row_like_sweep(res)
    assert row["equity_step_status"] == "insufficient_steps"
    assert row["equity_step_mean_return"] is None
    assert row["mean_return"] == 0.0 and row["sharpe"] == 0.0
