"""
Type definitions and result structures for ML backtest engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, TypedDict

Signal = Literal["LONG", "SHORT", "HOLD"]


class Trade(TypedDict):
    """Represents a single trade."""
    entry_time: str
    exit_time: str | None
    entry_price: float
    exit_price: float | None
    direction: Literal["LONG", "SHORT"]
    profit: float | None


class BacktestResult(TypedDict):
    """Result of a backtest run."""
    total_return: float
    win_rate: float
    max_drawdown: float
    trades: List[Trade]
    equity_curve: List[float]
    # Extended statistics
    total_trades: int
    long_trades: int
    short_trades: int
    avg_profit: float
    median_profit: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int

