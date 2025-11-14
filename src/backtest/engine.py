"""
Simple backtesting engine for EMA + RSI strategy.
"""
from __future__ import annotations

from typing import List, Literal, TypedDict

import pandas as pd

from src.indicators.basic import get_df_with_indicators


Signal = Literal["LONG", "SHORT", "HOLD"]


class Trade(TypedDict):
    entry_time: str
    exit_time: str | None
    entry_price: float
    exit_price: float | None
    direction: Literal["LONG", "SHORT"]
    profit: float | None


class BacktestResult(TypedDict):
    total_return: float
    win_rate: float
    max_drawdown: float
    trades: List[Trade]
    equity_curve: List[float]


def run_backtest() -> BacktestResult:
    """
    Run a simple EMA + RSI backtest across the entire dataset.
    """

    df: pd.DataFrame = get_df_with_indicators().copy()

    # Strategy signals
    df["signal"] = "HOLD"
    df.loc[(df["close"] > df["ema_20"]) & (df["rsi_14"] < 70), "signal"] = "LONG"
    df.loc[(df["close"] < df["ema_20"]) & (df["rsi_14"] > 30), "signal"] = "SHORT"

    trades: List[Trade] = []
    equity_curve: List[float] = []

    position: dict | None = None
    balance = 1.0  # Start with normalized capital of 1.0

    for row in df.itertuples():
        signal: Signal = getattr(row, "signal")

        if position is None:
            if signal in ("LONG", "SHORT"):
                position = {
                    "side": signal,
                    "entry_price": float(getattr(row, "close")),
                    "entry_time": str(getattr(row, "timestamp")),
                }
        else:
            if signal == "HOLD" or signal != position["side"]:
                exit_price = float(getattr(row, "close"))
                entry_price = position["entry_price"]
                direction: Literal["LONG", "SHORT"] = position["side"]

                if direction == "LONG":
                    profit = (exit_price - entry_price) / entry_price
                else:
                    profit = (entry_price - exit_price) / entry_price

                balance *= 1 + profit

                trades.append(
                    Trade(
                        entry_time=position["entry_time"],
                        exit_time=str(getattr(row, "timestamp")),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=direction,
                        profit=profit,
                    )
                )

                equity_curve.append(balance)
                position = None

    # Close remaining position at last candle if open
    if position is not None:
        last_row = df.iloc[-1]
        last_price = float(last_row["close"])
        entry_price = position["entry_price"]
        direction: Literal["LONG", "SHORT"] = position["side"]

        if direction == "LONG":
            profit = (last_price - entry_price) / entry_price
        else:
            profit = (entry_price - last_price) / entry_price

        balance *= 1 + profit

        trades.append(
            Trade(
                entry_time=position["entry_time"],
                exit_time=str(last_row["timestamp"]),
                entry_price=entry_price,
                exit_price=last_price,
                direction=direction,
                profit=profit,
            )
        )
        equity_curve.append(balance)

    if trades:
        wins = sum(1 for trade in trades if trade["profit"] is not None and trade["profit"] > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0.0

    # Max Drawdown calculation
    max_drawdown = 0.0
    running_max = float("-inf")

    for value in equity_curve:
        running_max = max(running_max, value)
        if running_max == 0:
            continue
        drawdown = (value - running_max) / running_max
        max_drawdown = min(max_drawdown, drawdown)

    return BacktestResult(
        total_return=balance - 1.0,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        trades=trades,
        equity_curve=equity_curve,
    )

