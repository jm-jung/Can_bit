"""
Backtest report generation and saving utilities.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.backtest.engine import BacktestResult
from src.core.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

BACKTEST_REPORTS_DIR = PROJECT_ROOT / "data" / "backtest_reports"
BACKTEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_backtest_report(
    result: BacktestResult,
    strategy_name: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    start_date: str | None = None,
    end_date: str | None = None,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
) -> Path:
    """
    Save backtest result to JSON file.
    
    Args:
        result: BacktestResult dictionary
        strategy_name: Strategy identifier (e.g., "ml_xgb")
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date (optional, for filename)
        end_date: End date (optional, for filename)
        long_threshold: Long threshold used (optional)
        short_threshold: Short threshold used (optional)
    
    Returns:
        Path to saved report file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename
    filename_parts = [strategy_name, symbol, timeframe]
    if start_date:
        filename_parts.append(start_date.replace("-", ""))
    if end_date:
        filename_parts.append(end_date.replace("-", ""))
    filename_parts.append(timestamp)
    filename = "_".join(filename_parts) + ".json"
    
    report_path = BACKTEST_REPORTS_DIR / filename
    
    # Prepare report data
    report_data = {
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": timestamp,
        "start_date": start_date,
        "end_date": end_date,
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "commission_rate": getattr(settings, "COMMISSION_RATE", 0.0004),
        "slippage_rate": getattr(settings, "SLIPPAGE_RATE", 0.0005),
        "metrics": {
            "total_return": result["total_return"],
            "win_rate": result["win_rate"],
            "max_drawdown": result["max_drawdown"],
            "total_trades": result["total_trades"],
            "long_trades": result["long_trades"],
            "short_trades": result["short_trades"],
            "avg_profit": result["avg_profit"],
            "median_profit": result["median_profit"],
            "avg_win": result["avg_win"],
            "avg_loss": result["avg_loss"],
            "max_consecutive_wins": result["max_consecutive_wins"],
            "max_consecutive_losses": result["max_consecutive_losses"],
        },
        "equity_curve_length": len(result["equity_curve"]),
        "trades_count": len(result["trades"]),
    }
    
    # Calculate additional metrics
    equity_curve = result["equity_curve"]
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        report_data["metrics"]["sharpe_ratio"] = float(
            returns.mean() / returns.std() if returns.std() > 0 else 0.0
        )
        report_data["metrics"]["volatility"] = float(returns.std())
        report_data["metrics"]["mean_return"] = float(returns.mean())
    
    # Separate long/short statistics
    long_trades_list = [t for t in result["trades"] if t["direction"] == "LONG"]
    short_trades_list = [t for t in result["trades"] if t["direction"] == "SHORT"]
    
    if long_trades_list:
        long_profits = [t["profit"] for t in long_trades_list if t["profit"] is not None]
        if long_profits:
            long_wins = [p for p in long_profits if p > 0]
            report_data["metrics"]["long_win_rate"] = len(long_wins) / len(long_profits)
            report_data["metrics"]["long_avg_profit"] = float(pd.Series(long_profits).mean())
            report_data["metrics"]["long_avg_win"] = float(pd.Series(long_wins).mean()) if long_wins else 0.0
            report_data["metrics"]["long_avg_loss"] = float(
                pd.Series([p for p in long_profits if p <= 0]).mean()
            ) if any(p <= 0 for p in long_profits) else 0.0
    
    if short_trades_list:
        short_profits = [t["profit"] for t in short_trades_list if t["profit"] is not None]
        if short_profits:
            short_wins = [p for p in short_profits if p > 0]
            report_data["metrics"]["short_win_rate"] = len(short_wins) / len(short_profits)
            report_data["metrics"]["short_avg_profit"] = float(pd.Series(short_profits).mean())
            report_data["metrics"]["short_avg_win"] = float(pd.Series(short_wins).mean()) if short_wins else 0.0
            report_data["metrics"]["short_avg_loss"] = float(
                pd.Series([p for p in short_profits if p <= 0]).mean()
            ) if any(p <= 0 for p in short_profits) else 0.0
    
    # Save to JSON
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Saved backtest report to {report_path}")
    
    return report_path


def print_backtest_summary(result: BacktestResult, strategy_name: str = "ML Strategy") -> None:
    """
    Print formatted backtest summary to console.
    
    Args:
        result: BacktestResult dictionary
        strategy_name: Strategy name for display
    """
    commission_rate = getattr(settings, "COMMISSION_RATE", 0.0004)
    slippage_rate = getattr(settings, "SLIPPAGE_RATE", 0.0005)
    
    print("")
    print("=" * 60)
    print(f"Backtest Summary: {strategy_name}")
    print("=" * 60)
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Win Rate: {result['win_rate']:.2%}")
    print(f"Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"Total Trades: {result['total_trades']} (Long: {result['long_trades']}, Short: {result['short_trades']})")
    # Note: HOLD count is logged separately in engine.py, not included in total_trades
    if 'hold_count' in result:
        print(f"HOLD Signals: {result['hold_count']} (not included in trades)")
    print(f"Avg Profit: {result['avg_profit']:.4%}")
    print(f"Avg Win: {result['avg_win']:.4%}")
    print(f"Avg Loss: {result['avg_loss']:.4%}")
    print(f"Max Consecutive Wins: {result['max_consecutive_wins']}")
    print(f"Max Consecutive Losses: {result['max_consecutive_losses']}")
    print(f"Commission Rate: {commission_rate:.4f} ({commission_rate*100:.2f}%)")
    print(f"Slippage Rate: {slippage_rate:.4f} ({slippage_rate*100:.2f}%)")
    
    # Calculate Sharpe if possible
    equity_curve = result["equity_curve"]
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std()
            print(f"Sharpe Ratio: {sharpe:.4f}")
    
    print("=" * 60)
    print("")

