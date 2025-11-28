"""
Backtest metrics for strategy evaluation.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def total_return_metric(backtest_result: Any) -> float:
    """
    Calculate total return metric.
    
    Higher is better. This is simply the total_return from the backtest result.
    
    Args:
        backtest_result: Backtest result dictionary
    
    Returns:
        Total return as float
    """
    return float(backtest_result["total_return"])


def sharpe_ratio_metric(
    backtest_result: Any,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sharpe ratio metric.
    
    Higher is better. Sharpe ratio = (mean_return - risk_free_rate) / std_return
    
    Args:
        backtest_result: Backtest result dictionary
        risk_free_rate: Risk-free rate (default: 0.0)
    
    Returns:
        Sharpe ratio as float
    """
    equity_curve = backtest_result["equity_curve"]
    
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate returns from equity curve
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    if len(returns) == 0:
        return 0.0
    
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return float(sharpe)


def get_metric_function(metric_name: str) -> Callable[[Any], float]:
    """
    Get metric function by name.
    
    Args:
        metric_name: Name of the metric ("total_return" or "sharpe")
    
    Returns:
        Metric function that takes a backtest result and returns float
    
    Raises:
        ValueError: If metric_name is not recognized
    """
    metric_map = {
        "total_return": total_return_metric,
        "sharpe": sharpe_ratio_metric,
        "sharpe_ratio": sharpe_ratio_metric,
    }
    
    if metric_name.lower() not in metric_map:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Available metrics: {list(metric_map.keys())}"
        )
    
    return metric_map[metric_name.lower()]

