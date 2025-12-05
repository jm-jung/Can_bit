"""
Wrapper function for threshold optimizer compatibility.
"""
from src.backtest.engine import BacktestResult, run_backtest_with_ml


def run_backtest_with_ml_optimized(
    long_threshold: float,
    short_threshold: float | None = None,
    *,
    strategy_name: str = "ml_xgb",
    symbol: str | None = None,
    timeframe: str | None = None,
    feature_preset: str = "extended_safe",
) -> BacktestResult:
    """
    Wrapper for run_backtest_with_ml that matches the signature expected by optimizer.
    
    Args:
        long_threshold: Probability threshold for LONG signal
        short_threshold: Probability threshold for SHORT signal (optional)
        strategy_name: Strategy identifier
        symbol: Trading symbol
        timeframe: Timeframe
        feature_preset: Feature preset for ml_xgb strategy
    
    Returns:
        BacktestResult
    """
    return run_backtest_with_ml(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_threshold=False,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        feature_preset=feature_preset,
    )

