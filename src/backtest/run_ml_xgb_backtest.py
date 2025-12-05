"""
Long-period backtest entry point for ML XGBoost strategy.

Usage:
    python -m src.backtest.run_ml_xgb_backtest --symbol BTCUSDT --timeframe 1m --start-date 2024-01-01 --end-date 2024-06-30 --use-optimized-threshold
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from src.backtest.backtest_report import print_backtest_summary, save_backtest_report
from src.backtest.engine import run_backtest_with_ml
from src.core.config import settings
from src.research.ml_strategy_research import run_all_experiments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run long-period backtest for ML XGBoost strategy"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ml_xgb",
        choices=["ml_xgb", "ml_lstm_attn"],
        help="Strategy name (default: ml_xgb)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Timeframe (default: 1m)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If not provided, uses all available data.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). If not provided, uses all available data.",
    )
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=None,
        help="Long threshold override (default: from optimized thresholds or settings)",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=None,
        help="Short threshold override (default: from optimized thresholds or settings)",
    )
    parser.add_argument(
        "--use-optimized-threshold",
        action="store_true",
        help="Use optimized ML thresholds from data/thresholds folder",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save backtest report to file",
    )
    parser.add_argument(
        "--research-mode",
        action="store_true",
        default=False,
        help="Run research experiments instead of normal backtest",
    )
    parser.add_argument(
        "--signal-confirmation-bars",
        type=int,
        default=1,
        help="Number of consecutive bars required to confirm signal (default: 1, no smoothing)",
    )
    parser.add_argument(
        "--use-trend-filter",
        action="store_true",
        default=False,
        help="Apply EMA-based trend filter (default: False)",
    )
    parser.add_argument(
        "--trend-ema-window",
        type=int,
        default=200,
        help="EMA window for trend filter (default: 200)",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=None,
        help="Take profit percentage (e.g., 0.003 = 0.3%), None to disable (default: None)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Stop loss percentage (e.g., 0.002 = 0.2%), None to disable (default: None)",
    )
    parser.add_argument(
        "--feature-preset",
        type=str,
        default="extended_safe",
        choices=["base", "extended_safe", "extended_full"],
        help="Feature preset for ml_xgb strategy (default: extended_safe)",
    )
    
    return parser.parse_args()


def filter_data_by_date(
    df,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Filter DataFrame by date range.
    
    Args:
        df: DataFrame with timestamp column
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        Filtered DataFrame
    """
    if start_date is None and end_date is None:
        return df
    
    if "timestamp" not in df.columns:
        logger.warning("DataFrame does not have 'timestamp' column. Skipping date filter.")
        return df
    
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        df = df[df["timestamp"] >= start_dt]
        logger.info(f"Filtered data: start_date >= {start_date}")
    
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        # Include the entire end date
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
        df = df[df["timestamp"] <= end_dt]
        logger.info(f"Filtered data: end_date <= {end_date}")
    
    logger.info(f"Filtered data shape: {df.shape}")
    return df


def main():
    """Main entry point for long-period backtest."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Long-Period ML Backtest")
    logger.info("=" * 60)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
    if args.strategy == "ml_xgb":
        logger.info(f"Feature preset: {args.feature_preset}")
    logger.info(f"Start date: {args.start_date or 'All available'}")
    logger.info(f"End date: {args.end_date or 'All available'}")
    logger.info("=" * 60)
    
    # Log threshold usage flag
    logger.info(
        "[ML Backtest] use_optimized_threshold=%s",
        args.use_optimized_threshold
    )
    
    # Research mode: run experiments instead of normal backtest
    if args.research_mode:
        logger.info("[ML Backtest] Research mode enabled - running experiments...")
        results = run_all_experiments(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            use_optimized_threshold=args.use_optimized_threshold,
        )
        return results
    
    # Normal backtest mode
    # Note: Date filtering would need to be implemented in the data loading layer
    # For now, we run backtest on all available data
    # TODO: Implement date filtering in load_ohlcv_df or get_df_with_indicators
    
    # Run backtest
    result = run_backtest_with_ml(
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        use_optimized_threshold=args.use_optimized_threshold,
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        feature_preset=args.feature_preset,
        signal_confirmation_bars=args.signal_confirmation_bars,
        use_trend_filter=args.use_trend_filter,
        trend_ema_window=args.trend_ema_window,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
    )
    
    # Print summary
    print_backtest_summary(result, strategy_name=args.strategy)
    
    # Save report
    if not args.no_save:
        report_path = save_backtest_report(
            result=result,
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
        )
        logger.info(f"Backtest report saved to: {report_path}")
    
    return result


if __name__ == "__main__":
    import pandas as pd
    main()

