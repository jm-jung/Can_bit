"""
Entry point for ML threshold optimization.
"""
from __future__ import annotations

import logging
import argparse

import numpy as np

from src.core.config import settings
from src.backtest.metrics import get_metric_function
from src.backtest.engine_optimized_wrapper import run_backtest_with_ml_optimized
from src.optimization.threshold_optimizer import (
    get_threshold_result_path,
    optimize_threshold_for_strategy,
    save_threshold_result,
    ThresholdOptimizerResult,
)
from src.strategies.ml_xgb import ml_xgb_strategy
from src.strategies.ml_lstm_attn import ml_lstm_attn_strategy

logger = logging.getLogger(__name__)


STRATEGY_REGISTRY = {
    "ml_xgb": ml_xgb_strategy,
    "ml_lstm_attn": ml_lstm_attn_strategy,
}


def run_threshold_optimization_for_ml_strategy(
    strategy_name: str = "ml_xgb",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    feature_preset: str = "extended_safe",
    metric_name: str | None = None,
    long_threshold_min: float | None = None,
    long_threshold_max: float | None = None,
    long_threshold_step: float | None = None,
    short_threshold_min: float | None = None,
    short_threshold_max: float | None = None,
    short_threshold_step: float | None = None,
    user_provided_short_range: bool = False,  # True if user explicitly provided --short-min/--short-max/--short-step
    # Signal filter parameters (for LSTM 3-class model)
    flat_threshold_min: float | None = None,
    flat_threshold_max: float | None = None,
    flat_threshold_step: float | None = None,
    confidence_margin_min: float | None = None,
    confidence_margin_max: float | None = None,
    confidence_margin_step: float | None = None,
    min_proba_dominance_min: float | None = None,
    min_proba_dominance_max: float | None = None,
    min_proba_dominance_step: float | None = None,
    save_result: bool = True,
    use_parallel: bool = True,
    n_jobs: int = -1,
    rebuild_proba_cache: bool = False,
    nthread: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    strategy_mode: str | None = None,
    min_trades_in: int | None = None,
    min_trades_out: int | None = None,
    min_sharpe_out: float | None = None,
) -> ThresholdOptimizerResult:
    """
    Run threshold optimization for ML strategy.
    
    Args:
        strategy_name: Name of the strategy
        symbol: Trading symbol
        timeframe: Timeframe
        metric_name: Metric to optimize ("total_return" or "sharpe")
        long_threshold_min: Minimum long threshold (default: from settings)
        long_threshold_max: Maximum long threshold (default: from settings)
        long_threshold_step: Step size for long threshold (default: from settings)
        short_threshold_min: Minimum short threshold (default: from settings)
        short_threshold_max: Maximum short threshold (default: from settings)
        short_threshold_step: Step size for short threshold (default: from settings)
        save_result: If True, save result to JSON file
        rebuild_proba_cache: If True, force rebuild of prediction cache even if cache exists
        nthread: Number of CPU threads for XGBoost prediction (default: auto-detected)
        start_date: Filter OHLCV and predictions to samples at or after this date (YYYY-MM-DD format)
        end_date: Filter OHLCV and predictions to samples at or before this date (YYYY-MM-DD format)
        strategy_mode: Strategy mode ("both", "long_only", "short_only") or None for default
        min_trades_in: Minimum trades in in-sample period (default: from config)
        min_trades_out: Minimum trades in out-of-sample period (default: from config)
        min_sharpe_out: Minimum acceptable out-of-sample Sharpe (default: from config)
    
    Returns:
        ThresholdOptimizerResult
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unsupported strategy '{strategy_name}'. Available: {list(STRATEGY_REGISTRY.keys())}")

    # Get metric function
    if metric_name is None:
        metric_name = settings.THRESHOLD_OPTIMIZATION_METRIC
    
    metric_fn = get_metric_function(metric_name)
    
    # Get threshold ranges from settings or parameters
    long_min = long_threshold_min if long_threshold_min is not None else settings.LONG_THRESHOLD_MIN
    long_max = long_threshold_max if long_threshold_max is not None else settings.LONG_THRESHOLD_MAX
    long_step = long_threshold_step if long_threshold_step is not None else settings.LONG_THRESHOLD_STEP
    
    short_min = short_threshold_min if short_threshold_min is not None else settings.SHORT_THRESHOLD_MIN
    short_max = short_threshold_max if short_threshold_max is not None else settings.SHORT_THRESHOLD_MAX
    short_step = short_threshold_step if short_threshold_step is not None else settings.SHORT_THRESHOLD_STEP
    
    # Generate threshold candidates
    long_candidates = list(np.arange(long_min, long_max + long_step, long_step))
    short_candidates = list(np.arange(short_min, short_max + short_step, short_step))
    
    # Add None to short_candidates to test long-only mode
    # BUT: If user explicitly provided short range (--short-min/--short-max/--short-step),
    #      do NOT add None (user wants numeric thresholds only)
    if not user_provided_short_range:
        short_candidates.append(None)
        logger.info(
            "[ThresholdOptimizer] Short threshold range not explicitly provided by user. "
            "Adding None to short_candidates to test long-only mode."
        )
    else:
        logger.info(
            "[ThresholdOptimizer] Short threshold range explicitly provided by user. "
            "NOT adding None to short_candidates (numeric thresholds only)."
        )
    
    # Generate signal filter candidates (for LSTM 3-class model)
    # flat_threshold: None (disabled) + optional range
    flat_threshold_candidates = [None]  # Always include None (disabled)
    if flat_threshold_min is not None and flat_threshold_max is not None and flat_threshold_step is not None:
        flat_range = list(np.arange(flat_threshold_min, flat_threshold_max + flat_threshold_step, flat_threshold_step))
        flat_threshold_candidates.extend(flat_range)
        logger.info(
            f"[ThresholdOptimizer] FLAT threshold candidates: {len(flat_threshold_candidates)} "
            f"(None + range [{flat_threshold_min:.3f}, {flat_threshold_max:.3f}], step={flat_threshold_step:.3f})"
        )
    
    # confidence_margin: 0.0 (disabled) + optional range
    confidence_margin_candidates = [0.0]  # Always include 0.0 (disabled)
    if confidence_margin_min is not None and confidence_margin_max is not None and confidence_margin_step is not None:
        conf_range = list(np.arange(confidence_margin_min, confidence_margin_max + confidence_margin_step, confidence_margin_step))
        confidence_margin_candidates.extend(conf_range)
        logger.info(
            f"[ThresholdOptimizer] Confidence margin candidates: {len(confidence_margin_candidates)} "
            f"(0.0 + range [{confidence_margin_min:.3f}, {confidence_margin_max:.3f}], step={confidence_margin_step:.3f})"
        )
    
    # min_proba_dominance: 0.0 (disabled) + optional range
    min_proba_dominance_candidates = [0.0]  # Always include 0.0 (disabled)
    if min_proba_dominance_min is not None and min_proba_dominance_max is not None and min_proba_dominance_step is not None:
        dom_range = list(np.arange(min_proba_dominance_min, min_proba_dominance_max + min_proba_dominance_step, min_proba_dominance_step))
        min_proba_dominance_candidates.extend(dom_range)
        logger.info(
            f"[ThresholdOptimizer] Min proba dominance candidates: {len(min_proba_dominance_candidates)} "
            f"(0.0 + range [{min_proba_dominance_min:.3f}, {min_proba_dominance_max:.3f}], step={min_proba_dominance_step:.3f})"
        )
    
    logger.info("=" * 60)
    logger.info("ML Threshold Optimization")
    logger.info("=" * 60)
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Symbol: {symbol}, Timeframe: {timeframe}")
    if strategy_name == "ml_xgb":
        logger.info(f"Feature preset: {feature_preset}")
    elif strategy_name == "ml_lstm_attn":
        logger.info(f"LSTM-Attention strategy: using cached probabilities from ml_proba_cache")
    if start_date or end_date:
        logger.info(f"Date range filter: start_date={start_date}, end_date={end_date}")
    logger.info(f"Metric: {metric_name}")
    
    # Resolve strategy_mode and min requirements for logging
    resolved_strategy_mode = strategy_mode if strategy_mode is not None else getattr(settings, "ML_STRATEGY_MODE_DEFAULT", "both")
    resolved_min_trades_in = min_trades_in if min_trades_in is not None else getattr(settings, "MIN_TRADES_IN_DEFAULT", 500)
    resolved_min_trades_out = min_trades_out if min_trades_out is not None else getattr(settings, "MIN_TRADES_OUT_DEFAULT", 50)
    resolved_min_sharpe_out = min_sharpe_out if min_sharpe_out is not None else getattr(settings, "MIN_SHARPE_OUT_DEFAULT", 0.0)
    
    logger.info(f"Strategy mode: {resolved_strategy_mode}")
    logger.info(f"Optimization constraints: min_trades_in={resolved_min_trades_in}, min_trades_out={resolved_min_trades_out}, min_sharpe_out={resolved_min_sharpe_out}")
    logger.info(f"Long threshold range: [{long_min:.3f}, {long_max:.3f}] step={long_step:.3f}")
    logger.info(f"Short threshold range: [{short_min:.3f}, {short_max:.3f}] step={short_step:.3f}")
    
    # Log short grid details including None inclusion
    short_has_none = None in short_candidates
    short_none_reason = (
        "user did not provide short range (default behavior)"
        if short_has_none and not user_provided_short_range
        else "N/A (None not included)"
    )
    logger.info(
        f"Short threshold grid: count={len(short_candidates)}, "
        f"includes_None={short_has_none}, reason={short_none_reason}, "
        f"user_provided_short_range={user_provided_short_range}"
    )
    
    # Calculate total combinations (including signal filters if provided)
    base_combinations = len(long_candidates) * len(short_candidates)
    filter_multiplier = (
        len(flat_threshold_candidates) *
        len(confidence_margin_candidates) *
        len(min_proba_dominance_candidates)
    )
    total_combinations = base_combinations * filter_multiplier
    
    logger.info(
        f"Total combinations: {total_combinations} "
        f"(base: {base_combinations} threshold combinations × "
        f"{filter_multiplier} filter combinations)"
    )
    
    if filter_multiplier > 1:
        logger.info(
            f"Signal filter grids: "
            f"flat_th={len(flat_threshold_candidates)}, "
            f"conf_margin={len(confidence_margin_candidates)}, "
            f"min_dominance={len(min_proba_dominance_candidates)}"
        )
    
    # Resolve strategy_mode and min requirements
    if strategy_mode is None:
        strategy_mode = getattr(settings, "ML_STRATEGY_MODE_DEFAULT", "both")
    if min_trades_in is None:
        min_trades_in = getattr(settings, "MIN_TRADES_IN_DEFAULT", 500)
    if min_trades_out is None:
        min_trades_out = getattr(settings, "MIN_TRADES_OUT_DEFAULT", 50)
    if min_sharpe_out is None:
        min_sharpe_out = getattr(settings, "MIN_SHARPE_OUT_DEFAULT", 0.0)
    
    # For LSTM strategy, default to serial execution to avoid MemoryError
    # User can still override with --use-parallel
    if strategy_name == "ml_lstm_attn":
        # For LSTM, be more conservative with parallel execution
        # If use_parallel is True and n_jobs is -1 (defaults), assume it wasn't explicitly set
        # and default to False for LSTM to avoid MemoryError
        if use_parallel and n_jobs == -1:
            # Likely using defaults - change to serial for LSTM
            use_parallel = False
            logger.info(
                "[ThresholdOptimizer] ml_lstm_attn: Defaulting to serial execution to avoid MemoryError. "
                "Use --use-parallel to enable parallel mode (n_jobs will be limited to 4 for memory safety)."
            )
        elif use_parallel:
            # User explicitly enabled parallel - limit n_jobs for safety
            import os
            max_safe_jobs = min(4, os.cpu_count() or 1)
            if n_jobs == -1 or n_jobs > max_safe_jobs:
                n_jobs = max_safe_jobs
                logger.info(
                    f"[ThresholdOptimizer] ml_lstm_attn: Limiting n_jobs to {n_jobs} for memory safety "
                    "(LSTM backtests use more memory than XGBoost)"
                )
    
    logger.info(
        "Running ML threshold optimization: strategy=%s, symbol=%s, timeframe=%s, "
        "feature_preset=%s, use_parallel=%s, n_jobs=%s",
        strategy_name,
        symbol,
        timeframe,
        feature_preset,
        use_parallel,
        n_jobs,
    )
    logger.info(
        "Strategy mode: %s, min_trades_in=%d, min_trades_out=%d, min_sharpe_out=%.3f",
        strategy_mode,
        min_trades_in,
        min_trades_out,
        min_sharpe_out,
    )
    # Check if signal filter parameters are provided (for future grid-search integration)
    has_signal_filters = (
        flat_threshold_min is not None or
        confidence_margin_min is not None or
        min_proba_dominance_min is not None
    )
    
    if has_signal_filters:
        logger.warning(
            "[ThresholdOptimizer] Extra signal-filter params provided "
            "(flat_threshold, confidence_margin, min_proba_dominance) "
            "but grid-search integration is not enabled yet; running with long/short only. "
            "These parameters will be ignored for now."
        )
    
    logger.info("=" * 60)
    
    # Data loader (closure with fixed params)
    def data_loader():
        """Load data for backtest (no parameters needed, uses global data)."""
        return None  # Backtest function loads data internally
    
    # Run optimization (with overfitting awareness)
    result = optimize_threshold_for_strategy(
        strategy_func=STRATEGY_REGISTRY[strategy_name],
        data_loader=data_loader,
        metric_fn=metric_fn,
        long_threshold_candidates=long_candidates,
        short_threshold_candidates=short_candidates,
        fixed_strategy_kwargs={},
        run_backtest_func=lambda long_thr, short_thr: run_backtest_with_ml_optimized(
            long_thr,
            short_thr,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            feature_preset=feature_preset,
        ),
        strategy_name=strategy_name,
        use_overfit_aware=True,
        symbol=symbol,
        timeframe=timeframe,
        feature_preset=feature_preset,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
        force_rebuild_proba_cache=rebuild_proba_cache,
        nthread=nthread,
        start_date=start_date,
        end_date=end_date,
        strategy_mode=strategy_mode,
        min_trades_in=min_trades_in,
        min_trades_out=min_trades_out,
        min_sharpe_out=min_sharpe_out,
        metric_name=metric_name,  # Pass original metric name for worker reconstruction
    )
    
    # Save result if requested
    if save_result:
        threshold_path = get_threshold_result_path(strategy_name, symbol, timeframe)
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        save_threshold_result(
            result,
            threshold_path,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
        )
        logger.info(f"Saved optimized thresholds to {threshold_path}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Optimization Complete - Final Summary")
    logger.info("=" * 60)
    logger.info(f"Strategy: {strategy_name}, Symbol: {symbol}, Timeframe: {timeframe}")
    if result.strategy_mode:
        logger.info(f"Strategy mode: {result.strategy_mode}")
    logger.info(f"Best long threshold: {result.best_long_threshold:.3f}")
    logger.info(f"Best short threshold: {result.best_short_threshold}")
    logger.info(f"Best metric value (out-of-sample): {result.best_metric_value:.6f}")
    
    if result.sharpe_in_sample is not None:
        logger.info(f"Sharpe (in-sample): {result.sharpe_in_sample:.6f}")
        logger.info(f"Sharpe (out-of-sample): {result.sharpe_out_sample:.6f}")
        logger.info(f"Gap (in - out): {result.gap_in_out:.6f}")
        logger.info(f"Score (overfit-adjusted): {result.score_overfit_adjusted:.6f}")
        logger.info(f"Total trades: in={result.total_trades_in}, out={result.total_trades_out}")
    
    if result.min_trades_in is not None or result.min_trades_out is not None or result.min_sharpe_out is not None:
        logger.info(
            f"Optimization constraints: min_trades_in={result.min_trades_in}, "
            f"min_trades_out={result.min_trades_out}, min_sharpe_out={result.min_sharpe_out}"
        )
    
    logger.info("=" * 60)
    
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize ML strategy thresholds. "
        "Supports ml_xgb (XGBoost) and ml_lstm_attn (LSTM-Attention) strategies."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ml_xgb",
        choices=list(STRATEGY_REGISTRY.keys()),
        help="Strategy name: 'ml_xgb' (XGBoost) or 'ml_lstm_attn' (LSTM-Attention). "
        "ml_lstm_attn uses cached probabilities from ml_proba_cache."
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe (default: 1m)")
    parser.add_argument(
        "--feature-preset",
        type=str,
        default="extended_safe",
        choices=["base", "extended_safe", "extended_full"],
        help="Feature preset for ml_xgb strategy only (default: extended_safe). "
        "Ignored for ml_lstm_attn (uses its own feature extraction)."
    )
    parser.add_argument("--metric", type=str, default=None, help="Metric name (e.g., sharpe)")
    parser.add_argument("--long-min", type=float, default=None)
    parser.add_argument("--long-max", type=float, default=None)
    parser.add_argument("--long-step", type=float, default=None)
    parser.add_argument("--short-min", type=float, default=None)
    parser.add_argument("--short-max", type=float, default=None)
    parser.add_argument("--short-step", type=float, default=None)
    
    # Signal filter parameters (for LSTM 3-class model)
    parser.add_argument("--flat-th-min", type=float, default=None, help="FLAT threshold minimum (None = disabled)")
    parser.add_argument("--flat-th-max", type=float, default=None, help="FLAT threshold maximum")
    parser.add_argument("--flat-th-step", type=float, default=None, help="FLAT threshold step")
    parser.add_argument("--conf-margin-min", type=float, default=None, help="Confidence margin minimum (0.0 = disabled)")
    parser.add_argument("--conf-margin-max", type=float, default=None, help="Confidence margin maximum")
    parser.add_argument("--conf-margin-step", type=float, default=None, help="Confidence margin step")
    parser.add_argument("--min-dominance-min", type=float, default=None, help="Min proba dominance minimum (0.0 = disabled)")
    parser.add_argument("--min-dominance-max", type=float, default=None, help="Min proba dominance maximum")
    parser.add_argument("--min-dominance-step", type=float, default=None, help="Min proba dominance step")
    
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON result")
    
    # Strategy mode and optimization constraints
    parser.add_argument(
        "--strategy-mode",
        type=str,
        default=None,
        choices=["both", "long_only", "short_only"],
        help="Strategy mode: 'both' (default), 'long_only', or 'short_only'"
    )
    parser.add_argument(
        "--min-trades-in",
        type=int,
        default=None,
        help="Minimum trades required in in-sample period (default: from config)"
    )
    parser.add_argument(
        "--min-trades-out",
        type=int,
        default=None,
        help="Minimum trades required in out-of-sample period (default: from config)"
    )
    parser.add_argument(
        "--min-sharpe-out",
        type=float,
        default=None,
        help="Minimum acceptable out-of-sample Sharpe ratio (default: from config)"
    )
    
    # Parallel execution options
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument(
        "--use-parallel",
        dest="use_parallel",
        action="store_true",
        help="Use parallel execution for threshold optimization (default).",
    )
    parallel_group.add_argument(
        "--no-parallel",
        dest="use_parallel",
        action="store_false",
        help="Disable parallel execution for threshold optimization.",
    )
    # Default depends on strategy: LSTM defaults to False, XGBoost defaults to True
    # We'll set default to True here, but override for LSTM in run_threshold_optimization_for_ml_strategy
    parser.set_defaults(use_parallel=True)
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of worker processes for parallel execution (default: -1 = use all CPUs).",
    )
    
    # Prediction cache and XGBoost thread options
    parser.add_argument(
        "--rebuild-proba-cache",
        action="store_true",
        help="Force rebuild of XGBoost probability cache even if cache file exists.",
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=None,
        help="Number of CPU threads for XGBoost; if omitted, it will be auto-detected.",
    )
    
    # Date range filtering options
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Filter OHLCV and ML predictions to samples at or after this date (inclusive). Format: YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Filter OHLCV and ML predictions to samples at or before this date (inclusive). Format: YYYY-MM-DD",
    )
    
    return parser.parse_args()


def main():
    """
    Main CLI entry point for ML threshold optimization.
    """
    # Configure logging at the start
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("ML Threshold Optimizer CLI")
    logger.info("=" * 60)
    
    try:
        args = _parse_args()
        
        # Log parsed arguments for debugging
        logger.info(f"[ThresholdOptimizer] Starting with args: strategy={args.strategy}, "
                   f"symbol={args.symbol}, timeframe={args.timeframe}, "
                   f"feature_preset={args.feature_preset}, metric={args.metric}")
        
        # Validate strategy is supported
        if args.strategy not in STRATEGY_REGISTRY:
            logger.error(
                f"[ThresholdOptimizer] Unsupported strategy '{args.strategy}'. "
                f"Available strategies: {list(STRATEGY_REGISTRY.keys())}"
            )
            raise ValueError(
                f"Unsupported strategy '{args.strategy}'. "
                f"Available: {list(STRATEGY_REGISTRY.keys())}"
            )
        
        # Detect if user explicitly provided short range
        # User is considered to have provided short range if ANY of --short-min/--short-max/--short-step is set
        user_provided_short_range = (
            args.short_min is not None or
            args.short_max is not None or
            args.short_step is not None
        )
        
        if user_provided_short_range:
            logger.info(
                "[ThresholdOptimizer] User explicitly provided short threshold range. "
                "None will NOT be added to short_candidates."
            )
        
        # Run optimization
        result = run_threshold_optimization_for_ml_strategy(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            feature_preset=args.feature_preset,
            metric_name=args.metric,
            long_threshold_min=args.long_min,
            long_threshold_max=args.long_max,
            long_threshold_step=args.long_step,
            short_threshold_min=args.short_min,
            short_threshold_max=args.short_max,
            short_threshold_step=args.short_step,
            user_provided_short_range=user_provided_short_range,
            flat_threshold_min=args.flat_th_min,
            flat_threshold_max=args.flat_th_max,
            flat_threshold_step=args.flat_th_step,
            confidence_margin_min=args.conf_margin_min,
            confidence_margin_max=args.conf_margin_max,
            confidence_margin_step=args.conf_margin_step,
            min_proba_dominance_min=args.min_dominance_min,
            min_proba_dominance_max=args.min_dominance_max,
            min_proba_dominance_step=args.min_dominance_step,
            save_result=not args.no_save,
            use_parallel=args.use_parallel,
            n_jobs=args.n_jobs,
            rebuild_proba_cache=args.rebuild_proba_cache,
            nthread=args.nthread,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy_mode=args.strategy_mode,
            min_trades_in=args.min_trades_in,
            min_trades_out=args.min_trades_out,
            min_sharpe_out=args.min_sharpe_out,
        )
        
        logger.info("[ThresholdOptimizer] Optimization completed successfully.")
        return result
        
    except KeyboardInterrupt:
        logger.warning("[ThresholdOptimizer] Interrupted by user.")
        raise
    except Exception as e:
        logger.exception(
            f"[ThresholdOptimizer] Fatal error during optimization: {type(e).__name__}: {e}"
        )
        raise


if __name__ == "__main__":
    main()

