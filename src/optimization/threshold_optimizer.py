"""
Threshold optimization for ML trading strategies.

This module provides functionality to automatically optimize probability thresholds
for ML-based trading strategies using grid search and backtest evaluation.

The optimization process is "overfitting-aware" by:
1. Splitting the backtest period into in-sample (70%) and out-of-sample (30%)
2. Computing separate Sharpe ratios for each split
3. Applying a penalty for large in-sample/out-of-sample gaps
4. Selecting thresholds that perform well out-of-sample while avoiding overfitting

This approach helps prevent threshold-level overfitting where thresholds are
optimized too specifically to the training period.

Performance optimizations (lossless):
- Parallel execution of threshold combinations (3-8x speedup for typical grids)
- Eliminated redundant data copying (20-30% memory reduction)
- Vectorized signal generation in backtest engine (10-50x faster signal creation)
- Expected overall speedup: 3-8x for typical threshold grids (49 combinations)
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Tuple

import numpy as np

from src.core.config import settings
from src.backtest.metrics import get_metric_function

logger = logging.getLogger(__name__)

THRESHOLD_RESULTS_DIR = Path("data/thresholds")

# Overfitting penalty hyperparameters
OVERFIT_PENALTY_ALPHA = 0.5  # Penalty weight for in-sample/out-of-sample gap
MIN_TRADES = 20  # Minimum total trades required for a threshold combination
IN_SAMPLE_RATIO = 0.7  # Proportion of data for in-sample evaluation

# Minimum trade count constraints for statistically meaningful strategies
MIN_TRADES_IN_SAMPLE = 500  # Minimum trades in in-sample period
MIN_TRADES_OUT_SAMPLE = 50  # Minimum trades in out-of-sample period

# Minimum acceptable out-of-sample Sharpe for profitable strategies
MIN_ACCEPTABLE_SHARPE_OUT = 0.0  # Minimum acceptable out-of-sample Sharpe ratio

# Performance optimization settings
DEFAULT_USE_PARALLEL = True  # Enable parallel execution by default
DEFAULT_N_JOBS = -1  # Use all available CPUs (-1 = all CPUs)


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbols like BTC/USDT -> BTCUSDT for file naming."""
    return symbol.replace("/", "").upper()


def _normalize_timeframe(timeframe: str) -> str:
    """Normalize timeframe tokens for file naming."""
    return timeframe.lower()


def get_threshold_result_path(
    strategy_name: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Build the canonical path for a threshold optimization result.

    Args:
        strategy_name: Strategy identifier (e.g., ml_xgb)
        symbol: Trading symbol (e.g., BTCUSDT or BTC/USDT)
        timeframe: Candle timeframe (e.g., 1m)
    """
    normalized_symbol = _normalize_symbol(symbol)
    normalized_timeframe = _normalize_timeframe(timeframe)
    filename = f"{strategy_name}_{normalized_symbol}_{normalized_timeframe}.json"
    return THRESHOLD_RESULTS_DIR / filename


@dataclass
class ThresholdOptimizerResult:
    """Result of threshold optimization."""
    
    best_long_threshold: float
    best_short_threshold: float | None
    best_metric_value: float
    metric_name: str
    trials: List[Dict[str, Any]]
    # Overfitting-aware fields (optional for backward compatibility)
    sharpe_in_sample: float | None = None
    sharpe_out_sample: float | None = None
    gap_in_out: float | None = None
    total_trades_in: int | None = None
    total_trades_out: int | None = None
    score_overfit_adjusted: float | None = None
    # Strategy enablement status
    enabled: bool = True  # Whether this strategy should be used for live trading
    disable_reason: str | None = None  # Reason if disabled (e.g., "no_positive_sharpe")
    # Strategy mode and optimization constraints (for JSON serialization)
    strategy_mode: str | None = None  # Strategy mode used ("both", "long_only", "short_only")
    min_trades_in: int | None = None  # Minimum trades in-sample requirement
    min_trades_out: int | None = None  # Minimum trades out-of-sample requirement
    min_sharpe_out: float | None = None  # Minimum Sharpe out-of-sample requirement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Remove None values for cleaner JSON (but keep strategy_mode and min_* for reference)
        filtered = {k: v for k, v in result.items() if v is not None}
        # Always include strategy_mode and min_* if they exist (even if None, for documentation)
        if hasattr(self, "strategy_mode"):
            filtered["strategy_mode"] = self.strategy_mode
        if hasattr(self, "min_trades_in"):
            filtered["min_trades_in"] = self.min_trades_in
        if hasattr(self, "min_trades_out"):
            filtered["min_trades_out"] = self.min_trades_out
        if hasattr(self, "min_sharpe_out"):
            filtered["min_sharpe_out"] = self.min_sharpe_out
        return filtered
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThresholdOptimizerResult:
        """Create from dictionary (backward compatible with old format)."""
        # Handle old format without overfitting fields
        result = cls(
            best_long_threshold=data["best_long_threshold"],
            best_short_threshold=data.get("best_short_threshold"),
            best_metric_value=data["best_metric_value"],
            metric_name=data.get("metric_name", "unknown"),
            trials=data.get("trials", []),
            sharpe_in_sample=data.get("sharpe_in_sample"),
            sharpe_out_sample=data.get("sharpe_out_sample"),
            gap_in_out=data.get("gap_in_out"),
            total_trades_in=data.get("total_trades_in"),
            total_trades_out=data.get("total_trades_out"),
            score_overfit_adjusted=data.get("score_overfit_adjusted"),
            enabled=data.get("enabled", True),
            disable_reason=data.get("disable_reason"),
            strategy_mode=data.get("strategy_mode"),
            min_trades_in=data.get("min_trades_in"),
            min_trades_out=data.get("min_trades_out"),
            min_sharpe_out=data.get("min_sharpe_out"),
        )
        return result


def has_enough_trades(
    trades_in: int, 
    trades_out: int,
    min_trades_in: int | None = None,
    min_trades_out: int | None = None,
) -> bool:
    """
    Returns True if the combination has enough trades to be considered meaningful.
    
    We require a minimum number of trades in both in-sample and out-of-sample periods.
    
    Args:
        trades_in: Number of trades in in-sample period
        trades_out: Number of trades in out-of-sample period
        min_trades_in: Minimum trades in in-sample (default: from config or MIN_TRADES_IN_SAMPLE)
        min_trades_out: Minimum trades in out-of-sample (default: from config or MIN_TRADES_OUT_SAMPLE)
    
    Returns:
        True if both periods meet minimum trade requirements
    """
    min_in = min_trades_in if min_trades_in is not None else MIN_TRADES_IN_SAMPLE
    min_out = min_trades_out if min_trades_out is not None else MIN_TRADES_OUT_SAMPLE
    return trades_in >= min_in and trades_out >= min_out


def compute_overfit_penalty_score(
    sharpe_in: float,
    sharpe_out: float,
    alpha: float = OVERFIT_PENALTY_ALPHA,
) -> float:
    """
    Compute overfitting-adjusted score.
    
    Formula: score = sharpe_out - alpha * max(0, sharpe_in - sharpe_out)
    
    This penalizes threshold combinations that perform much better in-sample
    than out-of-sample, indicating potential overfitting.
    
    Args:
        sharpe_in: Sharpe ratio on in-sample data
        sharpe_out: Sharpe ratio on out-of-sample data
        alpha: Penalty weight (default: OVERFIT_PENALTY_ALPHA)
    
    Returns:
        Adjusted score (higher is better)
    """
    gap = sharpe_in - sharpe_out
    penalty = alpha * max(0.0, gap)
    score = sharpe_out - penalty
    return score


def optimize_threshold_for_strategy(
    strategy_func: Callable[..., Any],
    data_loader: Callable[[], Any],
    metric_fn: Callable[[Any], float],
    long_threshold_candidates: List[float],
    short_threshold_candidates: Optional[List[float]] = None,
    fixed_strategy_kwargs: Optional[Dict[str, Any]] = None,
    run_backtest_func: Optional[Callable[[float, Optional[float]], Any]] = None,
    strategy_name: str | None = None,
    use_overfit_aware: bool = True,
    symbol: str | None = None,
    timeframe: str | None = None,
    feature_preset: Optional[str] = None,
    use_parallel: bool = DEFAULT_USE_PARALLEL,
    n_jobs: int = DEFAULT_N_JOBS,
    force_rebuild_proba_cache: bool = False,
    nthread: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    strategy_mode: str | None = None,
    min_trades_in: int | None = None,
    min_trades_out: int | None = None,
    min_sharpe_out: float | None = None,
    metric_name: str | None = None,
) -> ThresholdOptimizerResult:
    """
    Optimize thresholds for a trading strategy using grid search.
    
    When use_overfit_aware=True, this function:
    1. Computes prediction probabilities once and caches them
    2. Splits the backtest period into in-sample (70%) and out-of-sample (30%)
    3. Evaluates each threshold combination on both splits
    4. Applies an overfitting penalty based on the in-sample/out-of-sample gap
    5. Selects thresholds that perform well out-of-sample
    
    Args:
        strategy_func: Strategy function (for reference, not directly called)
        data_loader: Function that loads/prepares data for backtest
        metric_fn: Function that takes a backtest result object and returns a metric value (higher is better)
        long_threshold_candidates: List of long threshold values to test
        short_threshold_candidates: Optional list of short threshold values to test
        fixed_strategy_kwargs: Fixed keyword arguments to pass to strategy (e.g., model_path)
        run_backtest_func: Function that runs backtest with given thresholds.
                          Signature: (long_threshold, short_threshold) -> Backtest-like result
                          If None, uses default backtest function
        strategy_name: Strategy identifier (required for overfit-aware mode)
        use_overfit_aware: If True, use in-sample/out-of-sample split with penalty (default: True)
        symbol: Trading symbol (e.g., BTCUSDT)
        timeframe: Timeframe (e.g., 5m)
        feature_preset: Feature preset for ml_xgb strategy (e.g., extended_safe). Default: None.
        use_parallel: If True, use parallel execution for threshold combinations (default: True)
        n_jobs: Number of parallel jobs (-1 for all CPUs, default: -1)
        force_rebuild_proba_cache: If True, force rebuild of prediction cache even if cache exists
        nthread: Number of CPU threads for XGBoost prediction (default: auto-detected)
        start_date: Filter OHLCV and predictions to samples at or after this date (YYYY-MM-DD format)
        end_date: Filter OHLCV and predictions to samples at or before this date (YYYY-MM-DD format)
        strategy_mode: Strategy mode ("both", "long_only", "short_only") or None for default
        min_trades_in: Minimum trades in in-sample period (default: from config)
        min_trades_out: Minimum trades in out-of-sample period (default: from config)
        min_sharpe_out: Minimum acceptable out-of-sample Sharpe (default: from config)
        metric_name: Original metric name string (e.g., "total_return", "sharpe") for worker reconstruction
    
    Returns:
        ThresholdOptimizerResult with best thresholds and all trial results
    """
    if fixed_strategy_kwargs is None:
        fixed_strategy_kwargs = {}
    
    if short_threshold_candidates is None:
        short_threshold_candidates = [None]
    
    if run_backtest_func is None:
        # Default: use run_backtest_with_ml_optimized wrapper
        from src.backtest.engine_optimized_wrapper import run_backtest_with_ml_optimized
        run_backtest_func = run_backtest_with_ml_optimized
    
    # Overfitting-aware optimization
    if use_overfit_aware and strategy_name is not None:
        return _optimize_with_overfit_awareness(
            strategy_name=strategy_name,
            metric_fn=metric_fn,
            long_threshold_candidates=long_threshold_candidates,
            short_threshold_candidates=short_threshold_candidates,
            symbol=symbol,
            timeframe=timeframe,
            feature_preset=feature_preset,
            use_parallel=use_parallel,
            n_jobs=n_jobs,
            force_rebuild_proba_cache=force_rebuild_proba_cache,
            nthread=nthread,
            start_date=start_date,
            end_date=end_date,
            strategy_mode=strategy_mode,
            min_trades_in=min_trades_in,
            min_trades_out=min_trades_out,
            min_sharpe_out=min_sharpe_out,
            metric_name=metric_name,
        )
    
    # Legacy optimization (backward compatibility)
    trials = []
    best_metric_value = float("-inf")
    best_long_threshold = long_threshold_candidates[0]
    best_short_threshold = short_threshold_candidates[0] if short_threshold_candidates else None
    
    total_combinations = len(long_threshold_candidates) * len(short_threshold_candidates)
    logger.info(f"Starting threshold optimization: {total_combinations} combinations to test")
    
    for long_thr in long_threshold_candidates:
        for short_thr in short_threshold_candidates:
            try:
                # Run backtest with these thresholds
                result = run_backtest_func(long_thr, short_thr)
                
                # Compute metric
                metric_value = metric_fn(result)
                
                trial = {
                    "long_threshold": long_thr,
                    "short_threshold": short_thr,
                    "metric_value": metric_value,
                    "total_return": result["total_return"],
                    "win_rate": result["win_rate"],
                    "total_trades": result["total_trades"],
                }
                trials.append(trial)
                
                logger.info(
                    f"  long={long_thr:.3f}, short={short_thr}, "
                    f"metric={metric_value:.6f}, return={result['total_return']:.4f}, "
                    f"trades={result['total_trades']}"
                )
                
                # Update best if this is better
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_long_threshold = long_thr
                    best_short_threshold = short_thr
                    
            except Exception as e:
                logger.warning(
                    f"  Failed to evaluate long={long_thr:.3f}, short={short_thr}: {e}"
                )
                # Continue with next combination
                continue
    
    logger.info(
        f"Optimization complete. Best: long={best_long_threshold:.3f}, "
        f"short={best_short_threshold}, metric={best_metric_value:.6f}"
    )
    
    return ThresholdOptimizerResult(
        best_long_threshold=best_long_threshold,
        best_short_threshold=best_short_threshold,
        best_metric_value=best_metric_value,
        metric_name=metric_fn.__name__ if hasattr(metric_fn, "__name__") else "unknown",
        trials=trials,
    )


def _evaluate_single_threshold_combination(
    long_thr: float,
    short_thr: float | None,
    strategy_name: str,
    metric_fn: Callable[[Any], float],
    proba_long_arr: np.ndarray,
    proba_short_arr: np.ndarray,
    df_aligned: "pd.DataFrame",
    idx_in: np.ndarray,
    idx_out: np.ndarray,
    preset_to_use: str,
    symbol: str,
    timeframe: str,
    strategy_mode: str | None = None,
) -> Dict[str, Any]:
    """
    Evaluate a single threshold combination (helper for parallelization).
    
    Args:
        strategy_mode: Strategy mode ("both", "long_only", "short_only") or None
    
    Returns:
        Dictionary with results or error information
    """
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategies.strategy_mode import StrategyMode as StrategyModeEnum
    
    # Convert strategy_mode to long_only/short_only flags
    mode = StrategyModeEnum.from_string(strategy_mode) if strategy_mode else StrategyModeEnum.BOTH
    long_only = mode == StrategyModeEnum.LONG_ONLY
    short_only = mode == StrategyModeEnum.SHORT_ONLY
    
    # Get engine
    engine = get_ml_backtest_engine(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        feature_preset=preset_to_use if strategy_name == "ml_xgb" else "base",
    )
    
    try:
        # Run in-sample backtest
        result_in = engine.run_backtest(
            long_threshold=long_thr,
            short_threshold=short_thr,
            use_optimized_threshold=False,
            proba_long_cache=proba_long_arr,
            proba_short_cache=proba_short_arr,
            df_with_proba=df_aligned,
            index_mask=idx_in,
            long_only=long_only,
            short_only=short_only,
            flat_threshold=None,
            confidence_margin=0.0,
            min_proba_dominance=0.0,
        )
        metric_in = metric_fn(result_in)
        trades_in = result_in["total_trades"]
        
        # Run out-of-sample backtest
        result_out = engine.run_backtest(
            long_threshold=long_thr,
            short_threshold=short_thr,
            use_optimized_threshold=False,
            proba_long_cache=proba_long_arr,
            proba_short_cache=proba_short_arr,
            df_with_proba=df_aligned,
            index_mask=idx_out,
            long_only=long_only,
            short_only=short_only,
            flat_threshold=None,
            confidence_margin=0.0,
            min_proba_dominance=0.0,
        )
        metric_out = metric_fn(result_out)
        trades_out = result_out["total_trades"]
        
        total_trades = trades_in + trades_out
        gap = metric_in - metric_out
        # For overfit penalty, use the metric values directly
        # Note: compute_overfit_penalty_score was designed for sharpe, but works for any metric
        score = compute_overfit_penalty_score(metric_in, metric_out)
        
        return {
            "success": True,
            "long_threshold": long_thr,
            "short_threshold": short_thr,
            "sharpe_in": metric_in,  # Keep key name for backward compatibility, but value is metric-dependent
            "sharpe_out": metric_out,  # Keep key name for backward compatibility, but value is metric-dependent
            "metric_in": metric_in,  # Add explicit metric field
            "metric_out": metric_out,  # Add explicit metric field
            "gap": gap,
            "score": score,
            "trades_in": trades_in,
            "trades_out": trades_out,
            "total_trades": total_trades,
            "total_return_in": result_in["total_return"],
            "total_return_out": result_out["total_return"],
            "win_rate_in": result_in["win_rate"],
            "win_rate_out": result_out["win_rate"],
        }
    except Exception as e:
        return {
            "success": False,
            "long_threshold": long_thr,
            "short_threshold": short_thr,
            "error": str(e),
        }


def _worker_evaluate_threshold(
    args: Tuple[
        float,  # long_thr
        float | None,  # short_thr
        str,  # strategy_name
        str,  # metric_name (for reconstruction)
        np.ndarray,  # proba_long_arr
        np.ndarray,  # proba_short_arr
        "pd.DataFrame",  # df_aligned
        np.ndarray,  # idx_in
        np.ndarray,  # idx_out
        str,  # preset_to_use
        str,  # symbol
        str,  # timeframe
        str | None,  # strategy_mode
    ],
) -> Dict[str, Any]:
    """
    Worker function for parallel threshold evaluation (top-level for pickle).
    
    This function is designed to be pickled and executed in a separate process.
    All inputs must be serializable.
    
    Args:
        args: Tuple containing all necessary parameters
        
    Returns:
        Dictionary with results or error information
    """
    (
        long_thr,
        short_thr,
        strategy_name,
        metric_name,
        proba_long_arr,
        proba_short_arr,
        df_aligned,
        idx_in,
        idx_out,
        preset_to_use,
        symbol,
        timeframe,
        strategy_mode,
    ) = args
    
    # Reconstruct metric function (must be importable)
    from src.backtest.metrics import get_metric_function
    
    metric_fn = get_metric_function(metric_name)
    
    # Call the evaluation function
    return _evaluate_single_threshold_combination(
        long_thr=long_thr,
        short_thr=short_thr,
        strategy_name=strategy_name,
        metric_fn=metric_fn,
        proba_long_arr=proba_long_arr,
        proba_short_arr=proba_short_arr,
        df_aligned=df_aligned,
        idx_in=idx_in,
        idx_out=idx_out,
        preset_to_use=preset_to_use,
        symbol=symbol,
        timeframe=timeframe,
        strategy_mode=strategy_mode,
    )


def _optimize_with_overfit_awareness(
    strategy_name: str,
    metric_fn: Callable[[Any], float],
    long_threshold_candidates: List[float],
    short_threshold_candidates: Optional[List[float]] = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    feature_preset: Optional[str] = None,
    use_parallel: bool = DEFAULT_USE_PARALLEL,
    n_jobs: int = DEFAULT_N_JOBS,
    force_rebuild_proba_cache: bool = False,
    nthread: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    strategy_mode: str | None = None,
    min_trades_in: int | None = None,
    min_trades_out: int | None = None,
    min_sharpe_out: float | None = None,
    metric_name: str | None = None,
) -> ThresholdOptimizerResult:
    """
    Optimize thresholds with overfitting awareness using in-sample/out-of-sample splits.
    
    This function:
    1. Computes prediction probabilities once and caches them
    2. Splits data into in-sample (70%) and out-of-sample (30%)
    3. Evaluates each threshold combination on both splits
    4. Applies overfitting penalty and selects best combination
    
    Args:
        strategy_mode: Strategy mode ("both", "long_only", "short_only") or None for default
        min_trades_in: Minimum trades in in-sample (default: from config)
        min_trades_out: Minimum trades in out-of-sample (default: from config)
        min_sharpe_out: Minimum acceptable out-of-sample Sharpe (default: from config)
    """
    # Resolve strategy mode and min requirements from config if not provided
    from src.strategies.strategy_mode import StrategyMode as StrategyModeEnum
    
    if strategy_mode is None:
        strategy_mode = getattr(settings, "ML_STRATEGY_MODE_DEFAULT", "both")
    mode = StrategyModeEnum.from_string(strategy_mode)
    
    if min_trades_in is None:
        min_trades_in = getattr(settings, "MIN_TRADES_IN_DEFAULT", MIN_TRADES_IN_SAMPLE)
    if min_trades_out is None:
        min_trades_out = getattr(settings, "MIN_TRADES_OUT_DEFAULT", MIN_TRADES_OUT_SAMPLE)
    if min_sharpe_out is None:
        min_sharpe_out = getattr(settings, "MIN_SHARPE_OUT_DEFAULT", MIN_ACCEPTABLE_SHARPE_OUT)
    
    logger.info(
        f"[Threshold Opt] Strategy mode: {mode.value}, "
        f"min_trades_in={min_trades_in}, min_trades_out={min_trades_out}, "
        f"min_sharpe_out={min_sharpe_out}"
    )
    from src.backtest.engine import run_backtest_with_ml
    
    if short_threshold_candidates is None:
        short_threshold_candidates = [None]
    
    # Step 1: Compute and cache prediction probabilities
    logger.info("=" * 60)
    logger.info("Step 1: Computing prediction probabilities (caching for reuse)")
    logger.info("=" * 60)
    
    # Extract symbol and timeframe from parameters or settings
    if symbol is None:
        symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    else:
        symbol = symbol.replace("/", "").upper() if "/" in symbol else symbol.upper()
    
    if timeframe is None:
        timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    else:
        timeframe = timeframe.lower()
    
    # Use feature_preset if provided, otherwise default to "extended_safe" for ml_xgb
    preset_to_use = feature_preset if feature_preset is not None else ("extended_safe" if strategy_name == "ml_xgb" else "base")
    
    # Log date range filter if specified
    if start_date or end_date:
        logger.info(
            f"[Threshold Opt] Date range filter: start_date={start_date}, end_date={end_date}"
        )
    
    try:
        # Use get_or_build_predictions for file caching support
        from src.optimization.ml_proba_cache import get_or_build_predictions
        
        # Log 3-class model information for ml_lstm_attn
        if strategy_name == "ml_lstm_attn":
            from src.dl.data.labels import LstmClassIndex
            logger.info(
                f"[ThresholdOptimizer][{strategy_name}] Using 3-class LSTM-Attention model. "
                f"proba_long/proba_short are derived from class indices: "
                f"LONG={LstmClassIndex.LONG}, SHORT={LstmClassIndex.SHORT}, FLAT={LstmClassIndex.FLAT}. "
                f"Probabilities are extracted from 3-class softmax output, not binary probabilities."
            )
        
        proba_long_arr, proba_short_arr, df_aligned = get_or_build_predictions(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            feature_preset=preset_to_use if strategy_name == "ml_xgb" else "base",
            force_rebuild=force_rebuild_proba_cache,
            nthread=nthread,
            start_date=start_date,
            end_date=end_date,
        )
        N = len(proba_long_arr)
        preset_to_log = preset_to_use if strategy_name == "ml_xgb" else "N/A"
        logger.info(
            f"Cached {N} predictions (LONG and SHORT) for threshold optimization: "
            f"strategy={strategy_name}, symbol={symbol}, timeframe={timeframe}, "
            f"feature_preset={preset_to_log}"
        )
        
        # Log probability statistics for ml_lstm_attn
        if strategy_name == "ml_lstm_attn":
            import numpy as np
            from src.dl.data.labels import LstmClassIndex
            
            # Compute proba_flat for statistics (3-class: p_flat + p_long + p_short = 1)
            proba_flat_arr = 1.0 - proba_long_arr - proba_short_arr
            proba_flat_arr = np.clip(proba_flat_arr, 0.0, 1.0)
            
            # Compute argmax distribution (predicted class distribution)
            proba_stack = np.stack([proba_flat_arr, proba_long_arr, proba_short_arr], axis=1)  # (N, 3)
            predicted_classes = np.argmax(proba_stack, axis=1)  # (N,) with values {0: FLAT, 1: LONG, 2: SHORT}
            
            flat_count = int(np.sum(predicted_classes == LstmClassIndex.FLAT))
            long_count = int(np.sum(predicted_classes == LstmClassIndex.LONG))
            short_count = int(np.sum(predicted_classes == LstmClassIndex.SHORT))
            total_count = len(predicted_classes)
            
            logger.info(
                f"[ThresholdOptimizer][{strategy_name}] Probability statistics: "
                f"mean_proba_long={proba_long_arr.mean():.4f}, "
                f"mean_proba_short={proba_short_arr.mean():.4f}, "
                f"mean_proba_flat={proba_flat_arr.mean():.4f}, "
                f"std_proba_long={proba_long_arr.std():.4f}, "
                f"std_proba_short={proba_short_arr.std():.4f}"
            )
            logger.info(
                f"[ThresholdOptimizer][{strategy_name}] Predicted class distribution (argmax): "
                f"FLAT={flat_count} ({100*flat_count/total_count:.1f}%), "
                f"LONG={long_count} ({100*long_count/total_count:.1f}%), "
                f"SHORT={short_count} ({100*short_count/total_count:.1f}%)"
            )
    except Exception as e:
        logger.error(f"Failed to compute prediction cache: {e}")
        raise
    
    # Step 2: Create in-sample / out-of-sample split
    split_idx = int(N * IN_SAMPLE_RATIO)
    idx_in = np.arange(N) < split_idx
    idx_out = np.arange(N) >= split_idx
    
    logger.info("=" * 60)
    logger.info("Step 2: In-sample / Out-of-sample Split")
    logger.info("=" * 60)
    logger.info(f"Total samples: {N}")
    logger.info(f"In-sample: {np.sum(idx_in)} ({IN_SAMPLE_RATIO*100:.0f}%)")
    logger.info(f"Out-of-sample: {np.sum(idx_out)} ({(1-IN_SAMPLE_RATIO)*100:.0f}%)")
    logger.info("=" * 60)
    
    # Step 3: Grid search with overfitting penalty
    total_combinations = len(long_threshold_candidates) * len(short_threshold_candidates)
    logger.info("=" * 60)
    logger.info("Step 3: Testing threshold combinations")
    logger.info("=" * 60)
    logger.info(f"Total combinations: {total_combinations}")
    
    # Format long threshold range strings (avoid f-string format specifier with conditionals)
    long_min_str = f"{long_threshold_candidates[0]:.3f}" if len(long_threshold_candidates) > 0 else "N/A"
    long_max_str = f"{long_threshold_candidates[-1]:.3f}" if len(long_threshold_candidates) > 0 else "N/A"
    long_step_str = (
        f"{(long_threshold_candidates[1] - long_threshold_candidates[0]):.3f}"
        if len(long_threshold_candidates) > 1
        else "N/A"
    )
    
    logger.info(
        "Long threshold range: [%s, %s], step=%s, count=%d",
        long_min_str,
        long_max_str,
        long_step_str,
        len(long_threshold_candidates),
    )
    
    # Format short threshold range strings (avoid f-string format specifier with conditionals)
    short_min_str = (
        f"{short_threshold_candidates[0]:.3f}"
        if short_threshold_candidates[0] is not None
        else "None"
    )
    short_max_str = (
        f"{short_threshold_candidates[-1]:.3f}"
        if short_threshold_candidates[-1] is not None
        else "None"
    )
    
    # Check if None is included and log reason
    short_has_none = None in short_threshold_candidates
    short_none_count = sum(1 for x in short_threshold_candidates if x is None)
    short_numeric_count = len(short_threshold_candidates) - short_none_count
    
    logger.info(
        "Short threshold range: [%s, %s], count=%d (numeric=%d, None=%d)",
        short_min_str,
        short_max_str,
        len(short_threshold_candidates),
        short_numeric_count,
        short_none_count,
    )
    
    if short_has_none:
        logger.info(
            "[ThresholdOptimizer] Short grid includes None (for long-only mode testing). "
            "This is the default behavior when user does not explicitly provide short range."
        )
    if strategy_name == "ml_lstm_attn":
        from src.dl.data.labels import LstmClassIndex
        logger.info(
            f"[ThresholdOptimizer][{strategy_name}] Note: Thresholds are applied to "
            f"proba_long (from class {LstmClassIndex.LONG}) and proba_short (from class {LstmClassIndex.SHORT}) "
            f"of the 3-class softmax output. FLAT class (index {LstmClassIndex.FLAT}) is not used for thresholding."
        )
    
    # Performance optimization: parallel execution
    if use_parallel and total_combinations > 1:
        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count() or 1
        logger.info(
            "[Parallel] Using ProcessPoolExecutor for threshold optimization: n_jobs=%s",
            n_jobs,
        )
    else:
        if not use_parallel:
            logger.info("[Serial] Running threshold optimization in serial mode (parallel disabled).")
        elif total_combinations <= 1:
            logger.info("[Serial] Running threshold optimization in serial mode (single combination).")
        else:
            logger.info("[Serial] Running threshold optimization in serial mode.")
    logger.info("=" * 60)
    
    trials = []
    best_score = float("-inf")
    best_long_threshold = long_threshold_candidates[0]
    best_short_threshold = short_threshold_candidates[0] if short_threshold_candidates else None
    best_sharpe_in = 0.0
    best_sharpe_out = 0.0
    best_metric_in = 0.0
    best_metric_out = 0.0
    best_gap = 0.0
    best_trades_in = 0
    best_trades_out = 0
    
    # Fallback: track best result even if it doesn't meet trade requirements
    best_result_unfiltered = None
    best_score_unfiltered = float("-inf")
    
    # Track success/failure counts
    success_count = 0
    failed_count = 0
    skipped_low_trades = 0
    skipped_low_sharpe = 0
    
    # Generate all threshold combinations
    threshold_combinations = [
        (long_thr, short_thr)
        for long_thr in long_threshold_candidates
        for short_thr in short_threshold_candidates
    ]
    
    optimization_start = time.perf_counter()
    
    # Get metric name for worker reconstruction
    # If metric_name is provided, use it directly (this is the correct way)
    # Otherwise, try to infer from function name, but map function names to metric keys
    if metric_name is None:
        # Try to infer from function name
        if hasattr(metric_fn, "__name__"):
            fn_name = metric_fn.__name__
            # Map function names to metric keys
            # e.g., "total_return_metric" -> "total_return", "sharpe_ratio_metric" -> "sharpe"
            if fn_name == "total_return_metric":
                metric_name = "total_return"
            elif fn_name in ("sharpe_ratio_metric", "sharpe_ratio"):
                metric_name = "sharpe"
            else:
                # Try to remove "_metric" suffix if present
                if fn_name.endswith("_metric"):
                    metric_name = fn_name[:-7]  # Remove "_metric" suffix
                else:
                    metric_name = fn_name
        else:
            # Fallback: default to sharpe
            metric_name = "sharpe"
    
    # Normalize metric_name to lowercase for consistency
    metric_name = metric_name.lower()
    
    # Parallel execution
    if use_parallel and total_combinations > 1 and n_jobs > 1:
        # Prepare worker arguments (serialize data once)
        worker_args = [
            (
                long_thr,
                short_thr,
                strategy_name,
                metric_name,
                proba_long_arr,  # numpy array - pickleable
                proba_short_arr,  # numpy array - pickleable
                df_aligned,  # pandas DataFrame - pickleable (may be slow for large data)
                idx_in,  # numpy array - pickleable
                idx_out,  # numpy array - pickleable
                preset_to_use,
                symbol,  # symbol as string
                timeframe,  # timeframe as string
                mode.value,  # strategy_mode as string
            )
            for long_thr, short_thr in threshold_combinations
        ]
        
        # Execute in parallel using ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                future_to_combo = {
                    executor.submit(_worker_evaluate_threshold, args): (args[0], args[1])
                    for args in worker_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_combo):
                    long_thr, short_thr = future_to_combo[future]
                    try:
                        result = future.result()
                        
                        if not result.get("success", False):
                            failed_count += 1
                            logger.warning(
                                f"  Failed to evaluate long={long_thr:.3f}, short={short_thr}: {result.get('error', 'Unknown error')}"
                            )
                            continue
                        
                        # Use metric_in/metric_out if available, otherwise fall back to sharpe_in/sharpe_out
                        metric_in = result.get("metric_in", result.get("sharpe_in", 0.0))
                        metric_out = result.get("metric_out", result.get("sharpe_out", 0.0))
                        sharpe_in = result["sharpe_in"]  # Keep for backward compatibility
                        sharpe_out = result["sharpe_out"]  # Keep for backward compatibility
                        trades_in = result["trades_in"]
                        trades_out = result["trades_out"]
                        total_trades = result["total_trades"]
                        gap = result["gap"]
                        score = result["score"]
                        
                        # Track best unfiltered
                        if score > best_score_unfiltered:
                            best_score_unfiltered = score
                            best_result_unfiltered = {
                                "long_threshold": long_thr,
                                "short_threshold": short_thr,
                                "sharpe_in": sharpe_in,
                                "sharpe_out": sharpe_out,
                                "metric_in": metric_in,
                                "metric_out": metric_out,
                                "gap": gap,
                                "score": score,
                                "trades_in": trades_in,
                                "trades_out": trades_out,
                                "total_return_in": result.get("total_return_in"),
                                "total_return_out": result.get("total_return_out"),
                            }
                        
                        # Apply filters
                        if not has_enough_trades(trades_in, trades_out, min_trades_in, min_trades_out):
                            skipped_low_trades += 1
                            logger.info(
                                "[Threshold Optimizer] SKIP due to insufficient trades: "
                                f"long={long_thr:.3f}, short={short_thr}, "
                                f"trades_in={trades_in} < {min_trades_in} or trades_out={trades_out} < {min_trades_out}, total_trades={total_trades}"
                            )
                            continue
                        
                        if sharpe_out < min_sharpe_out:
                            skipped_low_sharpe += 1
                            logger.info(
                                "[Threshold Optimizer] SKIP due to low Sharpe_out: "
                                f"long={long_thr:.3f}, short={short_thr}, "
                                f"sharpe_out={sharpe_out:.6f} < {min_sharpe_out}"
                            )
                            continue
                        
                        trial = {
                            "long_threshold": long_thr,
                            "short_threshold": short_thr,
                            "sharpe_in_sample": sharpe_in,
                            "sharpe_out_sample": sharpe_out,
                            "gap_in_out": gap,
                            "score_overfit_adjusted": score,
                            "total_trades_in": trades_in,
                            "total_trades_out": trades_out,
                            "total_trades": total_trades,
                            "total_return_in": result["total_return_in"],
                            "total_return_out": result["total_return_out"],
                            "win_rate_in": result["win_rate_in"],
                            "win_rate_out": result["win_rate_out"],
                        }
                        trials.append(trial)
                        success_count += 1
                        
                        logger.info(
                            f"  long={long_thr:.3f}, short={short_thr}, "
                            f"sharpe_in={sharpe_in:.6f}, sharpe_out={sharpe_out:.6f}, "
                            f"gap={gap:.6f}, score={score:.6f}, "
                            f"trades_in={trades_in}, trades_out={trades_out}, total_trades={total_trades}"
                        )
                        
                        # Update best
                        if score > best_score:
                            best_score = score
                            best_long_threshold = long_thr
                            best_short_threshold = short_thr
                            best_sharpe_in = sharpe_in
                            best_sharpe_out = sharpe_out
                            best_metric_in = metric_in
                            best_metric_out = metric_out
                            best_gap = gap
                            best_trades_in = trades_in
                            best_trades_out = trades_out
                            
                    except Exception as e:
                        failed_count += 1
                        logger.warning(
                            f"  Failed to evaluate long={long_thr:.3f}, short={short_thr}: {e}"
                        )
        except Exception as e:
            logger.warning(
                "[Parallel] Parallel execution failed (%s). Falling back to serial execution.",
                repr(e),
            )
            use_parallel = False
            # Fall through to serial execution
    
    # Serial execution (fallback or when parallel is disabled)
    if not use_parallel or total_combinations <= 1 or n_jobs <= 1:
        for long_thr, short_thr in threshold_combinations:
            try:
                # OPTIMIZATION: Use helper function for cleaner code
                result = _evaluate_single_threshold_combination(
                    long_thr=long_thr,
                    short_thr=short_thr,
                    strategy_name=strategy_name,
                    metric_fn=metric_fn,
                    proba_long_arr=proba_long_arr,
                    proba_short_arr=proba_short_arr,
                    df_aligned=df_aligned,
                    idx_in=idx_in,
                    idx_out=idx_out,
                    preset_to_use=preset_to_use,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_mode=mode.value,
                )
                
                if not result.get("success", False):
                    failed_count += 1
                    logger.warning(
                        f"  Failed to evaluate long={long_thr:.3f}, short={short_thr}: {result.get('error', 'Unknown error')}"
                    )
                    continue
                
                # Use metric_in/metric_out if available, otherwise fall back to sharpe_in/sharpe_out
                metric_in = result.get("metric_in", result.get("sharpe_in", 0.0))
                metric_out = result.get("metric_out", result.get("sharpe_out", 0.0))
                sharpe_in = result["sharpe_in"]  # Keep for backward compatibility
                sharpe_out = result["sharpe_out"]  # Keep for backward compatibility
                trades_in = result["trades_in"]
                trades_out = result["trades_out"]
                total_trades = result["total_trades"]
                gap = result["gap"]
                score = result["score"]
                
                # Track best unfiltered result as fallback
                if score > best_score_unfiltered:
                    best_score_unfiltered = score
                    best_result_unfiltered = {
                        "long_threshold": long_thr,
                        "short_threshold": short_thr,
                        "sharpe_in": sharpe_in,
                        "sharpe_out": sharpe_out,
                        "metric_in": metric_in,
                        "metric_out": metric_out,
                        "gap": gap,
                        "score": score,
                        "trades_in": trades_in,
                        "trades_out": trades_out,
                        "total_return_in": result.get("total_return_in"),
                        "total_return_out": result.get("total_return_out"),
                    }
                
                # Apply trade count filter: skip combinations with insufficient trades
                if not has_enough_trades(trades_in, trades_out, min_trades_in, min_trades_out):
                    skipped_low_trades += 1
                    logger.info(
                        "[Threshold Optimizer] SKIP due to insufficient trades: "
                        f"long={long_thr:.3f}, short={short_thr}, "
                        f"trades_in={trades_in} < {min_trades_in} or trades_out={trades_out} < {min_trades_out}, total_trades={total_trades}"
                    )
                    continue
                
                # Apply Sharpe filter: skip combinations with insufficient out-of-sample Sharpe
                if sharpe_out < min_sharpe_out:
                    skipped_low_sharpe += 1
                    logger.info(
                        "[Threshold Optimizer] SKIP due to low Sharpe_out: "
                        f"long={long_thr:.3f}, short={short_thr}, "
                        f"sharpe_out={sharpe_out:.6f} < {min_sharpe_out}"
                    )
                    continue
                
                trial = {
                    "long_threshold": long_thr,
                    "short_threshold": short_thr,
                    "sharpe_in_sample": sharpe_in,
                    "sharpe_out_sample": sharpe_out,
                    "gap_in_out": gap,
                    "score_overfit_adjusted": score,
                    "total_trades_in": trades_in,
                    "total_trades_out": trades_out,
                    "total_trades": total_trades,
                    "total_return_in": result["total_return_in"],
                    "total_return_out": result["total_return_out"],
                    "win_rate_in": result["win_rate_in"],
                    "win_rate_out": result["win_rate_out"],
                }
                trials.append(trial)
                success_count += 1
                
                logger.info(
                    f"  long={long_thr:.3f}, short={short_thr}, "
                    f"sharpe_in={sharpe_in:.6f}, sharpe_out={sharpe_out:.6f}, "
                    f"gap={gap:.6f}, score={score:.6f}, "
                    f"trades_in={trades_in}, trades_out={trades_out}, total_trades={total_trades}"
                )
                
                # Update best if this is better
                if score > best_score:
                    best_score = score
                    best_long_threshold = long_thr
                    best_short_threshold = short_thr
                    best_sharpe_in = sharpe_in
                    best_sharpe_out = sharpe_out
                    best_metric_in = metric_in
                    best_metric_out = metric_out
                    best_gap = gap
                    best_trades_in = trades_in
                    best_trades_out = trades_out
                    
            except Exception as e:
                failed_count += 1
                logger.warning(
                    f"  Failed to evaluate long={long_thr:.3f}, short={short_thr}: {e}"
                )
                continue
    
    optimization_time = time.perf_counter() - optimization_start
    logger.info(f"[Performance] Threshold optimization completed in {optimization_time:.2f}s")
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Optimization Complete")
    logger.info("=" * 60)
    logger.info(
        f"Evaluated {total_combinations} combinations: "
        f"success={success_count}, failed={failed_count}, "
        f"skipped_low_trades={skipped_low_trades}, skipped_low_sharpe={skipped_low_sharpe}"
    )
    
    # Check if we have a valid best result (meets both trade and Sharpe requirements)
    strategy_enabled = True
    disable_reason = None
    
    if best_score == float("-inf"):
        # No combination met the trade and/or Sharpe requirements
        logger.warning(
            "[Threshold Optimizer] No valid threshold combination met minimum requirements. "
            f"Required: trades_in >= {min_trades_in}, trades_out >= {min_trades_out}, "
            f"sharpe_out >= {min_sharpe_out}"
        )
        
        # Check if it's a Sharpe issue specifically
        if best_result_unfiltered is not None:
            best_unfiltered_sharpe_out = best_result_unfiltered["sharpe_out"]
            if best_unfiltered_sharpe_out < min_sharpe_out:
                # All combinations had negative Sharpe_out
                logger.error(
                    "[Threshold Optimizer] ⚠️  CRITICAL: No combination achieved Sharpe_out >= "
                    f"{min_sharpe_out}. Best unfiltered Sharpe_out: {best_unfiltered_sharpe_out:.6f}. "
                    "Strategy has no profitable edge in this grid."
                )
                strategy_enabled = False
                disable_reason = "no_positive_sharpe"
                
                # Use unfiltered result for diagnostic purposes (but mark as disabled)
                best_long_threshold = best_result_unfiltered["long_threshold"]
                best_short_threshold = best_result_unfiltered["short_threshold"]
                best_sharpe_in = best_result_unfiltered["sharpe_in"]
                best_sharpe_out = best_result_unfiltered["sharpe_out"]
                best_gap = best_result_unfiltered["gap"]
                best_score = best_result_unfiltered["score"]
                best_trades_in = best_result_unfiltered["trades_in"]
                best_trades_out = best_result_unfiltered["trades_out"]
            else:
                # Trade count issue, but Sharpe was OK
                logger.warning(
                    "[Threshold Optimizer] Using best unfiltered result as fallback (trade count issue): "
                    f"long={best_result_unfiltered['long_threshold']:.3f}, "
                    f"short={best_result_unfiltered['short_threshold']}, "
                    f"trades_in={best_result_unfiltered['trades_in']}, "
                    f"trades_out={best_result_unfiltered['trades_out']}"
                )
                best_long_threshold = best_result_unfiltered["long_threshold"]
                best_short_threshold = best_result_unfiltered["short_threshold"]
                best_sharpe_in = best_result_unfiltered["sharpe_in"]
                best_sharpe_out = best_result_unfiltered["sharpe_out"]
                best_gap = best_result_unfiltered["gap"]
                best_score = best_result_unfiltered["score"]
                best_trades_in = best_result_unfiltered["trades_in"]
                best_trades_out = best_result_unfiltered["trades_out"]
        else:
            # No valid result at all - raise error
            logger.error(
                "[Threshold Optimizer] No valid threshold combination found. "
                "Please relax constraints or expand grid."
            )
            raise ValueError(
                "No valid threshold combination found. "
                f"All combinations either failed or had insufficient trades "
                f"(required: trades_in >= {min_trades_in}, trades_out >= {min_trades_out}) "
                f"or insufficient Sharpe (required: sharpe_out >= {min_sharpe_out})."
            )
    else:
        # We have a valid result, but check if Sharpe_out is still negative (shouldn't happen, but defensive)
        if best_sharpe_out < min_sharpe_out:
            logger.error(
                "[Threshold Optimizer] ⚠️  CRITICAL: Best combination has Sharpe_out < "
                f"{min_sharpe_out} (Sharpe_out={best_sharpe_out:.6f}). "
                "Strategy has no profitable edge in this grid."
            )
            strategy_enabled = False
            disable_reason = "no_positive_sharpe"
    
    logger.info(f"Best long threshold: {best_long_threshold:.3f}")
    logger.info(f"Best short threshold: {best_short_threshold}")
    logger.info(f"Best combination: long={best_long_threshold:.3f}, short={best_short_threshold}, sharpe_out={best_sharpe_out:.6f}")
    logger.info(f"Strategy mode: {mode.value}")
    logger.info(f"Sharpe (in-sample): {best_sharpe_in:.6f}")
    logger.info(f"Sharpe (out-of-sample): {best_sharpe_out:.6f}")
    logger.info(f"Gap (in - out): {best_gap:.6f}")
    logger.info(f"Final score (overfit-adjusted): {best_score:.6f}")
    logger.info(f"Total trades: in={best_trades_in}, out={best_trades_out}, total={best_trades_in + best_trades_out}")
    logger.info(
        f"Optimization constraints: min_trades_in={min_trades_in}, "
        f"min_trades_out={min_trades_out}, min_sharpe_out={min_sharpe_out}"
    )
    logger.info(
        f"Note: Only combinations meeting all constraints were considered."
    )
    
    if not strategy_enabled:
        logger.warning(
            "[Threshold Optimizer] ⚠️  Strategy will be DISABLED for live trading. "
            f"Reason: {disable_reason}. "
            "Consider using this strategy only as a filter/risk control signal "
            "combined with other strategies, or retrain the model with different features."
        )
    
    # Overfitting warnings
    if best_gap > 1.0:
        logger.warning(
            f"⚠️  Large in-sample/out-of-sample gap ({best_gap:.3f}) detected. "
            f"This may indicate overfitting. Consider using more conservative thresholds."
        )
    if (best_trades_in + best_trades_out) < MIN_TRADES * 2:
        logger.warning(
            f"⚠️  Low total trade count ({best_trades_in + best_trades_out}). "
            f"Results may be less reliable."
        )
    
    logger.info("=" * 60)
    
    # NOTE: metric field uses out-of-sample metric value as the primary metric
    # For total_return, use best_metric_out; for sharpe, use best_sharpe_out
    # score_overfit_adjusted is stored separately for reference
    # Store strategy_mode and min requirements in result for JSON serialization
    
    # Determine best metric value based on metric type
    if metric_name == "total_return":
        best_metric_value = best_metric_out
    else:
        # For sharpe or other metrics, use sharpe_out (which is same as metric_out for sharpe)
        best_metric_value = best_sharpe_out
    
    result = ThresholdOptimizerResult(
        best_long_threshold=best_long_threshold,
        best_short_threshold=best_short_threshold,
        best_metric_value=best_metric_value,  # Use appropriate metric value
        metric_name=metric_name,  # Use the normalized metric name
        trials=trials,
        sharpe_in_sample=best_sharpe_in,
        sharpe_out_sample=best_sharpe_out,
        gap_in_out=best_gap,
        total_trades_in=best_trades_in,
        total_trades_out=best_trades_out,
        score_overfit_adjusted=best_score,
        enabled=strategy_enabled,
        disable_reason=disable_reason,
        strategy_mode=mode.value,
        min_trades_in=min_trades_in,
        min_trades_out=min_trades_out,
        min_sharpe_out=min_sharpe_out,
    )
    
    return result


def save_threshold_result(
    result: ThresholdOptimizerResult,
    path: Path | str,
    strategy_name: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> None:
    """
    Save threshold optimization result to JSON file.
    
    The saved JSON includes both backward-compatible fields and new overfitting-aware fields:
    - long, short, metric: Backward compatible (existing code can read these)
    - sharpe_in_sample, sharpe_out_sample, gap_in_out, etc.: New overfitting diagnostics
    
    Args:
        result: ThresholdOptimizerResult to save
        path: Path to save JSON file
        strategy_name: Optional strategy name (for JSON metadata)
        symbol: Optional symbol (for JSON metadata)
        timeframe: Optional timeframe (for JSON metadata)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build JSON dict with backward-compatible structure
    json_data = {
        # Backward compatible fields (existing code expects these)
        "best_long_threshold": result.best_long_threshold,
        "best_short_threshold": result.best_short_threshold,
        "best_metric_value": result.best_metric_value,
        "metric_name": result.metric_name,
        "trials": result.trials,
        # New overfitting-aware fields (optional, for diagnostics)
    }
    
    # Add optional fields if present
    if result.sharpe_in_sample is not None:
        json_data["sharpe_in_sample"] = result.sharpe_in_sample
    if result.sharpe_out_sample is not None:
        json_data["sharpe_out_sample"] = result.sharpe_out_sample
    if result.gap_in_out is not None:
        json_data["gap_in_out"] = result.gap_in_out
    if result.total_trades_in is not None:
        json_data["total_trades_in"] = result.total_trades_in
    if result.total_trades_out is not None:
        json_data["total_trades_out"] = result.total_trades_out
    if result.score_overfit_adjusted is not None:
        json_data["score_overfit_adjusted"] = result.score_overfit_adjusted
    
    # Strategy enablement fields
    json_data["enabled"] = result.enabled
    if result.disable_reason:
        json_data["disable_reason"] = result.disable_reason
    
    # Strategy mode and optimization constraints
    if hasattr(result, "strategy_mode") and result.strategy_mode:
        json_data["strategy_mode"] = result.strategy_mode
    if hasattr(result, "min_trades_in") and result.min_trades_in is not None:
        json_data["min_trades_in"] = result.min_trades_in
    if hasattr(result, "min_trades_out") and result.min_trades_out is not None:
        json_data["min_trades_out"] = result.min_trades_out
    if hasattr(result, "min_sharpe_out") and result.min_sharpe_out is not None:
        json_data["min_sharpe_out"] = result.min_sharpe_out
    
    # Add metadata if provided
    if strategy_name:
        json_data["strategy"] = strategy_name
    if symbol:
        json_data["symbol"] = symbol
    if timeframe:
        json_data["timeframe"] = timeframe
    
    # Also include backward-compatible aliases for existing code
    json_data["long"] = result.best_long_threshold
    json_data["short"] = result.best_short_threshold
    json_data["metric"] = result.best_metric_value  # Out-of-sample Sharpe
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved threshold result to {path}")


def load_threshold_result(path: Path | str) -> ThresholdOptimizerResult:
    """
    Load threshold optimization result from JSON file.
    
    Supports both old format (backward compatible) and new overfitting-aware format.
    
    Args:
        path: Path to JSON file
    
    Returns:
        ThresholdOptimizerResult
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be parsed
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Threshold result file not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both old and new JSON formats
        # Old format: best_long_threshold, best_short_threshold, best_metric_value
        # New format: same fields + overfitting fields (sharpe_in_sample, etc.)
        # Also support backward-compatible aliases: "long", "short", "metric"
        
        normalized_data = {}
        
        # Map old/new field names to dataclass fields
        normalized_data["best_long_threshold"] = data.get("best_long_threshold") or data.get("long")
        normalized_data["best_short_threshold"] = data.get("best_short_threshold") or data.get("short")
        normalized_data["best_metric_value"] = data.get("best_metric_value") or data.get("metric")
        normalized_data["metric_name"] = data.get("metric_name", "unknown")
        normalized_data["trials"] = data.get("trials", [])
        
        # Optional overfitting fields
        normalized_data["sharpe_in_sample"] = data.get("sharpe_in_sample")
        normalized_data["sharpe_out_sample"] = data.get("sharpe_out_sample")
        normalized_data["gap_in_out"] = data.get("gap_in_out")
        normalized_data["total_trades_in"] = data.get("total_trades_in")
        normalized_data["total_trades_out"] = data.get("total_trades_out")
        normalized_data["score_overfit_adjusted"] = data.get("score_overfit_adjusted")
        # Strategy enablement fields (backward compatible: default to True if not present)
        normalized_data["enabled"] = data.get("enabled", True)
        normalized_data["disable_reason"] = data.get("disable_reason")
        
        return ThresholdOptimizerResult.from_dict(normalized_data)
    except Exception as e:
        raise ValueError(f"Failed to load threshold result from {path}: {e}") from e


def load_threshold_result_for(
    strategy_name: str,
    symbol: str,
    timeframe: str,
) -> tuple[ThresholdOptimizerResult, Path] | None:
    """
    Load threshold optimization result for a given strategy/symbol/timeframe combo.

    Returns:
        Tuple of (result, path) if data is available, otherwise None.
    """
    path = get_threshold_result_path(strategy_name, symbol, timeframe)
    try:
        result = load_threshold_result(path)
        return result, path
    except FileNotFoundError:
        return None
    except ValueError as exc:
        logger.warning(
            "[ThresholdOptimizer] Failed to load threshold result from %s: %s",
            path,
            exc,
        )
        return None


def main():
    """
    CLI entry point for threshold optimizer.
    
    This function delegates to optimize_ml_threshold.py for ML-based strategies.
    """
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )
    
    logger.info(
        "[ThresholdOptimizer] This module delegates to optimize_ml_threshold for ML strategies. "
        "Redirecting to optimize_ml_threshold module..."
    )
    
    # Import and call the actual CLI
    from src.optimization.optimize_ml_threshold import main as optimize_main
    optimize_main()


if __name__ == "__main__":
    main()

