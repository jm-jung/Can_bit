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
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np

from src.backtest.metrics import get_metric_function

logger = logging.getLogger(__name__)

THRESHOLD_RESULTS_DIR = Path("data/thresholds")

# Overfitting penalty hyperparameters
OVERFIT_PENALTY_ALPHA = 0.5  # Penalty weight for in-sample/out-of-sample gap
MIN_TRADES = 20  # Minimum total trades required for a threshold combination
IN_SAMPLE_RATIO = 0.7  # Proportion of data for in-sample evaluation


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in result.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThresholdOptimizerResult:
        """Create from dictionary (backward compatible with old format)."""
        # Handle old format without overfitting fields
        return cls(
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
        )


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


def _optimize_with_overfit_awareness(
    strategy_name: str,
    metric_fn: Callable[[Any], float],
    long_threshold_candidates: List[float],
    short_threshold_candidates: Optional[List[float]] = None,
) -> ThresholdOptimizerResult:
    """
    Optimize thresholds with overfitting awareness using in-sample/out-of-sample splits.
    
    This function:
    1. Computes prediction probabilities once and caches them
    2. Splits data into in-sample (70%) and out-of-sample (30%)
    3. Evaluates each threshold combination on both splits
    4. Applies overfitting penalty and selects best combination
    """
    from src.optimization.ml_proba_cache import compute_ml_proba_cache
    from src.backtest.engine import run_backtest_with_ml
    
    if short_threshold_candidates is None:
        short_threshold_candidates = [None]
    
    # Step 1: Compute and cache prediction probabilities
    logger.info("=" * 60)
    logger.info("Step 1: Computing prediction probabilities (caching for reuse)")
    logger.info("=" * 60)
    
    # Extract symbol and timeframe from settings
    symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    
    try:
        proba_arr, df_aligned = compute_ml_proba_cache(strategy_name, symbol=symbol, timeframe=timeframe)
        N = len(proba_arr)
        logger.info(f"Cached {N} predictions for threshold optimization")
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
    logger.info(f"Step 3: Testing {total_combinations} threshold combinations")
    logger.info("=" * 60)
    
    trials = []
    best_score = float("-inf")
    best_long_threshold = long_threshold_candidates[0]
    best_short_threshold = short_threshold_candidates[0] if short_threshold_candidates else None
    best_sharpe_in = 0.0
    best_sharpe_out = 0.0
    best_gap = 0.0
    best_trades_in = 0
    best_trades_out = 0
    
    for long_thr in long_threshold_candidates:
        for short_thr in short_threshold_candidates:
            try:
                # Run in-sample backtest
                result_in = run_backtest_with_ml(
                    long_threshold=long_thr,
                    short_threshold=short_thr,
                    use_optimized_thresholds=False,
                    strategy_name=strategy_name,
                    proba_up_cache=proba_arr,
                    df_with_proba=df_aligned,
                    index_mask=idx_in,
                )
                sharpe_in = metric_fn(result_in)
                trades_in = result_in["total_trades"]
                
                # Run out-of-sample backtest
                result_out = run_backtest_with_ml(
                    long_threshold=long_thr,
                    short_threshold=short_thr,
                    use_optimized_thresholds=False,
                    strategy_name=strategy_name,
                    proba_up_cache=proba_arr,
                    df_with_proba=df_aligned,
                    index_mask=idx_out,
                )
                sharpe_out = metric_fn(result_out)
                trades_out = result_out["total_trades"]
                
                total_trades = trades_in + trades_out
                
                # Skip if too few trades
                if total_trades < MIN_TRADES:
                    logger.debug(
                        f"  long={long_thr:.3f}, short={short_thr}: "
                        f"skipped (total_trades={total_trades} < MIN_TRADES={MIN_TRADES})"
                    )
                    continue
                
                # Compute overfitting-adjusted score
                gap = sharpe_in - sharpe_out
                score = compute_overfit_penalty_score(sharpe_in, sharpe_out)
                
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
                    "total_return_in": result_in["total_return"],
                    "total_return_out": result_out["total_return"],
                    "win_rate_in": result_in["win_rate"],
                    "win_rate_out": result_out["win_rate"],
                }
                trials.append(trial)
                
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
                    best_gap = gap
                    best_trades_in = trades_in
                    best_trades_out = trades_out
                    
            except Exception as e:
                logger.warning(
                    f"  Failed to evaluate long={long_thr:.3f}, short={short_thr}: {e}"
                )
                continue
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Optimization Complete")
    logger.info("=" * 60)
    logger.info(f"Best long threshold: {best_long_threshold:.3f}")
    logger.info(f"Best short threshold: {best_short_threshold}")
    logger.info(f"Sharpe (in-sample): {best_sharpe_in:.6f}")
    logger.info(f"Sharpe (out-of-sample): {best_sharpe_out:.6f}")
    logger.info(f"Gap (in - out): {best_gap:.6f}")
    logger.info(f"Final score (overfit-adjusted): {best_score:.6f}")
    logger.info(f"Total trades: in={best_trades_in}, out={best_trades_out}, total={best_trades_in + best_trades_out}")
    
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
    
    # NOTE: metric field uses out-of-sample Sharpe as the primary metric
    # score_overfit_adjusted is stored separately for reference
    return ThresholdOptimizerResult(
        best_long_threshold=best_long_threshold,
        best_short_threshold=best_short_threshold,
        best_metric_value=best_sharpe_out,  # Use out-of-sample Sharpe as primary metric
        metric_name=metric_fn.__name__ if hasattr(metric_fn, "__name__") else "unknown",
        trials=trials,
        sharpe_in_sample=best_sharpe_in,
        sharpe_out_sample=best_sharpe_out,
        gap_in_out=best_gap,
        total_trades_in=best_trades_in,
        total_trades_out=best_trades_out,
        score_overfit_adjusted=best_score,
    )


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

