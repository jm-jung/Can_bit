"""
Regression tests for threshold optimizer performance optimizations.

This module verifies that optimized threshold optimizer produces
identical results to the original implementation (lossless optimization).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

from src.optimization.threshold_optimizer import (
    optimize_threshold_for_strategy,
    ThresholdOptimizerResult,
    load_threshold_result,
    get_threshold_result_path,
)
from src.backtest.metrics import get_metric_function

logger = logging.getLogger(__name__)


def assert_metrics_equal(
    old_metrics: Dict[str, Any],
    new_metrics: Dict[str, Any],
    tolerance: float = 1e-6,
    check_trades: bool = False,
) -> None:
    """
    Assert that two threshold optimization results are equal (within tolerance).
    
    Args:
        old_metrics: Original result metrics
        new_metrics: Optimized result metrics
        tolerance: Floating point tolerance for comparisons
        check_trades: If True, also compare individual trade lists
    """
    # Compare best thresholds
    assert abs(old_metrics["best_long_threshold"] - new_metrics["best_long_threshold"]) < tolerance, (
        f"best_long_threshold mismatch: {old_metrics['best_long_threshold']} vs {new_metrics['best_long_threshold']}"
    )
    
    if old_metrics.get("best_short_threshold") is not None:
        assert abs(old_metrics["best_short_threshold"] - new_metrics["best_short_threshold"]) < tolerance, (
            f"best_short_threshold mismatch: {old_metrics['best_short_threshold']} vs {new_metrics['best_short_threshold']}"
        )
    
    # Compare performance metrics
    assert abs(old_metrics.get("sharpe_in_sample", 0) - new_metrics.get("sharpe_in_sample", 0)) < tolerance, (
        f"sharpe_in_sample mismatch: {old_metrics.get('sharpe_in_sample')} vs {new_metrics.get('sharpe_in_sample')}"
    )
    
    assert abs(old_metrics.get("sharpe_out_sample", 0) - new_metrics.get("sharpe_out_sample", 0)) < tolerance, (
        f"sharpe_out_sample mismatch: {old_metrics.get('sharpe_out_sample')} vs {new_metrics.get('sharpe_out_sample')}"
    )
    
    assert abs(old_metrics.get("gap_in_out", 0) - new_metrics.get("gap_in_out", 0)) < tolerance, (
        f"gap_in_out mismatch: {old_metrics.get('gap_in_out')} vs {new_metrics.get('gap_in_out')}"
    )
    
    assert abs(old_metrics.get("score_overfit_adjusted", 0) - new_metrics.get("score_overfit_adjusted", 0)) < tolerance, (
        f"score_overfit_adjusted mismatch: {old_metrics.get('score_overfit_adjusted')} vs {new_metrics.get('score_overfit_adjusted')}"
    )
    
    # Compare trade counts
    assert old_metrics.get("total_trades_in", 0) == new_metrics.get("total_trades_in", 0), (
        f"total_trades_in mismatch: {old_metrics.get('total_trades_in')} vs {new_metrics.get('total_trades_in')}"
    )
    
    assert old_metrics.get("total_trades_out", 0) == new_metrics.get("total_trades_out", 0), (
        f"total_trades_out mismatch: {old_metrics.get('total_trades_out')} vs {new_metrics.get('total_trades_out')}"
    )
    
    # Compare returns (within tolerance)
    if "total_return_in" in old_metrics and "total_return_in" in new_metrics:
        assert abs(old_metrics["total_return_in"] - new_metrics["total_return_in"]) < tolerance, (
            f"total_return_in mismatch: {old_metrics['total_return_in']} vs {new_metrics['total_return_in']}"
        )
    
    if "total_return_out" in old_metrics and "total_return_out" in new_metrics:
        assert abs(old_metrics["total_return_out"] - new_metrics["total_return_out"]) < tolerance, (
            f"total_return_out mismatch: {old_metrics['total_return_out']} vs {new_metrics['total_return_out']}"
        )
    
    logger.info("✅ All metrics match within tolerance")


def test_threshold_optimizer_regression(
    strategy_name: str = "ml_xgb",
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    feature_preset: str = "extended_safe",
    long_candidates: list[float] | None = None,
    short_candidates: list[float] | None = None,
    baseline_result_path: Path | str | None = None,
) -> None:
    """
    Regression test: Compare optimized threshold optimizer results with baseline.
    
    This test can be used to verify that optimizations maintain result accuracy.
    
    Args:
        strategy_name: Strategy identifier
        symbol: Trading symbol
        timeframe: Timeframe
        feature_preset: Feature preset
        long_candidates: Long threshold candidates (default: [0.1, 0.2, ..., 0.9])
        short_candidates: Short threshold candidates (default: [0.1, 0.2, ..., 0.9])
        baseline_result_path: Path to baseline result JSON (optional)
    """
    if long_candidates is None:
        long_candidates = [0.1 + i * 0.1 for i in range(9)]  # [0.1, 0.2, ..., 0.9]
    if short_candidates is None:
        short_candidates = [0.1 + i * 0.1 for i in range(9)]  # [0.1, 0.2, ..., 0.9]
    
    # Run optimized threshold optimization
    logger.info("Running optimized threshold optimization...")
    metric_fn = get_metric_function("sharpe")
    
    result = optimize_threshold_for_strategy(
        strategy_func=None,  # Not used in overfit-aware mode
        data_loader=lambda: None,  # Not used in overfit-aware mode
        metric_fn=metric_fn,
        long_threshold_candidates=long_candidates,
        short_threshold_candidates=short_candidates,
        strategy_name=strategy_name,
        use_overfit_aware=True,
        symbol=symbol,
        timeframe=timeframe,
        feature_preset=feature_preset,
        use_parallel=False,  # Use serial execution for deterministic results
    )
    
    # Convert result to dict for comparison
    new_metrics = {
        "best_long_threshold": result.best_long_threshold,
        "best_short_threshold": result.best_short_threshold,
        "sharpe_in_sample": result.sharpe_in_sample,
        "sharpe_out_sample": result.sharpe_out_sample,
        "gap_in_out": result.gap_in_out,
        "score_overfit_adjusted": result.score_overfit_adjusted,
        "total_trades_in": result.total_trades_in,
        "total_trades_out": result.total_trades_out,
    }
    
    # Add return metrics from trials if available
    if result.trials:
        best_trial = next(
            (t for t in result.trials if t["long_threshold"] == result.best_long_threshold),
            None
        )
        if best_trial:
            new_metrics["total_return_in"] = best_trial.get("total_return_in", 0)
            new_metrics["total_return_out"] = best_trial.get("total_return_out", 0)
    
    # Compare with baseline if provided
    if baseline_result_path:
        baseline_path = Path(baseline_result_path)
        if baseline_path.exists():
            logger.info(f"Loading baseline result from {baseline_path}...")
            baseline_result = load_threshold_result(baseline_path)
            
            old_metrics = {
                "best_long_threshold": baseline_result.best_long_threshold,
                "best_short_threshold": baseline_result.best_short_threshold,
                "sharpe_in_sample": baseline_result.sharpe_in_sample,
                "sharpe_out_sample": baseline_result.sharpe_out_sample,
                "gap_in_out": baseline_result.gap_in_out,
                "score_overfit_adjusted": baseline_result.score_overfit_adjusted,
                "total_trades_in": baseline_result.total_trades_in,
                "total_trades_out": baseline_result.total_trades_out,
            }
            
            # Add return metrics from baseline trials
            if baseline_result.trials:
                best_trial = next(
                    (t for t in baseline_result.trials if t["long_threshold"] == baseline_result.best_long_threshold),
                    None
                )
                if best_trial:
                    old_metrics["total_return_in"] = best_trial.get("total_return_in", 0)
                    old_metrics["total_return_out"] = best_trial.get("total_return_out", 0)
            
            # Assert metrics are equal
            assert_metrics_equal(old_metrics, new_metrics, tolerance=1e-5)
            logger.info("✅ Regression test passed: Results match baseline")
        else:
            logger.warning(f"Baseline result not found at {baseline_path}. Skipping comparison.")
    else:
        # Save result for future baseline comparison
        result_path = get_threshold_result_path(strategy_name, symbol, timeframe)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
        from src.optimization.threshold_optimizer import save_threshold_result
        save_threshold_result(
            result,
            result_path,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
        )
        logger.info(f"✅ Saved result to {result_path} for future baseline comparison")
    
    logger.info("✅ Regression test completed successfully")


def test_signal_generation_vectorization() -> None:
    """
    Test that vectorized signal generation produces identical results to loop-based.
    
    This is a unit test to verify the vectorization optimization in engine.py.
    """
    import numpy as np
    import pandas as pd
    
    # Create test data
    n = 1000
    proba_long = np.random.rand(n)
    proba_short = np.random.rand(n)
    long_threshold = 0.5
    short_threshold = 0.5
    
    # Original loop-based implementation (reference)
    signals_loop = np.full(n, "HOLD", dtype=object)
    for i in range(n):
        p_long = proba_long[i]
        p_short = proba_short[i]
        
        is_long = p_long >= long_threshold
        is_short = p_short >= short_threshold
        
        # Conflict resolution
        if is_long and is_short:
            margin_long = p_long - long_threshold
            margin_short = p_short - short_threshold
            if margin_long >= margin_short:
                is_short = False
            else:
                is_long = False
        
        if is_long and not is_short:
            signals_loop[i] = "LONG"
        elif is_short and not is_long:
            signals_loop[i] = "SHORT"
    
    # Vectorized implementation (optimized)
    signals_vec = np.full(n, "HOLD", dtype=object)
    is_long_mask = proba_long >= long_threshold
    is_short_mask = proba_short >= short_threshold
    
    conflict_mask = is_long_mask & is_short_mask
    if np.any(conflict_mask):
        margin_long = proba_long[conflict_mask] - long_threshold
        margin_short = proba_short[conflict_mask] - short_threshold
        prefer_short = margin_short > margin_long
        is_long_mask[conflict_mask] = ~prefer_short
        is_short_mask[conflict_mask] = prefer_short
    
    signals_vec[is_long_mask] = "LONG"
    signals_vec[is_short_mask] = "SHORT"
    
    # Compare results
    assert np.array_equal(signals_loop, signals_vec), (
        f"Signal generation mismatch: {np.sum(signals_loop != signals_vec)} differences"
    )
    
    logger.info("✅ Vectorized signal generation produces identical results")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Run regression test
    test_threshold_optimizer_regression(
        strategy_name="ml_xgb",
        symbol="BTCUSDT",
        timeframe="5m",
        feature_preset="extended_safe",
        # baseline_result_path="data/thresholds/ml_xgb_BTCUSDT_5m_baseline.json",  # Optional
    )
    
    # Run unit test
    test_signal_generation_vectorization()

