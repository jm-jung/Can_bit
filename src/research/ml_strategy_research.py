"""
ML Strategy Research Module

Automated experiments for analyzing and improving ML-based trading strategies.
This module provides tools to:
- Analyze cost impact (commission/slippage)
- Test long-only / short-only modes
- Grid search for threshold optimization
- Compare event feature impact
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest_with_ml
from src.core.config import PROJECT_ROOT
from src.features.ml_feature_config import MLFeatureConfig

logger = logging.getLogger(__name__)

# Research results directory
RESEARCH_RESULTS_DIR = PROJECT_ROOT / "data" / "research"
RESEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _extract_results(result) -> Dict[str, Any]:
    """Extract key metrics from BacktestResult (TypedDict)."""
    return {
        "total_return": result["total_return"],
        "sharpe": result.get("sharpe", 0.0),  # May not be in result
        "win_rate": result["win_rate"],
        "trades": result["total_trades"],
        "long_trades": result["long_trades"],
        "short_trades": result["short_trades"],
        "max_drawdown": result["max_drawdown"],
    }


def _calculate_sharpe(equity_curve: List[float]) -> float:
    """Calculate Sharpe ratio from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = (mean_return / std_return) * np.sqrt(252 * 24 * 60) if std_return > 0 else 0.0  # Annualized for 1m data
    return float(sharpe)


def experiment_cost_impact(
    strategy_name: str = "ml_xgb",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    use_optimized_threshold: bool = False,
) -> List[Dict[str, Any]]:
    """
    Compare backtest results with and without costs (commission/slippage).
    
    Returns:
        List of experiment results
    """
    logger.info("=" * 60)
    logger.info("[Research] Experiment: Cost Impact Analysis")
    logger.info("=" * 60)
    
    results = []
    
    # Run with costs (default)
    logger.info("[Research] Running backtest WITH costs (commission + slippage)...")
    result_with_costs = run_backtest_with_ml(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_threshold=use_optimized_threshold,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        commission_rate=None,  # Use default
        slippage_rate=None,  # Use default
        signal_confirmation_bars=1,
        use_trend_filter=False,
        trend_ema_window=200,
        take_profit_pct=None,
        stop_loss_pct=None,
    )
    
    result_with_costs_dict = _extract_results(result_with_costs)
    result_with_costs_dict["sharpe"] = _calculate_sharpe(result_with_costs["equity_curve"])
    results.append({
        "experiment": "cost_impact",
        "params": {"mode": "with_costs"},
        "results": result_with_costs_dict,
    })
    
    # Run without costs
    logger.info("[Research] Running backtest WITHOUT costs (commission=0, slippage=0)...")
    result_no_costs = run_backtest_with_ml(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_threshold=use_optimized_threshold,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        commission_rate=0.0,
        slippage_rate=0.0,
        signal_confirmation_bars=1,
        use_trend_filter=False,
        trend_ema_window=200,
        take_profit_pct=None,
        stop_loss_pct=None,
    )
    
    result_no_costs_dict = _extract_results(result_no_costs)
    result_no_costs_dict["sharpe"] = _calculate_sharpe(result_no_costs["equity_curve"])
    results.append({
        "experiment": "cost_impact",
        "params": {"mode": "no_costs"},
        "results": result_no_costs_dict,
    })
    
    # Compare and log
    logger.info("[Research] Cost Impact Comparison:")
    logger.info(f"  With costs:    return={result_with_costs_dict['total_return']:.4f}, sharpe={result_with_costs_dict['sharpe']:.4f}, trades={result_with_costs_dict['trades']}")
    logger.info(f"  Without costs: return={result_no_costs_dict['total_return']:.4f}, sharpe={result_no_costs_dict['sharpe']:.4f}, trades={result_no_costs_dict['trades']}")
    
    return_diff = result_no_costs_dict['total_return'] - result_with_costs_dict['total_return']
    sharpe_diff = result_no_costs_dict['sharpe'] - result_with_costs_dict['sharpe']
    
    logger.info(f"  Difference:    return_diff={return_diff:.4f}, sharpe_diff={sharpe_diff:.4f}")
    
    if result_no_costs_dict['total_return'] < 0:
        logger.warning("[Research] ⚠️  Strategy has no edge: negative return even without costs")
    elif return_diff > 0.05:  # 5% difference
        logger.info("[Research] ✓ Cost structure is causing significant losses")
    else:
        logger.info("[Research] Cost impact is minimal")
    
    return results


def experiment_long_short_only(
    strategy_name: str = "ml_xgb",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    use_optimized_threshold: bool = False,
) -> List[Dict[str, Any]]:
    """
    Test LONG-only and SHORT-only modes.
    
    Returns:
        List of experiment results
    """
    logger.info("=" * 60)
    logger.info("[Research] Experiment: LONG-only vs SHORT-only")
    logger.info("=" * 60)
    
    results = []
    
    # Determine optimal thresholds
    opt_long = long_threshold
    opt_short = short_threshold
    
    if use_optimized_threshold:
        from src.optimization.threshold_loader import load_optimized_thresholds
        try:
            opt_long, opt_short, _ = load_optimized_thresholds(strategy_name, symbol, timeframe)
        except Exception as e:
            logger.warning(f"[Research] Failed to load optimized thresholds: {e}. Using provided/default values.")
    
    if opt_long is None:
        logger.warning("[Research] long_threshold is None. Cannot run LONG-only experiment.")
        return results
    
    # LONG-only mode
    logger.info(f"[Research] Running LONG-only backtest (long_threshold={opt_long:.3f}, short_threshold=None)...")
    result_long_only = run_backtest_with_ml(
        long_threshold=opt_long,
        short_threshold=None,
        use_optimized_threshold=False,  # Already loaded
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        long_only=True,
        signal_confirmation_bars=1,
        use_trend_filter=False,
        trend_ema_window=200,
        take_profit_pct=None,
        stop_loss_pct=None,
    )
    
    result_long_dict = _extract_results(result_long_only)
    result_long_dict["sharpe"] = _calculate_sharpe(result_long_only["equity_curve"])
    results.append({
        "experiment": "long_short_only",
        "params": {"mode": "long_only", "long_threshold": opt_long},
        "results": result_long_dict,
    })
    
    # SHORT-only mode
    if opt_short is not None:
        logger.info(f"[Research] Running SHORT-only backtest (long_threshold=None, short_threshold={opt_short:.3f})...")
        result_short_only = run_backtest_with_ml(
            long_threshold=None,
            short_threshold=opt_short,
            use_optimized_threshold=False,  # Already loaded
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            short_only=True,
            signal_confirmation_bars=1,
            use_trend_filter=False,
            trend_ema_window=200,
            take_profit_pct=None,
            stop_loss_pct=None,
        )
        
        result_short_dict = _extract_results(result_short_only)
        result_short_dict["sharpe"] = _calculate_sharpe(result_short_only["equity_curve"])
        results.append({
            "experiment": "long_short_only",
            "params": {"mode": "short_only", "short_threshold": opt_short},
            "results": result_short_dict,
        })
    else:
        logger.warning("[Research] short_threshold is None. Skipping SHORT-only experiment.")
    
    # Summary table
    logger.info("[Research] LONG-only / SHORT-only Summary:")
    logger.info("  Mode       | Return   | Sharpe  | Trades | Win Rate")
    logger.info("  " + "-" * 50)
    for res in results:
        mode = res["params"]["mode"]
        r = res["results"]
        logger.info(f"  {mode:10} | {r['total_return']:8.4f} | {r['sharpe']:7.4f} | {r['trades']:6} | {r['win_rate']:8.2%}")
    
    return results


def experiment_threshold_strengthening(
    strategy_name: str = "ml_xgb",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    use_optimized_threshold: bool = False,
) -> List[Dict[str, Any]]:
    """
    Test stronger thresholds (higher long, lower short).
    
    Returns:
        List of experiment results sorted by Sharpe ratio
    """
    logger.info("=" * 60)
    logger.info("[Research] Experiment: Threshold Strengthening")
    logger.info("=" * 60)
    
    # Get baseline thresholds
    baseline_long = None
    baseline_short = None
    
    if use_optimized_threshold:
        from src.optimization.threshold_loader import load_optimized_thresholds
        try:
            baseline_long, baseline_short, _ = load_optimized_thresholds(strategy_name, symbol, timeframe)
        except Exception as e:
            logger.warning(f"[Research] Failed to load optimized thresholds: {e}. Using default grid.")
    
    # Define threshold grid
    if baseline_long is not None:
        long_candidates = [baseline_long, 0.75, 0.80, 0.85]
    else:
        long_candidates = [0.70, 0.75, 0.80, 0.85]
    
    if baseline_short is not None:
        short_candidates = [baseline_short, 0.25, 0.20, 0.15]
    else:
        short_candidates = [0.30, 0.25, 0.20, 0.15]
    
    results = []
    
    logger.info(f"[Research] Testing {len(long_candidates)} x {len(short_candidates)} = {len(long_candidates) * len(short_candidates)} threshold combinations...")
    
    for long_th in long_candidates:
        for short_th in short_candidates:
            logger.info(f"[Research] Testing: long={long_th:.2f}, short={short_th:.2f}...")
            
            result = run_backtest_with_ml(
                long_threshold=long_th,
                short_threshold=short_th,
                use_optimized_threshold=False,  # Using explicit values
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                signal_confirmation_bars=1,
                use_trend_filter=False,
                trend_ema_window=200,
                take_profit_pct=None,
                stop_loss_pct=None,
            )
            
            result_dict = _extract_results(result)
            result_dict["sharpe"] = _calculate_sharpe(result["equity_curve"])
            
            results.append({
                "experiment": "threshold_strengthening",
                "params": {"long_threshold": long_th, "short_threshold": short_th},
                "results": result_dict,
            })
    
    # Sort by Sharpe ratio
    results.sort(key=lambda x: x["results"]["sharpe"], reverse=True)
    
    # Log top 3
    logger.info("[Research] Top 3 Threshold Combinations (by Sharpe):")
    logger.info("  Rank | Long  | Short | Return   | Sharpe  | Trades | Win Rate")
    logger.info("  " + "-" * 60)
    for i, res in enumerate(results[:3], 1):
        p = res["params"]
        r = res["results"]
        logger.info(f"  {i:4} | {p['long_threshold']:5.2f} | {p['short_threshold']:5.2f} | {r['total_return']:8.4f} | {r['sharpe']:7.4f} | {r['trades']:6} | {r['win_rate']:8.2%}")
    
    return results


def experiment_event_features_on_off(
    strategy_name: str = "ml_xgb",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    use_optimized_threshold: bool = False,
) -> List[Dict[str, Any]]:
    """
    Compare backtest with event features ON vs OFF.
    
    Note: This requires modifying the model prediction to skip event features.
    For now, we'll run two backtests and note the difference.
    This is a placeholder - actual implementation may require model modification.
    
    Returns:
        List of experiment results
    """
    logger.info("=" * 60)
    logger.info("[Research] Experiment: Event Features ON/OFF")
    logger.info("=" * 60)
    logger.warning("[Research] ⚠️  Event feature toggle requires model modification.")
    logger.warning("[Research]    This experiment runs with event features ON (default behavior).")
    logger.warning("[Research]    To properly test OFF mode, model.predict_proba_latest needs to skip event features.")
    
    results = []
    
    # Run with event features (default behavior)
    logger.info("[Research] Running backtest WITH event features (default)...")
    result_with_events = run_backtest_with_ml(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_threshold=use_optimized_threshold,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        signal_confirmation_bars=1,
        use_trend_filter=False,
        trend_ema_window=200,
        take_profit_pct=None,
        stop_loss_pct=None,
    )
    
    result_with_dict = _extract_results(result_with_events)
    result_with_dict["sharpe"] = _calculate_sharpe(result_with_events["equity_curve"])
    results.append({
        "experiment": "event_features",
        "params": {"mode": "with_events"},
        "results": result_with_dict,
    })
    
    # Note: To properly test without events, we would need to:
    # 1. Modify the model's predict_proba_latest to accept a flag
    # 2. Or create a wrapper that filters out event features
    # For now, we log a placeholder
    logger.info("[Research] Event Features Comparison:")
    logger.info(f"  With events: return={result_with_dict['total_return']:.4f}, sharpe={result_with_dict['sharpe']:.4f}, trades={result_with_dict['trades']}")
    logger.warning("[Research]    Without events: Not implemented (requires model modification)")
    
    return results


def experiment_feature_presets(
    strategy_name: str = "ml_xgb",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    use_optimized_threshold: bool = False,
    model_paths: dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Compare backtest results across different feature presets.
    
    This experiment requires pre-trained models for each preset.
    If model_paths is not provided, it will attempt to use default paths
    with preset suffixes.
    
    Args:
        strategy_name: Strategy identifier
        symbol: Trading symbol
        timeframe: Timeframe
        long_threshold: Optional long threshold override
        short_threshold: Optional short threshold override
        use_optimized_threshold: Whether to use optimized thresholds
        model_paths: Dict mapping preset names to model file paths
            Example: {"base": "data/models/ml_xgb_base.pkl", ...}
            If None, will attempt to construct paths from default model path
    
    Returns:
        List of experiment results
    """
    logger.info("=" * 60)
    logger.info("[Research] Experiment: Feature Presets Comparison")
    logger.info("=" * 60)
    
    if model_paths is None:
        # Try to construct default paths
        from src.core.config import settings
        from pathlib import Path
        
        base_model_path = Path(settings.XGB_MODEL_PATH)
        model_paths = {
            "base": str(base_model_path),
            "extended_safe": str(base_model_path.parent / f"{base_model_path.stem}_extended_safe{base_model_path.suffix}"),
            "extended_full": str(base_model_path.parent / f"{base_model_path.stem}_extended_full{base_model_path.suffix}"),
        }
        logger.info(f"[Research] Using default model paths: {model_paths}")
    
    results = []
    presets_to_test = ["base", "extended_safe", "extended_full"]
    
    for preset in presets_to_test:
        if preset not in model_paths:
            logger.warning(f"[Research] Model path not provided for preset '{preset}'. Skipping.")
            continue
        
        model_path = model_paths[preset]
        if not Path(model_path).exists():
            logger.warning(f"[Research] Model file not found: {model_path}. Skipping preset '{preset}'.")
            continue
        
        logger.info(f"[Research] Testing preset: {preset} (model: {model_path})")
        
        # Note: This experiment assumes models are already trained.
        # For now, we'll run backtest with the existing model loading mechanism.
        # In a full implementation, we might need to modify run_backtest_with_ml
        # to accept a model_path parameter, or create a wrapper that loads the model
        # before calling run_backtest_with_ml.
        
        # For now, log a note that this requires manual model switching
        logger.warning(
            f"[Research] ⚠️  Feature preset experiment requires model switching. "
            f"Current implementation assumes models are pre-trained and available. "
            f"To fully test this, you need to: "
            f"1. Train models with each preset (--feature-preset flag)"
            f"2. Modify run_backtest_with_ml to accept model_path parameter"
            f"3. Or manually switch models between runs"
        )
        
        # For now, we'll just run with base preset and log the structure
        # A full implementation would require model path injection in run_backtest_with_ml
        result = run_backtest_with_ml(
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            use_optimized_threshold=use_optimized_threshold,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            signal_confirmation_bars=1,
            use_trend_filter=False,
            trend_ema_window=200,
            take_profit_pct=None,
            stop_loss_pct=None,
        )
        
        result_dict = _extract_results(result)
        result_dict["sharpe"] = _calculate_sharpe(result["equity_curve"])
        results.append({
            "experiment": "feature_presets",
            "params": {
                "preset": preset,
                "model_path": model_path,
            },
            "results": result_dict,
        })
    
    # Summary table
    if results:
        logger.info("[Research] Feature Presets Summary:")
        logger.info("  Preset         | Return   | Sharpe  | Trades | Win Rate")
        logger.info("  " + "-" * 60)
        for res in results:
            preset = res["params"]["preset"]
            r = res["results"]
            logger.info(f"  {preset:14} | {r['total_return']:8.4f} | {r['sharpe']:7.4f} | {r['trades']:6} | {r['win_rate']:8.2%}")
    
    return results


def run_all_experiments(
    strategy_name: str = "ml_xgb",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    use_optimized_threshold: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run all research experiments in sequence.
    
    Returns:
        List of all experiment results
    """
    logger.info("=" * 80)
    logger.info("[Research] ========================================")
    logger.info("[Research] ML Strategy Research Suite")
    logger.info("[Research] ========================================")
    logger.info(f"[Research] Strategy: {strategy_name}")
    logger.info(f"[Research] Symbol: {symbol}, Timeframe: {timeframe}")
    logger.info(f"[Research] Use optimized thresholds: {use_optimized_threshold}")
    logger.info("=" * 80)
    
    all_results = []
    
    # 1. Cost impact experiment
    logger.info("")
    logger.info("[Research] >>> Experiment 1: Cost Impact Analysis")
    logger.info("")
    cost_results = experiment_cost_impact(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_threshold=use_optimized_threshold,
    )
    all_results.extend(cost_results)
    
    # 2. Long-only / Short-only experiment
    logger.info("")
    logger.info("[Research] >>> Experiment 2: LONG-only / SHORT-only")
    logger.info("")
    long_short_results = experiment_long_short_only(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_threshold=use_optimized_threshold,
    )
    all_results.extend(long_short_results)
    
    # 3. Threshold strengthening experiment
    logger.info("")
    logger.info("[Research] >>> Experiment 3: Threshold Strengthening")
    logger.info("")
    threshold_results = experiment_threshold_strengthening(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        use_optimized_threshold=use_optimized_threshold,
    )
    all_results.extend(threshold_results)
    
    # 4. Event features ON/OFF experiment
    logger.info("")
    logger.info("[Research] >>> Experiment 4: Event Features ON/OFF")
    logger.info("")
    event_results = experiment_event_features_on_off(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_threshold=use_optimized_threshold,
    )
    all_results.extend(event_results)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESEARCH_RESULTS_DIR / f"research_results_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "use_optimized_threshold": use_optimized_threshold,
        "experiments": all_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("[Research] ========================================")
    logger.info("[Research] Research Suite Completed")
    logger.info("[Research] ========================================")
    logger.info(f"[Research] Total experiments: {len(all_results)}")
    logger.info(f"[Research] Results saved to: {output_path}")
    logger.info("=" * 80)
    
    return all_results

