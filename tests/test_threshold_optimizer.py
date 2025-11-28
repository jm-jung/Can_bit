"""
Tests for threshold optimization functionality.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.optimization.threshold_optimizer import (
    ThresholdOptimizerResult,
    save_threshold_result,
    load_threshold_result,
    optimize_threshold_for_strategy,
)
from src.backtest.engine import BacktestResult


def test_save_load_roundtrip():
    """Test saving and loading threshold result."""
    result = ThresholdOptimizerResult(
        best_long_threshold=0.55,
        best_short_threshold=0.45,
        best_metric_value=0.123,
        metric_name="sharpe",
        trials=[
            {"long_threshold": 0.5, "short_threshold": 0.5, "metric_value": 0.1},
            {"long_threshold": 0.55, "short_threshold": 0.45, "metric_value": 0.123},
        ],
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_threshold.json"
        save_threshold_result(result, path)
        
        assert path.exists()
        
        loaded = load_threshold_result(path)
        assert loaded.best_long_threshold == result.best_long_threshold
        assert loaded.best_short_threshold == result.best_short_threshold
        assert loaded.best_metric_value == result.best_metric_value
        assert loaded.metric_name == result.metric_name
        assert len(loaded.trials) == len(result.trials)


def test_threshold_optimizer_basic():
    """Test threshold optimizer picks best threshold combination."""
    # Mock backtest function
    def mock_backtest(long_threshold: float, short_threshold: float | None = None) -> BacktestResult:
        # Simulate: best performance at long=0.6, short=0.4
        if long_threshold == 0.6 and short_threshold == 0.4:
            return BacktestResult(
                total_return=0.2,
                win_rate=0.6,
                max_drawdown=-0.1,
                trades=[],
                equity_curve=[1.0, 1.1, 1.2],
                total_trades=10,
                long_trades=5,
                short_trades=5,
                avg_profit=0.02,
                median_profit=0.02,
                avg_win=0.05,
                avg_loss=-0.02,
                max_consecutive_wins=3,
                max_consecutive_losses=2,
            )
        else:
            return BacktestResult(
                total_return=0.1,
                win_rate=0.5,
                max_drawdown=-0.15,
                trades=[],
                equity_curve=[1.0, 1.05, 1.1],
                total_trades=8,
                long_trades=4,
                short_trades=4,
                avg_profit=0.01,
                median_profit=0.01,
                avg_win=0.03,
                avg_loss=-0.02,
                max_consecutive_wins=2,
                max_consecutive_losses=2,
            )
    
    # Mock metric function (total return)
    def metric_fn(result: BacktestResult) -> float:
        return result["total_return"]
    
    # Run optimization
    result = optimize_threshold_for_strategy(
        strategy_func=Mock(),
        data_loader=lambda: None,
        metric_fn=metric_fn,
        long_threshold_candidates=[0.5, 0.6, 0.7],
        short_threshold_candidates=[0.3, 0.4, 0.5],
        run_backtest_func=mock_backtest,
    )
    
    # Check that best threshold is selected
    assert result.best_long_threshold == 0.6
    assert result.best_short_threshold == 0.4
    assert result.best_metric_value == 0.2
    assert len(result.trials) == 9  # 3 long * 3 short


def test_strategy_uses_loaded_threshold():
    """Test that strategy can use loaded thresholds when flag is enabled."""
    from src.strategies.ml_xgb import ml_xgb_strategy, _load_optimized_thresholds
    
    # Create a temporary threshold file
    with tempfile.TemporaryDirectory() as tmpdir:
        threshold_dir = Path(tmpdir) / "data" / "thresholds"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        
        threshold_path = threshold_dir / "ml_xgb_BTCUSDT_1m.json"
        result = ThresholdOptimizerResult(
            best_long_threshold=0.6,
            best_short_threshold=0.4,
            best_metric_value=0.15,
            metric_name="sharpe",
            trials=[],
        )
        save_threshold_result(result, threshold_path)
        
        # Mock settings to enable optimized thresholds
        with patch("src.strategies.ml_xgb.settings") as mock_settings:
            mock_settings.USE_OPTIMIZED_THRESHOLDS = True
            
            # Mock Path to point to our temp directory
            with patch("src.strategies.ml_xgb.Path") as mock_path:
                mock_path.return_value = threshold_path
                
                # Test loading
                long_thr, short_thr = _load_optimized_thresholds()
                assert long_thr == 0.6
                assert short_thr == 0.4

