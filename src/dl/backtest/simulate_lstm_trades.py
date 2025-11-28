"""
Backtest simulation for LSTM+Attention model on test set.

This module provides a simple backtest skeleton that simulates trades
based on model predictions and actual future returns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

from src.core.config import settings
from src.indicators.basic import add_basic_indicators
from src.ml.features import build_ml_dataset
from src.services.ohlcv_service import load_ohlcv_df
from src.dl.models.lstm_attn import LSTMAttentionModel
from src.dl.data.split import make_time_series_splits
from src.dl.train.train_lstm_attn import create_sequences, TimeSeriesDataset
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compute_trade_stats(
    trade_returns_raw: np.ndarray,
    trade_returns_after_fee: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute statistics for a set of trades.
    
    Args:
        trade_returns_raw: Raw returns (before fees)
        trade_returns_after_fee: Net returns (after fees)
    
    Returns:
        Dictionary with n_trades, win_rate, avg_return, avg_return_after_fee, cum_return
    """
    n_trades = len(trade_returns_raw)
    
    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "avg_return_after_fee": 0.0,
            "cum_return": 0.0,
        }
    
    win_rate = float(np.mean(trade_returns_after_fee > 0))
    avg_return = float(np.mean(trade_returns_raw))
    avg_return_after_fee = float(np.mean(trade_returns_after_fee))
    cum_return = float(np.sum(trade_returns_after_fee))
    
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "avg_return_after_fee": avg_return_after_fee,
        "cum_return": cum_return,
    }


def _simulate_lstm_strategy(
    future_returns: np.ndarray,
    prob_up: np.ndarray,
    thresholds: list[float],
    strategy_mode: str,
    fee_rate: float,
) -> Dict[float, Dict[str, Any]]:
    """
    Simulate LSTM-driven strategy for a given strategy mode.
    
    Args:
        future_returns: Actual forward returns (shape: (N,))
        prob_up: Model predicted probabilities (shape: (N,))
        thresholds: List of probability thresholds to test
        strategy_mode: "long_only" or "long_short"
        fee_rate: Per-trade fee rate (0.0004 = 0.04% per trade, 0.08% round trip)
    
    Returns:
        Dictionary mapping threshold to stats dict
    """
    results = {}
    N = len(future_returns)
    
    for threshold in thresholds:
        if strategy_mode == "long_only":
            # Long-only: buy when prob_up >= threshold
            signal = prob_up >= threshold
            raw_returns = future_returns[signal]
        
        elif strategy_mode == "long_short":
            # Long/Short: symmetric around 0.5
            long_threshold = threshold
            short_threshold = 1.0 - threshold
            
            # Generate signals
            long_signal = prob_up >= long_threshold
            short_signal = prob_up <= short_threshold
            
            # Compute raw returns
            raw_returns_list = []
            for i in range(N):
                if long_signal[i]:
                    # Long position: profit when future_return > 0
                    raw_returns_list.append(future_returns[i])
                elif short_signal[i]:
                    # Short position: profit when future_return < 0
                    raw_returns_list.append(-future_returns[i])
                # else: no trade
            
            raw_returns = np.array(raw_returns_list) if raw_returns_list else np.array([])
        
        else:
            raise ValueError(f"Unknown strategy_mode: {strategy_mode}")
        
        # Apply fees
        if len(raw_returns) > 0:
            net_returns = raw_returns - 2 * fee_rate
        else:
            net_returns = np.array([])
        
        # Compute stats
        stats = _compute_trade_stats(raw_returns, net_returns)
        results[threshold] = stats
    
    return results


def _simulate_oracle_strategy(
    future_returns: np.ndarray,
    strategy_mode: str,
    fee_rate: float,
) -> Dict[str, Any]:
    """
    Simulate Oracle strategy (perfect future knowledge).
    
    Args:
        future_returns: Actual forward returns (shape: (N,))
        strategy_mode: "long_only" or "long_short"
        fee_rate: Per-trade fee rate
    
    Returns:
        Stats dictionary
    """
    if strategy_mode == "long_only":
        # Oracle long-only: trade whenever future_return > 0
        signal = future_returns > 0
        raw_returns = future_returns[signal]
    
    elif strategy_mode == "long_short":
        # Oracle long/short: always pick the correct direction
        raw_returns_list = []
        for ret in future_returns:
            if ret > 0:
                # Long position
                raw_returns_list.append(ret)
            elif ret < 0:
                # Short position
                raw_returns_list.append(-ret)
            # ret == 0: no trade
        
        raw_returns = np.array(raw_returns_list) if raw_returns_list else np.array([])
    
    else:
        raise ValueError(f"Unknown strategy_mode: {strategy_mode}")
    
    # Apply fees
    if len(raw_returns) > 0:
        net_returns = raw_returns - 2 * fee_rate
    else:
        net_returns = np.array([])
    
    # Compute stats
    return _compute_trade_stats(raw_returns, net_returns)


def _simulate_random_strategy(
    future_returns: np.ndarray,
    n_ref_trades: int,
    strategy_mode: str,
    fee_rate: float,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate Random strategy with fixed number of trades.
    
    Args:
        future_returns: Actual forward returns (shape: (N,))
        n_ref_trades: Number of trades to simulate (reference from LSTM)
        strategy_mode: "long_only" or "long_short"
        fee_rate: Per-trade fee rate
        random_seed: Random seed for reproducibility
    
    Returns:
        Stats dictionary
    """
    N = len(future_returns)
    
    if N == 0 or n_ref_trades <= 0:
        return _compute_trade_stats(np.array([]), np.array([]))
    
    # Use numpy random generator for reproducibility
    rng = np.random.default_rng(random_seed)
    
    # Randomly pick n_ref_trades indices (without replacement)
    n_actual = min(n_ref_trades, N)
    idxs = rng.choice(N, size=n_actual, replace=False)
    
    if strategy_mode == "long_only":
        # Random long-only: randomly pick indices, all long
        raw_returns = future_returns[idxs]
    
    elif strategy_mode == "long_short":
        # Random long/short: randomly pick indices, randomly choose direction
        raw_returns_list = []
        for idx in idxs:
            # Randomly choose long or short with equal probability
            is_long = rng.random() < 0.5
            if is_long:
                raw_returns_list.append(future_returns[idx])
            else:
                raw_returns_list.append(-future_returns[idx])
        
        raw_returns = np.array(raw_returns_list)
    
    else:
        raise ValueError(f"Unknown strategy_mode: {strategy_mode}")
    
    # Apply fees
    net_returns = raw_returns - 2 * fee_rate
    
    # Compute stats
    return _compute_trade_stats(raw_returns, net_returns)


def compute_buy_hold_return(future_returns: np.ndarray) -> float:
    """
    Buy & Hold baseline: buy at test[0], hold until test[-1].
    
    Approximate return = cumulative product of (1+future_return) - 1.
    We use horizon returns, so approximate using simple sum.
    """
    if len(future_returns) == 0:
        return 0.0
    # Approx: sum of future_returns across the test set
    cum_ret = float(np.sum(future_returns))
    return cum_ret


def main():
    """
    Main entry point for backtest simulation.
    
    This function:
    1. Loads OHLCV data and creates sequences (same as training)
    2. Splits data using same split utility
    3. Loads best LSTM model
    4. Computes predictions on test set
    5. Runs backtest simulations for both strategy modes and all signal sources
    """
    logger.info("=" * 60)
    logger.info("Backtest Simulation (LSTM+Attention) on Test Split")
    logger.info("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load data (same as training)
    logger.info("Loading OHLCV data...")
    df = load_ohlcv_df()
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of OHLCV data")
    
    # Get settings (same as training)
    window_size = 60
    horizon = settings.LSTM_RETURN_HORIZON
    pos_threshold = settings.LSTM_LABEL_POS_THRESHOLD
    neg_threshold = settings.LSTM_LABEL_NEG_THRESHOLD
    ignore_margin = settings.LSTM_LABEL_IGNORE_MARGIN
    
    logger.info(f"Window size: {window_size}, Horizon: {horizon}")
    logger.info(f"Label thresholds: pos={pos_threshold:.4f}, neg={neg_threshold:.4f}")
    logger.info(f"Ignore margin: {ignore_margin:.4f}")
    
    # Create sequences (same as training)
    logger.info("Creating sequences...")
    X, y, feature_cols, meta = create_sequences(
        df,
        window_size=window_size,
        horizon=horizon,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
        ignore_margin=ignore_margin,
        debug_inspect=False,
    )
    feature_dim = len(feature_cols)
    logger.info(f"Created {len(X)} sequences with {feature_dim} features")
    
    # Split data (same as training)
    logger.info("Splitting data...")
    splits = make_time_series_splits(
        X,
        y,
        train_ratio=0.7,
        valid_ratio=0.15,
        min_test_samples=200,
        meta={"future_returns": meta["future_returns"]},
    )
    
    X_test = splits["data"]["X_test"]
    y_test = splits["data"]["y_test"]
    future_returns_test = splits["meta"]["future_returns"]["test"]
    
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Load best model
    model_path = Path(settings.LSTM_ATTN_MODEL_PATH)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path.resolve()}")
        logger.error("Please train the model first using: python -m src.dl.train.train_lstm_attn")
        return
    
    logger.info(f"Loading model from {model_path.resolve()}")
    
    # Model architecture (should match training)
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    
    model = LSTMAttentionModel(
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully")
    
    # Compute predictions on test set
    logger.info("Computing predictions on test set...")
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
    
    prob_up_test = np.array(all_probs)
    logger.info(f"Computed {len(prob_up_test)} predictions")
    
    # Configuration
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    fee_rate = 0.0004  # 0.04% per trade, 0.08% round trip
    strategy_modes = ["long_only", "long_short"]
    
    # Storage for all results
    all_results = {
        "lstm": {},
        "oracle": {},
        "random": {},
    }
    
    # Run LSTM strategies for both modes
    logger.info("=" * 60)
    logger.info("Running backtest simulation...")
    logger.info("=" * 60)
    
    for strategy_mode in strategy_modes:
        all_results["lstm"][strategy_mode] = _simulate_lstm_strategy(
            future_returns=future_returns_test,
            prob_up=prob_up_test,
            thresholds=thresholds,
            strategy_mode=strategy_mode,
            fee_rate=fee_rate,
        )
    
    # Run Oracle strategies for both modes
    for strategy_mode in strategy_modes:
        all_results["oracle"][strategy_mode] = _simulate_oracle_strategy(
            future_returns=future_returns_test,
            strategy_mode=strategy_mode,
            fee_rate=fee_rate,
        )
    
    # Run Random strategies for both modes
    # Use LSTM threshold=0.5 as reference for number of trades
    for strategy_mode in strategy_modes:
        n_ref = 0
        if 0.5 in all_results["lstm"][strategy_mode]:
            n_ref = all_results["lstm"][strategy_mode][0.5]["n_trades"]
        
        all_results["random"][strategy_mode] = _simulate_random_strategy(
            future_returns=future_returns_test,
            n_ref_trades=n_ref,
            strategy_mode=strategy_mode,
            fee_rate=fee_rate,
            random_seed=42,
        )
    
    # Compute Buy & Hold
    buyhold_cum = compute_buy_hold_return(future_returns_test)
    
    # Log results
    logger.info("")
    logger.info("=" * 60)
    logger.info("Backtest Results")
    logger.info("=" * 60)
    
    # LSTM results for each strategy mode
    for strategy_mode in strategy_modes:
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"Strategy: {strategy_mode} (LSTM)")
        logger.info("-" * 60)
        logger.info(f"{'Threshold':<12} {'n_trades':<12} {'win_rate':<12} {'avg_ret':<12} {'avg_ret_fee':<15} {'cum_ret':<12}")
        logger.info("-" * 60)
        
        for threshold in thresholds:
            r = all_results["lstm"][strategy_mode][threshold]
            logger.info(
                f"{threshold:<12.2f} {r['n_trades']:<12} "
                f"{r['win_rate']:<12.4f} {r['avg_return']:<12.6f} "
                f"{r['avg_return_after_fee']:<15.6f} {r['cum_return']:<12.6f}"
            )
    
    # Baselines
    logger.info("")
    logger.info("-" * 60)
    logger.info("Baselines")
    logger.info("-" * 60)
    
    # Buy & Hold
    logger.info(f"Buy & Hold: cum_ret={buyhold_cum:.6f}")
    
    # Oracle baselines
    for strategy_mode in strategy_modes:
        r = all_results["oracle"][strategy_mode]
        logger.info(
            f"Oracle [{strategy_mode}]: n_trades={r['n_trades']}, "
            f"win_rate={r['win_rate']:.4f}, avg_ret={r['avg_return']:.6f}, "
            f"cum_ret={r['cum_return']:.6f}"
        )
    
    # Random baselines
    for strategy_mode in strategy_modes:
        r = all_results["random"][strategy_mode]
        n_ref = 0
        if 0.5 in all_results["lstm"][strategy_mode]:
            n_ref = all_results["lstm"][strategy_mode][0.5]["n_trades"]
        logger.info(
            f"Random [{strategy_mode}] (n={n_ref}): "
            f"win_rate={r['win_rate']:.4f}, avg_ret={r['avg_return']:.6f}, "
            f"cum_ret={r['cum_return']:.6f}"
        )
    
    logger.info("-" * 60)
    logger.info("Note: This is a skeleton implementation.")
    logger.info("Future enhancements: position overlap handling, holding period management, drawdown analysis, etc.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
