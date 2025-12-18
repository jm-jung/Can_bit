"""
LSTM + Attention ML strategy aligned with the XGB interface.
"""
from __future__ import annotations

import logging
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from src.core.config import settings
from src.dl.lstm_attn_model import get_lstm_attn_model
from src.indicators.basic import get_df_with_indicators
from src.strategies.ml_thresholds import resolve_ml_thresholds
from src.strategies.strategy_mode import StrategyMode
from src.dl.data.labels import LstmClassIndex

Signal = Literal["LONG", "SHORT", "HOLD"]

logger = logging.getLogger(__name__)


class MLStrategyOutput(TypedDict):
    """Output structure for ML strategy."""

    timestamp: str
    close: float
    proba_up: float | None
    signal: Signal


def ml_lstm_attn_strategy(
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    use_optimized_thresholds: bool = True,
    *,
    strategy_name: str = "ml_lstm_attn",
    symbol: str | None = None,
    timeframe: str | None = None,
) -> MLStrategyOutput:
    """
    LSTM Attention-based ML strategy that mirrors the XGB strategy contract.
    """
    long_threshold, short_threshold = resolve_ml_thresholds(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_thresholds=use_optimized_thresholds,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        default_long=settings.LSTM_ATTN_THRESHOLD_UP,
        default_short=settings.LSTM_ATTN_THRESHOLD_DOWN,
    )

    try:
        df = get_df_with_indicators()
        model = get_lstm_attn_model()

        if model is None or not model.is_loaded():
            last_row = df.iloc[-1]
            return MLStrategyOutput(
                timestamp=str(last_row["timestamp"]),
                close=float(last_row["close"]),
                proba_up=None,
                signal="HOLD",
            )

        proba_up = model.predict_proba_latest(df)
        last_row = df.iloc[-1]

        if proba_up >= long_threshold:
            signal: Signal = "LONG"
        elif short_threshold is not None and proba_up <= short_threshold:
            signal = "SHORT"
        else:
            signal = "HOLD"

        logger.debug(
            "[ml_lstm_attn] prob_up=%.4f, long_threshold=%.3f, short_threshold=%s, signal=%s",
            proba_up,
            long_threshold,
            f"{short_threshold:.3f}" if short_threshold is not None else "None",
            signal,
        )

        return MLStrategyOutput(
            timestamp=str(last_row["timestamp"]),
            close=float(last_row["close"]),
            proba_up=proba_up,
            signal=signal,
        )

    except Exception as exc:
        logger.exception("ml_lstm_attn_strategy failed: %s", exc)
        try:
            df = get_df_with_indicators()
            last_row = df.iloc[-1]
            return MLStrategyOutput(
                timestamp=str(last_row["timestamp"]),
                close=float(last_row["close"]),
                proba_up=None,
                signal="HOLD",
            )
        except Exception:
            return MLStrategyOutput(
                timestamp="",
                close=0.0,
                proba_up=None,
                signal="HOLD",
            )


def ml_lstm_attn_strategy_enhanced(
    proba_long_arr: np.ndarray | None = None,
    proba_short_arr: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    strategy_mode: str | StrategyMode = "both",
    use_optimized_thresholds: bool = True,
    *,
    strategy_name: str = "ml_lstm_attn",
    symbol: str | None = None,
    timeframe: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Enhanced LSTM-Attention strategy with mode support (both/long_only/short_only).
    
    This function is designed for backtest/cache generation where proba arrays
    are pre-computed. It generates signals based on strategy mode and thresholds.
    
    Args:
        proba_long_arr: Pre-computed long probabilities (shape: (N,))
        proba_short_arr: Pre-computed short probabilities (shape: (N,))
        df: DataFrame with OHLCV data (must align with proba arrays)
        long_threshold: Probability threshold for LONG signal
        short_threshold: Probability threshold for SHORT signal
        strategy_mode: Strategy mode ("both", "long_only", "short_only") or StrategyMode enum
        use_optimized_thresholds: If True, try to load optimized thresholds from JSON
        strategy_name: Strategy identifier for threshold lookup
        symbol: Optional override for symbol when resolving thresholds
        timeframe: Optional override for timeframe when resolving thresholds
    
    Returns:
        Tuple of (signals: np.ndarray, df_aligned: pd.DataFrame)
        - signals: Array of "LONG", "SHORT", or "HOLD" (shape: (N,))
        - df_aligned: DataFrame aligned with signals (may have fewer rows than input df)
    
    Strategy Mode Behavior:
        - "both": Check both long and short conditions, resolve conflicts
        - "long_only": Only check long condition, ignore short_threshold
        - "short_only": Only check short condition, ignore long_threshold
    
    Conflict Resolution (both mode):
        - If both long and short conditions are met:
          - Choose the one with higher confidence (larger margin above threshold)
          - If margins are equal, prefer LONG
        - This ensures no simultaneous long/short positions
    """
    # Convert strategy_mode to enum
    if isinstance(strategy_mode, str):
        mode = StrategyMode.from_string(strategy_mode)
    else:
        mode = strategy_mode
    
    # Resolve thresholds
    long_threshold, short_threshold = resolve_ml_thresholds(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_thresholds=use_optimized_thresholds,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        default_long=settings.LSTM_ATTN_THRESHOLD_UP,
        default_short=settings.LSTM_ATTN_THRESHOLD_DOWN,
    )
    
    # Load data if not provided
    if df is None:
        df = get_df_with_indicators()
    
    # Compute proba arrays if not provided
    if proba_long_arr is None or proba_short_arr is None:
        model = get_lstm_attn_model()
        if model is None or not model.is_loaded():
            # Return all HOLD signals
            signals = np.full(len(df), "HOLD", dtype=object)
            return signals, df
        
        # For single prediction, use predict_proba_latest with return_both=True
        # IMPORTANT: Use return_both=True to get correct proba_short from 3-class softmax
        # Do NOT use proba_short = 1.0 - proba_long (this is incorrect for 3-class)
        proba_long_val, proba_short_val = model.predict_proba_latest(
            df, 
            symbol=symbol, 
            timeframe=timeframe,
            return_both=True
        )
        proba_long_arr = np.array([proba_long_val])
        proba_short_arr = np.array([proba_short_val])
        # Truncate df to match
        df = df.iloc[-1:].reset_index(drop=True)
    
    # Ensure arrays are aligned
    n_proba = len(proba_long_arr)
    n_df = len(df)
    
    if n_proba != n_df:
        min_len = min(n_proba, n_df)
        proba_long_arr = proba_long_arr[:min_len]
        proba_short_arr = proba_short_arr[:min_len]
        df = df.iloc[:min_len].reset_index(drop=True)
        logger.warning(
            f"[ml_lstm_attn_enhanced] Array length mismatch: "
            f"proba={n_proba}, df={n_df}, using first {min_len} rows"
        )
    
    # Initialize signals array
    signals = np.full(len(df), "HOLD", dtype=object)
    
    # 3-class strategy: Use argmax to determine direction_class, then apply thresholds
    # Compute proba_flat from proba_long and proba_short (3-class softmax: p_flat + p_long + p_short = 1)
    proba_flat_arr = 1.0 - proba_long_arr - proba_short_arr
    proba_flat_arr = np.clip(proba_flat_arr, 0.0, 1.0)  # Ensure valid probability range
    
    # Determine direction_class using argmax
    # Stack probabilities: [p_flat, p_long, p_short] and find argmax
    proba_stack = np.stack([proba_flat_arr, proba_long_arr, proba_short_arr], axis=1)  # (N, 3)
    direction_classes = np.argmax(proba_stack, axis=1)  # (N,) with values {0: FLAT, 1: LONG, 2: SHORT}
    
    # Apply strategy mode logic with 3-class information
    if mode == StrategyMode.LONG_ONLY:
        # Only check long condition: direction_class == LONG AND p_long >= threshold
        long_mask = (
            (direction_classes == LstmClassIndex.LONG) &
            (proba_long_arr >= long_threshold if long_threshold is not None else np.ones(len(df), dtype=bool))
        )
        signals[long_mask] = "LONG"
        
    elif mode == StrategyMode.SHORT_ONLY:
        # Only check short condition: direction_class == SHORT AND p_short >= threshold
        short_mask = (
            (direction_classes == LstmClassIndex.SHORT) &
            (proba_short_arr >= short_threshold if short_threshold is not None else np.ones(len(df), dtype=bool))
        )
        signals[short_mask] = "SHORT"
        
    else:  # StrategyMode.BOTH
        # Check both conditions with direction_class constraint
        long_mask = (
            (direction_classes == LstmClassIndex.LONG) &
            (proba_long_arr >= long_threshold if long_threshold is not None else np.ones(len(df), dtype=bool))
        )
        short_mask = (
            (direction_classes == LstmClassIndex.SHORT) &
            (proba_short_arr >= short_threshold if short_threshold is not None else np.ones(len(df), dtype=bool))
        )
        
        # Conflict resolution: if both are true (shouldn't happen with direction_class, but defensive)
        conflict_mask = long_mask & short_mask
        if np.any(conflict_mask):
            # Calculate margins
            margin_long = proba_long_arr[conflict_mask] - long_threshold if long_threshold is not None else np.zeros(np.sum(conflict_mask))
            margin_short = proba_short_arr[conflict_mask] - short_threshold if short_threshold is not None else np.zeros(np.sum(conflict_mask))
            
            # Prefer the one with larger margin (or LONG if equal)
            prefer_short = margin_short > margin_long
            long_mask[conflict_mask] = ~prefer_short
            short_mask[conflict_mask] = prefer_short
        
        # Apply signals
        signals[long_mask] = "LONG"
        signals[short_mask] = "SHORT"
    
    # Log direction_class distribution for debugging
    flat_count = int(np.sum(direction_classes == LstmClassIndex.FLAT))
    long_count = int(np.sum(direction_classes == LstmClassIndex.LONG))
    short_count = int(np.sum(direction_classes == LstmClassIndex.SHORT))
    
    logger.debug(
        f"[ml_lstm_attn_enhanced] mode={mode.value}, "
        f"long_threshold={long_threshold}, short_threshold={short_threshold}, "
        f"direction_class: FLAT={flat_count}, LONG={long_count}, SHORT={short_count}, "
        f"signals: LONG={np.sum(signals == 'LONG')}, "
        f"SHORT={np.sum(signals == 'SHORT')}, HOLD={np.sum(signals == 'HOLD')}"
    )
    
    return signals, df


