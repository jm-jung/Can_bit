"""
XGBoost-based ML trading strategy.
"""
from __future__ import annotations

import logging
from typing import Literal, TypedDict

import pandas as pd

from src.core.config import settings
from src.indicators.basic import get_df_with_indicators
from src.ml.xgb_model import get_xgb_model
from src.strategies.ml_thresholds import resolve_ml_thresholds

Signal = Literal["LONG", "SHORT", "HOLD"]

logger = logging.getLogger(__name__)


class MLStrategyOutput(TypedDict):
    """Output structure for ML strategy."""

    timestamp: str
    close: float
    proba_up: float | None
    signal: Signal


def ml_xgb_strategy(
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    use_optimized_thresholds: bool = True,
    *,
    strategy_name: str = "ml_xgb",
    symbol: str | None = None,
    timeframe: str | None = None,
) -> MLStrategyOutput:
    """
    XGBoost-based ML trading strategy.

    Args:
        long_threshold: Probability threshold for LONG signal (default: 0.5)
        short_threshold: Probability threshold for SHORT signal (default: None, disabled)
        use_optimized_thresholds: If True, try to load optimized thresholds from JSON file
        strategy_name: Strategy identifier for threshold lookup
        symbol: Optional override for symbol when resolving optimized thresholds
        timeframe: Optional override for timeframe when resolving optimized thresholds

    Rules:
    - proba_up >= long_threshold → LONG
    - proba_up <= short_threshold → SHORT (if short_threshold is not None)
    - Otherwise → HOLD
    - Model or data failure → proba_up=None, signal="HOLD"

    Returns:
        MLStrategyOutput with prediction and signal
    """
    long_threshold, short_threshold = resolve_ml_thresholds(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_thresholds=use_optimized_thresholds,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        default_long=0.5,
        default_short=None,
    )
    
    try:
        df = get_df_with_indicators()
        model = get_xgb_model()

        if model is None or not model.is_loaded():
            last_row = df.iloc[-1]
            return MLStrategyOutput(
                timestamp=str(last_row["timestamp"]),
                close=float(last_row["close"]),
                proba_up=None,
                signal="HOLD",
            )

        # Check if model supports separate LONG/SHORT predictions
        has_separate_models = getattr(model, "has_separate_models", lambda: False)()
        
        if has_separate_models:
            # Use separate LONG/SHORT models
            proba_long, proba_short = model.predict_proba_latest(
                df, 
                symbol=symbol, 
                timeframe=timeframe,
                return_both=True
            )
            proba_up = proba_long  # For backward compatibility in output
        else:
            # Single model (backward compatibility)
            proba_up = model.predict_proba_latest(df, symbol=symbol, timeframe=timeframe)
            proba_long = proba_up
            proba_short = 1.0 - proba_up  # Approximate
        
        last_row = df.iloc[-1]

        # Determine signal based on probability thresholds (using separate LONG/SHORT proba)
        is_long = proba_long >= long_threshold if long_threshold is not None else False
        is_short = proba_short >= short_threshold if short_threshold is not None else False
        
        # Conflict resolution: if both are true, choose the one with larger margin
        if is_long and is_short:
            margin_long = proba_long - long_threshold if long_threshold is not None else 0.0
            margin_short = proba_short - short_threshold if short_threshold is not None else 0.0
            if margin_long >= margin_short:
                is_short = False  # Prefer LONG
            else:
                is_long = False  # Prefer SHORT
        
        if is_long:
            signal: Signal = "LONG"
        elif is_short:
            signal = "SHORT"
        else:
            signal = "HOLD"

        return MLStrategyOutput(
            timestamp=str(last_row["timestamp"]),
            close=float(last_row["close"]),
            proba_up=proba_up,  # For backward compatibility
            signal=signal,
        )

    except Exception as e:
        # On any error, return safe HOLD signal
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
            # Fallback if even getting data fails
            return MLStrategyOutput(
                timestamp="",
                close=0.0,
                proba_up=None,
                signal="HOLD",
            )

