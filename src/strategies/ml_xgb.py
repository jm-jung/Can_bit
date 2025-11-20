"""
XGBoost-based ML trading strategy.
"""
from __future__ import annotations

from typing import Literal, TypedDict

import pandas as pd

from src.indicators.basic import get_df_with_indicators
from src.ml.xgb_model import get_xgb_model

Signal = Literal["LONG", "SHORT", "HOLD"]


class MLStrategyOutput(TypedDict):
    """Output structure for ML strategy."""

    timestamp: str
    close: float
    proba_up: float | None
    signal: Signal


def ml_xgb_strategy() -> MLStrategyOutput:
    """
    XGBoost-based ML trading strategy.

    Rules:
    - proba_up >= 0.55 → LONG
    - proba_up <= 0.45 → SHORT
    - 0.45 < proba_up < 0.55 → HOLD
    - Model or data failure → proba_up=None, signal="HOLD"

    Returns:
        MLStrategyOutput with prediction and signal
    """
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

        proba_up = model.predict_proba_latest(df)
        last_row = df.iloc[-1]

        # Determine signal based on probability
        if proba_up >= 0.55:
            signal: Signal = "LONG"
        elif proba_up <= 0.45:
            signal = "SHORT"
        else:
            signal = "HOLD"

        return MLStrategyOutput(
            timestamp=str(last_row["timestamp"]),
            close=float(last_row["close"]),
            proba_up=proba_up,
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

