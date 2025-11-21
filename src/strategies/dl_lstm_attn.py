"""
LSTM + Attention-based deep learning trading strategy.
"""
from __future__ import annotations

from typing import Literal, TypedDict

import pandas as pd

from src.core.config import settings
from src.indicators.basic import get_df_with_indicators
from src.dl.lstm_attn_model import get_lstm_attn_model

Signal = Literal["LONG", "SHORT", "HOLD"]


class DLStrategyOutput(TypedDict):
    """Output structure for DL strategy."""

    timestamp: str
    close: float
    proba_up: float | None
    signal: Signal


def get_lstm_attn_signal(
    ohlcv_df: pd.DataFrame | None = None,
    threshold_up: float | None = None,
    threshold_down: float | None = None,
) -> DLStrategyOutput:
    """
    LSTM + Attention-based deep learning trading strategy.

    Rules:
    - proba_up >= threshold_up → LONG
    - proba_up <= threshold_down → SHORT
    - threshold_down < proba_up < threshold_up → HOLD
    - Model or data failure → proba_up=None, signal="HOLD"

    Args:
        ohlcv_df: Optional DataFrame with OHLCV data. If None, loads from service.
        threshold_up: Probability threshold for LONG signal (default: from settings)
        threshold_down: Probability threshold for SHORT signal (default: from settings)

    Returns:
        DLStrategyOutput with prediction and signal
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        if ohlcv_df is None:
            df = get_df_with_indicators()
        else:
            df = ohlcv_df.copy()
            from src.indicators.basic import add_basic_indicators

            df = add_basic_indicators(df)

        model = get_lstm_attn_model()

        if model is None or not model.is_loaded():
            last_row = df.iloc[-1]
            return DLStrategyOutput(
                timestamp=str(last_row["timestamp"]),
                close=float(last_row["close"]),
                proba_up=None,
                signal="HOLD",
            )

        # Use config defaults if not provided
        if threshold_up is None:
            threshold_up = settings.LSTM_ATTN_THRESHOLD_UP
        if threshold_down is None:
            threshold_down = settings.LSTM_ATTN_THRESHOLD_DOWN

        proba_up = model.predict_proba_latest(df)
        last_row = df.iloc[-1]

        # Use predict_label_latest for consistent signal generation
        signal: Signal = model.predict_label_latest(
            df,
            threshold_up=threshold_up,
            threshold_down=threshold_down,
        )

        # 디버그 로깅: threshold와 signal 확인
        logger.info(
            f"[LSTM Strategy] prob_up={proba_up:.4f}, "
            f"threshold_up={threshold_up:.2f}, threshold_down={threshold_down:.2f}, "
            f"signal={signal}"
        )

        return DLStrategyOutput(
            timestamp=str(last_row["timestamp"]),
            close=float(last_row["close"]),
            proba_up=proba_up,
            signal=signal,
        )

    except Exception as e:
        # On any error, return safe HOLD signal
        try:
            if ohlcv_df is None:
                df = get_df_with_indicators()
            else:
                df = ohlcv_df.copy()
                from src.indicators.basic import add_basic_indicators

                df = add_basic_indicators(df)
            last_row = df.iloc[-1]
            return DLStrategyOutput(
                timestamp=str(last_row["timestamp"]),
                close=float(last_row["close"]),
                proba_up=None,
                signal="HOLD",
            )
        except Exception:
            # Fallback if even getting data fails
            return DLStrategyOutput(
                timestamp="",
                close=0.0,
                proba_up=None,
                signal="HOLD",
            )

