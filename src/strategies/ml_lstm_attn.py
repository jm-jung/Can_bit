"""
LSTM + Attention ML strategy aligned with the XGB interface.
"""
from __future__ import annotations

import logging
from typing import Literal, TypedDict

from src.core.config import settings
from src.dl.lstm_attn_model import get_lstm_attn_model
from src.indicators.basic import get_df_with_indicators
from src.strategies.ml_thresholds import resolve_ml_thresholds

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


