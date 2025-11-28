"""
ML prediction probability caching for threshold optimization.

This module provides functions to compute and cache prediction probabilities
for ML strategies, allowing threshold optimization to reuse predictions
across multiple threshold combinations without recomputing.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import _get_ml_adapter
from src.core.config import settings
from src.indicators.basic import get_df_with_indicators

logger = logging.getLogger(__name__)


def compute_ml_proba_cache(
    strategy_name: str,
    df: pd.DataFrame | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute prediction probabilities (proba_up) for entire dataset.
    
    This function computes predictions once and caches them, allowing
    threshold optimization to reuse predictions across multiple threshold
    combinations without recomputing.
    
    Args:
        strategy_name: Strategy identifier (e.g., "ml_xgb", "ml_lstm_attn")
        df: Optional DataFrame with OHLCV + indicators. If None, loads from service.
        symbol: Trading symbol (default: from settings)
        timeframe: Timeframe (default: from settings)
    
    Returns:
        Tuple of (proba_up: np.ndarray of shape (N,), df: pd.DataFrame)
        where proba_up[i] corresponds to df.iloc[i]
        Note: proba_up may have fewer elements than df if some predictions fail.
    """
    if df is None:
        df = get_df_with_indicators().copy()
    else:
        df = df.copy()
    
    # Extract symbol and timeframe
    if symbol is None:
        symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    if timeframe is None:
        timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    
    adapter = _get_ml_adapter(strategy_name)
    model = adapter.get_model()
    
    if model is None:
        raise ValueError(f"{adapter.name} model instance is None. Cannot compute predictions.")
    
    if not getattr(model, "is_loaded", lambda: False)():
        model_path = getattr(model, "model_path", None)
        exists = model_path.exists() if model_path is not None else False
        raise ValueError(
            f"{adapter.name} model not loaded. Model path: {model_path}, exists={exists}"
        )
    
    min_rows = adapter.min_history_provider(model)
    proba_values: list[float] = []
    valid_indices: list[int] = []
    prediction_errors = 0
    
    logger.info(
        f"[Proba Cache][{adapter.name}] Computing predictions: "
        f"total_rows={len(df)}, min_history={min_rows}"
    )
    
    for i in range(min_rows, len(df)):
        df_slice = df.iloc[: i + 1]
        try:
            proba_up = float(model.predict_proba_latest(df_slice, symbol=symbol, timeframe=timeframe))
            proba_values.append(proba_up)
            valid_indices.append(i)
        except Exception as exc:
            prediction_errors += 1
            if prediction_errors <= 5:
                logger.warning(
                    f"[Proba Cache][{adapter.name}] Prediction failed at index {i}: "
                    f"{type(exc).__name__}: {exc}"
                )
            continue
    
    if not proba_values:
        raise ValueError(
            f"[Proba Cache][{adapter.name}] No successful predictions! "
            f"All {len(df) - min_rows} predictions failed."
        )
    
    proba_arr = np.array(proba_values, dtype=np.float32)
    
    logger.info(
        f"[Proba Cache][{adapter.name}] Completed: "
        f"successful={len(proba_values)}, errors={prediction_errors}, "
        f"mean_proba={proba_arr.mean():.4f}, std={proba_arr.std():.4f}"
    )
    
    return proba_arr, df.iloc[valid_indices].reset_index(drop=True)

