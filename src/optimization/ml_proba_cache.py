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
from src.indicators.basic import get_df_with_indicators, add_basic_indicators
from src.ml.xgb_model import MLXGBModel
from src.services.ohlcv_service import load_ohlcv_df
from src.ml.features import build_feature_frame
from src.features.ml_feature_config import MLFeatureConfig

logger = logging.getLogger(__name__)


def compute_ml_proba_cache(
    strategy_name: str,
    df: pd.DataFrame | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    feature_preset: str = "extended_safe",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Compute prediction probabilities (proba_long, proba_short) for entire dataset.
    
    This function computes predictions once and caches them, allowing
    threshold optimization to reuse predictions across multiple threshold
    combinations without recomputing.
    
    Args:
        strategy_name: Strategy identifier (e.g., "ml_xgb", "ml_lstm_attn")
        df: Optional DataFrame with OHLCV + indicators. If None, loads from service.
        symbol: Trading symbol (default: from settings)
        timeframe: Timeframe (default: from settings)
        feature_preset: Feature preset for ml_xgb strategy (default: "extended_safe")
    
    Returns:
        Tuple of (proba_long: np.ndarray, proba_short: np.ndarray, df: pd.DataFrame)
        where proba_long[i] and proba_short[i] correspond to df.iloc[i]
        Note: proba arrays may have fewer elements than df if some predictions fail.
    """
    # Extract symbol and timeframe
    if symbol is None:
        symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    if timeframe is None:
        timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    
    # Phase E: Load OHLCV data with timeframe support
    if df is None:
        # [ML] Pass symbol to use pre-resampled long-run CSV for 5m
        df_raw = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
        df = add_basic_indicators(df_raw.copy())
        logger.info(
            f"[Proba Cache] Loaded OHLCV data: symbol={symbol}, timeframe={timeframe}, rows={len(df)}"
        )
    else:
        df = df.copy()
    
    adapter = _get_ml_adapter(strategy_name)
    
    # Phase E: Use MLXGBModel for ml_xgb strategy
    if strategy_name == "ml_xgb":
        try:
            model = MLXGBModel(
                strategy=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                feature_preset=feature_preset,
            )
            logger.info(
                f"[Proba Cache][{adapter.name}] Using MLXGBModel: "
                f"long={model.long_model_path.name}, short={model.short_model_path.name}, "
                f"scaler={model.scaler_path.name if model.scaler_path and model.scaler_path.exists() else 'None'}, "
                f"label_mode={model.label_mode}, feature_preset={feature_preset}"
            )
            logger.info(
                f"[Proba Cache][{adapter.name}] Model metadata: "
                f"timeframe={model.meta_data.get('timeframe', 'unknown')}, "
                f"label_thresholds={model.label_thresholds}, "
                f"scaler_type={model.scaler_type}"
            )
            has_separate_models = True
            min_rows = 20  # Default min history for XGBoost
        except Exception as e:
            logger.warning(
                f"[Proba Cache][{adapter.name}] Failed to load MLXGBModel ({e}). "
                "Falling back to legacy model loader."
            )
            model = adapter.get_model()
            if model is None:
                raise ValueError(f"{adapter.name} model instance is None. Cannot compute predictions.")
            if not getattr(model, "is_loaded", lambda: False)():
                model_path = getattr(model, "model_path", None)
                exists = model_path.exists() if model_path is not None else False
                raise ValueError(
                    f"{adapter.name} model not loaded. Model path: {model_path}, exists={exists}"
                )
            has_separate_models = getattr(model, "has_separate_models", lambda: False)()
            min_rows = adapter.min_history_provider(model)
    else:
        model = adapter.get_model()
        if model is None:
            raise ValueError(f"{adapter.name} model instance is None. Cannot compute predictions.")
        if not getattr(model, "is_loaded", lambda: False)():
            model_path = getattr(model, "model_path", None)
            exists = model_path.exists() if model_path is not None else False
            raise ValueError(
                f"{adapter.name} model not loaded. Model path: {model_path}, exists={exists}"
            )
        has_separate_models = getattr(model, "has_separate_models", lambda: False)()
        min_rows = adapter.min_history_provider(model)
    
    proba_long_values: list[float] = []
    proba_short_values: list[float] = []
    valid_indices: list[int] = []
    prediction_errors = 0
    
    logger.info(
        f"[Proba Cache][{adapter.name}] Computing predictions: "
        f"total_rows={len(df)}, min_history={min_rows}, has_separate_models={has_separate_models}, "
        f"timeframe={timeframe}, feature_preset={feature_preset if strategy_name == 'ml_xgb' else 'N/A'}"
    )
    
    # Phase E: Use MLXGBModel's batch prediction if available
    if isinstance(model, MLXGBModel):
        # Extract features once for entire dataset
        feature_config = MLFeatureConfig.from_preset(feature_preset)
        full_features = build_feature_frame(
            df,
            symbol=symbol,
            timeframe=timeframe,
            use_events=settings.EVENTS_ENABLED,
            feature_config=feature_config,
        )
        
        for i in range(min_rows, len(df)):
            # Use features up to index i (sliding window)
            features_slice = full_features.iloc[: i + 1]
            last_features = features_slice.iloc[[-1]].copy()
            
            try:
                # MLXGBModel handles scaling internally
                proba_long = model.predict_proba_long(last_features)
                proba_short = model.predict_proba_short(last_features)
                proba_long_values.append(float(proba_long[0]) if isinstance(proba_long, np.ndarray) else float(proba_long))
                proba_short_values.append(float(proba_short[0]) if isinstance(proba_short, np.ndarray) else float(proba_short))
                valid_indices.append(i)
            except Exception as exc:
                prediction_errors += 1
                if prediction_errors <= 5:
                    logger.warning(
                        f"[Proba Cache][{adapter.name}] Prediction failed at index {i}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                continue
    else:
        # Legacy model: use predict_proba_latest
        for i in range(min_rows, len(df)):
            df_slice = df.iloc[: i + 1]
            try:
                if has_separate_models:
                    # Get both LONG and SHORT proba
                    proba_long, proba_short = model.predict_proba_latest(
                        df_slice, 
                        symbol=symbol, 
                        timeframe=timeframe,
                        return_both=True
                    )
                    proba_long_values.append(float(proba_long))
                    proba_short_values.append(float(proba_short))
                else:
                    # Single model (backward compatibility)
                    proba_up = float(model.predict_proba_latest(df_slice, symbol=symbol, timeframe=timeframe))
                    proba_long_values.append(proba_up)
                    proba_short_values.append(1.0 - proba_up)  # Approximate SHORT proba
                valid_indices.append(i)
            except Exception as exc:
                prediction_errors += 1
                if prediction_errors <= 5:
                    logger.warning(
                        f"[Proba Cache][{adapter.name}] Prediction failed at index {i}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                continue
    
    if not proba_long_values:
        raise ValueError(
            f"[Proba Cache][{adapter.name}] No successful predictions! "
            f"All {len(df) - min_rows} predictions failed."
        )
    
    proba_long_arr = np.array(proba_long_values, dtype=np.float32)
    proba_short_arr = np.array(proba_short_values, dtype=np.float32)
    
    logger.info(
        f"[Proba Cache][{adapter.name}] Completed: "
        f"successful={len(proba_long_values)}, errors={prediction_errors}, "
        f"mean_proba_long={proba_long_arr.mean():.4f}, std={proba_long_arr.std():.4f}, "
        f"mean_proba_short={proba_short_arr.mean():.4f}, std={proba_short_arr.std():.4f}"
    )
    
    return proba_long_arr, proba_short_arr, df.iloc[valid_indices].reset_index(drop=True)

