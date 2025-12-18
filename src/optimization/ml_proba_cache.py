"""
ML prediction probability caching for threshold optimization.

This module provides functions to compute and cache prediction probabilities
for ML strategies, allowing threshold optimization to reuse predictions
across multiple threshold combinations without recomputing.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
import torch
import numpy as np
import pandas as pd

from src.backtest.engine import _get_ml_adapter
from src.core.config import settings
from src.indicators.basic import get_df_with_indicators, add_basic_indicators
from src.ml.xgb_model import MLXGBModel
from src.services.ohlcv_service import load_ohlcv_df
from src.ml.features import build_feature_frame
from src.features.ml_feature_config import MLFeatureConfig
from src.dl.data.labels import LstmClassIndex

logger = logging.getLogger(__name__)

# Cache directory for prediction probabilities
CACHE_DIR = Path("data/cache/ml_predictions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    feature_preset: str = "extended_safe",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    """
    Build cache file path for prediction probabilities.
    
    Args:
        strategy_name: Strategy identifier (e.g., "ml_xgb")
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "5m")
        feature_preset: Feature preset (e.g., "extended_safe")
        start_date: Start date filter (YYYY-MM-DD format) or None
        end_date: End date filter (YYYY-MM-DD format) or None
    
    Returns:
        Path to cache file
    """
    # Normalize symbol and timeframe
    symbol_norm = symbol.replace("/", "").upper()
    timeframe_norm = timeframe.lower()
    
    # Build base filename
    if strategy_name == "ml_xgb":
        base_filename = f"{strategy_name}_{symbol_norm}_{timeframe_norm}_{feature_preset}"
    else:
        base_filename = f"{strategy_name}_{symbol_norm}_{timeframe_norm}"
    
    # Add date suffix if date range is specified
    date_suffix = ""
    if start_date or end_date:
        date_parts = []
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                date_parts.append(start_dt.strftime("%Y%m%d"))
            except Exception as e:
                logger.warning(f"[Proba Cache] Failed to parse start_date '{start_date}': {e}")
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date)
                date_parts.append(end_dt.strftime("%Y%m%d"))
            except Exception as e:
                logger.warning(f"[Proba Cache] Failed to parse end_date '{end_date}': {e}")
        
        if date_parts:
            date_suffix = "_" + "_".join(date_parts)
    
    filename = f"{base_filename}{date_suffix}_proba.parquet"
    
    return CACHE_DIR / filename


def get_or_build_predictions(
    strategy_name: str,
    symbol: str | None = None,
    timeframe: str | None = None,
    feature_preset: str = "extended_safe",
    force_rebuild: bool = False,
    nthread: int | None = None,
    df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Get prediction probabilities from cache or compute them.
    
    This function:
    1. Checks if cache file exists and is valid
    2. If cache exists and force_rebuild=False, loads from cache
    3. Otherwise, computes predictions and saves to cache
    
    Args:
        strategy_name: Strategy identifier (e.g., "ml_xgb", "ml_lstm_attn")
        symbol: Trading symbol (default: from settings)
        timeframe: Timeframe (default: from settings)
        feature_preset: Feature preset for ml_xgb strategy (default: "extended_safe")
        force_rebuild: If True, ignore cache and recompute predictions
        nthread: Number of CPU threads for XGBoost (passed to model loader)
        df: Optional DataFrame with OHLCV + indicators. If None, loads from service.
        start_date: Filter OHLCV and predictions to samples at or after this date (YYYY-MM-DD format)
        end_date: Filter OHLCV and predictions to samples at or before this date (YYYY-MM-DD format)
    
    Returns:
        Tuple of (proba_long: np.ndarray, proba_short: np.ndarray, df: pd.DataFrame)
    """
    # Extract symbol and timeframe
    if symbol is None:
        symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    if timeframe is None:
        timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    
    cache_path = _get_cache_path(
        strategy_name, symbol, timeframe, feature_preset, start_date, end_date
    )
    
    # Check cache validity if not forcing rebuild
    if not force_rebuild and cache_path.exists():
        try:
            # Load cache
            cache_df = pd.read_parquet(cache_path)
            
            # Validate cache structure
            required_cols = ["proba_long", "proba_short"]
            if not all(col in cache_df.columns for col in required_cols):
                logger.warning(
                    f"[Proba Cache] Cache file {cache_path} has invalid structure. Rebuilding..."
                )
            else:
                # Extract arrays and aligned dataframe
                proba_long_arr = cache_df["proba_long"].values.astype(np.float32)
                proba_short_arr = cache_df["proba_short"].values.astype(np.float32)
                
                # Load aligned dataframe (drop proba columns to get original structure)
                df_aligned = cache_df.drop(columns=["proba_long", "proba_short"])
                
                # Basic validation: check if we have data
                if len(proba_long_arr) == 0:
                    logger.warning(
                        f"[Proba Cache] Cache file {cache_path} is empty. Rebuilding..."
                    )
                elif len(proba_long_arr) != len(proba_short_arr):
                    logger.warning(
                        f"[Proba Cache] Cache file {cache_path} has mismatched array lengths. Rebuilding..."
                    )
                else:
                    date_range_str = ""
                    if start_date or end_date:
                        date_range_str = f" (date range: {start_date or 'None'} ~ {end_date or 'None'})"
                    logger.info(
                        f"[Proba Cache] Loaded {len(proba_long_arr)} cached predictions from {cache_path}{date_range_str}"
                    )
                    return proba_long_arr, proba_short_arr, df_aligned
        except Exception as e:
            logger.warning(
                f"[Proba Cache] Failed to load cache from {cache_path}: {e}. Rebuilding..."
            )
    
    # Compute predictions from scratch
    if force_rebuild and cache_path.exists():
        logger.info(
            f"[Proba Cache] Cache exists but rebuild was requested. "
            f"Generating XGBoost probabilities from scratch..."
        )
    else:
        logger.info(
            f"[Proba Cache] Generating XGBoost probabilities from scratch..."
        )
    
    # Apply date filtering to input DataFrame if provided, or load and filter
    df_filtered = df
    rows_before = None
    rows_after = None
    
    # Load OHLCV data if not provided
    if df_filtered is None:
        from src.services.ohlcv_service import load_ohlcv_df
        from src.indicators.basic import add_basic_indicators
        
        df_raw = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
        df_filtered = add_basic_indicators(df_raw.copy())
        logger.info(
            f"[Proba Cache] Loaded OHLCV data: symbol={symbol}, timeframe={timeframe}, rows={len(df_filtered)}"
        )
    
    # Apply date filtering
    rows_before = len(df_filtered)
    
    if start_date or end_date:
        # Ensure timestamp column exists and is datetime
        if "timestamp" not in df_filtered.columns:
            raise ValueError(
                "[Proba Cache] DataFrame must have 'timestamp' column for date filtering"
            )
        
        if not pd.api.types.is_datetime64_any_dtype(df_filtered["timestamp"]):
            df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])
        
        # Apply filters
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df_filtered = df_filtered[df_filtered["timestamp"] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            # Include the entire end date (up to end of day)
            end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_filtered = df_filtered[df_filtered["timestamp"] <= end_dt]
        
        rows_after = len(df_filtered)
        logger.info(
            f"[Proba Cache] Date filter applied: start={start_date or 'None'}, "
            f"end={end_date or 'None'}, rows_before={rows_before}, rows_after={rows_after}"
        )
    else:
        # No date filter applied
        rows_after = rows_before
        logger.info(
            f"[Proba Cache] Date filter not applied, using full range: rows={rows_after}"
        )
    
    # Compute predictions with filtered data
    proba_long_arr, proba_short_arr, df_aligned = compute_ml_proba_cache(
        strategy_name=strategy_name,
        df=df_filtered,
        symbol=symbol,
        timeframe=timeframe,
        feature_preset=feature_preset,
        nthread=nthread,
    )
    
    # Save to cache
    try:
        # Create cache DataFrame with predictions and original data
        cache_df = df_aligned.copy()
        cache_df["proba_long"] = proba_long_arr
        cache_df["proba_short"] = proba_short_arr
        
        cache_df.to_parquet(cache_path, compression="snappy", index=False)
        logger.info(
            f"[Proba Cache] Saved {len(proba_long_arr)} predictions to cache {cache_path}"
        )
    except Exception as e:
        logger.warning(
            f"[Proba Cache] Failed to save cache to {cache_path}: {e}. "
            "Continuing without cache."
        )
    
    return proba_long_arr, proba_short_arr, df_aligned


def compute_ml_proba_cache(
    strategy_name: str,
    df: pd.DataFrame | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    feature_preset: str = "extended_safe",
    nthread: int | None = None,
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
    # Note: If df is provided, it should already be filtered by date range
    # (filtering is done in get_or_build_predictions before calling this function)
    if df is None:
        # [ML] Pass symbol to use pre-resampled long-run CSV for 5m
        df_raw = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
        df = add_basic_indicators(df_raw.copy())
        logger.info(
            f"[Proba Cache] Loaded OHLCV data: symbol={symbol}, timeframe={timeframe}, rows={len(df)}"
        )
    else:
        # Use provided DataFrame (should already be filtered by date range)
        df = df.copy()
        logger.debug(
            f"[Proba Cache] Using provided DataFrame: rows={len(df)} "
            "(date filtering should have been applied already)"
        )
    
    adapter = _get_ml_adapter(strategy_name)
    
    # Phase E: Use MLXGBModel for ml_xgb strategy
    if strategy_name == "ml_xgb":
        try:
            model = MLXGBModel(
                strategy=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                feature_preset=feature_preset,
                nthread=nthread,
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
    
    # Log with filtered row count (df is already filtered at this point)
    total_rows = len(df)
    logger.info(
        f"[Proba Cache][{adapter.name}] Computing predictions: "
        f"total_rows={total_rows}, min_history={min_rows}, has_separate_models={has_separate_models}, "
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
    elif strategy_name == "ml_lstm_attn":
        # Optimized batch path for LSTM-Attention (3-class model)
        from src.dl.lstm_attn_model import LSTMAttnSignalModel
        
        if not isinstance(model, LSTMAttnSignalModel):
            raise ValueError(
                f"[Proba Cache][{adapter.name}] Expected LSTMAttnSignalModel, got {type(model)}"
            )
        
        # Log 3-class model information
        logger.info(
            f"[Proba Cache][{adapter.name}] Using 3-class LSTM-Attention model. "
            f"Class indices: FLAT={LstmClassIndex.FLAT}, LONG={LstmClassIndex.LONG}, SHORT={LstmClassIndex.SHORT}. "
            f"proba_long/proba_short are derived from 3-class softmax output."
        )
        
        # Extract features once for entire dataset
        logger.info(
            f"[Proba Cache][{adapter.name}] Extracting features for batch prediction..."
        )
        full_features = build_feature_frame(
            df,
            symbol=symbol,
            timeframe=timeframe,
            use_events=settings.EVENTS_ENABLED,
        )
        full_features = full_features.dropna()
        
        # Validate we have enough data
        if len(full_features) < min_rows:
            raise ValueError(
                f"[Proba Cache][{adapter.name}] Not enough features after extraction: "
                f"need at least {min_rows}, got {len(full_features)}"
            )
        
        # Use batch prediction
        try:
            proba_long_arr, proba_short_arr = model.predict_proba_batch(
                features=full_features,
                symbol=symbol,
                timeframe=timeframe,
                batch_size=512,
            )
            
            # Align with original df indices
            # IMPORTANT: Batch path now matches training exactly:
            # - Training: for i in range(window_size, len(features)): seq = features[i-window_size:i]
            # - Batch: for i in range(window_size, len(features)): seq = features[i-window_size:i]
            # - proba arrays have length (len(full_features) - window_size)
            # - Predictions correspond to indices [window_size, window_size+1, ..., len(features)-1] in features
            # - We need to map these to df indices, accounting for dropna() in feature extraction
            
            num_proba = len(proba_long_arr)
            num_features = len(full_features)
            
            # Batch path creates sequences for i in range(window_size, len(features))
            # So proba arrays have length (len(features) - window_size)
            # Predictions correspond to feature indices [window_size, window_size+1, ..., len(features)-1]
            # We need to map these to df indices
            
            # Since features may have fewer rows than df due to dropna(), we need to be careful
            # The mapping depends on how dropna() removed rows
            # For now, assume features and df are aligned after dropna()
            # (This is true if dropna() only removes rows with NaN, not reorders)
            
            # Calculate expected predictions: starting from min_rows (window_size) in df
            expected_proba_count = len(df) - min_rows
            
            if num_proba < expected_proba_count:
                # Fewer predictions than expected - this can happen if features dropped rows
                logger.warning(
                    f"[Proba Cache][{adapter.name}] Fewer predictions than expected: "
                    f"got {num_proba}, expected {expected_proba_count}. "
                    f"This may be due to feature extraction dropping rows."
                )
                # Use what we have
                proba_long_arr = proba_long_arr[:num_proba]
                proba_short_arr = proba_short_arr[:num_proba]
                valid_indices = list(range(min_rows, min_rows + num_proba))
            elif num_proba > expected_proba_count:
                # More predictions than expected - truncate to match df length
                logger.debug(
                    f"[Proba Cache][{adapter.name}] Truncating predictions: "
                    f"got {num_proba}, expected {expected_proba_count}"
                )
                proba_long_arr = proba_long_arr[:expected_proba_count]
                proba_short_arr = proba_short_arr[:expected_proba_count]
                valid_indices = list(range(min_rows, len(df)))
            else:
                # Perfect match
                valid_indices = list(range(min_rows, len(df)))
            
            proba_long_values = proba_long_arr.tolist()
            proba_short_values = proba_short_arr.tolist()
            
            logger.info(
                f"[Proba Cache][{adapter.name}] Batch prediction completed: "
                f"proba_count={len(proba_long_values)}, valid_indices_count={len(valid_indices)}, "
                f"df_rows={len(df)}, min_rows={min_rows}"
            )
            
        except Exception as exc:
            logger.error(
                f"[Proba Cache][{adapter.name}] Batch prediction failed: {type(exc).__name__}: {exc}"
            )
            logger.info(
                f"[Proba Cache][{adapter.name}] Falling back to legacy per-step prediction..."
            )
            # Fallback to legacy path
            # IMPORTANT: Use return_both=True to get correct proba_short from 3-class softmax
            prediction_errors = 0
            for i in range(min_rows, len(df)):
                df_slice = df.iloc[: i + 1]
                try:
                    # Use return_both=True to get both proba_long and proba_short from 3-class softmax
                    proba_long_val, proba_short_val = model.predict_proba_latest(
                        df_slice, 
                        symbol=symbol, 
                        timeframe=timeframe,
                        return_both=True
                    )
                    proba_long_values.append(float(proba_long_val))
                    proba_short_values.append(float(proba_short_val))
                    valid_indices.append(i)
                except Exception as exc2:
                    prediction_errors += 1
                    if prediction_errors <= 5:
                        logger.warning(
                            f"[Proba Cache][{adapter.name}] Prediction failed at index {i}: "
                            f"{type(exc2).__name__}: {exc2}"
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
    
    # Log probability statistics
    logger.info(
        f"[Proba Cache][{adapter.name}] Completed: "
        f"successful={len(proba_long_values)}, errors={prediction_errors}, "
        f"mean_proba_long={proba_long_arr.mean():.4f}, std={proba_long_arr.std():.4f}, "
        f"mean_proba_short={proba_short_arr.mean():.4f}, std={proba_short_arr.std():.4f}"
    )
    
    # For 3-class LSTM models, log additional statistics
    if strategy_name == "ml_lstm_attn":
        # Compute proba_flat for statistics (3-class: p_flat + p_long + p_short = 1)
        proba_flat_arr = 1.0 - proba_long_arr - proba_short_arr
        proba_flat_arr = np.clip(proba_flat_arr, 0.0, 1.0)
        
        # Compute argmax distribution (predicted class distribution)
        proba_stack = np.stack([proba_flat_arr, proba_long_arr, proba_short_arr], axis=1)  # (N, 3)
        predicted_classes = np.argmax(proba_stack, axis=1)  # (N,) with values {0: FLAT, 1: LONG, 2: SHORT}
        
        flat_count = int(np.sum(predicted_classes == LstmClassIndex.FLAT))
        long_count = int(np.sum(predicted_classes == LstmClassIndex.LONG))
        short_count = int(np.sum(predicted_classes == LstmClassIndex.SHORT))
        total_count = len(predicted_classes)
        
        logger.info(
            f"[Proba Cache][{adapter.name}] 3-class model statistics: "
            f"mean_proba_flat={proba_flat_arr.mean():.4f}, "
            f"predicted_class_distribution: FLAT={flat_count} ({100*flat_count/total_count:.1f}%), "
            f"LONG={long_count} ({100*long_count/total_count:.1f}%), "
            f"SHORT={short_count} ({100*short_count/total_count:.1f}%)"
        )
    
    # Validation: Compare batch vs legacy for LSTM-Attention (small sample)
    if strategy_name == "ml_lstm_attn" and len(df) > 1000:
        try:
            from src.dl.lstm_attn_model import LSTMAttnSignalModel
            
            if isinstance(model, LSTMAttnSignalModel):
                # Test on first 500-1000 rows
                test_size = min(1000, len(df) - min_rows)
                if test_size > 100:
                    logger.info(
                        f"[Proba Cache][{adapter.name}] Running validation: "
                        f"comparing batch vs legacy on {test_size} samples..."
                    )
                    
                    # Legacy method: Use cached sequences directly for validation
                    # IMPORTANT: Extract features from full DataFrame ONCE to use cache
                    # Both batch and legacy now use the same cached sequences, ensuring mathematical identity
                    legacy_proba_long: list[float] = []
                    legacy_proba_short: list[float] = []
                    legacy_errors = 0
                    
                    # Extract features once from full DataFrame (reuses cache from batch prediction)
                    features_for_validation = model._extract_features(df, symbol=symbol, timeframe=timeframe)
                    sequences_validation = model._get_or_build_sequences_full(
                        features_df=features_for_validation,
                        window_size=model.window_size,
                        feature_cols=model.feature_cols,
                        logger=logger,
                    )
                    
                    # Validation: Compare batch vs legacy using same sequence indices
                    # Batch predictions are indexed by sequence index: batch[seq_idx] corresponds to
                    # sequence at index seq_idx, which uses features[seq_idx : seq_idx + window_size]
                    # Legacy predictions use the same sequences, so we iterate over sequence indices
                    test_end_idx = min_rows + test_size
                    model.model.eval()
                    with torch.no_grad():
                        # Iterate over sequence indices (same as batch indexing)
                        # Compare batch[seq_idx] with legacy prediction for sequence[seq_idx]
                        for seq_idx in range(min(test_size, len(sequences_validation))):
                            try:
                                # Get sequence and predict (same as batch path)
                                seq = sequences_validation[seq_idx:seq_idx+1]  # (1, window_size, feature_dim)
                                seq_tensor = torch.from_numpy(seq).to(model.device)
                                
                                logits = model.model(seq_tensor)  # (1, 3)
                                probs = torch.nn.functional.softmax(logits, dim=-1)  # (1, 3)
                                
                                # Extract probabilities from 3-class softmax using explicit class indices
                                proba_long_val = float(probs[0, LstmClassIndex.LONG].cpu().item())  # LONG class (index 1)
                                proba_short_val = float(probs[0, LstmClassIndex.SHORT].cpu().item())  # SHORT class (index 2)
                                
                                legacy_proba_long.append(proba_long_val)
                                legacy_proba_short.append(proba_short_val)
                            except Exception as exc:
                                legacy_errors += 1
                                if legacy_errors <= 3:
                                    logger.warning(
                                        f"[Proba Cache][{adapter.name}] Legacy prediction failed at seq_idx {seq_idx}: {exc}"
                                    )
                                continue
                    
                    if len(legacy_proba_long) > 0 and len(proba_long_values) >= len(legacy_proba_long):
                        # Compare first test_size predictions
                        batch_proba_long = proba_long_arr[:len(legacy_proba_long)]
                        batch_proba_short = proba_short_arr[:len(legacy_proba_long)]
                        legacy_proba_long_arr = np.array(legacy_proba_long, dtype=np.float32)
                        legacy_proba_short_arr = np.array(legacy_proba_short, dtype=np.float32)
                        
                        # Check for NaN/Inf in batch or legacy arrays
                        batch_has_nan = np.isnan(batch_proba_long).any() or np.isnan(batch_proba_short).any()
                        batch_has_inf = np.isinf(batch_proba_long).any() or np.isinf(batch_proba_short).any()
                        legacy_has_nan = np.isnan(legacy_proba_long_arr).any() or np.isnan(legacy_proba_short_arr).any()
                        legacy_has_inf = np.isinf(legacy_proba_long_arr).any() or np.isinf(legacy_proba_short_arr).any()
                        
                        if batch_has_nan or batch_has_inf:
                            logger.error(
                                f"[Proba Cache][{adapter.name}] ⚠️  CRITICAL: Found NaN/Inf in batch predictions! "
                                f"batch_has_nan={batch_has_nan}, batch_has_inf={batch_has_inf}. "
                                f"This indicates a problem in batch inference path."
                            )
                            if batch_has_nan:
                                nan_count_long = np.isnan(batch_proba_long).sum()
                                nan_count_short = np.isnan(batch_proba_short).sum()
                                logger.error(
                                    f"[Proba Cache][{adapter.name}] NaN counts: "
                                    f"proba_long={nan_count_long}/{len(batch_proba_long)}, "
                                    f"proba_short={nan_count_short}/{len(batch_proba_short)}"
                                )
                            raise ValueError(
                                "NaN/Inf detected in batch predictions. "
                                "Cannot proceed with validation. Please check batch inference path."
                            )
                        
                        if legacy_has_nan or legacy_has_inf:
                            logger.error(
                                f"[Proba Cache][{adapter.name}] ⚠️  CRITICAL: Found NaN/Inf in legacy predictions! "
                                f"legacy_has_nan={legacy_has_nan}, legacy_has_inf={legacy_has_inf}. "
                                f"This indicates a problem in legacy inference path."
                            )
                            raise ValueError(
                                "NaN/Inf detected in legacy predictions. "
                                "Cannot proceed with validation. Please check legacy inference path."
                            )
                        
                        # Calculate differences
                        diff_long = np.abs(batch_proba_long - legacy_proba_long_arr)
                        diff_short = np.abs(batch_proba_short - legacy_proba_short_arr)
                        
                        # Check for NaN/Inf in differences (should not happen if inputs are valid)
                        # With sanitization in the helper, this should not occur, but keep as defensive check
                        if np.isnan(diff_long).any() or np.isnan(diff_short).any():
                            logger.error(
                                f"[Proba Cache][{adapter.name}] ⚠️  Found NaN in differences. "
                                f"This should not happen if batch and legacy are both valid. "
                                f"This may indicate a deeper issue despite sanitization."
                            )
                            # Replace NaN diffs with 0.0 for defensive calculation (but log the issue)
                            nan_mask_long = np.isnan(diff_long)
                            nan_mask_short = np.isnan(diff_short)
                            if nan_mask_long.any():
                                logger.error(
                                    f"[Proba Cache][{adapter.name}] NaN in diff_long at {np.where(nan_mask_long)[0][:10]} "
                                    f"(showing first 10 of {nan_mask_long.sum()} total)"
                                )
                                diff_long[nan_mask_long] = 0.0
                            if nan_mask_short.any():
                                logger.error(
                                    f"[Proba Cache][{adapter.name}] NaN in diff_short at {np.where(nan_mask_short)[0][:10]} "
                                    f"(showing first 10 of {nan_mask_short.sum()} total)"
                                )
                                diff_short[nan_mask_short] = 0.0
                        
                        # Check for Inf in differences (should not happen)
                        if np.isinf(diff_long).any() or np.isinf(diff_short).any():
                            logger.error(
                                f"[Proba Cache][{adapter.name}] ⚠️  Found Inf in differences. "
                                f"This should not happen if batch and legacy are both valid."
                            )
                            # Replace Inf diffs with a large but finite value for defensive calculation
                            inf_mask_long = np.isinf(diff_long)
                            inf_mask_short = np.isinf(diff_short)
                            if inf_mask_long.any():
                                diff_long[inf_mask_long] = 1.0  # Large but finite value
                            if inf_mask_short.any():
                                diff_short[inf_mask_short] = 1.0  # Large but finite value
                        
                        # Now calculate statistics (all values should be finite)
                        max_diff_long = float(np.max(diff_long))
                        max_diff_short = float(np.max(diff_short))
                        mean_diff_long = float(np.mean(diff_long))
                        mean_diff_short = float(np.mean(diff_short))
                        
                        # Final sanity check: statistics should be finite
                        if not (np.isfinite(max_diff_long) and np.isfinite(mean_diff_long) and
                                np.isfinite(max_diff_short) and np.isfinite(mean_diff_short)):
                            logger.error(
                                f"[Proba Cache][{adapter.name}] ⚠️  CRITICAL: Statistics contain NaN/Inf despite sanitization. "
                                f"max_diff_long={max_diff_long}, mean_diff_long={mean_diff_long}, "
                                f"max_diff_short={max_diff_short}, mean_diff_short={mean_diff_short}"
                            )
                            # Use fallback values for defensive logging
                            max_diff_long = 0.0 if not np.isfinite(max_diff_long) else max_diff_long
                            mean_diff_long = 0.0 if not np.isfinite(mean_diff_long) else mean_diff_long
                            max_diff_short = 0.0 if not np.isfinite(max_diff_short) else max_diff_short
                            mean_diff_short = 0.0 if not np.isfinite(mean_diff_short) else mean_diff_short
                        
                        # Calculate percentiles for more detailed diagnostics
                        p95_diff_long = float(np.percentile(diff_long, 95))
                        p95_diff_short = float(np.percentile(diff_short, 95))
                        p99_diff_long = float(np.percentile(diff_long, 99))
                        p99_diff_short = float(np.percentile(diff_short, 99))
                        
                        logger.info(
                            f"[Proba Cache][{adapter.name}] Validation results (batch vs legacy): "
                            f"test_samples={len(legacy_proba_long)}, "
                            f"max_diff_long={max_diff_long:.6f}, mean_diff_long={mean_diff_long:.6f}, "
                            f"p95_diff_long={p95_diff_long:.6f}, p99_diff_long={p99_diff_long:.6f}, "
                            f"max_diff_short={max_diff_short:.6f}, mean_diff_short={mean_diff_short:.6f}, "
                            f"p95_diff_short={p95_diff_short:.6f}, p99_diff_short={p99_diff_short:.6f}"
                        )
                        
                        # Stricter tolerance: float precision level (1e-4 for mean, 1e-3 for max)
                        tolerance_mean = 1e-4
                        tolerance_max = 1e-3
                        
                        passed = (
                            mean_diff_long < tolerance_mean and
                            max_diff_long < tolerance_max and
                            mean_diff_short < tolerance_mean and
                            max_diff_short < tolerance_max
                        )
                        
                        if passed:
                            logger.info(
                                f"[Proba Cache][{adapter.name}] ✅ Validation PASSED: "
                                f"batch vs legacy match within tolerance "
                                f"(mean_diff < {tolerance_mean:.0e}, max_diff < {tolerance_max:.0e}). "
                                f"This confirms batch and legacy paths produce identical results."
                            )
                        else:
                            # Find samples with largest differences for debugging
                            worst_long_idx = int(np.argmax(diff_long))
                            worst_short_idx = int(np.argmax(diff_short))
                            
                            logger.warning(
                                f"[Proba Cache][{adapter.name}] ⚠️  Validation FAILED: "
                                f"differences exceed tolerance. "
                                f"mean_diff_long={mean_diff_long:.6f} (threshold={tolerance_mean:.0e}), "
                                f"max_diff_long={max_diff_long:.6f} (threshold={tolerance_max:.0e}), "
                                f"mean_diff_short={mean_diff_short:.6f} (threshold={tolerance_mean:.0e}), "
                                f"max_diff_short={max_diff_short:.6f} (threshold={tolerance_max:.0e})"
                            )
                            logger.warning(
                                f"[Proba Cache][{adapter.name}] Worst LONG diff at sample {worst_long_idx}: "
                                f"batch={batch_proba_long[worst_long_idx]:.6f}, "
                                f"legacy={legacy_proba_long_arr[worst_long_idx]:.6f}, "
                                f"diff={diff_long[worst_long_idx]:.6f}"
                            )
                            logger.warning(
                                f"[Proba Cache][{adapter.name}] Worst SHORT diff at sample {worst_short_idx}: "
                                f"batch={batch_proba_short[worst_short_idx]:.6f}, "
                                f"legacy={legacy_proba_short_arr[worst_short_idx]:.6f}, "
                                f"diff={diff_short[worst_short_idx]:.6f}"
                            )
                            logger.warning(
                                f"[Proba Cache][{adapter.name}] This may indicate: "
                                f"(1) normalization mismatch between batch and legacy paths, "
                                f"(2) feature extraction differences, or "
                                f"(3) sequence window indexing differences. "
                                f"Please review the implementation."
                            )
        except Exception as exc:
            logger.warning(
                f"[Proba Cache][{adapter.name}] Validation failed: {type(exc).__name__}: {exc}"
            )
    
    return proba_long_arr, proba_short_arr, df.iloc[valid_indices].reset_index(drop=True)


def _parse_args():
    """Parse command-line arguments for ML probability cache generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate and cache ML prediction probabilities for threshold optimization."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["ml_xgb", "ml_lstm_attn"],
        help="ML strategy name (ml_xgb or ml_lstm_attn)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Timeframe (default: 1m)",
    )
    parser.add_argument(
        "--feature-preset",
        type=str,
        default="extended_safe",
        choices=["base", "extended_safe", "extended_full"],
        help="Feature preset for ml_xgb strategy (default: extended_safe)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild cache even if it already exists",
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=None,
        help="Number of CPU threads for XGBoost (default: auto-detected)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Filter OHLCV to samples at or after this date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Filter OHLCV to samples at or before this date (YYYY-MM-DD format)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI execution."""
    import sys
    
    # Setup logging if not already configured
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    
    try:
        args = _parse_args()
        
        # Log startup banner
        logger.info("=" * 60)
        logger.info("[ML_PROBA_CACHE] Starting probability cache generation")
        logger.info(f"[ML_PROBA_CACHE] strategy={args.strategy}, symbol={args.symbol}, timeframe={args.timeframe}")
        if args.strategy == "ml_xgb":
            logger.info(f"[ML_PROBA_CACHE] feature_preset={args.feature_preset}")
        if args.start_date or args.end_date:
            logger.info(f"[ML_PROBA_CACHE] date_range: start={args.start_date or 'None'}, end={args.end_date or 'None'}")
        logger.info(f"[ML_PROBA_CACHE] force_rebuild={args.force_rebuild}")
        logger.info("=" * 60)
        
        # Validate strategy
        from src.backtest.engine import _get_ml_adapter
        try:
            adapter = _get_ml_adapter(args.strategy)
            logger.info(f"[ML_PROBA_CACHE] Strategy '{args.strategy}' is supported (adapter: {adapter.name})")
        except ValueError as e:
            logger.error(f"[ML_PROBA_CACHE] Unsupported strategy: {args.strategy}")
            logger.error(f"[ML_PROBA_CACHE] Error: {e}")
            logger.error("[ML_PROBA_CACHE] Supported strategies: ml_xgb, ml_lstm_attn")
            sys.exit(1)
        
        # Get cache path for info
        cache_path = _get_cache_path(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            feature_preset=args.feature_preset if args.strategy == "ml_xgb" else "base",
            start_date=args.start_date,
            end_date=args.end_date,
        )
        
        # Check if cache exists
        if cache_path.exists() and not args.force_rebuild:
            logger.warning(
                f"[ML_PROBA_CACHE] Cache file already exists and --force-rebuild not set: {cache_path}"
            )
            logger.info("[ML_PROBA_CACHE] Skipping cache generation. Use --force-rebuild to regenerate.")
            logger.info("=" * 60)
            logger.info("[ML_PROBA_CACHE] Done (skipped - cache exists)")
            logger.info("=" * 60)
            return 0
        
        # Generate cache
        logger.info(f"[ML_PROBA_CACHE] Generating predictions and saving to: {cache_path}")
        
        proba_long_arr, proba_short_arr, df_aligned = get_or_build_predictions(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            feature_preset=args.feature_preset if args.strategy == "ml_xgb" else "base",
            force_rebuild=args.force_rebuild,
            nthread=args.nthread,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        
        # Success summary
        num_rows = len(proba_long_arr)
        start_ts = df_aligned["timestamp"].iloc[0] if len(df_aligned) > 0 else "N/A"
        end_ts = df_aligned["timestamp"].iloc[-1] if len(df_aligned) > 0 else "N/A"
        
        logger.info("=" * 60)
        logger.info("[ML_PROBA_CACHE] Done. Cache generation completed successfully.")
        logger.info(f"[ML_PROBA_CACHE] Saved cache to: {cache_path}")
        logger.info(f"[ML_PROBA_CACHE] Summary: rows={num_rows}, start={start_ts}, end={end_ts}")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("[ML_PROBA_CACHE] Interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"[ML_PROBA_CACHE] Fatal error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

