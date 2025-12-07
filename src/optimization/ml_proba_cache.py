"""
ML prediction probability caching for threshold optimization.

This module provides functions to compute and cache prediction probabilities
for ML strategies, allowing threshold optimization to reuse predictions
across multiple threshold combinations without recomputing.
"""
from __future__ import annotations

import logging
from pathlib import Path
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
        # Optimized batch path for LSTM-Attention
        from src.dl.lstm_attn_model import LSTMAttnSignalModel
        
        if not isinstance(model, LSTMAttnSignalModel):
            raise ValueError(
                f"[Proba Cache][{adapter.name}] Expected LSTMAttnSignalModel, got {type(model)}"
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
            # proba arrays have length (len(full_features) - window_size + 1)
            # But we need predictions starting from min_rows (window_size)
            # So we need to map feature indices back to df indices
            
            num_proba = len(proba_long_arr)
            num_features = len(full_features)
            
            # The proba arrays correspond to sequences starting from index (window_size - 1) in features
            # But we want predictions starting from min_rows in df
            # Since features may have fewer rows than df due to dropna(), we need to be careful
            
            # Calculate how many predictions we can use
            # We need at least min_rows predictions to match the old behavior
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
            prediction_errors = 0
            for i in range(min_rows, len(df)):
                df_slice = df.iloc[: i + 1]
                try:
                    proba_up = float(model.predict_proba_latest(df_slice, symbol=symbol, timeframe=timeframe))
                    proba_long_values.append(proba_up)
                    proba_short_values.append(1.0 - proba_up)
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
    
    logger.info(
        f"[Proba Cache][{adapter.name}] Completed: "
        f"successful={len(proba_long_values)}, errors={prediction_errors}, "
        f"mean_proba_long={proba_long_arr.mean():.4f}, std={proba_long_arr.std():.4f}, "
        f"mean_proba_short={proba_short_arr.mean():.4f}, std={proba_short_arr.std():.4f}"
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
                    
                    # Legacy method: predict_proba_latest for each step
                    legacy_proba_long: list[float] = []
                    legacy_proba_short: list[float] = []
                    legacy_errors = 0
                    
                    test_end_idx = min_rows + test_size
                    for i in range(min_rows, test_end_idx):
                        df_slice = df.iloc[: i + 1]
                        try:
                            proba_up = float(model.predict_proba_latest(df_slice, symbol=symbol, timeframe=timeframe))
                            legacy_proba_long.append(proba_up)
                            legacy_proba_short.append(1.0 - proba_up)
                        except Exception as exc:
                            legacy_errors += 1
                            if legacy_errors <= 3:
                                logger.warning(
                                    f"[Proba Cache][{adapter.name}] Legacy prediction failed at index {i}: {exc}"
                                )
                            continue
                    
                    if len(legacy_proba_long) > 0 and len(proba_long_values) >= len(legacy_proba_long):
                        # Compare first test_size predictions
                        batch_proba_long = proba_long_arr[:len(legacy_proba_long)]
                        batch_proba_short = proba_short_arr[:len(legacy_proba_long)]
                        legacy_proba_long_arr = np.array(legacy_proba_long, dtype=np.float32)
                        legacy_proba_short_arr = np.array(legacy_proba_short, dtype=np.float32)
                        
                        # Calculate differences
                        diff_long = np.abs(batch_proba_long - legacy_proba_long_arr)
                        diff_short = np.abs(batch_proba_short - legacy_proba_short_arr)
                        
                        max_diff_long = float(np.max(diff_long))
                        max_diff_short = float(np.max(diff_short))
                        mean_diff_long = float(np.mean(diff_long))
                        mean_diff_short = float(np.mean(diff_short))
                        
                        logger.info(
                            f"[Proba Cache][{adapter.name}] Validation results (batch vs legacy): "
                            f"test_samples={len(legacy_proba_long)}, "
                            f"max_diff_long={max_diff_long:.6f}, mean_diff_long={mean_diff_long:.6f}, "
                            f"max_diff_short={max_diff_short:.6f}, mean_diff_short={mean_diff_short:.6f}"
                        )
                        
                        # Warn if differences are too large (likely normalization mismatch)
                        if max_diff_long > 0.01 or max_diff_short > 0.01:
                            logger.warning(
                                f"[Proba Cache][{adapter.name}] Large differences detected! "
                                f"This may indicate a normalization mismatch. "
                                f"Please review the batch prediction implementation."
                            )
                        else:
                            logger.info(
                                f"[Proba Cache][{adapter.name}] Validation passed: "
                                f"differences are within acceptable range (<0.01)"
                            )
        except Exception as exc:
            logger.warning(
                f"[Proba Cache][{adapter.name}] Validation failed: {type(exc).__name__}: {exc}"
            )
    
    return proba_long_arr, proba_short_arr, df.iloc[valid_indices].reset_index(drop=True)

