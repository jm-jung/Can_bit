"""Feature engineering for ML models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from src.core.config import settings
from src.events.aggregator import build_event_feature_frame
from src.features.extended_features import (
    build_extended_trend_features,
    build_volatility_features,
    build_volume_features,
    build_structure_features,
)
from src.features.ml_feature_config import MLFeatureConfig
from src.indicators.basic import add_basic_indicators
from src.services.ohlcv_service import load_ohlcv_df

logger = logging.getLogger(__name__)


def _reindex_and_fill(features_df: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    """
    Reindex features DataFrame to match target index and fill NaN values with 0.
    
    This ensures all feature DataFrames are aligned to the same index and
    NaN values (which typically represent missing observations) are filled with 0.
    
    Args:
        features_df: Feature DataFrame to reindex
        target_index: Target index to align to
    
    Returns:
        Reindexed and filled DataFrame
    """
    if features_df is None or len(features_df) == 0:
        return pd.DataFrame(index=target_index)
    
    # Reindex to match target index
    features_df = features_df.reindex(target_index)
    
    # Fill NaN with 0 (for event/extended features, NaN typically means "no observation")
    return features_df.fillna(0.0)


def _log_feature_debug(
    tag: str,
    data: pd.DataFrame,
    *,
    debug_inspect: bool,
    debug_logger: logging.Logger | None,
) -> None:
    if not debug_inspect or debug_logger is None:
        return
    try:
        debug_logger.info("[DEBUG][FEATURE]%s shape=%s", tag, str(data.shape))
        debug_logger.info("[DEBUG][FEATURE]%s columns=%s", tag, list(data.columns))
        debug_logger.info("[DEBUG][FEATURE]%s head(5):\n%s", tag, data.head(5))
        debug_logger.info("[DEBUG][FEATURE]%s describe():\n%s", tag, data.describe().to_string())
    except Exception as e:
        debug_logger.warning("[DEBUG][FEATURE]%s failed to log: %s", tag, e)


def build_technical_feature_df(
    df_in: pd.DataFrame,
    *,
    debug_inspect: bool = False,
    debug_logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Build OHLCV + indicator-based technical features.
    """
    if "timestamp" in df_in.columns:
        df = df_in.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
    else:
        df = df_in.sort_index()

    tech_feat = pd.DataFrame(index=df.index)

    # Basic OHLCV features
    tech_feat["close"] = df["close"]
    tech_feat["high"] = df["high"]
    tech_feat["low"] = df["low"]
    tech_feat["volume"] = df["volume"]

    # Price-derived features
    tech_feat["high_low"] = df["high"] - df["low"]
    tech_feat["close_open"] = df["close"] - df["open"]
    open_nonzero = df["open"].replace(0.0, np.nan)
    close_open_pct = tech_feat["close_open"] / open_nonzero
    tech_feat["close_open_pct"] = close_open_pct.fillna(0.0)

    # Existing indicators
    tech_feat["ema_20"] = df["ema_20"]
    tech_feat["sma_20"] = df["sma_20"]
    tech_feat["rsi_14"] = df["rsi_14"]

    # Rolling statistics
    tech_feat["rolling_std_20"] = df["close"].rolling(window=20, min_periods=1).std()
    tech_feat["rolling_mean_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    # Ratios
    ema_nonzero = df["ema_20"].replace(0.0, np.nan)
    sma_nonzero = df["sma_20"].replace(0.0, np.nan)
    tech_feat["close_ema_ratio"] = (df["close"] / ema_nonzero).replace([np.inf, -np.inf], np.nan)
    tech_feat["close_sma_ratio"] = (df["close"] / sma_nonzero).replace([np.inf, -np.inf], np.nan)

    _log_feature_debug("[TECH_FEAT]", tech_feat, debug_inspect=debug_inspect, debug_logger=debug_logger)
    return tech_feat


def build_feature_frame(
    df: pd.DataFrame,
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    use_events: bool | None = None,
    event_lookback_minutes: int | None = None,
    feature_config: Optional[MLFeatureConfig] = None,
    debug_inspect: bool = False,
    debug_logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Build feature frame with technical features and optional event features.
    
    This is the unified entry point for feature generation used by:
    - XGB/LSTM training
    - ML backtesting
    - Threshold optimization
    
    Args:
        df: OHLCV DataFrame with timestamp column or timestamp index
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "1m")
        use_events: Whether to include event features (default: from settings)
            Note: This is overridden by feature_config.use_event_features if feature_config is provided
        event_lookback_minutes: Lookback window for event aggregation (default: from settings)
        feature_config: MLFeatureConfig instance (default: "base" preset)
        debug_inspect: Enable debug logging
        debug_logger: Logger instance for debug output
    
    Returns:
        DataFrame with features:
        - Technical features: close, high, low, volume, ema_20, sma_20, rsi_14, etc.
        - Event features (if enabled): event_count_*, event_sentiment_*, etc.
        - Extended features (if enabled): feat_trend_*, feat_vol_*, feat_volu_*, feat_struct_*
        - Index: timestamp (same as input df)
        - Column order: base features first, then extended features, then event features
    """
    if "timestamp" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must include timestamp column or have timestamp index.")

    # Use default "base" preset if no config provided (maintains backward compatibility)
    if feature_config is None:
        feature_config = MLFeatureConfig.from_preset("base")
    
    # Override use_events from config if provided
    if feature_config is not None:
        use_events = feature_config.use_event_features
    else:
        use_events = settings.EVENTS_ENABLED if use_events is None else use_events

    df = df.copy()
    _log_feature_debug("[IN]", df, debug_inspect=debug_inspect, debug_logger=debug_logger)
    
    # Ensure df has a consistent index (DatetimeIndex preferred)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        # If index is not DatetimeIndex, try to convert or use integer index
        df = df.sort_index()
    
    # Store target index for alignment
    target_index = df.index
    
    df = add_basic_indicators(df)
    _log_feature_debug("[TECH]", df, debug_inspect=debug_inspect, debug_logger=debug_logger)

    # Build base technical features
    tech_features = None
    if feature_config.use_base_price_features or feature_config.use_indicator_features:
        tech_features = build_technical_feature_df(
            df,
            debug_inspect=debug_inspect,
            debug_logger=debug_logger,
        )
        tech_features = _reindex_and_fill(tech_features, target_index)
    
    # Add extended features based on config
    trend_features = None
    if feature_config.use_extended_trend_features:
        trend_features = build_extended_trend_features(df)
        trend_features = _reindex_and_fill(trend_features, target_index)
        _log_feature_debug("[TREND]", trend_features, debug_inspect=debug_inspect, debug_logger=debug_logger)
    
    vol_features = None
    if feature_config.use_volatility_features:
        vol_features = build_volatility_features(df)
        vol_features = _reindex_and_fill(vol_features, target_index)
        _log_feature_debug("[VOL]", vol_features, debug_inspect=debug_inspect, debug_logger=debug_logger)
    
    volume_features = None
    if feature_config.use_volume_features:
        volume_features = build_volume_features(df)
        volume_features = _reindex_and_fill(volume_features, target_index)
        _log_feature_debug("[VOLUME]", volume_features, debug_inspect=debug_inspect, debug_logger=debug_logger)
    
    struct_features = None
    if feature_config.use_structure_features:
        struct_features = build_structure_features(df)
        struct_features = _reindex_and_fill(struct_features, target_index)
        _log_feature_debug("[STRUCT]", struct_features, debug_inspect=debug_inspect, debug_logger=debug_logger)
    
    # Combine all base and extended features (only non-None)
    feature_frames = []
    if tech_features is not None:
        feature_frames.append(tech_features)
    if trend_features is not None:
        feature_frames.append(trend_features)
    if vol_features is not None:
        feature_frames.append(vol_features)
    if volume_features is not None:
        feature_frames.append(volume_features)
    if struct_features is not None:
        feature_frames.append(struct_features)
    
    if feature_frames:
        feature_df = pd.concat(feature_frames, axis=1)
    else:
        # Fallback: create empty DataFrame with same index
        feature_df = pd.DataFrame(index=target_index)
    
    # Add event features if enabled
    event_df = None
    if feature_config.use_event_features:
        # Use v1 event feature builder: build_event_feature_frame
        # This function loads raw events and computes features aligned to OHLCV
        event_df = build_event_feature_frame(
            ohlcv_df=df,
            symbol=symbol,
            timeframe=timeframe,
            lookback_minutes=event_lookback_minutes,
        )
        
        # Reindex and fill event features
        event_df = _reindex_and_fill(event_df, target_index)
        
        # Merge with base features
        feature_df = pd.concat([feature_df, event_df], axis=1)
        
        _log_feature_debug("[EVENT]", feature_df, debug_inspect=debug_inspect, debug_logger=debug_logger)

    # Replace inf/-inf with NaN, then fill remaining NaN with 0
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(0.0)
    
    # Ensure index is sorted
    if isinstance(feature_df.index, pd.DatetimeIndex):
        feature_df = feature_df.sort_index()
    else:
        feature_df = feature_df.sort_index()
    
    _log_feature_debug("[POST]", feature_df, debug_inspect=debug_inspect, debug_logger=debug_logger)
    
    if debug_logger:
        debug_logger.info(f"[FEATURE_CONFIG] Using preset: {feature_config.preset_name}")
        debug_logger.info(f"[FEATURE_CONFIG] Total features: {len(feature_df.columns)}")
    
    return feature_df


def build_ml_dataset(
    df: pd.DataFrame,
    horizon: int = 20,
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    use_events: bool | None = None,
    event_lookback_minutes: int | None = None,
    feature_config: Optional[MLFeatureConfig] = None,
    label_threshold: float = 0.001,
    long_threshold: float = 0.002,
    short_threshold: float = 0.002,
    hold_threshold: float = 0.0005,
    enable_hold_labels: bool = False,
    debug_inspect: bool = False,
    debug_logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, pd.Series] | Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Build ML dataset with features and labels.
    
    Args:
        df: OHLCV DataFrame with timestamp column
        horizon: Number of periods ahead to predict (default: 20)
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "1m")
        use_events: Whether to include event features (default: from settings)
            Note: This is overridden by feature_config.use_event_features if feature_config is provided
        event_lookback_minutes: Lookback window for event aggregation (default: from settings)
        feature_config: MLFeatureConfig instance (default: "base" preset)
        label_threshold: Return threshold for positive label (default: 0.001 = 0.1%, backward compatibility)
        long_threshold: Return threshold for LONG label (default: 0.002 = 0.2%)
        short_threshold: Return threshold for SHORT label (default: 0.002 = 0.2%)
        hold_threshold: Return threshold for HOLD zone (abs(ret) <= hold_threshold, default: 0.0005 = 0.05%)
        enable_hold_labels: If True, generate y_hold and return it (default: False for backward compatibility)
        debug_inspect: Enable debug logging
        debug_logger: Logger instance for debug output
    
    Returns:
        If enable_hold_labels=False: (X, y, y_long, y_short) - backward compatible
        If enable_hold_labels=True: (X, y, y_long, y_short, y_hold)
        Note: y is y_long for backward compatibility
    """
    # Step 1: Copy and normalize index to RangeIndex for consistency
    df_original = df.copy()
    df = df.copy()
    
    # Reset index to RangeIndex to ensure consistent indexing
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)
    
    logger.info("[ML Features] Initial rows: %d", len(df))
    
    # Step 2: Ensure numeric types for OHLCV columns
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    for col in ["high", "low", "volume", "open"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Step 3: Build features using the full df (before label filtering)
    # This ensures features are calculated on the complete dataset
    features = build_feature_frame(
        df,
        symbol=symbol,
        timeframe=timeframe,
        use_events=use_events,
        event_lookback_minutes=event_lookback_minutes,
        feature_config=feature_config,
        debug_inspect=debug_inspect,
        debug_logger=debug_logger,
    )
    
    # Step 4: Ensure features index matches df index (both should be RangeIndex now)
    if not features.index.equals(df.index):
        # If indices don't match, reset both to ensure alignment
        df = df.reset_index(drop=True)
        features = features.reset_index(drop=True)
        logger.info(
            "[ML Features] Reset indices to RangeIndex for alignment. "
            "df rows: %d, features rows: %d",
            len(df), len(features)
        )
    
    # Step 5: Calculate labels and return_next_horizon
    # Label definition:
    # - Calculate future return: r = (price_{t+horizon} / price_t - 1)
    # - Label = 1 if r > threshold (0.1%), Label = 0 otherwise
    # - This is a binary classification task: predict significant price movement
    # - Horizon: number of periods (default: 30 for 30-minute ahead prediction on 1m data)
    # - Threshold: 0.001 (0.1%) to filter out noise and ensure meaningful price movements
    
    df["return_next_horizon"] = df["close"].shift(-horizon) / df["close"] - 1
    ret = df["return_next_horizon"]
    
    # Step 6: Create valid mask for return (NaN filtering)
    ret_valid_mask = ret.notna()
    valid_count = int(ret_valid_mask.sum())
    total_count = len(ret)
    
    logger.info("[ML Features] Rows after label/ret filtering: %d / %d", valid_count, total_count)
    
    if valid_count == 0:
        logger.error(
            "[Label Debug] All return_next_horizon values are NaN. "
            "Total rows: %d, Valid rows: %d. Check data and horizon setting.",
            total_count, valid_count
        )
        raise ValueError(
            f"All return_next_horizon values are NaN. "
            f"Total rows: {total_count}, Valid rows: {valid_count}. "
            "This may indicate insufficient data or horizon too large."
        )
    
    # Step 7: Generate labels with threshold (on valid rows only)
    # Step A: LONG/SHORT split + proba cache patch
    # Step C: HOLD label introduction
    # LONG label definition: y_long = 1 if ret > long_threshold, else 0
    #   - Predicts significant upward price movement (ret > long_threshold)
    #   - Used for LONG model training and LONG signal generation
    # SHORT label definition: y_short = 1 if ret < -short_threshold, else 0
    #   - Predicts significant downward price movement (ret < -short_threshold)
    #   - Used for SHORT model training and SHORT signal generation
    # HOLD label definition: y_hold = 1 if abs(ret) <= hold_threshold, else 0
    #   - Middle zone where price movement is too small to trade
    #   - Used to exclude ambiguous samples from training
    # Note: long_threshold=0.002 (0.2%), short_threshold=0.002 (0.2%), hold_threshold=0.0005 (0.05%)
    # Backward compatibility: if enable_hold_labels=False, use label_threshold for LONG/SHORT
    if enable_hold_labels:
        long_th = long_threshold
        short_th = short_threshold
        hold_th = hold_threshold
    else:
        # Backward compatibility: use label_threshold
        long_th = label_threshold
        short_th = label_threshold
        hold_th = 0.0  # No HOLD zone in backward compatibility mode
    
    y_long = pd.Series(index=df.index, dtype=int)
    y_short = pd.Series(index=df.index, dtype=int)
    y_hold = pd.Series(index=df.index, dtype=int) if enable_hold_labels else None
    
    ret_valid = ret.loc[ret_valid_mask]
    ret_abs = ret_valid.abs()
    
    # Generate LONG labels
    y_long.loc[ret_valid_mask] = (ret_valid > long_th).astype(int)
    y_long.loc[~ret_valid_mask] = np.nan
    
    # Generate SHORT labels
    y_short.loc[ret_valid_mask] = (ret_valid < -short_th).astype(int)
    y_short.loc[~ret_valid_mask] = np.nan
    
    # Generate HOLD labels (only if enable_hold_labels=True)
    if enable_hold_labels:
        y_hold.loc[ret_valid_mask] = (ret_abs <= hold_th).astype(int)
        y_hold.loc[~ret_valid_mask] = np.nan
    
    # For backward compatibility, y_long is used as the default y
    y = y_long.copy()
    
    # Step 8: Debug logging for label generation (after NaN filtering)
    y_long_valid = y_long.loc[ret_valid_mask]
    y_short_valid = y_short.loc[ret_valid_mask]
    y_hold_valid = y_hold.loc[ret_valid_mask] if enable_hold_labels and y_hold is not None else None
    
    if enable_hold_labels:
        logger.info("[Label Debug] horizon=%d, long_threshold=%.4f, short_threshold=%.4f, hold_threshold=%.4f", 
                   horizon, long_th, short_th, hold_th)
    else:
        logger.info("[Label Debug] horizon=%d, threshold=%.4f (backward compatibility)", horizon, label_threshold)
    
    logger.info(
        "[Label Debug] ret stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        float(ret_valid.mean()),
        float(ret_valid.std()),
        float(ret_valid.min()),
        float(ret_valid.max()),
    )
    
    pos_long = int((y_long_valid == 1).sum())
    neg_long = int((y_long_valid == 0).sum())
    pos_short = int((y_short_valid == 1).sum())
    neg_short = int((y_short_valid == 0).sum())
    pos_hold = int((y_hold_valid == 1).sum()) if y_hold_valid is not None else 0
    neg_hold = int((y_hold_valid == 0).sum()) if y_hold_valid is not None else 0
    
    total_valid = len(ret_valid)
    logger.info(
        "[Label Debug] LONG label distribution: positive=%d (%.2f%%), negative=%d (%.2f%%)",
        pos_long, pos_long / total_valid * 100.0 if total_valid > 0 else 0.0,
        neg_long, neg_long / total_valid * 100.0 if total_valid > 0 else 0.0,
    )
    logger.info(
        "[Label Debug] SHORT label distribution: positive=%d (%.2f%%), negative=%d (%.2f%%)",
        pos_short, pos_short / total_valid * 100.0 if total_valid > 0 else 0.0,
        neg_short, neg_short / total_valid * 100.0 if total_valid > 0 else 0.0,
    )
    if enable_hold_labels and y_hold_valid is not None:
        logger.info(
            "[Label Debug] HOLD label distribution: positive=%d (%.2f%%), negative=%d (%.2f%%)",
            pos_hold, pos_hold / total_valid * 100.0 if total_valid > 0 else 0.0,
            neg_hold, neg_hold / total_valid * 100.0 if total_valid > 0 else 0.0,
        )
    
    # Step 9: Label validation safety check
    if pos_long == 0 or neg_long == 0:
        logger.error(
            "[Label Debug] Invalid LONG label distribution after filtering: positive=%d, negative=%d",
            pos_long, neg_long,
        )
        raise ValueError(
            f"LONG label distribution must contain both positive and negative samples. "
            f"Current: positive={pos_long}, negative={neg_long}. "
            f"Consider adjusting long_threshold (current: {long_th}) or horizon (current: {horizon})."
        )
    
    if pos_short == 0 or neg_short == 0:
        logger.error(
            "[Label Debug] Invalid SHORT label distribution after filtering: positive=%d, negative=%d",
            pos_short, neg_short,
        )
        raise ValueError(
            f"SHORT label distribution must contain both positive and negative samples. "
            f"Current: positive={pos_short}, negative={neg_short}. "
            f"Consider adjusting short_threshold (current: {short_th}) or horizon (current: {horizon})."
        )
    
    # Step 10: Create final valid mask (combine ret valid mask and feature valid mask)
    # Features should already have NaN filled with 0, but check for safety
    feature_valid_mask = ~features.isna().any(axis=1)
    label_valid_mask = y_long.notna() & y_short.notna()
    if enable_hold_labels and y_hold is not None:
        label_valid_mask = label_valid_mask & y_hold.notna()
    
    # Final mask: both features and labels must be valid
    final_valid_mask = ret_valid_mask & feature_valid_mask & label_valid_mask
    final_valid_count = int(final_valid_mask.sum())
    
    logger.info("[ML Features] Rows after feature NaN filtering: %d / %d", final_valid_count, total_count)
    
    if final_valid_count == 0:
        logger.error(
            "[ML Features] All rows were filtered out by final valid_mask. "
            "ret_valid: %d, feature_valid: %d, label_valid: %d",
            int(ret_valid_mask.sum()),
            int(feature_valid_mask.sum()),
            int(label_valid_mask.sum())
        )
        # Show which columns have NaN
        na_cols = features.columns[features.isna().any()].tolist()
        if na_cols:
            logger.error("[ML Features] Columns with NaN: %s", na_cols[:10])  # Show first 10
        raise ValueError(
            "No valid rows after feature engineering (all rows contain NaN in features or labels). "
            f"Total rows: {total_count}, Final valid rows: {final_valid_count}. "
            "Check feature engineering logic and NaN handling."
        )
    
    # Step 11: Apply final valid mask to df, features, and labels simultaneously
    df_filtered = df.loc[final_valid_mask].copy()
    features_filtered = features.loc[final_valid_mask].copy()
    y_long_filtered = y_long.loc[final_valid_mask].copy()
    y_short_filtered = y_short.loc[final_valid_mask].copy()
    y_hold_filtered = y_hold.loc[final_valid_mask].copy() if enable_hold_labels and y_hold is not None else None
    y_filtered = y_long_filtered.copy()  # For backward compatibility
    
    # Step 12: Reset indices to RangeIndex for clean final output
    df_filtered = df_filtered.reset_index(drop=True)
    features_filtered = features_filtered.reset_index(drop=True)
    y_long_filtered = y_long_filtered.reset_index(drop=True)
    y_short_filtered = y_short_filtered.reset_index(drop=True)
    y_filtered = y_filtered.reset_index(drop=True)
    if y_hold_filtered is not None:
        y_hold_filtered = y_hold_filtered.reset_index(drop=True)
    
    # Step 13: NaN summary logging for final filtered features
    total_rows = len(features_filtered)
    if total_rows > 0:
        na_per_col = features_filtered.isna().sum()
        na_cols = na_per_col[na_per_col > 0]
        if len(na_cols) > 0:
            logger.info("[ML Features] NaN summary (final rows=%d):", total_rows)
            for col, cnt in na_cols.items():
                logger.info("  - %s: %d NaNs (%.2f%%)", col, cnt, cnt / total_rows * 100.0)
        else:
            logger.info("[ML Features] No NaN values in features (final rows=%d)", total_rows)
    else:
        logger.warning("[ML Features] Features DataFrame is empty (rows=0)")
    
    # Step 14: Final validation
    if len(features_filtered) != len(y_long_filtered) or len(features_filtered) != len(y_short_filtered):
        logger.error(
            "[ML Features] X and y length mismatch after filtering: X=%d, y_long=%d, y_short=%d",
            len(features_filtered), len(y_long_filtered), len(y_short_filtered)
        )
        raise ValueError(
            f"X and y length mismatch: X={len(features_filtered)}, y_long={len(y_long_filtered)}, y_short={len(y_short_filtered)}. "
            "This should not happen. Check feature engineering logic."
        )
    if enable_hold_labels and y_hold_filtered is not None:
        if len(features_filtered) != len(y_hold_filtered):
            logger.error(
                "[ML Features] X and y_hold length mismatch after filtering: X=%d, y_hold=%d",
                len(features_filtered), len(y_hold_filtered)
            )
            raise ValueError(
                f"X and y_hold length mismatch: X={len(features_filtered)}, y_hold={len(y_hold_filtered)}. "
                "This should not happen. Check feature engineering logic."
            )
    
    # Step 15: Store labels in df for access and return
    df_filtered["y_long"] = y_long_filtered
    df_filtered["y_short"] = y_short_filtered
    df_filtered["y"] = y_filtered  # For backward compatibility
    if enable_hold_labels and y_hold_filtered is not None:
        df_filtered["y_hold"] = y_hold_filtered
    
    # Step 16: Return final X, y (backward compatible), y_long, y_short, y_hold (if enabled)
    X = features_filtered
    y = y_filtered

    if debug_inspect and debug_logger is not None:
        try:
            debug_logger.info("[DEBUG][FEATURE][OUT] final feature_df shape=%s", str(features.shape))
            debug_logger.info("[DEBUG][FEATURE][OUT] columns=%s", list(features.columns))
            debug_logger.info("[DEBUG][FEATURE][OUT] valid rows=%d", final_valid_count)
            debug_logger.info("[DEBUG][FEATURE][OUT] describe():\n%s", features.describe().to_string())
        except Exception as e:
            debug_logger.warning("[DEBUG][FEATURE][OUT] failed to log feature_df: %s", e)

    if enable_hold_labels and y_hold_filtered is not None:
        return X, y, y_long_filtered, y_short_filtered, y_hold_filtered
    else:
        return X, y, y_long_filtered, y_short_filtered


def build_ml_dataset_from_csv(
    csv_path: str | Path,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build ML dataset from CSV file.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return build_ml_dataset(df, **kwargs)


