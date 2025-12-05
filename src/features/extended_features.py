"""
Extended feature engineering for ML models (Stage 2).

This module provides additional feature categories beyond the base set:
- Extended trend features
- Volatility features
- Volume features
- Structure features
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_extended_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build extended trend features.
    
    Features:
    - Log returns (1, 3, 5 periods)
    - Distance from EMA/SMA (normalized)
    - EMA/SMA ratios
    - Trend slopes
    
    Args:
        df: DataFrame with OHLCV + indicators (must have close, ema_20, sma_20)
    
    Returns:
        DataFrame with extended trend features (prefixed with 'feat_trend_')
    """
    # Ensure we have a consistent index
    if df.index is None or len(df.index) == 0:
        raise ValueError("Input DataFrame must have a valid index")
    
    features = pd.DataFrame(index=df.index)
    
    # Log returns
    close_shifted = df["close"].shift(1).replace(0.0, np.nan)
    features["feat_trend_log_return_1"] = np.log(df["close"] / close_shifted)
    
    close_shifted_3 = df["close"].shift(3).replace(0.0, np.nan)
    features["feat_trend_log_return_3"] = np.log(df["close"] / close_shifted_3)
    
    close_shifted_5 = df["close"].shift(5).replace(0.0, np.nan)
    features["feat_trend_log_return_5"] = np.log(df["close"] / close_shifted_5)
    
    # Distance from EMA/SMA (normalized by close)
    if "ema_20" in df.columns:
        ema_20_nonzero = df["ema_20"].replace(0.0, np.nan)
        close_nonzero = df["close"].replace(0.0, np.nan)
        features["feat_trend_dist_close_ema20"] = (df["close"] - df["ema_20"]) / close_nonzero
        features["feat_trend_ema20_over_close"] = ema_20_nonzero / close_nonzero
    
    if "sma_20" in df.columns:
        sma_20_nonzero = df["sma_20"].replace(0.0, np.nan)
        close_nonzero = df["close"].replace(0.0, np.nan)
        features["feat_trend_dist_close_sma20"] = (df["close"] - df["sma_20"]) / close_nonzero
        features["feat_trend_sma20_over_close"] = sma_20_nonzero / close_nonzero
    
    # EMA/SMA ratio (if both exist)
    if "ema_20" in df.columns and "sma_20" in df.columns:
        ema_20_nonzero = df["ema_20"].replace(0.0, np.nan)
        sma_20_nonzero = df["sma_20"].replace(0.0, np.nan)
        features["feat_trend_ema20_over_sma20"] = ema_20_nonzero / sma_20_nonzero
    
    # Trend slope (5-period rolling mean slope)
    rolling_mean_5 = df["close"].rolling(window=5, min_periods=1).mean()
    rolling_mean_5_shifted = rolling_mean_5.shift(5).replace(0.0, np.nan)
    features["feat_trend_slope_5"] = (rolling_mean_5 - rolling_mean_5_shifted) / rolling_mean_5_shifted
    
    # Replace inf/-inf with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    return features


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build volatility features.
    
    Features:
    - True Range (TR)
    - Average True Range (ATR)
    - Normalized ATR
    - Return volatility (rolling std)
    - High-low range (normalized)
    
    Args:
        df: DataFrame with OHLCV (must have high, low, close)
    
    Returns:
        DataFrame with volatility features (prefixed with 'feat_vol_')
    """
    # Ensure we have a consistent index
    if df.index is None or len(df.index) == 0:
        raise ValueError("Input DataFrame must have a valid index")
    
    features = pd.DataFrame(index=df.index)
    
    # True Range
    high_low = df["high"] - df["low"]
    high_prev_close = abs(df["high"] - df["close"].shift(1))
    low_prev_close = abs(df["low"] - df["close"].shift(1))
    features["feat_vol_tr"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # ATR (14-period)
    features["feat_vol_atr_14"] = features["feat_vol_tr"].rolling(window=14, min_periods=1).mean()
    
    # Normalized ATR
    close_nonzero = df["close"].replace(0.0, np.nan)
    features["feat_vol_atr_norm"] = features["feat_vol_atr_14"] / close_nonzero
    
    # Return volatility (20-period rolling std of log returns)
    close_shifted = df["close"].shift(1).replace(0.0, np.nan)
    log_return = np.log(df["close"] / close_shifted)
    features["feat_vol_volatility_20"] = log_return.rolling(window=20, min_periods=1).std()
    
    # High-low range (normalized)
    features["feat_vol_range_norm"] = (df["high"] - df["low"]) / close_nonzero
    
    # Replace inf/-inf with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    return features


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build volume-based features.
    
    Features:
    - Volume moving average
    - Volume z-score
    - Volume ratio (short-term vs long-term)
    
    Args:
        df: DataFrame with OHLCV (must have volume)
    
    Returns:
        DataFrame with volume features (prefixed with 'feat_volu_')
    """
    # Ensure we have a consistent index
    if df.index is None or len(df.index) == 0:
        raise ValueError("Input DataFrame must have a valid index")
    
    features = pd.DataFrame(index=df.index)
    
    if "volume" not in df.columns:
        logger.warning("Volume column not found. Skipping volume features.")
        return features
    
    # Volume moving average (20-period)
    vol_ma_20 = df["volume"].rolling(window=20, min_periods=1).mean()
    features["feat_volu_ma_20"] = vol_ma_20
    
    # Volume z-score (20-period)
    vol_std_20 = df["volume"].rolling(window=20, min_periods=1).std()
    vol_std_20_nonzero = vol_std_20.replace(0.0, np.nan)
    features["feat_volu_zscore_20"] = (df["volume"] - vol_ma_20) / vol_std_20_nonzero
    
    # Volume ratio (5-period MA / 20-period MA)
    vol_ma_5 = df["volume"].rolling(window=5, min_periods=1).mean()
    vol_ma_20_nonzero = vol_ma_20.replace(0.0, np.nan)
    features["feat_volu_ratio_5_20"] = vol_ma_5 / vol_ma_20_nonzero
    
    # Replace inf/-inf with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    return features


def build_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build candle structure features.
    
    Features:
    - Body size and ratios
    - Upper/lower shadow ratios
    - Direction counts (up/down bars)
    
    Args:
        df: DataFrame with OHLCV (must have open, high, low, close)
    
    Returns:
        DataFrame with structure features (prefixed with 'feat_struct_')
    """
    # Ensure we have a consistent index
    if df.index is None or len(df.index) == 0:
        raise ValueError("Input DataFrame must have a valid index")
    
    features = pd.DataFrame(index=df.index)
    
    # Body and shadows
    body = df["close"] - df["open"]
    upper_shadow = df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
    lower_shadow = pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]
    
    # Total range
    total_range = df["high"] - df["low"]
    total_range_nonzero = total_range.replace(0.0, 1e-8)  # Avoid division by zero
    
    # Normalized ratios
    features["feat_struct_body_norm"] = body / total_range_nonzero
    features["feat_struct_upper_ratio"] = upper_shadow / total_range_nonzero
    features["feat_struct_lower_ratio"] = lower_shadow / total_range_nonzero
    
    # Direction indicators
    close_diff = df["close"] - df["close"].shift(1)
    features["feat_struct_dir_1"] = np.sign(close_diff)  # +1, 0, or -1
    
    # Up/down counts (5-period)
    up_count = (df["close"] > df["close"].shift(1)).astype(int)
    down_count = (df["close"] < df["close"].shift(1)).astype(int)
    features["feat_struct_up_count_5"] = up_count.rolling(window=5, min_periods=1).sum()
    features["feat_struct_down_count_5"] = down_count.rolling(window=5, min_periods=1).sum()
    
    # Replace inf/-inf with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    return features

