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
    features["atr_14"] = features["feat_vol_atr_14"]
    features["true_range"] = features["feat_vol_tr"]
    
    # Normalized ATR
    close_nonzero = df["close"].replace(0.0, np.nan)
    features["feat_vol_atr_norm"] = features["feat_vol_atr_14"] / close_nonzero
    
    # Return volatility (20-period rolling std of log returns)
    close_shifted = df["close"].shift(1).replace(0.0, np.nan)
    log_return = np.log(df["close"] / close_shifted)
    features["feat_vol_volatility_20"] = log_return.rolling(window=20, min_periods=1).std()
    
    # High-low range (normalized)
    features["feat_vol_range_norm"] = (df["high"] - df["low"]) / close_nonzero
    
    # Spec: range_pct = (high - low) / close, range_ma_ratio = range_pct / rolling_mean(range_pct, 20)
    range_pct = (df["high"] - df["low"]) / close_nonzero
    features["range_pct"] = range_pct
    range_ma_20 = range_pct.rolling(window=20, min_periods=1).mean()
    range_ma_20_nonzero = range_ma_20.replace(0.0, np.nan)
    features["range_ma_ratio"] = (range_pct / range_ma_20_nonzero).fillna(1.0)
    
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
    
    # Spec: volume_zscore_50, volume_ma_ratio = volume / rolling_mean(volume, 20), volume_spike_flag
    vol_ma_50 = df["volume"].rolling(window=50, min_periods=1).mean()
    vol_std_50 = df["volume"].rolling(window=50, min_periods=1).std()
    vol_std_50_nonzero = vol_std_50.replace(0.0, np.nan)
    features["volume_zscore_50"] = (df["volume"] - vol_ma_50) / vol_std_50_nonzero
    features["volume_ma_ratio"] = df["volume"] / vol_ma_20_nonzero
    features["volume_zscore_20"] = features["feat_volu_zscore_20"]
    z20 = features["feat_volu_zscore_20"].fillna(0.0)
    features["volume_spike_flag"] = (z20 > 2).astype(float)
    
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
    body_size = body.abs()
    upper_shadow = df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
    lower_shadow = pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]
    
    # Total range
    total_range = df["high"] - df["low"]
    total_range_nonzero = total_range.replace(0.0, 1e-8)  # Avoid division by zero
    
    # Normalized ratios
    features["feat_struct_body_norm"] = body / total_range_nonzero
    features["feat_struct_upper_ratio"] = upper_shadow / total_range_nonzero
    features["feat_struct_lower_ratio"] = lower_shadow / total_range_nonzero
    
    # Spec names: body_size, body_ratio, upper_wick, lower_wick, ratios, bullish_flag, bearish_flag
    features["body_size"] = body_size
    features["body_ratio"] = body_size / total_range_nonzero
    features["upper_wick"] = upper_shadow
    features["lower_wick"] = lower_shadow
    features["upper_wick_ratio"] = upper_shadow / total_range_nonzero
    features["lower_wick_ratio"] = lower_shadow / total_range_nonzero
    features["bullish_flag"] = (df["close"] > df["open"]).astype(float)
    features["bearish_flag"] = (df["close"] < df["open"]).astype(float)
    
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


def build_realized_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build realized volatility features (rolling std of close returns over 12/24/48 bars).
    Features: realized_vol_12, realized_vol_24, realized_vol_48, rv_ratio_short.
    """
    if df.index is None or len(df.index) == 0:
        raise ValueError("Input DataFrame must have a valid index")
    features = pd.DataFrame(index=df.index)
    close = df["close"]
    close_prev = close.shift(1).replace(0.0, np.nan)
    log_ret = np.log(close / close_prev)
    features["realized_vol_12"] = log_ret.rolling(window=12, min_periods=1).std()
    features["realized_vol_24"] = log_ret.rolling(window=24, min_periods=1).std()
    features["realized_vol_48"] = log_ret.rolling(window=48, min_periods=1).std()
    rv48_nonzero = features["realized_vol_48"].replace(0.0, np.nan)
    features["rv_ratio_short"] = (features["realized_vol_12"] / rv48_nonzero).fillna(1.0)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def build_multitimeframe_trend_features(
    df: pd.DataFrame,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    """
    Multi-timeframe trend (15m, 1h): ema20_tf, close_ema_ratio_tf, trend_flag_tf.
    Forward-fill to base TF; no future leakage.
    Handles duplicate index by taking last value per timestamp for resample.
    """
    if df.index is None or len(df.index) == 0:
        raise ValueError("Input DataFrame must have a valid index")
    if "close" not in df.columns:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    if not isinstance(df.index, pd.DatetimeIndex):
        return out
    # Deduplicate index so resample/reindex don't raise (keep last per timestamp)
    close_series = df["close"]
    if close_series.index.duplicated().any():
        close_series = close_series.groupby(level=0).last()
    close = close_series.sort_index()
    for label, agg_bars in [("15m", 3), ("1h", 12)]:
        period = f"{agg_bars * timeframe_minutes}min"
        close_htf = close.resample(period, label="right", closed="right").last()
        close_htf = close_htf.dropna()
        if len(close_htf) < 20:
            continue
        ema20 = close_htf.ewm(span=20, adjust=False).mean()
        close_ema_ratio = (close_htf / ema20).replace([0, np.inf, -np.inf], np.nan)
        trend_flag = (close_htf > ema20).astype(float)
        # Map back to original df.index (may have duplicates; ffill fills them)
        ema20_reindex = ema20.reindex(df.index).ffill()
        close_ema_ratio_reindex = close_ema_ratio.reindex(df.index).ffill()
        trend_flag_reindex = trend_flag.reindex(df.index).ffill()
        out[f"ema20_tf_{label}"] = ema20_reindex
        out[f"close_ema_ratio_tf_{label}"] = close_ema_ratio_reindex
        out[f"trend_flag_tf_{label}"] = trend_flag_reindex
    out = out.fillna(0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

