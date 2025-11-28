"""Feature engineering for ML models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.core.config import settings
from src.events.aggregator import build_event_feature_frame
from src.indicators.basic import add_basic_indicators
from src.services.ohlcv_service import load_ohlcv_df

logger = logging.getLogger(__name__)


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
        event_lookback_minutes: Lookback window for event aggregation (default: from settings)
        debug_inspect: Enable debug logging
        debug_logger: Logger instance for debug output
    
    Returns:
        DataFrame with features:
        - Technical features: close, high, low, volume, ema_20, sma_20, rsi_14, etc.
        - Event features (if use_events=True): event_count_*, event_sentiment_*, etc.
        - Index: timestamp (same as input df)
        - Column order: technical features first, then event features
    """
    if "timestamp" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must include timestamp column or have timestamp index.")

    df = df.copy()
    _log_feature_debug("[IN]", df, debug_inspect=debug_inspect, debug_logger=debug_logger)
    df = add_basic_indicators(df)
    _log_feature_debug("[TECH]", df, debug_inspect=debug_inspect, debug_logger=debug_logger)

    tech_features = build_technical_feature_df(
        df,
        debug_inspect=debug_inspect,
        debug_logger=debug_logger,
    )

    use_events = settings.EVENTS_ENABLED if use_events is None else use_events
    feature_df = tech_features
    
    if use_events:
        # Use v1 event feature builder: build_event_feature_frame
        # This function loads raw events and computes features aligned to OHLCV
        event_df = build_event_feature_frame(
            ohlcv_df=df,
            symbol=symbol,
            timeframe=timeframe,
            lookback_minutes=event_lookback_minutes,
        )
        
        # Merge technical and event features
        # Ensure both have same index
        if isinstance(feature_df.index, pd.DatetimeIndex) and isinstance(event_df.index, pd.DatetimeIndex):
            feature_df = pd.concat([tech_features, event_df], axis=1)
        else:
            # Fallback: reset index and merge
            tech_reset = tech_features.reset_index() if "timestamp" not in tech_features.columns else tech_features
            event_reset = event_df.reset_index() if "timestamp" not in event_df.columns else event_df
            if "timestamp" in tech_reset.columns and "timestamp" in event_reset.columns:
                feature_df = pd.merge(tech_reset, event_reset, on="timestamp", how="left", suffixes=("", "_event"))
                feature_df = feature_df.set_index("timestamp")
            else:
                feature_df = pd.concat([tech_features, event_df], axis=1)
        
        # Fill NaN in event columns with 0
        event_cols = [col for col in feature_df.columns if col.startswith("event_")]
        if event_cols:
            feature_df[event_cols] = feature_df[event_cols].fillna(0.0)
        
        _log_feature_debug("[EVENT]", feature_df, debug_inspect=debug_inspect, debug_logger=debug_logger)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    _log_feature_debug("[POST]", feature_df, debug_inspect=debug_inspect, debug_logger=debug_logger)
    return feature_df


def build_ml_dataset(
    df: pd.DataFrame,
    horizon: int = 5,
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    use_events: bool | None = None,
    event_lookback_minutes: int | None = None,
    debug_inspect: bool = False,
    debug_logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build ML dataset with features and labels.
    
    Args:
        df: OHLCV DataFrame with timestamp column
        horizon: Number of periods ahead to predict
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "1m")
        use_events: Whether to include event features (default: from settings)
        event_lookback_minutes: Lookback window for event aggregation (default: from settings)
        debug_inspect: Enable debug logging
        debug_logger: Logger instance for debug output
    
    Returns:
        Tuple of (X: features DataFrame, y: labels Series)
    """
    df = df.copy()
    features = build_feature_frame(
        df,
        symbol=symbol,
        timeframe=timeframe,
        use_events=use_events,
        event_lookback_minutes=event_lookback_minutes,
        debug_inspect=debug_inspect,
        debug_logger=debug_logger,
    )

    # Label definition:
    # - Calculate future return: r = (price_{t+horizon} / price_t - 1)
    # - Label = 1 if r > 0 (price goes up), Label = 0 otherwise (price goes down or stays same)
    # - This is a binary classification task: predict direction of price movement
    # - Horizon: number of periods (e.g., 5 for 5-minute ahead prediction on 1m data)
    # - Note: No TP/SL thresholds here; simple directional prediction
    # - Holding period: determined by signal generation logic in backtest (entry/exit based on threshold)
    df["return_next_horizon"] = df["close"].shift(-horizon) / df["close"] - 1
    df["y"] = (df["return_next_horizon"] > 0).astype(int)

    valid_mask = ~(features.isna().any(axis=1) | df["y"].isna())
    X = features.loc[valid_mask].copy()
    y = df.loc[valid_mask, "y"].copy()
    if debug_inspect and debug_logger is not None:
        try:
            debug_logger.info("[DEBUG][FEATURE][OUT] final feature_df shape=%s", str(features.shape))
            debug_logger.info("[DEBUG][FEATURE][OUT] columns=%s", list(features.columns))
            debug_logger.info("[DEBUG][FEATURE][OUT] describe():\n%s", features.describe().to_string())
        except Exception as e:
            debug_logger.warning("[DEBUG][FEATURE][OUT] failed to log feature_df: %s", e)

    return X, y


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

