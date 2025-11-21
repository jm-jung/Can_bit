"""Feature engineering for ML models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.core.config import settings
from src.events.dataset import (
    build_event_feature_df,
    load_event_features,
    merge_price_and_event_features,
)
from src.indicators.basic import add_basic_indicators
from src.services.ohlcv_service import load_ohlcv_df

logger = logging.getLogger(__name__)


def build_feature_frame(
    df: pd.DataFrame,
    *,
    use_events: bool | None = None,
    event_df: pd.DataFrame | None = None,
    event_lookback_minutes: int | None = None,
    refresh_events: bool = False,
) -> pd.DataFrame:
    """
    OHLCV 기반 피처 프레임 생성 (이벤트 피처 optional).
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must include timestamp column.")

    df = df.copy()
    df = add_basic_indicators(df)
    timestamps = pd.to_datetime(df["timestamp"])

    features = pd.DataFrame(index=timestamps)
    features["close"] = df["close"]
    features["high"] = df["high"]
    features["low"] = df["low"]
    features["volume"] = df["volume"]
    features["high_low"] = df["high"] - df["low"]
    features["close_open"] = df["close"] - df["open"]
    features["close_open_pct"] = (df["close"] / df["open"]) - 1
    features["ema_20"] = df["ema_20"]
    features["sma_20"] = df["sma_20"]
    features["rsi_14"] = df["rsi_14"]
    features["rolling_std_20"] = df["close"].rolling(window=20, min_periods=1).std()
    features["rolling_mean_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    features["close_ema_ratio"] = df["close"] / df["ema_20"]
    features["close_sma_ratio"] = df["close"] / df["sma_20"]

    use_events = settings.EVENTS_ENABLED if use_events is None else use_events
    if use_events:
        if event_df is None and not refresh_events:
            cached = load_event_features()
            event_df = cached if not cached.empty else None
            if event_df is None:
                logger.info("캐시된 이벤트 피처가 없어 새로 생성합니다.")

        if event_df is None or refresh_events:
            event_df = build_event_feature_df(
                df,
                lookback_minutes=event_lookback_minutes,
            )

        features = merge_price_and_event_features(features, event_df)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def build_ml_dataset(
    df: pd.DataFrame,
    horizon: int = 5,
    *,
    use_events: bool | None = None,
    event_df: pd.DataFrame | None = None,
    event_lookback_minutes: int | None = None,
    refresh_events: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build ML dataset with features and labels.
    """
    df = df.copy()
    features = build_feature_frame(
        df,
        use_events=use_events,
        event_df=event_df,
        event_lookback_minutes=event_lookback_minutes,
        refresh_events=refresh_events,
    )

    df["return_next_horizon"] = df["close"].shift(-horizon) / df["close"] - 1
    df["y"] = (df["return_next_horizon"] > 0).astype(int)

    valid_mask = ~(features.isna().any(axis=1) | df["y"].isna())
    X = features.loc[valid_mask].copy()
    y = df.loc[valid_mask, "y"].copy()
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

