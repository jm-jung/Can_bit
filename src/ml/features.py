"""
Feature engineering for ML models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.indicators.basic import add_basic_indicators
from src.services.ohlcv_service import load_ohlcv_df


def build_ml_dataset(df: pd.DataFrame, horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build ML dataset with features and labels.

    Args:
        df: DataFrame with OHLCV data (must include timestamp, open, high, low, close, volume)
        horizon: Number of periods ahead to predict (default: 5 minutes)

    Returns:
        Tuple of (X: features DataFrame, y: binary labels Series)
    """
    df = df.copy()

    # Add indicators
    df = add_basic_indicators(df)

    # Calculate future return (label)
    df["return_next_horizon"] = df["close"].shift(-horizon) / df["close"] - 1
    df["y"] = (df["return_next_horizon"] > 0).astype(int)

    # Feature engineering
    features = pd.DataFrame(index=df.index)

    # Basic OHLCV features
    features["close"] = df["close"]
    features["high"] = df["high"]
    features["low"] = df["low"]
    features["volume"] = df["volume"]

    # Price differences
    features["high_low"] = df["high"] - df["low"]
    features["close_open"] = df["close"] - df["open"]
    features["close_open_pct"] = (df["close"] / df["open"]) - 1

    # Indicators
    features["ema_20"] = df["ema_20"]
    features["sma_20"] = df["sma_20"]
    features["rsi_14"] = df["rsi_14"]

    # Rolling statistics
    features["rolling_std_20"] = df["close"].rolling(window=20, min_periods=1).std()
    features["rolling_mean_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    # Price relative to moving averages
    features["close_ema_ratio"] = df["close"] / df["ema_20"]
    features["close_sma_ratio"] = df["close"] / df["sma_20"]

    # Remove rows with NaN (especially at the end due to horizon shift)
    valid_mask = ~(features.isna().any(axis=1) | df["y"].isna())
    X = features[valid_mask].copy()
    y = df.loc[valid_mask, "y"].copy()

    return X, y


def build_ml_dataset_from_csv(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build ML dataset from CSV file.

    Args:
        csv_path: Path to OHLCV CSV file

    Returns:
        Tuple of (X: features DataFrame, y: binary labels Series)
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return build_ml_dataset(df)

