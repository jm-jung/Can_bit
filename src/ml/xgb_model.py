"""
XGBoost model wrapper for prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.core.config import settings
from src.ml.features import build_ml_dataset


class XGBSignalModel:
    """XGBoost model wrapper for BTC price direction prediction."""

    def __init__(self, model_path: Optional[str | Path] = None):
        """
        Initialize XGBoost model.

        Args:
            model_path: Path to saved model file. If None, uses default from settings.
        """
        if model_path is None:
            model_path = settings.XGB_MODEL_PATH

        self.model_path = Path(model_path)
        self.model: Optional[XGBClassifier] = None

        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                raise ValueError(f"Failed to load model from {model_path}: {e}") from e
        else:
            self.model = None

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from DataFrame (same as training).

        Args:
            df: DataFrame with OHLCV + indicators

        Returns:
            Features DataFrame
        """
        from src.indicators.basic import add_basic_indicators

        df = df.copy()
        df = add_basic_indicators(df)

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

        return features

    def predict_proba_latest(self, df: pd.DataFrame) -> float:
        """
        Predict probability of price going up (next horizon periods).

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)

        Returns:
            Probability of price going up (0.0 to 1.0)
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Train model first or check model path.")

        features = self._extract_features(df)
        last_features = features.iloc[[-1]]

        # Check for NaN
        if last_features.isna().any().any():
            raise ValueError("Features contain NaN values. Ensure indicators are calculated.")

        proba = self.model.predict_proba(last_features)[0]
        # proba[0] = probability of class 0 (down), proba[1] = probability of class 1 (up)
        return float(proba[1])

    def predict_label_latest(self, df: pd.DataFrame) -> int:
        """
        Predict binary label (0 = down, 1 = up).

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)

        Returns:
            Binary label: 1 if price goes up, 0 otherwise
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Train model first or check model path.")

        features = self._extract_features(df)
        last_features = features.iloc[[-1]]

        if last_features.isna().any().any():
            raise ValueError("Features contain NaN values. Ensure indicators are calculated.")

        label = self.model.predict(last_features)[0]
        return int(label)


# Global model instance
_xgb_model: Optional[XGBSignalModel] = None


def get_xgb_model() -> Optional[XGBSignalModel]:
    """Get or create global XGBoost model instance."""
    global _xgb_model
    if _xgb_model is None:
        try:
            _xgb_model = XGBSignalModel()
        except Exception:
            _xgb_model = None
    return _xgb_model

