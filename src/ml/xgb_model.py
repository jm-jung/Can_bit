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

import logging

from src.core.config import settings
from src.ml.features import build_feature_frame

logger = logging.getLogger(__name__)


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

    def _extract_features(self, df: pd.DataFrame, symbol: str | None = None, timeframe: str | None = None) -> pd.DataFrame:
        """
        Extract features from DataFrame (same as training).

        Args:
            df: DataFrame with OHLCV + indicators
            symbol: Trading symbol (default: from settings)
            timeframe: Timeframe (default: from settings)

        Returns:
            Features DataFrame
        """
        if symbol is None:
            symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
        if timeframe is None:
            timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
        
        features = build_feature_frame(
            df,
            symbol=symbol,
            timeframe=timeframe,
            use_events=settings.EVENTS_ENABLED,
        )
        return features

    def predict_proba_latest(self, df: pd.DataFrame, symbol: str | None = None, timeframe: str | None = None) -> float:
        """
        Predict probability of price going up (next horizon periods).

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)
            symbol: Trading symbol (default: from settings)
            timeframe: Timeframe (default: from settings)

        Returns:
            Probability of price going up (0.0 to 1.0)
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Train model first or check model path.")

        features = self._extract_features(df, symbol=symbol, timeframe=timeframe)
        last_features = features.iloc[[-1]]

        # Check for NaN in critical columns (non-event features)
        critical_cols = [c for c in last_features.columns if not c.startswith("event_")]
        if last_features[critical_cols].isna().any().any():
            raise ValueError("Features contain NaN values. Ensure indicators are calculated.")

        # Align features with model's expected feature names
        # If model expects event features but they're missing, fill with 0
        model_feature_names = self.model.get_booster().feature_names
        if model_feature_names is None:
            # Fallback: use feature_importances_ to get feature count
            model_feature_names = [f"f{i}" for i in range(len(self.model.feature_importances_))]
        
        # Ensure all model-expected features are present
        missing_features = set(model_feature_names) - set(last_features.columns)
        if missing_features:
            # Fill missing features with 0 (typically event features)
            for feat in missing_features:
                last_features[feat] = 0.0
        
        # Reorder columns to match model's expected order
        if model_feature_names:
            last_features = last_features.reindex(columns=model_feature_names, fill_value=0.0)
        
        # Debug logging for feature alignment
        logger.debug(f"[XGB Inference] Model expects {len(model_feature_names)} features")
        logger.debug(f"[XGB Inference] Provided features: {list(last_features.columns)}")
        if settings.EVENTS_ENABLED:
            event_cols = [c for c in last_features.columns if c.startswith("event_")]
            logger.debug(f"[XGB Inference] Event features in input: {event_cols}")
        if missing_features:
            logger.debug(f"[XGB Inference] Missing features (filled with 0): {missing_features}")

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

