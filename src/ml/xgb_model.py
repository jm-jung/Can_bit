"""
XGBoost model wrapper for prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import logging

from src.core.config import settings
from src.features.ml_feature_config import MLFeatureConfig
from src.ml.features import build_feature_frame
from src.ml.scalers import load_scaler, apply_scaler, ScalerType

logger = logging.getLogger(__name__)


class XGBSignalModel:
    """XGBoost model wrapper for BTC price direction prediction.
    
    Supports both single model (backward compatibility) and separate LONG/SHORT models.
    """

    def __init__(
        self, 
        model_path: Optional[str | Path] = None,
        long_model_path: Optional[str | Path] = None,
        short_model_path: Optional[str | Path] = None,
        symbol: str | None = None,
        timeframe: str | None = None,
    ):
        """
        Initialize XGBoost model(s).

        Args:
            model_path: Path to saved model file (for backward compatibility).
                       If None and long_model_path/short_model_path are not provided,
                       uses default from settings.
            long_model_path: Path to LONG model file. If provided, loads separate LONG model.
            short_model_path: Path to SHORT model file. If provided, loads separate SHORT model.
            symbol: Trading symbol (e.g., "BTCUSDT") for auto-detecting model paths
            timeframe: Timeframe (e.g., "1m") for auto-detecting model paths
        """
        self.model_path = Path(model_path) if model_path else None
        self.long_model_path = Path(long_model_path) if long_model_path else None
        self.short_model_path = Path(short_model_path) if short_model_path else None
        self.symbol = symbol
        self.timeframe = timeframe
        
        self.model: Optional[XGBClassifier] = None  # For backward compatibility
        self.long_model: Optional[XGBClassifier] = None
        self.short_model: Optional[XGBClassifier] = None
        self.feature_columns: Optional[list[str]] = None  # Step D: Store feature order from metadata
        # Phase E: Scaler support
        self.scaler = None
        self.scaler_type: ScalerType | None = None
        
        # Determine if we're using separate LONG/SHORT models
        use_separate_models = (long_model_path is not None) or (short_model_path is not None)
        
        if not use_separate_models and model_path is None:
            # Try to auto-detect LONG/SHORT models based on symbol/timeframe
            if symbol and timeframe:
                model_dir = Path(settings.XGB_MODEL_PATH).parent
                long_path = model_dir / f"ml_xgb_long_{symbol}_{timeframe}.pkl"
                short_path = model_dir / f"ml_xgb_short_{symbol}_{timeframe}.pkl"
                
                if long_path.exists() and short_path.exists():
                    use_separate_models = True
                    self.long_model_path = long_path
                    self.short_model_path = short_path
                    logger.info(
                        f"[XGB Model] Auto-detected separate models: "
                        f"LONG={long_path}, SHORT={short_path}"
                    )
        
        if use_separate_models:
            # Load separate LONG and SHORT models
            if self.long_model_path and self.long_model_path.exists():
                try:
                    self.long_model = joblib.load(self.long_model_path)
                    logger.info(f"[XGB Model] Loaded LONG model from {self.long_model_path}")
                except Exception as e:
                    raise ValueError(f"Failed to load LONG model from {self.long_model_path}: {e}") from e
            else:
                raise ValueError(f"LONG model path not found: {self.long_model_path}")
            
            if self.short_model_path and self.short_model_path.exists():
                try:
                    self.short_model = joblib.load(self.short_model_path)
                    logger.info(f"[XGB Model] Loaded SHORT model from {self.short_model_path}")
                except Exception as e:
                    raise ValueError(f"Failed to load SHORT model from {self.short_model_path}: {e}") from e
            else:
                raise ValueError(f"SHORT model path not found: {self.short_model_path}")
            
            # For backward compatibility, set model to long_model
            self.model = self.long_model
            
            # Step D: Try to load feature_columns from metadata
            # Phase E: Also load scaler info
            meta_path = self.long_model_path.with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        meta_data = json.load(f)
                    if "feature_columns" in meta_data:
                        self.feature_columns = meta_data["feature_columns"]
                        logger.info(f"[XGB Model] Loaded feature_columns from metadata: {len(self.feature_columns)} features")
                    
                    # Phase E: Load scaler if present
                    scaler_type = meta_data.get("scaler_type", "none")
                    scaler_path_str = meta_data.get("scaler_path")
                    if scaler_type != "none" and scaler_path_str:
                        scaler_path = Path(scaler_path_str)
                        if scaler_path.exists():
                            self.scaler = load_scaler(scaler_path, scaler_type)
                            self.scaler_type = scaler_type
                            logger.info(f"[XGB Model] Loaded {scaler_type} scaler from {scaler_path}")
                        else:
                            logger.warning(f"[XGB Model] Scaler path not found: {scaler_path}")
                    else:
                        logger.debug(f"[XGB Model] No scaler configured (scaler_type={scaler_type})")
                except Exception as e:
                    logger.warning(f"[XGB Model] Failed to load metadata: {e}")
        else:
            # Load single model (backward compatibility)
            if model_path is None:
                model_path = settings.XGB_MODEL_PATH
            self.model_path = Path(model_path)
            
            if self.model_path.exists():
                try:
                    self.model = joblib.load(self.model_path)
                    self.long_model = self.model  # For compatibility
                    logger.info(f"[XGB Model] Loaded single model from {self.model_path}")
                except Exception as e:
                    raise ValueError(f"Failed to load model from {model_path}: {e}") from e
            else:
                self.model = None
                logger.warning(f"[XGB Model] Model file not found: {model_path}")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        if self.long_model is not None and self.short_model is not None:
            return True
        return self.model is not None
    
    def has_separate_models(self) -> bool:
        """Check if separate LONG/SHORT models are loaded."""
        return self.long_model is not None and self.short_model is not None

    def _extract_features(
        self, 
        df: pd.DataFrame, 
        symbol: str | None = None, 
        timeframe: str | None = None,
        feature_config: Optional[MLFeatureConfig] = None,
    ) -> pd.DataFrame:
        """
        Extract features from DataFrame (same as training).

        Args:
            df: DataFrame with OHLCV + indicators
            symbol: Trading symbol (default: from settings)
            timeframe: Timeframe (default: from settings)
            feature_config: MLFeatureConfig instance (default: "base" preset for backward compatibility)

        Returns:
            Features DataFrame
        """
        if symbol is None:
            symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
        if timeframe is None:
            timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
        
        # Use base preset by default for backward compatibility
        if feature_config is None:
            feature_config = MLFeatureConfig.from_preset("base")
        
        features = build_feature_frame(
            df,
            symbol=symbol,
            timeframe=timeframe,
            use_events=settings.EVENTS_ENABLED,
            feature_config=feature_config,
        )
        return features

    def predict_proba_latest(
        self, 
        df: pd.DataFrame, 
        symbol: str | None = None, 
        timeframe: str | None = None,
        return_both: bool = False,
        hold_margin: float | None = None,
    ) -> float | tuple[float, float] | tuple[float, float, str]:
        """
        Predict probability of price going up (next horizon periods).

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)
            symbol: Trading symbol (default: from settings)
            timeframe: Timeframe (default: from settings)
            return_both: If True and separate models are loaded, returns (proba_long, proba_short).
                        Otherwise returns single proba (backward compatible).
            hold_margin: If provided, returns (proba_long, proba_short, signal) where signal can be 'HOLD'
                        when abs(proba_long - 0.5) < hold_margin. Default: None (disabled)

        Returns:
            If hold_margin is provided and separate models: (proba_long, proba_short, signal)
            If return_both=True and separate models: (proba_long, proba_short)
            Otherwise: Probability of price going up (0.0 to 1.0) - backward compatible
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Train model first or check model path.")

        features = self._extract_features(df, symbol=symbol, timeframe=timeframe)
        last_features = features.iloc[[-1]]

        # Check for NaN in critical columns (non-event features)
        critical_cols = [c for c in last_features.columns if not c.startswith("event_")]
        if last_features[critical_cols].isna().any().any():
            raise ValueError("Features contain NaN values. Ensure indicators are calculated.")

        # Step D: Use feature_columns from metadata if available, otherwise fall back to model's feature names
        if self.feature_columns is not None:
            # Use feature order from metadata (most reliable)
            model_feature_names = self.feature_columns
            logger.debug(f"[XGB Inference] Using feature_columns from metadata: {len(model_feature_names)} features")
        else:
            # Fallback: get feature names from model
            ref_model = self.long_model if self.long_model is not None else self.model
            model_feature_names = ref_model.get_booster().feature_names
            if model_feature_names is None:
                # Fallback: use feature_importances_ to get feature count
                model_feature_names = [f"f{i}" for i in range(len(ref_model.feature_importances_))]
            logger.debug(f"[XGB Inference] Using feature names from model: {len(model_feature_names)} features")
        
        # Ensure all model-expected features are present
        missing_features = set(model_feature_names) - set(last_features.columns)
        if missing_features:
            # Fill missing features with 0 (typically event features)
            for feat in missing_features:
                last_features[feat] = 0.0
        
        # Reorder columns to match model's expected order
        if model_feature_names:
            last_features = last_features.reindex(columns=model_feature_names, fill_value=0.0)
        
        # Phase E: Apply scaler if available
        if self.scaler is not None:
            last_features = apply_scaler(last_features, self.scaler, fit=False)
            logger.debug(f"[XGB Inference] Applied {self.scaler_type} scaler")
        
        # Debug logging for feature alignment
        logger.debug(f"[XGB Inference] Model expects {len(model_feature_names)} features")
        logger.debug(f"[XGB Inference] Provided features: {list(last_features.columns)}")
        if settings.EVENTS_ENABLED:
            event_cols = [c for c in last_features.columns if c.startswith("event_")]
            logger.debug(f"[XGB Inference] Event features in input: {event_cols}")
        if missing_features:
            logger.debug(f"[XGB Inference] Missing features (filled with 0): {missing_features}")

        if self.has_separate_models() and (return_both or hold_margin is not None):
            # Predict with both LONG and SHORT models
            proba_long = self.long_model.predict_proba(last_features)[0]
            proba_short = self.short_model.predict_proba(last_features)[0]
            proba_long_val = float(proba_long[1])
            proba_short_val = float(proba_short[1])
            
            # Check for HOLD zone if hold_margin is provided
            if hold_margin is not None:
                # HOLD zone: when both probabilities are close to 0.5 (uncertain)
                # Use margin: abs(proba_long - 0.5) < hold_margin → HOLD
                if abs(proba_long_val - 0.5) < hold_margin:
                    signal = "HOLD"
                else:
                    signal = "LONG" if proba_long_val > 0.5 else "SHORT"
                return (proba_long_val, proba_short_val, signal)
            else:
                # return_both=True but no hold_margin
                return (proba_long_val, proba_short_val)
        else:
            # Use single model or long_model for backward compatibility
            model_to_use = self.long_model if self.long_model is not None else self.model
            proba = model_to_use.predict_proba(last_features)[0]
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


class MLXGBModel:
    """
    Phase E: ML XGBoost model loader with metadata and scaler support.
    
    Loads LONG/SHORT models, scaler, and metadata based on strategy/symbol/timeframe/feature_preset.
    """
    
    def __init__(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        feature_preset: str = "extended_safe",
    ):
        """
        Initialize ML XGBoost model loader.
        
        Args:
            strategy: Strategy identifier (e.g., "ml_xgb")
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "5m")
            feature_preset: Feature preset (e.g., "extended_safe", "base")
        """
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.feature_preset = feature_preset
        
        # Build model paths
        model_dir = Path(settings.XGB_MODEL_PATH).parent
        base_name = f"{strategy}_long_{symbol}_{timeframe}_{feature_preset}"
        self.long_model_path = model_dir / f"{base_name}.pkl"
        self.short_model_path = model_dir / f"{strategy}_short_{symbol}_{timeframe}_{feature_preset}.pkl"
        self.meta_path = model_dir / f"{base_name}.meta.json"
        
        # Load metadata
        self.meta_data = {}
        self.scaler_type: ScalerType = "none"
        self.scaler_path: Path | None = None
        self.scaler = None
        self.label_mode: str = "classification"
        self.label_thresholds: dict = {}
        self.feature_columns: list[str] | None = None
        
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r") as f:
                    self.meta_data = json.load(f)
                
                self.scaler_type = self.meta_data.get("scaler_type", "none")
                scaler_path_str = self.meta_data.get("scaler_path")
                if scaler_path_str:
                    self.scaler_path = Path(scaler_path_str)
                else:
                    # Fallback: assume scaler is in same directory with .scaler.pkl suffix
                    self.scaler_path = self.long_model_path.with_suffix(".scaler.pkl")
                
                self.label_mode = self.meta_data.get("label_mode", "classification")
                self.label_thresholds = self.meta_data.get("label_thresholds", {})
                self.feature_columns = self.meta_data.get("feature_columns")
                
                logger.info(
                    f"[XGB Model] Loaded metadata from {self.meta_path}: "
                    f"scaler_type={self.scaler_type}, label_mode={self.label_mode}, "
                    f"timeframe={self.meta_data.get('timeframe', 'unknown')}"
                )
            except Exception as e:
                logger.warning(f"[XGB Model] Failed to load metadata from {self.meta_path}: {e}")
        
        # Load models
        if not self.long_model_path.exists():
            raise FileNotFoundError(f"LONG model not found: {self.long_model_path}")
        if not self.short_model_path.exists():
            raise FileNotFoundError(f"SHORT model not found: {self.short_model_path}")
        
        try:
            self.long_model = joblib.load(self.long_model_path)
            logger.info(f"[XGB Model] Loaded LONG model from {self.long_model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load LONG model from {self.long_model_path}: {e}") from e
        
        try:
            self.short_model = joblib.load(self.short_model_path)
            logger.info(f"[XGB Model] Loaded SHORT model from {self.short_model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load SHORT model from {self.short_model_path}: {e}") from e
        
        # Load scaler if configured
        if self.scaler_type != "none" and self.scaler_path and self.scaler_path.exists():
            try:
                self.scaler = load_scaler(self.scaler_path, self.scaler_type)
                logger.info(f"[XGB Model] Loaded {self.scaler_type} scaler from {self.scaler_path}")
            except Exception as e:
                logger.warning(f"[XGB Model] Failed to load scaler from {self.scaler_path}: {e}")
                self.scaler = None
        else:
            self.scaler = None
            if self.scaler_type != "none":
                logger.debug(f"[XGB Model] Scaler type is {self.scaler_type} but scaler file not found")
        
        logger.info(
            f"[XGB Model] Loaded ML models for {symbol}-{timeframe} (preset={feature_preset}): "
            f"long={self.long_model_path.name}, short={self.short_model_path.name}, "
            f"scaler={self.scaler_path.name if self.scaler_path and self.scaler_path.exists() else 'None'}, "
            f"label_mode={self.label_mode}"
        )
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self.long_model is not None and self.short_model is not None
    
    def predict_proba_long(self, X_raw: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict LONG probability for raw features.
        
        Args:
            X_raw: Raw feature array/DataFrame (not scaled)
        
        Returns:
            Array of probabilities (class 1 probabilities)
        """
        # Apply scaler if available
        if self.scaler is not None:
            X_scaled = apply_scaler(X_raw, self.scaler, fit=False)
        else:
            X_scaled = X_raw
        
        # Ensure feature order matches training
        if isinstance(X_scaled, pd.DataFrame) and self.feature_columns is not None:
            # Reorder columns to match training order
            missing_cols = set(self.feature_columns) - set(X_scaled.columns)
            if missing_cols:
                # Add missing columns with zeros
                for col in missing_cols:
                    X_scaled[col] = 0.0
            X_scaled = X_scaled.reindex(columns=self.feature_columns, fill_value=0.0)
        
        proba = self.long_model.predict_proba(X_scaled)
        return proba[:, 1] if proba.ndim > 1 else proba
    
    def predict_proba_short(self, X_raw: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict SHORT probability for raw features.
        
        Args:
            X_raw: Raw feature array/DataFrame (not scaled)
        
        Returns:
            Array of probabilities (class 1 probabilities)
        """
        # Apply scaler if available
        if self.scaler is not None:
            X_scaled = apply_scaler(X_raw, self.scaler, fit=False)
        else:
            X_scaled = X_raw
        
        # Ensure feature order matches training
        if isinstance(X_scaled, pd.DataFrame) and self.feature_columns is not None:
            # Reorder columns to match training order
            missing_cols = set(self.feature_columns) - set(X_scaled.columns)
            if missing_cols:
                # Add missing columns with zeros
                for col in missing_cols:
                    X_scaled[col] = 0.0
            X_scaled = X_scaled.reindex(columns=self.feature_columns, fill_value=0.0)
        
        proba = self.short_model.predict_proba(X_scaled)
        return proba[:, 1] if proba.ndim > 1 else proba


def load_legacy_single_model(
    model_path: str | Path | None = None,
) -> XGBSignalModel:
    """
    Load legacy single XGBoost model (backward compatibility).
    
    Args:
        model_path: Path to legacy model file (default: from settings)
    
    Returns:
        XGBSignalModel instance
    
    WARNING: This function loads timeframe-agnostic legacy models.
    Consider migrating to MLXGBModel with metadata/scaler support.
    """
    if model_path is None:
        model_path = settings.XGB_MODEL_PATH
    
    logger.warning(
        f"[XGB Model] WARNING: Using legacy single model {model_path} (timeframe-agnostic). "
        "Consider migrating to MLXGBModel with meta/scaler."
    )
    
    return XGBSignalModel(model_path=model_path)


# Global model instance
_xgb_model: Optional[XGBSignalModel] = None


def get_xgb_model(
    symbol: str | None = None,
    timeframe: str | None = None,
) -> Optional[XGBSignalModel]:
    """
    Get or create global XGBoost model instance.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT") for auto-detecting separate LONG/SHORT models
        timeframe: Timeframe (e.g., "1m") for auto-detecting separate LONG/SHORT models
    
    Returns:
        XGBSignalModel instance (with separate LONG/SHORT models if available)
    """
    global _xgb_model
    if _xgb_model is None:
        try:
            # Try to load separate LONG/SHORT models if symbol/timeframe are provided
            if symbol and timeframe:
                _xgb_model = XGBSignalModel(symbol=symbol, timeframe=timeframe)
            else:
                _xgb_model = XGBSignalModel()
        except Exception:
            _xgb_model = None
    return _xgb_model

