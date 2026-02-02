"""
TCN (Temporal Convolutional Network) model wrapper for prediction.

This module provides a wrapper class for TCN model inference,
similar to LSTMAttnSignalModel, to ensure compatibility with
the existing backtest and cache generation pipeline.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.core.config import settings
from src.dl.models.tcn import TCNModel
from src.indicators.basic import add_basic_indicators
from src.ml.features import build_feature_frame
from src.dl.data.labels import LstmClassIndex

logger = logging.getLogger(__name__)


def build_normalized_tcn_sequences_full(
    features_df: pd.DataFrame,
    window_size: int,
    feature_cols: list[str],
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Build the full 3D TCN input tensor for all valid time indices.
    
    This function uses the same normalization logic as LSTM-Attn to ensure
    consistency in feature extraction and preprocessing.
    
    Args:
        features_df: Full features DataFrame (must have at least window_size rows)
        window_size: Sequence length (number of timesteps)
        feature_cols: List of feature column names in the correct order
        logger: Optional logger instance (uses module logger if None)
    
    Returns:
        sequences: np.ndarray of shape (num_sequences, window_size, feature_dim), dtype float32
        where num_sequences = len(features_df) - window_size
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Validate input
    if len(features_df) < window_size:
        raise ValueError(
            f"Not enough rows: need at least {window_size}, got {len(features_df)}"
        )
    
    # Ensure feature columns are in the correct order
    features_df = features_df.reindex(columns=feature_cols, fill_value=0.0)
    
    N = len(features_df)
    num_sequences = N - window_size
    feature_dim = len(feature_cols)
    
    logger.info(
        f"[TCN Normalize] Building {num_sequences} sequences from {N} feature rows "
        f"(window_size={window_size}, feature_dim={feature_dim})"
    )
    
    # Step 1: Normalize the entire DataFrame ONCE (same as LSTM-Attn)
    features_normalized = features_df.copy()
    
    for col in feature_cols:
        # Rolling statistics: window=window_size, min_periods=1
        rolling_mean = features_normalized[col].rolling(window=window_size, min_periods=1).mean()
        rolling_std = features_normalized[col].rolling(window=window_size, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero
        
        # Z-score normalization: (X - mean) / std
        features_normalized[col] = (features_normalized[col] - rolling_mean) / rolling_std
    
    # Convert to float32
    for col in feature_cols:
        features_normalized[col] = features_normalized[col].astype(np.float32)
    
    # Step 2: Sanitize NaN/Inf
    if not np.isfinite(features_normalized.to_numpy()).all():
        logger.warning(
            f"[TCN Normalize] Found NaN/Inf in features_normalized. Sanitizing..."
        )
        features_normalized = features_normalized.replace([np.inf, -np.inf], np.nan)
        features_normalized = features_normalized.fillna(0.0)
        
        if not np.isfinite(features_normalized.to_numpy()).all():
            raise ValueError("Non-finite values remain after sanitization.")
    
    # Step 3: Create sequences via sliding windows
    sequences_list: list[np.ndarray] = []
    for i in range(window_size, len(features_normalized)):
        seq = features_normalized.iloc[i - window_size : i][feature_cols].values
        sequences_list.append(seq)
    
    # Step 4: Convert to numpy array
    if sequences_list:
        sequences_array = np.array(sequences_list, dtype=np.float32)
    else:
        sequences_array = np.empty((0, window_size, feature_dim), dtype=np.float32)
    
    # Step 5: Final sanitization check
    if not np.isfinite(sequences_array).all():
        logger.warning(
            f"[TCN Normalize] Found NaN/Inf in sequences_array. Sanitizing..."
        )
        sequences_array[~np.isfinite(sequences_array)] = 0.0
        
        if not np.isfinite(sequences_array).all():
            raise ValueError("Non-finite values remain after sanitization.")
    
    return sequences_array


class TCNSignalModel:
    """TCN model wrapper for BTC price direction prediction."""
    
    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        window_size: int = 60,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize TCN model wrapper.
        
        Args:
            model_path: Path to saved model file. If None, uses default from settings.
            window_size: Sequence length (must match training)
            num_channels: List of channel sizes (must match training)
            kernel_size: Convolution kernel size (must match training)
            dropout: Dropout rate (must match training)
        """
        if model_path is None:
            # Default model path (can be customized via environment variable TCN_MODEL_PATH)
            env_path = os.getenv("TCN_MODEL_PATH")
            if env_path:
                default_path = env_path
                logger.info(
                    f"[TCN] Using model path from environment variable TCN_MODEL_PATH: {default_path}"
                )
            else:
                default_path = "models/tcn_v1.pt"
                logger.info(
                    f"[TCN] Using default model path (TCN_MODEL_PATH not set): {default_path}"
                )
            model_path = Path(default_path)
        else:
            logger.info(f"[TCN] Using model path from constructor argument: {model_path}")
        
        # Allow model_path to be set via environment or settings
        if isinstance(model_path, str):
            model_path = Path(model_path)
        
        self.model_path = Path(model_path)
        logger.info(f"[TCN] Final model path resolved: {self.model_path.resolve()}")
        self.window_size = window_size
        self.num_channels = num_channels if num_channels is not None else [64, 64, 64, 64]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.model: Optional[TCNModel] = None
        self.device: Optional[torch.device] = None
        self.feature_cols: Optional[list[str]] = None
        
        # Cache for sequences
        self._cached_features_df_id: int | None = None
        self._cached_sequences_full: np.ndarray | None = None
        self._cached_window_size: int | None = None
        self._cached_feature_cols: list[str] | None = None
        
        # Initialize device
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            logger.error(f"Failed to initialize device: {e}")
            self.device = torch.device("cpu")
        
        # Model will be loaded lazily when needed
        self.model = None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        if self.model is None:
            self._load_model()
        return self.model is not None
    
    def _load_model(self) -> None:
        """Load model from file (lazy loading)."""
        if not self.model_path.exists():
            logger.warning(
                f"TCN model file not found: {self.model_path.resolve()}. "
                f"Please train the model first or check the path."
            )
            self.model = None
            return
        
        logger.info(f"[TCN] Loading TCN model from: {self.model_path.resolve()}")
        
        # Determine feature dimension
        try:
            if self.feature_cols is None:
                from src.services.ohlcv_service import load_ohlcv_df
                
                df = load_ohlcv_df()
                df = df.sort_values("timestamp").reset_index(drop=True)
                symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
                timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
                
                X_features = build_feature_frame(
                    df,
                    symbol=symbol,
                    timeframe=timeframe,
                    use_events=settings.EVENTS_ENABLED,
                ).dropna()
                self.feature_cols = X_features.columns.tolist()
            
            feature_dim = len(self.feature_cols)
        except Exception as e:
            logger.error(f"Failed to determine feature dimension: {e}")
            self.model = None
            return
        
        # Load model
        try:
            self.model = TCNModel.load_model(
                path=self.model_path,
                input_size=feature_dim,
                num_channels=self.num_channels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                num_classes=3,
                device=self.device,
            )
            logger.info(
                f"[TCN] Successfully loaded TCN model from: {self.model_path.resolve()}"
            )
            logger.info(
                f"[TCN] Model configuration: "
                f"feature_dim={feature_dim}, num_channels={self.num_channels}, "
                f"kernel_size={self.kernel_size}, dropout={self.dropout}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load model from {self.model_path.resolve()}: {type(e).__name__}: {e}"
            )
            self.model = None
    
    def _extract_features(self, df: pd.DataFrame, symbol: str | None = None, timeframe: str | None = None) -> pd.DataFrame:
        """Extract features from DataFrame (same as LSTM-Attn)."""
        df = df.copy()
        
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
        features = features.dropna()
        
        if self.feature_cols is None:
            self.feature_cols = features.columns.tolist()
        
        features = features.reindex(columns=self.feature_cols, fill_value=0.0)
        return features
    
    def _get_or_build_sequences_full(
        self,
        features_df: pd.DataFrame,
        window_size: int,
        feature_cols: list[str],
        logger: logging.Logger,
    ) -> np.ndarray:
        """Return cached sequences or build them."""
        features_df_id = id(features_df)
        
        if (self._cached_features_df_id == features_df_id and
            self._cached_window_size == window_size and
            self._cached_feature_cols == feature_cols):
            logger.debug(
                f"[TCN Cache] Cache HIT: reusing {self._cached_sequences_full.shape[0]} sequences"
            )
            return self._cached_sequences_full
        
        logger.debug(
            f"[TCN Cache] Cache MISS: building sequences for features_df "
            f"(id={features_df_id}, len={len(features_df)}, window_size={window_size})"
        )
        
        sequences = build_normalized_tcn_sequences_full(
            features_df=features_df,
            window_size=window_size,
            feature_cols=feature_cols,
            logger=logger,
        )
        
        self._cached_features_df_id = features_df_id
        self._cached_window_size = window_size
        self._cached_feature_cols = list(feature_cols)
        self._cached_sequences_full = sequences
        
        return sequences
    
    def predict_proba_batch(
        self,
        features: pd.DataFrame,
        symbol: str | None = None,
        timeframe: str | None = None,
        batch_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for multiple sequences.
        
        Args:
            features: DataFrame with extracted features
            symbol: Trading symbol (default: from settings)
            timeframe: Timeframe (default: from settings)
            batch_size: Batch size for model forward passes
        
        Returns:
            Tuple of (proba_long: np.ndarray, proba_short: np.ndarray)
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Train model first or check model path.")
        
        if len(features) < self.window_size:
            raise ValueError(
                f"Not enough features: need at least {self.window_size} rows, got {len(features)}"
            )
        
        # Ensure feature columns match
        if self.feature_cols is None:
            self.feature_cols = features.columns.tolist()
        else:
            features = features.reindex(columns=self.feature_cols, fill_value=0.0)
        
        logger.info(
            f"[TCN Proba][Batch] total_rows={len(features)}, window_size={self.window_size}, "
            f"batch_size={batch_size}"
        )
        
        # Build sequences
        sequences_array = self._get_or_build_sequences_full(
            features_df=features,
            window_size=self.window_size,
            feature_cols=self.feature_cols,
            logger=logger,
        )
        
        num_sequences = sequences_array.shape[0]
        logger.info(
            f"[TCN Proba][Batch] Using {num_sequences} sequences from {len(features)} feature rows"
        )
        
        # Convert to tensor
        all_sequences = torch.from_numpy(sequences_array).to(self.device)
        
        # Batch forward passes
        self.model.eval()
        all_proba_long: list[float] = []
        all_proba_short: list[float] = []
        
        with torch.no_grad():
            for batch_start in range(0, num_sequences, batch_size):
                batch_end = min(batch_start + batch_size, num_sequences)
                batch_sequences = all_sequences[batch_start:batch_end]
                
                # Forward pass (3-class)
                logits = self.model(batch_sequences)  # (batch_size, 3)
                
                # Check for NaN/Inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.error(
                        f"[TCN Proba][Batch] Found NaN/Inf in logits at batch_start={batch_start}"
                    )
                    raise ValueError("NaN/Inf detected in model logits.")
                
                probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, 3)
                
                # Extract LONG and SHORT probabilities
                proba_long_batch = probs[:, LstmClassIndex.LONG].cpu().numpy()
                proba_short_batch = probs[:, LstmClassIndex.SHORT].cpu().numpy()
                
                all_proba_long.extend(proba_long_batch.tolist())
                all_proba_short.extend(proba_short_batch.tolist())
        
        proba_long_arr = np.array(all_proba_long, dtype=np.float32)
        proba_short_arr = np.array(all_proba_short, dtype=np.float32)
        
        logger.info(
            f"[TCN Proba][Batch] mean_proba_long={proba_long_arr.mean():.4f}, "
            f"mean_proba_short={proba_short_arr.mean():.4f}"
        )
        
        return proba_long_arr, proba_short_arr


# Global model instance
_tcn_model: Optional[TCNSignalModel] = None


def get_tcn_model() -> Optional[TCNSignalModel]:
    """Get global TCN model instance (lazy loading)."""
    global _tcn_model
    if _tcn_model is None:
        _tcn_model = TCNSignalModel()
    return _tcn_model

