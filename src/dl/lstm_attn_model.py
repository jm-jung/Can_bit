"""
LSTM + Attention model wrapper for prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.core.config import settings
from src.dl.models.lstm_attn import LSTMAttentionModel
from src.indicators.basic import add_basic_indicators
from src.ml.features import build_ml_dataset

logger = logging.getLogger(__name__)


class LSTMAttnSignalModel:
    """LSTM + Attention model wrapper for BTC price direction prediction."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        window_size: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize LSTM + Attention model.

        Args:
            model_path: Path to saved model file. If None, uses default from settings.
            window_size: Sequence length (must match training)
            hidden_size: LSTM hidden size (must match training)
            num_layers: Number of LSTM layers (must match training)
            dropout: Dropout rate (must match training)
        """
        if model_path is None:
            model_path = settings.LSTM_ATTN_MODEL_PATH

        self.model_path = Path(model_path)
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model: Optional[LSTMAttentionModel] = None
        self.device: Optional[torch.device] = None
        self.feature_cols: Optional[list[str]] = None

        # Initialize device (always set, even if model file doesn't exist)
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            logger.error(f"Failed to initialize device: {e}")
            self.device = torch.device("cpu")

        # Model will be loaded lazily when needed
        self.model = None

    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded and ready, False otherwise
        """
        if self.model is None:
            # Try to load model lazily
            self._load_model()
        return self.model is not None

    def _load_model(self) -> None:
        """
        Load model from file (lazy loading).
        
        This method handles errors gracefully and logs warnings instead of raising exceptions.
        """
        # Check if model file exists
        if not self.model_path.exists():
            logger.warning(
                f"LSTM model file not found: {self.model_path.resolve()}. "
                f"Please train the model first or check the path in settings."
            )
            self.model = None
            return

        logger.info(f"Loading LSTM model from: {self.model_path.resolve()}")

        # Determine feature dimension by creating a sample feature set
        try:
            if self.feature_cols is None:
                # Create dummy data to get feature columns
                from src.services.ohlcv_service import load_ohlcv_df

                df = load_ohlcv_df()
                df = df.sort_values("timestamp").reset_index(drop=True)
                X_features, _ = build_ml_dataset(df, horizon=5)
                self.feature_cols = X_features.columns.tolist()

            feature_dim = len(self.feature_cols)
        except Exception as e:
            logger.error(f"Failed to determine feature dimension: {e}")
            self.model = None
            return

        # Load model state_dict
        try:
            self.model = LSTMAttentionModel.load_model(
                path=self.model_path,
                input_size=feature_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                device=self.device,
            )
            logger.info(f"Successfully loaded LSTM model (feature_dim={feature_dim})")
        except RuntimeError as e:
            # Handle size mismatch or other model loading errors
            error_msg = str(e)
            if "size mismatch" in error_msg.lower() or "missing keys" in error_msg.lower():
                logger.error(
                    f"Model architecture mismatch when loading from {self.model_path.resolve()}. "
                    f"Error: {error_msg}. "
                    f"Please retrain the model with matching hyperparameters."
                )
            else:
                logger.error(
                    f"RuntimeError when loading model from {self.model_path.resolve()}: {error_msg}"
                )
            self.model = None
        except Exception as e:
            # Handle any other exceptions
            logger.error(
                f"Failed to load model from {self.model_path.resolve()}: {type(e).__name__}: {e}"
            )
            self.model = None

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from DataFrame (same as training).

        Args:
            df: DataFrame with OHLCV + indicators

        Returns:
            Features DataFrame
        """
        df = df.copy()
        df = add_basic_indicators(df)

        # Build features using same function as training
        X_features, _ = build_ml_dataset(df, horizon=5)

        # Cache feature columns if not set
        if self.feature_cols is None:
            self.feature_cols = X_features.columns.tolist()

        return X_features

    def _prepare_sequence(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Prepare sequence for LSTM model.

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)

        Returns:
            Tensor of shape (1, window_size, feature_dim)
        """
        features = self._extract_features(df)

        # Ensure we have enough data
        if len(features) < self.window_size:
            raise ValueError(
                f"Not enough data: need at least {self.window_size} rows, got {len(features)}"
            )

        # Get last window_size rows
        seq_features = features.iloc[-self.window_size :].copy()

        # Normalize features (using rolling statistics, same as training)
        # Use the full feature history for normalization to match training
        for col in seq_features.columns:
            # Calculate mean and std from the full feature history
            mean = features[col].rolling(window=self.window_size, min_periods=1).mean().iloc[-1]
            std = features[col].rolling(window=self.window_size, min_periods=1).std().iloc[-1]
            if std == 0:
                std = 1  # Avoid division by zero
            seq_features[col] = (seq_features[col] - mean) / std

        # Convert to numpy and then tensor
        seq_array = seq_features.values.astype(np.float32)
        seq_tensor = torch.FloatTensor(seq_array).unsqueeze(0)  # Add batch dimension

        return seq_tensor

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

        # Prepare sequence
        seq_tensor = self._prepare_sequence(df)
        seq_tensor = seq_tensor.to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(seq_tensor)  # (1, 1), raw logit
            prob = torch.sigmoid(logits)  # (1, 1), 0~1
            prob_up = float(prob.cpu().item())

        return prob_up

    def predict_label_latest(
        self,
        df: pd.DataFrame,
        threshold_up: float = 0.55,
        threshold_down: float = 0.45,
    ) -> str:
        """
        Predict trading signal based on probability.

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)
            threshold_up: Probability threshold for LONG signal (default: 0.55)
            threshold_down: Probability threshold for SHORT signal (default: 0.45)

        Returns:
            Trading signal: "LONG", "SHORT", or "HOLD"
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Train model first or check model path.")

        prob_up = self.predict_proba_latest(df)

        if prob_up >= threshold_up:
            return "LONG"
        elif prob_up <= threshold_down:
            return "SHORT"
        else:
            return "HOLD"


# Global model instance
_lstm_attn_model: Optional[LSTMAttnSignalModel] = None


def get_lstm_attn_model() -> Optional[LSTMAttnSignalModel]:
    """
    Get or create global LSTM + Attention model instance.
    
    Returns:
        LSTMAttnSignalModel instance if available, None otherwise.
        Use model.is_loaded() to check if the model was successfully loaded.
    """
    global _lstm_attn_model
    if _lstm_attn_model is None:
        try:
            _lstm_attn_model = LSTMAttnSignalModel()
            # Try to load model immediately to catch errors early
            if not _lstm_attn_model.is_loaded():
                logger.warning(
                    "LSTM model instance created but model file could not be loaded. "
                    "Check logs above for details."
                )
        except Exception as e:
            logger.error(f"Failed to create LSTMAttnSignalModel instance: {e}")
            _lstm_attn_model = None
    return _lstm_attn_model

