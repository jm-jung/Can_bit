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

from src.core.config import settings, get_lstm_attn_model_path
from src.dl.models.lstm_attn import LSTMAttentionModel
from src.indicators.basic import add_basic_indicators
from src.ml.features import build_feature_frame, build_ml_dataset
from src.dl.data.labels import LstmClassIndex

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
            # Generate dynamic model path based on configuration (3-class, horizon, feature preset)
            num_classes = 3  # 3-class: FLAT, LONG, SHORT
            horizon = getattr(settings, "LSTM_RETURN_HORIZON", 5)
            feature_preset = "events" if settings.EVENTS_ENABLED else "basic"
            model_path = get_lstm_attn_model_path(
                num_classes=num_classes,
                horizon=horizon,
                feature_preset=feature_preset,
            )

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
                # Extract symbol and timeframe from settings
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
            logger.info(
                f"Successfully loaded LSTM model: "
                f"feature_dim={feature_dim}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, dropout={self.dropout}"
            )
            logger.info(f"Model input_size attribute: {self.model.input_size}")
            logger.info(f"Feature columns count: {len(self.feature_cols)}")
            logger.info(f"Feature columns (first 10): {self.feature_cols[:10]}")
            
            # Feature columns detailed logging
            logger.info("=" * 60)
            logger.info("[LSTM Inference] Feature Columns")
            logger.info("=" * 60)
            logger.info(f"[LSTM] Using {len(self.feature_cols)} feature columns:")
            logger.info(f"[LSTM] FEATURE_COLS = {self.feature_cols}")
            if settings.EVENTS_ENABLED:
                event_cols = [c for c in self.feature_cols if c.startswith("event_")]
                logger.info(f"[LSTM] Event features ({len(event_cols)}): {event_cols}")
                basic_cols = [c for c in self.feature_cols if not c.startswith("event_")]
                logger.info(f"[LSTM] Basic features ({len(basic_cols)}): {basic_cols}")
            logger.info("=" * 60)
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
        df = df.copy()

        # Extract symbol and timeframe from settings if not provided
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

        # Cache feature columns if not set
        if self.feature_cols is None:
            self.feature_cols = features.columns.tolist()

        # 추론 시점에 피처 순서를 학습 시점과 동일하게 맞춘다
        features = features.reindex(columns=self.feature_cols, fill_value=0.0)
        return features

    def _prepare_sequence(self, df: pd.DataFrame, symbol: str | None = None, timeframe: str | None = None) -> torch.Tensor:
        """
        Prepare sequence for LSTM model.
        
        IMPORTANT: 이 함수는 학습 시점의 정규화 방식과 완전히 동일해야 함.
        - 학습 시: 각 시퀀스마다 rolling(window=window_size)로 정규화
        - 예측 시: 동일한 방식으로 정규화 (현재 시점 기준 rolling window)

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)
            symbol: Trading symbol (default: from settings)
            timeframe: Timeframe (default: from settings)

        Returns:
            Tensor of shape (1, window_size, feature_dim)
        """
        features = self._extract_features(df, symbol=symbol, timeframe=timeframe)

        # Ensure we have enough data
        if len(features) < self.window_size:
            raise ValueError(
                f"Not enough data: need at least {self.window_size} rows, got {len(features)}"
            )

        # Get last window_size rows
        seq_features = features.iloc[-self.window_size :].copy()

        # Normalize features (using rolling statistics, same as training)
        # IMPORTANT: 학습 시점과 동일한 방식으로 정규화
        #
        # 정규화 방식 (학습 시점과 동일):
        # - 학습 시: 각 행 i에 대해 [i-window_size+1:i+1] 구간의 평균/표준편차로 정규화
        # - 예측 시: 마지막 시퀀스의 마지막 행에서 [len-window_size:len] 구간의 평균/표준편차 사용
        #
        # 예: window_size=60, 전체 데이터가 1000행일 때
        #   - 학습 시: 행 60은 [0:61] 구간의 통계로 정규화
        #   - 예측 시: 행 999는 [939:1000] 구간의 통계로 정규화
        #
        # 이렇게 하면 학습 시점과 예측 시점의 정규화 방식이 일치함
        for col in seq_features.columns:
            # Calculate mean and std from the full feature history
            # 마지막 시퀀스의 마지막 행에서 rolling window의 마지막 값 사용
            mean = features[col].rolling(window=self.window_size, min_periods=1).mean().iloc[-1]
            std = features[col].rolling(window=self.window_size, min_periods=1).std().iloc[-1]
            if std == 0:
                std = 1  # Avoid division by zero
            seq_features[col] = (seq_features[col] - mean) / std

        # Convert to numpy and then tensor
        seq_array = seq_features.values.astype(np.float32)
        seq_tensor = torch.FloatTensor(seq_array).unsqueeze(0)  # Add batch dimension
        
        # Feature dimension 검증 로깅 (디버깅용)
        actual_feature_dim = seq_tensor.shape[2]
        if self.model is not None and hasattr(self.model, 'input_size'):
            expected_feature_dim = self.model.input_size
            if actual_feature_dim != expected_feature_dim:
                logger.error(
                    f"Feature dimension mismatch! "
                    f"Expected: {expected_feature_dim}, Got: {actual_feature_dim}. "
                    f"This will cause model inference to fail."
                )
                raise ValueError(
                    f"Feature dimension mismatch: expected {expected_feature_dim}, got {actual_feature_dim}"
                )
            logger.debug(
                f"Feature dimension check: expected={expected_feature_dim}, "
                f"actual={actual_feature_dim}, sequence shape={seq_tensor.shape}"
            )

        return seq_tensor

    def predict_proba_latest(self, df: pd.DataFrame, symbol: str | None = None, timeframe: str | None = None) -> float:
        """
        Predict probability of price going up (next horizon periods).
        
        This method maintains backward compatibility by returning proba_long
        from 3-class softmax output.

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)
            symbol: Trading symbol (default: from settings)
            timeframe: Timeframe (default: from settings)

        Returns:
            Probability of LONG class (0.0 to 1.0) from 3-class softmax
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Train model first or check model path.")

        # Prepare sequence (uses _extract_features which now accepts symbol/timeframe)
        # Note: _prepare_sequence calls _extract_features internally
        # We need to pass symbol/timeframe through _prepare_sequence
        seq_tensor = self._prepare_sequence(df, symbol=symbol, timeframe=timeframe)
        seq_tensor = seq_tensor.to(self.device)
        
        # 입력 텐서 shape 로깅 (디버깅용)
        logger.debug(
            f"predict_proba_latest: input tensor shape={seq_tensor.shape}, "
            f"feature_dim={seq_tensor.shape[2] if len(seq_tensor.shape) >= 3 else 'N/A'}"
        )

        # Predict (3-class)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(seq_tensor)  # (1, 3), raw logits
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (1, 3), probabilities
            prob_long = float(probs[0, LstmClassIndex.LONG].cpu().item())
            
            logger.debug(
                f"predict_proba_latest: logits={logits.cpu().numpy().flatten()}, "
                f"probs=[FLAT={probs[0, LstmClassIndex.FLAT]:.4f}, "
                f"LONG={probs[0, LstmClassIndex.LONG]:.4f}, "
                f"SHORT={probs[0, LstmClassIndex.SHORT]:.4f}], "
                f"proba_long={prob_long:.4f}"
            )

        return prob_long

    def predict_label_latest(
        self,
        df: pd.DataFrame,
        threshold_up: float | None = None,
        threshold_down: float | None = None,
    ) -> str:
        """
        Predict trading signal based on probability.

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)
            threshold_up: Probability threshold for LONG signal (default: from settings)
            threshold_down: Probability threshold for SHORT signal (default: from settings)

        Returns:
            Trading signal: "LONG", "SHORT", or "HOLD"
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Train model first or check model path.")

        # Use settings defaults if not provided
        if threshold_up is None:
            threshold_up = settings.LSTM_ATTN_THRESHOLD_UP
        if threshold_down is None:
            threshold_down = settings.LSTM_ATTN_THRESHOLD_DOWN

        prob_up = self.predict_proba_latest(df)

        if prob_up >= threshold_up:
            return "LONG"
        elif prob_up <= threshold_down:
            return "SHORT"
        else:
            return "HOLD"

    def predict_proba_batch(
        self,
        features: pd.DataFrame,
        symbol: str | None = None,
        timeframe: str | None = None,
        batch_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for multiple sequences (optimized for backtest/cache generation).
        
        This function is optimized for computing predictions for many timesteps at once.
        It extracts features once, creates all sequences, and processes them in batches.
        
        The model outputs 3-class softmax probabilities, which are mapped to:
        - proba_long: probability of LONG class (class 1)
        - proba_short: probability of SHORT class (class 2)
        
        Args:
            features: DataFrame with extracted features (already normalized if needed)
                     Shape: (N, feature_dim) where N >= window_size
            symbol: Trading symbol (default: from settings)
            timeframe: Timeframe (default: from settings)
            batch_size: Batch size for model forward passes (default: 512)
        
        Returns:
            Tuple of (proba_long: np.ndarray, proba_short: np.ndarray)
            Shape: (N - window_size + 1,) where N is the number of feature rows
            proba_long and proba_short are extracted from 3-class softmax output
            The first prediction corresponds to features[window_size-1], 
            and the last prediction corresponds to features[N-1]
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
        
        num_sequences = len(features) - self.window_size + 1
        feature_dim = len(self.feature_cols)
        
        logger.info(
            f"[LSTM Proba][Batch] total_rows={len(features)}, window_size={self.window_size}, "
            f"num_sequences={num_sequences}, batch_size={batch_size}"
        )
        
        # Create all sequences with rolling normalization
        # Each sequence i uses features[i:i+window_size] normalized by rolling stats
        # We create sequences starting from index (window_size - 1) to match training
        sequences: list[torch.Tensor] = []
        
        # Pre-compute rolling statistics for all features
        rolling_mean = features.rolling(window=self.window_size, min_periods=1).mean()
        rolling_std = features.rolling(window=self.window_size, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero
        
        # Create normalized sequences
        # Sequence at position i uses features[i-window_size+1:i+1] normalized by stats at i
        # This matches training: each sequence is normalized by rolling stats at the end of the window
        for i in range(self.window_size - 1, len(features)):
            # Get sequence window: from (i - window_size + 1) to i (inclusive)
            seq_start = i - self.window_size + 1
            seq_end = i + 1
            seq_features = features.iloc[seq_start:seq_end].copy()
            
            # Normalize using rolling statistics at the end of the window (index i)
            # This matches the training normalization: each row is normalized by
            # the rolling stats computed up to that row
            mean_at_end = rolling_mean.iloc[i]
            std_at_end = rolling_std.iloc[i]
            
            # Normalize sequence
            seq_normalized = (seq_features - mean_at_end) / std_at_end
            
            # Convert to tensor
            seq_array = seq_normalized.values.astype(np.float32)
            seq_tensor = torch.FloatTensor(seq_array).unsqueeze(0)  # (1, window_size, feature_dim)
            sequences.append(seq_tensor)
        
        # Concatenate all sequences into a single tensor
        if sequences:
            all_sequences = torch.cat(sequences, dim=0)  # (num_sequences, window_size, feature_dim)
        else:
            # Edge case: no sequences
            all_sequences = torch.empty((0, self.window_size, feature_dim), dtype=torch.float32)
        
        # Move to device
        all_sequences = all_sequences.to(self.device)
        
        # Batch forward passes
        self.model.eval()
        all_proba_long: list[float] = []
        all_proba_short: list[float] = []
        forward_calls = 0
        
        with torch.no_grad():
            for batch_start in range(0, num_sequences, batch_size):
                batch_end = min(batch_start + batch_size, num_sequences)
                batch_sequences = all_sequences[batch_start:batch_end]
                
                # Forward pass (3-class)
                logits = self.model(batch_sequences)  # (batch_size, 3)
                probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, 3)
                
                # Extract LONG and SHORT probabilities
                proba_long_batch = probs[:, LstmClassIndex.LONG].cpu().numpy()
                proba_short_batch = probs[:, LstmClassIndex.SHORT].cpu().numpy()
                
                all_proba_long.extend(proba_long_batch.tolist())
                all_proba_short.extend(proba_short_batch.tolist())
                forward_calls += 1
        
        proba_long_arr = np.array(all_proba_long, dtype=np.float32)
        proba_short_arr = np.array(all_proba_short, dtype=np.float32)
        
        logger.info(
            f"[LSTM Proba][Batch] forward_calls={forward_calls}, "
            f"mean_proba_long={proba_long_arr.mean():.4f}, std={proba_long_arr.std():.4f}, "
            f"mean_proba_short={proba_short_arr.mean():.4f}, std={proba_short_arr.std():.4f}"
        )
        
        return proba_long_arr, proba_short_arr


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

