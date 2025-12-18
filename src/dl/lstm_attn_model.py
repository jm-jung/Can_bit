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


def build_normalized_lstm_sequences_full(
    features_df: pd.DataFrame,
    window_size: int,
    feature_cols: list[str],
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Build the full 3D LSTM input tensor for all valid time indices using a shared normalization + sanitization.
    
    This function normalizes the entire DataFrame ONCE, then extracts all sequences via sliding windows.
    Time complexity: O(N × window_size) instead of O(N^2).
    
    This is the SHARED helper used by both batch and legacy prediction paths to ensure
    mathematical identity in feature extraction, normalization, sanitization, and window indexing.
    
    Window indexing rule (matching training):
    - For sequence index i (0-based), the corresponding time index is idx = i + window_size
    - Sequence uses rows [idx - window_size, idx) = [i, i + window_size)
    - This corresponds to `features_df.iloc[i : i + window_size]`
    
    Steps:
    1. Normalize the entire DataFrame using rolling statistics (vectorized, executed ONCE)
    2. Apply NaN/Inf sanitization ONCE on the normalized DataFrame
    3. Extract all sequences via sliding windows
    4. Return 3D array of shape (num_sequences, window_size, feature_dim)
    
    Args:
        features_df: Full features DataFrame (must have at least window_size rows)
        window_size: Sequence length (number of timesteps)
        feature_cols: List of feature column names in the correct order
        logger: Optional logger instance (uses module logger if None)
    
    Returns:
        sequences: np.ndarray of shape (num_sequences, window_size, feature_dim), dtype float32
        where num_sequences = len(features_df) - window_size
        All values are guaranteed to be finite (no NaN/Inf)
    
    Raises:
        ValueError: If len(features_df) < window_size
        ValueError: If sanitization fails (non-finite values remain)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    import numpy as np
    import pandas as pd
    
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
        f"[LSTM Normalize] Building {num_sequences} sequences from {N} feature rows "
        f"(window_size={window_size}, feature_dim={feature_dim})"
    )
    
    # Step 1: Normalize the entire DataFrame ONCE (vectorized, O(N))
    # Each row i is normalized by rolling stats at i: [i - window_size + 1, ..., i] inclusive
    # This matches the training pipeline: normalize entire DataFrame, then extract sequences
    features_normalized = features_df.copy()
    
    for col in feature_cols:
        # Rolling statistics: window=window_size, min_periods=1 (same as training)
        rolling_mean = features_normalized[col].rolling(window=window_size, min_periods=1).mean()
        rolling_std = features_normalized[col].rolling(window=window_size, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero
        
        # Z-score normalization: (X - mean) / std
        features_normalized[col] = (features_normalized[col] - rolling_mean) / rolling_std
    
    # Convert to float32
    for col in feature_cols:
        features_normalized[col] = features_normalized[col].astype(np.float32)
    
    # Step 2: Sanitize NaN/Inf ONCE on the normalized DataFrame
    features_bad = np.isnan(features_normalized.values).any() or np.isinf(features_normalized.values).any()
    if features_bad:
        # Count issues before sanitization (summarized logging)
        nan_cols = []
        total_nan_count = 0
        for col in feature_cols:
            col_values = features_normalized[col].values
            nan_count = np.isnan(col_values).sum()
            inf_count = np.isinf(col_values).sum()
            if nan_count > 0 or inf_count > 0:
                nan_cols.append(col)
                total_nan_count += nan_count + inf_count
        
        logger.warning(
            f"[LSTM Normalize] Found NaN/Inf in features_normalized DataFrame. "
            f"Sanitizing: {len(nan_cols)} columns affected, {total_nan_count} total NaN/Inf values. "
            f"Replacing with 0.0 (neutral z-score value)."
        )
        
        # Replace +/-Inf with NaN first (explicit reassignment)
        features_normalized = features_normalized.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaNs with 0.0 column-by-column (explicit reassignment)
        for col in feature_cols:
            if features_normalized[col].isna().sum() > 0:
                features_normalized[col] = features_normalized[col].fillna(0.0)
        
        # Final validation: ensure all values are now finite
        if not np.isfinite(features_normalized.to_numpy()).all():
            logger.error(
                "[LSTM Normalize] After sanitization, still found NaN/Inf in features_normalized. "
                "This indicates a bug in the sanitization logic."
            )
            raise ValueError("Non-finite values remain after sanitization in features_normalized.")
        
        logger.info(
            f"[LSTM Normalize] Sanitization complete. All NaN/Inf in features_normalized replaced with 0.0."
        )
    
    # Step 3: Extract all sequences via sliding windows (O(N × window_size))
    # Preallocate sequences array
    sequences_array = np.empty((num_sequences, window_size, feature_dim), dtype=np.float32)
    
    # Extract sequences: for sequence index i, use rows [i, i + window_size)
    # This matches training: for i in range(window_size, len), seq = iloc[i - window_size : i]
    # But we index sequences from 0, so for seq_i=0, we use rows [0, window_size)
    # For seq_i=1, we use rows [1, window_size+1), etc.
    # The corresponding time index for seq_i is idx = seq_i + window_size
    # So seq_i uses rows [idx - window_size, idx) = [seq_i, seq_i + window_size)
    for seq_i in range(num_sequences):
        seq_start = seq_i
        seq_end = seq_i + window_size
        window_slice = features_normalized.iloc[seq_start:seq_end]
        sequences_array[seq_i] = window_slice[feature_cols].values.astype(np.float32)
    
    # Final validation: ensure all sequences are finite
    if not np.isfinite(sequences_array).all():
        bad_mask = ~np.isfinite(sequences_array)
        bad_count = bad_mask.sum()
        total_elements = sequences_array.size
        
        logger.error(
            f"[LSTM Normalize] CRITICAL: Found NaN/Inf in sequences_array after all processing: "
            f"bad_count={bad_count}, total_elements={total_elements}, "
            f"bad_ratio={bad_count/total_elements:.6f}"
        )
        raise ValueError("Non-finite values in sequences_array after normalization and sanitization.")
    
    logger.info(
        f"[LSTM Normalize] Successfully built {num_sequences} sequences "
        f"(shape={sequences_array.shape}, dtype={sequences_array.dtype})"
    )
    
    return sequences_array


def build_normalized_lstm_sequence(
    features_df: pd.DataFrame,
    idx: int,
    window_size: int,
    feature_cols: list[str],
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Build a single normalized LSTM input sequence for time index `idx` using a lookback window.
    
    This is the SHARED helper used by both batch and legacy prediction paths to ensure
    mathematical identity in feature extraction, normalization, sanitization, and window indexing.
    
    Window indexing rule:
    - For time index `idx`, the sequence uses rows [idx - window_size + 1, ..., idx]
    - This corresponds to `features_df.iloc[idx - window_size + 1 : idx + 1]` (inclusive end)
    - Or equivalently: `features_df.iloc[idx - window_size : idx]` if using exclusive end (matching training)
    
    Steps:
    1. Extract the window slice from features_df
    2. Normalize the window using rolling statistics computed at each row
    3. Apply NaN/Inf sanitization (replace +/-Inf with NaN, then NaN with 0.0)
    4. Preserve the column ordering of feature_cols
    5. Return a float32 numpy array of shape (window_size, feature_dim)
    
    Args:
        features_df: Full features DataFrame (must have at least idx+1 rows)
        idx: Target time index (0-based). Sequence will use [idx - window_size + 1, ..., idx]
        window_size: Sequence length (number of timesteps)
        feature_cols: List of feature column names in the correct order
        logger: Optional logger instance (uses module logger if None)
    
    Returns:
        Normalized sequence array of shape (window_size, len(feature_cols)), dtype float32
        All values are guaranteed to be finite (no NaN/Inf)
    
    Raises:
        ValueError: If idx < window_size - 1 (not enough history)
        ValueError: If sanitization fails (non-finite values remain)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    import numpy as np
    import pandas as pd
    
    # Validate index
    if idx < window_size - 1:
        raise ValueError(
            f"Index {idx} is too small for window_size {window_size}. "
            f"Need at least {window_size} rows before index {idx}."
        )
    
    if len(features_df) <= idx:
        raise ValueError(
            f"Index {idx} is out of bounds. DataFrame has {len(features_df)} rows."
        )
    
    # Ensure feature columns are in the correct order
    features_df = features_df.reindex(columns=feature_cols, fill_value=0.0)
    
    # Step 1: Extract window slice
    # Training rule: seq = X_normalized.iloc[i - window_size : i] (exclusive end)
    # For index idx, we want [idx - window_size + 1, ..., idx] inclusive
    # This is equivalent to iloc[idx - window_size : idx] if we use 0-based indexing
    # But to match training exactly: for i in range(window_size, len), seq = iloc[i - window_size : i]
    # So for idx, we use: iloc[idx - window_size : idx] (exclusive end, matching training)
    seq_start = idx - window_size
    seq_end = idx
    window_slice = features_df.iloc[seq_start : seq_end].copy()
    
    if len(window_slice) != window_size:
        raise ValueError(
            f"Window slice has incorrect length: expected {window_size}, got {len(window_slice)}. "
            f"seq_start={seq_start}, seq_end={seq_end}, features_df_len={len(features_df)}"
        )
    
    # Step 2: Normalize using rolling statistics (same as training)
    # IMPORTANT: We normalize the entire features_df up to idx, then extract the window
    # This matches the training pipeline: normalize entire DataFrame, then extract sequences
    # Each row i is normalized by rolling stats at i: [i - window_size + 1, ..., i] inclusive
    
    # Normalize features_df up to idx (we only need up to idx for this sequence)
    features_normalized = features_df.iloc[:idx+1].copy()
    
    for col in feature_cols:
        # Rolling statistics: window=window_size, min_periods=1 (same as training)
        rolling_mean = features_normalized[col].rolling(window=window_size, min_periods=1).mean()
        rolling_std = features_normalized[col].rolling(window=window_size, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero
        
        # Z-score normalization: (X - mean) / std
        features_normalized[col] = (features_normalized[col] - rolling_mean) / rolling_std
    
    # Convert to float32
    for col in feature_cols:
        features_normalized[col] = features_normalized[col].astype(np.float32)
    
    # Extract the window from normalized features
    window_normalized = features_normalized.iloc[seq_start:seq_end].copy()
    
    # Step 3: Sanitize NaN/Inf (inference-time robustness)
    # Replace +/-Inf with NaN first, then fill NaN with 0.0
    has_nan_inf = not np.isfinite(window_normalized.values).all()
    if has_nan_inf:
        # Count issues before sanitization (minimal logging to avoid spam)
        nan_cols = []
        for col in feature_cols:
            if window_normalized[col].isna().any() or np.isinf(window_normalized[col].values).any():
                nan_cols.append(col)
        
        if len(nan_cols) > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[LSTM Sequence] Sanitizing {len(nan_cols)} columns with NaN/Inf at idx={idx}: "
                f"{nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}"
            )
        
        # Replace +/-Inf with NaN
        window_normalized = window_normalized.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with 0.0 column-by-column
        for col in feature_cols:
            if window_normalized[col].isna().sum() > 0:
                window_normalized[col] = window_normalized[col].fillna(0.0)
        
        # Final validation
        if not np.isfinite(window_normalized.values).all():
            logger.error(
                f"[LSTM Sequence] After sanitization, still found NaN/Inf at idx={idx}. "
                f"This indicates a bug in the sanitization logic."
            )
            raise ValueError(f"Non-finite values remain after sanitization at idx={idx}.")
    
    # Step 4: Convert to numpy array with correct ordering
    sequence_array = window_normalized[feature_cols].values.astype(np.float32)
    
    # Final shape check
    if sequence_array.shape != (window_size, len(feature_cols)):
        raise ValueError(
            f"Sequence array has incorrect shape: expected ({window_size}, {len(feature_cols)}), "
            f"got {sequence_array.shape}"
        )
    
    # Final finite check
    if not np.isfinite(sequence_array).all():
        logger.error(
            f"[LSTM Sequence] CRITICAL: Sequence array at idx={idx} contains NaN/Inf after all sanitization."
        )
        raise ValueError(f"Non-finite values in sequence array at idx={idx}.")
    
    return sequence_array


def normalize_and_make_sequences_for_lstm(
    features: pd.DataFrame,
    window_size: int,
    *,
    horizon: int | None = None,
    for_inference: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    학습 시 사용한 것과 동일한 방식으로 정규화 및 시퀀스 생성.
    
    이 함수는 train_lstm_attn.py의 create_sequences 함수와 완전히 동일한
    정규화/시퀀스 생성 로직을 사용합니다.
    
    Args:
        features: DataFrame with extracted features (shape: (N, feature_dim))
        window_size: Sequence length (number of timesteps)
        horizon: Number of periods ahead (for training, used to limit sequence range)
                If None and for_inference=False, uses default from settings
        for_inference: If True, creates sequences for all valid indices (no horizon limit)
    
    Returns:
        Tuple of:
        - features_normalized: Normalized DataFrame (same shape as input)
        - sequences_array: numpy array of shape (num_sequences, window_size, feature_dim)
    
    정규화 방식 (학습 시와 동일):
        - 각 행 i에 대해 [i-window_size+1:i+1] 구간의 평균/표준편차로 z-score 정규화
        - rolling(window=window_size, min_periods=1) 사용
        - std == 0인 경우 1로 치환 (division by zero 방지)
        - float32로 변환 (메모리 최적화)
    
    시퀀스 생성 방식 (학습 시와 동일):
        - Training: for i in range(window_size, len(X_normalized) - horizon + 1):
        - Inference: for i in range(window_size, len(features)):
        - Sequence: X_normalized.iloc[i - window_size : i]  # [i-window_size, i) exclusive
    """
    import numpy as np
    import pandas as pd
    
    # Step 1: Normalize features row-by-row (same as training)
    # 학습 코드: train_lstm_attn.py lines 627-632
    features_normalized = features.copy()
    feature_cols = features.columns.tolist()
    
    for col in feature_cols:
        # Rolling statistics: window=window_size, min_periods=1 (학습 시와 동일)
        mean = features_normalized[col].rolling(window=window_size, min_periods=1).mean()
        std = features_normalized[col].rolling(window=window_size, min_periods=1).std()
        # std == 0인 경우 1로 치환 (학습 시와 동일)
        std = std.replace(0, 1)  # Avoid division by zero
        
        # Z-score normalization: (X - mean) / std
        features_normalized[col] = (features_normalized[col] - mean) / std
    
    # Convert to float32 (학습 시와 동일한 메모리 최적화)
    for col in feature_cols:
        features_normalized[col] = features_normalized[col].astype(np.float32)
    
    # Step 2: Sanitize NaN/Inf in features_normalized (inference-time robustness)
    # Replace +/- Inf with NaN first, then fill all NaNs with 0.0 (neutral z-score value)
    features_bad = np.isnan(features_normalized.values).any() or np.isinf(features_normalized.values).any()
    if features_bad:
        logger.warning(
            f"[LSTM Normalize] Found NaN/Inf in features_normalized DataFrame. "
            f"Sanitizing by replacing with 0.0 (neutral z-score value)."
        )
        
        # Replace +/- Inf with NaN first (explicit reassignment - OPTION A)
        features_normalized = features_normalized.replace([np.inf, -np.inf], np.nan)
        
        # Log which columns have issues before sanitization
        for col in feature_cols:
            nan_count = features_normalized[col].isna().sum()
            if nan_count > 0:
                logger.warning(
                    f"[LSTM Normalize] Column '{col}' has {nan_count} NaN values (will be filled with 0.0)"
                )
        
        # Fill NaNs with 0.0 column-by-column (explicit reassignment - OPTION A)
        # This ensures each column is properly sanitized
        for col in feature_cols:
            if features_normalized[col].isna().sum() > 0:
                features_normalized[col] = features_normalized[col].fillna(0.0)
        
        # Final validation: ensure all values are now finite
        if not np.isfinite(features_normalized.to_numpy()).all():
            logger.error(
                "[LSTM Normalize] After sanitization, still found NaN/Inf in features_normalized. "
                "This indicates a bug in the sanitization logic."
            )
            raise ValueError("Non-finite values remain after sanitization in features_normalized.")
        
        logger.info(
            f"[LSTM Normalize] Sanitization complete. All NaN/Inf in features_normalized replaced with 0.0."
        )
    
    # Step 3: Create sequences (same as training)
    # 학습 코드: train_lstm_attn.py lines 662-664
    sequences_list: list[np.ndarray] = []
    
    if for_inference:
        # Inference: use all valid indices (no horizon constraint)
        sequence_range = range(window_size, len(features_normalized))
    else:
        # Training: limit by horizon
        if horizon is None:
            from src.core.config import settings
            horizon = getattr(settings, "LSTM_RETURN_HORIZON", 5)
        sequence_range = range(window_size, len(features_normalized) - horizon + 1)
    
    for i in sequence_range:
        # Extract sequence: [i - window_size, i) exclusive (학습 시와 동일)
        seq = features_normalized.iloc[i - window_size : i][feature_cols].values
        sequences_list.append(seq)
    
    # Step 4: Convert to numpy array (학습 시와 동일)
    # 학습 코드: train_lstm_attn.py line 698
    if sequences_list:
        sequences_array = np.array(sequences_list, dtype=np.float32)  # (num_sequences, window_size, feature_dim)
    else:
        sequences_array = np.empty((0, window_size, len(feature_cols)), dtype=np.float32)
    
    # Step 5: Sanitize NaN/Inf in sequences_array (inference-time robustness)
    # After sanitizing features_normalized, sequences_array should be clean, but check anyway
    if not np.isfinite(sequences_array).all():
        bad_mask = ~np.isfinite(sequences_array)
        bad_count = bad_mask.sum()
        total_elements = sequences_array.size
        
        logger.warning(
            f"[LSTM Normalize] Found NaN/Inf in sequences_array after sanitization: "
            f"bad_count={bad_count}, total_elements={total_elements}, "
            f"bad_ratio={bad_count/total_elements:.6f}. "
            f"Sanitizing by replacing with 0.0."
        )
        
        # Find problematic sequences for logging
        bad_sequences = bad_mask.any(axis=(1, 2))  # (num_sequences,)
        if bad_sequences.any():
            bad_seq_indices = np.where(bad_sequences)[0]
            logger.warning(
                f"[LSTM Normalize] Bad sequences at indices: {bad_seq_indices[:10]} "
                f"(showing first 10 of {len(bad_seq_indices)} total)"
            )
            
            # Log example bad sequence
            if len(bad_seq_indices) > 0:
                example_idx = bad_seq_indices[0]
                example_seq = sequences_array[example_idx]
                nan_count = np.isnan(example_seq).sum()
                inf_count = np.isinf(example_seq).sum()
                logger.warning(
                    f"[LSTM Normalize] Example bad sequence[{example_idx}]: "
                    f"shape={example_seq.shape}, NaN={nan_count}, Inf={inf_count}, "
                    f"min={np.nanmin(example_seq):.6f}, max={np.nanmax(example_seq):.6f}"
                )
        
        # Replace NaN/Inf with 0.0 (deterministic, training-compatible fallback)
        # Direct assignment to numpy array (this modifies in-place)
        sequences_array[~np.isfinite(sequences_array)] = 0.0
        
        # Final validation: ensure all values are now finite
        if not np.isfinite(sequences_array).all():
            logger.error(
                "[LSTM Normalize] After sanitization, still found NaN/Inf in sequences_array. "
                "This indicates a bug in the sanitization logic."
            )
            raise ValueError("Non-finite values remain after sanitization in sequences_array.")
        
        logger.info(
            f"[LSTM Normalize] Sanitization complete. All NaN/Inf in sequences_array replaced with 0.0."
        )
    
    return features_normalized, sequences_array


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
        
        # Cache for sequences to avoid repeated normalization (O(N^2) -> O(N × window_size))
        # Cache key: (id(features_df), window_size, tuple(feature_cols))
        self._cached_features_df_id: int | None = None
        self._cached_sequences_full: np.ndarray | None = None
        self._cached_window_size: int | None = None
        self._cached_feature_cols: list[str] | None = None

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

    def _get_or_build_sequences_full(
        self,
        features_df: pd.DataFrame,
        window_size: int,
        feature_cols: list[str],
        logger: logging.Logger,
    ) -> np.ndarray:
        """
        Return a cached full 3D sequence tensor if possible; otherwise build it once
        using build_normalized_lstm_sequences_full(), cache it, and return it.
        
        This ensures build_normalized_lstm_sequences_full() is called at most once
        per features_df + window_size + feature_cols configuration.
        
        Args:
            features_df: Full features DataFrame
            window_size: Sequence length
            feature_cols: List of feature column names
            logger: Logger instance
        
        Returns:
            sequences: np.ndarray of shape (num_sequences, window_size, feature_dim), dtype float32
        """
        # Cache key: use id() to check if it's the same DataFrame object
        features_df_id = id(features_df)
        
        # Check cache
        if (self._cached_features_df_id == features_df_id and
            self._cached_window_size == window_size and
            self._cached_feature_cols == feature_cols):
            logger.debug(
                f"[LSTM Cache] Cache HIT: reusing {self._cached_sequences_full.shape[0]} sequences "
                f"for features_df (id={features_df_id}, len={len(features_df)})"
            )
            return self._cached_sequences_full
        
        # Cache miss: build sequences
        logger.debug(
            f"[LSTM Cache] Cache MISS: building sequences for features_df "
            f"(id={features_df_id}, len={len(features_df)}, window_size={window_size})"
        )
        
        sequences = build_normalized_lstm_sequences_full(
            features_df=features_df,
            window_size=window_size,
            feature_cols=feature_cols,
            logger=logger,
        )
        
        # Update cache
        self._cached_features_df_id = features_df_id
        self._cached_window_size = window_size
        self._cached_feature_cols = list(feature_cols)  # Make a copy
        self._cached_sequences_full = sequences
        
        return sequences

    def _prepare_sequence(self, df: pd.DataFrame, symbol: str | None = None, timeframe: str | None = None) -> torch.Tensor:
        """
        Prepare sequence for LSTM model (legacy single-step prediction).
        
        IMPORTANT: This function now uses the shared build_normalized_lstm_sequence() helper
        to ensure mathematical identity with the batch prediction path.

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

        # Ensure feature columns match
        if self.feature_cols is None:
            self.feature_cols = features.columns.tolist()
        else:
            features = features.reindex(columns=self.feature_cols, fill_value=0.0)

        # IMPORTANT: Use cached sequences to ensure mathematical identity with batch path
        # This uses the same cached sequences as batch prediction, avoiding repeated normalization
        sequences_array = self._get_or_build_sequences_full(
            features_df=features,
            window_size=self.window_size,
            feature_cols=self.feature_cols,
            logger=logger,
        )
        
        # Extract the last sequence (corresponds to the latest time index)
        # The last sequence is at index len(sequences_array) - 1
        # This corresponds to time index: (len(sequences_array) - 1) + window_size = len(features) - 1
        latest_sequence = sequences_array[-1]  # shape: (window_size, feature_dim)
        
        # Defensive assertions
        assert latest_sequence.ndim == 2, f"Expected 2D sequence, got {latest_sequence.ndim}D"
        assert latest_sequence.shape[0] == self.window_size, \
            f"Expected window_size={self.window_size}, got {latest_sequence.shape[0]}"
        assert latest_sequence.shape[1] == len(self.feature_cols), \
            f"Expected feature_dim={len(self.feature_cols)}, got {latest_sequence.shape[1]}"
        
        # Convert to tensor: (1, window_size, feature_dim)
        seq_tensor = torch.from_numpy(latest_sequence).unsqueeze(0)
        
        # Feature dimension validation
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

    def predict_proba_latest(
        self, 
        df: pd.DataFrame, 
        symbol: str | None = None, 
        timeframe: str | None = None,
        return_both: bool = False,
    ) -> float | tuple[float, float]:
        """
        Predict probability of price going up (next horizon periods).
        
        This method maintains backward compatibility by returning proba_long
        from 3-class softmax output. When return_both=True, returns both
        proba_long and proba_short from 3-class softmax.

        Args:
            df: DataFrame with OHLCV + indicators (must be sorted by timestamp)
            symbol: Trading symbol (default: from settings)
            timeframe: Timeframe (default: from settings)
            return_both: If True, returns (proba_long, proba_short) tuple.
                        If False, returns only proba_long (backward compatible).

        Returns:
            If return_both=True: Tuple of (proba_long: float, proba_short: float)
            If return_both=False: Probability of LONG class (0.0 to 1.0) from 3-class softmax
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
            
            # Check for NaN/Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.error(
                    "[LSTM Proba][Legacy] Found NaN/Inf in logits. "
                    "This indicates a problem in model forward pass or input sequence."
                )
                raise ValueError("NaN/Inf detected in model logits during legacy forward pass.")
            
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (1, 3), probabilities
            
            # Check for NaN/Inf in probabilities
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                logger.error(
                    "[LSTM Proba][Legacy] Found NaN/Inf in probabilities. "
                    "This indicates a problem in softmax computation."
                )
                raise ValueError("NaN/Inf detected in model probabilities after softmax.")
            
            prob_long = float(probs[0, LstmClassIndex.LONG].cpu().item())
            prob_short = float(probs[0, LstmClassIndex.SHORT].cpu().item())
            
            # Final check on extracted probabilities
            if np.isnan(prob_long) or np.isinf(prob_long):
                logger.error(
                    f"[LSTM Proba][Legacy] Found NaN/Inf in prob_long: {prob_long}"
                )
                raise ValueError("NaN/Inf detected in prob_long after extraction.")
            if np.isnan(prob_short) or np.isinf(prob_short):
                logger.error(
                    f"[LSTM Proba][Legacy] Found NaN/Inf in prob_short: {prob_short}"
                )
                raise ValueError("NaN/Inf detected in prob_short after extraction.")
            
            logger.debug(
                f"predict_proba_latest: logits={logits.cpu().numpy().flatten()}, "
                f"probs=[FLAT={probs[0, LstmClassIndex.FLAT]:.4f}, "
                f"LONG={probs[0, LstmClassIndex.LONG]:.4f}, "
                f"SHORT={probs[0, LstmClassIndex.SHORT]:.4f}], "
                f"proba_long={prob_long:.4f}, proba_short={prob_short:.4f}"
            )

        if return_both:
            return prob_long, prob_short
        else:
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
        
        feature_dim = len(self.feature_cols)
        
        logger.info(
            f"[LSTM Proba][Batch] total_rows={len(features)}, window_size={self.window_size}, "
            f"batch_size={batch_size}"
        )
        
        # IMPORTANT: Use cached sequences for O(N × window_size) complexity
        # This normalizes the entire DataFrame ONCE, then extracts all sequences via sliding windows
        # Both batch and legacy paths use this same cached sequences to ensure mathematical identity
        sequences_array = self._get_or_build_sequences_full(
            features_df=features,
            window_size=self.window_size,
            feature_cols=self.feature_cols,
            logger=logger,
        )
        
        num_sequences = sequences_array.shape[0]
        
        # Defensive assertions
        assert sequences_array.ndim == 3, f"Expected 3D sequences array, got {sequences_array.ndim}D"
        assert sequences_array.shape[1] == self.window_size, \
            f"Expected window_size={self.window_size}, got {sequences_array.shape[1]}"
        assert sequences_array.shape[2] == len(self.feature_cols), \
            f"Expected feature_dim={len(self.feature_cols)}, got {sequences_array.shape[2]}"
        
        logger.info(
            f"[LSTM Proba][Batch] Using {num_sequences} sequences from {len(features)} feature rows "
            f"(from cache or newly built)"
        )
        
        # Convert to tensor
        all_sequences = torch.from_numpy(sequences_array)  # (num_sequences, window_size, feature_dim)
        
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
                
                # Check for NaN/Inf in logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.error(
                        f"[LSTM Proba][Batch] Found NaN/Inf in logits at batch_start={batch_start}. "
                        f"This indicates a problem in model forward pass or input sequences."
                    )
                    bad_mask = torch.isnan(logits) | torch.isinf(logits)
                    bad_count = bad_mask.sum().item()
                    logger.error(
                        f"[LSTM Proba][Batch] Bad logits count = {bad_count}, "
                        f"batch_size = {batch_sequences.shape[0]}, "
                        f"logits shape = {logits.shape}"
                    )
                    raise ValueError("NaN/Inf detected in model logits during batch forward pass.")
                
                probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, 3)
                
                # Check for NaN/Inf in probabilities
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    logger.error(
                        f"[LSTM Proba][Batch] Found NaN/Inf in probabilities at batch_start={batch_start}. "
                        f"This indicates a problem in softmax computation."
                    )
                    bad_mask = torch.isnan(probs) | torch.isinf(probs)
                    bad_count = bad_mask.sum().item()
                    logger.error(
                        f"[LSTM Proba][Batch] Bad probs count = {bad_count}, "
                        f"batch_size = {batch_sequences.shape[0]}, "
                        f"probs shape = {probs.shape}"
                    )
                    raise ValueError("NaN/Inf detected in model probabilities after softmax.")
                
                # Extract LONG and SHORT probabilities
                proba_long_batch = probs[:, LstmClassIndex.LONG].cpu().numpy()
                proba_short_batch = probs[:, LstmClassIndex.SHORT].cpu().numpy()
                
                # Final check on extracted probabilities
                if np.isnan(proba_long_batch).any() or np.isinf(proba_long_batch).any():
                    logger.error(
                        f"[LSTM Proba][Batch] Found NaN/Inf in proba_long_batch at batch_start={batch_start}"
                    )
                    raise ValueError("NaN/Inf detected in proba_long_batch after extraction.")
                if np.isnan(proba_short_batch).any() or np.isinf(proba_short_batch).any():
                    logger.error(
                        f"[LSTM Proba][Batch] Found NaN/Inf in proba_short_batch at batch_start={batch_start}"
                    )
                    raise ValueError("NaN/Inf detected in proba_short_batch after extraction.")
                
                all_proba_long.extend(proba_long_batch.tolist())
                all_proba_short.extend(proba_short_batch.tolist())
                forward_calls += 1
        
        proba_long_arr = np.array(all_proba_long, dtype=np.float32)
        proba_short_arr = np.array(all_proba_short, dtype=np.float32)
        
        # Final NaN/Inf check on output arrays
        if np.isnan(proba_long_arr).any() or np.isinf(proba_long_arr).any():
            logger.error(
                "[LSTM Proba][Batch] Found NaN/Inf in final proba_long_arr. "
                "This should have been caught earlier."
            )
            bad_count = (np.isnan(proba_long_arr) | np.isinf(proba_long_arr)).sum()
            logger.error(
                f"[LSTM Proba][Batch] Bad proba_long count = {bad_count}, "
                f"total = {len(proba_long_arr)}"
            )
            raise ValueError("NaN/Inf detected in final proba_long_arr.")
        
        if np.isnan(proba_short_arr).any() or np.isinf(proba_short_arr).any():
            logger.error(
                "[LSTM Proba][Batch] Found NaN/Inf in final proba_short_arr. "
                "This should have been caught earlier."
            )
            bad_count = (np.isnan(proba_short_arr) | np.isinf(proba_short_arr)).sum()
            logger.error(
                f"[LSTM Proba][Batch] Bad proba_short count = {bad_count}, "
                f"total = {len(proba_short_arr)}"
            )
            raise ValueError("NaN/Inf detected in final proba_short_arr.")
        
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

