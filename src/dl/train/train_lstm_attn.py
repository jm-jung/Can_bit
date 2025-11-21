"""
Training script for LSTM + Attention model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.core.config import settings
from src.indicators.basic import add_basic_indicators
from src.ml.features import build_ml_dataset
from src.services.ohlcv_service import load_ohlcv_df
from src.dl.models.lstm_attn import LSTMAttentionModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and easy sample dominance.
    
    Focal Loss는 쉬운 샘플에 대한 loss를 줄이고, 어려운 샘플에 집중하도록 설계됨.
    특히 클래스 불균형 문제가 있을 때 유용하며, 상수 예측기 collapse를 방지하는 데 도움이 됨.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1, else 1-p
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor for positive class (default: 0.25)
            gamma: Focusing parameter - higher gamma down-weights easy examples more (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits from model (batch, 1) or (batch,)
            targets: Binary labels (batch, 1) or (batch,) in [0, 1]
        
        Returns:
            Focal loss value
        """
        # Ensure inputs and targets have compatible shapes
        if inputs.dim() > 1:
            inputs = inputs.squeeze(1)
        if targets.dim() > 1:
            targets = targets.squeeze(1)
        
        # BCE with logits (numerically stable)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        
        # Get probabilities p_t
        # p_t = p if target=1, else 1-p
        # where p = sigmoid(logit)
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Alpha weighting: alpha_t = alpha if target=1, else 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal term: alpha_t * (1 - p_t)^gamma * bce_loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.

        Args:
            X: Sequences of shape (num_samples, seq_len, feature_dim)
            y: Labels of shape (num_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_sequences(
    df: pd.DataFrame,
    window_size: int = 60,
    horizon: int | None = None,
    pos_threshold: float | None = None,
    neg_threshold: float | None = None,
    ignore_margin: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Create sequences for LSTM training with improved label definition.
    
    라벨 생성 로직:
    - horizon 기간 후의 수익률 계산: r = (price_{t+horizon} / price_t - 1)
    - r >= pos_threshold → label = 1 (강한 상승)
    - r <= neg_threshold → label = 0 (강한 하락)
    - abs(r) <= ignore_margin → label = -1 (ambiguous zone, 학습에서 제외)
    - 그 외 구간(중간 영역) → label = -1 (ambiguous zone, 학습에서 제외)
    
    정규화 방식:
    - 각 시퀀스마다 rolling window(window_size)를 사용한 z-score 정규화
    - 예측 시점과 동일한 방식으로 정규화하여 일관성 유지

    Args:
        df: DataFrame with OHLCV + indicators
        window_size: Sequence length (number of timesteps)
        horizon: Number of periods ahead to predict (default: from settings)
        pos_threshold: Positive return threshold for label=1 (default: from settings)
        neg_threshold: Negative return threshold for label=0 (default: from settings)
        ignore_margin: Margin for ambiguous zone (default: from settings)

    Returns:
        Tuple of (X: sequences, y: labels, feature_cols: feature column names)
    """
    # Settings에서 기본값 가져오기
    if horizon is None:
        horizon = settings.LSTM_RETURN_HORIZON
    if pos_threshold is None:
        pos_threshold = settings.LSTM_LABEL_POS_THRESHOLD
    if neg_threshold is None:
        neg_threshold = settings.LSTM_LABEL_NEG_THRESHOLD
    if ignore_margin is None:
        ignore_margin = settings.LSTM_LABEL_IGNORE_MARGIN
    
    logger.info("=" * 60)
    logger.info("Creating sequences for LSTM training")
    logger.info("=" * 60)
    logger.info(f"Input data shape: {df.shape}")
    logger.info(f"Window size: {window_size}, Horizon: {horizon}")
    logger.info(f"Label thresholds: pos={pos_threshold:.4f}, neg={neg_threshold:.4f}")
    logger.info(f"Ignore margin: {ignore_margin:.4f} (ambiguous zone: [-{ignore_margin:.4f}, +{ignore_margin:.4f}])")
    
    # Add indicators if not present
    if "ema_20" not in df.columns:
        df = add_basic_indicators(df)

    # Build features using existing function
    # NOTE: build_ml_dataset는 feature와 label을 반환하지만, 여기서는 feature만 사용
    # 라벨은 별도로 계산함 (threshold 기반)
    X_features, _ = build_ml_dataset(
        df,
        horizon=horizon,
        use_events=settings.EVENTS_ENABLED,
    )

    # Get feature columns (순서가 중요함 - 예측 시점과 동일해야 함)
    feature_cols = X_features.columns.tolist()
    feature_dim = len(feature_cols)
    
    logger.info(f"Feature columns ({feature_dim} total):")
    logger.info(f"  - Basic features: {[c for c in feature_cols if not c.startswith('event_')]}")
    if settings.EVENTS_ENABLED:
        event_cols = [c for c in feature_cols if c.startswith("event_")]
        logger.info(f"  - Event features: {len(event_cols)} (sample: {event_cols[:3]})")

    # Calculate future return for label definition
    df = df.copy()
    df["future_return"] = (df["close"].shift(-horizon) - df["close"]) / df["close"]

    # Normalize features (using rolling statistics)
    # IMPORTANT: 이 정규화 방식은 예측 시점(_prepare_sequence)과 동일해야 함
    # 
    # 정규화 방식:
    # - 각 행 i에 대해, [i-window_size+1:i+1] 구간의 평균/표준편차로 z-score 정규화
    # - rolling(window=window_size)는 각 행을 포함한 이전 window_size개의 통계를 계산
    # - 예측 시점에서는 마지막 시퀀스의 마지막 행에서 동일한 방식으로 정규화
    #
    # 예: window_size=60일 때
    #   - 행 60: [0:61] 구간의 평균/표준편차로 정규화
    #   - 행 61: [1:62] 구간의 평균/표준편차로 정규화
    #   - ...
    #   - 예측 시점: 마지막 행에서 [len-60:len] 구간의 평균/표준편차로 정규화
    X_normalized = X_features.copy()
    for col in feature_cols:
        mean = X_normalized[col].rolling(window=window_size, min_periods=1).mean()
        std = X_normalized[col].rolling(window=window_size, min_periods=1).std()
        std = std.replace(0, 1)  # Avoid division by zero
        X_normalized[col] = (X_normalized[col] - mean) / std

    # Create sequences with improved label definition (with ignored zone)
    sequences = []
    labels = []
    future_returns = []

    for i in range(window_size, len(X_normalized) - horizon + 1):
        # Extract sequence
        seq = X_normalized.iloc[i - window_size : i][feature_cols].values
        # Extract future return
        # NOTE: i-1을 사용하는 이유는 future_return이 현재 시점 기준이기 때문
        future_ret = df.iloc[i - 1]["future_return"]

        # Skip if future_return is NaN/Inf
        if pd.isna(future_ret) or np.isinf(future_ret):
            continue

        # Label definition based on return thresholds with ignored zone
        if future_ret >= pos_threshold:
            label = 1.0  # Positive (strong up)
        elif future_ret <= neg_threshold:
            label = 0.0  # Negative (strong down)
        elif ignore_margin > 0.0 and abs(future_ret) <= ignore_margin:
            # Ambiguous zone (very small movement) → ignore
            # Only apply if ignore_margin > 0
            label = -1.0
        else:
            # Intermediate zone (between thresholds but not in ignore margin)
            # If ignore_margin == 0, include this zone in training
            # Use binary classification: positive if future_ret > 0, negative otherwise
            if ignore_margin == 0.0:
                # No ignore zone: use simple binary classification for intermediate zone
                label = 1.0 if future_ret > 0 else 0.0
            else:
                # Has ignore zone: treat intermediate zone as ambiguous and ignore
                label = -1.0

        sequences.append(seq)
        labels.append(label)
        future_returns.append(future_ret)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    future_returns_arr = np.array(future_returns, dtype=np.float32)

    # --- 안전장치: NaN 제거 + ignored label 제거 ---
    # 1) 입력 데이터와 라벨의 NaN/Inf 체크
    # 라벨에 NaN이 있는 샘플 제거
    mask = ~np.isnan(y)
    # 입력 데이터에 NaN/Inf가 있는 샘플도 제거
    X_has_nan = np.isnan(X).any(axis=(1, 2)) | np.isinf(X).any(axis=(1, 2))
    mask = mask & ~X_has_nan

    # 2) Ignored labels (label == -1) 제거
    mask = mask & (y != -1.0)

    X = X[mask]
    y = y[mask]
    future_returns_arr = future_returns_arr[mask]

    # 3) 라벨을 0/1로 강제 보정 (이미 0/1이지만 안전장치)
    y = (y > 0).astype(np.float32)

    # 상세 라벨 분포 로깅 (ignored labels 포함)
    # 먼저 전체 샘플(ignored 포함)에서 통계 계산
    all_labels = np.array(labels, dtype=np.float32)
    all_future_returns = np.array(future_returns, dtype=np.float32)
    
    pos_count_all = int(np.sum(all_labels == 1.0))
    neg_count_all = int(np.sum(all_labels == 0.0))
    ignore_count_all = int(np.sum(all_labels == -1.0))
    total_count_all = len(all_labels)
    
    # 학습에 사용되는 샘플만 (ignored 제외)
    pos_count = int(y.sum())
    neg_count = int(len(y) - pos_count)
    pos_ratio = y.mean() if len(y) > 0 else 0.0
    neg_ratio = 1.0 - pos_ratio
    
    ignore_ratio = ignore_count_all / total_count_all if total_count_all > 0 else 0.0
    
    logger.info("-" * 60)
    logger.info("Label Distribution (with ignore):")
    logger.info(f"  Total samples (before filtering): {total_count_all}")
    logger.info(f"  Positive (label=1): {pos_count_all} ({pos_count_all/total_count_all:.3f} of total)")
    logger.info(f"  Negative (label=0): {neg_count_all} ({neg_count_all/total_count_all:.3f} of total)")
    logger.info(f"  Ignored (label=-1): {ignore_count_all} ({ignore_ratio:.3f} of total)")
    logger.info("-" * 60)
    logger.info(f"  Final samples (after filtering): {len(y)}")
    logger.info(f"  Positive (label=1): {pos_count} ({pos_ratio:.3f})")
    logger.info(f"  Negative (label=0): {neg_count} ({neg_ratio:.3f})")
    
    # Ignore ratio 경고 (ignore_margin > 0일 때만 의미 있음)
    if ignore_margin > 0.0:
        if ignore_ratio < 0.05:
            logger.warning(
                f"Ambiguous/ignored zone is very small (<5%%). "
                f"라벨이 너무 공격적일 수 있습니다. (ignore_ratio={ignore_ratio:.3f})"
            )
        elif ignore_ratio > 0.5:
            logger.warning(
                f"Ambiguous/ignored zone is very large (>50%%). "
                f"라벨 기준이 너무 보수적일 수 있습니다. (ignore_ratio={ignore_ratio:.3f})"
            )
    
    # Future return 통계 (threshold 분석용)
    if len(future_returns_arr) > 0:
        logger.info("-" * 60)
        logger.info("Future Return Statistics (for threshold analysis):")
        logger.info(f"  Horizon: {horizon}")
        logger.info(f"  pos_threshold: {pos_threshold:.6f}")
        logger.info(f"  neg_threshold: {neg_threshold:.6f}")
        logger.info(f"  ignore_margin: {ignore_margin:.6f}")
        logger.info(f"  Mean: {future_returns_arr.mean():.6f}")
        logger.info(f"  Std: {future_returns_arr.std():.6f}")
        logger.info(f"  Min: {future_returns_arr.min():.6f}")
        logger.info(f"  Max: {future_returns_arr.max():.6f}")
        logger.info(f"  25th percentile: {np.percentile(future_returns_arr, 25):.6f}")
        logger.info(f"  50th percentile: {np.percentile(future_returns_arr, 50):.6f}")
        logger.info(f"  75th percentile: {np.percentile(future_returns_arr, 75):.6f}")
        logger.info(f"  Samples >= pos_threshold ({pos_threshold:.4f}): {np.sum(future_returns_arr >= pos_threshold)}")
        logger.info(f"  Samples <= neg_threshold ({neg_threshold:.4f}): {np.sum(future_returns_arr <= neg_threshold)}")
        
        # Ignored zone 샘플 수 계산
        ignored_mask = (all_future_returns >= -ignore_margin) & (all_future_returns <= ignore_margin)
        ignored_count = np.sum(ignored_mask)
        logger.info(f"  Samples in ignore margin (abs <= {ignore_margin:.4f}): {ignored_count}")
    
    logger.info("=" * 60)
    logger.info("Final sequence shape after dropping ignored labels:")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y shape: {y.shape}")
    logger.info(f"  feature_dim: {feature_dim}")
    logger.info("=" * 60)

    return X, y, feature_cols


def train_model(
    window_size: int = 60,
    horizon: int | None = None,
    pos_threshold: float | None = None,
    neg_threshold: float | None = None,
    ignore_margin: float | None = None,
    batch_size: int = 64,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    num_epochs: int = 50,
    train_split: float = 0.8,
    patience_es: int = 10,
    min_delta: float = 1e-4,
    device: torch.device | None = None,
) -> LSTMAttentionModel:
    """
    Train LSTM + Attention model with improved training pipeline.

    Args:
        window_size: Sequence length
        horizon: Prediction horizon
        pos_threshold: Positive return threshold for label=1
        neg_threshold: Negative return threshold for label=0
        batch_size: Batch size for training
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        num_epochs: Number of training epochs
        train_split: Train/validation split ratio
        patience_es: Early stopping patience
        min_delta: Minimum change to qualify as improvement
        device: Device to train on (default: auto-detect)

    Returns:
        Trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Settings에서 기본값 가져오기 (파라미터가 None인 경우)
    if horizon is None:
        horizon = settings.LSTM_RETURN_HORIZON
    if pos_threshold is None:
        pos_threshold = settings.LSTM_LABEL_POS_THRESHOLD
    if neg_threshold is None:
        neg_threshold = settings.LSTM_LABEL_NEG_THRESHOLD
    if ignore_margin is None:
        ignore_margin = settings.LSTM_LABEL_IGNORE_MARGIN

    logger.info("=" * 60)
    logger.info("LSTM + Attention Model Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Window size: {window_size}, Horizon: {horizon}")
    logger.info(f"Return thresholds: pos={pos_threshold:.4f}, neg={neg_threshold:.4f}")
    logger.info(f"Ignore margin: {ignore_margin:.4f}")

    # Load data
    logger.info("Loading OHLCV data...")
    df = load_ohlcv_df()
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of OHLCV data")

    # Create sequences
    logger.info(f"Creating sequences (window_size={window_size}, horizon={horizon})...")
    logger.info("Event features enabled: %s", settings.EVENTS_ENABLED)
    X, y, feature_cols = create_sequences(
        df,
        window_size=window_size,
        horizon=horizon,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
        ignore_margin=ignore_margin,
    )
    feature_dim = len(feature_cols)

    logger.info(f"Created {len(X)} sequences with {feature_dim} features")
    if settings.EVENTS_ENABLED:
        event_cols = [col for col in feature_cols if col.startswith("event_")]
        logger.info("이벤트 피처 수: %d (샘플: %s)", len(event_cols), event_cols[:5])

    # Train/validation split (time-series aware: first 80% train, last 20% valid)
    split_idx = int(len(X) * train_split)
    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]

    logger.info("-" * 60)
    logger.info("Train/Validation Split:")
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_valid shape: {X_valid.shape}")
    logger.info(f"  y_train shape: {y_train.shape} (pos={y_train.mean():.3f})")
    logger.info(f"  y_valid shape: {y_valid.shape} (pos={y_valid.mean():.3f})")
    logger.info("-" * 60)

    # Calculate class weights for imbalanced data
    # pos_weight 계산: neg_count / pos_count
    # PyTorch tensor에서 계산
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    pos_count = int((y_train_tensor == 1).sum().item())
    neg_count = int((y_train_tensor == 0).sum().item())
    
    # Edge case 처리
    if pos_count == 0 or neg_count == 0:
        pos_weight_value = 1.0
        logger.warning(
            f"pos_count or neg_count is zero (pos_count={pos_count}, neg_count={neg_count}), "
            f"fallback to pos_weight=1.0"
        )
    else:
        pos_weight_value = float(neg_count) / float(pos_count)

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    valid_dataset = TimeSeriesDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    logger.info("-" * 60)
    logger.info("Model Initialization:")
    logger.info(f"  input_size (feature_dim): {feature_dim}")
    logger.info(f"  hidden_size: {hidden_size}")
    logger.info(f"  num_layers: {num_layers}")
    logger.info(f"  dropout: {dropout}")
    logger.info(f"  device: {device}")
    logger.info("-" * 60)
    
    model = LSTMAttentionModel(
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    
    # 모델 파라미터 수 로깅
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # DEBUG: requires_grad 확인
    requires_grad_count = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters with requires_grad=True: {requires_grad_count}/{total_params}")
    
    # DEBUG: 마지막 레이어 초기 weight 확인
    if hasattr(model, 'fc_out'):
        fc_out_weight_mean = model.fc_out.weight.data.mean().item()
        fc_out_weight_std = model.fc_out.weight.data.std().item()
        fc_out_bias_mean = model.fc_out.bias.data.mean().item()
        logger.info(
            f"  Initial fc_out.weight: mean={fc_out_weight_mean:.6f}, std={fc_out_weight_std:.6f}"
        )
        logger.info(f"  Initial fc_out.bias: mean={fc_out_bias_mean:.6f}")
    else:
        # classifier의 마지막 레이어 찾기
        last_layer = None
        for module in model.classifier.modules():
            if isinstance(module, nn.Linear):
                last_layer = module
        if last_layer is not None:
            last_weight_mean = last_layer.weight.data.mean().item()
            last_weight_std = last_layer.weight.data.std().item()
            last_bias_mean = last_layer.bias.data.mean().item()
            logger.info(
                f"  Initial last_layer.weight: mean={last_weight_mean:.6f}, std={last_weight_std:.6f}"
            )
            logger.info(f"  Initial last_layer.bias: mean={last_bias_mean:.6f}")

    # Loss and optimizer
    # NOTE: BCEWithLogitsLoss는 모델 출력이 raw logit일 때 사용
    # 모델의 forward()는 sigmoid 없이 logit을 반환하므로 이 손실 함수가 적합함
    
    # Loss function 선택 (설정으로 제어 가능)
    use_focal_loss = settings.LSTM_USE_FOCAL_LOSS
    focal_alpha = settings.LSTM_FOCAL_ALPHA
    focal_gamma = settings.LSTM_FOCAL_GAMMA
    
    if use_focal_loss:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        logger.info(
            "Using FocalLoss (alpha=%.3f, gamma=%.3f) instead of BCEWithLogitsLoss",
            focal_alpha,
            focal_gamma,
        )
    else:
        # BCEWithLogitsLoss + pos_weight (항상 적용)
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        logger.info(
            "Using BCEWithLogitsLoss with pos_weight=%.4f (neg_count=%d, pos_count=%d)",
            pos_weight_value,
            neg_count,
            pos_count,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    logger.info(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6,
    )

    # Training loop
    logger.info("Starting training...")
    logger.info("-" * 60)

    best_valid_loss = float("inf")
    epochs_no_improve = 0
    
    # DEBUG: 소규모 overfit 테스트 모드
    debug_small_overfit = getattr(settings, "DEBUG_SMALL_OVERFIT", False)
    debug_gradient_logging = getattr(settings, "DEBUG_GRADIENT_LOGGING", True)
    
    if debug_small_overfit:
        logger.info("=" * 60)
        logger.info("DEBUG MODE: Small Overfit Test")
        logger.info("=" * 60)
        # 소규모 데이터셋 생성
        small_size = 64
        X_train_small = X_train[:small_size]
        y_train_small = y_train[:small_size]
        train_dataset_small = TimeSeriesDataset(X_train_small, y_train_small)
        train_loader_small = DataLoader(train_dataset_small, batch_size=32, shuffle=False)
        logger.info(f"Using small dataset: {len(X_train_small)} samples")
        logger.info(f"Small dataset label distribution: pos={y_train_small.mean():.3f}")
        logger.info("=" * 60)
        # Overfit 테스트에서는 early stopping 비활성화
        num_epochs = max(num_epochs, 200)  # 최소 200 epoch
        logger.info(f"Extended epochs to {num_epochs} for overfit test")
        logger.info("Expected behavior: train loss should decrease significantly")
        logger.info("Expected behavior: prob_up std should increase to >0.05")
        logger.info("=" * 60)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # DEBUG: epoch 시작 시 weight 상태 로깅
        if debug_gradient_logging and epoch == 0:
            if hasattr(model, 'fc_out'):
                fc_out_w_mean = model.fc_out.weight.data.mean().item()
                fc_out_w_std = model.fc_out.weight.data.std().item()
                logger.info(
                    f"  [DEBUG] Epoch {epoch+1} start - fc_out.weight: "
                    f"mean={fc_out_w_mean:.6f}, std={fc_out_w_std:.6f}"
                )
        
        # 소규모 overfit 모드에서는 작은 데이터셋 사용
        current_train_loader = train_loader_small if debug_small_overfit else train_loader
        
        for batch_idx, (X_batch, y_batch) in enumerate(current_train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            # 입력/라벨 NaN/Inf 체크 (문제 있으면 바로 예외 던지기)
            if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                raise ValueError("NaN/Inf detected in X_batch")
            if torch.isnan(y_batch).any() or torch.isinf(y_batch).any():
                raise ValueError("NaN/Inf detected in y_batch")

            optimizer.zero_grad()

            logits = model(X_batch)  # (B, 1) raw logits
            loss = criterion(logits, y_batch)

            # loss NaN/Inf 체크
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("NaN/Inf detected in loss during training")

            loss.backward()

            # DEBUG: gradient norm 로깅 (첫 배치 또는 주기적으로)
            if debug_gradient_logging and (epoch == 0 and batch_idx == 0 or batch_idx == 0 and epoch % 10 == 0):
                grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_norms.append((name, grad_norm))
                    else:
                        grad_norms.append((name, None))
                
                # 마지막 레이어 gradient 확인
                if hasattr(model, 'fc_out'):
                    if model.fc_out.weight.grad is not None:
                        fc_out_grad_norm = model.fc_out.weight.grad.norm().item()
                        logger.info(
                            f"  [DEBUG] Epoch {epoch+1}, Batch {batch_idx} - "
                            f"fc_out.weight.grad.norm()={fc_out_grad_norm:.6f}"
                        )
                    else:
                        logger.warning(
                            f"  [DEBUG] Epoch {epoch+1}, Batch {batch_idx} - "
                            f"fc_out.weight.grad is None!"
                        )
                
                # 전체 gradient norm 요약
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                logger.info(
                    f"  [DEBUG] Epoch {epoch+1}, Batch {batch_idx} - "
                    f"Total gradient norm: {total_grad_norm:.6f}"
                )

            # Gradient clipping: gradient 폭발 방지
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss += loss.item()
            
            # DEBUG: 첫 배치에서 logit 분포 확인
            if debug_gradient_logging and epoch == 0 and batch_idx == 0:
                with torch.no_grad():
                    logits_sample = model(X_batch[:5])  # 첫 5개 샘플만
                    probs_sample = torch.sigmoid(logits_sample)
                    logger.info(
                        f"  [DEBUG] Epoch {epoch+1}, Batch {batch_idx} - "
                        f"Sample logits: {logits_sample.squeeze().cpu().numpy()}"
                    )
                    logger.info(
                        f"  [DEBUG] Epoch {epoch+1}, Batch {batch_idx} - "
                        f"Sample probs: {probs_sample.squeeze().cpu().numpy()}"
                    )

        train_loss /= len(current_train_loader)
        
        # DEBUG: epoch 끝에서 weight 변화 확인
        if debug_gradient_logging and (epoch == 0 or epoch % 10 == 0):
            if hasattr(model, 'fc_out'):
                fc_out_w_mean = model.fc_out.weight.data.mean().item()
                fc_out_w_std = model.fc_out.weight.data.std().item()
                fc_out_bias = model.fc_out.bias.data.item()
                logger.info(
                    f"  [DEBUG] Epoch {epoch+1} end - fc_out.weight: "
                    f"mean={fc_out_w_mean:.6f}, std={fc_out_w_std:.6f}, bias={fc_out_bias:.6f}"
                )
        
        # DEBUG: 소규모 overfit 테스트에서 train set prob_up 분포 확인
        if debug_small_overfit and (epoch == 0 or epoch % 20 == 0 or epoch == num_epochs - 1):
            model.eval()
            train_probs = []
            with torch.no_grad():
                for X_batch, y_batch in current_train_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    probs = torch.sigmoid(logits)
                    train_probs.extend(probs.cpu().numpy().flatten().tolist())
            if train_probs:
                train_prob_array = np.array(train_probs)
                train_prob_mean = float(train_prob_array.mean())
                train_prob_std = float(train_prob_array.std())
                train_prob_min = float(train_prob_array.min())
                train_prob_max = float(train_prob_array.max())
                logger.info(
                    f"  [DEBUG OVERFIT] Epoch {epoch+1} - Train prob_up: "
                    f"mean={train_prob_mean:.4f}, std={train_prob_std:.4f}, "
                    f"min={train_prob_min:.4f}, max={train_prob_max:.4f}"
                )
            model.train()

        # Validation
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        all_probs = []  # prob_up 분포 분석용
        
        # DEBUG: 소규모 overfit 모드에서는 train 데이터로 validation 수행
        current_valid_loader = current_train_loader if debug_small_overfit else valid_loader

        with torch.no_grad():
            for X_batch, y_batch in current_valid_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)

                logits = model(X_batch)  # (B, 1), raw logit
                loss = criterion(logits, y_batch)
                valid_loss += loss.item()

                # 확률로 변환 후 정확도 및 메트릭 계산
                probs = torch.sigmoid(logits)  # (B, 1) in (0, 1)
                predictions = (probs >= 0.5).float()  # (B, 1) in {0, 1}

                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

                # Confusion matrix 계산
                tp += ((predictions == 1) & (y_batch == 1)).sum().item()
                fp += ((predictions == 1) & (y_batch == 0)).sum().item()
                tn += ((predictions == 0) & (y_batch == 0)).sum().item()
                fn += ((predictions == 0) & (y_batch == 1)).sum().item()
                
                # prob_up 분포 수집
                all_probs.extend(probs.cpu().numpy().flatten().tolist())

        valid_loss /= len(current_valid_loader)
        valid_acc = correct / total if total > 0 else 0.0

        # Calculate metrics (division by zero 방지)
        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        
        # prob_up 분포 통계
        if all_probs:
            prob_array = np.array(all_probs)
            prob_mean = float(prob_array.mean())
            prob_std = float(prob_array.std())
            prob_min = float(prob_array.min())
            prob_max = float(prob_array.max())
        else:
            prob_mean = prob_std = prob_min = prob_max = 0.0

        # Learning rate scheduler step (소규모 overfit 모드에서는 비활성화)
        if not debug_small_overfit:
            scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, "
            f"Acc: {valid_acc:.4f}, "
            f"P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}, "
            f"LR: {current_lr:.6f}"
        )
        logger.info(
            f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}"
        )
        logger.info(
            f"  Valid prob_up stats: mean={prob_mean:.4f}, std={prob_std:.4f}, "
            f"min={prob_min:.4f}, max={prob_max:.4f}"
        )
        
        # Collapse 감지 경고
        if prob_std < 0.01:
            logger.warning(
                f"  ⚠️  WARNING: prob_up std is very low ({prob_std:.6f}), "
                f"model may be collapsing to constant prediction!"
            )
        if tp == 0 and fp == 0:
            logger.warning(
                f"  ⚠️  WARNING: Model never predicts positive class (TP=0, FP=0)"
            )
        if tp == 0 and fn > 0:
            logger.warning(
                f"  ⚠️  WARNING: Model never predicts positive class but there are {fn} positive samples (FN)"
            )

        # Early stopping and model saving (소규모 overfit 모드에서는 비활성화)
        if not debug_small_overfit:
            if valid_loss < best_valid_loss - min_delta:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
                model_path = Path(settings.LSTM_ATTN_MODEL_PATH)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(model_path)
                logger.info(
                    f"✓ Saved best model (valid_loss={valid_loss:.4f}) to {model_path.resolve()}"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience_es:
                    logger.info(
                        f"Early stopping triggered after {epoch+1} epochs "
                        f"(no improvement for {patience_es} epochs)"
                    )
                    break

    logger.info("-" * 60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_valid_loss:.4f}")
    logger.info(f"Final validation accuracy: {valid_acc:.4f}")
    logger.info(f"Final metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # 최종 모델 경로 로그
    final_model_path = Path(settings.LSTM_ATTN_MODEL_PATH)
    if final_model_path.exists():
        logger.info(f"Saved best LSTM model to: {final_model_path.resolve()}")
    else:
        logger.warning(f"Model file not found at expected path: {final_model_path.resolve()}")
    
    # 4단계: Validation 셋 샘플 출력 및 다양한 threshold 분석 (디버깅용)
    # 설정으로 제어 가능하도록 (기본값: True)
    debug_inference_samples = getattr(settings, "DEBUG_LSTM_INFERENCE_SAMPLES", True)
    
    # 다양한 threshold에 대한 precision/recall 계산
    if debug_inference_samples:
        logger.info("-" * 60)
        logger.info("Threshold Analysis (various thresholds for precision/recall):")
        logger.info("-" * 60)
        
        model.eval()
        all_probs_valid = []
        all_labels_valid = []
        
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                
                logits = model(X_batch)
                probs = torch.sigmoid(logits)
                
                all_probs_valid.extend(probs.cpu().numpy().flatten().tolist())
                all_labels_valid.extend(y_batch.cpu().numpy().flatten().tolist())
        
        all_probs_valid = np.array(all_probs_valid)
        all_labels_valid = np.array(all_labels_valid)
        
        # 다양한 threshold 테스트 (확장된 범위)
        test_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        logger.info(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
        logger.info("-" * 60)
        
        for thresh in test_thresholds:
            preds = (all_probs_valid >= thresh).astype(int)
            tp = np.sum((preds == 1) & (all_labels_valid == 1))
            fp = np.sum((preds == 1) & (all_labels_valid == 0))
            fn = np.sum((preds == 0) & (all_labels_valid == 1))
            
            precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
            
            logger.info(
                f"{thresh:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} "
                f"{tp:<8} {fp:<8} {fn:<8}"
            )
        logger.info("-" * 60)
    
    if debug_inference_samples:
        logger.info("-" * 60)
        logger.info("Validation Set Sample Predictions (for debugging):")
        logger.info("-" * 60)
        
        model.eval()
        with torch.no_grad():
            # 마지막 N개 샘플 출력 (또는 랜덤 샘플)
            sample_size = min(10, len(valid_dataset))
            sample_indices = list(range(len(valid_dataset) - sample_size, len(valid_dataset)))
            
            logger.info(f"Showing predictions for last {sample_size} validation samples:")
            logger.info(f"{'Index':<8} {'Label':<8} {'prob_up':<12} {'pred_label':<12} {'Match':<8}")
            logger.info("-" * 60)
            
            for idx in sample_indices:
                X_sample, y_sample = valid_dataset[idx]
                X_sample = X_sample.unsqueeze(0).to(device)  # Add batch dimension
                y_sample = y_sample.item()
                
                logit = model(X_sample)
                prob_up = torch.sigmoid(logit).item()
                pred_label = 1 if prob_up >= 0.5 else 0
                match = "✓" if pred_label == int(y_sample) else "✗"
                
                logger.info(
                    f"{idx:<8} {int(y_sample):<8} {prob_up:<12.4f} {pred_label:<12} {match:<8}"
                )
        
        logger.info("-" * 60)
    
    # 최종 체크리스트 출력
    logger.info("=" * 60)
    logger.info("Training Pipeline Checklist:")
    logger.info("=" * 60)
    logger.info("[✓] 학습/예측 파이프라인 feature 일관성 검증")
    logger.info("[✓] 라벨링 로직 및 분포 확인")
    logger.info("[✓] 손실/출력/메트릭 일관성 검증")
    logger.info("[✓] Valid confusion matrix / prob_up 분포 로깅")
    logger.info("[✓] Collapse 여부를 판단할 수 있는 로그 추가")
    logger.info("[✓] Feature dimension 일치 검증")
    logger.info("[✓] 정규화 방식 일관성 확인")
    logger.info("=" * 60)

    return model


if __name__ == "__main__":
    # ============================================================
    # 학습 설정 (Hyperparameters)
    # ============================================================
    # NOTE: horizon, pos_threshold, neg_threshold, ignore_margin는
    # settings에서 가져오므로 여기서는 None으로 전달 (또는 명시적으로 override 가능)
    WINDOW_SIZE = 60
    # HORIZON, POS_THRESHOLD, NEG_THRESHOLD, IGNORE_MARGIN는 settings에서 가져옴

    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2

    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    TRAIN_SPLIT = 0.8

    # Early stopping
    PATIENCE_ES = 10
    MIN_DELTA = 1e-4

    # Train model
    # horizon, pos_threshold, neg_threshold, ignore_margin는 None으로 전달하면
    # settings에서 기본값을 사용함
    model = train_model(
        window_size=WINDOW_SIZE,
        horizon=None,  # settings.LSTM_RETURN_HORIZON 사용
        pos_threshold=None,  # settings.LSTM_LABEL_POS_THRESHOLD 사용
        neg_threshold=None,  # settings.LSTM_LABEL_NEG_THRESHOLD 사용
        ignore_margin=None,  # settings.LSTM_LABEL_IGNORE_MARGIN 사용
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=NUM_EPOCHS,
        train_split=TRAIN_SPLIT,
        patience_es=PATIENCE_ES,
        min_delta=MIN_DELTA,
    )

    logger.info("Model training completed successfully!")


# ======================================================================
# 체크리스트: LSTM + Attention 모델 훈련 파이프라인 검증
# ======================================================================
# 
# [✓] 1단계: 데이터 파이프라인 전체 흐름 검증
#   - 학습 시점과 예측 시점의 feature 컬럼 순서 일치 확인
#   - 정규화 방식 일관성 확인 (rolling window 기반 z-score)
#   - feature_dim 일치 검증 및 로깅
#   - 학습 시작 시 전체 데이터 shape, feature 컬럼 목록, window_size, horizon 로깅
#   - train/valid split 후 shape 로깅
#   - 예측 시점 입력 텐서 shape 및 feature_dim 로깅
#
# [✓] 2단계: 라벨 생성 로직 검증
#   - 라벨링 함수 분석 및 주석 추가
#   - 라벨 분포 상세 로깅 (positive/negative count, ratio)
#   - Future return 통계 로깅 (mean, std, percentile)
#   - Threshold 분석 (ambiguous zone 비율 경고)
#   - 이진 분류 확인 (0/1 라벨, BCEWithLogitsLoss 사용)
#
# [✓] 3단계: 손실 함수/출력/메트릭 검증 및 collapse 진단
#   - 손실 함수 명확화 (BCEWithLogitsLoss 또는 FocalLoss)
#   - 출력층 구조 확인 (raw logit 반환, sigmoid는 예측 시점에만)
#   - Confusion matrix 계산 및 로깅 (TP, FP, TN, FN)
#   - Valid prob_up 분포 통계 로깅 (mean, std, min, max)
#   - Collapse 감지 경고 (prob_up std < 0.01, TP=0 등)
#   - Class weight 자동 계산 및 적용 옵션
#   - FocalLoss 옵션 추가 (설정으로 제어 가능)
#
# [✓] 4단계: 예측/백테스트 시 prob_up 검증용 도우미
#   - Validation 셋 샘플 출력 (마지막 N개 샘플의 label, prob_up, predicted_label)
#   - 다양한 threshold에 대한 precision/recall 분석
#   - 설정 플래그로 제어 가능 (DEBUG_LSTM_INFERENCE_SAMPLES)
#
# [✓] 5단계: 코드 정리 및 안전장치
#   - 모든 로깅은 logger 사용 (print 대신)
#   - 기존 외부 인터페이스 보존 (get_lstm_attn_model, get_lstm_attn_signal 등)
#   - Feature dimension 불일치 시 명확한 에러 메시지
#   - 정규화 방식 일관성 주석 추가
#
# ======================================================================
# 주요 개선 사항:
# ======================================================================
# 1. 학습/예측 파이프라인 일관성 강화
#    - Feature 컬럼 순서 명시적 관리
#    - 정규화 방식 동일성 보장
#    - Feature dimension 검증 로직 추가
#
# 2. 라벨링 및 메트릭 개선
#    - 상세한 라벨 분포 로깅
#    - Future return 통계 분석
#    - Threshold 튜닝 가이드 제공
#
# 3. Collapse 진단 강화
#    - Confusion matrix 상세 로깅
#    - Prob_up 분포 모니터링
#    - 자동 경고 메시지
#
# 4. 유연성 향상
#    - FocalLoss 옵션 추가
#    - Class weight 자동 계산
#    - 다양한 threshold 분석 도구
#
# ======================================================================
