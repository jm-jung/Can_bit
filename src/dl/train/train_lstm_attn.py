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
    horizon: int = 5,
    pos_threshold: float = 0.0015,
    neg_threshold: float = -0.0015,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Create sequences for LSTM training with improved label definition.

    Args:
        df: DataFrame with OHLCV + indicators
        window_size: Sequence length (number of timesteps)
        horizon: Number of periods ahead to predict
        pos_threshold: Positive return threshold for label=1 (default: 0.0015)
        neg_threshold: Negative return threshold for label=0 (default: -0.0015)

    Returns:
        Tuple of (X: sequences, y: labels, feature_cols: feature column names)
    """
    # Add indicators if not present
    if "ema_20" not in df.columns:
        df = add_basic_indicators(df)

    # Build features using existing function
    X_features, _ = build_ml_dataset(df, horizon=horizon)

    # Get feature columns
    feature_cols = X_features.columns.tolist()
    feature_dim = len(feature_cols)

    # Calculate future return for label definition
    df = df.copy()
    df["future_return"] = (df["close"].shift(-horizon) - df["close"]) / df["close"]

    # Normalize features (using rolling statistics)
    X_normalized = X_features.copy()
    for col in feature_cols:
        mean = X_normalized[col].rolling(window=window_size, min_periods=1).mean()
        std = X_normalized[col].rolling(window=window_size, min_periods=1).std()
        std = std.replace(0, 1)  # Avoid division by zero
        X_normalized[col] = (X_normalized[col] - mean) / std

    # Create sequences with improved label definition
    sequences = []
    labels = []
    future_returns = []

    for i in range(window_size, len(X_normalized) - horizon + 1):
        # Extract sequence
        seq = X_normalized.iloc[i - window_size : i][feature_cols].values
        # Extract future return
        future_ret = df.iloc[i - 1]["future_return"]

        # Skip if future_return is NaN/Inf
        if pd.isna(future_ret) or np.isinf(future_ret):
            continue

        # Label definition based on return thresholds
        if future_ret >= pos_threshold:
            label = 1.0  # Positive (up)
        elif future_ret <= neg_threshold:
            label = 0.0  # Negative (down)
        else:
            # Ambiguous zone: skip this sample
            continue

        sequences.append(seq)
        labels.append(label)
        future_returns.append(future_ret)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # --- 안전장치: NaN 제거 + 라벨을 0/1 float로 보정 ---
    # 1) 입력 데이터와 라벨의 NaN/Inf 체크
    # 라벨에 NaN이 있는 샘플 제거
    mask = ~np.isnan(y)
    # 입력 데이터에 NaN/Inf가 있는 샘플도 제거
    X_has_nan = np.isnan(X).any(axis=(1, 2)) | np.isinf(X).any(axis=(1, 2))
    mask = mask & ~X_has_nan

    X = X[mask]
    y = y[mask]

    # 2) 라벨을 0/1로 강제 보정 (이미 0/1이지만 안전장치)
    y = (y > 0).astype(np.float32)

    # Log label distribution
    pos_ratio = y.mean()
    neg_ratio = 1.0 - pos_ratio
    logger.info(
        f"Label distribution: pos={pos_ratio:.3f}, neg={neg_ratio:.3f}, "
        f"total={len(y)} samples"
    )

    return X, y, feature_cols


def train_model(
    window_size: int = 60,
    horizon: int = 5,
    pos_threshold: float = 0.0015,
    neg_threshold: float = -0.0015,
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

    logger.info("=" * 60)
    logger.info("LSTM + Attention Model Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Window size: {window_size}, Horizon: {horizon}")
    logger.info(f"Return thresholds: pos={pos_threshold:.4f}, neg={neg_threshold:.4f}")

    # Load data
    logger.info("Loading OHLCV data...")
    df = load_ohlcv_df()
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of OHLCV data")

    # Create sequences
    logger.info(f"Creating sequences (window_size={window_size}, horizon={horizon})...")
    X, y, feature_cols = create_sequences(
        df,
        window_size=window_size,
        horizon=horizon,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
    )
    feature_dim = len(feature_cols)

    logger.info(f"Created {len(X)} sequences with {feature_dim} features")

    # Train/validation split (time-series aware: first 80% train, last 20% valid)
    split_idx = int(len(X) * train_split)
    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]

    logger.info(f"Train samples: {len(X_train)}, Valid samples: {len(X_valid)}")

    # Calculate class weights for imbalanced data
    pos_ratio_train = y_train.mean()
    neg_ratio_train = 1.0 - pos_ratio_train

    if 0.35 <= pos_ratio_train <= 0.65:
        # Balanced enough, use default loss
        pos_weight = None
        logger.info("Label distribution is balanced, using default loss")
    else:
        # Imbalanced, apply class weight
        pos_weight = neg_ratio_train / max(pos_ratio_train, 1e-3)
        logger.info(
            f"Label distribution imbalanced (pos={pos_ratio_train:.3f}), "
            f"applying pos_weight={pos_weight:.3f}"
        )

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    valid_dataset = TimeSeriesDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    logger.info(f"Initializing model: hidden_size={hidden_size}, num_layers={num_layers}")
    model = LSTMAttentionModel(
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Loss and optimizer
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

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

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
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

            # Gradient clipping: gradient 폭발 방지
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
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

                # Precision, Recall, F1 계산
                tp += ((predictions == 1) & (y_batch == 1)).sum().item()
                fp += ((predictions == 1) & (y_batch == 0)).sum().item()
                fn += ((predictions == 0) & (y_batch == 1)).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc = correct / total if total > 0 else 0.0

        # Calculate metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Learning rate scheduler step
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

        # Early stopping and model saving
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

    return model


if __name__ == "__main__":
    # ============================================================
    # 학습 설정 (Hyperparameters)
    # ============================================================
    WINDOW_SIZE = 60
    HORIZON = 5
    POS_THRESHOLD = 0.0015  # +0.15%
    NEG_THRESHOLD = -0.0015  # -0.15%

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
    model = train_model(
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        pos_threshold=POS_THRESHOLD,
        neg_threshold=NEG_THRESHOLD,
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
