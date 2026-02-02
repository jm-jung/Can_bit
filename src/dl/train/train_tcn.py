"""
Training script for TCN (Temporal Convolutional Network) model.

This script trains a TCN model for 3-class classification (FLAT/LONG/SHORT)
using the same dataset/feature pipeline as LSTM-Attn for fair comparison.

Usage:
    python -m src.dl.train.train_tcn --symbol BTCUSDT --timeframe 5m --epochs 50 --out-model models/tcn_v1.pt
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.core.config import settings
from src.dl.models.tcn import TCNModel
from src.dl.data.split import make_time_series_splits, log_split_summary
from src.dl.data.labels import LstmClassIndex
from src.dl.train.train_lstm_attn import create_sequences
from src.services.ohlcv_service import load_ohlcv_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Time series dataset for TCN training."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Sequences array of shape (N, seq_len, feature_dim)
            y: Labels array of shape (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    window_size: int = 60,
    horizon: int | None = None,
    pos_threshold: float | None = None,
    neg_threshold: float | None = None,
    ignore_margin: float | None = None,
    batch_size: int = 64,
    num_channels: list[int] | None = None,
    kernel_size: int = 3,
    dropout: float = 0.2,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    num_epochs: int = 50,
    train_split: float = 0.8,
    patience_es: int = 10,
    min_delta: float = 1e-4,
    device: torch.device | None = None,
    out_model_path: Path | str | None = None,
) -> tuple[TCNModel, float, Path]:
    """
    Train TCN model.
    
    Args:
        window_size: Sequence length
        horizon: Prediction horizon
        pos_threshold: Positive return threshold for label=1
        neg_threshold: Negative return threshold for label=0
        ignore_margin: Margin for ambiguous zone
        batch_size: Batch size for training
        num_channels: List of channel sizes for TCN layers
        kernel_size: Convolution kernel size
        dropout: Dropout rate
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        num_epochs: Number of training epochs
        train_split: Train/validation split ratio
        patience_es: Early stopping patience
        min_delta: Minimum change to qualify as improvement
        device: Device to train on (default: auto-detect)
        out_model_path: Path to save the trained model
        
    Returns:
        Tuple of (model, best_val_loss, best_model_path)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Settings에서 기본값 가져오기
    if horizon is None:
        horizon = settings.LSTM_RETURN_HORIZON
    if pos_threshold is None:
        pos_threshold = settings.LSTM_LABEL_POS_THRESHOLD
    if neg_threshold is None:
        neg_threshold = settings.LSTM_LABEL_NEG_THRESHOLD
    if ignore_margin is None:
        ignore_margin = settings.LSTM_LABEL_IGNORE_MARGIN
    
    if num_channels is None:
        num_channels = [64, 64, 64, 64]
    
    if out_model_path is None:
        out_model_path = Path("models/tcn_v1.pt")
    else:
        out_model_path = Path(out_model_path)
    
    logger.info("=" * 60)
    logger.info("TCN Model Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Window size: {window_size}, Horizon: {horizon}")
    logger.info(f"Return thresholds: pos={pos_threshold:.4f}, neg={neg_threshold:.4f}")
    logger.info(f"Ignore margin: {ignore_margin:.4f}")
    logger.info(f"Model path: {out_model_path.resolve()}")
    
    # Load data
    logger.info("Loading OHLCV data...")
    df = load_ohlcv_df()
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of OHLCV data")
    
    # Create sequences (same pipeline as LSTM-Attn)
    logger.info(f"Creating sequences (window_size={window_size}, horizon={horizon})...")
    X, y, feature_cols, meta = create_sequences(
        df,
        window_size=window_size,
        horizon=horizon,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
        ignore_margin=ignore_margin,
        debug_inspect=False,
    )
    feature_dim = len(feature_cols)
    
    logger.info(f"Created {len(X)} sequences with {feature_dim} features")
    
    # Train/validation/test split
    splits = make_time_series_splits(
        X,
        y,
        train_ratio=0.7,
        valid_ratio=0.15,
        min_test_samples=200,
        meta={"future_returns": meta["future_returns"]},
    )
    
    X_train = splits["data"]["X_train"]
    y_train = splits["data"]["y_train"]
    X_valid = splits["data"]["X_valid"]
    y_valid = splits["data"]["y_valid"]
    X_test = splits["data"]["X_test"]
    y_test = splits["data"]["y_test"]
    
    log_split_summary(
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        logger=logger,
    )
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    valid_dataset = TimeSeriesDataset(X_valid, y_valid)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Compute class weights for imbalanced data
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    flat_count = int((y_train_tensor == LstmClassIndex.FLAT).sum().item())
    long_count = int((y_train_tensor == LstmClassIndex.LONG).sum().item())
    short_count = int((y_train_tensor == LstmClassIndex.SHORT).sum().item())
    total_count = flat_count + long_count + short_count
    
    if total_count > 0:
        class_weights = torch.tensor([
            total_count / (3.0 * max(flat_count, 1)),
            total_count / (3.0 * max(long_count, 1)),
            total_count / (3.0 * max(short_count, 1)),
        ], dtype=torch.float32)
    else:
        class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    
    logger.info(f"Class weights: FLAT={class_weights[0]:.4f}, LONG={class_weights[1]:.4f}, SHORT={class_weights[2]:.4f}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Initialize model
    logger.info("-" * 60)
    logger.info("Model Initialization:")
    logger.info(f"  input_size (feature_dim): {feature_dim}")
    logger.info(f"  num_channels: {num_channels}")
    logger.info(f"  kernel_size: {kernel_size}")
    logger.info(f"  dropout: {dropout:.4f}")
    logger.info(f"  num_classes: 3")
    logger.info("-" * 60)
    
    model = TCNModel(
        input_size=feature_dim,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        num_classes=3,
    ).to(device)
    
    # Verify model output dimension
    with torch.no_grad():
        dummy_input = torch.zeros(1, window_size, feature_dim, device=device)
        dummy_output = model(dummy_input)
        actual_output_dim = dummy_output.shape[1]
        if actual_output_dim != 3:
            raise ValueError(
                f"Model output dimension mismatch: expected 3, got {actual_output_dim}"
            )
        logger.info(f"Model output dimension verified: {actual_output_dim}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    logger.info(f"Optimizer: AdamW (lr={learning_rate:.6f}, weight_decay={weight_decay:.6f})")
    
    # Learning rate scheduler
    # Note: verbose parameter may not be available in older torch versions
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6,
        )
    except TypeError:
        # Fallback for torch versions without verbose parameter
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    
    # Training loop
    logger.info("Starting training...")
    logger.info("-" * 60)
    
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            logits = model(X_batch)  # (B, 3)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            pred = logits.argmax(dim=1)
            train_correct += (pred == y_batch).sum().item()
            train_total += len(y_batch)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item()
                
                pred = logits.argmax(dim=1)
                val_correct += (pred == y_batch).sum().item()
                val_total += len(y_batch)
        
        val_loss /= len(valid_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s): "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        
        if epochs_no_improve >= patience_es:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(f"Loaded best model (val_loss={best_val_loss:.4f})")
    
    # Save model
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out_model_path)
    logger.info(f"Model saved to: {out_model_path.resolve()}")
    
    return model, best_val_loss, out_model_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train TCN model for 3-class classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol (default: from settings)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Timeframe (default: from settings)",
    )
    parser.add_argument(
        "--feature-preset",
        type=str,
        default="extended_safe",
        help="Feature preset (default: extended_safe)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sequence length / window size (default: 60)",
    )
    parser.add_argument(
        "--horizon-bars",
        type=int,
        default=None,
        help="Prediction horizon in bars (default: from settings)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--out-model",
        type=str,
        default="models/tcn_v1.pt",
        help="Output model path (default: models/tcn_v1.pt)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Train model
    model, best_val_loss, best_model_path = train_model(
        window_size=args.seq_len,
        horizon=args.horizon_bars,
        pos_threshold=None,  # Use settings defaults
        neg_threshold=None,
        ignore_margin=None,
        batch_size=args.batch_size,
        num_channels=[64, 64, 64, 64],  # Default TCN channels
        kernel_size=3,
        dropout=0.2,
        learning_rate=args.lr,
        weight_decay=1e-4,
        num_epochs=args.epochs,
        train_split=0.8,
        patience_es=10,
        min_delta=1e-4,
        device=device,
        out_model_path=args.out_model,
    )
    
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_model_path.resolve()}")
    logger.info("Model training completed successfully!")


if __name__ == "__main__":
    main()

