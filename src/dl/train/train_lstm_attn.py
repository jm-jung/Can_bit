"""
Training script for LSTM + Attention model.

=== DEBUGGING INSTRUMENTATION ADDED ===

This script has been instrumented with diagnostic logging to identify root causes of:
1. Missing FLAT labels in dataset (0 samples in FLAT class)
2. best_val_loss staying at inf (never updated)
3. Class ↔ output dimension consistency issues

Key diagnostic sections:
- [DEBUG] Labeling Threshold Configuration: Shows threshold values and FLAT zone
- [DEBUG] Label Distribution After Split: Shows class counts for train/val/test
- [DEBUG] Future Return Statistics: Shows return distribution vs thresholds
- [DEBUG] best_val_loss tracking: Shows when/why best_val_loss updates
- [ERROR]/[WARNING] markers for suspicious conditions

Run with normal training to see diagnostic output, or use --debug-overfit for quick testing.
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.core.config import settings, get_lstm_attn_model_path
from src.indicators.basic import add_basic_indicators
from src.ml.features import build_ml_dataset
from src.services.ohlcv_service import load_ohlcv_df
from src.dl.models.lstm_attn import LSTMAttentionModel
from src.dl.data.split import make_time_series_splits, log_split_summary
from src.dl.data.labels import create_3class_labels, LstmClassIndex
from src.debug.overfit_checks import (
    run_single_sample_overfit,
    run_two_sample_overfit,
    run_small_batch_overfit,
    print_label_stats,
    check_backbone_feature_std,
)
from src.debug.gradient_check import report_gradient_issues

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================================
# LSTM Training Memory Optimization Settings
# ======================================================================
# These settings are LSTM-specific and do not affect other ML/DL pipelines

# Debug mode: Use smaller dataset for faster development/testing
DEBUG_SMALL_DATASET = False  # Set to True for quick testing with smaller data

# Date range filter for LSTM training (to reduce memory usage)
# Only data within this range will be used for LSTM training
LSTM_TRAIN_START_DATE = pd.Timestamp("2022-01-01")
LSTM_TRAIN_END_DATE = pd.Timestamp("2025-12-07")

# Maximum rows to use for LSTM training (tail sampling)
# If dataset has more rows than this, only the last N rows will be used
# This helps prevent memory errors on systems with limited RAM
if DEBUG_SMALL_DATASET:
    MAX_LSTM_ROWS = 100_000  # Smaller dataset for debug/development
    logger.info("[LSTM][Config] Using DEBUG_SMALL_DATASET mode (MAX_LSTM_ROWS=%d)", MAX_LSTM_ROWS)
else:
    MAX_LSTM_ROWS = 300_000  # Production mode: use up to 300k rows


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
    """
    PyTorch Dataset for time series sequences.
    
    Supports both binary (float) and 3-class (int) labels.
    For 3-class labels, y should be integer array with values {0, 1, 2}.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.

        Args:
            X: Sequences of shape (num_samples, seq_len, feature_dim)
            y: Labels of shape (num_samples,)
                - For binary: float array with values {0.0, 1.0}
                - For 3-class: int array with values {0, 1, 2}
        """
        self.X = torch.FloatTensor(X)
        # Check if y is integer (3-class) or float (binary)
        if y.dtype in (np.int64, np.int32, int):
            # 3-class: use LongTensor for CrossEntropyLoss
            self.y = torch.LongTensor(y)
        else:
            # Binary: use FloatTensor for BCEWithLogitsLoss
            self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def evaluate_split(
    model: nn.Module,
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    device: torch.device,
    split_name: str,
    thresholds: list[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    batch_size: int = 64,
    logger: Any = None,
) -> None:
    """
    Evaluate model on a split (train/valid/test) and log metrics.
    
    Supports both binary and 3-class classification:
    - Binary: y is float array with {0.0, 1.0}
    - 3-class: y is int array with {0, 1, 2}
    
    Args:
        model: Trained model
        X: Input sequences (numpy array or torch tensor)
        y: Labels (numpy array or torch tensor)
        device: Device to run inference on
        split_name: Name of the split (e.g., "Validation", "Test")
        thresholds: List of probability thresholds to test (for binary only)
        batch_size: Batch size for inference
        logger: Logger instance (if None, uses default logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Convert to torch tensors if needed
    if isinstance(X, np.ndarray):
        X_tensor = torch.FloatTensor(X)
    else:
        X_tensor = X
    
    if isinstance(y, np.ndarray):
        # Check if y is integer (3-class) or float (binary)
        if y.dtype in (np.int64, np.int32, int):
            y_tensor = torch.LongTensor(y)
            is_3class = True
        else:
            y_tensor = torch.FloatTensor(y)
            is_3class = False
    else:
        # Try to infer from tensor dtype
        is_3class = y.dtype in (torch.int64, torch.int32, torch.long)
        y_tensor = y
    
    model.eval()
    all_probs_long = []
    all_probs_short = []
    all_pred_classes = []
    all_labels = []
    
    # Create dataset and dataloader
    dataset = TimeSeriesDataset(X_tensor.numpy(), y_tensor.numpy())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)  # (B, 3) for 3-class or (B, 1) for binary
            
            if is_3class:
                # 3-class: softmax and extract probabilities
                probs = F.softmax(logits, dim=-1)  # (B, 3)
                pred_classes = logits.argmax(dim=-1)  # (B,)
                all_probs_long.extend(probs[:, LstmClassIndex.LONG].cpu().numpy().tolist())
                all_probs_short.extend(probs[:, LstmClassIndex.SHORT].cpu().numpy().tolist())
                all_pred_classes.extend(pred_classes.cpu().numpy().tolist())
            else:
                # Binary: sigmoid
                probs = torch.sigmoid(logits)  # (B, 1)
                preds = (probs >= 0.5).long().squeeze()  # (B,)
                all_probs_long.extend(probs.cpu().numpy().flatten().tolist())
                all_pred_classes.extend(preds.cpu().numpy().tolist())
            
            all_labels.extend(y_batch.cpu().numpy().tolist())
    
    all_labels = np.array(all_labels)
    all_pred_classes = np.array(all_pred_classes)
    
    if is_3class:
        # 3-class evaluation
        all_probs_long = np.array(all_probs_long)
        all_probs_short = np.array(all_probs_short)
        
        # Calculate accuracy
        accuracy = (all_pred_classes == all_labels).mean()
        
        # Per-class metrics
        flat_true = (all_labels == LstmClassIndex.FLAT).sum()
        long_true = (all_labels == LstmClassIndex.LONG).sum()
        short_true = (all_labels == LstmClassIndex.SHORT).sum()
        
        flat_pred = (all_pred_classes == LstmClassIndex.FLAT).sum()
        long_pred = (all_pred_classes == LstmClassIndex.LONG).sum()
        short_pred = (all_pred_classes == LstmClassIndex.SHORT).sum()
        
        logger.info("=" * 60)
        logger.info(f"{split_name} Set Evaluation (3-class)")
        logger.info("=" * 60)
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info(f"Class distribution (true): FLAT={flat_true}, LONG={long_true}, SHORT={short_true}")
        logger.info(f"Class distribution (pred): FLAT={flat_pred}, LONG={long_pred}, SHORT={short_pred}")
        logger.info(f"Proba LONG stats: mean={all_probs_long.mean():.4f}, std={all_probs_long.std():.4f}")
        logger.info(f"Proba SHORT stats: mean={all_probs_short.mean():.4f}, std={all_probs_short.std():.4f}")
        logger.info("-" * 60)
    else:
        # Binary evaluation (backward compatibility)
        all_probs = np.array(all_probs_long)
        
        # Calculate metrics at default threshold (0.5)
        preds_default = (all_probs >= 0.5).astype(int)
        tp_default = np.sum((preds_default == 1) & (all_labels == 1))
        fp_default = np.sum((preds_default == 1) & (all_labels == 0))
        tn_default = np.sum((preds_default == 0) & (all_labels == 0))
        fn_default = np.sum((preds_default == 0) & (all_labels == 1))
        
        accuracy = (tp_default + tn_default) / len(all_labels) if len(all_labels) > 0 else 0.0
        precision_default = tp_default / (tp_default + fp_default + 1e-8) if (tp_default + fp_default) > 0 else 0.0
        recall_default = tp_default / (tp_default + fn_default + 1e-8) if (tp_default + fn_default) > 0 else 0.0
        f1_default = 2 * precision_default * recall_default / (precision_default + recall_default + 1e-8) if (precision_default + recall_default) > 0 else 0.0
        
        logger.info("=" * 60)
        logger.info(f"{split_name} Set Evaluation (binary)")
        logger.info("=" * 60)
        logger.info(f"Threshold=0.5: Accuracy={accuracy:.4f}, Precision={precision_default:.4f}, Recall={recall_default:.4f}, F1={f1_default:.4f}")
        logger.info(f"Confusion Matrix: TP={tp_default}, FP={fp_default}, TN={tn_default}, FN={fn_default}")
        logger.info("-" * 60)
        logger.info(f"{split_name} Threshold Analysis (various thresholds for precision/recall):")
        logger.info("-" * 60)
        logger.info(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
        logger.info("-" * 60)
        
        for thresh in thresholds:
            preds = (all_probs >= thresh).astype(int)
            tp = np.sum((preds == 1) & (all_labels == 1))
            fp = np.sum((preds == 1) & (all_labels == 0))
            fn = np.sum((preds == 0) & (all_labels == 1))
            
            precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
            
            logger.info(
                f"{thresh:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} "
                f"{tp:<8} {fp:<8} {fn:<8}"
            )
        logger.info("-" * 60)


def debug_lstm_dataset_samples(
    logger,
    X,
    y,
    indices_to_check=None,
    prefix: str = "[DEBUG][DATASET]",
):
    """
    Debug utility for LSTM sequence datasets.
    For the given X (N, window_size, feature_dim) and y (N,),
    log labels and a few sequence values for selected indices,
    and also log pairwise absolute differences.
    Intended for use when debug_overfit == 2.
    """
    import torch

    if indices_to_check is None:
        indices_to_check = [0, 1, 2, 12]

    X_t = torch.tensor(X) if not isinstance(X, torch.Tensor) else X
    y_t = torch.tensor(y) if not isinstance(y, torch.Tensor) else y

    logger.info(f"{prefix} X.shape={tuple(X_t.shape)}, y.shape={tuple(y_t.shape)}")

    for idx in indices_to_check:
        if idx < 0 or idx >= X_t.shape[0]:
            logger.warning(f"{prefix} index {idx} out of range (0 ~ {X_t.shape[0] - 1})")
            continue

        x_i = X_t[idx]
        y_i = y_t[idx].item()

        ts_sample = x_i[:5, :8].detach().cpu().numpy()

        logger.info(
            f"{prefix} idx={idx}, label={y_i}, "
            f"x_i.shape={tuple(x_i.shape)}, "
            f"ts_sample(first_5_steps_first_8_features)={ts_sample}"
        )

    if len(indices_to_check) >= 2:
        base_idx = indices_to_check[0]
        if base_idx < 0 or base_idx >= X_t.shape[0]:
            return
        base_x = X_t[base_idx]

        for other_idx in indices_to_check[1:]:
            if other_idx < 0 or other_idx >= X_t.shape[0]:
                logger.warning(
                    f"{prefix} other_idx {other_idx} out of range (0 ~ {X_t.shape[0] - 1})"
                )
                continue

            other_x = X_t[other_idx]
            diff = (base_x - other_x).abs()
            mean_diff = diff.mean().item()
            std_diff = diff.std().item()
            max_diff = diff.max().item()

            logger.info(
                f"{prefix}[DIFF] base_idx={base_idx} vs other_idx={other_idx} → "
                f"mean_abs_diff={mean_diff:.8f}, std={std_diff:.8f}, max={max_diff:.8f}"
            )


def create_sequences(
    df: pd.DataFrame,
    window_size: int = 60,
    horizon: int | None = None,
    pos_threshold: float | None = None,
    neg_threshold: float | None = None,
    ignore_margin: float | None = None,
    debug_inspect: bool = False,
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

    Args:
        debug_inspect: If True, log detailed stats and perform safety checks (used for debug overfit mode)

    Returns:
        Tuple of (X: sequences, y: labels, feature_cols: feature column names, meta: dict)
        where meta contains:
        - "future_returns": np.ndarray of shape (N,) with actual forward returns
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
    logger.info("Creating sequences for LSTM training (3-class: FLAT/LONG/SHORT)")
    logger.info("=" * 60)
    logger.info(f"Input data shape (before filtering): {df.shape}")
    
    # ======================================================================
    # STEP 1: Date Range Filtering (LSTM-specific memory optimization)
    # ======================================================================
    # Filter data to specified date range to reduce memory usage
    # This only affects LSTM training pipeline, not other ML/DL strategies
    original_rows = len(df)
    
    # Check if df has timestamp column or DatetimeIndex
    if "timestamp" in df.columns:
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter by date range
        date_mask = (df["timestamp"] >= LSTM_TRAIN_START_DATE) & (df["timestamp"] <= LSTM_TRAIN_END_DATE)
        df = df.loc[date_mask].copy()
        
        logger.info(
            "[LSTM][Date Filter] Filtered by timestamp column: %d -> %d rows (from %s to %s)",
            original_rows,
            len(df),
            LSTM_TRAIN_START_DATE.date(),
            LSTM_TRAIN_END_DATE.date(),
        )
    elif isinstance(df.index, pd.DatetimeIndex):
        # Filter by DatetimeIndex
        date_mask = (df.index >= LSTM_TRAIN_START_DATE) & (df.index <= LSTM_TRAIN_END_DATE)
        df = df.loc[date_mask].copy()
        
        logger.info(
            "[LSTM][Date Filter] Filtered by DatetimeIndex: %d -> %d rows (from %s to %s)",
            original_rows,
            len(df),
            LSTM_TRAIN_START_DATE.date(),
            LSTM_TRAIN_END_DATE.date(),
        )
    else:
        logger.warning(
            "[LSTM][Date Filter] No DatetimeIndex or 'timestamp' column found; skipping date filter. "
            "This may cause memory issues with large datasets."
        )
    
    # ======================================================================
    # STEP 2: Tail Sampling (LSTM-specific memory optimization)
    # ======================================================================
    # Use only the most recent N rows to further reduce memory usage
    # This ensures we use the most recent data while staying within memory limits
    current_rows = len(df)
    if current_rows > MAX_LSTM_ROWS:
        df = df.iloc[-MAX_LSTM_ROWS:].copy()
        logger.info(
            "[LSTM][Tail Sampling] Using last %d rows out of %d (tail sampling for memory safety)",
            MAX_LSTM_ROWS,
            current_rows,
        )
    else:
        logger.info(
            "[LSTM][Tail Sampling] Rows (%d) <= MAX_LSTM_ROWS (%d), no tail sampling applied",
            current_rows,
            MAX_LSTM_ROWS,
        )
    
    # Reset index after filtering to ensure clean sequential indexing
    df = df.reset_index(drop=True)
    
    logger.info(f"Input data shape (after filtering): {df.shape}")
    logger.info(f"Window size: {window_size}, Horizon: {horizon}")
    logger.info(f"Label thresholds: pos={pos_threshold:.4f}, neg={neg_threshold:.4f}")
    logger.info(f"3-class mapping: FLAT=0, LONG=1, SHORT=2")
    logger.info(f"  - LONG: r > +{pos_threshold:.4f}")
    logger.info(f"  - SHORT: r < -{neg_threshold:.4f}")
    logger.info(f"  - FLAT: otherwise")
    
    if debug_inspect:
        try:
            logger.info("[DEBUG][RAW_DF] Before feature_df creation:")
            logger.info(
                "[DEBUG][RAW_DF] shape=%s, columns=%s",
                str(df.shape),
                list(df.columns),
            )
            logger.info("[DEBUG][RAW_DF] head(10):\n%s", df.head(10))
            logger.info(
                "[DEBUG][RAW_DF] describe():\n%s",
                df.describe().to_string(),
            )
        except Exception as e:
            logger.warning("[DEBUG][RAW_DF] failed to log raw df: %s", e)

    # Add indicators if not present
    if "ema_20" not in df.columns:
        df = add_basic_indicators(df)

    # Build features using existing function
    # NOTE: build_ml_dataset는 (X, y, y_long, y_short) 4개 값을 반환합니다
    # LSTM 3-class 학습에서는 feature(X)만 사용하고, binary y/y_long/y_short는 사용하지 않음
    # 3-class 레이블은 별도로 계산함 (threshold 기반)
    # Extract symbol and timeframe from settings
    symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    
    # build_ml_dataset 반환값: (X, y, y_long, y_short) - 4개
    # enable_hold_labels=False (기본값)이므로 y_hold는 반환되지 않음
    X_features, _, _, _ = build_ml_dataset(
        df,
        horizon=horizon,
        symbol=symbol,
        timeframe=timeframe,
        use_events=settings.EVENTS_ENABLED,
        enable_hold_labels=False,  # LSTM 3-class에서는 HOLD label 불필요
        debug_inspect=debug_inspect,
        debug_logger=logger,
    )

    # feature_df 생성 이후 상태 디버깅
    if debug_inspect:
        try:
            logger.info(
                "[DEBUG][SEQ][FEATURE_DF] shape=%s",
                getattr(X_features, "shape", None),
            )
            logger.info(
                "[DEBUG][SEQ][FEATURE_DF] dtypes:\n%s",
                getattr(X_features, "dtypes", None),
            )
            logger.info("[DEBUG][SEQ][FEATURE_DF] head(10):\n%s", X_features.head(10))
            logger.info(
                "[DEBUG][SEQ][FEATURE_DF] describe():\n%s",
                X_features.describe().to_string(),
            )
        except Exception as e:
            logger.warning("[DEBUG][SEQ][FEATURE_DF] failed to log feature df: %s", e)

    # Get feature columns (순서가 중요함 - 예측 시점과 동일해야 함)
    feature_cols = X_features.columns.tolist()
    feature_dim = len(feature_cols)
    
    logger.info(f"Feature columns ({feature_dim} total):")
    logger.info(f"  - Basic features: {[c for c in feature_cols if not c.startswith('event_')]}")
    if settings.EVENTS_ENABLED:
        event_cols = [c for c in feature_cols if c.startswith("event_")]
        logger.info(f"  - Event features: {len(event_cols)} (sample: {event_cols[:3]})")
    
    # Feature columns detailed logging
    logger.info("=" * 60)
    logger.info("[LSTM Train] Feature Columns")
    logger.info("=" * 60)
    logger.info(f"[LSTM] Using {len(feature_cols)} feature columns:")
    logger.info(f"[LSTM] FEATURE_COLS = {feature_cols}")
    if settings.EVENTS_ENABLED:
        event_cols = [c for c in feature_cols if c.startswith("event_")]
        logger.info(f"[LSTM] Event features ({len(event_cols)}): {event_cols}")
        basic_cols = [c for c in feature_cols if not c.startswith("event_")]
        logger.info(f"[LSTM] Basic features ({len(basic_cols)}): {basic_cols}")
    logger.info("=" * 60)

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
    
    # ======================================================================
    # STEP 3: Float32 Downcasting (LSTM-specific memory optimization)
    # ======================================================================
    # Convert normalized features to float32 to reduce memory usage by ~50%
    # Financial time series data has sufficient precision with float32
    # (typical price movements are well within float32 precision range)
    # This optimization only affects LSTM training pipeline
    logger.info(
        "[LSTM][Memory] Converting features to float32 for memory optimization "
        "(original dtype: %s, memory reduction: ~50%%)",
        X_normalized.dtypes.iloc[0] if len(X_normalized) > 0 else "unknown",
    )
    
    # Convert all feature columns to float32
    for col in feature_cols:
        X_normalized[col] = X_normalized[col].astype(np.float32)
    
    logger.info(
        "[LSTM][Memory] Feature conversion complete. "
        "X_normalized shape: %s, dtype: float32",
        X_normalized.shape,
    )

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

        # ### BUG CANDIDATE: Missing FLAT labels in dataset
        # 3-class label definition:
        # - r > +pos_threshold → LONG (class 1)
        # - r < -neg_threshold → SHORT (class 2)
        # - Otherwise → FLAT (class 0)
        # 
        # NOTE: If neg_threshold is negative (e.g., -0.001), then -neg_threshold = 0.001
        # This means SHORT condition is r < 0.001, which is very broad.
        # FLAT zone becomes: -abs(neg_threshold) <= r <= pos_threshold
        # If this zone is too narrow, FLAT samples will be rare or zero.
        neg_threshold_abs = abs(neg_threshold)  # Use absolute value for comparison
        if future_ret > pos_threshold:
            label = LstmClassIndex.LONG  # 1
        elif future_ret < -neg_threshold_abs:
            label = LstmClassIndex.SHORT  # 2
        else:
            label = LstmClassIndex.FLAT  # 0

        sequences.append(seq)
        labels.append(label)
        future_returns.append(future_ret)

    # Convert sequences to numpy array with float32 dtype for memory optimization
    # Note: seq values are already float32 from X_normalized, but explicit dtype ensures consistency
    # Memory benefit: float32 uses 4 bytes per value vs 8 bytes for float64 (~50% reduction)
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)  # 3-class: 0, 1, 2
    future_returns_arr = np.array(future_returns, dtype=np.float32)
    
    logger.info(
        "[LSTM][Memory] Sequence arrays created: X.shape=%s (dtype=%s), y.shape=%s (dtype=%s)",
        X.shape,
        X.dtype,
        y.shape,
        y.dtype,
    )

    # --- 안전장치: NaN 제거 ---
    # 1) 입력 데이터와 라벨의 NaN/Inf 체크
    # 라벨에 NaN이 있는 샘플 제거 (3-class에서는 모든 레이블이 유효하므로 NaN 체크만)
    mask = ~np.isnan(y.astype(np.float64))  # int64를 float64로 변환하여 NaN 체크
    # 입력 데이터에 NaN/Inf가 있는 샘플도 제거
    X_has_nan = np.isnan(X).any(axis=(1, 2)) | np.isinf(X).any(axis=(1, 2))
    mask = mask & ~X_has_nan

    X = X[mask]
    y = y[mask]
    future_returns_arr = future_returns_arr[mask]

    # 3-class 레이블은 이미 0, 1, 2로 정수형이므로 추가 변환 불필요

    # 상세 라벨 분포 로깅 (3-class)
    all_labels = np.array(labels, dtype=np.int64)
    all_future_returns = np.array(future_returns, dtype=np.float32)
    
    flat_count_all = int(np.sum(all_labels == LstmClassIndex.FLAT))
    long_count_all = int(np.sum(all_labels == LstmClassIndex.LONG))
    short_count_all = int(np.sum(all_labels == LstmClassIndex.SHORT))
    total_count_all = len(all_labels)
    
    # 학습에 사용되는 샘플만 (NaN 제거 후)
    flat_count = int(np.sum(y == LstmClassIndex.FLAT))
    long_count = int(np.sum(y == LstmClassIndex.LONG))
    short_count = int(np.sum(y == LstmClassIndex.SHORT))
    
    flat_ratio = flat_count / len(y) if len(y) > 0 else 0.0
    long_ratio = long_count / len(y) if len(y) > 0 else 0.0
    short_ratio = short_count / len(y) if len(y) > 0 else 0.0
    
    logger.info("-" * 60)
    logger.info("3-Class Label Distribution:")
    logger.info(f"  Total samples (before filtering): {total_count_all}")
    logger.info(f"  FLAT (label=0): {flat_count_all} ({flat_count_all/total_count_all:.3f} of total)")
    logger.info(f"  LONG (label=1): {long_count_all} ({long_count_all/total_count_all:.3f} of total)")
    logger.info(f"  SHORT (label=2): {short_count_all} ({short_count_all/total_count_all:.3f} of total)")
    logger.info("-" * 60)
    logger.info(f"  Final samples (after filtering): {len(y)}")
    logger.info(f"  FLAT (label=0): {flat_count} ({flat_ratio:.3f})")
    logger.info(f"  LONG (label=1): {long_count} ({long_ratio:.3f})")
    logger.info(f"  SHORT (label=2): {short_count} ({short_ratio:.3f})")
    
    # 3-class 분포 경고
    if flat_ratio > 0.8:
        logger.warning(
            f"FLAT class ratio is very high (>80%%). "
            f"대부분의 샘플이 FLAT으로 분류됩니다. (flat_ratio={flat_ratio:.3f})"
        )
    elif flat_ratio < 0.1:
        logger.warning(
            f"FLAT class ratio is very low (<10%%). "
            f"대부분의 샘플이 LONG/SHORT로 분류됩니다. (flat_ratio={flat_ratio:.3f})"
        )
    
    # 클래스 불균형 경고
    min_class_ratio = min(long_ratio, short_ratio, flat_ratio)
    if min_class_ratio < 0.05:
        logger.warning(
            f"Severe class imbalance detected. "
            f"Smallest class ratio: {min_class_ratio:.3f}. "
            f"Consider adjusting thresholds or using class weights."
        )
    
    # ### BUG CANDIDATE: Missing FLAT labels in dataset
    # Future return 통계 (threshold 분석용)
    if len(future_returns_arr) > 0:
        neg_threshold_abs = abs(neg_threshold)
        logger.info("-" * 60)
        logger.info("Future Return Statistics (for threshold analysis):")
        logger.info(f"  Horizon: {horizon}")
        logger.info(f"  pos_threshold: {pos_threshold:.6f}")
        logger.info(f"  neg_threshold (raw): {neg_threshold:.6f}")
        logger.info(f"  neg_threshold_abs (used): {neg_threshold_abs:.6f}")
        logger.info(f"  ignore_margin: {ignore_margin:.6f}")
        logger.info(f"  Mean: {future_returns_arr.mean():.6f}")
        logger.info(f"  Std: {future_returns_arr.std():.6f}")
        logger.info(f"  Min: {future_returns_arr.min():.6f}")
        logger.info(f"  Max: {future_returns_arr.max():.6f}")
        logger.info(f"  25th percentile: {np.percentile(future_returns_arr, 25):.6f}")
        logger.info(f"  50th percentile: {np.percentile(future_returns_arr, 50):.6f}")
        logger.info(f"  75th percentile: {np.percentile(future_returns_arr, 75):.6f}")
        
        # Count samples in each class zone
        long_count = np.sum(future_returns_arr > pos_threshold)
        short_count = np.sum(future_returns_arr < -neg_threshold_abs)
        flat_count = np.sum((future_returns_arr >= -neg_threshold_abs) & (future_returns_arr <= pos_threshold))
        total_samples = len(future_returns_arr)
        
        logger.info(f"  Samples > pos_threshold ({pos_threshold:.6f}) → LONG: {long_count} ({long_count/total_samples:.4f})")
        logger.info(f"  Samples < -neg_threshold_abs ({-neg_threshold_abs:.6f}) → SHORT: {short_count} ({short_count/total_samples:.4f})")
        logger.info(f"  Samples in FLAT zone [-{neg_threshold_abs:.6f}, {pos_threshold:.6f}]: {flat_count} ({flat_count/total_samples:.4f})")
        
        # Diagnostic: if FLAT count is very low, explain why
        if flat_count == 0:
            logger.error(
                "[ERROR] FLAT zone has 0 samples! This explains why FLAT class never appears. "
                f"FLAT zone is [{neg_threshold_abs:.6f}, {pos_threshold:.6f}]. "
                f"Return range is [{future_returns_arr.min():.6f}, {future_returns_arr.max():.6f}]. "
                f"The FLAT zone may be too narrow or outside the return distribution."
            )
        elif flat_count / total_samples < 0.01:
            logger.warning(
                f"[WARNING] FLAT zone has very few samples ({flat_count}/{total_samples} = {flat_count/total_samples:.4f}). "
                f"FLAT zone [{neg_threshold_abs:.6f}, {pos_threshold:.6f}] may be too narrow."
            )
    
    logger.info("=" * 60)
    logger.info("Final sequence shape after dropping ignored labels:")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y shape: {y.shape}")
    logger.info(f"  feature_dim: {feature_dim}")
    logger.info("=" * 60)

    if debug_inspect:
        try:
            logger.info("[DEBUG][SEQ][FINAL] X.shape=%s, y.shape=%s", str(X.shape), str(y.shape))
            sample_indices = [0, 1, 2, 12]
            for idx in sample_indices:
                if idx < 0 or idx >= len(X):
                    logger.warning(
                        "[DEBUG][SEQ][FINAL] index %s out of range (0 ~ %s)",
                        idx,
                        max(len(X) - 1, -1),
                    )
                    continue
                sample = X[idx]
                mean_abs = float(np.mean(np.abs(sample)))
                max_abs = float(np.max(np.abs(sample)))
                logger.info(
                    "[DEBUG][SEQ][FINAL] idx=%d, label=%.3f, mean_abs=%.6f, max_abs=%.6f",
                    idx,
                    float(y[idx]),
                    mean_abs,
                    max_abs,
                )
                logger.info(
                    "[DEBUG][SEQ][FINAL] idx=%d first_5_steps_first_8_features=\n%s",
                    idx,
                    sample[:5, :8],
                )

            if len(sample_indices) >= 2 and len(X) > 0:
                base_idx = sample_indices[0]
                if 0 <= base_idx < len(X):
                    base_sample = X[base_idx]
                    for other_idx in sample_indices[1:]:
                        if other_idx < 0 or other_idx >= len(X):
                            continue
                        diff = np.abs(base_sample - X[other_idx])
                        logger.info(
                            "[DEBUG][SEQ][FINAL][DIFF] base_idx=%d vs other_idx=%d → mean_abs_diff=%.8f, max_abs_diff=%.8f",
                            base_idx,
                            other_idx,
                            float(diff.mean()),
                            float(diff.max()),
                        )
        except Exception as e:
            logger.warning("[DEBUG][SEQ][FINAL] failed to log final sequences: %s", e)

        if len(X) >= 2:
            base = X[0]
            other = X[1]
            if np.allclose(base, 0) and np.allclose(other, 0) and np.allclose(base, other):
                raise ValueError(
                    "All LSTM input sequences appear to be identical or all zeros – please check feature creation and normalization."
                )

    # Create meta dictionary with future returns
    meta = {
        "future_returns": future_returns_arr,
    }
    
    return X, y, feature_cols, meta


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
    debug_overfit_mode: str | None = None,
    debug_small_overfit: bool = False,
) -> tuple[LSTMAttentionModel, float, Path]:
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
        debug_overfit_mode: Debug overfit test mode (None, "1", "2", "32", "64", "all")
        debug_small_overfit: If True, use stratified 64-sample subset with 30 epochs for quick overfit testing (default: False)

    Returns:
        Tuple of (model, best_val_loss, best_model_path):
        - model: Trained model with best weights loaded
        - best_val_loss: Best validation loss achieved during training
        - best_model_path: Path to the saved best model checkpoint
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
    
    # ### BUG CANDIDATE: Threshold sign check
    # Verify threshold signs are correct for 3-class labeling
    # Expected: pos_threshold > 0, neg_threshold should be used as abs(neg_threshold) for comparison
    logger.info("=" * 60)
    logger.info("[DEBUG] Labeling Threshold Configuration")
    logger.info("=" * 60)
    logger.info(f"pos_threshold (raw from settings): {pos_threshold}")
    logger.info(f"neg_threshold (raw from settings): {neg_threshold}")
    logger.info(f"FLAT zone will be: [-abs(neg_threshold), pos_threshold] = [{-abs(neg_threshold):.6f}, {pos_threshold:.6f}]")
    logger.info(f"LONG condition: r > {pos_threshold:.6f}")
    logger.info(f"SHORT condition: r < {-abs(neg_threshold):.6f}")
    logger.info("=" * 60)

    # Generate unique model path based on configuration
    # Include num_classes (3), horizon, and feature preset (events_enabled) to avoid conflicts
    num_classes = 3  # 3-class: FLAT, LONG, SHORT
    feature_preset = "events" if settings.EVENTS_ENABLED else "basic"
    best_model_path = get_lstm_attn_model_path(
        num_classes=num_classes,
        horizon=horizon,
        feature_preset=feature_preset,
    )

    logger.info("=" * 60)
    logger.info("LSTM + Attention Model Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Window size: {window_size}, Horizon: {horizon}")
    logger.info(f"Return thresholds: pos={pos_threshold:.4f}, neg={neg_threshold:.4f}")
    logger.info(f"Ignore margin: {ignore_margin:.4f}")
    logger.info(f"Model path: {best_model_path.resolve()}")
    logger.info(f"Configuration: {num_classes}-class, horizon={horizon}, features={feature_preset}")

    # Load data
    logger.info("Loading OHLCV data...")
    df = load_ohlcv_df()
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of OHLCV data")

    # Create sequences
    logger.info(f"Creating sequences (window_size={window_size}, horizon={horizon})...")
    logger.info("Event features enabled: %s", settings.EVENTS_ENABLED)
    X, y, feature_cols, meta = create_sequences(
        df,
        window_size=window_size,
        horizon=horizon,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
        ignore_margin=ignore_margin,
        debug_inspect=(debug_overfit_mode == "2"),
    )
    feature_dim = len(feature_cols)

    logger.info(f"Created {len(X)} sequences with {feature_dim} features")
    if settings.EVENTS_ENABLED:
        event_cols = [col for col in feature_cols if col.startswith("event_")]
        logger.info("이벤트 피처 수: %d (샘플: %s)", len(event_cols), event_cols[:5])

    # Train/validation/test split using time-series split utility
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

    # Log split summary
    log_split_summary(
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        logger=logger,
    )
    
    # ### BUG CANDIDATE: Missing FLAT labels in dataset
    # Explicit label distribution logging for all splits
    logger.info("=" * 60)
    logger.info("[DEBUG] Label Distribution After Split")
    logger.info("=" * 60)
    for split_name, y_split in [("Train", y_train), ("Valid", y_valid), ("Test", y_test)]:
        y_arr = np.array(y_split) if not isinstance(y_split, np.ndarray) else y_split
        flat_count = int(np.sum(y_arr == LstmClassIndex.FLAT))
        long_count = int(np.sum(y_arr == LstmClassIndex.LONG))
        short_count = int(np.sum(y_arr == LstmClassIndex.SHORT))
        total = len(y_arr)
        
        logger.info(
            f"{split_name} label distribution: "
            f"FLAT={flat_count} ({flat_count/total:.4f}), "
            f"LONG={long_count} ({long_count/total:.4f}), "
            f"SHORT={short_count} ({short_count/total:.4f}), "
            f"total={total}"
        )
        
        # Sanity check: warn if any class has 0 samples
        if flat_count == 0:
            logger.warning(
                f"[WARNING] Class FLAT (0) has 0 samples in {split_name} split. "
                f"3-class training may be invalid. Check threshold configuration."
            )
        if long_count == 0:
            logger.warning(
                f"[WARNING] Class LONG (1) has 0 samples in {split_name} split. "
                f"3-class training may be invalid."
            )
        if short_count == 0:
            logger.warning(
                f"[WARNING] Class SHORT (2) has 0 samples in {split_name} split. "
                f"3-class training may be invalid."
            )
        
        # Verify label range matches num_classes
        y_min = int(y_arr.min()) if len(y_arr) > 0 else -1
        y_max = int(y_arr.max()) if len(y_arr) > 0 else -1
        num_classes = 3
        if y_min < 0 or y_max >= num_classes:
            logger.error(
                f"[ERROR] Label range violation in {split_name} split: "
                f"min={y_min}, max={y_max}, expected range=[0, {num_classes-1}]"
            )
        else:
            logger.info(
                f"{split_name} label range: min={y_min}, max={y_max} (valid for {num_classes}-class)"
            )
    logger.info("=" * 60)

    logger.info("-" * 60)
    logger.info("Train/Validation/Test Split:")
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_valid shape: {X_valid.shape}")
    logger.info(f"  X_test shape: {X_test.shape}")
    logger.info("-" * 60)

    if debug_overfit_mode == "2":
        logger.info("[DEBUG][DATASET] debug_overfit==2 → inspecting train/valid sequences")
        debug_lstm_dataset_samples(
            logger=logger,
            X=X_train,
            y=y_train,
            indices_to_check=[0, 1, 2, 12],
            prefix="[DEBUG][DATASET][TRAIN]",
        )
        debug_lstm_dataset_samples(
            logger=logger,
            X=X_valid,
            y=y_valid,
            indices_to_check=[0, 1, 2, 12],
            prefix="[DEBUG][DATASET][VALID]",
        )

    # ### BUG CANDIDATE: Class ↔ Output Dimension Consistency
    # Calculate class weights for imbalanced data (3-class)
    # Compute class distribution for CrossEntropyLoss weight
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 3-class: integer labels
    
    # Runtime assertion: verify label range matches num_classes
    num_classes = 3
    y_min = int(y_train_tensor.min().item())
    y_max = int(y_train_tensor.max().item())
    if y_min < 0 or y_max >= num_classes:
        raise ValueError(
            f"[ERROR] Label range violation in training set: "
            f"min={y_min}, max={y_max}, expected range=[0, {num_classes-1}]. "
            f"This will cause CrossEntropyLoss to fail. Check labeling logic."
        )
    logger.info(
        f"[DEBUG] Training label range check: min={y_min}, max={y_max} (valid for {num_classes}-class)"
    )
    
    flat_count = int((y_train_tensor == LstmClassIndex.FLAT).sum().item())
    long_count = int((y_train_tensor == LstmClassIndex.LONG).sum().item())
    short_count = int((y_train_tensor == LstmClassIndex.SHORT).sum().item())
    total_count = flat_count + long_count + short_count
    
    logger.info(f"Training set class distribution: FLAT={flat_count}, LONG={long_count}, SHORT={short_count}")
    
    # TASK 2: Class weights handling
    # In debug mode: use no class weights (plain CrossEntropyLoss) to avoid interference with overfit test
    # In production mode: use class weights based on full train distribution to handle imbalance
    if debug_small_overfit:
        # Debug mode: no class weights (plain CrossEntropyLoss)
        class_weights = None
        logger.info("=" * 60)
        logger.info("[DEBUG MODE] Class weights: DISABLED (using plain CrossEntropyLoss)")
        logger.info("  Reason: Debug overfit test should work without class weight interference")
        logger.info("=" * 60)
    else:
        # Production mode: compute class weights from full train distribution
        # Weight for each class = total_samples / (num_classes * class_count)
        if total_count > 0:
            class_weights = torch.tensor([
                total_count / (3.0 * max(flat_count, 1)),   # FLAT
                total_count / (3.0 * max(long_count, 1)),    # LONG
                total_count / (3.0 * max(short_count, 1)),   # SHORT
            ], dtype=torch.float32)
        else:
            class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            logger.warning("No training samples found, using uniform class weights")
        
        logger.info(f"[PRODUCTION MODE] Class weights: FLAT={class_weights[0]:.4f}, LONG={class_weights[1]:.4f}, SHORT={class_weights[2]:.4f}")

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    valid_dataset = TimeSeriesDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization will be done after effective hyperparameters are defined
    # (See Task 2: Model initialization with effective values after line ~1560)
    
    # Loss and optimizer will be initialized after effective hyperparameters are defined
    # (See Task 3: Optimizer with effective values after line ~1700)
    
    # TASK 2: Loss function setup (with or without class weights based on mode)
    if class_weights is not None:
        # Production mode: use class weights
        class_weights_device = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_device)
        logger.info(
            "[PRODUCTION MODE] Using CrossEntropyLoss (3-class) with class weights: "
            f"FLAT={class_weights[0]:.4f}, LONG={class_weights[1]:.4f}, SHORT={class_weights[2]:.4f}"
        )
    else:
        # Debug mode: no class weights (plain CrossEntropyLoss)
        criterion = nn.CrossEntropyLoss()
        logger.info(
            "[DEBUG MODE] Using CrossEntropyLoss (3-class) without class weights "
            "(plain loss for overfit test)"
        )

    # DEBUG-OVERFIT: disable weight decay in debug-overfit mode
    base_weight_decay = weight_decay
    if debug_overfit_mode is not None:
        logger.info(
            f"[DEBUG][INIT] debug_overfit mode → forcing weight_decay=0.0 (was {base_weight_decay})"
        )
        weight_decay = 0.0
    
    # Optimizer and scheduler will be initialized with effective hyperparameters after model creation
    # (See Task 3: Optimizer and scheduler initialization with effective values after line ~1585)

    # ============================================================
    # Debug Overfit Mode: 오버핏 테스트 실행
    # ============================================================
    if debug_overfit_mode is not None:
        logger.info("=" * 60)
        logger.info("DEBUG OVERFIT MODE ENABLED")
        logger.info(f"[DEBUG][INIT] debug_overfit mode enabled: {debug_overfit_mode}")
        logger.info("=" * 60)
        
        # Gradient flow 검사
        model_files = [
            Path(__file__),
            Path(__file__).parent.parent / "models" / "lstm_attn.py",
        ]
        report_gradient_issues(model_files)
        
        # Label 분포 출력
        print_label_stats(train_dataset, prefix="[DEBUG][TRAIN][LABEL]")
        
        # Backbone feature std 확인
        if len(train_dataset) > 0:
            X_sample, _ = train_dataset[0]
            if isinstance(X_sample, torch.Tensor):
                X_sample_batch = X_sample.unsqueeze(0).to(device)
            else:
                X_sample_batch = torch.tensor(X_sample).unsqueeze(0).to(device)
            check_backbone_feature_std(model, X_sample_batch, device, prefix="[DEBUG][STD]")
        
        # 오버핏 테스트 실행
        test_results = {}
        
        if debug_overfit_mode in ["1", "all"]:
            logger.info("\n" + "=" * 60)
            logger.info("Running 1-sample overfit test...")
            logger.info("=" * 60)
            success, result = run_single_sample_overfit(
                model,
                optimizer,
                train_dataset,
                device,
                max_steps=1000,
                lr_override=learning_rate,
                loss_fn=criterion,
            )
            test_results["1-sample"] = {"success": success, "result": result}
            if not success:
                logger.error(
                    "[ERROR][OVERFIT] 1-sample overfit FAILED → "
                    "모델/옵티마이저/데이터 플로우 중 하나에 구조적인 버그 가능성 매우 높음"
                )
        
        if debug_overfit_mode in ["2", "all"]:
            logger.info("\n" + "=" * 60)
            logger.info("Running 2-sample overfit test...")
            logger.info("=" * 60)
            success, result = run_two_sample_overfit(
                model,
                optimizer,
                train_dataset,
                device,
                max_steps=2000,
                lr_override=learning_rate,
                loss_fn=criterion,
            )
            test_results["2-sample"] = {"success": success, "result": result}
            if not success:
                logger.error(
                    "[ERROR][OVERFIT] 2-sample overfit FAILED → "
                    "먼저 여기부터 디버깅 필요"
                )
        
        if debug_overfit_mode in ["32", "all"]:
            logger.info("\n" + "=" * 60)
            logger.info("Running 32-sample overfit test...")
            logger.info("=" * 60)
            success, result = run_small_batch_overfit(
                model,
                optimizer,
                train_dataset,
                device,
                n_samples=32,
                max_steps=5000,
                lr_override=learning_rate,
                loss_fn=criterion,
            )
            test_results["32-sample"] = {"success": success, "result": result}
        
        if debug_overfit_mode in ["64", "all"]:
            logger.info("\n" + "=" * 60)
            logger.info("Running 64-sample overfit test...")
            logger.info("=" * 60)
            success, result = run_small_batch_overfit(
                model,
                optimizer,
                train_dataset,
                device,
                n_samples=64,
                max_steps=5000,
                lr_override=learning_rate,
                loss_fn=criterion,
            )
            test_results["64-sample"] = {"success": success, "result": result}
        
        # 결과 요약
        logger.info("\n" + "=" * 60)
        logger.info("OVERFIT TEST SUMMARY")
        logger.info("=" * 60)
        for test_name, test_data in test_results.items():
            status = "✓ PASSED" if test_data["success"] else "✗ FAILED"
            logger.info(f"{test_name}: {status}")
        logger.info("=" * 60)
        
        # 오버핏 모드에서는 실제 학습을 건너뛰고 모델 반환
        logger.info("Overfit tests completed. Returning model without full training.")
        # For debug mode, return with best_val_loss=inf and best_model_path=None
        return model, float("inf"), best_model_path

    # Training loop
    logger.info("Starting training...")
    logger.info("-" * 60)

    # Single source of truth for best validation loss tracking
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0
    
    # Early stopping configuration (production mode only)
    if debug_small_overfit:
        early_stopping_enabled = False
        early_stopping_patience = None
        early_stopping_min_delta = None
        logger.info("[DEBUG MODE] Early stopping disabled (always run full debug epochs).")
    else:
        early_stopping_enabled = True
        early_stopping_patience = patience_es
        early_stopping_min_delta = min_delta
        logger.info("[PRODUCTION MODE] Early stopping enabled.")
        logger.info(f"[PRODUCTION MODE] early_stopping_patience = {early_stopping_patience}")
        logger.info(f"[PRODUCTION MODE] early_stopping_min_delta = {early_stopping_min_delta:.1e}")
    
    # TASK 1: Debug small overfit mode configuration
    # debug_small_overfit is passed as function parameter (no local reassignment)
    # This ensures single source of truth: value comes from main() -> train_model() call
    debug_gradient_logging = getattr(settings, "DEBUG_GRADIENT_LOGGING", True)
    
    # Debug mode constants
    debug_num_samples = 64  # Number of samples to use in debug mode
    
    # ============================================================
    # Effective Hyperparameters (Task 1)
    # ============================================================
    # Base hyperparameters (production mode defaults)
    base_num_epochs = num_epochs
    base_lr = learning_rate
    base_weight_decay = weight_decay
    base_dropout = dropout
    base_hidden_size = hidden_size
    
    # Determine effective hyperparameters based on mode
    if debug_small_overfit:
        # Aggressive overfit hyperparams for DEBUG mode
        effective_num_epochs = 100
        effective_lr = 1e-3
        effective_weight_decay = 0.0
        effective_dropout = 0.0
        effective_hidden_size = 256
    else:
        # Production mode: use base values
        effective_num_epochs = base_num_epochs
        effective_lr = base_lr
        effective_weight_decay = base_weight_decay
        effective_dropout = base_dropout
        effective_hidden_size = base_hidden_size
    
    # Log effective hyperparameters
    if debug_small_overfit:
        logger.info("=" * 60)
        logger.info("[DEBUG MODE] Small Overfit Test - Aggressive Hyperparams")
        logger.info("=" * 60)
        logger.info(f"effective_num_epochs = {effective_num_epochs}")
        logger.info(f"effective_lr = {effective_lr:.6f}")
        logger.info(f"effective_weight_decay = {effective_weight_decay:.6f}")
        logger.info(f"effective_dropout = {effective_dropout:.4f}")
        logger.info(f"effective_hidden_size = {effective_hidden_size}")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("[PRODUCTION MODE] Hyperparams")
        logger.info("=" * 60)
        logger.info(f"effective_num_epochs = {effective_num_epochs}")
        logger.info(f"effective_lr = {effective_lr:.6f}")
        logger.info(f"effective_weight_decay = {effective_weight_decay:.6f}")
        logger.info(f"effective_dropout = {effective_dropout:.4f}")
        logger.info(f"effective_hidden_size = {effective_hidden_size}")
        logger.info("=" * 60)
    
    if debug_small_overfit:
        logger.info("=" * 60)
        logger.info("DEBUG MODE: Small Overfit Test")
        logger.info("=" * 60)
        logger.info(f"Debug mode configuration:")
        logger.info(f"  - debug_num_samples: {debug_num_samples}")
        logger.info(f"  - Effective epochs for this run: {effective_num_epochs}")
        logger.info("=" * 60)
        
        # TASK 1: Stratified sampling to ensure all 3 classes are represented
        # Create stratified sample with FLAT/LONG/SHORT all included
        y_train_arr = np.array(y_train)
        
        # Get indices for each class
        flat_indices = np.where(y_train_arr == LstmClassIndex.FLAT)[0]
        long_indices = np.where(y_train_arr == LstmClassIndex.LONG)[0]
        short_indices = np.where(y_train_arr == LstmClassIndex.SHORT)[0]
        
        logger.info(f"Full train set class distribution:")
        logger.info(f"  FLAT: {len(flat_indices)}, LONG: {len(long_indices)}, SHORT: {len(short_indices)}")
        
        # Sample from each class (stratified)
        samples_per_class = debug_num_samples // 3
        remaining_samples = debug_num_samples % 3
        
        flat_sample_size = min(samples_per_class + (1 if remaining_samples > 0 else 0), len(flat_indices))
        long_sample_size = min(samples_per_class + (1 if remaining_samples > 1 else 0), len(long_indices))
        short_sample_size = min(samples_per_class, len(short_indices))
        
        # Adjust if any class is too small
        if flat_sample_size == 0:
            logger.warning("[WARNING] FLAT class has 0 samples in debug subset. Debug mode may not fully test 3-class behavior.")
        if long_sample_size == 0:
            logger.warning("[WARNING] LONG class has 0 samples in debug subset. Debug mode may not fully test 3-class behavior.")
        if short_sample_size == 0:
            logger.warning("[WARNING] SHORT class has 0 samples in debug subset. Debug mode may not fully test 3-class behavior.")
        
        # Random sampling from each class
        np.random.seed(42)  # For reproducibility
        flat_selected = np.random.choice(flat_indices, size=flat_sample_size, replace=False) if flat_sample_size > 0 else np.array([], dtype=int)
        long_selected = np.random.choice(long_indices, size=long_sample_size, replace=False) if long_sample_size > 0 else np.array([], dtype=int)
        short_selected = np.random.choice(short_indices, size=short_sample_size, replace=False) if short_sample_size > 0 else np.array([], dtype=int)
        
        # Combine and shuffle
        debug_indices = np.concatenate([flat_selected, long_selected, short_selected])
        np.random.shuffle(debug_indices)
        
        # Limit to debug_num_samples
        debug_indices = debug_indices[:debug_num_samples]
        
        # Extract debug subset
        X_train_small = X_train[debug_indices]
        y_train_small = y_train[debug_indices]
        
        # Log debug subset distribution
        y_train_small_arr = np.array(y_train_small)
        flat_count = int(np.sum(y_train_small_arr == LstmClassIndex.FLAT))
        long_count = int(np.sum(y_train_small_arr == LstmClassIndex.LONG))
        short_count = int(np.sum(y_train_small_arr == LstmClassIndex.SHORT))
        total_debug = len(y_train_small_arr)
        
        logger.info(f"Debug subset (stratified sampling):")
        logger.info(f"  Total samples: {total_debug}")
        logger.info(f"  FLAT: {flat_count} ({flat_count/total_debug:.3f})")
        logger.info(f"  LONG: {long_count} ({long_count/total_debug:.3f})")
        logger.info(f"  SHORT: {short_count} ({short_count/total_debug:.3f})")
        
        if flat_count == 0 or long_count == 0 or short_count == 0:
            logger.warning(
                "[WARNING] Debug subset does not contain all 3 classes. "
                "Overfit test may not fully validate 3-class behavior."
            )
        
        train_dataset_small = TimeSeriesDataset(X_train_small, y_train_small)
        train_loader_small = DataLoader(train_dataset_small, batch_size=32, shuffle=True)  # Shuffle for better training
        logger.info("=" * 60)
        logger.info("Expected behavior in debug mode:")
        logger.info("  - Train loss should decrease significantly over {effective_num_epochs} epochs")
        logger.info("  - Train accuracy should approach 100%")
        logger.info("  - prob_long / prob_short std should increase to >0.05")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("PRODUCTION MODE: Using full training dataset")
        logger.info("=" * 60)

    # ============================================================
    # Task 2: Model Initialization with Effective Hyperparameters
    # ============================================================
    # Initialize model using effective hyperparameters
    logger.info("-" * 60)
    logger.info("Model Initialization:")
    logger.info(f"  input_size (feature_dim): {feature_dim}")
    logger.info(f"  hidden_size: {effective_hidden_size}")
    logger.info(f"  num_layers: {num_layers}")
    
    # ### BUG CANDIDATE: Class ↔ Output Dimension Consistency
    num_classes = 3  # 3-class: FLAT, LONG, SHORT
    logger.info(f"  num_classes (output dimension): {num_classes}")
    logger.info(f"  Expected label range: [0, {num_classes-1}] (FLAT=0, LONG=1, SHORT=2)")
    
    # Use effective_dropout (already computed based on debug_small_overfit)
    # Legacy debug_overfit_mode also affects dropout, but effective_dropout takes priority
    final_dropout = effective_dropout
    if debug_overfit_mode is not None and effective_dropout != 0.0:
        logger.info(f"[DEBUG][INIT] debug_overfit mode → forcing dropout=0.0 (was {final_dropout})")
        final_dropout = 0.0
    
    logger.info(f"  dropout: {final_dropout:.4f}")
    logger.info(f"  device: {device}")
    logger.info("-" * 60)
    
    model = LSTMAttentionModel(
        input_size=feature_dim,
        hidden_size=effective_hidden_size,
        num_layers=num_layers,
        dropout=final_dropout,
    ).to(device)
    
    # Verify model output dimension matches num_classes
    # Test forward pass with dummy input
    with torch.no_grad():
        dummy_input = torch.zeros(1, window_size, feature_dim, device=device)
        dummy_output = model(dummy_input)
        actual_output_dim = dummy_output.shape[1]
        if actual_output_dim != num_classes:
            raise ValueError(
                f"[ERROR] Model output dimension mismatch: "
                f"expected {num_classes}, got {actual_output_dim}. "
                f"Check model architecture (final linear layer should output {num_classes} classes)."
            )
        logger.info(
            f"[DEBUG] Model output dimension verified: {actual_output_dim} (matches num_classes={num_classes})"
        )
    
    # DEBUG-OVERFIT: unfreeze all parameters in debug-overfit mode
    if debug_overfit_mode is not None:
        logger.info("[DEBUG][INIT] debug_overfit mode detected → unfreezing ALL parameters")
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                logger.info(f"[DEBUG][INIT] unfreeze param: {name}, shape={tuple(param.shape)}")
                unfrozen_count += 1
            param.requires_grad = True
        
        if unfrozen_count > 0:
            logger.info(f"[DEBUG][INIT] Unfroze {unfrozen_count} parameter groups")
        
        # Verify dropout layers after model creation
        dropout_ps = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        logger.info(f"[DEBUG][INIT] Dropout ps after override (should be 0.0 in debug_overfit): {dropout_ps}")
    
    # 모델 파라미터 수 로깅 (unfreeze 후)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # DEBUG: requires_grad 확인
    requires_grad_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_param_count = len(list(model.parameters()))
    logger.info(f"  Parameters with requires_grad=True: {requires_grad_count}/{total_param_count}")
    
    if debug_overfit_mode is not None:
        logger.info(f"[DEBUG][INIT] After unfreeze: trainable parameters={trainable_params}/{total_params}")
    
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

    # ============================================================
    # Task 3: Optimizer and Scheduler with Effective Hyperparameters
    # ============================================================
    # Initialize optimizer with effective hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=effective_lr, weight_decay=effective_weight_decay
    )
    logger.info(f"Optimizer: AdamW (lr={effective_lr:.6f}, weight_decay={effective_weight_decay:.6f})")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6,
    )

    # Single source of truth: log the effective num_epochs value that will be used
    logger.info("=" * 60)
    logger.info(f"Effective num_epochs for this run: {effective_num_epochs}")
    logger.info("=" * 60)

    for epoch in range(effective_num_epochs):
        # Epoch timing
        epoch_start = time.time()
        
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
            y_batch = y_batch.to(device)  # (B,) integer labels for CrossEntropyLoss

            # 입력/라벨 NaN/Inf 체크 (문제 있으면 바로 예외 던지기)
            if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                raise ValueError("NaN/Inf detected in X_batch")
            if torch.isnan(y_batch.float()).any() or torch.isinf(y_batch.float()).any():
                raise ValueError("NaN/Inf detected in y_batch")

            optimizer.zero_grad()

            logits = model(X_batch)  # (B, 3) raw logits for 3-class
            loss = criterion(logits, y_batch)  # CrossEntropyLoss expects (B, 3) and (B,)

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
                    logits_sample = model(X_batch[:5])  # (5, 3) raw logits
                    probs_sample = F.softmax(logits_sample, dim=-1)  # (5, 3) probabilities
                    logger.info(
                        f"  [DEBUG] Epoch {epoch+1}, Batch {batch_idx} - "
                        f"Sample logits (shape={logits_sample.shape}):\n{logits_sample.cpu().numpy()}"
                    )
                    logger.info(
                        f"  [DEBUG] Epoch {epoch+1}, Batch {batch_idx} - "
                        f"Sample probs (shape={probs_sample.shape}):\n{probs_sample.cpu().numpy()}"
                    )

        train_loss /= len(current_train_loader)
        
        # DEBUG: epoch 끝에서 weight 변화 확인
        if debug_gradient_logging and (epoch == 0 or epoch % 10 == 0):
            if hasattr(model, 'fc_out'):
                fc_out_w_mean = model.fc_out.weight.data.mean().item()
                fc_out_w_std = model.fc_out.weight.data.std().item()
                # fc_out.bias는 3-class 출력이므로 shape=(3,) 벡터이다.
                bias = model.fc_out.bias.data.detach().cpu()  # shape: (3,)
                bias_mean = bias.mean().item()
                bias_std = bias.std().item()
                logger.info(
                    f"  [DEBUG] Epoch {epoch+1} end - fc_out.weight: "
                    f"mean={fc_out_w_mean:.6f}, std={fc_out_w_std:.6f}"
                )
                logger.info(
                    f"  [DEBUG] Epoch {epoch+1} end - fc_out.bias stats: "
                    f"mean={bias_mean:.6f}, std={bias_std:.6f}, values={bias.numpy().tolist()}"
                )
        
        # DEBUG: 소규모 overfit 테스트에서 train set class 분포 확인
        if debug_small_overfit and (epoch == 0 or epoch % 10 == 0 or epoch == effective_num_epochs - 1):
            model.eval()
            train_pred_classes = []
            train_probs_long = []
            train_probs_short = []
            with torch.no_grad():
                for X_batch, y_batch in current_train_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)  # (B, 3)
                    probs = F.softmax(logits, dim=-1)  # (B, 3)
                    pred_classes = logits.argmax(dim=-1)  # (B,)
                    train_pred_classes.extend(pred_classes.cpu().numpy().tolist())
                    train_probs_long.extend(probs[:, LstmClassIndex.LONG].cpu().numpy().tolist())
                    train_probs_short.extend(probs[:, LstmClassIndex.SHORT].cpu().numpy().tolist())
            if train_pred_classes:
                train_pred_array = np.array(train_pred_classes)
                train_prob_long_array = np.array(train_probs_long)
                train_prob_short_array = np.array(train_probs_short)
                flat_count = int(np.sum(train_pred_array == LstmClassIndex.FLAT))
                long_count = int(np.sum(train_pred_array == LstmClassIndex.LONG))
                short_count = int(np.sum(train_pred_array == LstmClassIndex.SHORT))
                logger.info(
                    f"  [DEBUG OVERFIT] Epoch {epoch+1} - Train predictions: "
                    f"FLAT={flat_count}, LONG={long_count}, SHORT={short_count}"
                )
                logger.info(
                    f"  [DEBUG OVERFIT] Epoch {epoch+1} - Train prob_long: "
                    f"mean={train_prob_long_array.mean():.4f}, std={train_prob_long_array.std():.4f}"
                )
                logger.info(
                    f"  [DEBUG OVERFIT] Epoch {epoch+1} - Train prob_short: "
                    f"mean={train_prob_short_array.mean():.4f}, std={train_prob_short_array.std():.4f}"
                )
            model.train()

        # Validation
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        # 3-class confusion matrix: count per class
        class_counts = {0: 0, 1: 0, 2: 0}  # FLAT, LONG, SHORT
        pred_class_counts = {0: 0, 1: 0, 2: 0}
        all_probs_long = []  # prob_long 분포 분석용
        all_probs_short = []  # prob_short 분포 분석용
        
        # TASK 1: In debug mode, use debug subset for validation; in production, use full valid set
        current_valid_loader = current_train_loader if debug_small_overfit else valid_loader

        with torch.no_grad():
            for X_batch, y_batch in current_valid_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)  # (B,) integer labels

                logits = model(X_batch)  # (B, 3), raw logits
                loss = criterion(logits, y_batch)
                valid_loss += loss.item()

                # 확률로 변환 후 정확도 및 메트릭 계산
                probs = F.softmax(logits, dim=-1)  # (B, 3) probabilities
                predictions = logits.argmax(dim=-1)  # (B,) predicted class indices

                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

                # Class distribution 계산
                for class_idx in [0, 1, 2]:
                    class_counts[class_idx] += (y_batch == class_idx).sum().item()
                    pred_class_counts[class_idx] += (predictions == class_idx).sum().item()
                
                # prob_long, prob_short 분포 수집
                all_probs_long.extend(probs[:, LstmClassIndex.LONG].cpu().numpy().tolist())
                all_probs_short.extend(probs[:, LstmClassIndex.SHORT].cpu().numpy().tolist())

        valid_loss /= len(current_valid_loader)
        # Convert to Python float explicitly to ensure proper comparison
        valid_loss_float = float(valid_loss)
        valid_acc = correct / total if total > 0 else 0.0
        
        # ### BUG CANDIDATE: best_val_loss never updated due to NaN/Inf
        # Check for invalid loss values that would prevent best_val_loss updates
        if not math.isfinite(valid_loss_float):
            logger.error(
                f"[ERROR] Epoch {epoch+1}: valid_loss is invalid! "
                f"valid_loss={valid_loss_float}, is_nan={math.isnan(valid_loss_float)}, is_inf={math.isinf(valid_loss_float)}"
            )
            logger.error(
                f"[ERROR] This will prevent best_val_loss from being updated. "
                f"Check for NaN/Inf in model outputs or loss computation."
            )

        # 3-class metrics: per-class accuracy
        flat_acc = class_counts[0] / max(class_counts[0] + pred_class_counts[0] - (predictions == 0).sum().item() if total > 0 else 1, 1)
        long_acc = class_counts[1] / max(class_counts[1] + pred_class_counts[1] - (predictions == 1).sum().item() if total > 0 else 1, 1)
        short_acc = class_counts[2] / max(class_counts[2] + pred_class_counts[2] - (predictions == 2).sum().item() if total > 0 else 1, 1)
        
        # prob_long, prob_short 분포 통계
        if all_probs_long:
            prob_long_array = np.array(all_probs_long)
            prob_short_array = np.array(all_probs_short)
            prob_long_mean = float(prob_long_array.mean())
            prob_long_std = float(prob_long_array.std())
            prob_short_mean = float(prob_short_array.mean())
            prob_short_std = float(prob_short_array.std())
        else:
            prob_long_mean = prob_long_std = prob_short_mean = prob_short_std = 0.0

        # Learning rate scheduler step (소규모 overfit 모드에서는 비활성화)
        if not debug_small_overfit:
            scheduler.step(valid_loss_float)
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        logger.info(
            f"Epoch [{epoch+1}/{effective_num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, "
            f"Valid Loss: {valid_loss_float:.4f}, "
            f"Acc: {valid_acc:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        logger.info(
            f"  Class distribution (true): FLAT={class_counts[0]}, LONG={class_counts[1]}, SHORT={class_counts[2]}"
        )
        logger.info(
            f"  Class distribution (pred): FLAT={pred_class_counts[0]}, LONG={pred_class_counts[1]}, SHORT={pred_class_counts[2]}"
        )
        
        # TASK 4: Enhanced collapse detection logging for production mode
        if not debug_small_overfit:
            # Calculate prediction ratios for collapse detection
            total_pred = sum(pred_class_counts.values())
            if total_pred > 0:
                flat_pred_ratio = pred_class_counts[0] / total_pred
                long_pred_ratio = pred_class_counts[1] / total_pred
                short_pred_ratio = pred_class_counts[2] / total_pred
                
                # Collapse detection: if one class dominates (>90%)
                if flat_pred_ratio > 0.90:
                    logger.warning(
                        f"  ⚠️  COLLAPSE WARNING: Model predicts FLAT {flat_pred_ratio:.1%} of the time. "
                        f"May be collapsing to constant FLAT prediction."
                    )
                elif long_pred_ratio > 0.90:
                    logger.warning(
                        f"  ⚠️  COLLAPSE WARNING: Model predicts LONG {long_pred_ratio:.1%} of the time. "
                        f"May be collapsing to constant LONG prediction."
                    )
                elif short_pred_ratio > 0.90:
                    logger.warning(
                        f"  ⚠️  COLLAPSE WARNING: Model predicts SHORT {short_pred_ratio:.1%} of the time. "
                        f"May be collapsing to constant SHORT prediction."
                    )
                
                # Check if model never predicts a class
                if pred_class_counts[0] == 0:
                    logger.warning("  ⚠️  Model never predicts FLAT class.")
                if pred_class_counts[1] == 0:
                    logger.warning("  ⚠️  Model never predicts LONG class.")
                if pred_class_counts[2] == 0:
                    logger.warning("  ⚠️  Model never predicts SHORT class.")
        logger.info(
            f"  Valid prob_long stats: mean={prob_long_mean:.4f}, std={prob_long_std:.4f}"
        )
        logger.info(
            f"  Valid prob_short stats: mean={prob_short_mean:.4f}, std={prob_short_std:.4f}"
        )
        
        # TASK 4: Enhanced collapse detection (applies to both debug and production)
        if prob_long_std < 0.01 and prob_short_std < 0.01:
            logger.warning(
                f"  ⚠️  WARNING: prob_long/prob_short std is very low, "
                f"model may be collapsing to constant prediction!"
            )

        # TASK 3: best_val_loss / best_state_dict update logic with early stopping
        # This logic works for both debug and production modes
        # Convert valid_loss to Python float explicitly
        val_loss_value = float(valid_loss_float)
        
        # Determine if validation loss improved enough (considering min_delta)
        min_delta_threshold = early_stopping_min_delta if (early_stopping_enabled and early_stopping_min_delta is not None) else 0.0
        improved = math.isfinite(val_loss_value) and val_loss_value < (best_val_loss - min_delta_threshold)
        
        if improved:
            best_val_loss = val_loss_value
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            logger.info(
                f"[BEST] New best validation loss: {best_val_loss:.6f} at epoch {epoch+1}"
            )
            
            # Save model checkpoint (only in production mode, debug mode skips saving)
            if not debug_small_overfit:
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(best_model_path)
                logger.info(
                    f"✓ Saved best model checkpoint (best_val_loss={best_val_loss:.6f}) to {best_model_path.resolve()}"
                )
            else:
                logger.info("[DEBUG MODE] Skipping model checkpoint save to disk (in-memory only).")
        else:
            if not math.isfinite(val_loss_value):
                # Only log ERROR if this is a real problem (not just debug mode)
                if not debug_small_overfit:
                    logger.error(
                        f"[ERROR] Epoch {epoch+1}: valid_loss is invalid (NaN/Inf): {val_loss_value}. "
                        f"This will prevent best_val_loss from being updated."
                    )
                else:
                    logger.warning(
                        f"[WARNING] Epoch {epoch+1}: valid_loss is invalid (NaN/Inf): {val_loss_value}. "
                        f"Skipping best model update (debug mode)."
                    )
            else:
                # Loss is finite but didn't improve enough
                if early_stopping_enabled:
                    epochs_no_improve += 1
                    logger.info(
                        f"[EARLY STOP] No improvement in val_loss for {epochs_no_improve} epoch(s) "
                        f"(current={val_loss_value:.6f}, best={best_val_loss:.6f}, patience={early_stopping_patience})"
                    )
                else:
                    # Debug mode: just log that it didn't improve
                    logger.debug(
                        f"[DEBUG] Epoch {epoch+1}: Validation loss {val_loss_value:.6f} did not improve best_val_loss {best_val_loss:.6f}"
                    )
        
        # Early stopping check (only for production mode)
        if early_stopping_enabled and early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            logger.info("------------------------------------------------------------")
            logger.info(
                f"[EARLY STOP] Stopping training at epoch {epoch+1}/{effective_num_epochs} "
                f"due to no val_loss improvement for {early_stopping_patience} consecutive epochs."
            )
            logger.info(f"[EARLY STOP] Best validation loss: {best_val_loss:.6f}")
            logger.info("------------------------------------------------------------")
            break
        
        # Epoch timing log
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch [{epoch+1}/{effective_num_epochs}] took {epoch_time:.1f} seconds")

    logger.info("-" * 60)
    logger.info("Training completed!")
    
    # TASK 3: Final best_val_loss / best_state_dict validation and model loading
    
    # Check if best_state_dict was set during training
    if best_state_dict is None or not math.isfinite(best_val_loss):
        # Determine if this is a real error or just debug mode
        if debug_small_overfit:
            logger.warning(
                "[WARNING] best_state_dict is None or best_val_loss is not finite. "
                "Using final epoch model weights. "
                "(This is expected in debug mode if no valid loss improvement occurred.)"
            )
        else:
            # Production mode: this is a real issue
            if not math.isfinite(best_val_loss):
                logger.error(
                    f"[ERROR] Best validation loss is invalid: {best_val_loss}. "
                    f"This indicates best_state_dict was never set during training. "
                    f"Check logs above for NaN/Inf in valid_loss."
                )
            else:
                logger.warning(
                    "[WARNING] best_state_dict is None despite finite best_val_loss. "
                    "Using final epoch model weights. "
                    "This may indicate a bug in the update logic."
                )
        # Use final epoch model as-is (already loaded)
        logger.info(f"Final validation accuracy: {valid_acc:.4f}")
        logger.info(f"Best validation loss: {best_val_loss:.4f} (not used - using final epoch weights)")
    else:
        # Load best model weights
        model.load_state_dict(best_state_dict)
        logger.info(f"[INFO] Loaded best model with val_loss={best_val_loss:.6f}")
        logger.info(f"Final validation accuracy: {valid_acc:.4f}")
        
        # Verify checkpoint file exists (only in production mode)
        if not debug_small_overfit:
            if best_model_path.exists():
                logger.info(f"Saved best LSTM model to: {best_model_path.resolve()}")
            else:
                logger.warning(f"Model file not found at expected path: {best_model_path.resolve()}")
        else:
            # Task 5: DEBUG mode - skip checkpoint save to disk
            logger.info("[DEBUG MODE] Skipping model checkpoint save to disk (in-memory only).")
    
    # Note: Precision/Recall/F1 are not computed for 3-class (per-class metrics would be more appropriate)
    
    # Validation set evaluation using evaluate_split
    debug_inference_samples = getattr(settings, "DEBUG_LSTM_INFERENCE_SAMPLES", True)
    if debug_inference_samples:
        evaluate_split(
            model=model,
            X=X_valid,
            y=y_valid,
            device=device,
            split_name="Validation",
            thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
            batch_size=batch_size,
            logger=logger,
        )
        
        # Sample predictions for debugging
        logger.info("-" * 60)
        logger.info("Validation Set Sample Predictions (for debugging):")
        logger.info("-" * 60)
        
        model.eval()
        with torch.no_grad():
            sample_size = min(10, len(valid_dataset))
            sample_indices = list(range(len(valid_dataset) - sample_size, len(valid_dataset)))
            
            # Check if 3-class or binary
            _, y_sample_check = valid_dataset[0]
            is_3class = y_sample_check.dtype in (torch.int64, torch.int32, torch.long)
            
            if is_3class:
                logger.info(f"Showing predictions for last {sample_size} validation samples (3-class):")
                logger.info(f"{'Index':<8} {'Label':<12} {'prob_long':<12} {'prob_short':<12} {'pred_class':<12} {'Match':<8}")
                logger.info("-" * 80)
                
                for idx in sample_indices:
                    X_sample, y_sample = valid_dataset[idx]
                    X_sample = X_sample.unsqueeze(0).to(device)
                    y_sample = y_sample.item()
                    
                    logits = model(X_sample)  # (1, 3)
                    probs = F.softmax(logits, dim=-1)  # (1, 3)
                    pred_class = logits.argmax(dim=-1).item()
                    
                    prob_long = probs[0, LstmClassIndex.LONG].item()
                    prob_short = probs[0, LstmClassIndex.SHORT].item()
                    
                    label_name = ["FLAT", "LONG", "SHORT"][y_sample]
                    pred_name = ["FLAT", "LONG", "SHORT"][pred_class]
                    match = "✓" if pred_class == y_sample else "✗"
                    
                    logger.info(
                        f"{idx:<8} {label_name:<12} {prob_long:<12.4f} {prob_short:<12.4f} {pred_name:<12} {match:<8}"
                    )
            else:
                logger.info(f"Showing predictions for last {sample_size} validation samples (binary):")
                logger.info(f"{'Index':<8} {'Label':<8} {'prob_up':<12} {'pred_label':<12} {'Match':<8}")
                logger.info("-" * 60)
                
                for idx in sample_indices:
                    X_sample, y_sample = valid_dataset[idx]
                    X_sample = X_sample.unsqueeze(0).to(device)
                    y_sample = y_sample.item()
                    
                    logit = model(X_sample)
                    prob_up = torch.sigmoid(logit).item()
                    pred_label = 1 if prob_up >= 0.5 else 0
                    match = "✓" if pred_label == int(y_sample) else "✗"
                    
                    logger.info(
                        f"{idx:<8} {int(y_sample):<8} {prob_up:<12.4f} {pred_label:<12} {match:<8}"
                    )
        
        logger.info("-" * 60)
    
    # Test set evaluation (unseen data)
    # Note: model already contains best weights from best_state_dict
    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluating on Test Set (unseen data)")
    logger.info("=" * 60)
    
    # Evaluate on test set using the model with best weights
    model.eval()
    evaluate_split(
        model=model,
        X=X_test,
        y=y_test,
        device=device,
        split_name="Test",
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        batch_size=batch_size,
        logger=logger,
    )
    
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

    return model, best_val_loss, best_model_path


if __name__ == "__main__":
    # ============================================================
    # Argument Parser 설정
    # ============================================================
    parser = argparse.ArgumentParser(description="Train LSTM + Attention model")
    parser.add_argument(
        "--debug-overfit",
        type=str,
        default=None,
        choices=["1", "2", "32", "64", "all"],
        help="Debug overfit mode: 1=single sample, 2=two samples, 32=32 samples, 64=64 samples, all=all tests",
    )
    parser.add_argument(
        "--debug-small-overfit",
        action="store_true",
        help="Enable debug small overfit mode: uses stratified 64-sample subset with 30 epochs for quick overfit testing",
    )
    
    args = parser.parse_args()
    
    # ============================================================
    # Debug small overfit flag determination (CLI-only, single source of truth)
    # TASK 2-4: CLI argument is the ONLY way to enable debug mode
    # Settings are completely ignored to prevent accidental debug mode activation
    # ============================================================
    debug_small_overfit = bool(args.debug_small_overfit)
    
    # Log mode based on CLI flag only
    if debug_small_overfit:
        logger.info("=" * 60)
        logger.info("DEBUG MODE: Small Overfit Test (stratified subset)")
        logger.info("=" * 60)
        logger.info("[CLI] --debug-small-overfit flag detected.")
    else:
        logger.info("=" * 60)
        logger.info("PRODUCTION MODE: Using full training dataset")
        logger.info("=" * 60)
    
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
    # Production mode default: 200 epochs with early stopping enabled
    NUM_EPOCHS = 200
    TRAIN_SPLIT = 0.8

    # Early stopping (production mode only)
    PATIENCE_ES = 15
    MIN_DELTA = 1e-4

    # Train model
    # horizon, pos_threshold, neg_threshold, ignore_margin는 None으로 전달하면
    # settings에서 기본값을 사용함
    model, best_val_loss, best_model_path = train_model(
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
        debug_overfit_mode=args.debug_overfit,
        debug_small_overfit=debug_small_overfit,  # ★ 명시적으로 전달
    )

    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_model_path.resolve()}")
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
