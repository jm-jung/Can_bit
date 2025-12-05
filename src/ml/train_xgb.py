"""
Train XGBoost model for BTC price prediction.

This script trains an XGBoost classifier with comprehensive logging for overfitting detection,
similar to the LSTM training pipeline.

Usage:
    python -m src.ml.train_xgb

Overfitting Detection:
    - Check Train vs Valid vs Test performance gaps
    - Large accuracy/precision/recall differences indicate overfitting
    - ROC-AUC should be similar across splits
    - Confusion matrix patterns should be consistent
    - Threshold analysis helps identify optimal decision boundaries
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

from src.core.config import settings, PROJECT_ROOT
from src.features.ml_feature_config import MLFeatureConfig
from src.ml.evaluation import evaluate_ml_split, log_label_distribution
from src.ml.features import build_ml_dataset
from src.ml.scalers import create_scaler, save_scaler, apply_scaler, ScalerType
from src.ml.label_processor import LabelProcessor, LabelConfig
from src.ml.features.feature_blocks import FeatureBlockManager
from src.services.ohlcv_service import load_ohlcv_df

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class XGBConfig:
    """XGBoost hyperparameter configuration."""
    n_estimators: int = 400
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    min_child_weight: int = 3
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 2.0
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for metadata storage."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_metric": self.eval_metric,
        }


def make_time_series_split_3way(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    min_test_samples: int = 200,
) -> dict:
    """
    Split time-series data into train/valid/test sets while preserving temporal order.
    
    Similar to LSTM's make_time_series_splits but for pandas DataFrames.
    
    Args:
        X: Feature DataFrame
        y: Label Series (must have same length as X)
        train_ratio: Proportion for training (default: 0.7)
        valid_ratio: Proportion for validation (default: 0.15)
        min_test_samples: Minimum test samples (default: 200)
    
    Returns:
        Dictionary with X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    # Use X.shape[0] as the reference for split boundaries
    N = X.shape[0]
    train_end = int(N * train_ratio)
    valid_end = int(N * (train_ratio + valid_ratio))
    
    # Ensure minimum test samples
    test_len = N - valid_end
    if test_len < min_test_samples:
        needed = min_test_samples - test_len
        valid_end = max(train_end + 1, valid_end - needed)
        test_len = N - valid_end
        
        if valid_end <= train_end:
            raise ValueError(
                f"Cannot create valid splits: N={N}, train_ratio={train_ratio}, "
                f"valid_ratio={valid_ratio}, min_test_samples={min_test_samples}"
            )
    
    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()
    X_valid = X.iloc[train_end:valid_end].copy()
    y_valid = y.iloc[train_end:valid_end].copy()
    X_test = X.iloc[valid_end:].copy()
    y_test = y.iloc[valid_end:].copy()
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_xgb_model(
    horizon: int = 20,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    xgb_config: XGBConfig | None = None,
    use_events: bool | None = None,
    feature_preset: str = "base",
    long_threshold: float = 0.002,
    short_threshold: float = 0.002,
    hold_threshold: float = 0.0005,
    enable_hold_labels: bool = False,
    # Phase E: New parameters
    scaler_type: ScalerType = "none",
    label_mode: str = "classification",
    timeframe: str | None = None,
    use_regime_features: bool = False,
    atr_percentile_threshold: float = 0.70,
) -> XGBClassifier:
    """
    Train XGBoost classifier for BTC price direction prediction with comprehensive logging.

    Args:
        horizon: Number of periods ahead to predict (default: 20)
        train_ratio: Proportion of data for training (default: 0.7)
        valid_ratio: Proportion of data for validation (default: 0.15)
        xgb_config: XGBConfig instance with hyperparameters (default: conservative defaults)
        use_events: Whether to use event features (default: from settings)
            Note: This is overridden by feature_config if feature_preset is provided
        feature_preset: Feature preset name ("base", "extended_safe", "extended_full")
            Default: "base" (maintains backward compatibility)
        long_threshold: Return threshold for LONG label (default: 0.002 = 0.2%)
        short_threshold: Return threshold for SHORT label (default: 0.002 = 0.2%)
        hold_threshold: Return threshold for HOLD zone (default: 0.0005 = 0.05%)
        enable_hold_labels: If True, generate HOLD labels and exclude them from training (default: False)
        scaler_type: Feature scaler type ("standard", "robust", "minmax", "none") - Phase E
        label_mode: Label mode ("classification" or "regression") - Phase E
        timeframe: Target timeframe (e.g., "1m", "3m", "5m", "15m", "30m") - Phase E
        use_regime_features: Enable volatility regime features - Phase E
        atr_percentile_threshold: ATR percentile threshold for regime classification - Phase E

    Returns:
        Trained XGBClassifier model
    """
    # Use conservative default config if not provided
    if xgb_config is None:
        xgb_config = XGBConfig()
    logger.info("=" * 60)
    logger.info("XGBoost Model Training")
    logger.info("=" * 60)
    
    # (0) Feature Config
    feature_config = MLFeatureConfig.from_preset(feature_preset)
    logger.info(f"Feature preset: {feature_preset}")
    logger.info(f"Feature config: {feature_config}")
    
    # (1) Data & Label Logging
    logger.info("Loading OHLCV data...")
    logger.info(f"Phase E: timeframe={timeframe}, scaler_type={scaler_type}, label_mode={label_mode}")
    df = load_ohlcv_df(timeframe=timeframe)  # Phase E: Multi-timeframe support
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of OHLCV data (timeframe: {timeframe})")

    if use_events is None:
        use_events = settings.EVENTS_ENABLED
    
    logger.info("=" * 60)
    logger.info("Building ML Dataset")
    logger.info("=" * 60)
    logger.info(f"Horizon: {horizon}, Feature preset: {feature_preset}")
    logger.info(f"Use events (from config): {feature_config.use_event_features}")
    logger.info(f"Label thresholds: long={long_threshold}, short={short_threshold}, hold={hold_threshold}")
    logger.info(f"Enable HOLD labels: {enable_hold_labels}")
    
    # Extract symbol and timeframe from settings or use defaults
    symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    if timeframe is None:
        timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    else:
        timeframe = timeframe.lower()
    
    dataset_result = build_ml_dataset(
        df,
        horizon=horizon,
        symbol=symbol,
        timeframe=timeframe,
        use_events=use_events,
        feature_config=feature_config,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        hold_threshold=hold_threshold,
        enable_hold_labels=enable_hold_labels,
    )
    
    # Handle return value based on enable_hold_labels
    if enable_hold_labels:
        X, y, y_long, y_short, y_hold = dataset_result
    else:
        X, y, y_long, y_short = dataset_result
        y_hold = None

    # Align X and y lengths (use X.shape[0] as the reference)
    X_len = X.shape[0]
    y_len = len(y)
    y_long_len = len(y_long)
    y_short_len = len(y_short)
    y_hold_len = len(y_hold) if y_hold is not None else 0
    
    if X_len != y_len or X_len != y_long_len or X_len != y_short_len:
        logger.warning(
            f"X and y length mismatch: X.shape[0]={X_len}, len(y)={y_len}, "
            f"len(y_long)={y_long_len}, len(y_short)={y_short_len}. "
            f"Truncating to match X length."
        )
        y = y.iloc[:X_len] if hasattr(y, 'iloc') else y[:X_len]
        y_long = y_long.iloc[:X_len] if hasattr(y_long, 'iloc') else y_long[:X_len]
        y_short = y_short.iloc[:X_len] if hasattr(y_short, 'iloc') else y_short[:X_len]
        if y_hold is not None:
            y_hold = y_hold.iloc[:X_len] if hasattr(y_hold, 'iloc') else y_hold[:X_len]
        logger.info(f"After alignment: X.shape[0]={X.shape[0]}, len(y)={len(y)}, len(y_long)={len(y_long)}, len(y_short)={len(y_short)}, len(y_hold)={len(y_hold) if y_hold is not None else 0}")
    else:
        logger.info(f"X and y lengths match: {X_len}")

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}, y_long={y_long.shape}, y_short={y_short.shape}, y_hold={y_hold.shape if y_hold is not None else 'N/A'}")
    logger.info(f"Feature count: {len(X.columns)}")
    
    # Step C: Exclude HOLD samples from training if enable_hold_labels=True
    if enable_hold_labels and y_hold is not None:
        hold_mask = (y_hold == 1)
        hold_count = int(hold_mask.sum())
        total_before = len(X)
        
        logger.info(f"[HOLD Filter] Excluding HOLD samples from training: {hold_count} / {total_before} ({hold_count / total_before * 100.0:.2f}%)")
        
        # Filter out HOLD samples
        non_hold_mask = ~hold_mask
        X = X.loc[non_hold_mask].reset_index(drop=True)
        y = y.loc[non_hold_mask].reset_index(drop=True)
        y_long = y_long.loc[non_hold_mask].reset_index(drop=True)
        y_short = y_short.loc[non_hold_mask].reset_index(drop=True)
        
        logger.info(f"[HOLD Filter] After filtering: {len(X)} / {total_before} samples remaining ({len(X) / total_before * 100.0:.2f}%)")
    
    # Check for empty dataset
    if X.shape[0] == 0 or y.shape[0] == 0 or y_long.shape[0] == 0 or y_short.shape[0] == 0:
        logger.error(
            "[XGB Train] Empty dataset after feature engineering. "
            "X shape=%s, y shape=%s, y_long shape=%s, y_short shape=%s, feature_preset=%s",
            X.shape, y.shape, y_long.shape, y_short.shape, feature_preset
        )
        raise ValueError(
            f"Empty dataset: no samples available for training. "
            f"X shape={X.shape}, y shape={y.shape}, y_long shape={y_long.shape}, y_short shape={y_short.shape}, feature_preset={feature_preset}. "
            "Check feature engineering and NaN handling."
        )
    
    # Feature breakdown by category
    event_cols = [c for c in X.columns if c.startswith("event_")]
    trend_cols = [c for c in X.columns if c.startswith("feat_trend_")]
    vol_cols = [c for c in X.columns if c.startswith("feat_vol_")]
    volume_cols = [c for c in X.columns if c.startswith("feat_volu_")]
    struct_cols = [c for c in X.columns if c.startswith("feat_struct_")]
    base_cols = [c for c in X.columns if not any(c.startswith(prefix) for prefix in ["event_", "feat_trend_", "feat_vol_", "feat_volu_", "feat_struct_"])]
    
    logger.info(f"  Base features: {len(base_cols)}")
    if event_cols:
        logger.info(f"  Event features: {len(event_cols)}")
    if trend_cols:
        logger.info(f"  Extended trend features: {len(trend_cols)}")
    if vol_cols:
        logger.info(f"  Volatility features: {len(vol_cols)}")
    if volume_cols:
        logger.info(f"  Volume features: {len(volume_cols)}")
    if struct_cols:
        logger.info(f"  Structure features: {len(struct_cols)}")
    
    # Feature columns logging
    logger.info("=" * 60)
    logger.info("[XGB Train] Feature Columns")
    logger.info("=" * 60)
    logger.info(f"[XGB Train] Using {len(X.columns)} feature columns:")
    logger.info(f"[XGB Train] FEATURE_COLS = {list(X.columns)}")
    logger.info("=" * 60)
    
    # Label distribution logging
    log_label_distribution(y, "Full Dataset", logger)
    log_label_distribution(y_long, "Full Dataset (LONG)", logger)
    log_label_distribution(y_short, "Full Dataset (SHORT)", logger)
    if enable_hold_labels and y_hold is not None:
        # Log HOLD distribution before filtering
        hold_total = len(y_hold)
        hold_count = int((y_hold == 1).sum())
        logger.info(
            "[Label Distribution] HOLD: positive=%d (%.2f%%), negative=%d (%.2f%%)",
            hold_count, hold_count / hold_total * 100.0 if hold_total > 0 else 0.0,
            hold_total - hold_count, (hold_total - hold_count) / hold_total * 100.0 if hold_total > 0 else 0.0,
        )

    # (2) Time-series Split
    logger.info("=" * 60)
    logger.info("Time-series Split")
    logger.info("=" * 60)
    logger.info(f"Train ratio: {train_ratio}, Valid ratio: {valid_ratio}, Test ratio: {1.0 - train_ratio - valid_ratio:.3f}")
    
    # Split for LONG model
    splits_long = make_time_series_split_3way(
        X, y_long,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        min_test_samples=200,
    )
    
    # Split for SHORT model (same X, different y)
    splits_short = make_time_series_split_3way(
        X, y_short,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        min_test_samples=200,
    )
    
    # Phase E: Apply feature scaling (fit on train only, transform all splits)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase E: Feature Scaling")
    logger.info("=" * 60)
    scaler = create_scaler(scaler_type)
    if scaler is not None:
        logger.info(f"Fitting {scaler_type} scaler on training data only...")
        # Fit scaler on training data
        X_train_long_scaled = apply_scaler(splits_long["X_train"], scaler, fit=True)
        # Transform validation and test sets
        splits_long["X_valid"] = apply_scaler(splits_long["X_valid"], scaler, fit=False)
        splits_long["X_test"] = apply_scaler(splits_long["X_test"], scaler, fit=False)
        splits_long["X_train"] = X_train_long_scaled
        
        # Apply same scaler to SHORT splits (already fitted)
        splits_short["X_train"] = apply_scaler(splits_short["X_train"], scaler, fit=False)
        splits_short["X_valid"] = apply_scaler(splits_short["X_valid"], scaler, fit=False)
        splits_short["X_test"] = apply_scaler(splits_short["X_test"], scaler, fit=False)
        
        logger.info(f"✅ Applied {scaler_type} scaling to all splits")
    else:
        logger.info("No scaling applied (scaler_type='none')")
        scaler = None
    
    # LONG splits
    X_train_long = splits_long["X_train"]
    y_train_long = splits_long["y_train"]
    X_valid_long = splits_long["X_valid"]
    y_valid_long = splits_long["y_valid"]
    X_test_long = splits_long["X_test"]
    y_test_long = splits_long["y_test"]
    
    # SHORT splits
    X_train_short = splits_short["X_train"]
    y_train_short = splits_short["y_train"]
    X_valid_short = splits_short["X_valid"]
    y_valid_short = splits_short["y_valid"]
    X_test_short = splits_short["X_test"]
    y_test_short = splits_short["y_test"]
    
    # For backward compatibility, also keep original splits
    splits = splits_long
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_valid = splits["X_valid"]
    y_valid = splits["y_valid"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    
    logger.info("-" * 60)
    logger.info("Split Summary:")
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_valid shape: {X_valid.shape}")
    logger.info(f"  X_test shape: {X_test.shape}")
    logger.info("-" * 60)
    
    # Check for empty train/valid splits
    if X_train.shape[0] == 0 or X_valid.shape[0] == 0:
        logger.error(
            "[XGB Train] Train/Valid split is empty. "
            "X_train shape=%s, X_valid shape=%s, feature_preset=%s",
            X_train.shape, X_valid.shape, feature_preset
        )
        raise ValueError(
            f"Empty train/valid split: X_train shape={X_train.shape}, "
            f"X_valid shape={X_valid.shape}, feature_preset={feature_preset}. "
            "Adjust split ratios or check feature filtering."
        )
    
    # Label distribution for each split
    log_label_distribution(y_train, "Train", logger)
    log_label_distribution(y_valid, "Valid", logger)
    log_label_distribution(y_test, "Test", logger)
    log_label_distribution(y_train_long, "Train (LONG)", logger)
    log_label_distribution(y_train_short, "Train (SHORT)", logger)

    # (3) Model Training - LONG and SHORT models
    logger.info("=" * 60)
    logger.info("Model Training (LONG and SHORT models)")
    logger.info("=" * 60)
    logger.info("Final XGBoost Hyperparameters:")
    logger.info(f"  n_estimators: {xgb_config.n_estimators}")
    logger.info(f"  max_depth: {xgb_config.max_depth}")
    logger.info(f"  learning_rate: {xgb_config.learning_rate}")
    logger.info(f"  subsample: {xgb_config.subsample}")
    logger.info(f"  colsample_bytree: {xgb_config.colsample_bytree}")
    logger.info(f"  min_child_weight: {xgb_config.min_child_weight}")
    logger.info(f"  gamma: {xgb_config.gamma}")
    logger.info(f"  reg_alpha (L1): {xgb_config.reg_alpha}")
    logger.info(f"  reg_lambda (L2): {xgb_config.reg_lambda}")
    logger.info(f"  early_stopping_rounds: {xgb_config.early_stopping_rounds}")
    logger.info(f"  eval_metric: {xgb_config.eval_metric}")
    logger.info("-" * 60)
    
    def create_xgb_model() -> XGBClassifier:
        """Create XGBClassifier with configured hyperparameters."""
        return XGBClassifier(
            n_estimators=xgb_config.n_estimators,
            max_depth=xgb_config.max_depth,
            learning_rate=xgb_config.learning_rate,
            subsample=xgb_config.subsample,
            colsample_bytree=xgb_config.colsample_bytree,
            min_child_weight=xgb_config.min_child_weight,
            gamma=xgb_config.gamma,
            reg_alpha=xgb_config.reg_alpha,
            reg_lambda=xgb_config.reg_lambda,
            eval_metric=xgb_config.eval_metric,
            random_state=42,
            n_jobs=-1,
        )
    
    # Train LONG model
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training LONG XGBoost model for {symbol}-{timeframe}...".format(symbol=symbol, timeframe=timeframe))
    logger.info("=" * 60)
    long_model = create_xgb_model()
    long_model.fit(
        X_train_long,
        y_train_long,
        eval_set=[(X_valid_long, y_valid_long)],
        early_stopping_rounds=xgb_config.early_stopping_rounds,
        verbose=False,
    )
    
    # Log early stopping info for LONG model
    if hasattr(long_model, "best_iteration") and long_model.best_iteration is not None:
        logger.info(f"LONG model: Early stopping triggered at iteration {long_model.best_iteration + 1} / {xgb_config.n_estimators}")
    else:
        logger.info(f"LONG model: Training completed all {xgb_config.n_estimators} iterations")
    logger.info("LONG model training completed!")
    
    # Train SHORT model
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training SHORT XGBoost model for {symbol}-{timeframe}...".format(symbol=symbol, timeframe=timeframe))
    logger.info("=" * 60)
    short_model = create_xgb_model()
    short_model.fit(
        X_train_short,
        y_train_short,
        eval_set=[(X_valid_short, y_valid_short)],
        early_stopping_rounds=xgb_config.early_stopping_rounds,
        verbose=False,
    )
    
    # Log early stopping info for SHORT model
    if hasattr(short_model, "best_iteration") and short_model.best_iteration is not None:
        logger.info(f"SHORT model: Early stopping triggered at iteration {short_model.best_iteration + 1} / {xgb_config.n_estimators}")
    else:
        logger.info(f"SHORT model: Training completed all {xgb_config.n_estimators} iterations")
    logger.info("SHORT model training completed!")
    
    # For backward compatibility, keep model as long_model
    model = long_model

    # (4) Performance Evaluation - LONG and SHORT models separately
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    
    threshold_list = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    def compute_roc_auc(model, X, y) -> float:
        """Compute ROC-AUC score."""
        try:
            proba = model.predict_proba(X)
            y_proba = proba[:, 1]
            y_labels = np.array(y) if not isinstance(y, np.ndarray) else y
            return float(roc_auc_score(y_labels, y_proba))
        except ValueError:
            return 0.0
    
    # ========== LONG Model Evaluation ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("[LONG Model] Evaluation")
    logger.info("=" * 60)
    
    # ROC-AUC for LONG model
    long_auc_train = compute_roc_auc(long_model, X_train_long, y_train_long)
    long_auc_valid = compute_roc_auc(long_model, X_valid_long, y_valid_long)
    long_auc_test = compute_roc_auc(long_model, X_test_long, y_test_long)
    
    logger.info("")
    logger.info("[LONG] ROC-AUC:")
    logger.info(f"  Train: {long_auc_train:.4f}")
    logger.info(f"  Valid: {long_auc_valid:.4f}")
    logger.info(f"  Test:  {long_auc_test:.4f}")
    logger.info("-" * 60)
    
    # Threshold analysis for LONG model
    logger.info("")
    logger.info("[LONG] Threshold Analysis - Train Set:")
    logger.info("-" * 60)
    evaluate_ml_split(
        long_model,
        X_train_long,
        y_train_long,
        split_name="Train",
        thresholds=threshold_list,
        logger_instance=logger,
    )
    
    logger.info("")
    logger.info("[LONG] Threshold Analysis - Validation Set:")
    logger.info("-" * 60)
    long_valid_metrics = evaluate_ml_split(
        long_model,
        X_valid_long,
        y_valid_long,
        split_name="Validation",
        thresholds=threshold_list,
        logger_instance=logger,
        return_metrics=True,
    )
    
    logger.info("")
    logger.info("[LONG] Threshold Analysis - Test Set:")
    logger.info("-" * 60)
    evaluate_ml_split(
        long_model,
        X_test_long,
        y_test_long,
        split_name="Test",
        thresholds=threshold_list,
        logger_instance=logger,
    )
    
    # Best threshold for LONG model (based on Validation F1)
    logger.info("")
    logger.info("[LONG] Best Threshold Selection (based on Validation F1):")
    logger.info("-" * 60)
    if long_valid_metrics is None or "threshold_metrics" not in long_valid_metrics:
        logger.warning("Could not extract validation metrics. Using default threshold 0.5.")
        long_best_threshold = 0.5
        long_best_f1 = 0.0
    else:
        threshold_metrics = long_valid_metrics["threshold_metrics"]
        f1_list = [m["f1"] for m in threshold_metrics]
        best_idx = int(np.argmax(f1_list))
        long_best_threshold = threshold_metrics[best_idx]["threshold"]
        long_best_f1 = threshold_metrics[best_idx]["f1"]
        logger.info(f"  Best threshold: {long_best_threshold:.2f}")
        logger.info(f"  Best Validation F1: {long_best_f1:.4f}")
        logger.info(f"  Best Validation Precision: {threshold_metrics[best_idx]['precision']:.4f}")
        logger.info(f"  Best Validation Recall: {threshold_metrics[best_idx]['recall']:.4f}")
    
    # ========== SHORT Model Evaluation ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("[SHORT Model] Evaluation")
    logger.info("=" * 60)
    
    # ROC-AUC for SHORT model
    short_auc_train = compute_roc_auc(short_model, X_train_short, y_train_short)
    short_auc_valid = compute_roc_auc(short_model, X_valid_short, y_valid_short)
    short_auc_test = compute_roc_auc(short_model, X_test_short, y_test_short)
    
    logger.info("")
    logger.info("[SHORT] ROC-AUC:")
    logger.info(f"  Train: {short_auc_train:.4f}")
    logger.info(f"  Valid: {short_auc_valid:.4f}")
    logger.info(f"  Test:  {short_auc_test:.4f}")
    logger.info("-" * 60)
    
    # Threshold analysis for SHORT model
    logger.info("")
    logger.info("[SHORT] Threshold Analysis - Train Set:")
    logger.info("-" * 60)
    evaluate_ml_split(
        short_model,
        X_train_short,
        y_train_short,
        split_name="Train",
        thresholds=threshold_list,
        logger_instance=logger,
    )
    
    logger.info("")
    logger.info("[SHORT] Threshold Analysis - Validation Set:")
    logger.info("-" * 60)
    short_valid_metrics = evaluate_ml_split(
        short_model,
        X_valid_short,
        y_valid_short,
        split_name="Validation",
        thresholds=threshold_list,
        logger_instance=logger,
        return_metrics=True,
    )
    
    logger.info("")
    logger.info("[SHORT] Threshold Analysis - Test Set:")
    logger.info("-" * 60)
    evaluate_ml_split(
        short_model,
        X_test_short,
        y_test_short,
        split_name="Test",
        thresholds=threshold_list,
        logger_instance=logger,
    )
    
    # Best threshold for SHORT model (based on Validation F1)
    logger.info("")
    logger.info("[SHORT] Best Threshold Selection (based on Validation F1):")
    logger.info("-" * 60)
    if short_valid_metrics is None or "threshold_metrics" not in short_valid_metrics:
        logger.warning("Could not extract validation metrics. Using default threshold 0.5.")
        short_best_threshold = 0.5
        short_best_f1 = 0.0
    else:
        threshold_metrics = short_valid_metrics["threshold_metrics"]
        f1_list = [m["f1"] for m in threshold_metrics]
        best_idx = int(np.argmax(f1_list))
        short_best_threshold = threshold_metrics[best_idx]["threshold"]
        short_best_f1 = threshold_metrics[best_idx]["f1"]
        logger.info(f"  Best threshold: {short_best_threshold:.2f}")
        logger.info(f"  Best Validation F1: {short_best_f1:.4f}")
        logger.info(f"  Best Validation Precision: {threshold_metrics[best_idx]['precision']:.4f}")
        logger.info(f"  Best Validation Recall: {threshold_metrics[best_idx]['recall']:.4f}")
    
    # Use LONG model's best threshold for backward compatibility
    best_threshold = long_best_threshold
    best_f1 = long_best_f1
    valid_metrics = long_valid_metrics
    
    # (4-2) Re-evaluate Train/Valid/Test with best threshold (LONG model for backward compatibility)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Re-evaluation with Best Threshold (LONG model)")
    logger.info("=" * 60)
    
    def evaluate_with_threshold(model, X, y, split_name: str, threshold: float) -> dict[str, float]:
        """Evaluate model with specific threshold and return metrics."""
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        proba = model.predict_proba(X)
        y_proba = proba[:, 1]
        y_labels = np.array(y) if not isinstance(y, np.ndarray) else y
        
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_labels, y_pred)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            if y_pred[0] == 0:
                tn, fp, fn, tp = len(y_labels), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(y_labels)
        else:
            tn = int(np.sum((y_pred == 0) & (y_labels == 0)))
            fp = int(np.sum((y_pred == 1) & (y_labels == 0)))
            fn = int(np.sum((y_pred == 0) & (y_labels == 1)))
            tp = int(np.sum((y_pred == 1) & (y_labels == 1)))
        
        accuracy = float(accuracy_score(y_labels, y_pred))
        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        
        try:
            roc_auc = float(roc_auc_score(y_labels, y_proba))
        except ValueError:
            roc_auc = 0.0
        
        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": roc_auc,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    
    # Train set with best threshold (using LONG model and y_train for backward compatibility)
    train_best_metrics = evaluate_with_threshold(long_model, X_train_long, y_train_long, "Train", best_threshold)
    logger.info("")
    logger.info("Train Set Evaluation (Best Threshold)")
    logger.info("=" * 60)
    logger.info(
        "Best Threshold=%.2f: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, ROC-AUC=%.4f",
        best_threshold,
        train_best_metrics["accuracy"],
        train_best_metrics["precision"],
        train_best_metrics["recall"],
        train_best_metrics["f1"],
        train_best_metrics["roc_auc"],
    )
    logger.info(
        "Confusion Matrix: TP=%d, FP=%d, TN=%d, FN=%d",
        train_best_metrics["tp"],
        train_best_metrics["fp"],
        train_best_metrics["tn"],
        train_best_metrics["fn"],
    )
    
    # Validation set with best threshold (using LONG model and y_valid for backward compatibility)
    valid_best_metrics = evaluate_with_threshold(long_model, X_valid_long, y_valid_long, "Validation", best_threshold)
    logger.info("")
    logger.info("Validation Set Evaluation (Best Threshold)")
    logger.info("=" * 60)
    logger.info(
        "Best Threshold=%.2f: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, ROC-AUC=%.4f",
        best_threshold,
        valid_best_metrics["accuracy"],
        valid_best_metrics["precision"],
        valid_best_metrics["recall"],
        valid_best_metrics["f1"],
        valid_best_metrics["roc_auc"],
    )
    logger.info(
        "Confusion Matrix: TP=%d, FP=%d, TN=%d, FN=%d",
        valid_best_metrics["tp"],
        valid_best_metrics["fp"],
        valid_best_metrics["tn"],
        valid_best_metrics["fn"],
    )
    
    # Test set with best threshold (using LONG model and y_test for backward compatibility)
    test_best_metrics = evaluate_with_threshold(long_model, X_test_long, y_test_long, "Test", best_threshold)
    logger.info("")
    logger.info("Test Set Evaluation (Best Threshold)")
    logger.info("=" * 60)
    logger.info(
        "Best Threshold=%.2f: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, ROC-AUC=%.4f",
        best_threshold,
        test_best_metrics["accuracy"],
        test_best_metrics["precision"],
        test_best_metrics["recall"],
        test_best_metrics["f1"],
        test_best_metrics["roc_auc"],
    )
    logger.info(
        "Confusion Matrix: TP=%d, FP=%d, TN=%d, FN=%d",
        test_best_metrics["tp"],
        test_best_metrics["fp"],
        test_best_metrics["tn"],
        test_best_metrics["fn"],
    )
    logger.info("=" * 60)

    # (5) Feature Importance (LONG and SHORT models)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Feature Importance")
    logger.info("=" * 60)
    
    def get_feature_importance(model, X_cols):
        """Extract feature importance from model."""
        try:
            booster = model.get_booster()
            importance_dict = booster.get_score(importance_type="gain")
            importance_list = [(feat, importance_dict.get(feat, 0.0)) for feat in X_cols]
            importance_list.sort(key=lambda x: x[1], reverse=True)
        except Exception:
            # Fallback to feature_importances_ if booster method fails
            importance_list = list(zip(X_cols, model.feature_importances_))
            importance_list.sort(key=lambda x: x[1], reverse=True)
        return importance_list
    
    # LONG model feature importance
    logger.info("")
    logger.info("LONG Model - Top 20 Features by Importance:")
    long_importance_list = get_feature_importance(long_model, X.columns)
    long_feature_importance = pd.DataFrame(
        {
            "feature": [feat[0] for feat in long_importance_list],
            "importance": [feat[1] for feat in long_importance_list],
        }
    )
    logger.info(long_feature_importance.head(20).to_string(index=False))
    
    # SHORT model feature importance
    logger.info("")
    logger.info("SHORT Model - Top 20 Features by Importance:")
    short_importance_list = get_feature_importance(short_model, X.columns)
    short_feature_importance = pd.DataFrame(
        {
            "feature": [feat[0] for feat in short_importance_list],
            "importance": [feat[1] for feat in short_importance_list],
        }
    )
    logger.info(short_feature_importance.head(20).to_string(index=False))
    logger.info("-" * 60)
    
    # Save feature importance to JSON (both models)
    research_dir = PROJECT_ROOT / "data" / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    importance_path = research_dir / f"ml_xgb_feature_importance_{feature_preset}_{timestamp}.json"
    
    importance_data = {
        "timestamp": timestamp,
        "strategy": "ml_xgb",
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_preset": feature_preset,
        "long_model": {
            "feature_importance": [
                {
                    "feature": feat[0],
                    "gain": float(feat[1]),
                    "rank": idx + 1,
                }
                for idx, feat in enumerate(long_importance_list)
            ],
        },
        "short_model": {
            "feature_importance": [
                {
                    "feature": feat[0],
                    "gain": float(feat[1]),
                    "rank": idx + 1,
                }
                for idx, feat in enumerate(short_importance_list)
            ],
        },
    }
    
    with open(importance_path, "w") as f:
        json.dump(importance_data, f, indent=2)
    
    logger.info(f"Feature importance saved to: {importance_path}")

    # (6) Save Models (LONG and SHORT)
    # Build model paths: ml_xgb_long_{symbol}_{timeframe}.pkl and ml_xgb_short_{symbol}_{timeframe}.pkl
    base_model_path = Path(settings.XGB_MODEL_PATH)
    model_dir = base_model_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract base filename (e.g., "ml_xgb_BTCUSDT_1m" from "ml_xgb_BTCUSDT_1m.pkl")
    base_stem = base_model_path.stem
    # Remove symbol and timeframe if present, or use default pattern
    if feature_preset == "base":
        long_model_path = model_dir / f"ml_xgb_long_{symbol}_{timeframe}.pkl"
        short_model_path = model_dir / f"ml_xgb_short_{symbol}_{timeframe}.pkl"
    else:
        long_model_path = model_dir / f"ml_xgb_long_{symbol}_{timeframe}_{feature_preset}.pkl"
        short_model_path = model_dir / f"ml_xgb_short_{symbol}_{timeframe}_{feature_preset}.pkl"
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Saving Models")
    logger.info("=" * 60)
    
    # Save LONG model
    logger.info(f"Saving LONG model to {long_model_path.resolve()}...")
    logger.info(f"Model preset: {feature_preset}")
    joblib.dump(long_model, long_model_path)
    logger.info("✅ LONG model saved successfully!")
    
    # Save SHORT model
    logger.info(f"Saving SHORT model to {short_model_path.resolve()}...")
    logger.info(f"Model preset: {feature_preset}")
    joblib.dump(short_model, short_model_path)
    logger.info("✅ SHORT model saved successfully!")
    logger.info("=" * 60)
    
    # Phase E: Save scaler if used
    scaler_path = None
    if scaler is not None:
        scaler_path = long_model_path.with_suffix(".scaler.pkl")
        save_scaler(scaler, scaler_path, scaler_type)
        logger.info(f"✅ Saved scaler to {scaler_path}")
    
    # (6-1) Save Model Metadata (for LONG model, including best_threshold)
    # Step A: LONG/SHORT split + proba cache patch
    # Note: best_threshold is from LONG model evaluation (validation F1 기준)
    #   - This threshold is used for LONG signal generation
    #   - SHORT model has its own optimal threshold (can be optimized separately)
    #   - For threshold optimization, use threshold_optimizer.py which tests both LONG/SHORT combinations
    # Phase E: Add scaler, label_mode, and other Phase E metadata
    meta_path = long_model_path.with_suffix(".meta.json")
    meta_data = {
        "timestamp": timestamp,
        "strategy": "ml_xgb",
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_preset": feature_preset,
        "horizon": horizon,
        "label_threshold": 0.001,  # From build_ml_dataset default (backward compatibility)
        "best_threshold": float(best_threshold),  # LONG model 기준 (validation F1)
        "best_threshold_long": float(best_threshold),  # 명시적 필드명 (Step A patch)
        "best_threshold_short": float(short_best_threshold),  # SHORT model best threshold
        "best_validation_f1": float(best_f1),
        "best_validation_f1_long": float(long_best_f1),
        "best_validation_f1_short": float(short_best_f1),
        "long_model_path": str(long_model_path),
        "short_model_path": str(short_model_path),
        "xgb_hyperparams": xgb_config.to_dict(),  # Step D: Save XGBConfig
        "feature_columns": list(X.columns),  # Step D: Save feature order for consistency
        # Phase E: New metadata fields
        "scaler_type": scaler_type,
        "scaler_path": str(scaler_path) if scaler_path else None,
        "label_mode": label_mode,
        "label_thresholds": {
            "long": float(long_threshold),
            "short": float(short_threshold),
            "hold": float(hold_threshold),
        },
        "use_regime_features": use_regime_features,
        "atr_percentile_threshold": float(atr_percentile_threshold) if use_regime_features else None,
        "long_model_roc_auc": {
            "train": float(long_auc_train),
            "valid": float(long_auc_valid),
            "test": float(long_auc_test),
        },
        "short_model_roc_auc": {
            "train": float(short_auc_train),
            "valid": float(short_auc_valid),
            "test": float(short_auc_test),
        },
        "train_metrics": {
            "roc_auc": float(train_best_metrics["roc_auc"]),
            "accuracy": float(train_best_metrics["accuracy"]),
            "precision": float(train_best_metrics["precision"]),
            "recall": float(train_best_metrics["recall"]),
            "f1": float(train_best_metrics["f1"]),
        },
        "valid_metrics": {
            "roc_auc": float(valid_best_metrics["roc_auc"]),
            "accuracy": float(valid_best_metrics["accuracy"]),
            "precision": float(valid_best_metrics["precision"]),
            "recall": float(valid_best_metrics["recall"]),
            "f1": float(valid_best_metrics["f1"]),
        },
        "test_metrics": {
            "roc_auc": float(test_best_metrics["roc_auc"]),
            "accuracy": float(test_best_metrics["accuracy"]),
            "precision": float(test_best_metrics["precision"]),
            "recall": float(test_best_metrics["recall"]),
            "f1": float(test_best_metrics["f1"]),
        },
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    
    logger.info(f"Model metadata saved to: {meta_path}")
    logger.info(f"  Best threshold: {best_threshold:.2f} (Validation F1: {best_f1:.4f})")
    logger.info(f"  LONG model: {long_model_path}")
    logger.info(f"  SHORT model: {short_model_path}")
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Pipeline Checklist:")
    logger.info("=" * 60)
    logger.info("[✓] 데이터/라벨 분포 로깅")
    logger.info("[✓] Time-series split (Train/Valid/Test)")
    logger.info("[✓] Train/Valid/Test 성능 평가 (Accuracy, Precision, Recall, F1, ROC-AUC)")
    logger.info("[✓] Confusion Matrix 로깅")
    logger.info("[✓] Threshold 스윕 분석")
    logger.info("[✓] Feature importance 분석")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Overfitting Detection Tips:")
    logger.info("  - Train accuracy >> Valid/Test accuracy → Overfitting")
    logger.info("  - Train ROC-AUC >> Valid/Test ROC-AUC → Overfitting")
    logger.info("  - Large gap in confusion matrix patterns → Overfitting")
    logger.info("  - Check threshold analysis for optimal decision boundaries")
    logger.info("=" * 60)

    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost model for BTC price prediction")
    parser.add_argument("--tune-xgb", action="store_true", help="Run hyperparameter tuning before training")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of CV folds for tuning")
    parser.add_argument("--max-trials", type=int, default=30, help="Maximum hyperparameter combinations to try")
    parser.add_argument("--metric", type=str, default="roc_auc", choices=["roc_auc", "logloss", "accuracy"],
                       help="Metric to optimize during tuning")
    parser.add_argument("--horizon", type=int, default=20, help="Number of periods ahead to predict (default: 20)")
    parser.add_argument("--long-threshold", type=float, default=0.002, help="Return threshold for LONG label (default: 0.002 = 0.2%%)")
    parser.add_argument("--short-threshold", type=float, default=0.002, help="Return threshold for SHORT label (default: 0.002 = 0.2%%)")
    parser.add_argument("--hold-threshold", type=float, default=0.0005, help="Return threshold for HOLD zone (default: 0.0005 = 0.05%%)")
    parser.add_argument("--enable-hold-labels", action="store_true", help="Enable HOLD labels and exclude them from training (default: False)")
    parser.add_argument("--use-events", action="store_true", help="Use event features (overrides settings)")
    parser.add_argument("--no-events", action="store_true", help="Disable event features (overrides settings)")
    parser.add_argument("--feature-preset", type=str, default="base", 
                       choices=["base", "extended_safe", "extended_full"],
                       help="Feature preset to use (default: base)")
    # XGBoost hyperparameter overrides (optional)
    parser.add_argument("--xgb-max-depth", type=int, default=None, help="XGBoost max_depth override (default: 3)")
    parser.add_argument("--xgb-n-estimators", type=int, default=None, help="XGBoost n_estimators override (default: 400)")
    parser.add_argument("--xgb-learning-rate", type=float, default=None, help="XGBoost learning_rate override (default: 0.05)")
    parser.add_argument("--xgb-subsample", type=float, default=None, help="XGBoost subsample override (default: 0.7)")
    parser.add_argument("--xgb-colsample-bytree", type=float, default=None, help="XGBoost colsample_bytree override (default: 0.7)")
    # Phase E: New CLI arguments
    parser.add_argument("--scaler-type", type=str, default="none", choices=["standard", "robust", "minmax", "none"],
                       help="Feature scaler type (default: none)")
    parser.add_argument("--label-mode", type=str, default="classification", choices=["classification", "regression"],
                       help="Label mode: classification or regression (default: classification)")
    parser.add_argument("--timeframe", type=str, default=None, choices=["1m", "3m", "5m", "15m", "30m"],
                       help="Target timeframe (default: from settings)")
    parser.add_argument("--use-regime", action="store_true", help="Enable volatility regime features")
    parser.add_argument("--atr-percentile", type=float, default=0.70, help="ATR percentile threshold for regime (default: 0.70)")
    
    args = parser.parse_args()
    
    use_events = None
    if args.use_events:
        use_events = True
    elif args.no_events:
        use_events = False
    
    # Create XGBConfig with CLI overrides if provided
    xgb_config = XGBConfig()
    if args.xgb_max_depth is not None:
        xgb_config.max_depth = args.xgb_max_depth
    if args.xgb_n_estimators is not None:
        xgb_config.n_estimators = args.xgb_n_estimators
    if args.xgb_learning_rate is not None:
        xgb_config.learning_rate = args.xgb_learning_rate
    if args.xgb_subsample is not None:
        xgb_config.subsample = args.xgb_subsample
    if args.xgb_colsample_bytree is not None:
        xgb_config.colsample_bytree = args.xgb_colsample_bytree
    
    if args.tune_xgb:
        logger.info("=" * 60)
        logger.info("Starting hyperparameter tuning...")
        logger.info("=" * 60)
        
        from src.ml.xgb_tuning import tune_xgb_hyperparameters
        
        tuning_result = tune_xgb_hyperparameters(
            horizon=args.horizon,
            n_folds=args.cv_folds,
            metric=args.metric,
            use_events=use_events,
            max_combinations=args.max_trials,
            random_sample=True,
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training final model with best hyperparameters...")
        logger.info("=" * 60)
        
        # Create XGBConfig from tuning result
        tuned_config = XGBConfig(
            n_estimators=tuning_result.best_config.n_estimators,
            max_depth=tuning_result.best_config.max_depth,
            learning_rate=tuning_result.best_config.learning_rate,
            subsample=tuning_result.best_config.subsample,
            colsample_bytree=tuning_result.best_config.colsample_bytree,
            min_child_weight=tuning_result.best_config.min_child_weight,
            gamma=tuning_result.best_config.gamma,
        )
        # Train final model with best config
        model = train_xgb_model(
            horizon=args.horizon,
            xgb_config=tuned_config,
            use_events=use_events,
            feature_preset=args.feature_preset,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            hold_threshold=args.hold_threshold,
            enable_hold_labels=args.enable_hold_labels,
            scaler_type=args.scaler_type,
            label_mode=args.label_mode,
            timeframe=args.timeframe,
            use_regime_features=args.use_regime,
            atr_percentile_threshold=args.atr_percentile,
        )
    else:
        # Standard training without tuning
        model = train_xgb_model(
            horizon=args.horizon,
            xgb_config=xgb_config,
            use_events=use_events,
            feature_preset=args.feature_preset,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            hold_threshold=args.hold_threshold,
            enable_hold_labels=args.enable_hold_labels,
            scaler_type=args.scaler_type,
            label_mode=args.label_mode,
            timeframe=args.timeframe,
            use_regime_features=args.use_regime,
            atr_percentile_threshold=args.atr_percentile,
        )

