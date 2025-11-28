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
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.core.config import settings
from src.ml.evaluation import evaluate_ml_split, log_label_distribution
from src.ml.features import build_ml_dataset
from src.services.ohlcv_service import load_ohlcv_df

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    horizon: int = 5,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    use_events: bool | None = None,
) -> XGBClassifier:
    """
    Train XGBoost classifier for BTC price direction prediction with comprehensive logging.

    Args:
        horizon: Number of periods ahead to predict
        train_ratio: Proportion of data for training (default: 0.7)
        valid_ratio: Proportion of data for validation (default: 0.15)
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        subsample: Subsample ratio of training instances
        colsample_bytree: Subsample ratio of columns when constructing each tree
        min_child_weight: Minimum sum of instance weight needed in a child
        gamma: Minimum loss reduction required to make a further partition
        use_events: Whether to use event features (default: from settings)

    Returns:
        Trained XGBClassifier model
    """
    logger.info("=" * 60)
    logger.info("XGBoost Model Training")
    logger.info("=" * 60)
    
    # (1) Data & Label Logging
    logger.info("Loading OHLCV data...")
    df = load_ohlcv_df()
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of OHLCV data")

    if use_events is None:
        use_events = settings.EVENTS_ENABLED
    
    logger.info("=" * 60)
    logger.info("Building ML Dataset")
    logger.info("=" * 60)
    logger.info(f"Horizon: {horizon}, Use events: {use_events}")
    
    # Extract symbol and timeframe from settings or use defaults
    symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    
    X, y = build_ml_dataset(
        df,
        horizon=horizon,
        symbol=symbol,
        timeframe=timeframe,
        use_events=use_events,
    )

    # Align X and y lengths (use X.shape[0] as the reference)
    X_len = X.shape[0]
    y_len = len(y)
    
    if X_len != y_len:
        logger.warning(
            f"X and y length mismatch: X.shape[0]={X_len}, len(y)={y_len}. "
            f"Truncating y to match X length."
        )
        y = y.iloc[:X_len] if hasattr(y, 'iloc') else y[:X_len]
        logger.info(f"After alignment: X.shape[0]={X.shape[0]}, len(y)={len(y)}")
    else:
        logger.info(f"X and y lengths match: {X_len}")

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Feature count: {len(X.columns)}")
    if use_events:
        event_cols = [c for c in X.columns if c.startswith("event_")]
        logger.info(f"Event features: {len(event_cols)} (sample: {event_cols[:3] if event_cols else []})")
    
    # Feature columns logging
    logger.info("=" * 60)
    logger.info("[XGB Train] Feature Columns")
    logger.info("=" * 60)
    logger.info(f"[XGB Train] Using {len(X.columns)} feature columns:")
    logger.info(f"[XGB Train] FEATURE_COLS = {list(X.columns)}")
    if use_events:
        event_cols = [c for c in X.columns if c.startswith("event_")]
        logger.info(f"[XGB Train] Event features ({len(event_cols)}): {event_cols}")
        basic_cols = [c for c in X.columns if not c.startswith("event_")]
        logger.info(f"[XGB Train] Basic features ({len(basic_cols)}): {basic_cols}")
    logger.info("=" * 60)
    
    # Label distribution logging
    log_label_distribution(y, "Full Dataset", logger)

    # (2) Time-series Split
    logger.info("=" * 60)
    logger.info("Time-series Split")
    logger.info("=" * 60)
    logger.info(f"Train ratio: {train_ratio}, Valid ratio: {valid_ratio}, Test ratio: {1.0 - train_ratio - valid_ratio:.3f}")
    
    splits = make_time_series_split_3way(
        X, y,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        min_test_samples=200,
    )
    
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
    
    # Label distribution for each split
    log_label_distribution(y_train, "Train", logger)
    log_label_distribution(y_valid, "Valid", logger)
    log_label_distribution(y_test, "Test", logger)

    # (3) Model Training
    logger.info("=" * 60)
    logger.info("Model Training")
    logger.info("=" * 60)
    logger.info(f"Hyperparameters:")
    logger.info(f"  n_estimators: {n_estimators}")
    logger.info(f"  max_depth: {max_depth}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  subsample: {subsample}")
    logger.info(f"  colsample_bytree: {colsample_bytree}")
    logger.info(f"  min_child_weight: {min_child_weight}")
    logger.info(f"  gamma: {gamma}")
    logger.info("-" * 60)
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    logger.info("Training completed!")

    # (4) Performance Evaluation
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    
    # Train set evaluation
    evaluate_ml_split(
        model,
        X_train,
        y_train,
        split_name="Train",
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        logger_instance=logger,
    )
    
    # Validation set evaluation
    evaluate_ml_split(
        model,
        X_valid,
        y_valid,
        split_name="Validation",
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        logger_instance=logger,
    )
    
    # Test set evaluation
    evaluate_ml_split(
        model,
        X_test,
        y_test,
        split_name="Test",
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        logger_instance=logger,
    )

    # (5) Feature Importance
    logger.info("")
    logger.info("=" * 60)
    logger.info("Feature Importance")
    logger.info("=" * 60)
    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    logger.info("\nTop 20 Features by Importance:")
    logger.info(feature_importance.head(20).to_string(index=False))
    logger.info("-" * 60)

    # (6) Save Model
    model_path = Path(settings.XGB_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Saving model to {model_path.resolve()}...")
    joblib.dump(model, model_path)
    logger.info("✅ Model saved successfully!")
    logger.info("=" * 60)
    
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
    parser.add_argument("--horizon", type=int, default=5, help="Number of periods ahead to predict")
    parser.add_argument("--use-events", action="store_true", help="Use event features (overrides settings)")
    parser.add_argument("--no-events", action="store_true", help="Disable event features (overrides settings)")
    
    args = parser.parse_args()
    
    use_events = None
    if args.use_events:
        use_events = True
    elif args.no_events:
        use_events = False
    
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
        
        # Train final model with best config
        model = train_xgb_model(
            horizon=args.horizon,
            n_estimators=tuning_result.best_config.n_estimators,
            max_depth=tuning_result.best_config.max_depth,
            learning_rate=tuning_result.best_config.learning_rate,
            subsample=tuning_result.best_config.subsample,
            colsample_bytree=tuning_result.best_config.colsample_bytree,
            min_child_weight=tuning_result.best_config.min_child_weight,
            gamma=tuning_result.best_config.gamma,
            use_events=use_events,
        )
    else:
        # Standard training without tuning
        model = train_xgb_model(
            horizon=args.horizon,
            use_events=use_events,
        )

