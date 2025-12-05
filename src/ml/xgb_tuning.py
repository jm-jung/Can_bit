"""
Time-series Cross-Validation based hyperparameter tuning for XGBoost.

This module provides functionality to tune XGBoost hyperparameters using
time-series cross-validation to find parameters that perform consistently
across multiple time periods, reducing overfitting risk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.core.config import settings
from src.ml.features import build_ml_dataset
from src.services.ohlcv_service import load_ohlcv_df

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for XGBoost."""
    
    max_depth: int = 4
    n_estimators: int = 300
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XGBoost parameters."""
        return {
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
        }


@dataclass
class CVFoldResult:
    """Result for a single CV fold."""
    
    fold_idx: int
    train_size: int
    valid_size: int
    metric_value: float
    config: HyperparameterConfig


@dataclass
class HyperparameterTuningResult:
    """Result of hyperparameter tuning."""
    
    best_config: HyperparameterConfig
    best_mean_metric: float
    best_std_metric: float
    all_results: List[CVFoldResult]
    fold_summary: Dict[str, Any]


def create_time_series_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 3,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Create time-series cross-validation folds (roll-forward style).
    
    Args:
        X: Feature DataFrame
        y: Label Series
        n_folds: Number of CV folds
        train_ratio: Proportion of data for training in each fold
        valid_ratio: Proportion of data for validation in each fold
    
    Returns:
        List of fold dictionaries with X_train, y_train, X_valid, y_valid
    """
    N = len(X)
    folds = []
    
    # Calculate fold sizes
    total_ratio = train_ratio + valid_ratio
    if total_ratio >= 1.0:
        raise ValueError(f"train_ratio + valid_ratio ({total_ratio}) must be < 1.0")
    
    # For roll-forward CV:
    # Fold 1: train=0:train_end_1, valid=train_end_1:valid_end_1
    # Fold 2: train=0:train_end_2, valid=train_end_2:valid_end_2
    # ... where train_end_i and valid_end_i grow with each fold
    
    base_train_size = int(N * train_ratio)
    base_valid_size = int(N * valid_ratio)
    
    # Calculate increment per fold
    remaining_size = N - base_train_size - base_valid_size
    if n_folds > 1:
        train_increment = remaining_size // (n_folds * 2)  # Conservative increment
        valid_increment = remaining_size // (n_folds * 2)
    else:
        train_increment = 0
        valid_increment = 0
    
    for fold_idx in range(n_folds):
        train_end = base_train_size + (fold_idx * train_increment)
        valid_end = train_end + base_valid_size + (fold_idx * valid_increment)
        
        # Ensure valid_end doesn't exceed data length
        valid_end = min(valid_end, N)
        
        if train_end >= valid_end or valid_end > N:
            logger.warning(
                f"Fold {fold_idx + 1}: Skipping (train_end={train_end}, valid_end={valid_end}, N={N})"
            )
            continue
        
        X_train = X.iloc[:train_end].copy()
        y_train = y.iloc[:train_end].copy()
        X_valid = X.iloc[train_end:valid_end].copy()
        y_valid = y.iloc[train_end:valid_end].copy()
        
        folds.append({
            "fold_idx": fold_idx + 1,
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "train_size": len(X_train),
            "valid_size": len(X_valid),
        })
    
    return folds


def evaluate_config_on_fold(
    config: HyperparameterConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    fold_idx: int,
    metric: str = "roc_auc",
) -> CVFoldResult:
    """
    Train XGBoost with given config on a fold and evaluate.
    
    Args:
        config: Hyperparameter configuration
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        fold_idx: Fold index (for logging)
        metric: Metric to evaluate ("roc_auc", "logloss", "accuracy")
    
    Returns:
        CVFoldResult with metric value
    """
    model = XGBClassifier(
        **config.to_dict(),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    
    # Calculate metric
    if metric == "roc_auc":
        from sklearn.metrics import roc_auc_score
        y_proba = model.predict_proba(X_valid)[:, 1]
        metric_value = roc_auc_score(y_valid, y_proba)
    elif metric == "logloss":
        from sklearn.metrics import log_loss
        y_proba = model.predict_proba(X_valid)
        metric_value = -log_loss(y_valid, y_proba)  # Negative because higher is better
    elif metric == "accuracy":
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_valid)
        metric_value = accuracy_score(y_valid, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return CVFoldResult(
        fold_idx=fold_idx,
        train_size=len(X_train),
        valid_size=len(X_valid),
        metric_value=metric_value,
        config=config,
    )


def generate_hyperparameter_grid(
    max_depth_range: List[int] = [3, 4, 5, 6],
    n_estimators_range: List[int] = [200, 400, 600],
    learning_rate_range: List[float] = [0.01, 0.02, 0.05, 0.1],
    subsample_range: List[float] = [0.6, 0.8, 1.0],
    colsample_bytree_range: List[float] = [0.6, 0.8, 1.0],
    min_child_weight_range: List[int] = [1, 3, 5],
    gamma_range: List[float] = [0, 0.1, 0.3],
    max_combinations: Optional[int] = None,
    random_sample: bool = True,
) -> List[HyperparameterConfig]:
    """
    Generate hyperparameter grid for tuning.
    
    Args:
        max_depth_range: Range of max_depth values
        n_estimators_range: Range of n_estimators values
        learning_rate_range: Range of learning_rate values
        subsample_range: Range of subsample values
        colsample_bytree_range: Range of colsample_bytree values
        min_child_weight_range: Range of min_child_weight values
        gamma_range: Range of gamma values
        max_combinations: Maximum number of combinations to generate (None = all)
        random_sample: If True and max_combinations is set, randomly sample combinations
    
    Returns:
        List of HyperparameterConfig objects
    """
    import itertools
    
    all_combinations = list(itertools.product(
        max_depth_range,
        n_estimators_range,
        learning_rate_range,
        subsample_range,
        colsample_bytree_range,
        min_child_weight_range,
        gamma_range,
    ))
    
    if max_combinations is not None and len(all_combinations) > max_combinations:
        if random_sample:
            np.random.seed(42)
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        else:
            all_combinations = all_combinations[:max_combinations]
    
    configs = []
    for combo in all_combinations:
        configs.append(HyperparameterConfig(
            max_depth=combo[0],
            n_estimators=combo[1],
            learning_rate=combo[2],
            subsample=combo[3],
            colsample_bytree=combo[4],
            min_child_weight=combo[5],
            gamma=combo[6],
        ))
    
    return configs


def tune_xgb_hyperparameters(
    horizon: int = 30,
    n_folds: int = 3,
    metric: str = "roc_auc",
    use_events: bool | None = None,
    max_combinations: Optional[int] = 30,
    random_sample: bool = True,
    hyperparameter_ranges: Optional[Dict[str, List[Any]]] = None,
) -> HyperparameterTuningResult:
    """
    Tune XGBoost hyperparameters using time-series cross-validation.
    
    Args:
        horizon: Number of periods ahead to predict (default: 30)
        n_folds: Number of CV folds
        metric: Metric to optimize ("roc_auc", "logloss", "accuracy")
        use_events: Whether to use event features (default: from settings)
        max_combinations: Maximum number of hyperparameter combinations to try
        random_sample: If True, randomly sample combinations when max_combinations is set
        hyperparameter_ranges: Optional custom ranges for hyperparameters
    
    Returns:
        HyperparameterTuningResult with best config and all fold results
    """
    logger.info("=" * 60)
    logger.info("XGBoost Hyperparameter Tuning (Time-series CV)")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading OHLCV data...")
    df = load_ohlcv_df()
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of OHLCV data")
    
    if use_events is None:
        use_events = settings.EVENTS_ENABLED
    
    # Build dataset
    logger.info("Building ML dataset...")
    # Extract symbol and timeframe from settings
    symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    timeframe = getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    
    X, y = build_ml_dataset(
        df,
        horizon=horizon,
        symbol=symbol,
        timeframe=timeframe,
        use_events=use_events,
    )
    
    # Align X and y lengths
    X_len = X.shape[0]
    y_len = len(y)
    if X_len != y_len:
        y = y.iloc[:X_len] if hasattr(y, 'iloc') else y[:X_len]
    
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Feature count: {len(X.columns)}")
    
    # Create CV folds
    logger.info(f"Creating {n_folds} time-series CV folds...")
    folds = create_time_series_folds(X, y, n_folds=n_folds)
    logger.info(f"Created {len(folds)} folds")
    for fold in folds:
        logger.info(
            f"  Fold {fold['fold_idx']}: train={fold['train_size']}, valid={fold['valid_size']}"
        )
    
    # Generate hyperparameter grid
    if hyperparameter_ranges is None:
        configs = generate_hyperparameter_grid(max_combinations=max_combinations, random_sample=random_sample)
    else:
        configs = generate_hyperparameter_grid(
            max_depth_range=hyperparameter_ranges.get("max_depth", [3, 4, 5, 6]),
            n_estimators_range=hyperparameter_ranges.get("n_estimators", [200, 400, 600]),
            learning_rate_range=hyperparameter_ranges.get("learning_rate", [0.01, 0.02, 0.05, 0.1]),
            subsample_range=hyperparameter_ranges.get("subsample", [0.6, 0.8, 1.0]),
            colsample_bytree_range=hyperparameter_ranges.get("colsample_bytree", [0.6, 0.8, 1.0]),
            min_child_weight_range=hyperparameter_ranges.get("min_child_weight", [1, 3, 5]),
            gamma_range=hyperparameter_ranges.get("gamma", [0, 0.1, 0.3]),
            max_combinations=max_combinations,
            random_sample=random_sample,
        )
    
    logger.info(f"Testing {len(configs)} hyperparameter combinations...")
    logger.info("=" * 60)
    
    # Evaluate each config on all folds
    all_results: List[CVFoldResult] = []
    config_scores: Dict[str, List[float]] = {}
    
    for config_idx, config in enumerate(configs):
        config_key = f"{config.max_depth}_{config.n_estimators}_{config.learning_rate:.3f}"
        fold_metrics = []
        
        logger.info(
            f"[{config_idx + 1}/{len(configs)}] Testing config: "
            f"max_depth={config.max_depth}, n_estimators={config.n_estimators}, "
            f"learning_rate={config.learning_rate:.3f}, subsample={config.subsample:.2f}, "
            f"colsample_bytree={config.colsample_bytree:.2f}, "
            f"min_child_weight={config.min_child_weight}, gamma={config.gamma:.2f}"
        )
        
        for fold in folds:
            try:
                result = evaluate_config_on_fold(
                    config=config,
                    X_train=fold["X_train"],
                    y_train=fold["y_train"],
                    X_valid=fold["X_valid"],
                    y_valid=fold["y_valid"],
                    fold_idx=fold["fold_idx"],
                    metric=metric,
                )
                all_results.append(result)
                fold_metrics.append(result.metric_value)
                logger.debug(
                    f"  Fold {fold['fold_idx']}: {metric}={result.metric_value:.6f}"
                )
            except Exception as e:
                logger.warning(f"  Fold {fold['fold_idx']} failed: {e}")
                continue
        
        if fold_metrics:
            config_scores[config_key] = fold_metrics
            mean_metric = np.mean(fold_metrics)
            std_metric = np.std(fold_metrics)
            logger.info(
                f"  Mean {metric}: {mean_metric:.6f}, Std: {std_metric:.6f}"
            )
    
    # Find best config (highest mean, lowest std)
    if not config_scores:
        raise ValueError("No successful evaluations. Check data and hyperparameter ranges.")
    
    best_config_key = None
    best_mean = float("-inf")
    best_std = float("inf")
    
    for config_key, metrics in config_scores.items():
        mean_metric = np.mean(metrics)
        std_metric = np.std(metrics)
        
        # Prefer higher mean, but if similar means, prefer lower std
        if mean_metric > best_mean or (abs(mean_metric - best_mean) < 1e-6 and std_metric < best_std):
            best_mean = mean_metric
            best_std = std_metric
            best_config_key = config_key
    
    # Find the config object
    best_config = None
    for config in configs:
        config_key = f"{config.max_depth}_{config.n_estimators}_{config.learning_rate:.3f}"
        if config_key == best_config_key:
            best_config = config
            break
    
    if best_config is None:
        raise ValueError("Could not find best config. This should not happen.")
    
    # Create fold summary
    fold_summary = {
        "n_folds": len(folds),
        "metric": metric,
        "best_config_key": best_config_key,
        "config_scores": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} 
                         for k, v in config_scores.items()},
    }
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Tuning Complete")
    logger.info("=" * 60)
    logger.info(f"Best config: {best_config_key}")
    logger.info(f"  max_depth: {best_config.max_depth}")
    logger.info(f"  n_estimators: {best_config.n_estimators}")
    logger.info(f"  learning_rate: {best_config.learning_rate}")
    logger.info(f"  subsample: {best_config.subsample}")
    logger.info(f"  colsample_bytree: {best_config.colsample_bytree}")
    logger.info(f"  min_child_weight: {best_config.min_child_weight}")
    logger.info(f"  gamma: {best_config.gamma}")
    logger.info(f"Mean {metric}: {best_mean:.6f}")
    logger.info(f"Std {metric}: {best_std:.6f}")
    logger.info("=" * 60)
    
    return HyperparameterTuningResult(
        best_config=best_config,
        best_mean_metric=best_mean,
        best_std_metric=best_std,
        all_results=all_results,
        fold_summary=fold_summary,
    )

