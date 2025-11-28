"""
Time-series data splitting utilities.

This module provides functions for splitting time-series data into train/valid/test
sets while preserving temporal order, and logging split summaries.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import logging


def make_time_series_splits(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    min_test_samples: int = 200,
    meta: Dict[str, np.ndarray] | None = None,
) -> Dict[str, Any]:
    """
    Split time-series data into train/valid/test sets while preserving temporal order.
    
    This function splits the data sequentially from the beginning, ensuring that
    train, valid, and test sets maintain the temporal order of the original data.
    The test set is guaranteed to have at least min_test_samples samples by
    potentially adjusting the valid/test boundary.
    
    The function returns both indices and actual data slices, allowing other models
    (e.g., XGBoost) to reuse the same split indices for consistency.
    
    Args:
        X: Input sequences of shape (N, window_size, feature_dim)
        y: Labels of shape (N,)
        train_ratio: Proportion of data for training (default: 0.7)
        valid_ratio: Proportion of data for validation (default: 0.15)
        min_test_samples: Minimum number of samples in test set (default: 200)
        meta: Optional dictionary with additional arrays to split (e.g., {"future_returns": np.ndarray})
    
    Returns:
        Dictionary with keys:
        - "idx": Dictionary containing train_idx, valid_idx, test_idx (numpy arrays)
        - "data": Dictionary containing X_train, y_train, X_valid, y_valid, X_test, y_test
        - "meta": Dictionary containing split meta arrays (if meta was provided)
    
    Raises:
        AssertionError: If X and y have incompatible shapes
        ValueError: If data is too short to create valid splits
    """
    # Basic validation
    assert len(X) == len(y), f"X and y must have the same length: {len(X)} != {len(y)}"
    assert X.ndim == 3, f"X must be 3D (N, window, feature_dim), got shape {X.shape}"
    assert y.ndim == 1, f"y must be 1D (N,), got shape {y.shape}"
    
    N = len(X)
    
    # Calculate split boundaries
    train_end = int(N * train_ratio)
    valid_end = int(N * (train_ratio + valid_ratio))
    
    # Check minimum test samples and adjust if needed
    test_len = N - valid_end
    if test_len < min_test_samples:
        needed = min_test_samples - test_len
        # Move valid_end backward to give more samples to test
        valid_end = max(train_end + 1, valid_end - needed)
        test_len = N - valid_end
        
        # Ensure valid set has at least 1 sample
        if valid_end <= train_end:
            raise ValueError(
                f"Cannot create valid splits with given constraints. "
                f"N={N}, train_ratio={train_ratio}, valid_ratio={valid_ratio}, "
                f"min_test_samples={min_test_samples}. "
                f"Result: train_end={train_end}, valid_end={valid_end}, "
                f"test_len={test_len}. Valid set would be empty or negative."
            )
    
    # Calculate indices
    train_idx = np.arange(0, train_end)
    valid_idx = np.arange(train_end, valid_end)
    test_idx = np.arange(valid_end, N)
    
    # Slice data
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    result = {
        "idx": {
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
        },
        "data": {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test,
        },
    }
    
    # Split meta if provided
    if meta is not None:
        meta_split = {}
        for key, arr in meta.items():
            assert len(arr) == N, f"meta['{key}'] length {len(arr)} != X length {N}"
            meta_split[key] = {
                "train": arr[train_idx],
                "valid": arr[valid_idx],
                "test": arr[test_idx],
            }
        result["meta"] = meta_split
    
    return result


def log_split_summary(
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    logger: Any = None,
    label_positive: int = 1,
    label_negative: int = 0,
) -> None:
    """
    Log summary statistics for train/valid/test splits.
    
    This function logs the number of samples and label distribution (positive/negative
    ratio) for each split.
    
    Args:
        y_train: Training labels
        y_valid: Validation labels
        y_test: Test labels
        logger: Logger instance (if None, uses default logger)
        label_positive: Label value for positive class (default: 1)
        label_negative: Label value for negative class (default: 0)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Calculate statistics for each split
    def calc_stats(y: np.ndarray) -> tuple[int, int, int, float, float]:
        n = len(y)
        pos = int(np.sum(y == label_positive))
        neg = int(np.sum(y == label_negative))
        total = pos + neg
        ratio_pos = pos / total if total > 0 else 0.0
        ratio_neg = neg / total if total > 0 else 0.0
        return n, pos, neg, ratio_pos, ratio_neg
    
    n_train, pos_train, neg_train, ratio_train_pos, ratio_train_neg = calc_stats(y_train)
    n_valid, pos_valid, neg_valid, ratio_valid_pos, ratio_valid_neg = calc_stats(y_valid)
    n_test, pos_test, neg_test, ratio_test_pos, ratio_test_neg = calc_stats(y_test)
    
    # Log summary
    logger.info("-" * 60)
    logger.info("Time-series Split Summary:")
    logger.info("-" * 60)
    logger.info(
        f"Train: n={n_train}, pos={pos_train} (p={ratio_train_pos:.3f}), "
        f"neg={neg_train} (p={ratio_train_neg:.3f})"
    )
    logger.info(
        f"Valid: n={n_valid}, pos={pos_valid} (p={ratio_valid_pos:.3f}), "
        f"neg={neg_valid} (p={ratio_valid_neg:.3f})"
    )
    logger.info(
        f"Test : n={n_test}, pos={pos_test} (p={ratio_test_pos:.3f}), "
        f"neg={neg_test} (p={ratio_test_neg:.3f})"
    )
    logger.info("-" * 60)

