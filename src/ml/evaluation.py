"""
Evaluation utilities for ML models (XGBoost, etc.).

This module provides functions for evaluating ML models with detailed metrics
and threshold analysis, similar to the LSTM training pipeline.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_ml_split(
    model: Any,
    X: Any,
    y: np.ndarray | Any,
    split_name: str,
    thresholds: list[float] | None = None,
    logger_instance: logging.Logger | None = None,
    return_metrics: bool = False,
) -> dict[str, Any] | None:
    """
    Evaluate ML model on a split (train/valid/test) and log metrics.
    
    Similar to LSTM's evaluate_split but adapted for sklearn-style models.
    
    Args:
        model: Trained model with predict_proba method
        X: Input features (DataFrame or array)
        y: True labels (array or Series)
        split_name: Name of the split (e.g., "Train", "Validation", "Test")
        thresholds: List of probability thresholds to test (default: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
        logger_instance: Logger instance (if None, uses module logger)
        return_metrics: If True, return dict with metrics for each threshold
    
    Returns:
        If return_metrics=True, returns dict with keys:
        - "threshold_metrics": List of dicts with keys: threshold, precision, recall, f1, tp, fp, fn
        - "default_metrics": Dict with default threshold=0.5 metrics
        - "roc_auc": ROC-AUC score
        Otherwise returns None
    """
    if logger_instance is None:
        logger_instance = logger
    
    # Check for empty X
    if X is None or len(X) == 0:
        logger_instance.warning("[ML Eval] Split '%s' has empty X. Skipping evaluation.", split_name)
        return None
    
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    # Get predictions with defensive checks
    try:
        proba = model.predict_proba(X)
    except Exception as e:
        logger_instance.error(
            "[ML Eval] predict_proba failed for split '%s': %s. X shape=%s",
            split_name, str(e), X.shape if hasattr(X, 'shape') else 'unknown'
        )
        raise
    
    # Check predict_proba shape
    if proba.ndim != 2 or proba.shape[1] < 2:
        logger_instance.error(
            "[ML Eval] Unexpected predict_proba shape for split '%s': %s (expected (n_samples, 2)).",
            split_name, proba.shape
        )
        raise ValueError(
            f"Unexpected predict_proba shape: {proba.shape} (expected (n_samples, 2)). "
            f"Split: {split_name}, X shape: {X.shape if hasattr(X, 'shape') else 'unknown'}"
        )
    
    y_proba = proba[:, 1]  # Probability of positive class
    y_labels = np.array(y) if not isinstance(y, np.ndarray) else y
    
    # Calculate metrics at default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    cm_default = confusion_matrix(y_labels, y_pred_default)
    
    if cm_default.size == 4:
        tn_default, fp_default, fn_default, tp_default = cm_default.ravel()
    elif cm_default.size == 1:
        # Edge case: all predictions are the same
        if y_pred_default[0] == 0:
            tn_default, fp_default, fn_default, tp_default = len(y_labels), 0, 0, 0
        else:
            tn_default, fp_default, fn_default, tp_default = 0, 0, 0, len(y_labels)
    else:
        # Fallback for unusual cases
        tn_default = int(np.sum((y_pred_default == 0) & (y_labels == 0)))
        fp_default = int(np.sum((y_pred_default == 1) & (y_labels == 0)))
        fn_default = int(np.sum((y_pred_default == 0) & (y_labels == 1)))
        tp_default = int(np.sum((y_pred_default == 1) & (y_labels == 1)))
    
    accuracy = accuracy_score(y_labels, y_pred_default)
    precision_default = precision_score(y_labels, y_pred_default, zero_division=0.0)
    recall_default = recall_score(y_labels, y_pred_default, zero_division=0.0)
    f1_default = f1_score(y_labels, y_pred_default, zero_division=0.0)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_labels, y_proba)
    except ValueError:
        # Edge case: only one class present
        roc_auc = 0.0
        logger_instance.warning(f"  ROC-AUC calculation failed (only one class present), setting to 0.0")
    
    logger_instance.info("=" * 60)
    logger_instance.info(f"{split_name} Set Evaluation")
    logger_instance.info("=" * 60)
    logger_instance.info(
        f"Threshold=0.5: Accuracy={accuracy:.4f}, Precision={precision_default:.4f}, "
        f"Recall={recall_default:.4f}, F1={f1_default:.4f}, ROC-AUC={roc_auc:.4f}"
    )
    logger_instance.info(
        f"Confusion Matrix: TP={tp_default}, FP={fp_default}, TN={tn_default}, FN={fn_default}"
    )
    logger_instance.info("-" * 60)
    logger_instance.info(f"{split_name} Threshold Analysis (various thresholds for precision/recall):")
    logger_instance.info("-" * 60)
    logger_instance.info(
        f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8}"
    )
    logger_instance.info("-" * 60)
    
    # Store metrics for each threshold if return_metrics is True
    threshold_metrics_list = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
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
        
        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        
        logger_instance.info(
            f"{thresh:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} "
            f"{tp:<8} {fp:<8} {fn:<8}"
        )
        
        if return_metrics:
            threshold_metrics_list.append({
                "threshold": thresh,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            })
    
    logger_instance.info("-" * 60)
    
    if return_metrics:
        return {
            "threshold_metrics": threshold_metrics_list,
            "default_metrics": {
                "threshold": 0.5,
                "accuracy": float(accuracy),
                "precision": float(precision_default),
                "recall": float(recall_default),
                "f1": float(f1_default),
                "roc_auc": float(roc_auc),
                "tp": int(tp_default),
                "fp": int(fp_default),
                "fn": int(fn_default),
                "tn": int(tn_default),
            },
            "roc_auc": float(roc_auc),
        }
    return None


def log_label_distribution(
    y: np.ndarray | Any,
    dataset_name: str = "Dataset",
    logger_instance: logging.Logger | None = None,
) -> None:
    """
    Log label distribution statistics.
    
    Args:
        y: Labels (array or Series)
        dataset_name: Name of the dataset (e.g., "Full Dataset", "Train")
        logger_instance: Logger instance (if None, uses module logger)
    """
    if logger_instance is None:
        logger_instance = logger
    
    y_arr = np.array(y) if not isinstance(y, np.ndarray) else y
    n_total = len(y_arr)
    pos_count = int(np.sum(y_arr == 1))
    neg_count = int(np.sum(y_arr == 0))
    pos_ratio = pos_count / n_total if n_total > 0 else 0.0
    neg_ratio = neg_count / n_total if n_total > 0 else 0.0
    
    logger_instance.info(f"{dataset_name} Label Distribution:")
    logger_instance.info(f"  Total samples: {n_total}")
    logger_instance.info(f"  Positive (1): {pos_count} ({pos_ratio:.4f} = {pos_ratio*100:.2f}%)")
    logger_instance.info(f"  Negative (0): {neg_count} ({neg_ratio:.4f} = {neg_ratio*100:.2f}%)")
    logger_instance.info("-" * 60)

