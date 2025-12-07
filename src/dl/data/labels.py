"""
3-class label generation utilities for LSTM training.

This module provides functions to create 3-class labels (FLAT/LONG/SHORT)
from future returns for LSTM classification.
"""
from __future__ import annotations

import enum
from typing import Literal

import numpy as np


class LstmClassIndex(enum.IntEnum):
    """
    Class index mapping for 3-class LSTM classification.
    
    - FLAT = 0: Ambiguous/neutral zone (no clear direction)
    - LONG = 1: Strong upward movement expected
    - SHORT = 2: Strong downward movement expected
    """
    FLAT = 0
    LONG = 1
    SHORT = 2


def create_3class_labels(
    returns: np.ndarray,
    pos_threshold: float = 0.001,
    neg_threshold: float = 0.001,
) -> np.ndarray:
    """
    Create 3-class labels from future returns.
    
    Label definition:
    - r > +pos_threshold → LONG (class 1)
    - r < -neg_threshold → SHORT (class 2)
    - Otherwise → FLAT (class 0)
    
    Args:
        returns: Array of future returns (shape: (N,))
        pos_threshold: Positive return threshold for LONG class (default: 0.001 = 0.1%)
        neg_threshold: Negative return threshold for SHORT class (default: 0.001 = 0.1%)
    
    Returns:
        Array of integer labels (shape: (N,), dtype: np.int64)
        Values are in {0: FLAT, 1: LONG, 2: SHORT}
    """
    labels = np.zeros_like(returns, dtype=np.int64)
    
    # LONG: returns > pos_threshold
    labels[returns > pos_threshold] = LstmClassIndex.LONG
    
    # SHORT: returns < -neg_threshold
    labels[returns < -neg_threshold] = LstmClassIndex.SHORT
    
    # FLAT: everything else (already 0)
    
    return labels


def get_class_name(class_idx: int) -> Literal["FLAT", "LONG", "SHORT"]:
    """
    Get class name from class index.
    
    Args:
        class_idx: Class index (0, 1, or 2)
    
    Returns:
        Class name: "FLAT", "LONG", or "SHORT"
    """
    if class_idx == LstmClassIndex.FLAT:
        return "FLAT"
    elif class_idx == LstmClassIndex.LONG:
        return "LONG"
    elif class_idx == LstmClassIndex.SHORT:
        return "SHORT"
    else:
        raise ValueError(f"Invalid class index: {class_idx}")

