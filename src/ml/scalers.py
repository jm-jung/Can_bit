"""
Feature scaling utilities for ML pipeline.

This module provides functions to save, load, and apply scalers (StandardScaler,
RobustScaler, MinMaxScaler) for feature normalization in the ML pipeline.

Phase E: Structural improvements - Feature scaling.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

ScalerType = Literal["standard", "robust", "minmax", "none"]


def create_scaler(scaler_type: ScalerType):
    """
    Create a scaler instance based on type.
    
    Args:
        scaler_type: Type of scaler ("standard", "robust", "minmax", or "none")
    
    Returns:
        Scaler instance or None if scaler_type is "none"
    
    Raises:
        ValueError: If scaler_type is not supported
    """
    if scaler_type == "none":
        return None
    elif scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}. Choose from: standard, robust, minmax, none")


def save_scaler(scaler, scaler_path: Path | str, scaler_type: ScalerType) -> None:
    """
    Save scaler to disk.
    
    Args:
        scaler: Scaler instance (or None if no scaling)
        scaler_path: Path to save scaler file
        scaler_type: Type of scaler (for metadata)
    
    Raises:
        ValueError: If scaler is None but scaler_type is not "none"
    """
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    
    if scaler_type == "none" or scaler is None:
        logger.info(f"[Scaler] No scaling enabled (scaler_type={scaler_type}). Skipping scaler save.")
        return
    
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"[Scaler] Saved {scaler_type} scaler to {scaler_path}")
    except Exception as e:
        logger.error(f"[Scaler] Failed to save scaler to {scaler_path}: {e}")
        raise


def load_scaler(scaler_path: Path | str, scaler_type: ScalerType | None = None):
    """
    Load scaler from disk.
    
    Args:
        scaler_path: Path to scaler file
        scaler_type: Optional scaler type (for validation)
    
    Returns:
        Scaler instance or None if file doesn't exist and scaler_type is "none"
    
    Raises:
        FileNotFoundError: If scaler file doesn't exist and scaler_type is not "none"
    """
    scaler_path = Path(scaler_path)
    
    if not scaler_path.exists():
        if scaler_type == "none" or scaler_type is None:
            logger.info(f"[Scaler] No scaler file found at {scaler_path} and scaler_type is 'none'. Using no scaling.")
            return None
        else:
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"[Scaler] Loaded scaler from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"[Scaler] Failed to load scaler from {scaler_path}: {e}")
        raise


def apply_scaler(
    X: pd.DataFrame | np.ndarray,
    scaler,
    fit: bool = False,
) -> pd.DataFrame | np.ndarray:
    """
    Apply scaler to features.
    
    Args:
        X: Feature DataFrame or array
        scaler: Scaler instance (or None for no scaling)
        fit: If True, fit the scaler before transforming
    
    Returns:
        Scaled features (same type as input)
    """
    if scaler is None:
        return X
    
    is_dataframe = isinstance(X, pd.DataFrame)
    
    if fit:
        if is_dataframe:
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                index=X.index,
                columns=X.columns,
            )
        else:
            X_scaled = scaler.fit_transform(X)
        logger.debug(f"[Scaler] Fitted and transformed {X.shape[0]} samples, {X.shape[1]} features")
    else:
        if is_dataframe:
            X_scaled = pd.DataFrame(
                scaler.transform(X),
                index=X.index,
                columns=X.columns,
            )
        else:
            X_scaled = scaler.transform(X)
        logger.debug(f"[Scaler] Transformed {X.shape[0]} samples, {X.shape[1]} features")
    
    return X_scaled

