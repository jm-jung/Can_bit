"""
Label processing for ML pipeline.

This module provides LabelProcessor class that handles both classification
and regression label generation, with support for dynamic thresholds and
volatility regime adjustments.

Phase E: Structural improvements - Labeling strategy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LabelMode = Literal["classification", "regression"]


@dataclass
class LabelConfig:
    """Configuration for label generation."""
    mode: LabelMode = "classification"
    long_threshold: float = 0.0030  # 0.3%
    short_threshold: float = 0.0030  # 0.3%
    hold_threshold: float = 0.0010  # 0.1%
    horizon: int = 20
    enable_hold_labels: bool = False
    # Regime adjustments (multipliers)
    high_vol_multiplier: float = 1.3
    low_vol_multiplier: float = 0.8


class LabelProcessor:
    """
    Process labels for ML training and inference.
    
    Supports:
    - Classification mode: Binary labels (LONG/SHORT/HOLD)
    - Regression mode: Continuous return predictions
    - Volatility regime-aware thresholding
    """
    
    def __init__(self, config: LabelConfig):
        """
        Initialize label processor.
        
        Args:
            config: LabelConfig instance with labeling parameters
        """
        self.config = config
        logger.info(
            f"[LabelProcessor] Initialized with mode={config.mode}, "
            f"long_thr={config.long_threshold}, short_thr={config.short_threshold}, "
            f"hold_thr={config.hold_threshold}, horizon={config.horizon}"
        )
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        returns: pd.Series | None = None,
        regime_labels: pd.Series | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series | None]:
        """
        Generate labels from OHLCV data.
        
        Args:
            df: OHLCV DataFrame with 'close' column
            returns: Optional pre-computed returns (if None, computed from df)
            regime_labels: Optional volatility regime labels ("HIGH_VOL" or "LOW_VOL")
        
        Returns:
            Tuple of (y_long, y_short, y_hold) Series
            - y_hold is None if enable_hold_labels=False
            - For regression mode, y_long and y_short represent return predictions
        """
        if returns is None:
            # Compute forward returns
            returns = df["close"].shift(-self.config.horizon) / df["close"] - 1
        
        # Apply regime adjustments if provided
        long_thr = self.config.long_threshold
        short_thr = self.config.short_threshold
        hold_thr = self.config.hold_threshold
        
        if regime_labels is not None:
            long_thr = self._apply_regime_threshold(long_thr, regime_labels)
            short_thr = self._apply_regime_threshold(short_thr, regime_labels)
            hold_thr = self._apply_regime_threshold(hold_thr, regime_labels)
        
        if self.config.mode == "classification":
            return self._generate_classification_labels(returns, long_thr, short_thr, hold_thr)
        else:  # regression
            return self._generate_regression_labels(returns, long_thr, short_thr, hold_thr)
    
    def _apply_regime_threshold(
        self,
        base_threshold: float,
        regime_labels: pd.Series,
    ) -> pd.Series:
        """
        Apply volatility regime multipliers to threshold.
        
        Args:
            base_threshold: Base threshold value
            regime_labels: Series with "HIGH_VOL" or "LOW_VOL" labels
        
        Returns:
            Series of adjusted thresholds
        """
        adjusted = pd.Series(index=regime_labels.index, dtype=float)
        adjusted[regime_labels == "HIGH_VOL"] = base_threshold * self.config.high_vol_multiplier
        adjusted[regime_labels == "LOW_VOL"] = base_threshold * self.config.low_vol_multiplier
        # Default to base_threshold for NaN or other values
        adjusted = adjusted.fillna(base_threshold)
        return adjusted
    
    def _generate_classification_labels(
        self,
        returns: pd.Series,
        long_thr: float | pd.Series,
        short_thr: float | pd.Series,
        hold_thr: float | pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series | None]:
        """
        Generate classification labels (binary).
        
        Args:
            returns: Forward returns
            long_thr: Long threshold (float or Series for regime-aware)
            short_thr: Short threshold (float or Series for regime-aware)
            hold_thr: Hold threshold (float or Series for regime-aware)
        
        Returns:
            Tuple of (y_long, y_short, y_hold)
        """
        y_long = pd.Series(index=returns.index, dtype=int)
        y_short = pd.Series(index=returns.index, dtype=int)
        y_hold = pd.Series(index=returns.index, dtype=int) if self.config.enable_hold_labels else None
        
        valid_mask = returns.notna()
        
        if isinstance(long_thr, pd.Series):
            # Regime-aware thresholds
            y_long.loc[valid_mask] = (returns.loc[valid_mask] > long_thr.loc[valid_mask]).astype(int)
            y_short.loc[valid_mask] = (returns.loc[valid_mask] < -short_thr.loc[valid_mask]).astype(int)
            if y_hold is not None:
                abs_returns = returns.loc[valid_mask].abs()
                y_hold.loc[valid_mask] = (abs_returns <= hold_thr.loc[valid_mask]).astype(int)
        else:
            # Fixed thresholds
            y_long.loc[valid_mask] = (returns.loc[valid_mask] > long_thr).astype(int)
            y_short.loc[valid_mask] = (returns.loc[valid_mask] < -short_thr).astype(int)
            if y_hold is not None:
                abs_returns = returns.loc[valid_mask].abs()
                y_hold.loc[valid_mask] = (abs_returns <= hold_thr).astype(int)
        
        y_long.loc[~valid_mask] = np.nan
        y_short.loc[~valid_mask] = np.nan
        if y_hold is not None:
            y_hold.loc[~valid_mask] = np.nan
        
        return y_long, y_short, y_hold
    
    def _generate_regression_labels(
        self,
        returns: pd.Series,
        long_thr: float | pd.Series,
        short_thr: float | pd.Series,
        hold_thr: float | pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series | None]:
        """
        Generate regression labels (continuous returns).
        
        In regression mode:
        - y_long: Future return (for LONG model to predict)
        - y_short: Negative of future return (for SHORT model to predict)
        - y_hold: Binary indicator (1 if abs(return) <= hold_thr)
        
        Args:
            returns: Forward returns
            long_thr: Long threshold (for hold zone, not used in regression)
            short_thr: Short threshold (for hold zone, not used in regression)
            hold_thr: Hold threshold (for hold zone)
        
        Returns:
            Tuple of (y_long, y_short, y_hold)
        """
        y_long = returns.copy()  # Direct return prediction
        y_short = -returns.copy()  # Negative return for SHORT model
        
        y_hold = None
        if self.config.enable_hold_labels:
            valid_mask = returns.notna()
            y_hold = pd.Series(index=returns.index, dtype=int)
            if isinstance(hold_thr, pd.Series):
                abs_returns = returns.loc[valid_mask].abs()
                y_hold.loc[valid_mask] = (abs_returns <= hold_thr.loc[valid_mask]).astype(int)
            else:
                abs_returns = returns.loc[valid_mask].abs()
                y_hold.loc[valid_mask] = (abs_returns <= hold_thr).astype(int)
            y_hold.loc[~valid_mask] = np.nan
        
        return y_long, y_short, y_hold
    
    def predict_to_signal(
        self,
        prediction_long: float,
        prediction_short: float | None = None,
        regime_label: str | None = None,
    ) -> Literal["LONG", "SHORT", "HOLD"]:
        """
        Convert model prediction to trading signal.
        
        Args:
            prediction_long: LONG model prediction (probability or return)
            prediction_short: Optional SHORT model prediction
            regime_label: Optional volatility regime ("HIGH_VOL" or "LOW_VOL")
        
        Returns:
            Trading signal: "LONG", "SHORT", or "HOLD"
        """
        # Apply regime adjustments
        long_thr = self.config.long_threshold
        short_thr = self.config.short_threshold
        
        if regime_label == "HIGH_VOL":
            long_thr *= self.config.high_vol_multiplier
            short_thr *= self.config.high_vol_multiplier
        elif regime_label == "LOW_VOL":
            long_thr *= self.config.low_vol_multiplier
            short_thr *= self.config.low_vol_multiplier
        
        if self.config.mode == "classification":
            # Classification: prediction is probability
            is_long = prediction_long >= long_thr
            is_short = (prediction_short is not None and prediction_short >= short_thr) if prediction_short is not None else False
            
            if is_long and is_short:
                # Conflict: choose based on margin
                margin_long = prediction_long - long_thr
                margin_short = prediction_short - short_thr if prediction_short is not None else 0.0
                return "LONG" if margin_long >= margin_short else "SHORT"
            elif is_long:
                return "LONG"
            elif is_short:
                return "SHORT"
            else:
                return "HOLD"
        else:
            # Regression: prediction is return
            is_long = prediction_long >= long_thr
            is_short = (prediction_short is not None and prediction_short >= short_thr) if prediction_short is not None else False
            
            if is_long and is_short:
                # Conflict: choose based on absolute value
                return "LONG" if abs(prediction_long) >= abs(prediction_short) else "SHORT"
            elif is_long:
                return "LONG"
            elif is_short:
                return "SHORT"
            else:
                return "HOLD"

