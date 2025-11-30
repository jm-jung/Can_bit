"""
Utility for loading optimized ML thresholds from JSON files.

Threshold files are stored in data/thresholds/ directory with naming convention:
{strategy}_{symbol}_{timeframe}.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def load_optimized_thresholds(strategy: str, symbol: str, timeframe: str):
    """
    Load optimized thresholds from JSON file.
    
    Args:
        strategy: Strategy name (e.g., "ml_xgb", "ml_lstm_attn")
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "1m")
    
    Returns:
        Tuple of (long_threshold, short_threshold, path)
        - long_threshold: float
        - short_threshold: float | None
        - path: str (absolute path to the JSON file)
    
    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    filename = f"{strategy}_{symbol}_{timeframe}.json"
    path = os.path.join("data", "thresholds", filename)
    
    # Convert to absolute path
    abs_path = os.path.abspath(path)
    
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Optimized threshold JSON not found: {abs_path}")
    
    with open(abs_path, "r") as f:
        data = json.load(f)
    
    long_th = data.get("best_long_threshold")
    short_th = data.get("best_short_threshold")
    
    return long_th, short_th, abs_path

