"""
Raw event storage utilities for append-only event logging.

This module provides functions to store and load raw events in a simple,
append-only format using Parquet files organized by symbol and timeframe.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Base directory for raw event storage
RAW_EVENTS_BASE_DIR = PROJECT_ROOT / "data" / "events" / "raw"


def get_raw_events_path(symbol: str, timeframe: str) -> Path:
    """
    Get the path for raw events file for a given symbol and timeframe.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "1m")
    
    Returns:
        Path to the raw events parquet file
        Format: data/events/raw/{symbol}_{timeframe}_events.parquet
    """
    # Normalize symbol (remove / if present)
    normalized_symbol = symbol.replace("/", "").upper()
    normalized_timeframe = timeframe.lower()
    
    filename = f"{normalized_symbol}_{normalized_timeframe}_events.parquet"
    file_path = RAW_EVENTS_BASE_DIR / filename
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    return file_path


def load_raw_events(
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Load raw events from parquet file.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "1m")
    
    Returns:
        DataFrame with raw events. Columns:
        - timestamp: datetime64[ns] (UTC, tz-naive)
        - category: str
        - sentiment_score: float
        - intensity: float
        - source_type: str
        - symbol: str
        - raw_json: str (optional, JSON serialized)
        - id: str (optional)
    
    If file doesn't exist, returns empty DataFrame with expected columns.
    """
    file_path = get_raw_events_path(symbol, timeframe)
    
    if not file_path.exists():
        logger.debug(f"Raw events file not found: {file_path}. Returning empty DataFrame.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "timestamp",
            "category",
            "sentiment_score",
            "intensity",
            "source_type",
            "symbol",
            "raw_json",
            "id",
        ])
    
    try:
        df = pd.read_parquet(file_path, engine="pyarrow")
        
        # Ensure timestamp is datetime64[ns] and tz-naive (UTC)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Remove timezone if present
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        
        # Sort by timestamp
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.debug(f"Loaded {len(df)} raw events from {file_path}")
        return df
    
    except ImportError as e:
        logger.error(f"pyarrow not installed. Please install: pip install pyarrow. Error: {e}")
        return pd.DataFrame(columns=[
            "timestamp",
            "category",
            "sentiment_score",
            "intensity",
            "source_type",
            "symbol",
            "raw_json",
            "id",
        ])
    except Exception as e:
        logger.error(f"Failed to load raw events from {file_path}: {e}")
        return pd.DataFrame(columns=[
            "timestamp",
            "category",
            "sentiment_score",
            "intensity",
            "source_type",
            "symbol",
            "raw_json",
            "id",
        ])


def append_raw_events(
    symbol: str,
    timeframe: str,
    new_events: pd.DataFrame,
) -> None:
    """
    Append new events to the raw events file (append-only).
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "1m")
        new_events: DataFrame with new events to append. Must have columns:
            - timestamp: datetime
            - category: str
            - sentiment_score: float
            - intensity: float
            - source_type: str
            - symbol: str (optional, will be filled if missing)
            - raw_json: str (optional)
            - id: str (optional)
    
    Raises:
        ValueError: If new_events is empty or missing required columns
    """
    if new_events.empty:
        logger.warning("new_events is empty. Nothing to append.")
        return
    
    # Ensure required columns exist
    required_columns = ["timestamp", "category", "sentiment_score", "intensity", "source_type"]
    missing_columns = [col for col in required_columns if col not in new_events.columns]
    if missing_columns:
        raise ValueError(f"new_events missing required columns: {missing_columns}")
    
    # Fill symbol if missing
    if "symbol" not in new_events.columns:
        new_events = new_events.copy()
        new_events["symbol"] = symbol
    
    # Normalize timestamp to datetime64[ns] (UTC, tz-naive)
    new_events = new_events.copy()
    new_events["timestamp"] = pd.to_datetime(new_events["timestamp"])
    if new_events["timestamp"].dt.tz is not None:
        new_events["timestamp"] = new_events["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    
    # Load existing events
    existing_df = load_raw_events(symbol, timeframe)
    
    if existing_df.empty:
        # No existing events, just save new ones
        combined_df = new_events.copy()
    else:
        # Combine and deduplicate
        combined_df = pd.concat([existing_df, new_events], ignore_index=True)
        
        # Deduplicate based on (timestamp, source_type, symbol, id if available)
        # Priority: id > (timestamp, source_type, symbol)
        if "id" in combined_df.columns:
            # Use id for deduplication if available
            combined_df = combined_df.drop_duplicates(subset=["id"], keep="last")
        else:
            # Use timestamp + source_type + symbol for deduplication
            dedup_cols = ["timestamp", "source_type", "symbol"]
            if all(col in combined_df.columns for col in dedup_cols):
                combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep="last")
        
        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
    
    # Save to parquet
    file_path = get_raw_events_path(symbol, timeframe)
    try:
        combined_df.to_parquet(file_path, engine="pyarrow", index=False)
        logger.info(
            f"Appended {len(new_events)} events to {file_path}. "
            f"Total events: {len(combined_df)}"
        )
    except ImportError as e:
        logger.error(f"pyarrow not installed. Please install: pip install pyarrow. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to save raw events to {file_path}: {e}")
        raise

