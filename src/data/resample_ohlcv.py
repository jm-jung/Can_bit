"""
OHLCV data resampling utility.

This module provides functionality to resample 1m OHLCV data to higher timeframes
(e.g., 5m, 15m, 30m) and save the resampled data to CSV files for long-run backtesting.

Usage:
    python -m src.data.resample_ohlcv --from-csv src/data/btc_ohlcv.csv --to-csv data/ohlcv/BTCUSDT_5m_full.csv --from-timeframe 1m --to-timeframe 5m --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resample_ohlcv(
    from_csv: str | Path,
    to_csv: str | Path,
    from_timeframe: str = "1m",
    to_timeframe: str = "5m",
    symbol: str = "BTCUSDT",
) -> None:
    """
    Resample OHLCV data from one timeframe to another.
    
    Args:
        from_csv: Path to source CSV file (1m OHLCV data)
        to_csv: Path to output CSV file (resampled data)
        from_timeframe: Source timeframe (default: "1m")
        to_timeframe: Target timeframe (default: "5m")
        symbol: Trading symbol (default: "BTCUSDT")
    
    Raises:
        FileNotFoundError: If source CSV file not found
        ValueError: If timeframes are invalid
    """
    from_path = Path(from_csv)
    to_path = Path(to_csv)
    
    # Validate source file
    if not from_path.exists():
        raise FileNotFoundError(f"Source CSV file not found: {from_path}")
    
    # Validate timeframes
    timeframe_map = {
        "1m": "1T",
        "3m": "3T",
        "5m": "5T",
        "15m": "15T",
        "30m": "30T",
    }
    
    if from_timeframe not in timeframe_map:
        raise ValueError(f"Unsupported source timeframe: {from_timeframe}")
    if to_timeframe not in timeframe_map:
        raise ValueError(f"Unsupported target timeframe: {to_timeframe}")
    
    if from_timeframe == to_timeframe:
        logger.warning(f"Source and target timeframes are the same ({from_timeframe}). No resampling needed.")
        return
    
    # Load source data
    logger.info(f"[Resample OHLCV] Loading {from_timeframe} data from {from_path}...")
    df = pd.read_csv(from_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"[Resample OHLCV] Loaded {from_timeframe} data: rows={len(df)}")
    logger.info(f"[Resample OHLCV] Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Set timestamp as index for resampling
    df = df.set_index("timestamp")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Resample OHLCV
    offset = timeframe_map[to_timeframe]
    logger.info(f"[Resample OHLCV] Resampling to {to_timeframe} (offset={offset})...")
    
    resampled = pd.DataFrame()
    resampled["open"] = df["open"].resample(offset).first()
    resampled["high"] = df["high"].resample(offset).max()
    resampled["low"] = df["low"].resample(offset).min()
    resampled["close"] = df["close"].resample(offset).last()
    resampled["volume"] = df["volume"].resample(offset).sum()
    
    # Drop rows with NaN (incomplete candles at the beginning/end)
    resampled = resampled.dropna()
    
    # Reset index to have timestamp as column
    resampled = resampled.reset_index()
    resampled.rename(columns={"index": "timestamp"}, inplace=True)
    
    logger.info(f"[Resample OHLCV] Resampled to {to_timeframe}: rows={len(resampled)}")
    
    # Add symbol column if not present
    if "symbol" not in resampled.columns:
        resampled["symbol"] = symbol
    
    # Create output directory if it doesn't exist
    to_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    resampled.to_csv(to_path, index=False)
    logger.info(f"[Resample OHLCV] Saved {to_timeframe} data to {to_path}")
    logger.info(f"[Resample OHLCV] Final date range: {resampled['timestamp'].min()} to {resampled['timestamp'].max()}")


def main():
    """CLI entry point for OHLCV resampling."""
    parser = argparse.ArgumentParser(
        description="Resample OHLCV data from one timeframe to another"
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default="src/data/btc_ohlcv.csv",
        help="Path to source CSV file (default: src/data/btc_ohlcv.csv)",
    )
    parser.add_argument(
        "--to-csv",
        type=str,
        default="data/ohlcv/BTCUSDT_5m_full.csv",
        help="Path to output CSV file (default: data/ohlcv/BTCUSDT_5m_full.csv)",
    )
    parser.add_argument(
        "--from-timeframe",
        type=str,
        default="1m",
        choices=["1m", "3m", "5m", "15m", "30m"],
        help="Source timeframe (default: 1m)",
    )
    parser.add_argument(
        "--to-timeframe",
        type=str,
        default="5m",
        choices=["1m", "3m", "5m", "15m", "30m"],
        help="Target timeframe (default: 5m)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    
    args = parser.parse_args()
    
    try:
        resample_ohlcv(
            from_csv=args.from_csv,
            to_csv=args.to_csv,
            from_timeframe=args.from_timeframe,
            to_timeframe=args.to_timeframe,
            symbol=args.symbol,
        )
        logger.info("[Resample OHLCV] ✅ Resampling completed successfully")
    except Exception as e:
        logger.error(f"[Resample OHLCV] ❌ Resampling failed: {e}")
        raise


if __name__ == "__main__":
    main()

