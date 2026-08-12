from pathlib import Path
from functools import lru_cache
from typing import List

import pandas as pd

from src.schemas.ohlcv import OHLCVCandle


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "btc_ohlcv.csv"

# Supported timeframes for resampling
SUPPORTED_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m"]

# [ML] OHLCV CSV path mapping for long-run resampled data
# When timeframe=5m, use the pre-resampled full CSV instead of on-the-fly resampling
# Path is relative to project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OHLCV_CSV_PATHS = {
    ("BTCUSDT", "1m"): PROJECT_ROOT / "src" / "data" / "btc_ohlcv.csv",
    ("BTCUSDT", "5m"): PROJECT_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv",
    # Add more mappings as needed for other symbols/timeframes
}
OUT_DIR = PROJECT_ROOT / "data" / "ohlcv"


def _parse_timeframe(timeframe: str) -> str:
    """
    Parse and validate timeframe string.
    
    Args:
        timeframe: Timeframe string (e.g., "1m", "3m", "5m", "15m", "30m")
    
    Returns:
        Normalized timeframe string
    
    Raises:
        ValueError: If timeframe is not supported
    """
    timeframe = timeframe.lower().strip()
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported: {SUPPORTED_TIMEFRAMES}"
        )
    return timeframe


def _resample_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to target timeframe.
    
    Args:
        df: DataFrame with OHLCV columns and timestamp index
        target_timeframe: Target timeframe (e.g., "3m", "5m", "15m", "30m")
    
    Returns:
        Resampled DataFrame
    """
    if target_timeframe == "1m":
        # No resampling needed
        return df
    
    # Convert timeframe to pandas offset
    timeframe_map = {
        "3m": "3T",   # 3 minutes
        "5m": "5T",   # 5 minutes
        "15m": "15T", # 15 minutes
        "30m": "30T", # 30 minutes
    }
    
    offset = timeframe_map.get(target_timeframe)
    if offset is None:
        raise ValueError(f"Cannot resample to timeframe: {target_timeframe}")
    
    # Ensure timestamp is index
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Resample OHLCV
    resampled = pd.DataFrame()
    resampled["open"] = df["open"].resample(offset).first()
    resampled["high"] = df["high"].resample(offset).max()
    resampled["low"] = df["low"].resample(offset).min()
    resampled["close"] = df["close"].resample(offset).last()
    resampled["volume"] = df["volume"].resample(offset).sum()
    
    # Drop rows with NaN (incomplete candles)
    resampled = resampled.dropna()
    
    # Reset index to have timestamp as column
    resampled = resampled.reset_index()
    resampled.rename(columns={"index": "timestamp"}, inplace=True)
    
    return resampled


def _load_ohlcv_df_impl(
    timeframe: str,
    symbol: str,
    include_microstructure: bool,
) -> pd.DataFrame:
    # Parse and validate timeframe
    target_timeframe = _parse_timeframe(timeframe)
    symbol_upper = symbol.upper().replace("/", "")

    # [ML] Check if we have a pre-resampled CSV for this symbol/timeframe combination
    csv_key = (symbol_upper, target_timeframe)
    if csv_key in OHLCV_CSV_PATHS:
        csv_path = OHLCV_CSV_PATHS[csv_key]
        if csv_path.exists():
            # Load pre-resampled data directly
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"[OHLCV Loader] Loading pre-resampled data: "
                f"symbol={symbol_upper}, timeframe={target_timeframe}, path={csv_path}"
            )
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(
                f"[OHLCV Loader] Loaded data: symbol={symbol_upper}, timeframe={target_timeframe}, "
                f"path={csv_path.name}, rows={len(df)}"
            )
            if include_microstructure and symbol_upper == "BTCUSDT" and target_timeframe == "5m":
                micro_path = OUT_DIR / "BTCUSDT_5m_microstructure.parquet"
                if micro_path.exists():
                    micro = pd.read_parquet(micro_path)
                    mts = pd.to_datetime(micro["timestamp"])
                    if mts.dt.tz is not None:
                        mts = mts.dt.tz_localize(None)
                    micro["timestamp"] = mts.astype("datetime64[ns]")
                    ots = pd.to_datetime(df["timestamp"])
                    if ots.dt.tz is not None:
                        ots = ots.dt.tz_localize(None)
                    df["timestamp"] = ots.astype("datetime64[ns]")
                    df = df.merge(
                        micro,
                        on="timestamp",
                        how="left",
                        suffixes=("", "_micro"),
                    )
                    # Drop duplicate columns if any (keep left)
                    for c in micro.columns:
                        if c != "timestamp" and c in df.columns:
                            pass  # keep
                    logger.info("[OHLCV Loader] Merged microstructure columns: %s", [c for c in micro.columns if c != "timestamp"])
                else:
                    logger.warning("[OHLCV Loader] Microstructure parquet not found: %s. Run fetch_futures_microstructure.", micro_path)
            return df
        else:
            # Pre-resampled file doesn't exist, fall back to on-the-fly resampling
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"[OHLCV Loader] Pre-resampled CSV not found: {csv_path}. "
                f"Falling back to on-the-fly resampling."
            )
    
    # Fallback: Load 1m data and resample on-the-fly
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"OHLCV CSV not found: {DATA_PATH}")
    
    # Load raw data (assumed to be 1m)
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[OHLCV Loader] Loading fallback data from: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], low_memory=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Resample if needed
    if target_timeframe != "1m":
        df = _resample_ohlcv(df, target_timeframe)
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 진단 로그 추가 (필수)
    ts_max = df["timestamp"].max()
    ts_min = df["timestamp"].min()
    logger.info(
        f"[OHLCV] loaded file={DATA_PATH}, rows={len(df)}, ts_max={ts_max}, ts_min={ts_min}"
    )
    
    return df


@lru_cache(maxsize=8)
def load_ohlcv_df(
    timeframe: str = "1m",
    symbol: str = "BTCUSDT",
    include_microstructure: bool = False,
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV and optionally resample / merge microstructure.

    For timeframe=5m and symbol=BTCUSDT, when include_microstructure=True,
    merges data/ohlcv/BTCUSDT_5m_microstructure.parquet if present (FR2).
    """
    return _load_ohlcv_df_impl(timeframe, symbol, include_microstructure)


def get_last_candle() -> OHLCVCandle:
    """Return the most recent OHLCV candle."""
    df = load_ohlcv_df()
    row = df.iloc[-1]
    return OHLCVCandle(**row.to_dict())


def get_recent_candles(limit: int = 100) -> List[OHLCVCandle]:
    """Return the most recent `limit` OHLCV candles."""
    df = load_ohlcv_df()
    df_recent = df.tail(limit)
    return [OHLCVCandle(**row.to_dict()) for _, row in df_recent.iterrows()]
