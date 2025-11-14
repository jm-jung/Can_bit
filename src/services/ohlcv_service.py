from pathlib import Path
from functools import lru_cache
from typing import List

import pandas as pd

from src.schemas.ohlcv import OHLCVCandle


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "btc_ohlcv.csv"


@lru_cache(maxsize=1)
def load_ohlcv_df() -> pd.DataFrame:
    """Load OHLCV data from CSV into a DataFrame."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"OHLCV CSV not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


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
