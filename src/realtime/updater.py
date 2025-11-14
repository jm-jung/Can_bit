"""
Realtime OHLCV updater using Binance data.
"""
from __future__ import annotations

from typing import Any, Dict

import ccxt
import pandas as pd

from src.services.ohlcv_service import DATA_PATH, load_ohlcv_df

exchange = ccxt.binance({"enableRateLimit": True})


def fetch_latest_candle(symbol: str = "BTC/USDT") -> Dict[str, Any]:
    """
    Fetch the latest 1m OHLCV candle for the given symbol.
    """
    data = exchange.fetch_ohlcv(symbol, "1m", limit=2)
    ts, o, h, l, c, v = data[-1]
    return {
        "timestamp": pd.to_datetime(ts, unit="ms"),
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
        "volume": float(v),
    }


def update_latest_candle(symbol: str = "BTC/USDT") -> bool:
    """
    Append the latest candle to the CSV if it's new and refresh cached data.

    Returns:
        True if a new candle was appended, False otherwise.
    """
    latest = fetch_latest_candle(symbol)

    df = load_ohlcv_df()
    last_ts = df.iloc[-1]["timestamp"]

    # Ensure timestamps are comparable
    if not isinstance(last_ts, pd.Timestamp):
        last_ts = pd.to_datetime(last_ts)

    latest_ts = latest["timestamp"]

    if latest_ts <= last_ts:
        return False

    new_df = pd.DataFrame([latest])
    updated = pd.concat([df, new_df], ignore_index=True)
    updated.to_csv(DATA_PATH, index=False)

    # Clear cached dataframe
    load_ohlcv_df.cache_clear()
    return True

