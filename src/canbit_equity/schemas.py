"""QQQ schema helpers."""
from __future__ import annotations

NORMALIZED_COLUMNS = [
    "symbol",
    "session_date",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "adj_close",
    "adjustment_factor",
    "open_adj",
    "high_adj",
    "low_adj",
    "close_adj",
    "volume",
    "dividends",
    "stock_splits",
    "provider",
    "provider_provenance",
    "session_open_utc",
    "session_close_utc",
    "source_timestamp",
    "downloaded_at_utc",
]

FEATURE_LOOKBACK_MIN = 252
