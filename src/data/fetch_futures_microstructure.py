"""
Fetch Binance futures microstructure data (taker buy/sell, funding rate)
and save as parquet for FR2 feature merge.

Usage:
    python -m src.data.fetch_futures_microstructure --since 2023-01-01 --days 720
    python -m src.data.fetch_futures_microstructure --since 2023-01-01 --days 720 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "ohlcv"
SYMBOL = "BTCUSDT"
INTERVAL_5M = "5m"


def _ms_to_ts(ms: int) -> pd.Timestamp:
    return pd.to_datetime(ms, unit="ms", utc=True)


def fetch_futures_klines(since_ms: int, until_ms: int | None = None) -> pd.DataFrame:
    """Fetch 5m klines from Binance futures. Columns: timestamp, open, high, low, close, volume, taker_buy_volume."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_rows = []
    current = since_ms
    limit = 1000
    while True:
        params = {"symbol": SYMBOL, "interval": "5m", "limit": limit, "startTime": current}
        if until_ms is not None:
            params["endTime"] = until_ms
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for c in data:
            # [0 open time, 1 open, 2 high, 3 low, 4 close, 5 volume, ... 9 taker buy base volume]
            ts = int(c[0])
            vol = float(c[5])
            taker_buy = float(c[9])
            all_rows.append({
                "timestamp": _ms_to_ts(ts),
                "volume": vol,
                "taker_buy_volume": taker_buy,
                "taker_sell_volume": vol - taker_buy,
            })
        current = data[-1][0] + 1
        if until_ms is not None and current >= until_ms:
            break
        if len(data) < limit:
            break
        time.sleep(0.2)
    return pd.DataFrame(all_rows)


def fetch_funding_rate(since_ms: int, until_ms: int | None = None) -> pd.DataFrame:
    """Fetch funding rate history. Binance returns 8h intervals."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_rows = []
    current = since_ms
    limit = 1000
    while True:
        params = {"symbol": SYMBOL, "limit": limit, "startTime": current}
        if until_ms is not None:
            params["endTime"] = until_ms
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for c in data:
            ts = int(c["fundingTime"])
            rate = float(c["fundingRate"])
            all_rows.append({"timestamp": _ms_to_ts(ts), "funding_rate": rate})
        current = data[-1]["fundingTime"] + 1
        if until_ms is not None and current >= until_ms:
            break
        if len(data) < limit:
            break
        time.sleep(0.2)
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance futures microstructure for FR2")
    parser.add_argument("--since", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=720, help="Number of days to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Do not write file")
    parser.add_argument("--out", default=None, help="Output path (default: data/ohlcv/BTCUSDT_5m_microstructure.parquet)")
    args = parser.parse_args()

    since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    since_ms = int(since_dt.timestamp() * 1000)
    until_dt = since_dt + pd.Timedelta(days=args.days)
    until_ms = int(until_dt.timestamp() * 1000)

    logger.info("Fetching futures 5m klines (taker buy/sell)...")
    klines = fetch_futures_klines(since_ms, until_ms)
    logger.info("Fetching funding rate...")
    funding = fetch_funding_rate(since_ms, until_ms)

    # Align funding (8h) to 5m bars: last funding rate at or before each 5m timestamp
    out_df = klines[["timestamp", "taker_buy_volume", "taker_sell_volume"]].copy()
    out_df = out_df.sort_values("timestamp")
    funding = funding.sort_values("timestamp")
    out_df = pd.merge_asof(
        out_df,
        funding[["timestamp", "funding_rate"]],
        on="timestamp",
        direction="backward",
    )
    out_df["funding_rate"] = out_df["funding_rate"].fillna(0.0)

    # Placeholder columns (no public liquidation/OI history in simple form here)
    out_df["open_interest"] = 0.0
    out_df["liq_long_volume"] = 0.0
    out_df["liq_short_volume"] = 0.0

    out_path = Path(args.out) if args.out else OUT_DIR / "BTCUSDT_5m_microstructure.parquet"
    out_path = out_path.resolve()
    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)
        logger.info("Wrote %s (rows=%d)", out_path, len(out_df))
    else:
        logger.info("Dry-run: would write %s (rows=%d)", out_path, len(out_df))
    return out_df


if __name__ == "__main__":
    main()
