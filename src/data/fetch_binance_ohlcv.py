"""
Binance OHLCV Data Fetcher with Full History Support

Fetches OHLCV (Open, High, Low, Close, Volume) data from Binance
and saves it to a CSV file. Supports full history download with
iterative fetching and rate limit protection.

Usage:
    # Fetch full history from a specific date
    python -m src.data.fetch_binance_ohlcv --since 2024-01-01 --symbol BTC/USDT --timeframe 1m

    # Append to existing CSV
    python -m src.data.fetch_binance_ohlcv --since 2024-01-01 --append

    # Dry run (test without saving)
    python -m src.data.fetch_binance_ohlcv --since 2024-01-01 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import ccxt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_public_binance_client() -> ccxt.binance:
    """
    Create a public Binance client.

    We do NOT use API keys here, because OHLCV market data is public.
    This avoids AuthenticationError (-2015) from private SAPI endpoints.
    """
    exchange = ccxt.binance({
        "enableRateLimit": True,
    })

    # 안전빵으로 계정 관련 통화 정보 요청 기능 비활성화
    # (이게 True면 sapi/v1/capital/config/getall 같은 private API를 호출하려고 해서 2015가 날 수 있음)
    exchange.has["fetchCurrencies"] = False

    return exchange


def parse_since(since_str: str) -> int:
    """
    Parse since string to milliseconds timestamp.

    Args:
        since_str: Date string in format "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"

    Returns:
        Timestamp in milliseconds
    """
    try:
        # Try parsing with time
        dt = datetime.strptime(since_str, "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            # Try parsing date only
            dt = datetime.strptime(since_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Invalid since format: {since_str}. "
                f"Expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'"
            )
    
    return int(dt.timestamp() * 1000)


def load_existing_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load existing CSV file if it exists.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with existing data, or empty DataFrame if file doesn't exist
    """
    if not csv_path.exists():
        logger.info(f"[Fetch] Existing CSV not found: {csv_path}. Starting fresh.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            f"[Fetch] Loaded existing CSV: rows={len(df)}, "
            f"date_range={df['timestamp'].min()} to {df['timestamp'].max()}"
        )
        return df
    except Exception as e:
        logger.warning(f"[Fetch] Failed to load existing CSV: {e}. Starting fresh.")
        return pd.DataFrame()


def fetch_full_history(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    since: Optional[str] = None,
    max_rows: Optional[int] = None,
    append: bool = False,
    dry_run: bool = False,
    outfile: str | Path = "src/data/btc_ohlcv.csv",
) -> Path:
    """
    Fetch full OHLCV history from Binance with iterative fetching.

    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        timeframe: Timeframe for OHLCV data (e.g., "1m", "5m", "1h", "1d")
        since: Start date string (format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM")
               If None, fetches from earliest available data
        max_rows: Maximum number of rows to fetch (None = unlimited)
        append: If True, append to existing CSV and remove duplicates
        dry_run: If True, fetch but don't save to file
        outfile: Output file path (relative to project root)

    Returns:
        Path object of the saved CSV file

    Raises:
        Exception: If data fetching or file saving fails
    """
    output_path = Path(outfile) if isinstance(outfile, str) else outfile
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if append mode
    existing_df = pd.DataFrame()
    if append:
        existing_df = load_existing_csv(output_path)
        if len(existing_df) > 0:
            # Use latest timestamp as since if not provided
            if since is None:
                latest_ts = existing_df["timestamp"].max()
                since_ms = int(pd.Timestamp(latest_ts).timestamp() * 1000)
                since = datetime.fromtimestamp(since_ms / 1000).strftime("%Y-%m-%d %H:%M")
                logger.info(
                    f"[Fetch] Append mode: Using latest timestamp from existing data: {since}"
                )

    # Parse since timestamp
    since_ms: Optional[int] = None
    if since:
        since_ms = parse_since(since)
        since_dt = datetime.fromtimestamp(since_ms / 1000)
        logger.info(f"[Fetch] Starting from: {since_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        exchange = create_public_binance_client()

        all_rows: List[List] = []
        current_since = since_ms
        batch_count = 0
        total_fetched = 0

        logger.info(
            f"[Fetch] Starting full history fetch: "
            f"symbol={symbol}, timeframe={timeframe}, "
            f"max_rows={max_rows if max_rows else 'unlimited'}"
        )

        while True:
            try:
                # Check max_rows limit
                if max_rows and total_fetched >= max_rows:
                    logger.info(
                        f"[Fetch] Reached max_rows limit ({max_rows}). Stopping."
                    )
                    break

                # Fetch batch
                batch_count += 1
                logger.debug(
                    f"[Fetch] Batch {batch_count}: Fetching since={current_since} "
                    f"(limit=1000)..."
                )

                # Rate limit protection
                time.sleep(0.5)

                # Fetch OHLCV data
                fetch_params = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "limit": 1000,
                }
                if current_since:
                    fetch_params["since"] = current_since

                ohlcv_data = exchange.fetch_ohlcv(**fetch_params)

                if not ohlcv_data:
                    logger.info("[Fetch] Empty batch received. No more data available.")
                    break

                # Check if we got new data
                if len(ohlcv_data) == 0:
                    logger.info("[Fetch] Empty batch received. No more data available.")
                    break

                # Add to accumulated rows
                all_rows.extend(ohlcv_data)
                total_fetched = len(all_rows)

                # Get latest timestamp for next iteration
                latest_timestamp = ohlcv_data[-1][0]  # First element is timestamp in ms
                latest_dt = datetime.fromtimestamp(latest_timestamp / 1000)

                logger.info(
                    f"[Fetch] Batch {batch_count} fetched: rows={len(ohlcv_data)}, "
                    f"Latest timestamp: {latest_dt.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                logger.info(f"[Fetch] Total accumulated rows: {total_fetched}")

                # Check if we got less than limit (last batch)
                if len(ohlcv_data) < 1000:
                    logger.info(
                        "[Fetch] Received less than 1000 rows. "
                        "This is likely the last batch."
                    )
                    break

                # Set next since to latest timestamp + 1ms to avoid duplicates
                current_since = latest_timestamp + 1

                # Check max_rows limit after batch
                if max_rows and total_fetched >= max_rows:
                    # Trim to max_rows
                    all_rows = all_rows[:max_rows]
                    total_fetched = len(all_rows)
                    logger.info(
                        f"[Fetch] Reached max_rows limit ({max_rows}). "
                        f"Trimming to {total_fetched} rows."
                    )
                    break

            except ccxt.NetworkError as e:
                logger.error(f"[Fetch] Network error in batch {batch_count}: {e}")
                logger.info("[Fetch] Retrying after 2 seconds...")
                time.sleep(2)
                continue

            except ccxt.ExchangeError as e:
                logger.error(f"[Fetch] Exchange error in batch {batch_count}: {e}")
                # Some exchange errors might be recoverable, but stop for now
                break

            except Exception as e:
                logger.error(f"[Fetch] Unexpected error in batch {batch_count}: {e}")
                raise

        if not all_rows:
            raise ValueError(f"No data fetched from Binance for {symbol}")

        # Convert to DataFrame
        df_new = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        # Convert timestamp from milliseconds to datetime
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")

        # Sort by timestamp (oldest first)
        df_new = df_new.sort_values("timestamp").reset_index(drop=True)

        # Merge with existing data if append mode
        if append and len(existing_df) > 0:
            # Combine dataframes
            df_combined = pd.concat([existing_df, df_new], ignore_index=True)

            # Remove duplicates based on timestamp
            df_combined = df_combined.drop_duplicates(
                subset=["timestamp"], keep="last"
            )

            # Sort again
            df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)

            logger.info(
                f"[Fetch] Merged with existing data: "
                f"existing={len(existing_df)}, new={len(df_new)}, "
                f"combined={len(df_combined)}, duplicates_removed={len(existing_df) + len(df_new) - len(df_combined)}"
            )

            df_final = df_combined
        else:
            df_final = df_new

        # Dry run check
        if dry_run:
            logger.info("[Fetch] DRY RUN: Would save but skipping file write.")
            logger.info(
                f"[Fetch] DRY RUN: Would save OHLCV: rows={len(df_final)}, "
                f"date_range={df_final['timestamp'].min()} to {df_final['timestamp'].max()}"
            )
            return output_path

        # Save to CSV
        logger.info(f"[Fetch] Saving data to {output_path}...")
        df_final.to_csv(output_path, index=False)

        logger.info(
            f"[Fetch] ✅ Saved OHLCV: rows={len(df_final)}, "
            f"date_range={df_final['timestamp'].min()} to {df_final['timestamp'].max()}"
        )

        return output_path

    except ccxt.NetworkError as e:
        error_msg = f"Network error while fetching data from Binance: {str(e)}"
        logger.error(f"[Fetch] ❌ {error_msg}")
        raise Exception(error_msg) from e

    except ccxt.ExchangeError as e:
        error_msg = f"Binance exchange error: {str(e)}"
        logger.error(f"[Fetch] ❌ {error_msg}")
        raise Exception(error_msg) from e

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[Fetch] ❌ {error_msg}")
        raise


def main():
    """CLI entry point for OHLCV fetching."""
    parser = argparse.ArgumentParser(
        description="Fetch full OHLCV history from Binance"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading pair symbol (default: BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Timeframe for OHLCV data (default: 1m)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Start date (format: YYYY-MM-DD or YYYY-MM-DD HH:MM). "
        "If not provided, fetches from earliest available.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to fetch (default: unlimited)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV and remove duplicates",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch but don't save to file (test mode)",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="src/data/btc_ohlcv.csv",
        help="Output file path (default: src/data/btc_ohlcv.csv)",
    )

    args = parser.parse_args()

    try:
        output_path = fetch_full_history(
            symbol=args.symbol,
            timeframe=args.timeframe,
            since=args.since,
            max_rows=args.max_rows,
            append=args.append,
            dry_run=args.dry_run,
            outfile=args.outfile,
        )

        if not args.dry_run:
            logger.info(f"\n📁 Saved OHLCV data to: {output_path.absolute()}")

    except Exception as e:
        logger.error(f"\n❌ Failed to fetch OHLCV data: {e}")
        raise


if __name__ == "__main__":
    main()
