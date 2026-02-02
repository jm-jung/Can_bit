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
import requests

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
def create_public_binance_client(timeout_ms: int = 30000) -> ccxt.binance:
    """
    Create a public Binance client.

    We do NOT use API keys here, because OHLCV market data is public.
    This avoids AuthenticationError (-2015) from private SAPI endpoints.
    """
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "timeout": timeout_ms,  # timeout in milliseconds
        "options": {
            "defaultType": "spot",  # Spot trading only
        },
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
        # CSV는 timestamp를 문자열로 저장하므로, low_memory=False로 로드
        df = pd.read_csv(csv_path, low_memory=False)
        
        # 타입 정규화: symbol을 항상 문자열로
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str)
        
        # timestamp를 datetime으로 파싱하여 내부 처리용 컬럼 생성
        if "timestamp" in df.columns:
            # timestamp가 이미 datetime이면 그대로 사용, 아니면 파싱
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["_ts_dt"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            else:
                df["_ts_dt"] = pd.to_datetime(df["timestamp"], utc=True)
            
            # NaT 제거
            df = df[df["_ts_dt"].notna()].copy()
            df = df.sort_values("_ts_dt").reset_index(drop=True)
            
            logger.info(
                f"[Fetch] Loaded existing CSV: rows={len(df)}, "
                f"date_range={df['_ts_dt'].min()} to {df['_ts_dt'].max()}"
            )
        else:
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(
                f"[Fetch] Loaded existing CSV: rows={len(df)} (no timestamp column)"
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
    insecure_ssl: bool = False,
    fallback_direct_klines: bool = False,
    timeout_ms: int = 30000,
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
                # _ts_dt 컬럼이 있으면 사용, 없으면 timestamp 파싱
                if "_ts_dt" in existing_df.columns:
                    latest_ts_dt = existing_df["_ts_dt"].max()
                    since_ms = int(latest_ts_dt.timestamp() * 1000)
                else:
                    latest_ts = existing_df["timestamp"].max()
                    latest_ts_dt = pd.to_datetime(latest_ts, utc=True)
                    since_ms = int(latest_ts_dt.timestamp() * 1000)
                since = datetime.fromtimestamp(since_ms / 1000).strftime("%Y-%m-%d %H:%M")
                logger.info(
                    f"[Fetch] Append mode: Using latest timestamp from existing data: {since}"
                )

    # Parse since timestamp
    since_ms: Optional[int] = None
    if since:
        since_ms = parse_since(since)
        since_dt = datetime.fromtimestamp(since_ms / 1000)
        # Binance API가 since 이전 데이터를 반환할 수 있으므로, 
        # 1 bar(5m) 앞당겨서 요청하여 중복을 허용하고 나중에 dedup
        # 단, append 모드일 때만 (전체 fetch 시에는 정확한 since 사용)
        if append:
            # 5분 타임프레임이면 5분(300000ms) 앞당김
            timeframe_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
            minutes = timeframe_minutes.get(timeframe.lower(), 5)
            since_ms = max(0, since_ms - (minutes * 60 * 1000))
            since_dt = datetime.fromtimestamp(since_ms / 1000)
            logger.info(f"[Fetch] Starting from: {since_dt.strftime('%Y-%m-%d %H:%M:%S')} (adjusted -{minutes}m for dedup safety)")
        else:
            logger.info(f"[Fetch] Starting from: {since_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # CCXT를 기본 경로로 사용 (fallback_direct_klines가 True일 때만 직접 API 사용)
        use_direct_klines = fallback_direct_klines
        exchange = None
        
        if not use_direct_klines:
            exchange = create_public_binance_client(timeout_ms=timeout_ms)
            # load_markets는 선택적으로만 시도 (실패해도 계속 진행)
            try:
                logger.info("[Fetch] Attempting to load markets (optional, 1회만)...")
                exchange.load_markets()
                logger.info("[Fetch] Markets loaded successfully")
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"[Fetch] Failed to load markets: {e}. Continuing without markets cache...")
                # markets 로드 실패해도 fetch_ohlcv는 시도 가능
        else:
            logger.info("[Fetch] Using direct klines API (fallback mode).")

        all_rows: List[List] = []
        current_since = since_ms
        batch_count = 0
        total_fetched = 0
        
        # 증분 vs Full history 로그 구분
        if since_ms is not None:
            since_dt = datetime.fromtimestamp(since_ms / 1000)
            logger.info(
                f"[Fetch] Starting incremental fetch: "
                f"symbol={symbol}, timeframe={timeframe}, "
                f"since={since_dt.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"max_rows={max_rows if max_rows else 'unlimited'}"
            )
        else:
            logger.info(
                f"[Fetch] Starting full history fetch: "
                f"symbol={symbol}, timeframe={timeframe}, "
                f"max_rows={max_rows if max_rows else 'unlimited'}"
            )

        # 재시도 설정
        max_retries = 10
        retry_count = 0
        base_delay = 2  # 초기 지연 2초

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
                if since_ms is not None:
                    logger.info(
                        f"[Fetch] Batch {batch_count}: Incremental fetch since={current_since} "
                        f"(limit=1000)..."
                    )
                else:
                    logger.debug(
                        f"[Fetch] Batch {batch_count}: Fetching since={current_since} "
                        f"(limit=1000)..."
                    )

                # Rate limit protection
                time.sleep(0.5)

                # Fetch OHLCV data
                if use_direct_klines:
                    # Fallback: 직접 klines API 호출
                    url = "https://api.binance.com/api/v3/klines"
                    params = {
                        "symbol": symbol.replace("/", ""),  # BTC/USDT -> BTCUSDT
                        "interval": timeframe,
                        "limit": 1000,
                    }
                    if current_since:
                        params["startTime"] = current_since
                    
                    # SSL verify 옵션화 (기본값 True)
                    verify_ssl = not insecure_ssl
                    if not verify_ssl:
                        import urllib3
                        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                        logger.warning("[Fetch] [WARN] insecure ssl verify disabled")
                    
                    response = requests.get(url, params=params, timeout=30, verify=verify_ssl)
                    response.raise_for_status()
                    raw_data = response.json()
                    
                    # CCXT 형식으로 변환: [timestamp, open, high, low, close, volume, ...]
                    ohlcv_data = [[int(candle[0]), float(candle[1]), float(candle[2]), 
                                  float(candle[3]), float(candle[4]), float(candle[5])] 
                                 for candle in raw_data]
                else:
                    # 기본 경로: CCXT 사용
                    fetch_params = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": 1000,
                    }
                    if current_since:
                        fetch_params["since"] = current_since
                    
                    ohlcv_data = exchange.fetch_ohlcv(**fetch_params)
                
                # 성공 시 retry_count 리셋
                retry_count = 0

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

                # Latest timestamp를 UTC로 명시적으로 표시
                latest_dt_utc = pd.to_datetime(latest_timestamp, unit="ms", utc=True)
                logger.info(
                    f"[Fetch] Batch {batch_count} fetched: rows={len(ohlcv_data)}, "
                    f"Latest timestamp: {latest_dt_utc.strftime('%Y-%m-%d %H:%M:%S')} (UTC)"
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
                retry_count += 1
                if retry_count > max_retries:
                    # CCXT 실패 시 fallback으로 전환 시도 (1회만, fallback_direct_klines가 False일 때)
                    if not use_direct_klines and not fallback_direct_klines:
                        logger.warning(f"[Fetch] CCXT failed after {max_retries} retries: {e}. Switching to direct klines fallback...")
                        use_direct_klines = True
                        retry_count = 0  # fallback 전환 시 retry_count 리셋
                        continue
                    else:
                        error_msg = (
                            f"[Fetch] ❌ Network error exceeded max retries ({max_retries}): {e}\n"
                            f"  Endpoint: {getattr(e, 'url', 'unknown')}\n"
                            f"  Retries used: {retry_count - 1}\n"
                            f"  Suggestion: curl exchangeInfo OK -> likely ccxt http client settings; "
                            f"try --timeout-ms {timeout_ms} or --fallback-direct-klines 1"
                        )
                        logger.error(error_msg)
                        raise Exception(error_msg) from e
                
                # Exponential backoff: 2, 4, 8, 16, 32, 60, 60, ...
                delay = min(base_delay * (2 ** (retry_count - 1)), 60)
                logger.error(
                    f"[Fetch] Network error in batch {batch_count} (retry {retry_count}/{max_retries}): {e}"
                )
                logger.info(f"[Fetch] Retrying after {delay} seconds...")
                time.sleep(delay)
                continue
                
            except requests.exceptions.SSLError as e:
                # SSL 에러 발생 시 fallback으로 전환 시도 (1회만)
                if not use_direct_klines and not fallback_direct_klines:
                    logger.warning(f"[Fetch] SSL error in CCXT path: {e}. Switching to direct klines fallback...")
                    use_direct_klines = True
                    retry_count = 0  # fallback 전환 시 retry_count 리셋
                    continue
                else:
                    retry_count += 1
                    if retry_count > max_retries:
                        error_msg = (
                            f"[Fetch] ❌ SSL error exceeded max retries ({max_retries}): {e}\n"
                            f"  Retries used: {retry_count - 1}\n"
                            f"  Suggestion: If curl works, try --insecure-ssl 1 (not recommended)"
                        )
                        logger.error(error_msg)
                        raise Exception(error_msg) from e
                    delay = min(base_delay * (2 ** (retry_count - 1)), 60)
                    logger.error(f"[Fetch] SSL error in batch {batch_count} (retry {retry_count}/{max_retries}): {e}")
                    logger.info(f"[Fetch] Retrying after {delay} seconds...")
                    time.sleep(delay)
                    continue
                    
            except requests.exceptions.RequestException as e:
                # Direct klines fallback에서 네트워크 에러
                retry_count += 1
                if retry_count > max_retries:
                    error_msg = (
                        f"[Fetch] ❌ Request error exceeded max retries ({max_retries}): {e}\n"
                        f"  Retries used: {retry_count - 1}\n"
                        f"  Endpoint: {getattr(e, 'url', 'unknown')}"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg) from e
                
                delay = min(base_delay * (2 ** (retry_count - 1)), 60)
                logger.error(
                    f"[Fetch] Request error in batch {batch_count} (retry {retry_count}/{max_retries}): {e}"
                )
                logger.info(f"[Fetch] Retrying after {delay} seconds...")
                time.sleep(delay)
                continue

            except ccxt.ExchangeError as e:
                logger.error(f"[Fetch] Exchange error in batch {batch_count}: {e}")
                # Exchange errors are usually not recoverable
                raise Exception(f"Binance exchange error: {e}") from e

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

        # Convert timestamp from milliseconds to datetime (내부 처리용 컬럼)
        df_new["_ts_dt"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
        
        # symbol 컬럼 추가 (정규화)
        symbol_normalized = symbol.replace("/", "").upper()
        df_new["symbol"] = symbol_normalized

        # Sort by _ts_dt (oldest first)
        df_new = df_new.sort_values("_ts_dt").reset_index(drop=True)

        # Merge with existing data if append mode
        if append and len(existing_df) > 0:
            # 기존 데이터의 최신 timestamp 저장 (신규 row 계산용)
            if "_ts_dt" in existing_df.columns:
                existing_ts_max_dt = existing_df["_ts_dt"].max()
            else:
                existing_ts_max_dt = pd.to_datetime(existing_df["timestamp"].max(), utc=True) if "timestamp" in existing_df.columns else None
            
            # Combine dataframes
            df_combined = pd.concat([existing_df, df_new], ignore_index=True)

            # Remove duplicates based on _ts_dt and symbol (명확한 dedup 기준)
            duplicates_before = len(df_combined)
            df_combined = df_combined.drop_duplicates(
                subset=["_ts_dt", "symbol"], keep="last"
            )
            duplicates_removed = duplicates_before - len(df_combined)

            # Sort again by _ts_dt
            df_combined = df_combined.sort_values("_ts_dt").reset_index(drop=True)
            
            # 신규 row 계산 (existing_ts_max_dt 이후의 데이터)
            if existing_ts_max_dt is not None:
                new_rows = len(df_combined[df_combined["_ts_dt"] > existing_ts_max_dt])
                # 가져온 데이터의 최신 timestamp 확인
                new_ts_max = df_new["_ts_dt"].max() if len(df_new) > 0 else None
            else:
                new_rows = len(df_new)
                new_ts_max = df_new["_ts_dt"].max() if len(df_new) > 0 else None
            
            # 운영 안전장치: fetched rows > 0인데 new_rows == 0이고, 
            # 가져온 데이터의 최신 timestamp가 기존 최신보다 크면 문제
            if len(df_new) > 0 and new_rows == 0 and existing_ts_max_dt is not None and new_ts_max is not None:
                if new_ts_max > existing_ts_max_dt:
                    # 실제로 새로운 데이터가 있어야 하는데 반영되지 않음 → 문제
                    error_msg = (
                        f"[Fetch][ERROR] Fetched {len(df_new)} rows but no new rows were appended. "
                        f"Possible dedup/type bug. existing_ts_max={existing_ts_max_dt}, "
                        f"new_ts_min={df_new['_ts_dt'].min() if len(df_new) > 0 else 'N/A'}, "
                        f"new_ts_max={new_ts_max} (should be appended but wasn't)"
                    )
                    logger.error(error_msg)
                    # 에러 발생 시 예외를 발생시켜 update_ohlcv에서 처리
                    raise ValueError(error_msg)
                else:
                    # 가져온 데이터가 모두 기존 데이터 범위 내 → 정상 (중복)
                    logger.debug(
                        f"[Fetch] Fetched {len(df_new)} rows but all are duplicates. "
                        f"existing_ts_max={existing_ts_max_dt}, new_ts_max={new_ts_max}"
                    )

            logger.info(
                f"[Fetch] Merged with existing data: "
                f"existing={len(existing_df)}, new={len(df_new)}, "
                f"combined={len(df_combined)}, duplicates_removed={duplicates_removed}, "
                f"new_rows={new_rows}"
            )

            df_final = df_combined
        else:
            df_final = df_new
            new_rows = len(df_new)

        # Dry run check
        if dry_run:
            logger.info("[Fetch] DRY RUN: Would save but skipping file write.")
            logger.info(
                f"[Fetch] DRY RUN: Would save OHLCV: rows={len(df_final)}, "
                f"date_range={df_final['timestamp'].min()} to {df_final['timestamp'].max()}"
            )
            return output_path

        # Save to CSV
        # 저장 직전에 timestamp를 문자열로 변환
        df_save = df_final.copy()
        if "_ts_dt" in df_save.columns:
            df_save["timestamp"] = df_save["_ts_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
            # 내부 처리용 컬럼 제거
            df_save = df_save.drop(columns=["_ts_dt"])
        
        logger.info(f"[Fetch] Saving data to {output_path}...")
        df_save.to_csv(output_path, index=False)
        
        # 저장 후 최종 timestamp 범위 로그
        if "_ts_dt" in df_final.columns:
            saved_ts_max = df_final["_ts_dt"].max()
            saved_ts_min = df_final["_ts_dt"].min()
        else:
            saved_ts_max = df_final["timestamp"].max()
            saved_ts_min = df_final["timestamp"].min()

        logger.info(
            f"[Fetch] ✅ Saved OHLCV: rows={len(df_final)}, "
            f"date_range={saved_ts_min} to {saved_ts_max}"
        )

        return output_path

    except ccxt.NetworkError as e:
        error_msg = (
            f"Network error while fetching data from Binance: {str(e)}\n"
            f"  Endpoint: {getattr(e, 'url', 'unknown')}\n"
            f"  Hint: curl works -> likely CCXT client config/IPv6/proxy/timeout issue"
        )
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
    parser.add_argument(
        "--insecure-ssl",
        type=int,
        default=0,
        choices=[0, 1],
        help="Disable SSL certificate verification (not recommended) (default: 0)",
    )
    parser.add_argument(
        "--fallback-direct-klines",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use direct klines API instead of CCXT (last resort) (default: 0)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Network timeout in milliseconds (default: 30000)",
    )

    args = parser.parse_args()
    
    # 환경변수 확인 (CLI 인자보다 우선순위 낮음)
    import os
    insecure_ssl_env = os.getenv("CANBIT_INSECURE_SSL", "0") == "1"
    insecure_ssl = bool(args.insecure_ssl) or insecure_ssl_env

    try:
        output_path = fetch_full_history(
            symbol=args.symbol,
            timeframe=args.timeframe,
            since=args.since,
            max_rows=args.max_rows,
            append=args.append,
            dry_run=args.dry_run,
            outfile=args.outfile,
            insecure_ssl=insecure_ssl,
            fallback_direct_klines=bool(args.fallback_direct_klines),
            timeout_ms=args.timeout_ms,
        )

        if not args.dry_run:
            logger.info(f"\n📁 Saved OHLCV data to: {output_path.absolute()}")

    except Exception as e:
        logger.error(f"\n❌ Failed to fetch OHLCV data: {e}")
        raise


if __name__ == "__main__":
    main()
