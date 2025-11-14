"""
Binance OHLCV Data Fetcher

Fetches OHLCV (Open, High, Low, Close, Volume) data from Binance
and saves it to a CSV file.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import ccxt


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


def fetch_and_save_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    limit: int = 1000,
    outfile: str | Path = "src/data/btc_ohlcv.csv",
) -> Path:
    """
    Fetch OHLCV data from Binance and save to CSV file.

    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        timeframe: Timeframe for OHLCV data (e.g., "1m", "5m", "1h", "1d")
        limit: Number of candles to fetch (max 1000 for most exchanges)
        outfile: Output file path (relative to project root)

    Returns:
        Path object of the saved CSV file

    Raises:
        Exception: If data fetching or file saving fails
    """
    # Convert outfile to Path object if it's a string
    output_path = Path(outfile) if isinstance(outfile, str) else outfile

    # Create data directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # ✅ Public Binance client (no API key)
        exchange = create_public_binance_client()

        print(f"Fetching {limit} candles of {symbol} ({timeframe}) from Binance...")

        # Fetch OHLCV data from Binance
        ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        if not ohlcv_data:
            raise ValueError(f"No data received from Binance for {symbol}")

        # Convert to pandas DataFrame
        df = pd.DataFrame(
            ohlcv_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Sort by timestamp (oldest first)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Save to CSV file
        print(f"Saving data to {output_path}...")
        df.to_csv(output_path, index=False)

        print(f"✅ Successfully saved {len(df)} records to {output_path}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return output_path

    except ccxt.NetworkError as e:
        error_msg = f"Network error while fetching data from Binance: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg) from e

    except ccxt.ExchangeError as e:
        error_msg = f"Binance exchange error: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg) from e

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        raise


if __name__ == "__main__":
    output_path = fetch_and_save_ohlcv(
        symbol="BTC/USDT",
        timeframe="1m",
        limit=1000,
        outfile=Path("src/data/btc_ohlcv.csv"),
    )

    print(f"\n📁 Saved OHLCV data to: {output_path.absolute()}")
