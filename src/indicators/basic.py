import pandas as pd

from src.services.ohlcv_service import load_ohlcv_df


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with SMA/EMA/RSI columns added."""
    df = df.copy()

    # Simple Moving Average (20)
    df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    # Exponential Moving Average (20)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (14)
    window = 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


def get_df_with_indicators() -> pd.DataFrame:
    """Load OHLCV data and append indicators."""
    df = load_ohlcv_df()
    return add_basic_indicators(df)


def get_last_row_with_indicators() -> dict:
    """Return the last row (as dict) with indicators."""
    df = get_df_with_indicators()
    return df.iloc[-1].to_dict()
