from typing import Literal, TypedDict

import pandas as pd

from src.indicators.basic import get_df_with_indicators


Signal = Literal["LONG", "SHORT", "HOLD"]


class StrategyOutput(TypedDict):
    timestamp: str
    close: float
    ema_20: float | None
    rsi_14: float | None
    signal: Signal


def simple_ema_rsi_strategy() -> StrategyOutput:
    """Return a simple EMA + RSI strategy signal."""
    df = get_df_with_indicators()
    last = df.iloc[-1]

    close = float(last["close"])
    ema = float(last["ema_20"]) if not pd.isna(last["ema_20"]) else None
    rsi = float(last["rsi_14"]) if not pd.isna(last["rsi_14"]) else None

    signal: Signal = "HOLD"

    if ema is not None and rsi is not None:
        if close > ema and rsi < 70:
            signal = "LONG"
        elif close < ema and rsi > 30:
            signal = "SHORT"
        else:
            signal = "HOLD"

    return StrategyOutput(
        timestamp=str(last["timestamp"]),
        close=close,
        ema_20=ema,
        rsi_14=rsi,
        signal=signal,
    )
