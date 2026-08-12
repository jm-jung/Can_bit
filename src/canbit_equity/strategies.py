"""LONG/FLAT rule baselines — signal at t close, execute t+1 open."""
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index, name="signal")


def signal_sma_200(df: pd.DataFrame) -> pd.Series:
    return (df["close_adj"] > df["sma_200"]).astype(float)


def signal_dual_trend(df: pd.DataFrame) -> pd.Series:
    cond = (df["close_adj"] > df["sma_200"]) & (df["sma_50"] > df["sma_200"]) & (df["ret_20d"] > 0)
    return cond.astype(float)


def signal_trend_vol(df: pd.DataFrame, vol_cap: float) -> pd.Series:
    cond = (
        (df["close_adj"] > df["sma_200"])
        & (df["sma_50"] > df["sma_200"])
        & (df["ret_60d"] > 0)
        & (df["realized_vol_20"] <= vol_cap)
    )
    return cond.astype(float)


RULE_SIGNAL_FNS = {
    "BUY_AND_HOLD": signal_buy_and_hold,
    "SMA_200_FILTER": signal_sma_200,
    "DUAL_TREND_FILTER": signal_dual_trend,
}
