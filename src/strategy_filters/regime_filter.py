"""
EMA200 regime filter: Long-only entry gating.
- Rule: if close <= ema200 then skip LONG entry (엄격히 <= 사용, 문서화).
- No lookahead: EMA is computed from past~current close only.
"""
from __future__ import annotations

import pandas as pd


def compute_ema(series_close: pd.Series, span: int = 200) -> pd.Series:
    """
    Compute EMA of close. No lookahead: each value uses only past and current data.
    Uses pandas ewm(span=span, adjust=False).mean().
    """
    return series_close.ewm(span=span, adjust=False).mean()


def allow_long_entry(close: float, ema: float) -> bool:
    """
    Allow LONG entry only when close > ema200.
    If close <= ema: entry is blocked (return False).
    엄격히 "<=" 사용: close <= ema 이면 Long 진입 금지.
    """
    if ema != ema:  # NaN
        return True  # no filter when ema not yet available
    return close > ema


def compute_ema_slope(ema_series: pd.Series, lookback: int = 48) -> pd.Series:
    """
    ema200_slope[t] = ema[t] - ema[t - lookback]. No lookahead (uses only past).
    First `lookback` bars are NaN.
    """
    return ema_series - ema_series.shift(lookback)


def compute_vol_compress(series_close: pd.Series, window: int = 48) -> pd.Series:
    """
    Rolling std of pct_change (volatility). No lookahead: rolling uses past and current only.
    ret = close.pct_change(); vol = ret.rolling(window=window).std()
    Used for vol_compress regime: if vol <= threshold then skip Long entry (횡보 차단).
    """
    ret = series_close.pct_change()
    return ret.rolling(window=window, min_periods=1).std()
