"""
FR2: Microstructure & market state features.

Order flow, funding/position pressure, liquidation cluster, aggressive volume.
All features use past-only data; lookback ≤ 50 bars. No future leakage.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional raw columns (if missing, filled with 0)
_TAKER_BUY = "taker_buy_volume"
_TAKER_SELL = "taker_sell_volume"
_FUNDING = "funding_rate"
_OPEN_INTEREST = "open_interest"
_LIQ_LONG = "liq_long_volume"
_LIQ_SHORT = "liq_short_volume"


def _ensure_series(df: pd.DataFrame, key: str, index: pd.Index) -> pd.Series:
    if key in df.columns:
        return df[key].reindex(index).fillna(0.0)
    return pd.Series(0.0, index=index)


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build FR2 microstructure features.

    Expects optional columns: taker_buy_volume, taker_sell_volume, funding_rate,
    open_interest, liq_long_volume, liq_short_volume. If missing, uses 0.

    Rules: past-only, lookback ≤ 50, no future data.
    """
    if df.index is None or len(df.index) == 0:
        raise ValueError("Input DataFrame must have a valid index")

    idx = df.index
    features = pd.DataFrame(index=idx)

    # ----- Order flow -----
    buy = _ensure_series(df, _TAKER_BUY, idx)
    sell = _ensure_series(df, _TAKER_SELL, idx)
    vol = df["volume"].reindex(idx).fillna(0.0).replace(0.0, 1e-12)

    ofi_raw = buy - sell
    features["ofi_raw"] = ofi_raw
    denom = (buy + sell).replace(0.0, np.nan)
    features["ofi_ratio"] = (ofi_raw / denom).fillna(0.0)

    for w in [5, 10, 20]:
        features[f"ofi_mean_{w}"] = ofi_raw.rolling(window=w, min_periods=1).mean()

    features["buy_pressure"] = (buy / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["sell_pressure"] = (sell / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["ofi_delta"] = ofi_raw - ofi_raw.shift(1)

    # ----- Funding & position pressure -----
    funding = _ensure_series(df, _FUNDING, idx)
    features["funding_rate"] = funding
    for w in [24, 72]:
        m = funding.rolling(window=w, min_periods=1).mean()
        s = funding.rolling(window=w, min_periods=1).std().replace(0.0, np.nan)
        features[f"funding_zscore_{w}"] = ((funding - m) / s).fillna(0.0)

    # Extreme flags (top/bottom 10% over 72 bars; cap window at 50 for consistency)
    w_fund = min(50, 72)
    roll_hi = funding.rolling(window=w_fund, min_periods=1).quantile(0.9)
    roll_lo = funding.rolling(window=w_fund, min_periods=1).quantile(0.1)
    features["funding_extreme_pos_flag"] = (funding >= roll_hi).astype(float).fillna(0.0)
    features["funding_extreme_neg_flag"] = (funding <= roll_lo).astype(float).fillna(0.0)

    # Open interest
    oi = _ensure_series(df, _OPEN_INTEREST, idx)
    features["oi_change"] = oi - oi.shift(1)
    oi_prev = oi.shift(1).replace(0.0, np.nan)
    features["oi_pct_change"] = (oi - oi.shift(1)) / oi_prev
    features["oi_pct_change"] = features["oi_pct_change"].fillna(0.0)

    # OI–price divergence: price up + OI down -> short covering; price down + OI up -> short build
    close = df["close"].reindex(idx).ffill().bfill()
    price_up = (close > close.shift(1)).astype(float)
    oi_down = (oi < oi.shift(1)).astype(float)
    price_down = (close < close.shift(1)).astype(float)
    oi_up = (oi > oi.shift(1)).astype(float)
    features["oi_divergence_flag"] = (price_up * oi_down + price_down * oi_up).clip(0, 1)

    # ----- Liquidation cluster -----
    liq_long = _ensure_series(df, _LIQ_LONG, idx)
    liq_short = _ensure_series(df, _LIQ_SHORT, idx)
    features["liq_long_volume"] = liq_long
    features["liq_short_volume"] = liq_short
    features["liq_imbalance"] = liq_long - liq_short
    features["liq_ratio"] = (liq_long / (liq_short + 1e-12)).replace(np.inf, 0.0)

    liq_total = liq_long + liq_short
    for w in [5, 10, 20]:
        features[f"liq_cluster_{w}"] = liq_total.rolling(window=w, min_periods=1).sum()

    roll95 = liq_total.rolling(window=50, min_periods=5).quantile(0.95)
    features["large_liq_event_flag"] = (liq_total > roll95).astype(float).fillna(0.0)

    # ----- Aggressive volume -----
    vol_ma5 = vol.rolling(window=5, min_periods=1).mean().replace(0.0, np.nan)
    vol_ma20 = vol.rolling(window=20, min_periods=1).mean().replace(0.0, np.nan)
    features["vol_spike_5"] = (vol / vol_ma5).fillna(1.0)
    features["vol_spike_20"] = (vol / vol_ma20).fillna(1.0)

    features["taker_buy_ratio"] = (buy / (buy + sell + 1e-12)).fillna(0.5)
    features["taker_sell_ratio"] = (sell / (buy + sell + 1e-12)).fillna(0.5)
    features["vol_imbalance"] = (buy - sell) / (vol + 1e-12)

    features["vol_momentum_3"] = vol / vol.shift(3).replace(0.0, np.nan)
    features["vol_momentum_6"] = vol / vol.shift(6).replace(0.0, np.nan)
    features["vol_momentum_3"] = features["vol_momentum_3"].fillna(1.0)
    features["vol_momentum_6"] = features["vol_momentum_6"].fillna(1.0)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features
