"""Past-only daily features for QQQ."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import QQQConfig, paths
from .schemas import FEATURE_LOOKBACK_MIN


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(
    df: pd.DataFrame,
    cfg: QQQConfig = QQQConfig(),
    *,
    persist: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df.copy()
    d["session_date"] = pd.to_datetime(d["session_date"]).dt.normalize()
    d = d.sort_values("session_date").reset_index(drop=True)
    c = d["close_adj"]
    o = d["open_adj"]
    h = d["high_adj"]
    l = d["low_adj"]
    v = d["volume"].astype(float)

    out = d[["symbol", "session_date", "open_adj", "high_adj", "low_adj", "close_adj", "volume"]].copy()
    for n in (1, 2, 5, 10, 20, 60, 120, 252):
        out[f"ret_{n}d"] = c.pct_change(n)

    for n in (5, 10, 20, 50, 100, 200):
        sma = c.rolling(n, min_periods=n).mean()
        out[f"close_sma_ratio_{n}"] = c / sma

    sma10 = c.rolling(10, min_periods=10).mean()
    sma20 = c.rolling(20, min_periods=20).mean()
    sma50 = c.rolling(50, min_periods=50).mean()
    sma200 = c.rolling(200, min_periods=200).mean()
    out["sma_10_sma_20_ratio"] = sma10 / sma20
    out["sma_20_sma_50_ratio"] = sma20 / sma50
    out["sma_50_sma_200_ratio"] = sma50 / sma200

    for n in (10, 20, 50):
        ema = c.ewm(span=n, adjust=False).mean()
        out[f"ema_{n}_ratio"] = c / ema

    logret = np.log(c / c.shift(1))
    for n in (5, 10, 20, 60):
        out[f"realized_vol_{n}"] = logret.rolling(n, min_periods=n).std() * np.sqrt(252)

    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    out["atr_14_ratio"] = atr14 / c
    out["range_1d_ratio"] = (h - l) / c
    out["range_5d_mean"] = out["range_1d_ratio"].rolling(5, min_periods=5).mean()
    neg = logret.clip(upper=0.0)
    out["downside_vol_20"] = neg.rolling(20, min_periods=20).std() * np.sqrt(252)

    out["momentum_20"] = c / c.shift(20) - 1
    out["momentum_60"] = c / c.shift(60) - 1
    out["momentum_120"] = c / c.shift(120) - 1
    out["rsi_14"] = _rsi(c, 14)

    v5 = v.rolling(5, min_periods=5).mean()
    v20 = v.rolling(20, min_periods=20).mean()
    v60 = v.rolling(60, min_periods=60).mean()
    out["volume_ratio_5_20"] = v5 / v20.replace(0, np.nan)
    out["volume_ratio_20_60"] = v20 / v60.replace(0, np.nan)
    out["volume_zscore_20"] = (v - v20) / v.rolling(20, min_periods=20).std()
    out["volume_zscore_60"] = (v - v60) / v.rolling(60, min_periods=60).std()
    out["dollar_volume"] = c * v

    out["drawdown_from_20d_high"] = c / c.rolling(20, min_periods=20).max() - 1
    out["drawdown_from_60d_high"] = c / c.rolling(60, min_periods=60).max() - 1
    out["drawdown_from_252d_high"] = c / c.rolling(252, min_periods=252).max() - 1
    out["distance_from_20d_high"] = out["drawdown_from_20d_high"]
    out["distance_from_52w_high"] = out["drawdown_from_252d_high"]
    pos = (c.pct_change() > 0).astype(float)
    out["positive_return_ratio_20"] = pos.rolling(20, min_periods=20).mean()
    out["positive_return_ratio_60"] = pos.rolling(60, min_periods=60).mean()

    out["overnight_gap"] = o / prev_close - 1
    out["close_to_open_return"] = out["overnight_gap"]
    out["intraday_return"] = c / o - 1

    # helper columns for strategies
    out["sma_50"] = sma50
    out["sma_200"] = sma200
    out["ret_20d"] = out["ret_20d"]
    out["ret_60d"] = out["ret_60d"]

    feature_cols = [c for c in out.columns if c not in {"symbol", "session_date", "open_adj", "high_adj", "low_adj", "close_adj", "volume", "sma_50", "sma_200"}]
    # drop initial lookback
    out = out.iloc[FEATURE_LOOKBACK_MIN:].reset_index(drop=True)

    manifest = {
        "symbol": cfg.symbol,
        "rows": int(len(out)),
        "start": str(pd.Timestamp(out["session_date"].iloc[0]).date()) if len(out) else None,
        "end": str(pd.Timestamp(out["session_date"].iloc[-1]).date()) if len(out) else None,
        "min_lookback": FEATURE_LOOKBACK_MIN,
        "feature_columns": feature_cols,
        "note": "All features use only information available at session t close. No centered rolling / global normalize.",
        "production_ready": False,
        "promotion_ready": False,
    }
    if persist:
        p = paths(cfg)
        out.to_parquet(p["features_file"], index=False)
        p["feature_manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return out, manifest


def logistic_feature_columns() -> List[str]:
    return [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ret_60d",
        "close_sma_ratio_20",
        "close_sma_ratio_50",
        "close_sma_ratio_200",
        "sma_50_sma_200_ratio",
        "realized_vol_20",
        "momentum_20",
        "momentum_60",
        "rsi_14",
        "volume_ratio_5_20",
        "drawdown_from_60d_high",
        "distance_from_52w_high",
        "overnight_gap",
        "atr_14_ratio",
    ]
