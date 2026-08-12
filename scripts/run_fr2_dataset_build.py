#!/usr/bin/env python3
"""
FR2: Build dataset with preset microstructure_v1 and write parquet + feature_summary.
Output: data/datasets/tcn_microstructure_v1.parquet, data/diagnostics/fr2/feature_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dl.data.labels import create_3class_labels
from src.features.ml_feature_config import MLFeatureConfig
from src.ml.features import build_feature_frame
from src.services.ohlcv_service import load_ohlcv_df

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
HORIZON = 15
THRESHOLD = 0.004
PRESET = "microstructure_v1"

DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
FR2_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_PARQUET = DATASETS_DIR / "tcn_microstructure_v1.parquet"
FEATURE_SUMMARY_CSV = FR2_DIR / "feature_summary.csv"


def main():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    FR2_DIR.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv_df(timeframe=TIMEFRAME, symbol=SYMBOL, include_microstructure=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    config = MLFeatureConfig.from_preset(PRESET)
    features = build_feature_frame(
        df,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        feature_config=config,
    )
    if not features.index.equals(df.index):
        df = df.reset_index(drop=True)
        features = features.reset_index(drop=True)

    ret = (df["close"].shift(-HORIZON) / df["close"]) - 1
    direction = create_3class_labels(
        ret.values,
        pos_threshold=THRESHOLD,
        neg_threshold=THRESHOLD,
    )
    valid = ret.notna()
    features = features.loc[valid].copy()
    ts = df.loc[valid, "timestamp"].reset_index(drop=True)
    ret_valid = ret.loc[valid].reset_index(drop=True)
    direction_valid = pd.Series(direction[valid.values], index=features.index)

    out = features.copy()
    out["timestamp"] = ts.values
    out["return_next_horizon"] = ret_valid.values
    out["direction"] = direction_valid.values
    out = out.reset_index(drop=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"[FR2] Wrote {OUT_PARQUET} rows={len(out)} cols={len(out.columns)}")

    # Feature summary: feature count, null ratio per column, correlation path
    feature_cols = [c for c in features.columns if c not in ("timestamp", "return_next_horizon", "direction")]
    null_ratio = features[feature_cols].isna().mean()
    summary_rows = [{"feature": c, "null_ratio": null_ratio[c]} for c in feature_cols]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(FEATURE_SUMMARY_CSV, index=False)
    print(f"[FR2] Feature count={len(feature_cols)} null_ratio written to {FEATURE_SUMMARY_CSV}")

    corr = features[feature_cols].corr()
    corr_path = FR2_DIR / "fr2_feature_correlation.csv"
    corr.to_csv(corr_path)
    print(f"[FR2] Correlation matrix saved to {corr_path}")


if __name__ == "__main__":
    main()
