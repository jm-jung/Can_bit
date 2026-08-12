#!/usr/bin/env python3
"""Feature preset 스모크: preset으로 feature frame 생성 후 컬럼 목록 / NaN / unique count 출력."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Feature preset smoke test")
    parser.add_argument("--preset", type=str, default="calendar_e0")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()

    import pandas as pd
    from src.ml.features import build_feature_frame
    from src.services.ohlcv_service import load_ohlcv_df
    from src.features.ml_feature_config import MLFeatureConfig

    print("=" * 60)
    print(f"Feature preset smoke: preset={args.preset}, days={args.days}, tf={args.timeframe}, symbol={args.symbol}")
    print("=" * 60)

    df = load_ohlcv_df(timeframe=args.timeframe, symbol=args.symbol)
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = datetime.utcnow() - timedelta(days=args.days)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    df = df.loc[ts >= cutoff].copy()
    df = df.reset_index(drop=True)

    try:
        config = MLFeatureConfig.from_preset(args.preset)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    features = build_feature_frame(
        df,
        symbol=args.symbol,
        timeframe=args.timeframe,
        feature_config=config,
    )
    if "timestamp" in features.columns and features.index.name != "timestamp":
        features = features.set_index("timestamp") if "timestamp" in features.columns else features

    print(f"\nColumns ({len(features.columns)}): {list(features.columns)}")
    print("\nNaN check:")
    for c in features.columns:
        nan_count = features[c].isna().sum()
        print(f"  {c}: NaN={nan_count}")
    print("\nUnique count (const detection):")
    for c in features.columns:
        u = features[c].nunique()
        print(f"  {c}: unique={u}")
    print("=" * 60)


if __name__ == "__main__":
    main()
