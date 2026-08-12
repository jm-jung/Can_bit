#!/usr/bin/env python3
"""
검증 3) feature_preset=events 검증.
- events가 어떤 피처 컬럼인지 코드 추적
- 최근 7일 추론 입력에서 이벤트 피처 통계 (non-null, mean/std, unique, NaN)
- 학습 vs 추론 스키마 일치 여부 (가능한 범위)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics"


def main():
    parser = argparse.ArgumentParser(description="TCN feature preset (events) verification")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    import pandas as pd
    from src.ml.features import build_feature_frame
    from src.services.ohlcv_service import load_ohlcv_df
    from src.core.config import settings

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("검증 3) feature_preset=events 검증")
    print("=" * 60)

    # 1) "events" preset 추적: build_feature_frame(use_events=True) -> build_event_feature_frame
    print("[1] events preset 코드 추적")
    print("  - feature_preset 표시: settings.EVENTS_ENABLED=True -> 'events' (아님 'base')")
    print("  - build_feature_frame(use_events=True) 시 event_df = build_event_feature_frame(...) 호출")
    print("  - src/events/aggregator.py: event_count_*, event_sentiment_*, event_time_since_last_min, event_share_*")
    print()

    df = load_ohlcv_df(timeframe="5m", symbol="BTCUSDT")
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = datetime.now() - timedelta(days=args.days)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    use_events = getattr(settings, "EVENTS_ENABLED", True)
    features = build_feature_frame(
        df,
        symbol="BTCUSDT",
        timeframe="5m",
        use_events=use_events,
    )
    features = features.dropna(how="all", subset=[c for c in features.columns if c != "timestamp"])
    if "timestamp" in features.columns:
        features = features.set_index("timestamp")
    event_cols = [c for c in features.columns if c.startswith("event_")]
    other_cols = [c for c in features.columns if not c.startswith("event_")]

    print("[2] 최근 {}일 추론 입력 피처".format(args.days))
    print(f"  전체 컬럼 수: {len(features.columns)}, 이벤트 컬럼 수: {len(event_cols)}")
    print("  events_feature_list (길이 {}): {}".format(len(event_cols), event_cols))
    print()

    # 3) 이벤트 피처별 통계
    print("[3] 이벤트 피처별 통계 (상위 20개 콘솔)")
    print("-" * 60)
    feature_stats = []
    for col in event_cols:
        s = features[col]
        non_null = s.notna().sum()
        n = len(s)
        uniq = s.nunique()
        nan_ratio = 1.0 - (non_null / n) if n else 0
        stat = {
            "col": col,
            "non_null_ratio": round(non_null / n, 4) if n else 0,
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "unique_count": int(uniq),
            "nan_ratio": round(nan_ratio, 4),
        }
        feature_stats.append(stat)
        if len(feature_stats) <= 20:
            print(f"  {col}: non_null_ratio={stat['non_null_ratio']:.2%}, mean={stat['mean']}, std={stat['std']}, unique={stat['unique_count']}, nan_ratio={stat['nan_ratio']:.2%}")
    print()

    # 4) 학습 vs 추론 스키마
    print("[4] 학습 vs 추론 스키마")
    print("  - 학습: train_tcn.py -> create_sequences(train_lstm_attn) -> build_feature_frame(use_events=settings.EVENTS_ENABLED)")
    print("  - 추론: tcn_model / ml_proba_cache -> build_feature_frame(use_events=settings.EVENTS_ENABLED)")
    print("  - 동일 설정이면 스키마 일치. 모델 체크포인트에 feature_names 미저장 -> diff 불가.")
    training_schema_vs_infer_schema_diff = "unknown (checkpoint does not store feature_names); assume same if EVENTS_ENABLED unchanged"
    print(f"  training_schema_vs_infer_schema_diff: {training_schema_vs_infer_schema_diff}")
    print()

    out_path = DIAG_DIR / f"tcn_feature_preset_debug_{datetime.now().strftime('%Y%m%d')}.json"
    payload = {
        "events_enabled": use_events,
        "events_feature_list": event_cols,
        "events_feature_count": len(event_cols),
        "each_feature_stats": feature_stats,
        "training_schema_vs_infer_schema_diff": training_schema_vs_infer_schema_diff,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {out_path}")
    print()
    print("Feature Preset Verdict:")
    print(f"  events features actually present? {'YES' if len(event_cols) > 0 else 'NO'}")
    nan_issues = [s for s in feature_stats if s["nan_ratio"] > 0.5 or s["unique_count"] <= 1]
    print(f"  NaN/constant issues: {len(nan_issues)} cols with nan_ratio>0.5 or unique<=1")
    if nan_issues:
        for x in nan_issues[:5]:
            print(f"    - {x['col']}: nan_ratio={x['nan_ratio']:.2%}, unique={x['unique_count']}")
    print("  training vs inference schema match? assumed YES (same build_feature_frame path)")
    print("=" * 60)


if __name__ == "__main__":
    main()
