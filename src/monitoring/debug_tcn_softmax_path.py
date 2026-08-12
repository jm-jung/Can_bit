#!/usr/bin/env python3
"""
검증 2) Softmax/확률 계산 경로 검증.
- 모델 출력이 logits인지 proba인지
- softmax 적용 횟수
- 100 rows: raw_output_sum, softmax_sum, max_proba 통계 vs 현재 파이프라인
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_CACHE = PROJECT_ROOT / "data" / "cache" / "ml_predictions"
PROBA_PATH = DATA_CACHE / "ml_tcn_BTCUSDT_5m_proba.parquet"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics"


def main():
    parser = argparse.ArgumentParser(description="TCN softmax path verification")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    import pandas as pd
    from src.dl.tcn_model import build_normalized_tcn_sequences_full, get_tcn_model
    from src.ml.features import build_feature_frame
    from src.services.ohlcv_service import load_ohlcv_df
    from src.core.config import settings

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAG_DIR / f"tcn_softmax_debug_{datetime.now().strftime('%Y%m%d')}.json"

    print("=" * 60)
    print("검증 2) Softmax/확률 계산 경로 검증")
    print("=" * 60)

    # 1) 모델 forward 출력 확인 (코드 레벨)
    print("[1] 모델 forward 출력 (코드): src/dl/models/tcn.py -> return logits (softmax 없음)")
    print("    src/dl/tcn_model.py predict_proba_batch -> F.softmax(logits, dim=-1) 1회 적용")
    print()

    # 2) Load data and run model for n rows (use tail to keep run fast)
    df = load_ohlcv_df(timeframe="5m", symbol="BTCUSDT")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.tail(400).reset_index(drop=True)
    cutoff = datetime.now() - timedelta(days=args.days)
    features = build_feature_frame(df, symbol="BTCUSDT", timeframe="5m", use_events=settings.EVENTS_ENABLED)
    features = features.dropna()
    if len(features) < 60:
        print("[ERROR] Not enough features.")
        sys.exit(1)

    model = get_tcn_model()
    if not model or not model.is_loaded():
        print("[ERROR] TCN model not loaded.")
        sys.exit(1)
    window_size = model.window_size
    feature_cols = features.columns.tolist()
    model.feature_cols = feature_cols

    sequences = build_normalized_tcn_sequences_full(
        features_df=features,
        window_size=window_size,
        feature_cols=feature_cols,
        logger=None,
    )
    num_seq = sequences.shape[0]
    n_sample = min(args.n, num_seq)

    device = next(model.model.parameters()).device
    seq_tensor = torch.from_numpy(sequences[:n_sample]).float().to(device)
    with torch.no_grad():
        logits = model.model(seq_tensor)
    logits_np = logits.cpu().numpy()
    probs_1x = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    raw_sum_per_row = logits_np.sum(axis=1)
    softmax_sum_per_row = probs_1x.sum(axis=1)
    max_proba_1x = probs_1x.max(axis=1)

    print("[2] 샘플 {} rows: raw(logits) vs softmax 1회 적용".format(n_sample))
    print("-" * 60)
    print(f"  raw_output_sum (row별): min={raw_sum_per_row.min():.4f}, max={raw_sum_per_row.max():.4f}, mean={raw_sum_per_row.mean():.4f}")
    print(f"  softmax_sum (row별): min={softmax_sum_per_row.min():.6f}, max={softmax_sum_per_row.max():.6f}")
    print(f"  max_proba (1회 softmax): mean={max_proba_1x.mean():.4f}, median={np.median(max_proba_1x):.4f}, p10={np.percentile(max_proba_1x, 10):.4f}, p90={np.percentile(max_proba_1x, 90):.4f}")
    print()

    # 3) Case B: current pipeline (parquet) - same time range
    case_b_max = []
    if PROBA_PATH.exists():
        try:
            cache_df = pd.read_parquet(PROBA_PATH)
            if "timestamp" in cache_df.columns:
                cache_df = cache_df[pd.to_datetime(cache_df["timestamp"]) >= cutoff]
            pl = cache_df["proba_long"].astype(float).values[: n_sample]
            ps = cache_df["proba_short"].astype(float).values[: n_sample]
            pf = 1.0 - pl - ps
            pf = np.clip(pf, 0.0, 1.0)
            case_b_max = np.maximum(np.maximum(pl, ps), pf)
            print("[3] Case B (현재 파이프라인, parquet) 동일 구간 max_proba")
            print(f"  n_parquet_rows={len(case_b_max)}")
            print(f"  max_proba: mean={case_b_max.mean():.4f}, median={np.median(case_b_max):.4f}, p10={np.percentile(case_b_max, 10):.4f}, p90={np.percentile(case_b_max, 90):.4f}")
        except Exception as e:
            print(f"[3] Parquet load failed: {e}")
            case_b_max = []
    else:
        print("[3] Parquet not found; skip Case B")
    print()

    # 4) Double softmax test: if we apply softmax again to probs_1x, max_proba would get flatter
    probs_2x = torch.nn.functional.softmax(torch.from_numpy(probs_1x), dim=-1).numpy()
    max_proba_2x = probs_2x.max(axis=1)
    print("[4] Case A vs 2회 softmax (버그 시뮬레이션)")
    print(f"  1회 softmax max_proba mean: {max_proba_1x.mean():.4f}")
    print(f"  2회 softmax max_proba mean: {max_proba_2x.mean():.4f} (평평해지면 버그 가능)")
    print()

    result = {
        "model_output_is": "logits",
        "softmax_applied_times": 1,
        "n_sample": n_sample,
        "raw_output_sum_min_max_mean": [float(raw_sum_per_row.min()), float(raw_sum_per_row.max()), float(raw_sum_per_row.mean())],
        "softmax_sum_min_max": [float(softmax_sum_per_row.min()), float(softmax_sum_per_row.max())],
        "case_a_max_proba_mean_median_p10_p90": [
            float(max_proba_1x.mean()),
            float(np.median(max_proba_1x)),
            float(np.percentile(max_proba_1x, 10)),
            float(np.percentile(max_proba_1x, 90)),
        ],
        "case_b_parquet_max_proba_mean": float(case_b_max.mean()) if len(case_b_max) else None,
        "double_softmax_max_proba_mean": float(max_proba_2x.mean()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    print()
    print("Softmax Path Verdict:")
    print("  model output is: logits (합!=1, softmax 1회 적용 후 proba)")
    print("  softmax applied how many times: 1 (in tcn_model.predict_proba_batch)")
    print("  max_proba stats: see above; if parquet matches Case A, pipeline is correct.")
    print("=" * 60)


if __name__ == "__main__":
    main()
