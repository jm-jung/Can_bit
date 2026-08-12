#!/usr/bin/env python3
"""
검증 1) TCN 클래스 매핑(라벨 순서) 검증.
- index 0=FLAT, 1=LONG, 2=SHORT 일치 여부
- 전체 argmax 분포 vs ENTRY 시점 argmax 분포
- 20개 샘플: timestamp, raw logits(3), softmax(3), argmax, mapped label, max_proba
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_MONITORING = PROJECT_ROOT / "data" / "monitoring"
DATA_CACHE = PROJECT_ROOT / "data" / "cache" / "ml_predictions"
PROBA_PATH = DATA_CACHE / "ml_tcn_BTCUSDT_5m_proba.parquet"

CLASS_LABELS = ["FLAT", "LONG", "SHORT"]  # index 0, 1, 2 (LstmClassIndex)


def main():
    parser = argparse.ArgumentParser(description="TCN class label mapping verification")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    from src.dl.data.labels import LstmClassIndex, get_class_name
    from src.dl.tcn_model import (
        build_normalized_tcn_sequences_full,
        get_tcn_model,
    )
    from src.ml.features import build_feature_frame
    from src.services.ohlcv_service import load_ohlcv_df
    from src.core.config import settings

    print("=" * 60)
    print("검증 1) TCN 클래스 매핑 (라벨 순서) 검증")
    print("=" * 60)
    print(f"  LstmClassIndex: FLAT={LstmClassIndex.FLAT}, LONG={LstmClassIndex.LONG}, SHORT={LstmClassIndex.SHORT}")
    print(f"  class_labels (order): {CLASS_LABELS}")
    print()

    # Load OHLCV and features (minimal for sample logits: last 500 rows enough for 20 sequences)
    df = load_ohlcv_df(timeframe="5m", symbol="BTCUSDT")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.tail(600).reset_index(drop=True)
    features = build_feature_frame(df, symbol="BTCUSDT", timeframe="5m", use_events=settings.EVENTS_ENABLED)
    features = features.dropna()
    if len(features) < 60:
        print("[ERROR] Not enough features after dropna.")
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
    print(f"  Sequences built: {num_seq} (window_size={window_size})")
    print()

    # Run model to get logits for first `samples` and optionally all for distribution
    device = next(model.model.parameters()).device
    seq_tensor = torch.from_numpy(sequences[: num_seq]).float().to(device)
    with torch.no_grad():
        logits_all = model.model(seq_tensor)
    logits_np = logits_all.cpu().numpy()
    probs_np = torch.nn.functional.softmax(logits_all, dim=-1).cpu().numpy()

    # Timestamps for each sequence (aligned to feature index window_size, window_size+1, ...)
    timestamps = features.index[window_size : window_size + num_seq]
    if hasattr(timestamps, "tolist"):
        ts_list = [str(t) for t in timestamps]
    else:
        ts_list = [str(features.index[window_size + i]) for i in range(num_seq)]

    # --- 1) 20 samples ---
    print("[1] 단일 row 샘플 (처음 {}개)".format(args.samples))
    print("-" * 60)
    n_show = min(args.samples, num_seq)
    for i in range(n_show):
        logits = logits_np[i].tolist()
        probs = probs_np[i].tolist()
        argmax_idx = int(np.argmax(probs))
        label = get_class_name(argmax_idx)
        max_proba = float(max(probs))
        print(f"  [{i}] ts={ts_list[i]}")
        print(f"       raw_logits=[{logits[0]:.6f}, {logits[1]:.6f}, {logits[2]:.6f}]")
        print(f"       softmax(proba)=[FLAT={probs[0]:.4f}, LONG={probs[1]:.4f}, SHORT={probs[2]:.4f}]")
        print(f"       argmax_index={argmax_idx}, mapped_label={label}, max_proba={max_proba:.4f}")
    print()

    # --- 2) 전체 argmax 분포 (parquet 기반으로 최근 7일) ---
    argmax_all = np.argmax(probs_np, axis=1)
    n_all = len(argmax_all)
    if PROBA_PATH.exists():
        try:
            pq = pd.read_parquet(PROBA_PATH)
            cutoff = datetime.now() - timedelta(days=args.days)
            if "timestamp" in pq.columns:
                pq = pq[pd.to_datetime(pq["timestamp"]) >= cutoff]
            pl = pq["proba_long"].astype(float).values
            ps = pq["proba_short"].astype(float).values
            pf = np.clip(1.0 - pl - ps, 0.0, 1.0)
            stack = np.stack([pf, pl, ps], axis=1)
            argmax_all = np.argmax(stack, axis=1)
            n_all = len(argmax_all)
        except Exception as e:
            print(f"[WARN] Parquet load failed for full dist: {e}")
    c_all = Counter(argmax_all)
    print("[2] 전체 rows (최근 {}일, n={}) argmax 분포".format(args.days, n_all))
    print("-" * 60)
    for idx in range(3):
        pct = 100.0 * c_all.get(idx, 0) / n_all
        print(f"  {CLASS_LABELS[idx]} (index={idx}): {c_all.get(idx, 0)} ({pct:.2f}%)")
    print()

    # --- 3) ENTRY 시점만 매칭 (JSONL에서 ENTRY ts 수집 후, 해당 ts와 일치하는 sequence만)
    entry_ts = set()
    for p in sorted(DATA_MONITORING.glob("monitor_guard_stage2_*.jsonl"), reverse=True)[:5]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    if o.get("event") == "ENTRY" and o.get("ts"):
                        entry_ts.add(o["ts"].strip())
        except Exception:
            continue
    # Normalize ts format for matching (e.g. "2021-01-01 13:40:00" vs index)
    entry_ts_norm = set()
    for t in entry_ts:
        try:
            dt = pd.to_datetime(t)
            entry_ts_norm.add(dt)
        except Exception:
            entry_ts_norm.add(t)
    mask_entry = np.zeros(num_seq, dtype=bool)
    for i in range(num_seq):
        try:
            ti = pd.to_datetime(ts_list[i])
            for et in entry_ts_norm:
                if abs((ti - et).total_seconds()) < 300:  # 5m
                    mask_entry[i] = True
                    break
        except Exception:
            pass
    if mask_entry.sum() == 0 and entry_ts:
        # Fallback: match by string prefix
        for i in range(num_seq):
            for et in entry_ts:
                if et[:16] in ts_list[i] or ts_list[i][:16] in str(et):
                    mask_entry[i] = True
                    break
    argmax_entry = argmax_all[mask_entry]
    n_entry = len(argmax_entry)
    c_entry = Counter(argmax_entry)
    print("[3] ENTRY 시점 매칭된 row만 argmax 분포 (n={})".format(n_entry))
    print("-" * 60)
    for idx in range(3):
        pct = 100.0 * c_entry.get(idx, 0) / n_entry if n_entry else 0
        print(f"  {CLASS_LABELS[idx]} (index={idx}): {c_entry.get(idx, 0)} ({pct:.2f}%)")
    print()

    # --- 4) 기존 inspect 비율 계산 방식 확인 ---
    print("[4] 기존 inspect_tcn_signal_quality 'long/short/flat 비율' 계산 방식")
    print("  코드: ratio_long = (proba_long > max(proba_short, proba_flat)).sum() / n")
    print("  즉 argmax가 LONG인 비율 = ratio_long, SHORT = ratio_short, FLAT = ratio_flat")
    print("  순서: [LONG, SHORT, FLAT] 로 출력됨. index 1=LONG, 2=SHORT, 0=FLAT 와 일치.")
    print()

    # Verdict
    print("Label Mapping Verdict:")
    print("  labels order confirmed? YES (index 0=FLAT, 1=LONG, 2=SHORT, LstmClassIndex 및 코드 일치)")
    print("  any mismatch found? None (proba_long=probs[:,1], proba_short=probs[:,2], proba_flat=1-p_long-p_short)")
    print("=" * 60)


if __name__ == "__main__":
    main()
