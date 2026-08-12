#!/usr/bin/env python3
"""
고정 t_max + 최근 60d window에서
ensemble proba(p_long, p_short)를 만든 뒤
current_max_proba = max(p_long, p_short, p_flat) 분포를 숫자로 출력.

출력 예:
  n_bars, max_proba mean/min/max, quantiles
  threshold별 max_proba >= th 비율
  rounded(2dp) 기준 값 빈도(대개 0.4/0.6 같이 이산화될 때 확인용)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_meta_source_latest() -> dict[str, Any]:
    p = PROJECT_ROOT / "data" / "diagnostics" / "fr2" / "meta_metrics_source_latest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def main() -> int:
    # ✅ 사용자 요구: t_max 변경 금지 (이전 지정값 고정)
    t_max = _to_utc_ts("2026-03-20T07:40:00+00:00")

    window_days = 60
    window_start = t_max - pd.Timedelta(days=window_days)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    # fast sweep와 동일한 모델/파이프라인 사용
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask

    BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
    FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"

    # fast sweep와 동일한 여유(60d slice + window_size/horizon/기본 계산 여유)
    DAYS_FULL = 62

    # fast sweep는 end_date를 "YYYY-MM-DD"로 넘김
    end_date_utc_str = t_max.strftime("%Y-%m-%d")

    # proba 생성
    print(f"[INSPECT] t_max={t_max.isoformat()}", flush=True)
    print(f"[INSPECT] window_start={window_start.isoformat()} window_end={t_max.isoformat()}", flush=True)
    print(f"[INSPECT] DAYS_FULL={DAYS_FULL}", flush=True)

    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, BASELINE_PT, "base", False, end_date=end_date_utc_str)
    if err_b or triple_base is None:
        raise RuntimeError(f"get_ohlcv_and_proba(base) failed: {err_b}")
    df_b, pl_b, ps_b, _ = triple_base

    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, FR2_PT, "microstructure_v1", True, end_date=end_date_utc_str)
    if err_f or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba(fr2) failed: {err_f}")
    df_f, pl_f, ps_f, _ = triple_fr2

    # timestamp alignment (fast sweep와 동일한 join 방식)
    d1 = df_b.copy()
    d2 = df_f.copy()
    d1["timestamp"] = pd.to_datetime(d1["timestamp"])
    d2["timestamp"] = pd.to_datetime(d2["timestamp"])

    d1["pl_base"] = np.asarray(pl_b, dtype=np.float32)
    d1["ps_base"] = np.asarray(ps_b, dtype=np.float32)
    d2["pl_fr2"] = np.asarray(pl_f, dtype=np.float32)
    d2["ps_fr2"] = np.asarray(ps_f, dtype=np.float32)

    joined = d1.merge(d2[["timestamp", "pl_fr2", "ps_fr2"]], on="timestamp", how="inner", validate="one_to_one")
    joined = joined.sort_values("timestamp").reset_index(drop=True)

    df_bt = joined[["timestamp", "close", "high", "low"]].copy()
    pl_base = joined["pl_base"].to_numpy(dtype=np.float32)
    ps_base = joined["ps_base"].to_numpy(dtype=np.float32)
    pl_fr2 = joined["pl_fr2"].to_numpy(dtype=np.float32)
    ps_fr2 = joined["ps_fr2"].to_numpy(dtype=np.float32)

    # regime + C4 mask
    df_bt_reg = add_regime_columns(DAYS_FULL, df_bt.copy())
    base_c4 = ((df_bt_reg["trend_regime"] == "uptrend") & (df_bt_reg["vol_regime"] == "high_vol")).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)

    ensemble_inputs = EnsembleInputs(
        pl_base=pl_base,
        ps_base=ps_base,
        pl_fr2=pl_fr2,
        ps_fr2=ps_fr2,
        c4_active=c4_mask,
    )
    pl_primary, ps_primary = build_ensemble_proba(ensemble_inputs, mode="override")

    # slice 60d
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    mask = (t >= window_start) & (t <= t_max)
    n = int(mask.sum())
    if n <= 0:
        raise RuntimeError(f"[INSPECT] window slice is empty (n={n})")

    pl = np.asarray(pl_primary, dtype=np.float32)[mask.to_numpy()]
    ps = np.asarray(ps_primary, dtype=np.float32)[mask.to_numpy()]
    pf = np.clip(1.0 - pl - ps, 0.0, 1.0)

    mp = np.maximum(np.maximum(pl, ps), pf)

    # stats
    print(f"[INSPECT] n_bars(window)={n}", flush=True)
    print(f"[INSPECT] max_proba mean={float(mp.mean()):.6f} min={float(mp.min()):.6f} max={float(mp.max()):.6f}", flush=True)
    qs = np.quantile(mp, [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]).tolist()
    labels = ["min", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
    print("[INSPECT] quantiles:", flush=True)
    for lab, qv in zip(labels, qs):
        print(f"  {lab}={float(qv):.6f}", flush=True)

    # threshold pass rates
    print("[INSPECT] pass rates (max_proba >= th):", flush=True)
    for th in thresholds:
        rate = float((mp >= th).mean())
        cnt = int((mp >= th).sum())
        print(f"  th={th:.2f}: count={cnt}/{n} rate={rate:.4f}", flush=True)

    # discrete value check (rounded)
    rounded = np.round(mp, 2)
    uniq, counts = np.unique(rounded, return_counts=True)
    # print top by count
    order = np.argsort(-counts)
    print("[INSPECT] rounded(max_proba,2dp) top-10 by frequency:", flush=True)
    for idx in order[:10]:
        print(f"  {float(uniq[idx]):.2f}: {int(counts[idx])}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

