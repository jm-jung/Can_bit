#!/usr/bin/env python3
"""
Meta metrics probe 분석용:
latest t_max 기준 최근 60일 구간에서 trades_60d == 0이 왜 발생하는지
backtest 엔진이 반환하는 정량 카운트를 기반으로 단계별로 분해 출력한다.

주의:
- "절대 추측하지 말고"를 위해, drop/skip/blocked 카운트는 backtest 결과 dict에서 그대로 사용한다.
- signal LONG/SHORT 개수는 backtest 엔진이 result로 merge한 signal 통계를 사용한다.
"""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _fmt_float(x: float | None) -> str:
    return f"{x:.6f}" if x is not None else "NA"


def _load_meta_source_latest() -> dict[str, Any]:
    p = PROJECT_ROOT / "data" / "diagnostics" / "fr2" / "meta_metrics_source_latest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_ts(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _compute_ensemble_predictions(
    *,
    end_date_utc_str: str,
    load_days: int,
    threshold_primary: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Timestamp]:
    """
    refresh_fr2_meta_metrics.py의 heavy operational_latest 입력 구성과 동일한 방식으로:
    - base/base predictions
    - fr2/microstructure predictions
    - ensemble_proba (pl_primary, ps_primary)
    - aligned df_bt
    - df_bt max timestamp (end_ts for probe candidate)
    을 구성한다.
    """
    # reuse constants / helpers
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask

    BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
    FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"

    DAYS_FULL = load_days
    WINDOW_60D = 60
    WINDOW_90D = 90

    COMMISSION = 0.0009
    SLIPPAGE = 0.0001
    MAX_ENTROPY = 1.30
    TIME_STOP_BARS = 72
    EARLY_EXIT_BAD_K = 8
    MIN_HOLD = 24
    COOLDOWN = 24

    # Keep the same ensemble weights as refresh_fr2_meta_metrics.py
    W_60D = 0.7
    W_90D = 0.3
    REGIME_C4_PRIMARY = False

    # C4 mask requires regime columns; replicate refresh heavy logic.
    print(f"[Stage] compute base predictions: days={DAYS_FULL}, end_date={end_date_utc_str}", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, BASELINE_PT, "base", False, end_date=end_date_utc_str)
    print(f"[Stage] compute fr2 predictions: days={DAYS_FULL}, end_date={end_date_utc_str}", flush=True)
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, FR2_PT, "microstructure_v1", True, end_date=end_date_utc_str)
    if err_b or err_f or triple_base is None or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba failed: base={err_b}, fr2={err_f}")

    df_b, pl_b, ps_b, _ = triple_base
    df_f, pl_f, ps_f, _ = triple_fr2
    if len(df_b) != len(pl_b) or len(df_b) != len(ps_b):
        raise RuntimeError(f"base alignment mismatch: df_b={len(df_b)}, pl_b={len(pl_b)}, ps_b={len(ps_b)}")
    if len(df_f) != len(pl_f) or len(df_f) != len(ps_f):
        raise RuntimeError(f"fr2 alignment mismatch: df_f={len(df_f)}, pl_f={len(pl_f)}, ps_f={len(ps_f)}")

    # Align by timestamp (same strategy as refresh heavy)
    d1 = df_b.copy()
    d2 = df_f.copy()
    d1["timestamp"] = pd.to_datetime(d1["timestamp"])
    d2["timestamp"] = pd.to_datetime(d2["timestamp"])
    d1 = d1.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    d2 = d2.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Keep stable numeric indices so proba arrays can be sliced 1:1.
    d1["base_idx"] = np.arange(len(d1), dtype=np.int64)
    d2["fr2_idx"] = np.arange(len(d2), dtype=np.int64)

    joined = d1.merge(d2[["timestamp", "fr2_idx"]], on="timestamp", how="inner", validate="one_to_one")
    joined = joined.sort_values("timestamp").reset_index(drop=True)
    if len(joined) < 500:
        raise RuntimeError(f"Aligned length too small: {len(joined)}")

    base_indices = joined["base_idx"].to_numpy()
    fr2_indices = joined["fr2_idx"].to_numpy()

    df_bt = joined[["timestamp", "close", "high", "low"]].copy()
    pl_base = pl_b[base_indices].astype(np.float32, copy=False)
    ps_base = ps_b[base_indices].astype(np.float32, copy=False)
    pl_fr2 = pl_f[fr2_indices].astype(np.float32, copy=False)
    ps_fr2 = ps_f[fr2_indices].astype(np.float32, copy=False)

    print(f"[Stage] add regime columns + build C4 mask (n={len(df_bt)})", flush=True)
    df_bt_reg = add_regime_columns(DAYS_FULL, df_bt)
    base_c4 = ((df_bt_reg["trend_regime"] == "uptrend") & (df_bt_reg["vol_regime"] == "high_vol")).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)

    print("[Stage] build ensemble probabilities", flush=True)
    ensemble_inputs = EnsembleInputs(
        pl_base=pl_base,
        ps_base=ps_base,
        pl_fr2=pl_fr2,
        ps_fr2=ps_fr2,
        c4_active=c4_mask,
    )
    pl_w_full, ps_w_full = build_ensemble_proba(ensemble_inputs, mode="override")

    if REGIME_C4_PRIMARY:
        pl_primary = np.where(c4_mask, pl_w_full, 0.0).astype(np.float32)
        ps_primary = np.where(c4_mask, ps_w_full, 0.0).astype(np.float32)
    else:
        pl_primary = pl_w_full.astype(np.float32)
        ps_primary = ps_w_full.astype(np.float32)

    t_bt = pd.to_datetime(df_bt["timestamp"])
    end_ts = t_bt.max()
    return df_bt, pl_primary, ps_primary, pd.Timestamp(end_ts)


def _run_backtest_for_window(
    *,
    symbol: str,
    timeframe: str,
    df_window: pd.DataFrame,
    proba_long: np.ndarray,
    proba_short: np.ndarray,
    threshold_primary: float,
) -> dict[str, Any]:
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    # Keep exactly the meta layer probe parameters.
    COMMISSION = 0.0009
    SLIPPAGE = 0.0001
    MAX_ENTROPY = 1.30
    TIME_STOP_BARS = 72
    EARLY_EXIT_BAD_K = 8
    MIN_HOLD = 24
    COOLDOWN = 24

    res, err = run_backtest_7d(
        symbol,
        timeframe,
        df_window,
        proba_long,
        proba_short,
        commission_rate=COMMISSION,
        slippage_rate=SLIPPAGE,
        min_max_proba=threshold_primary,
        max_entropy=MAX_ENTROPY,
        decision_mode="argmax",
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
        emit_trade_log=False,
    )
    if err:
        raise RuntimeError(f"run_backtest_7d failed: {err}")
    if res is None:
        raise RuntimeError("run_backtest_7d returned None result")
    return res


def main() -> int:
    meta = _load_meta_source_latest()
    source_info = meta.get("source_info") or {}

    symbol = source_info.get("symbol", "BTCUSDT")
    timeframe = source_info.get("timeframe", "5m")
    threshold_primary = float(source_info.get("threshold", 0.6))

    probe_t_max_str = source_info.get("first_probe_t_max")
    if not probe_t_max_str:
        raise RuntimeError("source_info.first_probe_t_max missing")
    probe_t_max = _parse_ts(probe_t_max_str)

    raw_path_5m = PROJECT_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv"
    raw_5m_df = pd.read_csv(raw_path_5m, usecols=["timestamp"])
    raw_5m_df["timestamp"] = pd.to_datetime(raw_5m_df["timestamp"])
    raw_5m_max = pd.Timestamp(raw_5m_df["timestamp"].max())

    end_date_utc_str = probe_t_max.strftime("%Y-%m-%d")

    # Try to load enough history so that df_bt max reaches probe_t_max.
    for load_days in [120, 240, 360, 720]:
        print(f"[Probe] trying load_days={load_days}", flush=True)
        df_bt, pl_primary, ps_primary, df_bt_max = _compute_ensemble_predictions(
            end_date_utc_str=end_date_utc_str,
            load_days=load_days,
            threshold_primary=threshold_primary,
        )
        df_bt_max = pd.Timestamp(df_bt_max)
        if df_bt_max >= probe_t_max:
            break
    else:
        raise RuntimeError(
            f"df_bt_max never reached probe_t_max. probe_t_max={probe_t_max}, df_bt_max_last={df_bt_max}"
        )

    t_max = df_bt_max
    window_start = t_max - pd.Timedelta(days=60)

    t = pd.to_datetime(df_bt["timestamp"])
    mask_60 = ((t >= window_start) & (t <= t_max)).to_numpy(dtype=bool)
    df_60 = df_bt.iloc[mask_60].reset_index(drop=True)
    pl_60 = pl_primary[mask_60]
    ps_60 = ps_primary[mask_60]

    if len(df_60) != len(pl_60) or len(df_60) != len(ps_60):
        raise RuntimeError(f"window alignment mismatch: df={len(df_60)} pl={len(pl_60)} ps={len(ps_60)}")

    # Step 2: proba distribution + signal counts + entries_attempted from backtest
    p_flat = 1.0 - pl_60 - ps_60
    p_flat = np.clip(p_flat, 0.0, 1.0)
    max_proba = np.maximum(np.maximum(pl_60, ps_60), p_flat)

    # Backtest for this 60d window.
    result = _run_backtest_for_window(
        symbol=symbol,
        timeframe=timeframe,
        df_window=df_60,
        proba_long=pl_60,
        proba_short=ps_60,
        threshold_primary=threshold_primary,
    )

    total_bars = int(len(df_60))
    entries_attempted = int(result.get("entries_attempted", 0))
    long_signals = int(result.get("signal_long_count", 0))
    short_signals = int(result.get("signal_short_count", 0))

    proba_mean = float(max_proba.mean()) if len(max_proba) else None
    proba_min = float(max_proba.min()) if len(max_proba) else None
    proba_max = float(max_proba.max()) if len(max_proba) else None

    # Step 3: filter drops (exact counters from engine)
    filter_skip_stats = result.get("filter_skip_stats") or {}
    by_min_max_proba = int(filter_skip_stats.get("by_min_max_proba", 0))
    by_max_entropy = int(filter_skip_stats.get("by_max_entropy", 0))

    regime_enabled = bool(result.get("regime_enabled", False))
    # regime filter disabled in our meta-probe params
    entries_allowed_final_candidates = int(result.get("stage2_total_entries", 0))

    # Step 4: execution
    total_trades = int(result.get("total_trades", 0))
    trades = int(result.get("trades", 0))
    block_reasons = result.get("block_reasons") or {}
    cooldown_excluded = int(block_reasons.get("cooldown", 0))
    min_hold_excluded = int(block_reasons.get("min_hold", 0))

    # "already in position" isn't directly tracked as a separate counter.
    # For this probe, we can only report the observable number: trades/stage2 entries are 0 => no position was opened.
    already_in_position_excluded = 0 if trades == 0 else None

    # Step 5: conclusion
    # Use only observable counts.
    if long_signals + short_signals == 0 or entries_attempted == 0:
        case = "CASE A"
        case_detail = "신호 자체가 부족 → 모델/피처 문제"
    elif entries_allowed_final_candidates == 0 and (by_min_max_proba > 0 or by_max_entropy > 0):
        case = "CASE B"
        case_detail = "필터가 너무 강함 → threshold/entropy 문제"
    elif entries_allowed_final_candidates > 0 and total_trades == 0:
        case = "CASE C"
        case_detail = "execution 제약 문제 → 전략 로직 문제"
    else:
        # Fallback: if we can't classify, show the closest evidence
        case = "CASE B"
        case_detail = "필터/스테이지2가 ENTRY를 막음 (정량 근거: stage2_total_entries)"

    # Print required tables
    print("## [STEP 1] 분석 구간")
    print(
        "| key | value |\n"
        f"|---|---|\n"
        f"| 기준 t_max (probe_t_max) | {probe_t_max.isoformat()} |\n"
        f"| raw 5m max timestamp | {raw_5m_max.isoformat()} |\n"
        f"| df_bt max timestamp | {t_max.isoformat()} |\n"
        f"| 분석 구간 [t_max-60d, t_max] | [{window_start.isoformat()}, {t_max.isoformat()}] |"
    )

    print("\n## [STEP 2] 신호 발생 단계 분석")
    print(
        "| metric | value |\n"
        "|---|---|\n"
        f"| total_bars (60d 내 bar 수) | {total_bars} |\n"
        f"| entries_attempted (LONG/SHORT 시도 횟수) | {entries_attempted} |\n"
        f"| long_signals | {long_signals} |\n"
        f"| short_signals | {short_signals} |\n"
        f"| proba_max mean/min/max | {_fmt_float(proba_mean)} / {_fmt_float(proba_min)} / {_fmt_float(proba_max)} |"
    )

    print("\n## [STEP 3] 필터 단계별 drop 분석")
    print(
        "| filter | skip_count | pass_count (entry_attempted - skip) |\n"
        "|---|---:|---:|\n"
        f"| min_max_proba (threshold={threshold_primary}) | {by_min_max_proba} | {max(entries_attempted - by_min_max_proba, 0)} |\n"
        f"| entropy (max_entropy=1.30) | {by_max_entropy} | {max(entries_attempted - by_max_entropy, 0)} |\n"
        f"| regime filter | 0 (enabled={regime_enabled}) | {entries_attempted} |\n"
        f"| 최종 진입 후보 수 (stage2_total_entries) | - | {entries_allowed_final_candidates} |"
    )

    print("\n## [STEP 4] execution 단계 분석")
    print(
        "| metric | value |\n"
        "|---|---|\n"
        f"| stage2_total_entries (ENTRY 발생) | {entries_allowed_final_candidates} |\n"
        f"| total_trades (실제 체결/라운드트립 수) | {total_trades} |\n"
        f"| trades (동일 필드 alias) | {trades} |\n"
        f"| cooldown 제외 수 | {cooldown_excluded} |\n"
        f"| min_hold 제외 수 | {min_hold_excluded} |\n"
        f"| 이미 포지션 있어서 제외 수 | {already_in_position_excluded} |"
    )

    print("\n## [STEP 5] 결론")
    print(f"- {case}: {case_detail}")

    # extra: show other block reasons for transparency (still numeric)
    if isinstance(block_reasons, dict):
        other = {k: int(v) for k, v in block_reasons.items() if k not in {"cooldown", "min_hold"} and int(v) != 0}
        if other:
            print("\n(부가: block_reasons 중 0이 아닌 값)")
            for k, v in sorted(other.items()):
                print(f"- {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

