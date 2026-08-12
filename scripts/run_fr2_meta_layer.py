#!/usr/bin/env python3
"""
FR2 전략 ON/OFF 자동화 (Meta Layer).

- Primary: B (threshold=0.6, regime=off), Secondary: A (threshold=0.64, regime=off)
- execution: min_hold=24, cooldown=24
- 최근 성과(rolling 60d, 90d) 기반으로 FULL / REDUCED / OFF 상태 자동 전환
- 히스테리시스: 동일 조건 2회 연속 만족 시 전환
- 거래량 보호: rolling trades < 50 → REDUCED

출력: 기본 data/diagnostics/fr2/state_log.csv
      운영 state_log.csv(append 스키마)를 보호하려면 --output 로 state_log_legacy_*.csv 등에 기록.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
COMMISSION = 0.0009
SLIPPAGE = 0.0001
MAX_ENTROPY = 1.30
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8
MIN_HOLD = 24
COOLDOWN = 24
DAYS_FULL = 720

# Primary 전략 B (threshold=0.6, regime=off)
THRESHOLD_PRIMARY = 0.6
REGIME_C4_PRIMARY = False

# 상태 전환 가중치
W_60D = 0.7
W_90D = 0.3
# 구간
SCORE_FULL_MIN = 0.0
SCORE_OFF_MAX = -0.2
ALPHA_FULL_MIN = 1.0
TRADES_MIN = 50
HYSTERESIS_COUNT = 2
POSITION_SCALE_REDUCED = 0.4

EVAL_STEP_DAYS = 30
WINDOW_60D = 60
WINDOW_90D = 90

from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
from scripts.run_fr2_regime_conditioning import add_regime_columns
from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
from src.strategies.ensemble_strategy import (
    EnsembleInputs,
    build_ensemble_proba,
    build_fr2_c4_mask,
)

BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"


def _align_by_timestamp(
    df_base: pd.DataFrame, pl_base: np.ndarray, ps_base: np.ndarray,
    df_fr2: pd.DataFrame, pl_fr2: np.ndarray, ps_fr2: np.ndarray,
):
    df_b = df_base.copy()
    df_f = df_fr2.copy()
    for d in (df_b, df_f):
        d["timestamp"] = pd.to_datetime(d["timestamp"])
    df_b = df_b.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df_f = df_f.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df_b["pl_base"] = pl_base[: len(df_b)]
    df_b["ps_base"] = ps_base[: len(df_b)]
    df_f["pl_fr2"] = pl_fr2[: len(df_f)]
    df_f["ps_fr2"] = ps_fr2[: len(df_f)]
    joined = df_b.merge(df_f[["timestamp", "pl_fr2", "ps_fr2"]], on="timestamp", how="inner")
    joined = joined.sort_values("timestamp").reset_index(drop=True)
    if len(joined) < 500:
        raise RuntimeError("Aligned length too small")
    df_bt = joined[["timestamp", "close", "high", "low"]].copy()
    pl_b = joined["pl_base"].to_numpy(dtype=float)
    ps_b = joined["ps_base"].to_numpy(dtype=float)
    pl_f = joined["pl_fr2"].to_numpy(dtype=float)
    ps_f = joined["ps_fr2"].to_numpy(dtype=float)
    return df_bt, pl_b, ps_b, pl_f, ps_f


def _run_backtest(
    df_w: pd.DataFrame,
    pl: np.ndarray,
    ps: np.ndarray,
    threshold: float,
) -> Tuple[float, float, int]:
    """Returns (cost_on, alpha_fee_ratio, trades)."""
    res_on, _ = run_backtest_7d(
        SYMBOL, TIMEFRAME, df_w, pl, ps,
        commission_rate=COMMISSION, slippage_rate=SLIPPAGE,
        min_max_proba=threshold, max_entropy=MAX_ENTROPY, decision_mode="argmax",
        min_hold=MIN_HOLD, cooldown=COOLDOWN,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
    )
    res_off, _ = run_backtest_7d(
        SYMBOL, TIMEFRAME, df_w, pl, ps,
        commission_rate=0.0, slippage_rate=0.0,
        min_max_proba=threshold, max_entropy=MAX_ENTROPY, decision_mode="argmax",
        min_hold=MIN_HOLD, cooldown=COOLDOWN,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
    )
    if res_on is None:
        return np.nan, np.nan, 0
    cost_on = float(res_on.get("total_return", np.nan))
    trades = int(res_on.get("total_trades", 0))
    cost_off = float(res_off.get("total_return", np.nan)) if res_off else np.nan
    if trades > 0 and res_off is not None:
        alpha_per_trade = cost_off / trades
        fee_per_trade = (cost_off - cost_on) / trades
        afr = alpha_per_trade / fee_per_trade if abs(fee_per_trade) > 1e-12 else np.nan
    else:
        afr = np.nan
    return cost_on, afr, trades


def _raw_state(score: float, alpha_score: float) -> str:
    if score >= SCORE_FULL_MIN and (np.isnan(alpha_score) or alpha_score >= ALPHA_FULL_MIN):
        return "FULL"
    if score >= SCORE_OFF_MAX:
        return "REDUCED"
    return "OFF"


def _eval_dates(df_bt: pd.DataFrame) -> List[pd.Timestamp]:
    t = pd.to_datetime(df_bt["timestamp"])
    t_min, t_max = t.min(), t.max()
    first_eval = t_min + pd.Timedelta(days=WINDOW_90D)
    out = []
    d = first_eval
    while d <= t_max:
        out.append(d)
        d += pd.Timedelta(days=EVAL_STEP_DAYS)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="FR2 meta layer offline state_log generator.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 CSV 경로 (미지정 시 data/diagnostics/fr2/state_log.csv)",
    )
    args = parser.parse_args()

    print("[FR2-META] Loading data (BTC 720d)...", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, BASELINE_PT, "base", False)
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, FR2_PT, "microstructure_v1", True)
    if err_b or err_f or triple_base is None or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba failed: base={err_b}, fr2={err_f}")

    df_b, pl_b, ps_b, _ = triple_base
    df_f, pl_f, ps_f, _ = triple_fr2
    df_bt, pl_b, ps_b, pl_f, ps_f = _align_by_timestamp(df_b, pl_b, ps_b, df_f, pl_f, ps_f)
    df_bt_reg = add_regime_columns(DAYS_FULL, df_bt)
    base_c4 = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)
    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b, ps_base=ps_b, pl_fr2=pl_f, ps_fr2=ps_f, c4_active=c4_mask
    )
    pl_w_full, ps_w_full = build_ensemble_proba(ensemble_inputs, mode="override")

    if REGIME_C4_PRIMARY:
        pl_primary = np.where(c4_mask, pl_w_full, 0.0).astype(np.float32)
        ps_primary = np.where(c4_mask, ps_w_full, 0.0).astype(np.float32)
    else:
        pl_primary = pl_w_full.copy()
        ps_primary = ps_w_full.copy()

    t = pd.to_datetime(df_bt["timestamp"])
    eval_dates = _eval_dates(df_bt)
    print(f"[FR2-META] Eval dates: {len(eval_dates)} (step {EVAL_STEP_DAYS}d)", flush=True)

    state = "FULL"
    pending_state: str | None = None
    consecutive_count = 0
    rows: List[Dict[str, Any]] = []

    for eval_date in eval_dates:
        end_ts = eval_date
        start_90 = eval_date - pd.Timedelta(days=WINDOW_90D)
        start_60 = eval_date - pd.Timedelta(days=WINDOW_60D)
        mask_90 = (t >= start_90) & (t <= end_ts)
        mask_60 = (t >= start_60) & (t <= end_ts)
        if mask_90.sum() < 200 or mask_60.sum() < 200:
            continue

        df_90 = df_bt.loc[mask_90].reset_index(drop=True)
        pl_90 = pl_primary[mask_90]
        ps_90 = ps_primary[mask_90]
        df_60 = df_bt.loc[mask_60].reset_index(drop=True)
        pl_60 = pl_primary[mask_60]
        ps_60 = ps_primary[mask_60]

        cost_on_90d, alpha_90d, trades_90d = _run_backtest(df_90, pl_90, ps_90, THRESHOLD_PRIMARY)
        cost_on_60d, alpha_60d, trades_60d = _run_backtest(df_60, pl_60, ps_60, THRESHOLD_PRIMARY)

        score = W_60D * (cost_on_60d if not np.isnan(cost_on_60d) else 0.0) + W_90D * (cost_on_90d if not np.isnan(cost_on_90d) else 0.0)
        a60 = alpha_60d if not np.isnan(alpha_60d) else 0.0
        a90 = alpha_90d if not np.isnan(alpha_90d) else 0.0
        alpha_score = W_60D * a60 + W_90D * a90

        raw = _raw_state(score, alpha_score)
        if raw != state:
            if pending_state == raw:
                consecutive_count += 1
                if consecutive_count >= HYSTERESIS_COUNT:
                    state = raw
                    pending_state = None
                    consecutive_count = 0
            else:
                pending_state = raw
                consecutive_count = 1
        else:
            pending_state = None
            consecutive_count = 0

        if trades_60d < TRADES_MIN:
            state = "REDUCED"

        position_scale = 1.0 if state == "FULL" else (POSITION_SCALE_REDUCED if state == "REDUCED" else 0.0)

        rows.append({
            "timestamp": eval_date.isoformat(),
            "state": state,
            "cost_on_60d": cost_on_60d,
            "cost_on_90d": cost_on_90d,
            "alpha_60d": alpha_60d,
            "alpha_90d": alpha_90d,
            "score": score,
            "alpha_score": alpha_score,
            "trades_60d": trades_60d,
            "position_scale": position_scale,
        })
        print(f"  {eval_date.date()} score={score:.4f} alpha_score={alpha_score:.3f} trades_60d={trades_60d} -> {state}", flush=True)

    out_df = pd.DataFrame(rows)
    out_df = out_df[[
        "timestamp", "state",
        "cost_on_60d", "cost_on_90d", "alpha_60d", "alpha_90d",
        "score", "alpha_score", "trades_60d", "position_scale",
    ]]
    out_path = Path(args.output) if args.output else OUT_DIR / "state_log.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[FR2-META] Wrote {out_path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
