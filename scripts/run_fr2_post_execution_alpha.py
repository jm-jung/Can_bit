#!/usr/bin/env python3
"""
FR2 post-execution alpha improvement 실험.

목적: execution 고정(min_hold=24, cooldown=24) 상태에서 alpha_per_trade를 올려
      alpha_fee_ratio ≥ 1 또는 cost_on ≥ 0 도달 가능 여부 검증.

실험 순서 (한 레버씩):
  1) threshold 구간 탐색 (0.60 ~ 0.70 step 0.02)
  2) regime binary (OFF vs C4 only, high_vol only)
  3) position sizing sweep (p_floor × p_full)

고정: strategy=override_ensemble, min_hold=24, cooldown=24.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

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
STRATEGY = "override_ensemble"
WINDOW_DAYS = 720
MIN_HOLD = 24
COOLDOWN = 24

# Threshold sweep (구간 탐색)
THRESHOLD_VALUES = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]

# Regime: binary 먼저
REGIME_OFF = "off"
REGIME_C4_ONLY = "c4_only"
REGIME_HIGH_VOL_ONLY = "high_vol_only"
REGIME_VALUES = [REGIME_OFF, REGIME_C4_ONLY, REGIME_HIGH_VOL_ONLY]

# Position sizing sweep
POSITION_P_FLOOR_VALUES = [0.2, 0.3, 0.4]
POSITION_P_FULL_VALUES = [0.6, 0.7, 0.8]

from scripts.run_fr2_diagnostics import (
    DAYS,
    MODELS_DIR,
    get_ohlcv_and_proba,
)
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
    df_base: pd.DataFrame,
    pl_base: np.ndarray,
    ps_base: np.ndarray,
    df_fr2: pd.DataFrame,
    pl_fr2: np.ndarray,
    ps_fr2: np.ndarray,
):
    df_b = df_base.copy()
    df_f = df_fr2.copy()
    for df in (df_b, df_f):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
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


def _window_mask(df_bt: pd.DataFrame, window_days: int) -> np.ndarray:
    t = pd.to_datetime(df_bt["timestamp"])
    t_max = t.max()
    if window_days >= 720:
        return np.ones(len(df_bt), dtype=bool)
    start = t_max - pd.Timedelta(days=window_days)
    return (t >= start).to_numpy()


def _bar_index_for_time(df_bt: pd.DataFrame, time_str: str | None) -> int | None:
    if not time_str:
        return None
    ts = pd.to_datetime(df_bt["timestamp"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    if len(ts) == 0:
        return None
    t = pd.to_datetime(time_str, errors="coerce")
    if pd.isna(t):
        return None
    t64 = t.to_datetime64()
    i = int(np.searchsorted(ts, t64, side="left"))
    if i <= 0:
        return 0
    if i >= len(ts):
        return len(ts) - 1
    prev_dt = abs(ts[i - 1] - t64)
    next_dt = abs(ts[i] - t64)
    return i - 1 if prev_dt <= next_dt else i


def _avg_gap_between_trades(trades: List[Any], df_bt: pd.DataFrame) -> float:
    """연속 trade 사이 평균 bar 간격 (exit_i ~ entry_{i+1})."""
    if not trades or len(trades) < 2:
        return np.nan
    # sort by entry_time
    def _entry_ts(t):
        if isinstance(t, dict):
            return t.get("entry_time") or ""
        return getattr(t, "entry_time", "") or ""

    sorted_trades = sorted(trades, key=_entry_ts)
    gaps = []
    for i in range(len(sorted_trades) - 1):
        t_cur = sorted_trades[i]
        t_next = sorted_trades[i + 1]
        exit_time = t_cur.get("exit_time") if isinstance(t_cur, dict) else getattr(t_cur, "exit_time", None)
        entry_next = t_next.get("entry_time") if isinstance(t_next, dict) else getattr(t_next, "entry_time", None)
        exit_bar = _bar_index_for_time(df_bt, exit_time)
        entry_bar = _bar_index_for_time(df_bt, entry_next)
        if exit_bar is not None and entry_bar is not None:
            gap = max(0, entry_bar - exit_bar)
            gaps.append(float(gap))
    if not gaps:
        return np.nan
    return float(np.mean(gaps))


def _compute_holding_bars_from_times(
    df_bt: pd.DataFrame,
    entry_time: str | None,
    exit_time: str | None,
) -> float | None:
    if not entry_time or not exit_time:
        return None
    ts = pd.to_datetime(df_bt["timestamp"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    if len(ts) == 0:
        return None
    e = pd.to_datetime(entry_time, errors="coerce")
    x = pd.to_datetime(exit_time, errors="coerce")
    if pd.isna(e) or pd.isna(x):
        return None
    e64 = e.to_datetime64()
    x64 = x.to_datetime64()

    def _nearest_idx(t64) -> int:
        i = int(np.searchsorted(ts, t64, side="left"))
        if i <= 0:
            return 0
        if i >= len(ts):
            return len(ts) - 1
        prev_dt = abs(ts[i - 1] - t64)
        next_dt = abs(ts[i] - t64)
        return i - 1 if prev_dt <= next_dt else i

    ei = _nearest_idx(e64)
    xi = _nearest_idx(x64)
    return float(max(0, xi - ei))


def _extract_trade_stats(trades: List[Any], df_bt: pd.DataFrame) -> Dict[str, float]:
    if not trades:
        return {
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "mean_holding_bars": np.nan,
            "median_holding_bars": np.nan,
        }
    holdings = []
    for t in trades:
        h = t.get("holding_bars") if isinstance(t, dict) else getattr(t, "holding_bars", None)
        if h is None:
            h = t.get("bars_held") if isinstance(t, dict) else getattr(t, "bars_held", None)
        if h is None:
            entry_time = t.get("entry_time") if isinstance(t, dict) else getattr(t, "entry_time", None)
            exit_time = t.get("exit_time") if isinstance(t, dict) else getattr(t, "exit_time", None)
            h = _compute_holding_bars_from_times(df_bt, entry_time, exit_time)
        if h is not None:
            holdings.append(float(h))
    pnls = []
    for t in trades:
        p = t.get("profit") if isinstance(t, dict) else getattr(t, "profit", None)
        if p is None:
            p = t.get("net_return", t.get("pnl", 0.0))
        pnls.append(float(p))
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else np.nan
    mean_hold = float(np.mean(holdings)) if holdings else np.nan
    median_hold = float(np.median(holdings)) if holdings else np.nan
    return {
        "win_rate": float(len(wins) / len(pnls)),
        "profit_factor": float(pf) if not np.isnan(pf) else np.nan,
        "mean_holding_bars": mean_hold,
        "median_holding_bars": median_hold,
    }


def _run_one(
    df_w: pd.DataFrame,
    pl: np.ndarray,
    ps: np.ndarray,
    threshold: float,
    position_scaling_enabled: bool,
    position_p_floor: float,
    position_p_full: float,
) -> tuple[dict | None, dict | None]:
    """Run backtest with fee and without. Returns (res_on, res_off)."""
    res_on, _ = run_backtest_7d(
        SYMBOL,
        TIMEFRAME,
        df_w,
        pl,
        ps,
        commission_rate=COMMISSION,
        slippage_rate=SLIPPAGE,
        min_max_proba=threshold,
        max_entropy=MAX_ENTROPY,
        decision_mode="argmax",
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
        position_scaling_enabled=position_scaling_enabled,
        position_p_floor=position_p_floor,
        position_p_full=position_p_full,
    )
    res_off, _ = run_backtest_7d(
        SYMBOL,
        TIMEFRAME,
        df_w,
        pl,
        ps,
        commission_rate=0.0,
        slippage_rate=0.0,
        min_max_proba=threshold,
        max_entropy=MAX_ENTROPY,
        decision_mode="argmax",
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
        position_scaling_enabled=position_scaling_enabled,
        position_p_floor=position_p_floor,
        position_p_full=position_p_full,
    )
    return (res_on, res_off)


def main() -> None:
    print("[FR2-POST] Loading data...", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS, BASELINE_PT, "base", False)
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS, FR2_PT, "microstructure_v1", True)
    if err_b or err_f or triple_base is None or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba failed: base={err_b}, fr2={err_f}")

    df_b, pl_b, ps_b, _ = triple_base
    df_f, pl_f, ps_f, _ = triple_fr2
    df_bt, pl_b, ps_b, pl_f, ps_f = _align_by_timestamp(df_b, pl_b, ps_b, df_f, pl_f, ps_f)
    df_bt_reg = add_regime_columns(DAYS, df_bt)
    base_c4 = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)
    high_vol_mask = (df_bt_reg["vol_regime"] == "high_vol").fillna(False).to_numpy()

    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b, ps_base=ps_b, pl_fr2=pl_f, ps_fr2=ps_f, c4_active=c4_mask
    )
    pl_w_full, ps_w_full = build_ensemble_proba(ensemble_inputs, mode="override")

    mask = _window_mask(df_bt, WINDOW_DAYS)
    n_bars = int(mask.sum())
    df_w = df_bt.loc[mask].reset_index(drop=True)
    pl_w = pl_w_full[mask].copy()
    ps_w = ps_w_full[mask].copy()
    c4_w = c4_mask[mask]
    high_vol_w = high_vol_mask[mask]

    rows: List[Dict] = []

    # --- 1) Threshold sweep (regime=off, position_scaling=off) ---
    print("[FR2-POST] 1) Threshold sweep...", flush=True)
    for th in THRESHOLD_VALUES:
        res_on, res_off = _run_one(
            df_w, pl_w, ps_w,
            threshold=th,
            position_scaling_enabled=False,
            position_p_floor=0.55,
            position_p_full=0.65,
        )
        row = _row_from_results(
            res_on, res_off, df_w, n_bars,
            strategy=STRATEGY, threshold=th, regime=REGIME_OFF,
            position_scaling="off", position_p_floor=None, position_p_full=None,
        )
        rows.append(row)
        print(f"  threshold={th} -> trades={row['trades']} cost_on={row['cost_on']:.4f} alpha_fee_ratio={row['alpha_fee_ratio']:.3f}", flush=True)

    # --- 2) Regime binary (threshold=0.60, position_scaling=off) ---
    print("[FR2-POST] 2) Regime binary...", flush=True)
    for regime_name in [REGIME_C4_ONLY, REGIME_HIGH_VOL_ONLY]:
        if regime_name == REGIME_C4_ONLY:
            pl_g = np.where(c4_w, pl_w, 0.0).astype(np.float32)
            ps_g = np.where(c4_w, ps_w, 0.0).astype(np.float32)
        else:
            pl_g = np.where(high_vol_w, pl_w, 0.0).astype(np.float32)
            ps_g = np.where(high_vol_w, ps_w, 0.0).astype(np.float32)
        res_on, res_off = _run_one(
            df_w, pl_g, ps_g,
            threshold=0.60,
            position_scaling_enabled=False,
            position_p_floor=0.55,
            position_p_full=0.65,
        )
        row = _row_from_results(
            res_on, res_off, df_w, n_bars,
            strategy=STRATEGY, threshold=0.60, regime=regime_name,
            position_scaling="off", position_p_floor=None, position_p_full=None,
        )
        rows.append(row)
        print(f"  regime={regime_name} -> trades={row['trades']} cost_on={row['cost_on']:.4f} alpha_fee_ratio={row['alpha_fee_ratio']:.3f}", flush=True)

    # --- 3) Position sizing (threshold=0.60, regime=off) ---
    print("[FR2-POST] 3) Position sizing sweep...", flush=True)
    for p_floor in POSITION_P_FLOOR_VALUES:
        for p_full in POSITION_P_FULL_VALUES:
            if p_floor >= p_full:
                continue
            res_on, res_off = _run_one(
                df_w, pl_w, ps_w,
                threshold=0.60,
                position_scaling_enabled=True,
                position_p_floor=p_floor,
                position_p_full=p_full,
            )
            pos_str = f"p_floor={p_floor},p_full={p_full}"
            row = _row_from_results(
                res_on, res_off, df_w, n_bars,
                strategy=STRATEGY, threshold=0.60, regime=REGIME_OFF,
                position_scaling=pos_str, position_p_floor=p_floor, position_p_full=p_full,
            )
            rows.append(row)
            print(f"  {pos_str} -> trades={row['trades']} cost_on={row['cost_on']:.4f} alpha_fee_ratio={row['alpha_fee_ratio']:.3f}", flush=True)

    out_df = pd.DataFrame(rows)
    out_df = out_df[[
        "strategy", "threshold", "regime", "position_scaling",
        "trades", "cost_on", "cost_off",
        "alpha_per_trade", "fee_per_trade", "alpha_fee_ratio",
        "mean_holding_bars", "avg_gap_between_trades",
    ]]
    out_df.to_csv(OUT_DIR / "fr2_post_execution_alpha.csv", index=False)
    print(f"[FR2-POST] Wrote fr2_post_execution_alpha.csv ({len(rows)} rows)", flush=True)


def _row_from_results(
    res_on: dict | None,
    res_off: dict | None,
    df_w: pd.DataFrame,
    n_bars: int,
    strategy: str,
    threshold: float,
    regime: str,
    position_scaling: str,
    position_p_floor: float | None,
    position_p_full: float | None,
) -> Dict[str, Any]:
    if res_on is None:
        cost_on = np.nan
        trades_count = 0
        mean_holding_bars = np.nan
        avg_gap = np.nan
    else:
        cost_on = float(res_on.get("total_return", np.nan))
        trades_count = int(res_on.get("total_trades", 0))
        trades_list = res_on.get("trades", [])
        st = _extract_trade_stats(trades_list, df_w)
        mean_holding_bars = st["mean_holding_bars"]
        avg_gap = _avg_gap_between_trades(trades_list, df_w)
    if res_off is None:
        cost_off = np.nan
    else:
        cost_off = float(res_off.get("total_return", np.nan))
    if trades_count > 0:
        alpha_per_trade = cost_off / trades_count
        fee_per_trade = (cost_off - cost_on) / trades_count
        afr = alpha_per_trade / fee_per_trade if abs(fee_per_trade) > 1e-12 else np.nan
    else:
        alpha_per_trade = fee_per_trade = afr = np.nan
    return {
        "strategy": strategy,
        "threshold": threshold,
        "regime": regime,
        "position_scaling": position_scaling,
        "trades": trades_count,
        "cost_on": cost_on,
        "cost_off": cost_off,
        "alpha_per_trade": alpha_per_trade,
        "fee_per_trade": fee_per_trade if trades_count > 0 else np.nan,
        "alpha_fee_ratio": afr,
        "mean_holding_bars": mean_holding_bars,
        "avg_gap_between_trades": avg_gap,
    }


if __name__ == "__main__":
    main()
