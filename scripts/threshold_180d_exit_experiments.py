#!/usr/bin/env python3
"""
180d / threshold=0.60 고정에서 exit 구조 실험 비교.

실험:
- BASELINE
- EXP1: signal_exit 조기화(early_exit 조건 강화)
- EXP2: time_stop 단축(72 -> 48)
- EXP3: SHORT OFF (long_only=True)
- EXP4: EXP1 + EXP2 + EXP3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _cost_on(res: dict[str, Any]) -> float:
    ec = res.get("equity_curve") or []
    if isinstance(ec, list) and len(ec) >= 1:
        return float(ec[-1] - 1.0)
    return float(res.get("total_return", 0.0) or 0.0)


def _mean_profit_rt(res: dict[str, Any]) -> float | None:
    trades = list(res.get("trades") or [])
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _max_loss_trade(res: dict[str, Any]) -> dict[str, Any] | None:
    trades = list(res.get("trades") or [])
    if not trades:
        return None
    with_profit = [t for t in trades if t.get("profit") is not None]
    if not with_profit:
        return None
    t = min(with_profit, key=lambda x: float(x["profit"]))
    return {
        "entry_time": t.get("entry_time"),
        "exit_time": t.get("exit_time"),
        "direction": t.get("direction"),
        "profit": float(t["profit"]),
    }


def _avg_win_loss(res: dict[str, Any]) -> tuple[float | None, float | None]:
    trades = list(res.get("trades") or [])
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None, None
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v <= 0]
    avg_win = float(np.mean(wins)) if wins else None
    avg_loss = float(np.mean(losses)) if losses else None
    return avg_win, avg_loss


def _build_pipeline(days_full: int, t_max: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask

    base_pt = MODELS_DIR / "tcn_h15_t0p004.pt"
    fr2_pt = MODELS_DIR / "tcn_h15_micro_v1.pt"
    end_date_utc_str = t_max.strftime("%Y-%m-%d")

    triple_base, err_b = get_ohlcv_and_proba(days_full, base_pt, "base", False, end_date=end_date_utc_str)
    if err_b or triple_base is None:
        raise RuntimeError(f"get_ohlcv_and_proba(base) failed: {err_b}")
    df_b, pl_b, ps_b, _ = triple_base

    triple_fr2, err_f = get_ohlcv_and_proba(days_full, fr2_pt, "microstructure_v1", True, end_date=end_date_utc_str)
    if err_f or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba(fr2) failed: {err_f}")
    df_f, pl_f, ps_f, _ = triple_fr2

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
    pl_base_arr = joined["pl_base"].to_numpy(dtype=np.float32)
    ps_base_arr = joined["ps_base"].to_numpy(dtype=np.float32)
    pl_fr2_arr = joined["pl_fr2"].to_numpy(dtype=np.float32)
    ps_fr2_arr = joined["ps_fr2"].to_numpy(dtype=np.float32)

    df_bt_reg = add_regime_columns(days_full, df_bt.copy())
    base_c4 = ((df_bt_reg["trend_regime"] == "uptrend") & (df_bt_reg["vol_regime"] == "high_vol")).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)
    inputs = EnsembleInputs(
        pl_base=pl_base_arr,
        ps_base=ps_base_arr,
        pl_fr2=pl_fr2_arr,
        ps_fr2=ps_fr2_arr,
        c4_active=c4_mask,
    )
    pl_primary, ps_primary = build_ensemble_proba(inputs, mode="override")
    return df_bt, pl_primary, ps_primary


def _run_exp(
    *,
    name: str,
    df_bt: pd.DataFrame,
    pl_primary: np.ndarray,
    ps_primary: np.ndarray,
    t_max: pd.Timestamp,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategies.ml_thresholds import resolve_ml_thresholds
    from scripts.threshold_sweep_latest_tmax_60d_fast import equity_step_metrics

    t_start = t_max - pd.Timedelta(days=180)
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    mask = (t >= t_start) & (t <= t_max)
    df_w = df_bt.loc[mask].reset_index(drop=True)
    pl_w = np.asarray(pl_primary, dtype=np.float32)[mask.to_numpy()]
    ps_w = np.asarray(ps_primary, dtype=np.float32)[mask.to_numpy()]

    engine = get_ml_backtest_engine(
        strategy_name="ml_tcn",
        symbol="BTCUSDT",
        timeframe="5m",
        feature_preset="base",
        tcn_preset=None,
    )
    long_th, short_th = resolve_ml_thresholds(
        strategy_name="ml_tcn",
        symbol="BTCUSDT",
        timeframe="5m",
        use_optimized_thresholds=True,
    )
    short_th = short_th if short_th is not None else 0.5

    base_kwargs = dict(
        long_threshold=long_th,
        short_threshold=short_th,
        use_optimized_threshold=True,
        use_stage2=True,
        use_strategy_guard_v2=True,
        proba_long_cache=pl_w,
        proba_short_cache=ps_w,
        df_with_proba=df_w,
        commission_rate=0.0009,
        slippage_rate=0.0001,
        min_max_proba=0.60,
        max_entropy=1.30,
        min_proba_gap=0.0,
        min_directional_gap=None,
        max_flat_entry_proba=None,
        decision_mode="argmax",
        min_directional_edge=None,
        require_direction_gt_flat=True,
        min_side_flat_margin=None,
        min_confidence=None,
        use_legacy_entry_gates_with_directional=True,
        min_hold_bars_override=24,
        cooldown_bars_override=24,
        regime_filter_enabled=False,
        regime_ema_span=200,
        regime_rule="ema_only",
        regime_slope_lookback=48,
        vol_window=48,
        vol_threshold=0.0010,
        breakout_lookback=96,
        breakout_mode="ema_high",
        slope_threshold=1e-5,
        q_window=8640,
        q=0.90,
        p_floor=0.55,
        position_scaling_enabled=False,
        position_scaling_mode="linear",
        position_p_floor=0.55,
        position_p_full=0.65,
        position_size_min=0.25,
        position_size_max=1.0,
        position_p_mid=0.60,
        position_k=25.0,
        early_exit_enabled=True,
        early_exit_lookback=12,
        early_exit_p_floor=0.55,
        early_exit_bad_k=8,
        entry_flat_gate_enabled=False,
        max_flat_proba=0.45,
        flat_exit_aware_enabled=False,
        flat_exit_threshold=0.30,
        flat_exit_badk_delta=2,
        time_stop_enabled=True,
        time_stop_bars=72,
        partial_tp_enabled=False,
        partial_tp_threshold=0.0025,
        partial_tp_ratio=0.5,
        break_even_stop_enabled=False,
        be_threshold=0.0,
        emit_trade_log=False,
        long_only=False,
        short_only=False,
    )
    base_kwargs.update(overrides)
    res = engine.run_backtest(**base_kwargs)

    status, _, step_mean = equity_step_metrics(res.get("equity_curve"))
    avg_win, avg_loss = _avg_win_loss(res)
    max_loss = _max_loss_trade(res)
    return {
        "experiment": name,
        "unique_round_trips": int(res.get("unique_round_trips", 0) or 0),
        "win_rate": float(res.get("win_rate", 0.0) or 0.0),
        "mean_profit_roundtrip": _mean_profit_rt(res),
        "cost_on": _cost_on(res),
        "total_return": float(res.get("total_return", 0.0) or 0.0),
        "equity_step_mean_return": step_mean,
        "equity_step_status": status,
        "max_loss_trade": max_loss,
        "avg_loss": avg_loss,
        "avg_win": avg_win,
    }


def _fmt(v: Any, n: int = 8) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{n}f}"
    return str(v)


def main() -> int:
    t_max = _to_utc_ts("2026-03-20T07:40:00+00:00")
    days_full = 182
    print(f"[EXP] t_max={t_max.isoformat()} days_full={days_full}", flush=True)
    df_bt, pl_primary, ps_primary = _build_pipeline(days_full=days_full, t_max=t_max)
    print(f"[EXP] joined_rows={len(df_bt)}", flush=True)

    experiments = [
        ("BASELINE", {}),
        # EXP1: signal_exit 조기화(확률 기반 조기 청산 민감도 강화)
        ("EXP1_signal_exit_early", {"early_exit_lookback": 8, "early_exit_p_floor": 0.60, "early_exit_bad_k": 4}),
        # EXP2: time_stop 단축
        ("EXP2_time_stop_48", {"time_stop_bars": 48}),
        # EXP3: SHORT OFF (LONG only)
        ("EXP3_short_off", {"long_only": True}),
        # EXP4: 결합
        (
            "EXP4_combo",
            {
                "early_exit_lookback": 8,
                "early_exit_p_floor": 0.60,
                "early_exit_bad_k": 4,
                "time_stop_bars": 48,
                "long_only": True,
            },
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, overrides in experiments:
        print(f"[EXP] running {name} ...", flush=True)
        rows.append(
            _run_exp(
                name=name,
                df_bt=df_bt,
                pl_primary=pl_primary,
                ps_primary=ps_primary,
                t_max=t_max,
                overrides=overrides,
            )
        )

    print("\n| experiment | unique_round_trips | win_rate | mean_profit_roundtrip | cost_on | total_return | equity_step_mean_return | max_loss_trade | avg_loss | avg_win |", flush=True)
    print("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|", flush=True)
    for r in rows:
        mlt = r["max_loss_trade"]
        if mlt is None:
            mlt_s = "N/A"
        else:
            mlt_s = (
                f"{mlt.get('profit'):.6f} ({mlt.get('direction')} {mlt.get('entry_time')}->{mlt.get('exit_time')})"
            )
        print(
            f"| {r['experiment']} | {r['unique_round_trips']} | {r['win_rate']:.4f} | "
            f"{_fmt(r['mean_profit_roundtrip'])} | {_fmt(r['cost_on'])} | {_fmt(r['total_return'])} | "
            f"{_fmt(r['equity_step_mean_return'])} | {mlt_s} | {_fmt(r['avg_loss'])} | {_fmt(r['avg_win'])} |",
            flush=True,
        )

    # Delta vs baseline
    base = rows[0]
    print("\n[DELTA vs BASELINE]", flush=True)
    print("| experiment | d_total_return | d_cost_on | d_mean_profit_rt | d_unique_round_trips |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    for r in rows[1:]:
        d_tr = float(r["total_return"]) - float(base["total_return"])
        d_cost = float(r["cost_on"]) - float(base["cost_on"])
        b_mpr = float(base["mean_profit_roundtrip"]) if base["mean_profit_roundtrip"] is not None else 0.0
        r_mpr = float(r["mean_profit_roundtrip"]) if r["mean_profit_roundtrip"] is not None else 0.0
        d_mpr = r_mpr - b_mpr
        d_n = int(r["unique_round_trips"]) - int(base["unique_round_trips"])
        print(f"| {r['experiment']} | {d_tr:.8f} | {d_cost:.8f} | {d_mpr:.8f} | {d_n} |", flush=True)

    best = max(rows, key=lambda x: float(x["total_return"]))
    print("\n[SUMMARY]", flush=True)
    print(
        f"best_by_total_return={best['experiment']} total_return={float(best['total_return']):.8f} "
        f"cost_on={float(best['cost_on']):.8f} unique_round_trips={int(best['unique_round_trips'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

