#!/usr/bin/env python3
"""
180d / threshold=0.60 / emit_trade_log=True 손실 꼬리 분석.

출력:
- A. worst loss top 10
- B. exit_reason별 집계
- C. bars_held bucket별 집계
- D. direction별 집계
- E. 60d vs 90d 동일 현상 점검
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


def _mean_profit_roundtrip(res: dict[str, Any]) -> float | None:
    trades = list(res.get("trades") or [])
    if not trades:
        return None
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _run_window(
    *,
    df_bt: pd.DataFrame,
    pl_primary: np.ndarray,
    ps_primary: np.ndarray,
    t_max: pd.Timestamp,
    window_days: int,
    threshold: float,
    emit_trade_log: bool,
) -> dict[str, Any]:
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
    from scripts.threshold_sweep_latest_tmax_60d_fast import equity_step_metrics

    window_start = t_max - pd.Timedelta(days=window_days)
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    mask = (t >= window_start) & (t <= t_max)
    df_w = df_bt.loc[mask].reset_index(drop=True)
    pl_w = np.asarray(pl_primary, dtype=np.float32)[mask.to_numpy()]
    ps_w = np.asarray(ps_primary, dtype=np.float32)[mask.to_numpy()]

    COMMISSION = 0.0009
    SLIPPAGE = 0.0001
    MAX_ENTROPY = 1.30
    MIN_HOLD = 24
    COOLDOWN = 24
    TIME_STOP_BARS = 72
    EARLY_EXIT_BAD_K = 8

    res, err = run_backtest_7d(
        "BTCUSDT",
        "5m",
        df_w,
        pl_w,
        ps_w,
        commission_rate=COMMISSION,
        slippage_rate=SLIPPAGE,
        min_max_proba=float(threshold),
        max_entropy=MAX_ENTROPY,
        decision_mode="argmax",
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
        emit_trade_log=emit_trade_log,
    )
    if err:
        raise RuntimeError(f"run_backtest_7d failed (window={window_days}, thr={threshold}): {err}")
    if res is None:
        res = {}

    status, step_count, step_mean = equity_step_metrics(res.get("equity_curve"))
    out = {
        "window_days": window_days,
        "threshold": threshold,
        "rows": len(df_w),
        "entries_attempted": int(res.get("entries_attempted", 0) or 0),
        "total_trades": int(res.get("total_trades", 0) or 0),
        "unique_round_trips": int(res.get("unique_round_trips", 0) or 0),
        "win_rate": float(res.get("win_rate", 0.0) or 0.0),
        "mean_profit_roundtrip": _mean_profit_roundtrip(res),
        "cost_on": _cost_on(res),
        "equity_step_status": status,
        "equity_step_count": int(step_count),
        "equity_step_mean_return": step_mean,
        "result": res,
    }
    return out


def _to_exit_df(res: dict[str, Any]) -> pd.DataFrame:
    events = list(res.get("trade_events") or [])
    exit_events = [e for e in events if str(e.get("event", "")).startswith("EXIT")]
    if not exit_events:
        return pd.DataFrame(
            columns=["entry_ts", "exit_ts", "direction", "bars_held", "exit_reason", "profit"]
        )
    df = pd.DataFrame(exit_events)
    cols = ["entry_ts", "exit_ts", "direction", "bars_held", "exit_reason", "profit"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["bars_held"] = pd.to_numeric(df["bars_held"], errors="coerce")
    df["direction"] = df["direction"].astype(str)
    df["exit_reason"] = df["exit_reason"].astype(str)
    return df


def _bucketize_bars(v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "unknown"
    x = int(v)
    if x <= 12:
        return "0~12"
    if x <= 24:
        return "13~24"
    if x <= 48:
        return "25~48"
    if x <= 72:
        return "49~72"
    return "73+"


def main() -> int:
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask

    t_max = _to_utc_ts("2026-03-20T07:40:00+00:00")
    end_date_utc_str = t_max.strftime("%Y-%m-%d")
    DAYS_FULL = 182

    base_pt = MODELS_DIR / "tcn_h15_t0p004.pt"
    fr2_pt = MODELS_DIR / "tcn_h15_micro_v1.pt"

    print(f"[TAIL] t_max={t_max.isoformat()} days_full={DAYS_FULL}", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, base_pt, "base", False, end_date=end_date_utc_str)
    if err_b or triple_base is None:
        raise RuntimeError(f"get_ohlcv_and_proba(base) failed: {err_b}")
    df_b, pl_b, ps_b, _ = triple_base
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, fr2_pt, "microstructure_v1", True, end_date=end_date_utc_str)
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
    print(f"[TAIL] joined_rows={len(df_bt)}", flush=True)

    pl_base_arr = joined["pl_base"].to_numpy(dtype=np.float32)
    ps_base_arr = joined["ps_base"].to_numpy(dtype=np.float32)
    pl_fr2_arr = joined["pl_fr2"].to_numpy(dtype=np.float32)
    ps_fr2_arr = joined["ps_fr2"].to_numpy(dtype=np.float32)

    df_bt_reg = add_regime_columns(DAYS_FULL, df_bt.copy())
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

    # Main: 180d, thr=0.60, emit_trade_log=True
    out_180 = _run_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_max=t_max,
        window_days=180,
        threshold=0.60,
        emit_trade_log=True,
    )
    res_180 = out_180["result"]
    exit_df = _to_exit_df(res_180)

    # A. worst loss top 10
    worst10 = exit_df.sort_values("profit", ascending=True).head(10).copy()
    worst10 = worst10.reset_index(drop=True)
    worst10["rank"] = worst10.index + 1
    worst10 = worst10[["rank", "entry_ts", "exit_ts", "direction", "bars_held", "exit_reason", "profit"]]

    # B. exit_reason aggregate
    by_reason = (
        exit_df.groupby("exit_reason", dropna=False)["profit"]
        .agg(count="count", mean_profit="mean", median_profit="median", total_profit="sum")
        .reset_index()
        .sort_values("total_profit", ascending=True)
    )

    # C. bars_held bucket aggregate
    exit_df["bars_bucket"] = exit_df["bars_held"].apply(_bucketize_bars)
    bucket_order = ["0~12", "13~24", "25~48", "49~72", "73+", "unknown"]
    by_bucket = (
        exit_df.groupby("bars_bucket", dropna=False)["profit"]
        .agg(count="count", mean_profit="mean", total_profit="sum")
        .reset_index()
    )
    by_bucket["bars_bucket"] = pd.Categorical(by_bucket["bars_bucket"], categories=bucket_order, ordered=True)
    by_bucket = by_bucket.sort_values("bars_bucket")

    # D. direction aggregate
    by_dir = (
        exit_df.groupby("direction", dropna=False)
        .agg(
            count=("profit", "count"),
            win_rate=("profit", lambda s: float((s > 0).mean()) if len(s) > 0 else 0.0),
            mean_profit=("profit", "mean"),
            total_profit=("profit", "sum"),
        )
        .reset_index()
        .sort_values("direction")
    )

    # E. 60d vs 90d check
    out_60 = _run_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_max=t_max,
        window_days=60,
        threshold=0.60,
        emit_trade_log=False,
    )
    out_90 = _run_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_max=t_max,
        window_days=90,
        threshold=0.60,
        emit_trade_log=False,
    )
    t60 = list((out_60["result"].get("trades") or []))
    t90 = list((out_90["result"].get("trades") or []))
    key = lambda t: (str(t.get("entry_time")), str(t.get("exit_time")), str(t.get("direction")))
    k60 = {key(t) for t in t60}
    k90 = {key(t) for t in t90}
    additional_90 = sorted(list(k90 - k60))

    # Print outputs
    print("\nA. 180d worst loss top 10", flush=True)
    print("| rank | entry_ts | exit_ts | direction | bars_held | exit_reason | profit |", flush=True)
    print("|---:|---|---|---|---:|---|---:|", flush=True)
    for _, r in worst10.iterrows():
        print(
            f"| {int(r['rank'])} | {r['entry_ts']} | {r['exit_ts']} | {r['direction']} | "
            f"{int(r['bars_held']) if pd.notna(r['bars_held']) else 'N/A'} | {r['exit_reason']} | {float(r['profit']):.8f} |",
            flush=True,
        )

    print("\nB. exit_reason별 집계", flush=True)
    print("| exit_reason | count | mean_profit | median_profit | total_profit |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    for _, r in by_reason.iterrows():
        print(
            f"| {r['exit_reason']} | {int(r['count'])} | {float(r['mean_profit']):.8f} | "
            f"{float(r['median_profit']):.8f} | {float(r['total_profit']):.8f} |",
            flush=True,
        )

    print("\nC. bars_held bucket별 집계", flush=True)
    print("| bars_bucket | count | mean_profit | total_profit |", flush=True)
    print("|---|---:|---:|---:|", flush=True)
    for _, r in by_bucket.iterrows():
        print(
            f"| {r['bars_bucket']} | {int(r['count'])} | {float(r['mean_profit']):.8f} | {float(r['total_profit']):.8f} |",
            flush=True,
        )

    print("\nD. direction별 집계", flush=True)
    print("| direction | count | win_rate | mean_profit | total_profit |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    for _, r in by_dir.iterrows():
        print(
            f"| {r['direction']} | {int(r['count'])} | {float(r['win_rate']):.4f} | "
            f"{float(r['mean_profit']):.8f} | {float(r['total_profit']):.8f} |",
            flush=True,
        )

    print("\nE. 60d vs 90d 동일 현상 점검", flush=True)
    print(
        f"- 60d rows={out_60['rows']}, trades={out_60['total_trades']}, unique_rt={out_60['unique_round_trips']}, "
        f"mean_profit_rt={out_60['mean_profit_roundtrip']}, cost_on={out_60['cost_on']:.8f}",
        flush=True,
    )
    print(
        f"- 90d rows={out_90['rows']}, trades={out_90['total_trades']}, unique_rt={out_90['unique_round_trips']}, "
        f"mean_profit_rt={out_90['mean_profit_roundtrip']}, cost_on={out_90['cost_on']:.8f}",
        flush=True,
    )
    print(f"- 90d 추가 구간 trade count(vs60d): {len(additional_90)}", flush=True)
    if additional_90:
        print(f"- sample additional trade key: {additional_90[0]}", flush=True)
    else:
        print("- additional trade 없음 또는 영향 미미", flush=True)

    # compact context for conclusion
    print("\n[180d context]", flush=True)
    print(
        f"- unique_round_trips={out_180['unique_round_trips']} win_rate={out_180['win_rate']:.4f} "
        f"mean_profit_roundtrip={out_180['mean_profit_roundtrip']} cost_on={out_180['cost_on']:.8f} "
        f"equity_step_status={out_180['equity_step_status']} "
        f"equity_step_mean_return={out_180['equity_step_mean_return']}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
