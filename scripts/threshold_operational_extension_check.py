#!/usr/bin/env python3
"""
운영 해석 정교화 점검:
- 60d에서 threshold 0.60 vs 0.61 비교
- 90d / 180d에서 threshold 0.60 확장 검증
- emit_trade_log=True 최소 루트로 exit/tail 분석 가능 필드 확인
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


def _cost_on_from_result(res: dict[str, Any]) -> float:
    ec = res.get("equity_curve") or []
    if isinstance(ec, list) and len(ec) >= 1:
        return float(ec[-1] - 1.0)
    return float(res.get("total_return", 0.0) or 0.0)


def _mean_profit_roundtrip_from_result(res: dict[str, Any]) -> float | None:
    trades = list(res.get("trades") or [])
    if not trades:
        return None
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _run_window_backtest(
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

    ec_status, ec_count, ec_mean = equity_step_metrics(res.get("equity_curve"))
    row = {
        "window_days": int(window_days),
        "threshold": float(threshold),
        "entries_attempted": int(res.get("entries_attempted", 0) or 0),
        "total_trades": int(res.get("total_trades", 0) or 0),
        "unique_round_trips": int(res.get("unique_round_trips", 0) or 0),
        "win_rate": float(res.get("win_rate", 0.0) or 0.0),
        "mean_profit_roundtrip": _mean_profit_roundtrip_from_result(res),
        "cost_on": _cost_on_from_result(res),
        "equity_step_status": ec_status,
        "equity_step_mean_return": ec_mean,
        "equity_step_count": int(ec_count),
    }
    return {"row": row, "result": res}


def _print_markdown_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n## {title}", flush=True)
    print(
        "| threshold | entries_attempted | total_trades | unique_round_trips | win_rate | mean_profit_roundtrip | cost_on | equity_step_status | equity_step_mean_return |",
        flush=True,
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---|---:|", flush=True)
    for r in rows:
        mpr = r["mean_profit_roundtrip"]
        esm = r["equity_step_mean_return"]
        mpr_s = f"{float(mpr):.8f}" if mpr is not None else "N/A"
        esm_s = f"{float(esm):.8f}" if esm is not None else "N/A"
        print(
            f"| {r['threshold']:.2f} | {r['entries_attempted']} | {r['total_trades']} | "
            f"{r['unique_round_trips']} | {r['win_rate']:.4f} | {mpr_s} | {r['cost_on']:.8f} | "
            f"{r['equity_step_status']} | {esm_s} |",
            flush=True,
        )


def main() -> int:
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask

    t_max = _to_utc_ts("2026-03-20T07:40:00+00:00")
    end_date_utc_str = t_max.strftime("%Y-%m-%d")
    DAYS_FULL = 182  # 180d window + minimal safety

    base_pt = MODELS_DIR / "tcn_h15_t0p004.pt"
    fr2_pt = MODELS_DIR / "tcn_h15_micro_v1.pt"

    print(f"[EXT] t_max={t_max.isoformat()} days_full={DAYS_FULL}", flush=True)
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
    print(f"[EXT] joined_rows={len(df_bt)}", flush=True)

    # [B] 60d: 0.60 vs 0.61
    compare_rows: list[dict[str, Any]] = []
    for thr in (0.60, 0.61):
        out = _run_window_backtest(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_max=t_max,
            window_days=60,
            threshold=thr,
            emit_trade_log=False,
        )
        compare_rows.append(out["row"])
    _print_markdown_table("60d Re-Validation (0.60 vs 0.61)", compare_rows)

    # [C] 90d / 180d at 0.60
    ext_rows: list[dict[str, Any]] = []
    for wd in (90, 180):
        out = _run_window_backtest(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_max=t_max,
            window_days=wd,
            threshold=0.60,
            emit_trade_log=False,
        )
        ext_rows.append(out["row"])
    _print_markdown_table("Window Extension (0.60 @ 90d/180d)", ext_rows)

    # [D] emit_trade_log=True minimal route check
    out_log = _run_window_backtest(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_max=t_max,
        window_days=60,
        threshold=0.60,
        emit_trade_log=True,
    )
    res_log = out_log["result"]
    trade_events = list(res_log.get("trade_events") or [])
    exit_events = [e for e in trade_events if str(e.get("event", "")).startswith("EXIT")]
    required = {"exit_reason", "bars_held", "entry_ts", "exit_ts", "direction", "profit"}
    available = set(exit_events[0].keys()) if exit_events else set()
    missing = sorted(list(required - available))

    print("\n## emit_trade_log=True Readiness", flush=True)
    print(f"- trade_events_count: {len(trade_events)}", flush=True)
    print(f"- exit_events_count: {len(exit_events)}", flush=True)
    print(f"- required_fields: {sorted(required)}", flush=True)
    print(f"- missing_fields: {missing}", flush=True)
    if exit_events:
        sample = exit_events[0]
        print("- sample_exit_event:", flush=True)
        print(
            f"  exit_reason={sample.get('exit_reason')} bars_held={sample.get('bars_held')} "
            f"entry_time={sample.get('entry_ts')} exit_time={sample.get('exit_ts')} "
            f"direction={sample.get('direction')} profit={sample.get('profit')}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
