#!/usr/bin/env python3
"""
latest t_max 기준 최근 60일 구간에서 min_max_proba(=threshold) 스윕.

요청 컬럼:
- entries_attempted
- total_trades
- win_rate
- mean_return
- sharpe (equity_curve 기반)
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


def _mean_return_from_equity_curve(equity_curve: list[float] | None) -> tuple[float, float]:
    """
    Returns: (mean_return, sharpe_ratio) where
    - returns = diff(equity_curve)/equity_curve[:-1]
    - sharpe = mean_return / std_return (risk_free_rate=0)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0
    arr = np.asarray(equity_curve, dtype=float)
    if np.any(~np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return 0.0, 0.0
    rets = np.diff(arr) / arr[:-1]
    if len(rets) == 0:
        return 0.0, 0.0
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets))
    if std_ret == 0.0:
        sharpe = 0.0
    else:
        sharpe = float(mean_ret / std_ret)
    return mean_ret, sharpe


def main() -> int:
    meta = _load_meta_source_latest()
    src_info = meta.get("source_info") or {}

    symbol = src_info.get("symbol", "BTCUSDT")
    timeframe = src_info.get("timeframe", "5m")
    threshold_default = float(src_info.get("threshold", 0.6))

    t_max = _to_utc_ts(src_info["first_probe_t_max"])
    end_date_utc_str = t_max.strftime("%Y-%m-%d")

    WINDOW_DAYS = 60
    window_start = t_max - pd.Timedelta(days=WINDOW_DAYS)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    # Build ensemble proba ONCE for the full backtest canvas up to t_max.
    # (threshold only affects entry filter in backtest.)
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
    FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"

    # Safe load horizon for feature windows.
    DAYS_FULL = 120

    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, BASELINE_PT, "base", False, end_date=end_date_utc_str)
    if err_b or triple_base is None:
        raise RuntimeError(f"get_ohlcv_and_proba(base) failed: {err_b}")
    df_b, pl_b, ps_b, _ = triple_base

    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, FR2_PT, "microstructure_v1", True, end_date=end_date_utc_str)
    if err_f or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba(fr2) failed: {err_f}")
    df_f, pl_f, ps_f, _ = triple_fr2

    if not (len(df_b) == len(pl_b) == len(ps_b)):
        raise RuntimeError(f"base length mismatch: df_b={len(df_b)} pl_b={len(pl_b)} ps_b={len(ps_b)}")
    if not (len(df_f) == len(pl_f) == len(ps_f)):
        raise RuntimeError(f"fr2 length mismatch: df_f={len(df_f)} pl_f={len(pl_f)} ps_f={len(ps_f)}")

    # One-to-one timestamp alignment.
    d1 = df_b.copy()
    d2 = df_f.copy()
    d1["timestamp"] = pd.to_datetime(d1["timestamp"])
    d2["timestamp"] = pd.to_datetime(d2["timestamp"])

    d1["pl_base"] = np.asarray(pl_b, dtype=np.float32)
    d1["ps_base"] = np.asarray(ps_b, dtype=np.float32)
    d2["pl_fr2"] = np.asarray(pl_f, dtype=np.float32)
    d2["ps_fr2"] = np.asarray(ps_f, dtype=np.float32)

    # Join on timestamp.
    joined = d1.merge(
        d2[["timestamp", "pl_fr2", "ps_fr2"]],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp").reset_index(drop=True)

    if len(joined) < 500:
        raise RuntimeError(f"Aligned joined length too small: {len(joined)}")

    df_bt = joined[["timestamp", "close", "high", "low"]].copy()
    pl_base = joined["pl_base"].to_numpy(dtype=np.float32)
    ps_base = joined["ps_base"].to_numpy(dtype=np.float32)
    pl_fr2 = joined["pl_fr2"].to_numpy(dtype=np.float32)
    ps_fr2 = joined["ps_fr2"].to_numpy(dtype=np.float32)

    # Regime + C4 gate for ensemble override mode.
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

    # Slice exactly the fixed 60d window.
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    t_max_utc = _to_utc_ts(t_max)
    mask = (t >= window_start) & (t <= t_max_utc)

    df_60 = df_bt.loc[mask].reset_index(drop=True)
    pl_60 = np.asarray(pl_primary, dtype=np.float32)[mask.to_numpy()]
    ps_60 = np.asarray(ps_primary, dtype=np.float32)[mask.to_numpy()]

    if not (len(df_60) == len(pl_60) == len(ps_60)):
        raise RuntimeError(f"Window length mismatch: df_60={len(df_60)} pl_60={len(pl_60)} ps_60={len(ps_60)}")

    COMMISSION = 0.0009
    SLIPPAGE = 0.0001
    MAX_ENTROPY = 1.30
    MIN_HOLD = 24
    COOLDOWN = 24
    TIME_STOP_BARS = 72
    EARLY_EXIT_BAD_K = 8

    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        res, err = run_backtest_7d(
            symbol,
            timeframe,
            df_60,
            pl_60,
            ps_60,
            commission_rate=COMMISSION,
            slippage_rate=SLIPPAGE,
            min_max_proba=float(thr),
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
            raise RuntimeError(f"run_backtest_7d failed at thr={thr}: {err}")
        if res is None:
            # Should not happen, but keep numeric.
            res = {}

        total_trades = int(res.get("total_trades", 0) or 0)
        entries_attempted = int(res.get("entries_attempted", 0) or 0)
        win_rate = float(res.get("win_rate", 0.0) or 0.0)
        equity_curve = res.get("equity_curve")
        mean_return, sharpe = _mean_return_from_equity_curve(equity_curve)

        rows.append({
            "threshold": float(thr),
            "entries_attempted": entries_attempted,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "mean_return": mean_return,
            "sharpe": sharpe,
        })

    # Find first threshold where trades > 0.
    alive = [r for r in rows if int(r["total_trades"]) > 0]
    min_alive_thr = min(alive, key=lambda r: r["threshold"])["threshold"] if alive else None

    # Print tables.
    print("## Threshold Sweep (fixed t_max, fixed 60d window)", flush=True)
    print(
        "| t_max | window_start | threshold | entries_attempted | total_trades | win_rate | mean_return | sharpe |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        print(
            f"| {t_max.isoformat()} | {window_start.isoformat()} | {r['threshold']:.2f} | {r['entries_attempted']} | {r['total_trades']} | {r['win_rate']:.4f} | {r['mean_return']:.8f} | {r['sharpe']:.4f} |",
            flush=True,
        )

    print("\n## Result", flush=True)
    if min_alive_thr is None:
        print("trade가 살아나는 최소 threshold: NONE (전 구간 total_trades=0)", flush=True)
    else:
        print(f"trade가 살아나는 최소 threshold: {min_alive_thr:.2f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

