#!/usr/bin/env python3
"""
TCN 실전 후보 1개(Primary Candidate)에 대해 기간 확장 검증.
고정 필터(min_max_proba, max_entropy, min_hold, cooldown)로 days_list 및 연도별 구간에서
백테스트를 1회씩 수행하고, JSON/MD + 요약을 출력.

사용법:
  python -m scripts.run_tcn_candidate_validation \\
    --id h30_t0p004 --symbol BTCUSDT --timeframe 5m \\
    --min-max-proba 0.50 --max-entropy 1.35 --min-hold 36 --cooldown 6 \\
    [--days-list "30,60,90,180,365"] [--year-spans "2024-01-01:2024-12-31,2025-01-01:2025-12-31"]
  # Regime 3-way: off | ema_only | ema_plus_slope
  python -m scripts.run_tcn_candidate_validation \\
    --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \\
    --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \\
    --regime-filter ema_only
"""
from __future__ import annotations

# Segfault(exit 139) 완화: OMP/MKL/OpenBLAS 스레드 충돌 방지 — 반드시 다른 모듈 import 전에 적용
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"


def _write_trade_log_csv(trade_events: list, run_id: str, suffix: str | int, trade_log_path: str) -> None:
    """Write EXIT-only trade log CSV: entry_ts, exit_ts, exit_reason, holding_bars, entry_price, exit_price, net_return, MFE, MAE, be_armed, be_arm_ts. suffix e.g. 30 -> 30d.csv, 365 -> 365d.csv, or '2024'."""
    exit_events = [e for e in trade_events if e.get("event", "").startswith("EXIT") and "exit_reason" in e]
    if not exit_events:
        return
    out_dir = Path(trade_log_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{run_id}_{suffix}d.csv" if isinstance(suffix, int) else f"{run_id}_{suffix}.csv"
    out_file = out_dir / name
    rows = []
    for e in exit_events:
        rows.append({
            "entry_ts": e.get("entry_ts") or e.get("ts"),
            "exit_ts": e.get("exit_ts") or e.get("ts"),
            "exit_reason": e.get("exit_reason", ""),
            "holding_bars": e.get("holding_bars") if e.get("holding_bars") is not None else e.get("bars_held"),
            "entry_price": e.get("entry_price"),
            "exit_price": e.get("exit_price"),
            "net_return": e.get("net_return") if e.get("net_return") is not None else e.get("profit"),
            "position_scale": e.get("position_scale"),
            "max_favorable_excursion": e.get("max_favorable_excursion"),
            "max_adverse_excursion": e.get("max_adverse_excursion"),
            "be_armed": e.get("be_armed", False),
            "be_arm_ts": e.get("be_arm_ts") or "",
        })
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"[TRADE_LOG] {out_file} ({len(rows)} exits)", flush=True)

# GO 조건 상수
COST_ON_POSITIVE_RATIO_MIN = 0.60  # days-list 결과 중 cost_on >= 0 비율
MDD_MAX = 0.05  # 최악 span max_drawdown
TRADES_365_MIN = 50  # 365d에서 trades 최소 (너무 낮으면 운빨)


def main() -> int:
    parser = argparse.ArgumentParser(description="TCN candidate validation: multi-period backtest for one fixed filter")
    parser.add_argument("--id", type=str, required=True, help="Model id, e.g. h30_t0p004")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--min-max-proba", type=float, required=True)
    parser.add_argument("--max-entropy", type=float, required=True)
    parser.add_argument("--min-hold", type=int, required=True)
    parser.add_argument("--cooldown", type=int, required=True)
    parser.add_argument("--days-list", type=str, default="30,60,90,180,365", help="Comma-separated days, e.g. 30,60,90,180,365")
    parser.add_argument("--year-spans", type=str, default=None, help="Optional: start:end per year, e.g. 2024-01-01:2024-12-31,2025-01-01:2025-12-31")
    parser.add_argument("--out-prefix", type=str, default="tcn_candidate_validation")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--force-rebuild", action="store_true", help="Placeholder; no-op")
    parser.add_argument("--regime-filter", type=str, choices=["off", "on", "ema_only", "ema_plus_slope", "vol_compress", "vol_slope", "vol_breakout", "flat_slope", "downtrend_block", "proba_quantile"], default="off",
                        help="Regime: off | ... | downtrend_block | proba_quantile (확률 분위수 게이트)")
    parser.add_argument("--regime-ema-span", type=int, default=200, help="EMA span for regime filter (default 200)")
    parser.add_argument("--regime-slope-lookback", type=int, default=48, help="Bars for EMA slope (default 48 = 4h on 5m)")
    parser.add_argument("--vol-window", type=int, default=48, help="Rolling window for vol (default 48)")
    parser.add_argument("--vol-threshold", type=float, default=0.0010, help="Vol threshold: compress when vol < this (default 0.0010)")
    parser.add_argument("--breakout-lookback", type=int, default=96, help="RollingHigh lookback for vol_breakout (default 96 = 8h on 5m)")
    parser.add_argument("--breakout-mode", type=str, choices=["ema", "ema_high"], default="ema_high", help="vol_breakout: ema only or ema+rolling_high")
    parser.add_argument("--slope-threshold", type=float, default=1e-5, help="flat_slope: skip Long when abs(ema_slope) <= this (default 1e-5)")
    parser.add_argument("--q-window", type=int, default=8640, help="proba_quantile: rolling window in bars (default 8640 = 30d @ 5m)")
    parser.add_argument("--q", type=float, default=0.90, help="proba_quantile: quantile threshold, e.g. 0.90 = top 10%% (default 0.90)")
    parser.add_argument("--p-floor", type=float, default=0.55, help="proba_quantile: minimum p gate (default 0.55)")
    parser.add_argument("--position-scaling", type=str, choices=["off", "linear", "sigmoid"], default="off",
                        help="Position scaling: off | linear | sigmoid (proba-based size)")
    parser.add_argument("--position-p-floor", type=float, default=0.55, help="position scaling linear: p_floor (default 0.55)")
    parser.add_argument("--position-p-full", type=float, default=0.65, help="position scaling linear: p_full (default 0.65)")
    parser.add_argument("--position-size-min", type=float, default=0.25, help="position scaling: size_min (default 0.25)")
    parser.add_argument("--position-size-max", type=float, default=1.0, help="position scaling: size_max (default 1.0)")
    parser.add_argument("--position-p-mid", type=float, default=0.60, help="position scaling sigmoid: p_mid (default 0.60)")
    parser.add_argument("--position-k", type=float, default=25.0, help="position scaling sigmoid: k (default 25.0)")
    parser.add_argument("--early-exit", type=str, choices=["on", "off"], default="off", help="Early exit (proba-based): off | on (default off)")
    parser.add_argument("--early-exit-lookback", type=int, default=12, help="Early exit: lookback bars (default 12 = 1h @ 5m)")
    parser.add_argument("--early-exit-p-floor", type=float, default=0.55, help="Early exit: proba below this counts as bad bar (default 0.55)")
    parser.add_argument("--early-exit-bad-k", type=int, default=8, help="Early exit: exit when bad bars >= this in lookback (default 8)")
    parser.add_argument("--commission", type=float, default=None, help="Commission rate (default 0.0009). Used for Phase A sensitivity.")
    parser.add_argument("--slippage", type=float, default=None, help="Slippage rate (default 0.0001). Used for Phase A sensitivity.")
    parser.add_argument("--entry-flat-gate", type=str, choices=["on", "off"], default="off", help="Phase C: skip entry when proba_flat > max_flat_proba (default off)")
    parser.add_argument("--max-flat-proba", type=float, default=0.45, help="Phase C: max proba_flat allowed for entry (default 0.45)")
    parser.add_argument("--calibration", type=str, choices=["off", "temp"], default="off", help="Phase B1: off | temp (temperature scaling, default off)")
    parser.add_argument("--calibration-T", type=float, default=1.0, dest="calibration_T", help="Phase B1: temperature for temp scaling (default 1.0)")
    parser.add_argument("--flat-exit-aware", type=str, choices=["on", "off"], default="off", help="Phase C3: when proba_flat > threshold use lower effective_bad_k (default off)")
    parser.add_argument("--flat-exit-threshold", type=float, default=0.30, help="Phase C3: proba_flat above this triggers lower bad_k (default 0.30)")
    parser.add_argument("--flat-exit-badk-delta", type=int, default=2, help="Phase C3: subtract from bad_k when flat above threshold (default 2)")
    parser.add_argument("--time-stop", type=str, choices=["on", "off"], default="off", help="Phase C4: force exit when holding_bars >= time_stop_bars (default off)")
    parser.add_argument("--time-stop-bars", type=int, default=96, help="Phase C4: max holding bars before time-stop exit (default 96 = 8h @ 5m)")
    parser.add_argument("--partial-tp", type=str, choices=["on", "off"], default="off", help="Phase C7: partial take-profit (default off)")
    parser.add_argument("--partial-tp-threshold", type=float, default=0.0025, help="Phase C7: profit threshold (e.g. 0.0025 = +0.25%%) to trigger partial TP")
    parser.add_argument("--partial-tp-ratio", type=float, default=0.5, help="Phase C7: fraction to close on partial TP (e.g. 0.5 = 50%%)")
    parser.add_argument("--break-even-stop", type=str, choices=["on", "off"], default="off", help="Phase C9: break-even stop (default off)")
    parser.add_argument("--be-threshold", type=float, default=0.002, help="Phase C9: unrealized return threshold to activate BE stop (e.g. 0.002 = 0.2%%)")
    parser.add_argument("--end-date", type=str, default=None, help="Ops pinning: YYYY-MM-DD end of window (with start-date-30/365 or derived)")
    parser.add_argument("--start-date-30", type=str, default=None, help="Ops pinning: 30d window start (optional, default end_date - 31)")
    parser.add_argument("--start-date-365", type=str, default=None, help="Ops pinning: 365d window start (optional, default end_date - 366)")
    parser.add_argument("--emit-trade-log", type=str, choices=["on", "off"], default="off", help="Emit trade log CSV for sanity check (default off)")
    parser.add_argument("--trade-log-path", type=str, default=None, help="Dir or path for trade log CSV (default data/diagnostics/trade_logs)")
    args = parser.parse_args()

    regime_enabled = args.regime_filter not in ("off",)
    regime_rule = "ema_only" if args.regime_filter == "on" else (args.regime_filter if regime_enabled else "ema_only")
    early_exit_on = args.early_exit == "on"
    entry_flat_gate_on = args.entry_flat_gate == "on"
    calibration_on = args.calibration == "temp"
    calibration_T = getattr(args, "calibration_T", 1.0)
    flat_exit_aware_on = args.flat_exit_aware == "on"
    flat_exit_threshold = args.flat_exit_threshold
    flat_exit_badk_delta = args.flat_exit_badk_delta
    time_stop_on = args.time_stop == "on"
    time_stop_bars = args.time_stop_bars
    partial_tp_on = args.partial_tp == "on"
    partial_tp_threshold = args.partial_tp_threshold
    partial_tp_ratio = args.partial_tp_ratio
    break_even_on = args.break_even_stop == "on"
    be_threshold = args.be_threshold
    emit_trade_log = args.emit_trade_log == "on"
    trade_log_path = args.trade_log_path or str(DIAG / "trade_logs")

    model_path = MODELS_DIR / f"tcn_{args.id}.pt"
    if not model_path.exists():
        print(f"ERROR: model not found {model_path}", file=sys.stderr)
        return 1

    days_list = [int(x.strip()) for x in args.days_list.split(",") if x.strip()]
    year_spans = []
    if args.year_spans:
        for part in args.year_spans.split(","):
            part = part.strip()
            if ":" in part:
                a, b = part.split(":", 1)
                year_spans.append((a.strip(), b.strip()))

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = DIAG / f"{args.out_prefix}_{args.symbol}_{args.timeframe}_{args.id}_{run_id}.json"
    out_md = DIAG / f"{args.out_prefix}_{args.symbol}_{args.timeframe}_{args.id}_{run_id}.md"

    from scripts.run_tcn_label_sweep_v2 import get_7d_ohlcv_and_proba, run_backtest_7d
    from src.utils.proba_calibration import temperature_scale_3class

    commission = args.commission if args.commission is not None else 0.0009
    slippage = args.slippage if args.slippage is not None else 0.0001
    results = []
    calibration_stats_for_meta = None
    _cal_eps = 1e-8

    # 기간 pinning: end_date가 있으면 각 span별 start/end 고정
    pinned_start_dates: dict[int, str] = {}
    pinned_end_date: str | None = None
    if args.end_date:
        try:
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            pinned_end_date = args.end_date
            for d in days_list:
                if d == 30 and args.start_date_30:
                    pinned_start_dates[30] = args.start_date_30
                elif d == 365 and args.start_date_365:
                    pinned_start_dates[365] = args.start_date_365
                else:
                    start_dt = end_dt - timedelta(days=d)  # d일 치 데이터 (end 당일 포함)
                    pinned_start_dates[d] = start_dt.strftime("%Y-%m-%d")
        except ValueError:
            print(f"WARNING: invalid end-date/start-date, ignoring pinning", file=sys.stderr)
            pinned_end_date = None
            pinned_start_dates = {}

    # --- days_list: 각 기간별로 OHLCV+proba 로드 후 백테스트 1회
    for days in days_list:
        start_date_arg = pinned_start_dates.get(days)
        end_date_arg = pinned_end_date
        if start_date_arg and end_date_arg:
            triple, err = get_7d_ohlcv_and_proba(
                model_path, args.symbol, args.timeframe, days,
                log_prefix=f"[{days}d] ",
                start_date=start_date_arg,
                end_date=end_date_arg,
            )
        else:
            triple, err = get_7d_ohlcv_and_proba(
                model_path, args.symbol, args.timeframe, days, log_prefix=f"[{days}d] ",
            )
        if triple is None:
            results.append({
                "span": f"{days}d",
                "days": days,
                "error": err or "get_ohlcv_and_proba returned None",
                "trades": None,
                "cost_on_return": None,
                "cost_off_return": None,
                "win_rate_on": None,
                "win_rate_off": None,
                "max_drawdown": None,
                "entries_attempted": None,
                "entries_executed": None,
                "filter_skip_stats": None,
                "regime_enabled": None,
                "regime_rule": None,
                "regime_span": None,
                "regime_slope_lookback": None,
                "vol_window": None,
                "vol_threshold": None,
                "pct_above_ema200": None,
                "pct_compress": None,
                "entries_blocked_by_regime": None,
                "entries_blocked_by_regime_price": None,
                "entries_blocked_by_regime_slope": None,
                "entries_blocked_by_regime_vol": None,
                "entries_blocked_by_regime_vol_slope": None,
                "entries_blocked_by_regime_vol_breakout": None,
                "entries_allowed_on_decompress": None,
                "pct_vol_slope_block": None,
                "breakout_lookback": None,
                "breakout_mode": None,
                "slope_threshold": None,
                "pct_flat_slope_block": None,
                "entries_blocked_by_regime_flat_slope": None,
                "pct_downtrend_block": None,
                "entries_blocked_by_regime_downtrend": None,
                "q_window": None,
                "q": None,
                "p_floor": None,
                "q_threshold_mean": None,
                "pct_proba_quantile_block": None,
                "entries_blocked_by_regime_proba_quantile": None,
                "blocked_ratio": None,
                "position_scaling": None,
                "position_p_floor": None,
                "position_p_full": None,
                "position_size_min": None,
                "position_size_max": None,
                "position_p_mid": None,
                "position_k": None,
                "entries_scaled_count": None,
                "entries_scaled_applied_count": None,
                "scale_mean": None,
                "scale_min": None,
                "scale_max": None,
                "scale_bins": None,
                "early_exit_enabled": None,
                "early_exit_count": None,
                "early_exit_rate": None,
                "early_exit_lookback": None,
                "early_exit_p_floor": None,
                "early_exit_bad_k": None,
                "entry_flat_gate_enabled": None,
                "entries_blocked_by_flat_gate": None,
                "pct_flat_gate_block": None,
                "max_flat_proba": None,
                "flat_exit_trigger_count": None,
                "pct_flat_exit_adjusted": None,
                "time_stop_enabled": None,
                "time_stop_bars": None,
                "time_stop_exit_count": None,
                "pct_time_stop_exits": None,
                "partial_tp_enabled": None,
                "partial_tp_threshold": None,
                "partial_tp_ratio": None,
                "partial_tp_count": None,
                "pct_partial_tp": None,
                "partial_tp_avg_pnl": None,
                "exit_reason_counts": None,
                "exit_reason_avg_pnl": None,
            })
            continue
        df_bt, pl, ps = triple
        if len(df_bt) != len(pl):
            min_len = min(len(df_bt), len(pl))
            df_bt = df_bt.tail(min_len).copy()
            pl = np.asarray(pl[-min_len:], dtype=pl.dtype)
            ps = np.asarray(ps[-min_len:], dtype=ps.dtype)

        if calibration_on:
            if calibration_stats_for_meta is None:
                pf_before = np.clip(1.0 - pl.astype(np.float64) - ps.astype(np.float64), _cal_eps, 1.0)
                calibration_stats_for_meta = {
                    "calibration_pl_mean_before": float(np.mean(pl)),
                    "calibration_ps_mean_before": float(np.mean(ps)),
                    "calibration_pf_mean_before": float(np.mean(pf_before)),
                }
            pl, ps = temperature_scale_3class(pl, ps, calibration_T, eps=_cal_eps)
            if calibration_stats_for_meta is not None and "calibration_pl_mean_after" not in calibration_stats_for_meta:
                pf_after = np.clip(1.0 - pl.astype(np.float64) - ps.astype(np.float64), _cal_eps, 1.0)
                calibration_stats_for_meta["calibration_pl_mean_after"] = float(np.mean(pl))
                calibration_stats_for_meta["calibration_ps_mean_after"] = float(np.mean(ps))
                calibration_stats_for_meta["calibration_pf_mean_after"] = float(np.mean(pf_after))

        res_on, err_on = run_backtest_7d(
            args.symbol, args.timeframe, df_bt, pl, ps,
            commission, slippage,
            min_max_proba=args.min_max_proba, max_entropy=args.max_entropy,
            min_hold=args.min_hold, cooldown=args.cooldown,
            regime_filter_enabled=regime_enabled,
            regime_ema_span=args.regime_ema_span,
            regime_rule=regime_rule,
            regime_slope_lookback=args.regime_slope_lookback,
            vol_window=args.vol_window,
            vol_threshold=args.vol_threshold,
            breakout_lookback=args.breakout_lookback,
            breakout_mode=args.breakout_mode,
            slope_threshold=args.slope_threshold,
            q_window=args.q_window,
            q=args.q,
            p_floor=args.p_floor,
            position_scaling_enabled=args.position_scaling != "off",
            position_scaling_mode=args.position_scaling if args.position_scaling != "off" else "linear",
            position_p_floor=args.position_p_floor,
            position_p_full=args.position_p_full,
            position_size_min=args.position_size_min,
            position_size_max=args.position_size_max,
            position_p_mid=args.position_p_mid,
            position_k=args.position_k,
            early_exit_enabled=early_exit_on,
            early_exit_lookback=args.early_exit_lookback,
            early_exit_p_floor=args.early_exit_p_floor,
            early_exit_bad_k=args.early_exit_bad_k,
            entry_flat_gate_enabled=entry_flat_gate_on,
            max_flat_proba=args.max_flat_proba,
            flat_exit_aware_enabled=flat_exit_aware_on,
            flat_exit_threshold=flat_exit_threshold,
            flat_exit_badk_delta=flat_exit_badk_delta,
            time_stop_enabled=time_stop_on,
            time_stop_bars=time_stop_bars,
            partial_tp_enabled=partial_tp_on,
            partial_tp_threshold=partial_tp_threshold,
            partial_tp_ratio=partial_tp_ratio,
            break_even_stop_enabled=break_even_on,
            be_threshold=be_threshold,
            emit_trade_log=emit_trade_log,
        )
        res_off, _ = run_backtest_7d(
            args.symbol, args.timeframe, df_bt, pl, ps,
            0.0, 0.0,
            min_max_proba=args.min_max_proba, max_entropy=args.max_entropy,
            min_hold=args.min_hold, cooldown=args.cooldown,
            regime_filter_enabled=regime_enabled,
            regime_ema_span=args.regime_ema_span,
            regime_rule=regime_rule,
            regime_slope_lookback=args.regime_slope_lookback,
            vol_window=args.vol_window,
            vol_threshold=args.vol_threshold,
            breakout_lookback=args.breakout_lookback,
            breakout_mode=args.breakout_mode,
            slope_threshold=args.slope_threshold,
            q_window=args.q_window,
            q=args.q,
            p_floor=args.p_floor,
            position_scaling_enabled=args.position_scaling != "off",
            position_scaling_mode=args.position_scaling if args.position_scaling != "off" else "linear",
            position_p_floor=args.position_p_floor,
            position_p_full=args.position_p_full,
            position_size_min=args.position_size_min,
            position_size_max=args.position_size_max,
            position_p_mid=args.position_p_mid,
            position_k=args.position_k,
            early_exit_enabled=early_exit_on,
            early_exit_lookback=args.early_exit_lookback,
            early_exit_p_floor=args.early_exit_p_floor,
            early_exit_bad_k=args.early_exit_bad_k,
            entry_flat_gate_enabled=entry_flat_gate_on,
            max_flat_proba=args.max_flat_proba,
            flat_exit_aware_enabled=flat_exit_aware_on,
            flat_exit_threshold=flat_exit_threshold,
            flat_exit_badk_delta=flat_exit_badk_delta,
            time_stop_enabled=time_stop_on,
            time_stop_bars=time_stop_bars,
            partial_tp_enabled=partial_tp_on,
            partial_tp_threshold=partial_tp_threshold,
            partial_tp_ratio=partial_tp_ratio,
            break_even_stop_enabled=break_even_on,
            be_threshold=be_threshold,
            emit_trade_log=emit_trade_log,
        )
        res_on = res_on or {}
        res_off = res_off or {}
        if emit_trade_log and res_on.get("trade_events"):
            _write_trade_log_csv(res_on["trade_events"], run_id, days, trade_log_path)
        cts = res_on.get("cap_trigger_stats") or {}
        row = {
            "span": f"{days}d",
            "days": days,
            "error": None,
            "trades": int(res_on.get("total_trades", 0)),
            "cost_on_return": float(res_on.get("total_return", 0.0)),
            "cost_off_return": float(res_off.get("total_return", 0.0)),
            "win_rate_on": float(res_on.get("win_rate", 0.0)),
            "win_rate_off": float(res_off.get("win_rate", 0.0)),
            "max_drawdown": float(res_on.get("max_drawdown", 0.0)),
            "entries_attempted": res_on.get("entries_attempted"),
            "entries_executed": cts.get("entries_executed"),
            "filter_skip_stats": res_on.get("filter_skip_stats"),
            "regime_enabled": res_on.get("regime_enabled"),
            "regime_rule": res_on.get("regime_rule"),
            "regime_span": res_on.get("regime_span"),
            "regime_slope_lookback": res_on.get("regime_slope_lookback"),
            "vol_window": res_on.get("vol_window"),
            "vol_threshold": res_on.get("vol_threshold"),
            "breakout_lookback": res_on.get("breakout_lookback"),
            "breakout_mode": res_on.get("breakout_mode"),
            "pct_above_ema200": res_on.get("pct_above_ema200"),
            "pct_compress": res_on.get("pct_compress"),
            "pct_vol_slope_block": res_on.get("pct_vol_slope_block"),
            "entries_blocked_by_regime": res_on.get("entries_blocked_by_regime"),
            "entries_blocked_by_regime_price": res_on.get("entries_blocked_by_regime_price"),
            "entries_blocked_by_regime_slope": res_on.get("entries_blocked_by_regime_slope"),
            "entries_blocked_by_regime_vol": res_on.get("entries_blocked_by_regime_vol"),
            "entries_blocked_by_regime_vol_slope": res_on.get("entries_blocked_by_regime_vol_slope"),
            "entries_blocked_by_regime_vol_breakout": res_on.get("entries_blocked_by_regime_vol_breakout"),
            "entries_allowed_on_decompress": res_on.get("entries_allowed_on_decompress"),
            "slope_threshold": res_on.get("slope_threshold"),
            "pct_flat_slope_block": res_on.get("pct_flat_slope_block"),
            "entries_blocked_by_regime_flat_slope": res_on.get("entries_blocked_by_regime_flat_slope"),
            "pct_downtrend_block": res_on.get("pct_downtrend_block"),
            "entries_blocked_by_regime_downtrend": res_on.get("entries_blocked_by_regime_downtrend"),
            "q_window": res_on.get("q_window"),
            "q": res_on.get("q"),
            "p_floor": res_on.get("p_floor"),
            "q_threshold_mean": res_on.get("q_threshold_mean"),
            "pct_proba_quantile_block": res_on.get("pct_proba_quantile_block"),
            "entries_blocked_by_regime_proba_quantile": res_on.get("entries_blocked_by_regime_proba_quantile"),
            "blocked_ratio": res_on.get("blocked_ratio"),
            "position_scaling": res_on.get("position_scaling"),
            "position_p_floor": res_on.get("position_p_floor"),
            "position_p_full": res_on.get("position_p_full"),
            "position_size_min": res_on.get("position_size_min"),
            "position_size_max": res_on.get("position_size_max"),
            "position_p_mid": res_on.get("position_p_mid"),
            "position_k": res_on.get("position_k"),
            "entries_scaled_count": res_on.get("entries_scaled_count"),
            "entries_scaled_applied_count": res_on.get("entries_scaled_applied_count"),
            "scale_mean": res_on.get("scale_mean"),
            "scale_min": res_on.get("scale_min"),
            "scale_max": res_on.get("scale_max"),
            "scale_bins": res_on.get("scale_bins"),
            "early_exit_enabled": res_on.get("early_exit_enabled"),
            "early_exit_count": res_on.get("early_exit_count"),
            "early_exit_rate": res_on.get("early_exit_rate"),
            "early_exit_lookback": res_on.get("early_exit_lookback"),
            "early_exit_p_floor": res_on.get("early_exit_p_floor"),
            "early_exit_bad_k": res_on.get("early_exit_bad_k"),
            "entry_flat_gate_enabled": res_on.get("entry_flat_gate_enabled"),
            "entries_blocked_by_flat_gate": res_on.get("entries_blocked_by_flat_gate"),
            "pct_flat_gate_block": res_on.get("pct_flat_gate_block"),
            "max_flat_proba": res_on.get("max_flat_proba"),
            "flat_exit_trigger_count": res_on.get("flat_exit_trigger_count"),
            "pct_flat_exit_adjusted": res_on.get("pct_flat_exit_adjusted"),
            "time_stop_enabled": res_on.get("time_stop_enabled"),
            "time_stop_bars": res_on.get("time_stop_bars"),
            "time_stop_exit_count": res_on.get("time_stop_exit_count"),
            "pct_time_stop_exits": res_on.get("pct_time_stop_exits"),
            "partial_tp_enabled": res_on.get("partial_tp_enabled"),
            "partial_tp_threshold": res_on.get("partial_tp_threshold"),
            "partial_tp_ratio": res_on.get("partial_tp_ratio"),
            "partial_tp_count": res_on.get("partial_tp_count"),
            "pct_partial_tp": res_on.get("pct_partial_tp"),
            "partial_tp_avg_pnl": res_on.get("partial_tp_avg_pnl"),
            "exit_reason_counts": res_on.get("exit_reason_counts"),
            "exit_reason_avg_pnl": res_on.get("exit_reason_avg_pnl"),
        }
        results.append(row)
        print(f"[VALID] {days}d trades={row['trades']} cost_on={row['cost_on_return']:.4f} cost_off={row['cost_off_return']:.4f} mdd={row['max_drawdown']:.4f}", flush=True)

    # --- year_spans: 충분한 기간 로드 후 구간 필터로 슬라이스 백테스트
    if year_spans:
        load_days = max(max(days_list) if days_list else 365, 730)
        triple, err = get_7d_ohlcv_and_proba(
            model_path, args.symbol, args.timeframe, load_days, log_prefix="[year] ",
        )
        if triple is not None:
            df_bt, pl, ps = triple
            if len(df_bt) != len(pl):
                min_len = min(len(df_bt), len(pl))
                df_bt = df_bt.tail(min_len).copy()
                pl = np.asarray(pl[-min_len:], dtype=pl.dtype)
                ps = np.asarray(ps[-min_len:], dtype=ps.dtype)
            df_bt = df_bt.copy()
            if "timestamp" not in df_bt.columns and hasattr(df_bt.index, "date"):
                df_bt["timestamp"] = pd.to_datetime(df_bt.index)
            df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])

            for start_s, end_s in year_spans:
                start_ts = pd.Timestamp(start_s)
                end_ts = pd.Timestamp(end_s)
                mask = (df_bt["timestamp"] >= start_ts) & (df_bt["timestamp"] <= end_ts)
                if mask.sum() < 50:
                    results.append({
                        "span": start_s[:4],
                        "start": start_s,
                        "end": end_s,
                        "error": f"insufficient data (rows={int(mask.sum())})",
                        "trades": None,
                        "cost_on_return": None,
                        "cost_off_return": None,
                        "win_rate_on": None,
                        "win_rate_off": None,
                        "max_drawdown": None,
                        "entries_attempted": None,
                        "entries_executed": None,
                        "filter_skip_stats": None,
                        "regime_enabled": None,
                        "regime_rule": None,
                        "regime_span": None,
                        "regime_slope_lookback": None,
                    "vol_window": None,
                    "vol_threshold": None,
                    "breakout_lookback": None,
                    "breakout_mode": None,
                    "pct_above_ema200": None,
                    "pct_compress": None,
                    "entries_blocked_by_regime": None,
                    "entries_blocked_by_regime_price": None,
                    "entries_blocked_by_regime_slope": None,
                    "entries_blocked_by_regime_vol": None,
                    "entries_blocked_by_regime_vol_slope": None,
                    "entries_blocked_by_regime_vol_breakout": None,
                    "entries_allowed_on_decompress": None,
                    "slope_threshold": None,
                    "pct_flat_slope_block": None,
                    "entries_blocked_by_regime_flat_slope": None,
                    "pct_downtrend_block": None,
                    "entries_blocked_by_regime_downtrend": None,
                    "q_window": None,
                    "q": None,
                    "p_floor": None,
                    "q_threshold_mean": None,
                    "pct_proba_quantile_block": None,
                    "entries_blocked_by_regime_proba_quantile": None,
                    "blocked_ratio": None,
                    "position_scaling": None,
                    "position_p_floor": None,
                    "position_p_full": None,
                    "position_size_min": None,
                    "position_size_max": None,
                    "position_p_mid": None,
                    "position_k": None,
                    "entries_scaled_count": None,
                    "entries_scaled_applied_count": None,
                    "scale_mean": None,
                    "scale_min": None,
                    "scale_max": None,
                    "scale_bins": None,
                    "pct_vol_slope_block": None,
                    "early_exit_enabled": None,
                    "early_exit_count": None,
                    "early_exit_rate": None,
                    "early_exit_lookback": None,
                    "early_exit_p_floor": None,
                    "early_exit_bad_k": None,
                    "entry_flat_gate_enabled": None,
                    "entries_blocked_by_flat_gate": None,
                    "pct_flat_gate_block": None,
                    "max_flat_proba": None,
                    "flat_exit_trigger_count": None,
                    "pct_flat_exit_adjusted": None,
                    "time_stop_enabled": None,
                    "time_stop_bars": None,
                    "time_stop_exit_count": None,
                    "pct_time_stop_exits": None,
                    "partial_tp_enabled": None,
                    "partial_tp_threshold": None,
                    "partial_tp_ratio": None,
                    "partial_tp_count": None,
                    "pct_partial_tp": None,
                    "partial_tp_avg_pnl": None,
                    "exit_reason_counts": None,
                    "exit_reason_avg_pnl": None,
                })
                    print(f"[VALID] {start_s[:4]} skip: insufficient data", flush=True)
                    continue
                positions = np.where(mask.values)[0]
                df_span = df_bt.iloc[positions].copy().reset_index(drop=True)
                pl_span = pl[positions]
                ps_span = ps[positions]
                if calibration_on:
                    pl_span, ps_span = temperature_scale_3class(pl_span, ps_span, calibration_T, eps=_cal_eps)

                res_on, err_on = run_backtest_7d(
                    args.symbol, args.timeframe, df_span, pl_span, ps_span,
                    commission, slippage,
                    min_max_proba=args.min_max_proba, max_entropy=args.max_entropy,
                    min_hold=args.min_hold, cooldown=args.cooldown,
                    regime_filter_enabled=regime_enabled,
                    regime_ema_span=args.regime_ema_span,
                    regime_rule=regime_rule,
                    regime_slope_lookback=args.regime_slope_lookback,
                    vol_window=args.vol_window,
                    vol_threshold=args.vol_threshold,
                    breakout_lookback=args.breakout_lookback,
                    breakout_mode=args.breakout_mode,
                    slope_threshold=args.slope_threshold,
                    q_window=args.q_window,
                    q=args.q,
                    p_floor=args.p_floor,
                    position_scaling_enabled=args.position_scaling != "off",
                    position_scaling_mode=args.position_scaling if args.position_scaling != "off" else "linear",
                    position_p_floor=args.position_p_floor,
                    position_p_full=args.position_p_full,
                    position_size_min=args.position_size_min,
                    position_size_max=args.position_size_max,
                    position_p_mid=args.position_p_mid,
                    position_k=args.position_k,
                    early_exit_enabled=early_exit_on,
                    early_exit_lookback=args.early_exit_lookback,
                    early_exit_p_floor=args.early_exit_p_floor,
                    early_exit_bad_k=args.early_exit_bad_k,
            entry_flat_gate_enabled=entry_flat_gate_on,
            max_flat_proba=args.max_flat_proba,
            flat_exit_aware_enabled=flat_exit_aware_on,
            flat_exit_threshold=flat_exit_threshold,
                    flat_exit_badk_delta=flat_exit_badk_delta,
                    time_stop_enabled=time_stop_on,
                    time_stop_bars=time_stop_bars,
                    partial_tp_enabled=partial_tp_on,
                    partial_tp_threshold=partial_tp_threshold,
                    partial_tp_ratio=partial_tp_ratio,
                    break_even_stop_enabled=break_even_on,
                    be_threshold=be_threshold,
                    emit_trade_log=emit_trade_log,
                )
                res_off, _ = run_backtest_7d(
                    args.symbol, args.timeframe, df_span, pl_span, ps_span,
                    0.0, 0.0,
                    min_max_proba=args.min_max_proba, max_entropy=args.max_entropy,
                    min_hold=args.min_hold, cooldown=args.cooldown,
                    regime_filter_enabled=regime_enabled,
                    regime_ema_span=args.regime_ema_span,
                    regime_rule=regime_rule,
                    regime_slope_lookback=args.regime_slope_lookback,
                    vol_window=args.vol_window,
                    vol_threshold=args.vol_threshold,
                    breakout_lookback=args.breakout_lookback,
                    breakout_mode=args.breakout_mode,
                    slope_threshold=args.slope_threshold,
                    q_window=args.q_window,
                    q=args.q,
                    p_floor=args.p_floor,
                    position_scaling_enabled=args.position_scaling != "off",
                    position_scaling_mode=args.position_scaling if args.position_scaling != "off" else "linear",
                    position_p_floor=args.position_p_floor,
                    position_p_full=args.position_p_full,
                    position_size_min=args.position_size_min,
                    position_size_max=args.position_size_max,
                    position_p_mid=args.position_p_mid,
                    position_k=args.position_k,
                    early_exit_enabled=early_exit_on,
                    early_exit_lookback=args.early_exit_lookback,
                    early_exit_p_floor=args.early_exit_p_floor,
                    early_exit_bad_k=args.early_exit_bad_k,
                    entry_flat_gate_enabled=entry_flat_gate_on,
                    max_flat_proba=args.max_flat_proba,
                    flat_exit_aware_enabled=flat_exit_aware_on,
                    flat_exit_threshold=flat_exit_threshold,
                    flat_exit_badk_delta=flat_exit_badk_delta,
                    time_stop_enabled=time_stop_on,
            time_stop_bars=time_stop_bars,
            partial_tp_enabled=partial_tp_on,
            partial_tp_threshold=partial_tp_threshold,
            partial_tp_ratio=partial_tp_ratio,
            break_even_stop_enabled=break_even_on,
            be_threshold=be_threshold,
            emit_trade_log=emit_trade_log,
        )
                res_on = res_on or {}
                res_off = res_off or {}
                if emit_trade_log and res_on.get("trade_events"):
                    _write_trade_log_csv(res_on["trade_events"], run_id, start_s[:4], trade_log_path)
                cts = res_on.get("cap_trigger_stats") or {}
                row = {
                    "span": start_s[:4],
                    "start": start_s,
                    "end": end_s,
                    "error": None,
                    "trades": int(res_on.get("total_trades", 0)),
                    "cost_on_return": float(res_on.get("total_return", 0.0)),
                    "cost_off_return": float(res_off.get("total_return", 0.0)),
                    "win_rate_on": float(res_on.get("win_rate", 0.0)),
                    "win_rate_off": float(res_off.get("win_rate", 0.0)),
                    "max_drawdown": float(res_on.get("max_drawdown", 0.0)),
                    "entries_attempted": res_on.get("entries_attempted"),
                    "entries_executed": cts.get("entries_executed"),
                    "filter_skip_stats": res_on.get("filter_skip_stats"),
                    "regime_enabled": res_on.get("regime_enabled"),
                    "regime_rule": res_on.get("regime_rule"),
                    "regime_span": res_on.get("regime_span"),
                    "regime_slope_lookback": res_on.get("regime_slope_lookback"),
                    "vol_window": res_on.get("vol_window"),
                    "vol_threshold": res_on.get("vol_threshold"),
                    "breakout_lookback": res_on.get("breakout_lookback"),
                    "breakout_mode": res_on.get("breakout_mode"),
                    "pct_above_ema200": res_on.get("pct_above_ema200"),
                    "pct_compress": res_on.get("pct_compress"),
                    "pct_vol_slope_block": res_on.get("pct_vol_slope_block"),
                    "entries_blocked_by_regime": res_on.get("entries_blocked_by_regime"),
                    "entries_blocked_by_regime_price": res_on.get("entries_blocked_by_regime_price"),
                    "entries_blocked_by_regime_slope": res_on.get("entries_blocked_by_regime_slope"),
                    "entries_blocked_by_regime_vol": res_on.get("entries_blocked_by_regime_vol"),
                    "entries_blocked_by_regime_vol_slope": res_on.get("entries_blocked_by_regime_vol_slope"),
                    "entries_blocked_by_regime_vol_breakout": res_on.get("entries_blocked_by_regime_vol_breakout"),
                    "entries_allowed_on_decompress": res_on.get("entries_allowed_on_decompress"),
                    "slope_threshold": res_on.get("slope_threshold"),
                    "pct_flat_slope_block": res_on.get("pct_flat_slope_block"),
                    "entries_blocked_by_regime_flat_slope": res_on.get("entries_blocked_by_regime_flat_slope"),
                    "pct_downtrend_block": res_on.get("pct_downtrend_block"),
                    "entries_blocked_by_regime_downtrend": res_on.get("entries_blocked_by_regime_downtrend"),
                    "q_window": res_on.get("q_window"),
                    "q": res_on.get("q"),
                    "p_floor": res_on.get("p_floor"),
                    "q_threshold_mean": res_on.get("q_threshold_mean"),
                    "pct_proba_quantile_block": res_on.get("pct_proba_quantile_block"),
                    "entries_blocked_by_regime_proba_quantile": res_on.get("entries_blocked_by_regime_proba_quantile"),
                    "blocked_ratio": res_on.get("blocked_ratio"),
                    "position_scaling": res_on.get("position_scaling"),
                    "position_p_floor": res_on.get("position_p_floor"),
                    "position_p_full": res_on.get("position_p_full"),
                    "position_size_min": res_on.get("position_size_min"),
                    "position_size_max": res_on.get("position_size_max"),
                    "position_p_mid": res_on.get("position_p_mid"),
                    "position_k": res_on.get("position_k"),
                    "entries_scaled_count": res_on.get("entries_scaled_count"),
                    "entries_scaled_applied_count": res_on.get("entries_scaled_applied_count"),
                    "scale_mean": res_on.get("scale_mean"),
                    "scale_min": res_on.get("scale_min"),
                    "scale_max": res_on.get("scale_max"),
                    "scale_bins": res_on.get("scale_bins"),
                    "early_exit_enabled": res_on.get("early_exit_enabled"),
                    "early_exit_count": res_on.get("early_exit_count"),
                    "early_exit_rate": res_on.get("early_exit_rate"),
                    "early_exit_lookback": res_on.get("early_exit_lookback"),
                    "early_exit_p_floor": res_on.get("early_exit_p_floor"),
                    "early_exit_bad_k": res_on.get("early_exit_bad_k"),
                    "entry_flat_gate_enabled": res_on.get("entry_flat_gate_enabled"),
                    "entries_blocked_by_flat_gate": res_on.get("entries_blocked_by_flat_gate"),
                    "pct_flat_gate_block": res_on.get("pct_flat_gate_block"),
                    "max_flat_proba": res_on.get("max_flat_proba"),
                    "flat_exit_trigger_count": res_on.get("flat_exit_trigger_count"),
                    "pct_flat_exit_adjusted": res_on.get("pct_flat_exit_adjusted"),
                    "time_stop_enabled": res_on.get("time_stop_enabled"),
                    "time_stop_bars": res_on.get("time_stop_bars"),
                    "time_stop_exit_count": res_on.get("time_stop_exit_count"),
                    "pct_time_stop_exits": res_on.get("pct_time_stop_exits"),
                    "partial_tp_enabled": res_on.get("partial_tp_enabled"),
                    "partial_tp_threshold": res_on.get("partial_tp_threshold"),
                    "partial_tp_ratio": res_on.get("partial_tp_ratio"),
                    "partial_tp_count": res_on.get("partial_tp_count"),
                    "pct_partial_tp": res_on.get("pct_partial_tp"),
                    "partial_tp_avg_pnl": res_on.get("partial_tp_avg_pnl"),
                    "exit_reason_counts": res_on.get("exit_reason_counts"),
                    "exit_reason_avg_pnl": res_on.get("exit_reason_avg_pnl"),
                }
                results.append(row)
                print(f"[VALID] {start_s[:4]} trades={row['trades']} cost_on={row['cost_on_return']:.4f} mdd={row['max_drawdown']:.4f}", flush=True)
        else:
            for start_s, end_s in year_spans:
                results.append({
                    "span": start_s[:4],
                    "start": start_s,
                    "end": end_s,
                    "error": err or "load failed",
                    "trades": None,
                    "cost_on_return": None,
                    "cost_off_return": None,
                    "win_rate_on": None,
                    "win_rate_off": None,
                    "max_drawdown": None,
                    "entries_attempted": None,
                    "entries_executed": None,
                    "filter_skip_stats": None,
                    "regime_enabled": None,
                    "regime_rule": None,
                    "regime_span": None,
                    "regime_slope_lookback": None,
                    "vol_window": None,
                    "vol_threshold": None,
                    "breakout_lookback": None,
                    "breakout_mode": None,
                    "pct_above_ema200": None,
                    "pct_compress": None,
                    "entries_blocked_by_regime": None,
                    "entries_blocked_by_regime_price": None,
                    "entries_blocked_by_regime_slope": None,
                    "entries_blocked_by_regime_vol": None,
                    "entries_blocked_by_regime_vol_slope": None,
                    "entries_blocked_by_regime_vol_breakout": None,
                    "entries_allowed_on_decompress": None,
                    "slope_threshold": None,
                    "pct_flat_slope_block": None,
                    "entries_blocked_by_regime_flat_slope": None,
                    "pct_downtrend_block": None,
                    "entries_blocked_by_regime_downtrend": None,
                    "q_window": None,
                    "q": None,
                    "p_floor": None,
                    "q_threshold_mean": None,
                    "pct_proba_quantile_block": None,
                    "entries_blocked_by_regime_proba_quantile": None,
                    "pct_vol_slope_block": None,
                    "blocked_ratio": None,
                    "position_scaling": None,
                    "position_p_floor": None,
                    "position_p_full": None,
                    "position_size_min": None,
                    "position_size_max": None,
                    "position_p_mid": None,
                    "position_k": None,
                    "entries_scaled_count": None,
                    "entries_scaled_applied_count": None,
                    "scale_mean": None,
                    "scale_min": None,
                    "scale_max": None,
                    "scale_bins": None,
                    "early_exit_enabled": None,
                    "early_exit_count": None,
                    "early_exit_rate": None,
                    "early_exit_lookback": None,
                    "early_exit_p_floor": None,
                    "early_exit_bad_k": None,
                    "entry_flat_gate_enabled": None,
                    "entries_blocked_by_flat_gate": None,
                    "pct_flat_gate_block": None,
                    "max_flat_proba": None,
                    "flat_exit_trigger_count": None,
                    "pct_flat_exit_adjusted": None,
                    "time_stop_enabled": None,
                    "time_stop_bars": None,
                    "time_stop_exit_count": None,
                    "pct_time_stop_exits": None,
                    "partial_tp_enabled": None,
                    "partial_tp_threshold": None,
                    "partial_tp_ratio": None,
                    "partial_tp_count": None,
                    "pct_partial_tp": None,
                    "partial_tp_avg_pnl": None,
                    "exit_reason_counts": None,
                    "exit_reason_avg_pnl": None,
                })

    # --- Summary & GO/TUNE
    days_results = [r for r in results if r.get("days") is not None and r.get("error") is None]
    year_results = [r for r in results if "start" in r and r.get("error") is None]
    cost_on_vals = [r["cost_on_return"] for r in results if r.get("cost_on_return") is not None]
    mdd_vals = [r["max_drawdown"] for r in results if r.get("max_drawdown") is not None]
    trades_365 = None
    for r in days_results:
        if r.get("days") == 365:
            trades_365 = r.get("trades")
            break

    n_days = len(days_results)
    cost_on_positive_ratio = (sum(1 for r in days_results if r.get("cost_on_return", -1) >= 0) / n_days) if n_days else 0.0
    worst_mdd = max(mdd_vals) if mdd_vals else 0.0
    avg_cost_on = sum(cost_on_vals) / len(cost_on_vals) if cost_on_vals else 0.0
    avg_mdd = sum(mdd_vals) / len(mdd_vals) if mdd_vals else 0.0

    best_span = None
    worst_span = None
    if cost_on_vals:
        by_cost = [r for r in results if r.get("cost_on_return") is not None]
        by_cost.sort(key=lambda x: x["cost_on_return"], reverse=True)
        best_span = by_cost[0].get("span")
        worst_span = by_cost[-1].get("span")

    pass_cost = cost_on_positive_ratio >= COST_ON_POSITIVE_RATIO_MIN
    pass_mdd = worst_mdd <= MDD_MAX
    pass_trades = (trades_365 is not None and trades_365 >= TRADES_365_MIN) or trades_365 is None
    decision = "GO" if (pass_cost and pass_mdd and pass_trades) else "TUNE"

    tune_hint = ""
    if decision == "TUNE":
        if not pass_cost and cost_on_vals:
            tune_hint = "cost_on 음수인데 cost_off 양수 → 수수료/슬리피지 민감 → trades 더 줄이는 방향 (min_max_proba↑ 또는 max_entropy↓)"
        elif not pass_mdd:
            tune_hint = "MDD 급증 → cooldown↑ 또는 min_hold↑ 후보"
        elif not pass_trades and trades_365 is not None:
            tune_hint = "365d trades 과소 → 운빨 가능성; 필터 완화 검토 또는 표본 확대"
        else:
            tune_hint = "cost_on은 양수인데 trades 과다 시 min_max_proba↑ 또는 entropy↓"

    pass_flags = {
        "cost_on_positive_ratio": cost_on_positive_ratio,
        "mdd_ok_ratio": 1.0 if pass_mdd else 0.0,
        "trades_ok_ratio": 1.0 if pass_trades else 0.0,
    }
    first_regime = next((r for r in results if r.get("blocked_ratio") is not None), None)
    overblock_warning = first_regime is not None and float(first_regime.get("blocked_ratio", 0)) >= 0.9
    overblock_nogo = first_regime is not None and float(first_regime.get("blocked_ratio", 0)) >= 0.95
    summary = {
        "best_span_by_cost_on": best_span,
        "worst_span_by_cost_on": worst_span,
        "avg_cost_on": avg_cost_on,
        "avg_mdd": avg_mdd,
        "pass_flags": pass_flags,
        "decision": decision,
        "tune_hint": tune_hint,
        "overblock_warning": overblock_warning,
        "overblock_nogo": overblock_nogo,
    }

    meta = {
        "run_id": run_id,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "id": args.id,
        "min_max_proba": args.min_max_proba,
        "max_entropy": args.max_entropy,
        "min_hold": args.min_hold,
        "cooldown": args.cooldown,
        "regime_enabled": regime_enabled,
        "regime_rule": regime_rule,
        "regime_span": args.regime_ema_span,
        "regime_slope_lookback": args.regime_slope_lookback,
        "vol_window": args.vol_window,
        "vol_threshold": args.vol_threshold,
        "breakout_lookback": args.breakout_lookback,
        "breakout_mode": args.breakout_mode,
        "slope_threshold": args.slope_threshold,
        "q_window": args.q_window,
        "q": args.q,
        "p_floor": args.p_floor,
        "position_scaling": args.position_scaling,
        "position_p_floor": args.position_p_floor,
        "position_p_full": args.position_p_full,
        "position_size_min": args.position_size_min,
        "position_size_max": args.position_size_max,
        "position_p_mid": args.position_p_mid,
        "position_k": args.position_k,
        "early_exit": args.early_exit,
        "early_exit_enabled": early_exit_on,
        "early_exit_lookback": args.early_exit_lookback,
        "early_exit_p_floor": args.early_exit_p_floor,
        "early_exit_bad_k": args.early_exit_bad_k,
        "entry_flat_gate": args.entry_flat_gate,
        "entry_flat_gate_enabled": entry_flat_gate_on,
        "max_flat_proba": args.max_flat_proba,
        "flat_exit_aware_enabled": flat_exit_aware_on,
        "flat_exit_threshold": flat_exit_threshold,
        "flat_exit_badk_delta": flat_exit_badk_delta,
        "time_stop_enabled": time_stop_on,
        "time_stop_bars": time_stop_bars,
        "partial_tp": args.partial_tp,
        "partial_tp_enabled": partial_tp_on,
        "partial_tp_threshold": partial_tp_threshold,
        "partial_tp_ratio": partial_tp_ratio,
        "partial_tp_count_365": None,  # 아래에서 365d 결과로 채움
        "break_even_stop": args.break_even_stop,
        "break_even_stop_enabled": break_even_on,
        "be_threshold": be_threshold if break_even_on else None,
        "calibration": args.calibration,
        "calibration_T": calibration_T,
        "commission": commission,
        "slippage": slippage,
        "created_at": datetime.now().isoformat(),
        "days_list": days_list,
        "year_spans": year_spans,
        "out_json_path": str(out_json),
        "out_md_path": str(out_md),
        "start_date_30": pinned_start_dates.get(30) if pinned_end_date else None,
        "start_date_365": pinned_start_dates.get(365) if pinned_end_date else None,
        "end_date": pinned_end_date,
    }
    if calibration_stats_for_meta:
        meta.update(calibration_stats_for_meta)
    row_365 = next((r for r in results if r.get("days") == 365), None)
    if row_365 is not None and row_365.get("partial_tp_count") is not None:
        meta["partial_tp_count_365"] = row_365["partial_tp_count"]

    payload = {"meta": meta, "results": results, "summary": summary}

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(type(obj))

    DIAG.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    # MD
    lines = [
        "# TCN 후보 기간 확장 검증",
        f"생성: {datetime.now().isoformat()}",
        "",
        "## 실행 파라미터",
        f"- id={args.id}, symbol={args.symbol}, timeframe={args.timeframe}",
        f"- min_max_proba={args.min_max_proba}, max_entropy={args.max_entropy}, min_hold={args.min_hold}, cooldown={args.cooldown}",
        f"- regime_filter={args.regime_filter}, regime_ema_span={args.regime_ema_span}, regime_slope_lookback={args.regime_slope_lookback}, vol_window={args.vol_window}, vol_threshold={args.vol_threshold}, breakout_lookback={args.breakout_lookback}, breakout_mode={args.breakout_mode}, slope_threshold={args.slope_threshold}",
        f"- position_scaling={args.position_scaling}, position_p_floor={args.position_p_floor}, position_p_full={args.position_p_full}, position_size_min={args.position_size_min}, position_size_max={args.position_size_max}",
        f"- early_exit={args.early_exit}, early_exit_lookback={args.early_exit_lookback}, early_exit_p_floor={args.early_exit_p_floor}, early_exit_bad_k={args.early_exit_bad_k}",
        f"- days_list={days_list}, year_spans={year_spans or 'None'}",
        "",
        "## span별 결과",
        "| span | trades | cost_on_return | cost_off_return | max_drawdown | win_rate_on | win_rate_off | error |",
        "|------|--------|----------------|----------------|--------------|-------------|--------------|-------|",
    ]
    for r in results:
        span = r.get("span", "-")
        tr = r.get("trades") if r.get("trades") is not None else "-"
        co = f"{r['cost_on_return']:.4f}" if r.get("cost_on_return") is not None else "-"
        cf = f"{r['cost_off_return']:.4f}" if r.get("cost_off_return") is not None else "-"
        mdd = f"{r['max_drawdown']:.4f}" if r.get("max_drawdown") is not None else "-"
        wo = f"{r['win_rate_on']:.2f}" if r.get("win_rate_on") is not None else "-"
        wf = f"{r['win_rate_off']:.2f}" if r.get("win_rate_off") is not None else "-"
        err = (r.get("error") or "-")[:40] if r.get("error") else "-"
        lines.append(f"| {span} | {tr} | {co} | {cf} | {mdd} | {wo} | {wf} | {err} |")
    lines.append("")
    if regime_enabled:
        first_with_regime = next((r for r in results if r.get("entries_blocked_by_regime") is not None), None)
        if first_with_regime:
            lines.append("## Regime 필터")
            lines.append(f"- regime_rule: {first_with_regime.get('regime_rule')}")
            if first_with_regime.get("regime_rule") == "vol_compress":
                lines.append(f"- vol_window: {first_with_regime.get('vol_window')}, vol_threshold: {first_with_regime.get('vol_threshold')}")
                lines.append(f"- pct_compress: {first_with_regime.get('pct_compress')}")
                lines.append(f"- entries_blocked_by_regime_vol: {first_with_regime.get('entries_blocked_by_regime_vol')}")
            elif first_with_regime.get("regime_rule") == "vol_slope":
                lines.append(f"- vol_window: {first_with_regime.get('vol_window')}, vol_threshold: {first_with_regime.get('vol_threshold')}, regime_slope_lookback: {first_with_regime.get('regime_slope_lookback')}")
                lines.append(f"- pct_vol_slope_block: {first_with_regime.get('pct_vol_slope_block')}")
                lines.append(f"- entries_blocked_by_regime_vol_slope: {first_with_regime.get('entries_blocked_by_regime_vol_slope')}")
            elif first_with_regime.get("regime_rule") == "vol_breakout":
                lines.append(f"- vol_window: {first_with_regime.get('vol_window')}, vol_threshold: {first_with_regime.get('vol_threshold')}, breakout_lookback: {first_with_regime.get('breakout_lookback')}, breakout_mode: {first_with_regime.get('breakout_mode')}")
                lines.append(f"- pct_compress: {first_with_regime.get('pct_compress')}")
                lines.append(f"- entries_blocked_by_regime_vol: {first_with_regime.get('entries_blocked_by_regime_vol')}")
                lines.append(f"- entries_blocked_by_regime_vol_breakout: {first_with_regime.get('entries_blocked_by_regime_vol_breakout')}")
                lines.append(f"- entries_allowed_on_decompress: {first_with_regime.get('entries_allowed_on_decompress')}")
            elif first_with_regime.get("regime_rule") == "flat_slope":
                lines.append(f"- regime_span: {first_with_regime.get('regime_span')}, regime_slope_lookback: {first_with_regime.get('regime_slope_lookback')}, slope_threshold: {first_with_regime.get('slope_threshold')}")
                lines.append(f"- pct_flat_slope_block: {first_with_regime.get('pct_flat_slope_block')}")
                lines.append(f"- entries_blocked_by_regime_flat_slope: {first_with_regime.get('entries_blocked_by_regime_flat_slope')}")
            elif first_with_regime.get("regime_rule") == "downtrend_block":
                lines.append(f"- regime_span: {first_with_regime.get('regime_span')}, regime_slope_lookback: {first_with_regime.get('regime_slope_lookback')}")
                lines.append(f"- pct_downtrend_block: {first_with_regime.get('pct_downtrend_block')}")
                lines.append(f"- entries_blocked_by_regime_downtrend: {first_with_regime.get('entries_blocked_by_regime_downtrend')}")
            elif first_with_regime.get("regime_rule") == "proba_quantile":
                lines.append(f"- q_window: {first_with_regime.get('q_window')}, q: {first_with_regime.get('q')}, p_floor: {first_with_regime.get('p_floor')}")
                lines.append(f"- q_threshold_mean: {first_with_regime.get('q_threshold_mean')}")
                lines.append(f"- pct_proba_quantile_block: {first_with_regime.get('pct_proba_quantile_block')}")
                lines.append(f"- entries_blocked_by_regime_proba_quantile: {first_with_regime.get('entries_blocked_by_regime_proba_quantile')}")
                lines.append(f"- blocked_ratio: {first_with_regime.get('blocked_ratio')}")
            else:
                lines.append(f"- entry_price_basis: close, ema_align_basis: same_bar")
                lines.append(f"- pct_above_ema200: {first_with_regime.get('pct_above_ema200')}")
                lines.append(f"- entries_blocked_by_regime_price: {first_with_regime.get('entries_blocked_by_regime_price')}")
                lines.append(f"- entries_blocked_by_regime_slope: {first_with_regime.get('entries_blocked_by_regime_slope')}")
            lines.append(f"- entries_blocked_by_regime: {first_with_regime.get('entries_blocked_by_regime')}")
            lines.append(f"- blocked_ratio: {first_with_regime.get('blocked_ratio')}")
            br = first_with_regime.get("blocked_ratio")
            if br is not None and float(br) >= 0.9:
                lines.append("- **경고**: blocked_ratio >= 90% → 레짐 과차단 가능성")
            if br is not None and float(br) >= 0.95:
                lines.append("- **NO-GO**: blocked_ratio >= 95% → 과차단")
            lines.append("")
    # Position Scaling 절
    first_with_scaling = next((r for r in results if r.get("position_scaling") is not None), None)
    if first_with_scaling and first_with_scaling.get("position_scaling") != "off":
        lines.append("## Position Scaling")
        lines.append(f"- position_scaling: {first_with_scaling.get('position_scaling')}")
        lines.append(f"- p_floor: {first_with_scaling.get('position_p_floor')}, p_full: {first_with_scaling.get('position_p_full')}, size_min: {first_with_scaling.get('position_size_min')}, size_max: {first_with_scaling.get('position_size_max')}")
        if first_with_scaling.get("position_scaling") == "sigmoid":
            lines.append(f"- p_mid: {first_with_scaling.get('position_p_mid')}, k: {first_with_scaling.get('position_k')}")
        lines.append(f"- scale_mean: {first_with_scaling.get('scale_mean')}, scale_min: {first_with_scaling.get('scale_min')}, scale_max: {first_with_scaling.get('scale_max')}")
        lines.append(f"- entries_scaled_count: {first_with_scaling.get('entries_scaled_count')}, entries_scaled_applied_count: {first_with_scaling.get('entries_scaled_applied_count')}")
        lines.append(f"- scale_bins: {first_with_scaling.get('scale_bins')}")
        lines.append("")
    elif first_with_scaling and first_with_scaling.get("position_scaling") == "off":
        lines.append("## Position Scaling")
        lines.append("- OFF")
        lines.append("")
    lines.append("## 결론")
    lines.append(f"- **decision**: {decision}")
    lines.append(f"- cost_on >= 0 비율: {cost_on_positive_ratio:.0%} (목표 >={COST_ON_POSITIVE_RATIO_MIN:.0%})")
    lines.append(f"- 최악 MDD: {worst_mdd:.4f} (한도 {MDD_MAX})")
    lines.append(f"- 365d trades: {trades_365} (최소 {TRADES_365_MIN})")
    if tune_hint:
        lines.append(f"- 튜닝 힌트: {tune_hint}")
    lines.append("")
    lines.append(f"저장: {out_md} | {out_json}")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 콘솔 요약 (그대로 출력)
    print("")
    print("=" * 60)
    print("TCN 후보 검증 요약")
    print("=" * 60)
    print(f"id={args.id} min_max_proba={args.min_max_proba} max_entropy={args.max_entropy} min_hold={args.min_hold} cooldown={args.cooldown} regime_filter={args.regime_filter}")
    for r in results:
        if r.get("error"):
            print(f"  {r.get('span', '?')}: error={r['error'][:50]}")
        else:
            print(f"  {r.get('span', '?')}: trades={r.get('trades')} cost_on={r.get('cost_on_return'):.4f} cost_off={r.get('cost_off_return'):.4f} mdd={r.get('max_drawdown'):.4f}")
    print("")
    n_ok = sum(1 for r in results if r.get("cost_on_return") is not None and r["cost_on_return"] >= 0)
    print(f"cost_on_return >= 0: {n_ok}/{len(results)} span")
    if worst_span and cost_on_vals:
        worst_row = next((r for r in results if r.get("span") == worst_span and r.get("cost_on_return") is not None), None)
        if worst_row:
            print(f"최악 span: {worst_span} (cost_on={worst_row['cost_on_return']:.4f}, mdd={worst_row['max_drawdown']:.4f}, trades={worst_row.get('trades')})")
    print(f"평균 cost_on={avg_cost_on:.4f}, 평균 mdd={avg_mdd:.4f}")
    print(f"결론: {decision}" + (f" — {tune_hint}" if tune_hint else ""))
    print("")
    print(f"저장: {out_json}, {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
