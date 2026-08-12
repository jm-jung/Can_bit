#!/usr/bin/env python3
"""
Phase D11: Directional gap + Flat suppression filter experiment.

- D11-A: min_directional_gap only (LONG: p_long-p_short >= gap, SHORT: p_short-p_long >= gap)
- D11-B: max_flat_entry_proba only (entry when p_flat <= threshold)
- D11-C: combination (gap + flat)

Usage:
  python scripts/run_phase_d11_filter_sweep.py --help
  python scripts/run_phase_d11_filter_sweep.py  # baseline + sample sweep
  python scripts/run_phase_d11_filter_sweep.py --min-directional-gap 0.04 --days 720
  python scripts/run_phase_d11_filter_sweep.py --max-flat-entry-proba 0.35 --days 365
  python scripts/run_phase_d11_filter_sweep.py --sweep full  # full A/B/C sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
END_DATE = "2026-03-03"
DAYS_LIST = [180, 365, 720]
MIN_MAX_PROBA = 0.575
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8
COMMISSION = 0.0009
SLIPPAGE = 0.0001

DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

# D11-A: directional gap only
D11_A_GAPS = [0.02, 0.04, 0.06, 0.08, 0.10]
# D11-B: flat suppression only
D11_B_FLATS = [0.25, 0.30, 0.35, 0.40, 0.45]
# D11-C: combinations (min_directional_gap, max_flat_entry_proba)
D11_C_COMBOS = [(0.04, 0.35), (0.04, 0.40), (0.06, 0.35), (0.06, 0.40)]


def _get_ohlcv_and_proba(symbol: str, timeframe: str, days: int, end_date: str, feature_config, use_events: bool, model_path: Path):
    from src.services.ohlcv_service import load_ohlcv_df
    from src.indicators.basic import add_basic_indicators
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel

    df = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
    df = add_basic_indicators(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_ts = pd.Timestamp(end_date).tz_localize("UTC") - pd.Timedelta(days=days)
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
    if df["timestamp"].dt.tz is None:
        start_ts, end_ts = start_ts.tz_localize(None), end_ts.tz_localize(None)
    df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    if len(df) < 100:
        return (None, f"rows={len(df)} < 100")
    features = build_feature_frame(df, symbol=symbol, timeframe=timeframe, feature_config=feature_config)
    features = features.dropna()
    if len(features) < 60:
        return (None, f"features rows={len(features)} < 60")
    model = TCNSignalModel(model_path=model_path, use_events=use_events)
    if not model.is_loaded():
        return (None, "TCN model failed to load")
    pl_arr, ps_arr = model.predict_proba_batch(features=features, symbol=symbol, timeframe=timeframe, batch_size=512)
    pl = np.asarray(pl_arr, dtype=np.float32)
    ps = np.asarray(ps_arr, dtype=np.float32)
    if "close" in features.columns and "high" in features.columns and "low" in features.columns:
        df_bt = features[["close", "high", "low"]].copy()
        if isinstance(features.index, pd.DatetimeIndex):
            df_bt["timestamp"] = features.index
            df_bt = df_bt.reset_index(drop=True)
        else:
            df_bt["timestamp"] = features["timestamp"].values
    else:
        df_indexed = df.set_index("timestamp")
        df_bt = df_indexed.reindex(features.index)[["close", "high", "low"]].dropna(how="any")
        if len(df_bt) == 0:
            return (None, "reindex df_bt empty")
        common = df_bt.index
        features = features.loc[common]
        pos = [list(features.index).index(i) for i in common]
        pl = np.asarray([pl[i] for i in pos], dtype=np.float32)
        ps = np.asarray([ps[i] for i in pos], dtype=np.float32)
        df_bt = df_bt.reset_index()
        if df_bt.columns[0] != "timestamp":
            df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})
    if "timestamp" not in df_bt.columns and isinstance(features.index, pd.DatetimeIndex):
        df_bt = df_bt.reset_index()
        if len(df_bt.columns) > 0 and df_bt.columns[0] != "timestamp":
            df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})
    if isinstance(df_bt.index, pd.DatetimeIndex) and "timestamp" in df_bt.columns:
        df_bt = df_bt.reset_index(drop=True)
    if len(df_bt) != len(pl):
        window_size = int(getattr(model, "window_size", 60))
        proba_len = len(pl)
        feat_index = pd.DatetimeIndex(pd.to_datetime(features["timestamp"] if "timestamp" in features.columns else features.index))
        align_ts = feat_index[window_size : window_size + proba_len]
        if len(align_ts) != proba_len:
            return (None, "align_ts mismatch")
        proba_df = pd.DataFrame({"timestamp": pd.to_datetime(align_ts.values), "pl": pl, "ps": ps})
        proba_df = proba_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df_bt = df_bt.copy()
        if "timestamp" not in df_bt.columns and isinstance(df_bt.index, pd.DatetimeIndex):
            df_bt = df_bt.reset_index()
            if df_bt.columns[0] != "timestamp":
                df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})
        df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
        df_bt = df_bt.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df_bt_aligned = df_bt[df_bt["timestamp"].isin(proba_df["timestamp"])].copy()
        df_bt_aligned = df_bt_aligned.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        joined = df_bt_aligned.merge(proba_df, on="timestamp", how="inner", validate="one_to_one")
        df_bt = joined[["timestamp", "close", "high", "low"]].copy()
        pl = joined["pl"].values.astype(np.float32)
        ps = joined["ps"].values.astype(np.float32)
    if len(df_bt) != len(pl):
        return (None, "len mismatch")
    return ((df_bt, pl, ps), None)


def _slice_period(df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray, days: int, end_date: str):
    df_bt = df_bt.copy()
    df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
    if df_bt["timestamp"].dt.tz is None:
        end_ts = end_ts.tz_localize(None)
    start_ts = end_ts - pd.Timedelta(days=days)
    mask = (df_bt["timestamp"] >= start_ts) & (df_bt["timestamp"] < end_ts)
    df_slice = df_bt.loc[mask].copy()
    if len(df_slice) < 60:
        return None
    idx = df_bt.index[mask]
    pos = [list(df_bt.index).index(i) for i in idx]
    pl_slice = np.asarray([pl[i] for i in pos], dtype=np.float32)
    ps_slice = np.asarray([ps[i] for i in pos], dtype=np.float32)
    return (df_slice.reset_index(drop=True), pl_slice, ps_slice)


def _run_backtest(symbol: str, timeframe: str, df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray,
                  min_directional_gap: float | None, max_flat_entry_proba: float | None):
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
    return run_backtest_7d(
        symbol, timeframe, df_bt, pl, ps, COMMISSION, SLIPPAGE,
        min_max_proba=MIN_MAX_PROBA, max_entropy=MAX_ENTROPY,
        min_proba_gap=0.0,
        min_directional_gap=min_directional_gap,
        max_flat_entry_proba=max_flat_entry_proba,
        min_hold=MIN_HOLD, cooldown=COOLDOWN,
        regime_filter_enabled=False, position_scaling_enabled=False,
        early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        emit_trade_log=False,
    )


def _mean_hold(trade_events: list) -> float:
    if not trade_events:
        return 0.0
    bars = [e.get("bars_held") or e.get("holding_bars") or 0 for e in trade_events if e.get("event", "").startswith("EXIT")]
    bars = [b for b in bars if b > 0]
    return float(np.mean(bars)) if bars else 0.0


def run_one(period_data: dict, days: int, min_directional_gap: float | None, max_flat_entry_proba: float | None) -> dict | None:
    df_bt, pl, ps = period_data[days]
    res, err = _run_backtest(SYMBOL, TIMEFRAME, df_bt, pl, ps, min_directional_gap, max_flat_entry_proba)
    if err:
        return None
    cap = res.get("cap_trigger_stats") or {}
    return {
        "days": days,
        "min_directional_gap": min_directional_gap,
        "max_flat_entry_proba": max_flat_entry_proba,
        "cost_on": res.get("total_return"),
        "MDD": res.get("max_drawdown"),
        "trades": res.get("total_trades", 0),
        "win_rate": res.get("win_rate"),
        "mean_hold": _mean_hold(res.get("trade_events") or []),
        "entries_attempted": res.get("entries_attempted"),
        "entries_executed": cap.get("entries_executed"),
        "directional_gap_fail_count": res.get("directional_gap_fail_count", 0),
        "flat_suppression_fail_count": res.get("flat_suppression_fail_count", 0),
    }


def main() -> int:
    from src.features.ml_feature_config import MLFeatureConfig

    parser = argparse.ArgumentParser(description="Phase D11: Directional gap + Flat suppression filter sweep")
    parser.add_argument("--min-directional-gap", type=float, default=None, help="D11 directional gap (LONG: p_long-p_short >= gap)")
    parser.add_argument("--max-flat-entry-proba", type=float, default=None, help="D11 flat suppression (entry when p_flat <= this)")
    parser.add_argument("--days", type=int, default=None, choices=[180, 365, 720], help="Single window; if omitted run 180/365/720")
    parser.add_argument("--sweep", type=str, default="sample", choices=["none", "sample", "full"],
                        help="none=baseline only, sample=baseline+few runs, full=all A/B/C sweeps")
    args = parser.parse_args()

    config = MLFeatureConfig.from_preset("base")
    config.use_event_features = True
    model_id = os.environ.get("PHASE_D10_MODEL_ID", "h15_t0p004")
    model_path = MODELS_DIR / f"tcn_{model_id}.pt"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    # Load 720d once, slice 180/365
    print("[D11] 720d load + inference (once) ...", flush=True)
    triple_720, err = _get_ohlcv_and_proba(SYMBOL, TIMEFRAME, 720, END_DATE, config, True, model_path)
    if err:
        print(f"[D11] ERROR: {err}", file=sys.stderr)
        return 1
    period_data = {720: triple_720}
    df_720, pl_720, ps_720 = triple_720
    for d in [180, 365]:
        sl = _slice_period(df_720, pl_720, ps_720, d, END_DATE)
        if sl is None:
            print(f"[D11] slice {d}d failed", file=sys.stderr)
            return 1
        period_data[d] = sl

    windows = [args.days] if args.days else DAYS_LIST
    all_results = []
    baseline_720 = None

    # Baseline (no D11 filters)
    print("[D11] baseline (no D11 filters) ...", flush=True)
    for days in windows:
        row = run_one(period_data, days, None, None)
        if row:
            row["run_id"] = "phase_d11_baseline"
            all_results.append(row)
            if days == 720:
                baseline_720 = row

    if args.sweep == "none" and not args.min_directional_gap and not args.max_flat_entry_proba:
        # Single baseline run only
        _write_summary(all_results, baseline_720, [])
        print("[D11] baseline only done.", flush=True)
        return 0

    # Single custom run from CLI
    if args.min_directional_gap is not None or args.max_flat_entry_proba is not None:
        for days in windows:
            row = run_one(period_data, days, args.min_directional_gap, args.max_flat_entry_proba)
            if row:
                row["run_id"] = f"d11_gap{args.min_directional_gap}_flat{args.max_flat_entry_proba}"
                all_results.append(row)
        _write_summary(all_results, baseline_720, [])
        return 0

    # Sweep: sample or full
    sweep_runs = []
    if args.sweep in ("sample", "full"):
        # D11-A sample: gap 0.04 only (full = all D11_A_GAPS)
        gaps = D11_A_GAPS if args.sweep == "full" else [0.04]
        for g in gaps:
            for days in windows:
                row = run_one(period_data, days, g, None)
                if row:
                    row["run_id"] = f"d11_a_gap{g}"
                    all_results.append(row)
                    sweep_runs.append(row)

        # D11-B sample: flat 0.35 only (full = all D11_B_FLATS)
        flats = D11_B_FLATS if args.sweep == "full" else [0.35]
        for f in flats:
            for days in windows:
                row = run_one(period_data, days, None, f)
                if row:
                    row["run_id"] = f"d11_b_flat{f}"
                    all_results.append(row)
                    sweep_runs.append(row)

        # D11-C sample: (0.04, 0.35) only (full = all D11_C_COMBOS)
        combos = D11_C_COMBOS if args.sweep == "full" else [(0.04, 0.35)]
        for gap, flat in combos:
            for days in windows:
                row = run_one(period_data, days, gap, flat)
                if row:
                    row["run_id"] = f"d11_c_gap{gap}_flat{flat}"
                    all_results.append(row)
                    sweep_runs.append(row)

    _write_summary(all_results, baseline_720, sweep_runs)
    print(f"[D11] done. results={len(all_results)}", flush=True)
    return 0


def _write_summary(all_results: list, baseline_720: dict | None, sweep_runs: list):
    out_json = BACKTESTS_DIR / "phase_d11_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "baseline_720": baseline_720, "created_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    print(f"[D11] wrote {out_json}", flush=True)

    # Markdown summary
    bl = baseline_720 or {}
    lines = [
        "# Phase D11: Directional gap + Flat suppression summary",
        "",
        "## Baseline (no D11)",
        f"- 720d cost_on: {bl.get('cost_on')}",
        f"- 720d MDD: {bl.get('MDD')}",
        f"- 720d trades: {bl.get('trades')}",
        f"- entries_attempted: {bl.get('entries_attempted')}, entries_executed: {bl.get('entries_executed')}",
        "",
        "## All runs (by run_id)",
        "| run_id | days | min_directional_gap | max_flat_entry_proba | cost_on | MDD | trades | entries_attempted | entries_executed | directional_gap_fail | flat_suppression_fail |",
        "|--------|------|---------------------|----------------------|---------|-----|--------|-------------------|------------------|----------------------|------------------------|",
    ]
    for r in all_results:
        ea = r.get("entries_attempted") if r.get("entries_attempted") is not None else "-"
        ee = r.get("entries_executed") if r.get("entries_executed") is not None else "-"
        c = r.get("cost_on")
        cstr = f"{c:.4f}" if c is not None else "-"
        mdd = r.get("MDD")
        mstr = f"{mdd:.4f}" if mdd is not None else "-"
        lines.append(
            f"| {r.get('run_id', '')} | {r.get('days', '')} | {r.get('min_directional_gap')} | {r.get('max_flat_entry_proba')} | {cstr} | {mstr} | {r.get('trades', '')} | {ea} | {ee} | {r.get('directional_gap_fail_count', 0)} | {r.get('flat_suppression_fail_count', 0)} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "- If directional_gap_fail_count or flat_suppression_fail_count is large but entries_executed barely changed vs baseline, filter has low practical effect (D10.1-like).",
        "- Compare 720d cost_on and trades vs baseline for improvement.",
    ])
    report_path = REPORTS_DIR / "phase_d11_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[D11] wrote {report_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
