#!/usr/bin/env python3
"""
Phase D13: Directional edge + flat margin + confidence decision rule sweep.

- Baseline: decision_mode=argmax (optional --run-baseline)
- D13: decision_mode=directional_edge_margin_conf

Sweeps:
  1) edge: min_directional_edge in [0.04, 0.08, 0.12, 0.16], flat_margin=0.03, confidence=0.40
  2) flat: edge=0.08, min_flat_margin in [0.00, 0.03, 0.05, 0.07], confidence=0.40
  3) conf: edge=0.08, flat_margin=0.03, min_confidence in [0.35, 0.40, 0.45, 0.50]
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

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
END_DATE = "2026-03-03"
DAYS_LIST = [180, 365, 720]
COMMISSION = 0.0009
SLIPPAGE = 0.0001
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests"
DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)


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
        window_size = 60
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


def _slice_period(df_bt, pl, ps, days: int, end_date: str):
    df_bt = df_bt.copy()
    df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=days)
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


def _run_backtest(
    symbol: str,
    timeframe: str,
    df_bt,
    pl,
    ps,
    decision_mode: str,
    min_directional_edge: float | None,
    min_flat_margin: float | None,
    min_confidence: float | None,
):
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    return run_backtest_7d(
        symbol,
        timeframe,
        df_bt,
        pl,
        ps,
        COMMISSION,
        SLIPPAGE,
        min_max_proba=None,
        max_entropy=None,
        min_proba_gap=0.0,
        min_directional_gap=None,
        max_flat_entry_proba=None,
        decision_mode=decision_mode,
        min_directional_edge=min_directional_edge,
        require_direction_gt_flat=True,
        min_side_flat_margin=min_flat_margin,
        min_confidence=min_confidence,
        use_legacy_entry_gates_with_directional=False,
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
        regime_filter_enabled=False,
        position_scaling_enabled=False,
        early_exit_enabled=True,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        emit_trade_log=False,
    )


def _mean_hold(trade_events: list) -> float:
    if not trade_events:
        return 0.0
    bars = [e.get("bars_held") or e.get("holding_bars") or 0 for e in trade_events if e.get("event", "").startswith("EXIT")]
    bars = [b for b in bars if b > 0]
    return float(np.mean(bars)) if bars else 0.0


def run_one(
    period_data: dict,
    days: int,
    decision_mode: str,
    min_directional_edge: float | None,
    min_flat_margin: float | None,
    min_confidence: float | None,
) -> dict | None:
    if days not in period_data:
        return None
    df_bt, pl, ps = period_data[days]
    res, err = _run_backtest(
        SYMBOL,
        TIMEFRAME,
        df_bt,
        pl,
        ps,
        decision_mode=decision_mode,
        min_directional_edge=min_directional_edge,
        min_flat_margin=min_flat_margin,
        min_confidence=min_confidence,
    )
    if err:
        print(f"[D13] Backtest failed (days={days}, mode={decision_mode}): {err[:500]}", file=sys.stderr)
        return None
    cap = res.get("cap_trigger_stats") or {}
    return {
        "days": days,
        "decision_mode": res.get("decision_mode", decision_mode),
        "min_directional_edge": min_directional_edge,
        "min_flat_margin": min_flat_margin,
        "min_confidence": min_confidence,
        "cost_on": res.get("total_return"),
        "MDD": res.get("max_drawdown"),
        "trades": res.get("total_trades", 0),
        "win_rate": res.get("win_rate"),
        "mean_hold": _mean_hold(res.get("trade_events") or []),
        "entries_attempted": res.get("entries_attempted"),
        "entries_executed": cap.get("entries_executed"),
        "signal_long_count": res.get("signal_long_count"),
        "signal_short_count": res.get("signal_short_count"),
        "signal_flat_count": res.get("signal_flat_count"),
        "directional_rule_reject_count": res.get("directional_rule_reject_count"),
        "avg_long_edge_entry": res.get("avg_long_edge_entry"),
        "avg_short_edge_entry": res.get("avg_short_edge_entry"),
        "avg_pflat_rejected": res.get("avg_pflat_rejected"),
    }


def main() -> int:
    from src.features.ml_feature_config import MLFeatureConfig

    parser = argparse.ArgumentParser(description="Phase D13: directional_edge + flat_margin + confidence sweep")
    parser.add_argument("--days", type=int, default=None, choices=[180, 365, 720], help="Single window; default: 180/365/720")
    parser.add_argument("--run-baseline", action="store_true", help="Also run argmax baseline for comparison.")
    parser.add_argument("--sweep-mode", type=str, default="all", choices=["edge", "flat", "conf", "all"], help="Which D13 sweeps to run.")
    args = parser.parse_args()

    config = MLFeatureConfig.from_preset("base")
    config.use_event_features = True
    model_id = os.environ.get("PHASE_D10_MODEL_ID", "h15_t0p004")
    model_path = MODELS_DIR / f"tcn_{model_id}.pt"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    print("[D13] 720d load + inference (once) ...", flush=True)
    triple_720, err = _get_ohlcv_and_proba(SYMBOL, TIMEFRAME, 720, END_DATE, config, True, model_path)
    if err:
        print(f"[D13] ERROR: {err}", file=sys.stderr)
        return 1
    period_data: dict[int, tuple] = {720: triple_720}
    df_720, pl_720, ps_720 = triple_720
    for d in [180, 365]:
        sl = _slice_period(df_720, pl_720, ps_720, d, END_DATE)
        if sl is None:
            print(f"[D13] slice {d}d failed", file=sys.stderr)
            return 1
        period_data[d] = sl

    windows = [args.days] if args.days else DAYS_LIST
    all_results: list[dict] = []
    baseline_720: dict | None = None

    if args.run_baseline:
        print("[D13] baseline (decision_mode=argmax) ...", flush=True)
        from scripts.run_d12_directional_strategy_sweep import run_one as run_one_d12

        for days in windows:
            row = run_one_d12(period_data, days, "argmax", None, True, None, True, None, None)
            if row:
                row["run_id"] = "phase_d13_baseline"
                all_results.append(row)
                if days == 720:
                    baseline_720 = row

    def _add_run_set(tag: str, edge_list, flat_list, conf_list):
        nonlocal all_results
        for edge in edge_list:
            for flat in flat_list:
                for conf in conf_list:
                    for days in windows:
                        row = run_one(
                            period_data,
                            days,
                            "directional_edge_margin_conf",
                            edge,
                            flat,
                            conf,
                        )
                        if row:
                            row["run_id"] = f"d13_{tag}_e{edge}_f{flat}_c{conf}"
                            all_results.append(row)

    if args.sweep_mode in ("edge", "all"):
        _add_run_set("edge", [0.04, 0.08, 0.12, 0.16], [0.03], [0.40])
    if args.sweep_mode in ("flat", "all"):
        _add_run_set("flat", [0.08], [0.00, 0.03, 0.05, 0.07], [0.40])
    if args.sweep_mode in ("conf", "all"):
        _add_run_set("conf", [0.08], [0.03], [0.35, 0.40, 0.45, 0.50])

    _write_summary(all_results, baseline_720)
    print(f"[D13] done. results={len(all_results)}", flush=True)
    return 0


def _write_summary(all_results: list, baseline_720: dict | None):
    out_json = BACKTESTS_DIR / "phase_d13_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"results": all_results, "baseline_720": baseline_720, "created_at": datetime.now(timezone.utc).isoformat()},
            f,
            indent=2,
        )
    print(f"[D13] wrote {out_json}", flush=True)

    bl = baseline_720 or {}
    lines = [
        "# Phase D13: Directional edge + flat margin + confidence summary",
        "",
        "## Baseline (argmax)",
        f"- 720d cost_on: {bl.get('cost_on')}",
        f"- 720d MDD: {bl.get('MDD')}",
        f"- 720d trades: {bl.get('trades')}",
        f"- entries_attempted: {bl.get('entries_attempted')}, entries_executed: {bl.get('entries_executed')}",
        "",
        "## All runs",
        "| run_id | days | decision_mode | min_directional_edge | min_flat_margin | min_confidence | cost_on | MDD | trades | entries_attempted | entries_executed | signal_long | signal_short | signal_flat | directional_rule_reject | avg_long_edge | avg_short_edge | avg_pflat_rejected |",
        "|--------|------|---------------|---------------------|-----------------|----------------|---------|-----|--------|-------------------|------------------|-------------|---------------|--------------|--------------------------|---------------|----------------|--------------------|",
    ]
    for r in all_results:
        c = r.get("cost_on")
        cstr = f"{c:.4f}" if c is not None else "-"
        mdd = r.get("MDD")
        mstr = f"{mdd:.4f}" if mdd is not None else "-"
        lines.append(
            f"| {r.get('run_id', '')} | {r.get('days', '')} | {r.get('decision_mode', '')} | "
            f"{r.get('min_directional_edge')} | {r.get('min_flat_margin')} | {r.get('min_confidence')} | "
            f"{cstr} | {mstr} | {r.get('trades', '')} | {r.get('entries_attempted')} | {r.get('entries_executed')} | "
            f"{r.get('signal_long_count')} | {r.get('signal_short_count')} | {r.get('signal_flat_count')} | "
            f"{r.get('directional_rule_reject_count', '')} | {r.get('avg_long_edge_entry', '')} | {r.get('avg_short_edge_entry', '')} | {r.get('avg_pflat_rejected', '')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "- Compare D13 vs baseline: 720d cost_on, MDD, trades, entries_attempted/entries_executed.",
        "- Target: entries_executed 300~600, cost_on improvement, trades not collapsed.",
    ])
    report_path = REPORTS_DIR / "phase_d13_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[D13] wrote {report_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
