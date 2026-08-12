#!/usr/bin/env python3
"""
Phase D10: Confidence Spread Filter experiment.

Optional gate: spread = top1_proba - top2_proba; allow entry only if spread >= min_proba_gap.
"""
from __future__ import annotations

import os
import sys
import json
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
EARLY_EXIT_ON = True
REGIME_FILTER = "off"
POSITION_SCALING = "off"
COMMISSION = 0.0009
SLIPPAGE = 0.0001

# Updated baseline reference (latest execution)
BASELINE_180D = 0.012621292690840447
BASELINE_365D = -0.0733818966155878
BASELINE_720D = -0.111968647467793
BASELINE_720D_TRADES = 1491
BASELINE_TOLERANCE = 0.10  # 10%: STOP if deviation > 10%

RUNS = [
    ("phase_d10_baseline", 0.00),
    ("phase_d10_gap003", 0.03),
    ("phase_d10_gap005", 0.05),
    ("phase_d10_gap007", 0.07),
    ("phase_d10_gap010", 0.10),
]

DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_ohlcv_and_proba(symbol: str, timeframe: str, days: int, end_date: str, feature_config, use_events: bool, model_path: Path):
    """Load OHLCV, build features, run TCN inference, align df_bt and pl/ps."""
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
        pl, ps = np.asarray([pl[i] for i in pos], dtype=np.float32), np.asarray([ps[i] for i in pos], dtype=np.float32)
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
            return (None, f"align_ts len={len(align_ts)} != proba_len={proba_len}")
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
        return (None, f"len(df_bt)={len(df_bt)} != len(pl)={len(pl)}")
    return ((df_bt, pl, ps), None)


def _run_backtest(symbol: str, timeframe: str, df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray,
                  commission: float, slippage: float, min_proba_gap: float, emit_trade_log: bool = False):
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
    return run_backtest_7d(
        symbol, timeframe, df_bt, pl, ps, commission, slippage,
        min_max_proba=MIN_MAX_PROBA, max_entropy=MAX_ENTROPY,
        min_proba_gap=min_proba_gap,
        min_hold=MIN_HOLD, cooldown=COOLDOWN,
        regime_filter_enabled=False, position_scaling_enabled=False,
        early_exit_enabled=EARLY_EXIT_ON, early_exit_bad_k=EARLY_EXIT_BAD_K,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        emit_trade_log=emit_trade_log,
    )


def _slice_period(df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray, days: int, end_date: str):
    """Slice (df_bt, pl, ps) to last `days` ending at end_date."""
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


def _mean_median_hold(trade_events: list) -> tuple[float, float]:
    if not trade_events:
        return 0.0, 0.0
    bars = [e.get("bars_held") or e.get("holding_bars") or 0 for e in trade_events if e.get("event", "").startswith("EXIT")]
    bars = [b for b in bars if b > 0]
    return (float(np.mean(bars)), float(np.median(bars))) if bars else (0.0, 0.0)


def _export_trades(trade_events: list, out_path: Path, include_proba: bool = True) -> None:
    exits = [e for e in trade_events if e.get("event", "").startswith("EXIT") and "exit_reason" in e]
    if not exits:
        return
    rows = []
    for e in exits:
        r = {
            "entry_time": e.get("entry_ts") or e.get("ts"),
            "exit_time": e.get("exit_ts") or e.get("ts"),
            "pnl": e.get("net_return") if e.get("net_return") is not None else e.get("profit"),
            "exit_reason": e.get("exit_reason", ""),
            "holding_bars": e.get("holding_bars") if e.get("holding_bars") is not None else e.get("bars_held"),
        }
        if include_proba:
            r["top1_proba"] = e.get("top1_proba")
            r["top2_proba"] = e.get("top2_proba")
            r["proba_gap"] = e.get("proba_gap")
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[TRADE_LOG] {out_path} ({len(rows)} rows)", flush=True)


def main() -> int:
    from src.features.ml_feature_config import MLFeatureConfig

    config = MLFeatureConfig.from_preset("base")
    config.use_event_features = True
    model_id = os.environ.get("PHASE_D10_MODEL_ID", "h15_t0p004")
    model_path = MODELS_DIR / f"tcn_{model_id}.pt"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    # Load 720d once, slice for 180d/365d (same as baseline_only)
    print("[D10] 720d: loading OHLCV + inference (once) ...", flush=True)
    triple_720, err = _get_ohlcv_and_proba(SYMBOL, TIMEFRAME, 720, END_DATE, config, True, model_path)
    if err:
        print(f"[D10] 720d: {err}", file=sys.stderr)
        return 1
    period_data = {720: triple_720}
    df_720, pl_720, ps_720 = triple_720
    for days in [180, 365]:
        sliced = _slice_period(df_720, pl_720, ps_720, days, END_DATE)
        if sliced is None:
            print(f"[D10] slice {days}d failed", file=sys.stderr)
            return 1
        period_data[days] = sliced

    cost_off = {}
    for days in DAYS_LIST:
        df_bt, pl, ps = period_data[days]
        res, err = _run_backtest(SYMBOL, TIMEFRAME, df_bt, pl, ps, 0.0, 0.0, 0.0, emit_trade_log=False)
        if err:
            return 1
        cost_off[days] = res.get("total_return")

    results = []
    best_trade_events = {}  # 180, 365, 720 -> list (for BEST run later)

    for run_id, min_proba_gap in RUNS:
        row = {
            "run_id": run_id,
            "min_proba_gap": min_proba_gap,
            "180d_cost_on": None, "365d_cost_on": None, "720d_cost_on": None,
            "720d_mdd": None, "720d_trades": None,
            "win_rate_on": None, "mean_hold": None, "median_hold": None,
            "180d_cost_off": cost_off[180], "365d_cost_off": cost_off[365], "720d_cost_off": cost_off[720],
            "filtered_trade_count": None, "filtered_trade_ratio": None,
            "mean_spread_executed": None, "mean_spread_rejected": None,
        }
        for days in DAYS_LIST:
            df_bt, pl, ps = period_data[days]
            emit = (run_id == RUNS[0][0] and days == 365)
            print(f"[D10] {run_id} {days}d ...", flush=True)
            res, err = _run_backtest(SYMBOL, TIMEFRAME, df_bt, pl, ps, COMMISSION, SLIPPAGE, min_proba_gap, emit_trade_log=emit)
            if err:
                print(f"[D10] backtest failed: {err}", file=sys.stderr)
                return 1
            c = res.get("total_return")
            if days == 180:
                row["180d_cost_on"] = c
            elif days == 365:
                row["365d_cost_on"] = c
                row["win_rate_on"] = res.get("win_rate")
                mean_h, median_h = _mean_median_hold(res.get("trade_events") or [])
                row["mean_hold"], row["median_hold"] = mean_h, median_h
            elif days == 720:
                row["720d_cost_on"] = c
                row["720d_mdd"] = res.get("max_drawdown")
                row["720d_trades"] = res.get("total_trades", 0)
                row["filtered_trade_count"] = res.get("filtered_trade_count")
                row["filtered_trade_ratio"] = res.get("filtered_trade_ratio")
                row["mean_spread_executed"] = res.get("mean_spread_of_executed_trades")
                row["mean_spread_rejected"] = res.get("mean_spread_of_rejected_trades")
        results.append(row)

    baseline_warning = None
    baseline_row = next((r for r in results if r["run_id"] == "phase_d10_baseline"), None)
    if baseline_row:
        for label, val, exp in [
            ("180d", baseline_row.get("180d_cost_on"), BASELINE_180D),
            ("365d", baseline_row.get("365d_cost_on"), BASELINE_365D),
            ("720d", baseline_row.get("720d_cost_on"), BASELINE_720D),
        ]:
            if val is not None and exp is not None and abs(exp) > 1e-9:
                dev = abs(val - exp) / abs(exp)
                if dev > BASELINE_TOLERANCE:
                    msg = f"{label} cost_on deviation {dev:.2%} > 10% (expected ≈{exp}). Possible data drift."
                    print(f"[D10] STOP: {msg}", file=sys.stderr)
                    return 1
                elif dev > 0.05:
                    baseline_warning = f"{label} cost_on deviation {dev:.2%} (expected ≈{exp})."
                    print(f"[D10] WARN: {baseline_warning}", file=sys.stderr)
        # 720d trades validation
        t_val, t_exp = baseline_row.get("720d_trades"), BASELINE_720D_TRADES
        if t_val is not None and t_exp is not None and t_exp > 0:
            dev = abs(t_val - t_exp) / t_exp
            if dev > BASELINE_TOLERANCE:
                print(f"[D10] STOP: 720d trades deviation {dev:.2%} > 10% (expected ≈{t_exp}). Possible data drift.", file=sys.stderr)
                return 1

    baseline_720d_mdd = baseline_row.get("720d_mdd") if baseline_row else None

    non_reject = [r for r in results if not (
        (r.get("720d_cost_on") is not None and BASELINE_720D is not None and r["720d_cost_on"] < BASELINE_720D - 0.005)
        or (r.get("180d_cost_on") is not None and r["180d_cost_on"] < 0)
    )]
    best_row = max(non_reject, key=lambda x: x.get("720d_cost_on") or -999.0) if non_reject else results[0]

    adopt = (
        best_row.get("720d_cost_on") is not None and best_row["720d_cost_on"] >= BASELINE_720D + 0.015
        and (baseline_720d_mdd is None or (best_row.get("720d_mdd") is not None and best_row["720d_mdd"] <= baseline_720d_mdd))
        and best_row.get("720d_trades") is not None and best_row["720d_trades"] <= BASELINE_720D_TRADES * 0.80
        and best_row.get("180d_cost_on") is not None and best_row["180d_cost_on"] >= 0
    )
    reject = (
        (best_row.get("720d_cost_on") is not None and BASELINE_720D is not None and best_row["720d_cost_on"] < BASELINE_720D - 0.005)
        or (best_row.get("180d_cost_on") is not None and best_row["180d_cost_on"] < 0)
    )
    if adopt:
        verdict = "ADOPT_CANDIDATE"
    elif reject:
        verdict = "REJECT"
    else:
        verdict = "NO_IMPROVE"

    spot_runs = []
    for _ in range(2):
        row_spot = {"180d_cost_on": None, "365d_cost_on": None, "720d_cost_on": None, "720d_mdd": None, "720d_trades": None}
        for days in DAYS_LIST:
            df_bt, pl, ps = period_data[days]
            res, _ = _run_backtest(SYMBOL, TIMEFRAME, df_bt, pl, ps, COMMISSION, SLIPPAGE, best_row["min_proba_gap"], emit_trade_log=False)
            if res:
                if days == 180:
                    row_spot["180d_cost_on"] = res.get("total_return")
                elif days == 365:
                    row_spot["365d_cost_on"] = res.get("total_return")
                elif days == 720:
                    row_spot["720d_cost_on"] = res.get("total_return")
                    row_spot["720d_mdd"] = res.get("max_drawdown")
                    row_spot["720d_trades"] = res.get("total_trades")
        spot_runs.append(row_spot)

    all_720 = [best_row.get("720d_cost_on")] + [s.get("720d_cost_on") for s in spot_runs if s.get("720d_cost_on") is not None]
    range_720 = max(all_720) - min(all_720) if all_720 else 0.0
    spot_stability = "STABLE" if range_720 <= 0.005 else "FLAG"

    for days in DAYS_LIST:
        df_bt, pl, ps = period_data[days]
        res, _ = _run_backtest(SYMBOL, TIMEFRAME, df_bt, pl, ps, COMMISSION, SLIPPAGE, best_row["min_proba_gap"], emit_trade_log=True)
        if res and res.get("trade_events"):
            best_trade_events[days] = res["trade_events"]

    best_config = {
        "run_id": best_row["run_id"],
        "min_proba_gap": best_row["min_proba_gap"],
        "180d_cost_on": best_row.get("180d_cost_on"),
        "365d_cost_on": best_row.get("365d_cost_on"),
        "720d_cost_on": best_row.get("720d_cost_on"),
        "720d_mdd": best_row.get("720d_mdd"),
        "720d_trades": best_row.get("720d_trades"),
        "filtered_trade_ratio": best_row.get("filtered_trade_ratio"),
        "spot_stability": spot_stability,
        "verdict": verdict,
    }
    with open(BACKTESTS_DIR / "phase_d10_best_run.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    for days in [180, 365, 720]:
        if days in best_trade_events:
            _export_trades(best_trade_events[days], BACKTESTS_DIR / f"phase_d10_best_trades_{days}d.csv", include_proba=True)

    top3 = sorted([r for r in results if r.get("720d_cost_on") is not None], key=lambda x: x["720d_cost_on"], reverse=True)[:3]

    lines = [
        "# Phase D10: Confidence Spread Summary",
        "",
    ]
    if baseline_warning:
        lines.append("**Note:** " + baseline_warning)
        lines.append("")
    lines.extend([
        "## A) Strategy parameters",
        f"- symbol={SYMBOL}, timeframe={TIMEFRAME}, end_date={END_DATE}",
        f"- min_max_proba={MIN_MAX_PROBA}, max_entropy={MAX_ENTROPY}",
        f"- min_hold={MIN_HOLD}, cooldown={COOLDOWN}",
        f"- time_stop={TIME_STOP_BARS}, early_exit_bad_k={EARLY_EXIT_BAD_K}, early_exit=on",
        f"- regime_filter={REGIME_FILTER}, position_scaling={POSITION_SCALING}",
        "",
        "## B) Updated baseline reference",
        f"- 180d cost_on ≈ {BASELINE_180D}",
        f"- 365d cost_on ≈ {BASELINE_365D}",
        f"- 720d cost_on ≈ {BASELINE_720D}",
        f"- 720d trades = {BASELINE_720D_TRADES}",
        "",
        "## C) Full result table",
        "",
        "| run_id | min_proba_gap | 180d cost_on | 365d cost_on | 720d cost_on | 720d MDD | 720d trades | filtered_trade_ratio |",
        "|--------|---------------|--------------|--------------|--------------|----------|-------------|----------------------|",
    ])
    for r in results:
        lines.append("| {} | {:.2f} | {} | {} | {} | {} | {} | {} |".format(
            r["run_id"], r["min_proba_gap"],
            f"{r['180d_cost_on']:.4f}" if r.get("180d_cost_on") is not None else "-",
            f"{r['365d_cost_on']:.4f}" if r.get("365d_cost_on") is not None else "-",
            f"{r['720d_cost_on']:.4f}" if r.get("720d_cost_on") is not None else "-",
            f"{r['720d_mdd']:.4f}" if r.get("720d_mdd") is not None else "-",
            r.get("720d_trades") if r.get("720d_trades") is not None else "-",
            f"{r['filtered_trade_ratio']:.2%}" if r.get("filtered_trade_ratio") is not None else "-",
        ))
    lines.extend([
        "",
        "## D) Top3 by 720d cost_on",
        "",
    ])
    for i, r in enumerate(top3, 1):
        lines.append(f"{i}. **{r['run_id']}** min_proba_gap={r['min_proba_gap']:.2f} 720d cost_on={r.get('720d_cost_on'):.4f}")
    lines.extend([
        "",
        "## E) Verdict summary",
        f"- BEST run_id: **{best_row['run_id']}**",
        f"- BEST min_proba_gap: **{best_row['min_proba_gap']}**",
        f"- Spot stability: **{spot_stability}**",
        f"- Final verdict: **{verdict}**",
        "",
    ])
    report_path = REPORTS_DIR / "phase_d10_confidence_spread_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("", flush=True)
    print("=" * 60, flush=True)
    print("Phase D10 Confidence Spread", flush=True)
    print("=" * 60, flush=True)
    print(f"1) report path: {report_path}", flush=True)
    print(f"2) BEST run_id: {best_row['run_id']}", flush=True)
    print(f"3) BEST min_proba_gap: {best_row['min_proba_gap']}", flush=True)
    print(f"4) 180d cost_on: {best_row.get('180d_cost_on')}", flush=True)
    print(f"5) 365d cost_on: {best_row.get('365d_cost_on')}", flush=True)
    print(f"6) 720d cost_on: {best_row.get('720d_cost_on')}", flush=True)
    print(f"7) 720d MDD: {best_row.get('720d_mdd')}", flush=True)
    print(f"8) 720d trades: {best_row.get('720d_trades')}", flush=True)
    print(f"9) filtered_trade_ratio: {best_row.get('filtered_trade_ratio')}", flush=True)
    print(f"10) spot stability: {spot_stability}", flush=True)
    print(f"11) final verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
