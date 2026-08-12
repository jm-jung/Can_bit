#!/usr/bin/env python3
"""
Phase E1: Event Feature Impact Test.

Compare performance with (extended_safe / base+events) vs without (base, no events)
event features. Fixed strategy params; only feature preset changes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Thread limits before other imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# Fixed strategy parameters (do not change)
SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
END_DATE = "2026-03-03"
# Set to [365] for quick test; [180, 365, 720] for full report
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

DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline check (extended_safe / base+events)
BASELINE_365D_COST_ON = -0.057
BASELINE_720D_COST_ON = -0.112
BASELINE_TOLERANCE = 0.05  # 5% deviation → STOP


def _get_ohlcv_and_proba(
    symbol: str,
    timeframe: str,
    days: int,
    end_date: str,
    feature_config,
    use_events: bool,
    model_path: Path,
) -> tuple[tuple[pd.DataFrame, np.ndarray, np.ndarray] | None, str | None]:
    """Load OHLCV, build features with config, run TCN inference, align df_bt and pl/ps."""
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
        start_ts = start_ts.tz_localize(None)
        end_ts = end_ts.tz_localize(None)
    df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    if len(df) < 100:
        return (None, f"rows={len(df)} < 100")

    features = build_feature_frame(
        df, symbol=symbol, timeframe=timeframe, feature_config=feature_config
    )
    features = features.dropna()
    if len(features) < 60:
        return (None, f"features rows={len(features)} < 60")

    model = TCNSignalModel(model_path=model_path, use_events=use_events)
    if not model.is_loaded():
        return (None, "TCN model failed to load")
    pl_arr, ps_arr = model.predict_proba_batch(
        features=features, symbol=symbol, timeframe=timeframe, batch_size=512
    )
    pl = np.asarray(pl_arr, dtype=np.float32)
    ps = np.asarray(ps_arr, dtype=np.float32)

    if "close" in features.columns and "high" in features.columns and "low" in features.columns:
        df_bt = features[["close", "high", "low"]].copy()
        if isinstance(features.index, pd.DatetimeIndex):
            df_bt["timestamp"] = features.index
            df_bt = df_bt.reset_index(drop=True)
        elif "timestamp" in features.columns:
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

    # Ensure timestamp is only a column (no index/column ambiguity)
    if isinstance(df_bt.index, pd.DatetimeIndex) and "timestamp" in df_bt.columns:
        df_bt = df_bt.reset_index(drop=True)
    elif isinstance(df_bt.index, pd.DatetimeIndex):
        df_bt = df_bt.reset_index()
        if df_bt.columns[0] != "timestamp":
            df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})

    # Align lengths (TCN returns len - window_size)
    if len(df_bt) != len(pl):
        window_size = int(getattr(model, "window_size", 60))
        proba_len = len(pl)
        if "timestamp" in features.columns:
            feat_index = pd.DatetimeIndex(pd.to_datetime(features["timestamp"]))
        else:
            feat_index = pd.DatetimeIndex(pd.to_datetime(features.index))
        align_ts = feat_index[window_size : window_size + proba_len]
        if len(align_ts) != proba_len:
            return (None, f"align_ts len={len(align_ts)} != proba_len={proba_len}")
        proba_df = pd.DataFrame({
            "timestamp": pd.to_datetime(align_ts.values),
            "pl": pl,
            "ps": ps,
        })
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


def _run_backtest(
    symbol: str,
    timeframe: str,
    df_bt: pd.DataFrame,
    pl: np.ndarray,
    ps: np.ndarray,
    emit_trade_log: bool = False,
) -> tuple[dict | None, str | None]:
    """Run backtest with fixed Phase E1 params; returns (result_dict, error)."""
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    return run_backtest_7d(
        symbol,
        timeframe,
        df_bt,
        pl,
        ps,
        COMMISSION,
        SLIPPAGE,
        min_max_proba=MIN_MAX_PROBA,
        max_entropy=MAX_ENTROPY,
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
        regime_filter_enabled=False,
        position_scaling_enabled=False,
        early_exit_enabled=EARLY_EXIT_ON,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        emit_trade_log=emit_trade_log,
    )


def _feature_count(feature_config) -> int:
    from src.services.ohlcv_service import load_ohlcv_df
    from src.indicators.basic import add_basic_indicators
    from src.ml.features import build_feature_frame
    df = load_ohlcv_df(timeframe=TIMEFRAME, symbol=SYMBOL)
    df = add_basic_indicators(df)
    if "timestamp" not in df.columns:
        df = df.reset_index()
    feats = build_feature_frame(df, symbol=SYMBOL, timeframe=TIMEFRAME, feature_config=feature_config)
    return len([c for c in feats.columns if c not in ("close", "high", "low", "timestamp")])


def _mean_median_hold(trade_events: list) -> tuple[float, float]:
    if not trade_events:
        return 0.0, 0.0
    bars = [
        e.get("bars_held") or e.get("holding_bars") or 0
        for e in trade_events
        if e.get("event", "").startswith("EXIT")
    ]
    bars = [b for b in bars if b > 0]
    if not bars:
        return 0.0, 0.0
    return float(np.mean(bars)), float(np.median(bars))


def _export_trade_log_365d(trade_events: list, out_path: Path) -> None:
    """Export entry_time, exit_time, pnl, exit_reason, holding_bars."""
    exits = [e for e in trade_events if e.get("event", "").startswith("EXIT") and "exit_reason" in e]
    if not exits:
        return
    rows = []
    for e in exits:
        rows.append({
            "entry_time": e.get("entry_ts") or e.get("ts"),
            "exit_time": e.get("exit_ts") or e.get("ts"),
            "pnl": e.get("net_return") if e.get("net_return") is not None else e.get("profit"),
            "exit_reason": e.get("exit_reason", ""),
            "holding_bars": e.get("holding_bars") if e.get("holding_bars") is not None else e.get("bars_held"),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[TRADE_LOG] {out_path} ({len(rows)} rows)", flush=True)


def main() -> int:
    from src.features.ml_feature_config import MLFeatureConfig

    # Preset A: base, use_event_features=False (technical only, ~14)
    config_base_no_events = MLFeatureConfig.from_preset("base")
    config_base_no_events.use_event_features = False
    # Preset B: extended_safe (technical + event; use base+events so TCN input dim matches)
    config_extended_safe = MLFeatureConfig.from_preset("extended_safe")
    # If TCN was trained with base+events only, use base+events for B to avoid dim mismatch
    config_base_events = MLFeatureConfig.from_preset("base")
    config_base_events.use_event_features = True

    no_events_path = Path(os.environ.get("TCN_MODEL_NO_EVENTS_PATH", "") or str(DIAG / "tcn_no_events.pt"))
    if not no_events_path.is_absolute():
        no_events_path = PROJECT_ROOT / no_events_path
    model_id = os.environ.get("PHASE_E1_MODEL_ID", "h15_t0p004")
    with_events_path = MODELS_DIR / f"tcn_{model_id}.pt"
    if not with_events_path.exists():
        print(f"ERROR: with-events model not found: {with_events_path}", file=sys.stderr)
        return 1
    if not no_events_path.exists():
        print(f"ERROR: no-events model not found: {no_events_path}. Set TCN_MODEL_NO_EVENTS_PATH or add tcn_no_events.pt", file=sys.stderr)
        return 1

    n_base = _feature_count(config_base_no_events)
    n_ext = _feature_count(config_extended_safe)
    # Use base+events for B so model input dim matches; report as "extended_safe" (technical+event)
    n_ext_report = _feature_count(config_base_events)
    print(f"[E1] Feature count: base (no events)={n_base}, extended_safe (with events)={n_ext_report}", flush=True)

    results = []
    trade_events_365_base = None
    trade_events_365_ext = None

    for preset_name, config, use_events, mpath in [
        ("base", config_base_no_events, False, no_events_path),
        ("extended_safe", config_base_events, True, with_events_path),
    ]:
        n_feat = n_base if preset_name == "base" else n_ext_report
        row = {
            "feature_preset": preset_name,
            "feature_count": n_feat,
            "180d_cost_on": None,
            "365d_cost_on": None,
            "720d_cost_on": None,
            "720d_mdd": None,
            "720d_trades": None,
            "win_rate": None,
            "mean_hold": None,
            "median_hold": None,
        }
        for days in DAYS_LIST:
            print(f"[E1] {preset_name} {days}d: loading OHLCV + inference ...", flush=True)
            triple, err = _get_ohlcv_and_proba(
                SYMBOL, TIMEFRAME, days, END_DATE,
                config, use_events, mpath,
            )
            if err:
                print(f"[E1] {preset_name} {days}d: {err}", flush=True)
                continue
            df_bt, pl, ps = triple
            emit_log = preset_name == "base" and days == 365 or preset_name == "extended_safe" and days == 365
            print(f"[E1] {preset_name} {days}d: backtest ...", flush=True)
            res, err_bt = _run_backtest(SYMBOL, TIMEFRAME, df_bt, pl, ps, emit_trade_log=emit_log)
            if err_bt:
                print(f"[E1] {preset_name} {days}d backtest: {err_bt}", flush=True)
                continue
            cost_on = res.get("total_return")
            trades = res.get("total_trades", 0)
            mdd = res.get("max_drawdown")
            wr = res.get("win_rate")
            if days == 180:
                row["180d_cost_on"] = cost_on
            elif days == 365:
                row["365d_cost_on"] = cost_on
                row["win_rate"] = wr
                if res.get("trade_events"):
                    if preset_name == "base":
                        trade_events_365_base = res["trade_events"]
                    else:
                        trade_events_365_ext = res["trade_events"]
                    mean_h, median_h = _mean_median_hold(res["trade_events"])
                    row["mean_hold"] = mean_h
                    row["median_hold"] = median_h
            elif days == 720:
                row["720d_cost_on"] = cost_on
                row["720d_mdd"] = mdd
                row["720d_trades"] = trades
        results.append(row)

    # Baseline check (extended_safe 365d / 720d cost_on) — warn only, still write full report
    baseline_warning = None
    ext_row = next((r for r in results if r["feature_preset"] == "extended_safe"), None)
    if ext_row and ext_row.get("365d_cost_on") is not None:
        dev_365 = abs(ext_row["365d_cost_on"] - BASELINE_365D_COST_ON) / max(abs(BASELINE_365D_COST_ON), 1e-8)
        if dev_365 > BASELINE_TOLERANCE:
            baseline_warning = f"365d cost_on deviation {dev_365:.2%} > 5% (expected ≈{BASELINE_365D_COST_ON}). Possible data drift."
            print(f"[E1] WARN: {baseline_warning}", file=sys.stderr)
    if ext_row and ext_row.get("720d_cost_on") is not None:
        dev_720 = abs(ext_row["720d_cost_on"] - BASELINE_720D_COST_ON) / max(abs(BASELINE_720D_COST_ON), 1e-8)
        if dev_720 > BASELINE_TOLERANCE:
            msg = f"720d cost_on deviation {dev_720:.2%} > 5% (expected ≈{BASELINE_720D_COST_ON}). Possible data drift."
            print(f"[E1] WARN: {msg}", file=sys.stderr)
            if baseline_warning is None:
                baseline_warning = msg

    # Export 365d trade logs
    if trade_events_365_base:
        _export_trade_log_365d(trade_events_365_base, BACKTESTS_DIR / "phase_e1_base_trades_365d.csv")
    if trade_events_365_ext:
        _export_trade_log_365d(trade_events_365_ext, BACKTESTS_DIR / "phase_e1_extended_trades_365d.csv")

    # Verdict
    base_row = next((r for r in results if r["feature_preset"] == "base"), None)
    if not ext_row or not base_row:
        verdict = "EVENT_FEATURE_NEUTRAL"
    else:
        c365_ext = ext_row.get("365d_cost_on") or -999.0
        c365_base = base_row.get("365d_cost_on") or -999.0
        c720_ext = ext_row.get("720d_cost_on") or -999.0
        c720_base = base_row.get("720d_cost_on") or -999.0
        mdd_ext = ext_row.get("720d_mdd") or 999.0
        mdd_base = base_row.get("720d_mdd") or 999.0
        improve_365 = (c365_ext - c365_base) >= 0.005
        improve_720 = (c720_ext - c720_base) >= 0.01
        mdd_better = (mdd_base - mdd_ext) >= 0.01
        if improve_365 or improve_720 or mdd_better:
            verdict = "EVENT_FEATURE_HELPFUL"
        else:
            if c365_ext < c365_base and c720_ext < c720_base:
                verdict = "EVENT_FEATURE_HARMFUL"
            else:
                verdict = "EVENT_FEATURE_NEUTRAL"

    # Report
    lines = [
        "# Phase E1: Event Feature Impact Summary",
        "",
    ]
    if baseline_warning:
        lines.extend([
            "**Note:** " + baseline_warning,
            "",
        ])
    lines.extend([
        "## A) Strategy parameters (fixed)",
        f"- symbol={SYMBOL}, timeframe={TIMEFRAME}, end_date={END_DATE}",
        f"- min_max_proba={MIN_MAX_PROBA}, max_entropy={MAX_ENTROPY}",
        f"- min_hold={MIN_HOLD}, cooldown={COOLDOWN}",
        f"- time_stop={TIME_STOP_BARS}, early_exit_bad_k={EARLY_EXIT_BAD_K}, early_exit=on",
        f"- regime_filter={REGIME_FILTER}, position_scaling={POSITION_SCALING}",
        f"- days_list={DAYS_LIST}",
        "",
        "## B) Result table",
        "",
        "| feature_preset | feature_count | 180d cost_on | 365d cost_on | 720d cost_on | 720d MDD | 720d trades | win_rate |",
        "|----------------|---------------|--------------|--------------|--------------|----------|-------------|----------|",
    ])
    for r in results:
        c180 = r.get("180d_cost_on")
        c365 = r.get("365d_cost_on")
        c720 = r.get("720d_cost_on")
        mdd = r.get("720d_mdd")
        tr = r.get("720d_trades")
        wr = r.get("win_rate")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["feature_preset"],
                r["feature_count"],
                f"{c180:.4f}" if c180 is not None else "—",
                f"{c365:.4f}" if c365 is not None else "—",
                f"{c720:.4f}" if c720 is not None else "—",
                f"{mdd:.4f}" if mdd is not None else "—",
                tr if tr is not None else "—",
                f"{wr:.2%}" if wr is not None else "—",
            )
        )
    lines.extend([
        "",
        "## Verdict",
        f"- **{verdict}**",
        "",
        "Export:",
        "- data/backtests/phase_e1_base_trades_365d.csv",
        "- data/backtests/phase_e1_extended_trades_365d.csv",
        "",
    ])
    report_path = REPORTS_DIR / "phase_e1_feature_impact_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Console output
    print("", flush=True)
    print("=" * 60, flush=True)
    print("Phase E1 Feature Impact", flush=True)
    print("=" * 60, flush=True)
    print(f"1) Report path: {report_path}", flush=True)
    print(f"2) Feature count: base={n_base}, extended_safe={n_ext_report}", flush=True)
    if base_row and ext_row:
        print(f"3) 180d cost_on: base={base_row.get('180d_cost_on')}, extended_safe={ext_row.get('180d_cost_on')}", flush=True)
        print(f"4) 365d cost_on: base={base_row.get('365d_cost_on')}, extended_safe={ext_row.get('365d_cost_on')}", flush=True)
        print(f"5) 720d cost_on: base={base_row.get('720d_cost_on')}, extended_safe={ext_row.get('720d_cost_on')}", flush=True)
        print(f"6) 720d MDD: base={base_row.get('720d_mdd')}, extended_safe={ext_row.get('720d_mdd')}", flush=True)
    print(f"7) Final verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
