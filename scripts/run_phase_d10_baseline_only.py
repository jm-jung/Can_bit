#!/usr/bin/env python3
"""Run only D10 baseline (min_proba_gap=0) and print 180d/365d/720d cost_on + 720d trades."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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


def _run_backtest(symbol: str, timeframe: str, df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray,
                  commission: float, slippage: float, min_proba_gap: float):
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
    return run_backtest_7d(
        symbol, timeframe, df_bt, pl, ps, commission, slippage,
        min_max_proba=MIN_MAX_PROBA, max_entropy=MAX_ENTROPY,
        min_proba_gap=min_proba_gap,
        min_hold=MIN_HOLD, cooldown=COOLDOWN,
        regime_filter_enabled=False, position_scaling_enabled=False,
        early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        emit_trade_log=False,
    )


def _slice_period(df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray, days: int, end_date: str):
    """Slice (df_bt, pl, ps) to last `days` ending at end_date. All same-length arrays."""
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
    # pl, ps are aligned to df_bt by index position; slice same rows
    idx = df_bt.index[mask]
    pos = [list(df_bt.index).index(i) for i in idx]
    pl_slice = np.asarray([pl[i] for i in pos], dtype=np.float32)
    ps_slice = np.asarray([ps[i] for i in pos], dtype=np.float32)
    return (df_slice.reset_index(drop=True), pl_slice, ps_slice)


def main() -> int:
    from src.features.ml_feature_config import MLFeatureConfig

    config = MLFeatureConfig.from_preset("base")
    config.use_event_features = True
    model_id = os.environ.get("PHASE_D10_MODEL_ID", "h15_t0p004")
    model_path = MODELS_DIR / f"tcn_{model_id}.pt"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    print("[D10 baseline] 720d load + predict (once) ...", flush=True)
    triple, err = _get_ohlcv_and_proba(SYMBOL, TIMEFRAME, 720, END_DATE, config, True, model_path)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    df_720, pl_720, ps_720 = triple

    out = {"180d_cost_on": None, "365d_cost_on": None, "720d_cost_on": None, "trades": None}
    for days in DAYS_LIST:
        print(f"[D10 baseline] {days}d backtest ...", flush=True)
        if days == 720:
            df_bt, pl, ps = df_720, pl_720, ps_720
        else:
            sliced = _slice_period(df_720, pl_720, ps_720, days, END_DATE)
            if sliced is None:
                print(f"ERROR: slice {days}d failed", file=sys.stderr)
                return 1
            df_bt, pl, ps = sliced
        res, err_bt = _run_backtest(SYMBOL, TIMEFRAME, df_bt, pl, ps, COMMISSION, SLIPPAGE, 0.0)
        if err_bt:
            print(f"ERROR: {err_bt}", file=sys.stderr)
            return 1
        c = res.get("total_return")
        if days == 180:
            out["180d_cost_on"] = c
        elif days == 365:
            out["365d_cost_on"] = c
        elif days == 720:
            out["720d_cost_on"] = c
            out["trades"] = res.get("total_trades")

    print("", flush=True)
    print("baseline run (this execution)", flush=True)
    print(f"  180d cost_on = {out['180d_cost_on']}", flush=True)
    print(f"  365d cost_on = {out['365d_cost_on']}", flush=True)
    print(f"  720d cost_on = {out['720d_cost_on']}", flush=True)
    print(f"  trades = {out['trades']}", flush=True)

    out_path = PROJECT_ROOT / "data" / "reports" / "d10_baseline_values.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("baseline run (this execution)\n")
        f.write(f"  180d cost_on = {out['180d_cost_on']}\n")
        f.write(f"  365d cost_on = {out['365d_cost_on']}\n")
        f.write(f"  720d cost_on = {out['720d_cost_on']}\n")
        f.write(f"  trades = {out['trades']}\n")
    print(f"\n(Written to {out_path})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
