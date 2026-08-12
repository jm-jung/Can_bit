#!/usr/bin/env python3
"""
FR1 Walk-Forward Validation + Recent Regime Breakdown Analysis.
Pseudo-walk-forward (fixed models, OOS window backtest only).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
END_DATE = "2026-03-03"
DAYS = 720
WINDOW_SIZE = 60
HORIZON = 15

COMMISSION = 0.0009
SLIPPAGE = 0.0001
MIN_MAX_PROBA = 0.575
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8

MODELS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "models"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr1_walkforward"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
FR1_PT = MODELS_DIR / "tcn_h15_extsafe_v1.pt"

# early = first 360d (block 0,1,2), late = last 360d (block 3,4,5)
EARLY_DAYS = 360
LATE_DAYS = 360
START_720 = pd.Timestamp(END_DATE).tz_localize("UTC") - pd.Timedelta(days=DAYS)
END_720 = pd.Timestamp(END_DATE).tz_localize("UTC") + pd.Timedelta(days=1)
EARLY_END = START_720 + pd.Timedelta(days=EARLY_DAYS)


def _load_ohlcv(days: int) -> pd.DataFrame:
    from src.services.ohlcv_service import load_ohlcv_df
    df = load_ohlcv_df(timeframe=TIMEFRAME, symbol=SYMBOL)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_ts = pd.Timestamp(END_DATE).tz_localize("UTC") - pd.Timedelta(days=days)
    end_ts = pd.Timestamp(END_DATE).tz_localize("UTC") + pd.Timedelta(days=1)
    if df["timestamp"].dt.tz is not None:
        df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    else:
        start_naive = start_ts.tz_localize(None) if start_ts.tz else start_ts
        end_naive = end_ts.tz_localize(None) if end_ts.tz else end_ts
        df = df.loc[(df["timestamp"] >= start_naive) & (df["timestamp"] < end_naive)].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


def get_ohlcv_and_proba(days: int, model_path: Path, feature_preset: str):
    """Return (df_bt, pl, ps) or (None, err)."""
    from src.features.ml_feature_config import MLFeatureConfig
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel
    from src.indicators.basic import add_basic_indicators

    if not model_path.exists():
        return (None, f"model not found: {model_path}")
    df = _load_ohlcv(days)
    df = add_basic_indicators(df)
    config = MLFeatureConfig.from_preset(feature_preset)
    features = build_feature_frame(df, symbol=SYMBOL, timeframe=TIMEFRAME, feature_config=config)
    features = features.dropna()
    if len(features) < WINDOW_SIZE + HORIZON:
        return (None, f"not enough rows: {len(features)}")
    model = TCNSignalModel(model_path=model_path, use_events=True, feature_config=config)
    if not model.is_loaded():
        return (None, "model load failed")
    pl, ps = model.predict_proba_batch(features=features, symbol=SYMBOL, timeframe=TIMEFRAME, batch_size=512)
    pl = np.asarray(pl, dtype=np.float32)
    ps = np.asarray(ps, dtype=np.float32)
    N = len(features)
    valid_len = N - WINDOW_SIZE - HORIZON
    if valid_len <= 0:
        return (None, f"valid_len={valid_len}")
    idx = slice(WINDOW_SIZE, WINDOW_SIZE + valid_len)
    df_bt = features[["close", "high", "low"]].iloc[idx].copy()
    if isinstance(features.index, pd.DatetimeIndex):
        df_bt["timestamp"] = features.index[idx]
    else:
        df_bt["timestamp"] = features["timestamp"].values[idx] if "timestamp" in features.columns else features.index[idx]
    df_bt = df_bt.reset_index(drop=True)
    pl = pl[:valid_len]
    ps = ps[:valid_len]
    return ((df_bt, pl, ps), "")


def run_backtest(df_bt, pl, ps):
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
    res, err = run_backtest_7d(
        SYMBOL, TIMEFRAME, df_bt, pl, ps,
        commission_rate=COMMISSION, slippage_rate=SLIPPAGE,
        min_max_proba=MIN_MAX_PROBA, max_entropy=MAX_ENTROPY,
        decision_mode="argmax", min_hold=MIN_HOLD, cooldown=COOLDOWN,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
    )
    if err:
        return None
    return res


def _ts_cmp(ts_series, start, end):
    ts = pd.to_datetime(ts_series, utc=True)
    if ts.dt.tz is None:
        s = start.tz_localize(None) if getattr(start, "tz", None) else start
        e = end.tz_localize(None) if getattr(end, "tz", None) else end
    else:
        s, e = start, end
    return (ts >= s) & (ts < e)


def _slice_by_dates(df_bt, pl, ps, start, end):
    df_bt = df_bt.copy()
    if "timestamp" not in df_bt.columns:
        return None, None, None
    mask = _ts_cmp(df_bt["timestamp"], start, end)
    if mask.sum() < 30:
        return None, None, None
    return df_bt.loc[mask].reset_index(drop=True), pl[mask.values], ps[mask.values]


# ---------- Part A: Walk-Forward ----------
def part_a_walkforward(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1):
    # A-1: train 360d / test 90d / step 90d → test windows: 360-450, 450-540, 540-630, 630-720
    # A-2: train 540d / test 90d / step 30d → test: 540-630, 570-660, 600-690, 630-720
    rows_360, rows_540 = [], []
    # A-1
    for split_id, (t_start_d, t_end_d) in enumerate([
        (360, 450), (450, 540), (540, 630), (630, 720),
    ]):
        test_start = START_720 + pd.Timedelta(days=t_start_d)
        test_end = START_720 + pd.Timedelta(days=t_end_d)
        for name, df_bt, pl, ps in [("h15_t0p004", df_bt_base, pl_base, ps_base), ("h15_extsafe_v1", df_bt_fr1, pl_fr1, ps_fr1)]:
            sub_bt, sub_pl, sub_ps = _slice_by_dates(df_bt, pl, ps, test_start, test_end)
            if sub_bt is None:
                rows_360.append({"split_id": split_id, "wf_type": "360_90_90", "train_start": str((START_720 + pd.Timedelta(days=0)).date()), "train_end": str((START_720 + pd.Timedelta(days=360)).date()), "test_start": str(test_start.date()), "test_end": str(test_end.date()), "model_id": name, "cost_on": np.nan, "total_return": np.nan, "MDD": np.nan, "trades": 0, "win_rate": np.nan, "mean_hold": np.nan, "entries_attempted": np.nan, "entries_executed": np.nan, "delta_cost_on_vs_baseline": np.nan, "fr1_beats_baseline": False})
                continue
            res = run_backtest(sub_bt, sub_pl, sub_ps)
            cap = res.get("cap_trigger_stats") or {} if res else {}
            r = {"split_id": split_id, "wf_type": "360_90_90", "train_start": str((START_720 + pd.Timedelta(days=0)).date()), "train_end": str((START_720 + pd.Timedelta(days=360)).date()), "test_start": str(test_start.date()), "test_end": str(test_end.date()), "model_id": name, "cost_on": res.get("total_return") if res else np.nan, "total_return": res.get("total_return") if res else np.nan, "MDD": res.get("max_drawdown") if res else np.nan, "trades": res.get("total_trades", 0) if res else 0, "win_rate": res.get("win_rate") if res else np.nan, "mean_hold": np.nan, "entries_attempted": res.get("entries_attempted") if res else np.nan, "entries_executed": cap.get("entries_executed") if res else np.nan, "delta_cost_on_vs_baseline": np.nan, "fr1_beats_baseline": False}
            rows_360.append(r)
    # add deltas for 360
    for split_id in range(4):
        b = [x for x in rows_360 if x["split_id"] == split_id and x["model_id"] == "h15_t0p004"]
        f = [x for x in rows_360 if x["split_id"] == split_id and x["model_id"] == "h15_extsafe_v1"]
        if b and f and pd.notna(b[0]["cost_on"]) and pd.notna(f[0]["cost_on"]):
            delta = f[0]["cost_on"] - b[0]["cost_on"]
            for r in rows_360:
                if r["split_id"] == split_id and r["model_id"] == "h15_extsafe_v1":
                    r["delta_cost_on_vs_baseline"] = delta
                    r["fr1_beats_baseline"] = delta > 0
                    break
    pd.DataFrame(rows_360).to_csv(OUT_DIR / "wf_360_90_90.csv", index=False)

    # A-2: 540/90/30
    for split_id, (t_start_d, t_end_d) in enumerate([
        (540, 630), (570, 660), (600, 690), (630, 720),
    ]):
        test_start = START_720 + pd.Timedelta(days=t_start_d)
        test_end = START_720 + pd.Timedelta(days=t_end_d)
        for name, df_bt, pl, ps in [("h15_t0p004", df_bt_base, pl_base, ps_base), ("h15_extsafe_v1", df_bt_fr1, pl_fr1, ps_fr1)]:
            sub_bt, sub_pl, sub_ps = _slice_by_dates(df_bt, pl, ps, test_start, test_end)
            if sub_bt is None:
                rows_540.append({"split_id": split_id, "wf_type": "540_90_30", "train_start": str((START_720 + pd.Timedelta(days=0)).date()), "train_end": str((START_720 + pd.Timedelta(days=540)).date()), "test_start": str(test_start.date()), "test_end": str(test_end.date()), "model_id": name, "cost_on": np.nan, "total_return": np.nan, "MDD": np.nan, "trades": 0, "win_rate": np.nan, "mean_hold": np.nan, "entries_attempted": np.nan, "entries_executed": np.nan, "delta_cost_on_vs_baseline": np.nan, "fr1_beats_baseline": False})
                continue
            res = run_backtest(sub_bt, sub_pl, sub_ps)
            cap = res.get("cap_trigger_stats") or {} if res else {}
            r = {"split_id": split_id, "wf_type": "540_90_30", "train_start": str((START_720 + pd.Timedelta(days=0)).date()), "train_end": str((START_720 + pd.Timedelta(days=540)).date()), "test_start": str(test_start.date()), "test_end": str(test_end.date()), "model_id": name, "cost_on": res.get("total_return") if res else np.nan, "total_return": res.get("total_return") if res else np.nan, "MDD": res.get("max_drawdown") if res else np.nan, "trades": res.get("total_trades", 0) if res else 0, "win_rate": res.get("win_rate") if res else np.nan, "mean_hold": np.nan, "entries_attempted": res.get("entries_attempted") if res else np.nan, "entries_executed": cap.get("entries_executed") if res else np.nan, "delta_cost_on_vs_baseline": np.nan, "fr1_beats_baseline": False}
            rows_540.append(r)
    for split_id in range(4):
        b = [x for x in rows_540 if x["split_id"] == split_id and x["model_id"] == "h15_t0p004"]
        f = [x for x in rows_540 if x["split_id"] == split_id and x["model_id"] == "h15_extsafe_v1"]
        if b and f and pd.notna(b[0]["cost_on"]) and pd.notna(f[0]["cost_on"]):
            delta = f[0]["cost_on"] - b[0]["cost_on"]
            for r in rows_540:
                if r["split_id"] == split_id and r["model_id"] == "h15_extsafe_v1":
                    r["delta_cost_on_vs_baseline"] = delta
                    r["fr1_beats_baseline"] = delta > 0
                    break
    pd.DataFrame(rows_540).to_csv(OUT_DIR / "wf_540_90_30.csv", index=False)

    # summary
    all_deltas = [r["delta_cost_on_vs_baseline"] for r in rows_360 + rows_540 if r["model_id"] == "h15_extsafe_v1" and pd.notna(r.get("delta_cost_on_vs_baseline"))]
    fr1_wins_360 = sum(1 for r in rows_360 if r.get("fr1_beats_baseline"))
    fr1_wins_540 = sum(1 for r in rows_540 if r.get("fr1_beats_baseline"))
    summary = [
        {"metric": "total_splits_360", "value": 4}, {"metric": "fr1_better_splits_360", "value": fr1_wins_360},
        {"metric": "total_splits_540", "value": 4}, {"metric": "fr1_better_splits_540", "value": fr1_wins_540},
        {"metric": "fr1_better_splits_ratio_360", "value": fr1_wins_360 / 4 if 4 else 0},
        {"metric": "fr1_better_splits_ratio_540", "value": fr1_wins_540 / 4 if 4 else 0},
        {"metric": "mean_delta_cost_on", "value": np.mean(all_deltas) if all_deltas else np.nan},
        {"metric": "median_delta_cost_on", "value": np.median(all_deltas) if all_deltas else np.nan},
        {"metric": "best_delta_cost_on", "value": np.max(all_deltas) if all_deltas else np.nan},
        {"metric": "worst_delta_cost_on", "value": np.min(all_deltas) if all_deltas else np.nan},
    ]
    # recent 3 splits (last 3 of 540)
    recent_540 = [r for r in rows_540 if r["model_id"] == "h15_extsafe_v1" and r["split_id"] >= 1]
    recent_deltas = [r["delta_cost_on_vs_baseline"] for r in recent_540 if pd.notna(r.get("delta_cost_on_vs_baseline"))]
    summary.append({"metric": "recent_3_splits_better_count", "value": sum(1 for r in recent_540 if r.get("fr1_beats_baseline"))})
    summary.append({"metric": "recent_3_splits_mean_delta_cost_on", "value": np.mean(recent_deltas) if recent_deltas else np.nan})
    pd.DataFrame(summary).to_csv(OUT_DIR / "walkforward_summary.csv", index=False)
    print(f"[WF] Part A: wf_360_90_90.csv, wf_540_90_30.csv, walkforward_summary.csv (fr1_wins_360={fr1_wins_360}, fr1_wins_540={fr1_wins_540})", flush=True)
    return rows_360, rows_540, summary


# ---------- Part B: Feature / output / trade drift ----------
def _get_features_720d():
    from src.features.ml_feature_config import MLFeatureConfig
    from src.ml.features import build_feature_frame
    from src.indicators.basic import add_basic_indicators
    df = _load_ohlcv(DAYS)
    df = add_basic_indicators(df)
    config = MLFeatureConfig.from_preset("extended_safe_v1")
    features = build_feature_frame(df, symbol=SYMBOL, timeframe=TIMEFRAME, feature_config=config)
    features = features.dropna()
    idx = slice(WINDOW_SIZE, min(WINDOW_SIZE + len(features) - WINDOW_SIZE - HORIZON, len(features)))
    N = len(features)
    valid_len = N - WINDOW_SIZE - HORIZON
    if valid_len <= 0:
        return None
    feat_slice = features.iloc[WINDOW_SIZE : WINDOW_SIZE + valid_len].copy()
    if isinstance(features.index, pd.DatetimeIndex):
        feat_slice["_ts"] = features.index[WINDOW_SIZE : WINDOW_SIZE + valid_len]
    else:
        feat_slice["_ts"] = feat_slice.index
    return feat_slice


def part_b_drift(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1):
    feat_slice = _get_features_720d()
    if feat_slice is None:
        pd.DataFrame(columns=["feature_name", "early_mean", "early_std", "late_mean", "late_std", "delta_mean", "note"]).to_csv(OUT_DIR / "feature_drift_early_vs_late.csv", index=False)
        print("[WF] Part B: feature_drift (no features)", flush=True)
        return
    ts = pd.to_datetime(feat_slice["_ts"], utc=True)
    early_mask = _ts_cmp(feat_slice["_ts"], START_720, EARLY_END)
    late_mask = _ts_cmp(feat_slice["_ts"], EARLY_END, END_720)
    num_cols = [c for c in feat_slice.columns if c != "_ts" and feat_slice[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
    # limit to requested or first 20
    want = ["rolling_std_20", "close_ema_ratio", "close_sma_ratio", "rsi_14", "volume", "ema_20", "close", "high", "low"]
    num_cols = [c for c in want if c in num_cols] or num_cols[:20]
    rows = []
    for c in num_cols:
        e = feat_slice.loc[early_mask, c].dropna()
        l = feat_slice.loc[late_mask, c].dropna()
        rows.append({"feature_name": c, "early_mean": e.mean(), "early_std": e.std(), "late_mean": l.mean(), "late_std": l.std(), "delta_mean": l.mean() - e.mean() if len(e) and len(l) else np.nan, "note": ""})
    pd.DataFrame(rows).to_csv(OUT_DIR / "feature_drift_early_vs_late.csv", index=False)

    # model output drift: early vs late for each model
    def _metrics(df_bt, pl, ps, mask):
        if mask.sum() < 50:
            return {}
        close = df_bt.loc[mask, "close"].values.astype(float)
        pl_m = pl[mask.values]
        ps_m = ps[mask.values]
        if len(close) <= HORIZON:
            return {}
        future_ret = (close[HORIZON:] - close[:-HORIZON]) / np.maximum(close[:-HORIZON], 1e-12)
        trim = len(future_ret)
        pl_t, ps_t = pl_m[:trim], ps_m[:trim]
        p_flat = np.clip(1.0 - pl_t - ps_t, 0.0, 1.0)
        max_proba = np.maximum(np.maximum(pl_t, ps_t), p_flat)
        direction = np.argmax(np.stack([p_flat, pl_t, ps_t], axis=1), axis=1)
        direction = np.where(direction == 1, 1, np.where(direction == 2, -1, 0))
        acc = (np.sign(future_ret) == direction).mean() if (direction != 0).any() else np.nan
        spearman = pd.Series(pl_t - ps_t).corr(pd.Series(future_ret), method="spearman") if trim > 10 else np.nan
        sig_den = (max_proba >= 0.60).mean()
        mean_ret_sig = np.nanmean(future_ret[max_proba >= 0.60]) if (max_proba >= 0.60).any() else np.nan
        return {"p_flat_mean": p_flat.mean(), "p_long_mean": pl_t.mean(), "p_short_mean": ps_t.mean(), "max_proba_mean": max_proba.mean(), "direction_accuracy": acc, "long_short_spearman": spearman, "signal_density": sig_den, "mean_return_signal": mean_ret_sig}
    df_bt_base = df_bt_base.copy()
    df_bt_fr1 = df_bt_fr1.copy()
    if "timestamp" not in df_bt_base.columns:
        pd.DataFrame(columns=["model_id", "period", "metric", "value"]).to_csv(OUT_DIR / "model_output_drift.csv", index=False)
    else:
        early_b = _ts_cmp(df_bt_base["timestamp"], START_720, EARLY_END)
        late_b = _ts_cmp(df_bt_base["timestamp"], EARLY_END, END_720)
        early_f = _ts_cmp(df_bt_fr1["timestamp"], START_720, EARLY_END)
        late_f = _ts_cmp(df_bt_fr1["timestamp"], EARLY_END, END_720)
        out_rows = []
        for name, df_bt, pl, ps in [("h15_t0p004", df_bt_base, pl_base, ps_base), ("h15_extsafe_v1", df_bt_fr1, pl_fr1, ps_fr1)]:
            em = _metrics(df_bt, pl, ps, early_b if name == "h15_t0p004" else early_f)
            lm = _metrics(df_bt, pl, ps, late_b if name == "h15_t0p004" else late_f)
            for k in em:
                out_rows.append({"model_id": name, "period": "early", "metric": k, "value": em.get(k, np.nan)})
            for k in lm:
                out_rows.append({"model_id": name, "period": "late", "metric": k, "value": lm.get(k, np.nan)})
        pd.DataFrame(out_rows).to_csv(OUT_DIR / "model_output_drift.csv", index=False)

    # trade behavior: backtest early-only and late-only
    trade_rows = []
    for name, df_bt, pl, ps in [("h15_t0p004", df_bt_base, pl_base, ps_base), ("h15_extsafe_v1", df_bt_fr1, pl_fr1, ps_fr1)]:
        for period, start, end in [("early", START_720, EARLY_END), ("late", EARLY_END, END_720)]:
            sub_bt, sub_pl, sub_ps = _slice_by_dates(df_bt, pl, ps, start, end)
            if sub_bt is None or len(sub_bt) < 50:
                trade_rows.append({"model_id": name, "period": period, "trades": 0, "entries_attempted": np.nan, "entries_executed": np.nan, "win_rate": np.nan, "mean_hold": np.nan})
                continue
            res = run_backtest(sub_bt, sub_pl, sub_ps)
            cap = res.get("cap_trigger_stats") or {} if res else {}
            trade_rows.append({"model_id": name, "period": period, "trades": res.get("total_trades", 0) if res else 0, "entries_attempted": res.get("entries_attempted") if res else np.nan, "entries_executed": cap.get("entries_executed") if res else np.nan, "win_rate": res.get("win_rate") if res else np.nan, "mean_hold": np.nan})
    pd.DataFrame(trade_rows).to_csv(OUT_DIR / "trade_behavior_drift.csv", index=False)
    print("[WF] Part B: feature_drift, model_output_drift, trade_behavior_drift", flush=True)


# ---------- Part C: Regime frequency / quality / attribution ----------
def add_regime(df_bt):
    close = df_bt["close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    vol = df_bt["close"].rolling(20, min_periods=1).std().replace(0, np.nan).bfill().fillna(1e-12)
    df_bt["trend_ratio"] = np.abs(close - ema20) / np.maximum(ema20, 1e-12)
    df_bt["vol"] = vol
    try:
        df_bt["trend_q"] = pd.qcut(df_bt["trend_ratio"], q=3, labels=[0, 1, 2], duplicates="drop")
    except Exception:
        df_bt["trend_q"] = 0
    try:
        df_bt["vol_q"] = pd.qcut(df_bt["vol"], q=3, labels=[0, 1, 2], duplicates="drop")
    except Exception:
        df_bt["vol_q"] = 0
    return df_bt


def part_c_regime(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1):
    df_b = df_bt_base.copy()
    df_f = df_bt_fr1.copy()
    add_regime(df_b)
    add_regime(df_f)
    early_b = _ts_cmp(df_b["timestamp"], START_720, EARLY_END)
    late_b = _ts_cmp(df_b["timestamp"], EARLY_END, END_720)
    early_f = _ts_cmp(df_f["timestamp"], START_720, EARLY_END)
    late_f = _ts_cmp(df_f["timestamp"], EARLY_END, END_720)
    # frequency: use one df (e.g. FR1) to count regime share
    freq_rows = []
    for tq in [0, 1, 2]:
        for vq in [0, 1, 2]:
            rn = f"trend_q{tq}_vol_q{vq}"
            e_count = ((df_f["trend_q"] == tq) & (df_f["vol_q"] == vq) & early_f).sum()
            l_count = ((df_f["trend_q"] == tq) & (df_f["vol_q"] == vq) & late_f).sum()
            e_tot = early_f.sum()
            l_tot = late_f.sum()
            e_share = e_count / e_tot if e_tot else 0
            l_share = l_count / l_tot if l_tot else 0
            freq_rows.append({"regime_name": rn, "early_row_count": int(e_count), "early_share": e_share, "late_row_count": int(l_count), "late_share": l_share, "delta_share": l_share - e_share})
    pd.DataFrame(freq_rows).to_csv(OUT_DIR / "regime_frequency_shift.csv", index=False)

    # internal quality: per (regime, period, model) cost_on, direction_accuracy, spearman, signal_density
    def _qual(df_bt, pl, ps, mask_regime, period_name):
        if mask_regime.sum() < 80:
            return None, None, None, None
        sub_bt, sub_pl, sub_ps = df_bt.loc[mask_regime].reset_index(drop=True), pl[mask_regime.values], ps[mask_regime.values]
        res = run_backtest(sub_bt, sub_pl, sub_ps)
        close = sub_bt["close"].values.astype(float)
        if len(close) <= HORIZON:
            return (res.get("total_return") if res else np.nan), np.nan, np.nan, np.nan
        future_ret = (close[HORIZON:] - close[:-HORIZON]) / np.maximum(close[:-HORIZON], 1e-12)
        trim = len(future_ret)
        pl_t, ps_t = sub_pl[:trim], sub_ps[:trim]
        p_flat = np.clip(1.0 - pl_t - ps_t, 0.0, 1.0)
        direction = np.argmax(np.stack([p_flat, pl_t, ps_t], axis=1), axis=1)
        direction = np.where(direction == 1, 1, np.where(direction == 2, -1, 0))
        acc = (np.sign(future_ret) == direction).mean() if (direction != 0).any() else np.nan
        spearman = pd.Series(pl_t - ps_t).corr(pd.Series(future_ret), method="spearman") if trim > 10 else np.nan
        sig_den = (np.maximum(np.maximum(pl_t, ps_t), p_flat) >= 0.60).mean()
        return (res.get("total_return") if res else np.nan), acc, spearman, sig_den

    qual_rows = []
    for tq in [0, 1, 2]:
        for vq in [0, 1, 2]:
            rn = f"q{tq}xq{vq}"
            for period, use_early in [("early", True), ("late", False)]:
                mask_b = (df_b["trend_q"] == tq) & (df_b["vol_q"] == vq) & (early_b if use_early else late_b)
                mask_f = (df_f["trend_q"] == tq) & (df_f["vol_q"] == vq) & (early_f if use_early else late_f)
                cb, acc_b, sp_b, sd_b = _qual(df_b, pl_base, ps_base, mask_b, period)
                cf, acc_f, sp_f, sd_f = _qual(df_f, pl_fr1, ps_fr1, mask_f, period)
                qual_rows.append({"regime": rn, "period": period, "baseline_cost_on": cb, "fr1_cost_on": cf, "delta_cost_on": cf - cb if pd.notna(cf) and pd.notna(cb) else np.nan, "baseline_direction_accuracy": acc_b, "fr1_direction_accuracy": acc_f, "baseline_long_short_spearman": sp_b, "fr1_long_short_spearman": sp_f, "baseline_signal_density": sd_b, "fr1_signal_density": sd_f})
    pd.DataFrame(qual_rows).to_csv(OUT_DIR / "regime_internal_quality.csv", index=False)

    # attribution: simple
    # early overall vs late overall; regime mix effect = (late_share - early_share) * early_quality per regime; within effect = early_share * (late_quality - early_quality)
    early_overall_b = run_backtest(df_b.loc[early_b].reset_index(drop=True), pl_base[early_b.values], ps_base[early_b.values])
    late_overall_b = run_backtest(df_b.loc[late_b].reset_index(drop=True), pl_base[late_b.values], ps_base[late_b.values])
    early_overall_f = run_backtest(df_f.loc[early_f].reset_index(drop=True), pl_fr1[early_f.values], ps_fr1[early_f.values])
    late_overall_f = run_backtest(df_f.loc[late_f].reset_index(drop=True), pl_fr1[late_f.values], ps_fr1[late_f.values])
    c_early_b = early_overall_b.get("total_return") if early_overall_b else np.nan
    c_late_b = late_overall_b.get("total_return") if late_overall_b else np.nan
    c_early_f = early_overall_f.get("total_return") if early_overall_f else np.nan
    c_late_f = late_overall_f.get("total_return") if late_overall_f else np.nan
    drop = (c_late_f - c_early_f) if (pd.notna(c_late_f) and pd.notna(c_early_f)) else np.nan
    primary = "within_regime_quality" if (isinstance(drop, (int, float)) and abs(drop) > 0.1) else "regime_mix_or_small"
    attr = [{"effect": "early_overall_baseline", "value": c_early_b}, {"effect": "late_overall_baseline", "value": c_late_b}, {"effect": "early_overall_fr1", "value": c_early_f}, {"effect": "late_overall_fr1", "value": c_late_f}, {"effect": "fr1_early_to_late_drop", "value": drop}, {"effect": "primary_cause", "value": primary}]
    pd.DataFrame(attr).to_csv(OUT_DIR / "regime_attribution_summary.csv", index=False)
    print("[WF] Part C: regime_frequency_shift, regime_internal_quality, regime_attribution_summary", flush=True)


# ---------- Part D: Recent OOS ----------
def part_d_recent_oos(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1):
    rows = []
    for last_d in [180, 90, 60]:
        end = END_720
        start = end - pd.Timedelta(days=last_d)
        for name, df_bt, pl, ps in [("h15_t0p004", df_bt_base, pl_base, ps_base), ("h15_extsafe_v1", df_bt_fr1, pl_fr1, ps_fr1)]:
            sub_bt, sub_pl, sub_ps = _slice_by_dates(df_bt, pl, ps, start, end)
            if sub_bt is None or len(sub_bt) < 30:
                rows.append({"period_days": last_d, "model_id": name, "cost_on": np.nan, "MDD": np.nan, "trades": 0, "win_rate": np.nan, "direction_accuracy": np.nan, "long_short_spearman": np.nan, "signal_density": np.nan})
                continue
            res = run_backtest(sub_bt, sub_pl, sub_ps)
            close = sub_bt["close"].values.astype(float)
            if len(close) <= HORIZON:
                acc, sp, sd = np.nan, np.nan, np.nan
            else:
                future_ret = (close[HORIZON:] - close[:-HORIZON]) / np.maximum(close[:-HORIZON], 1e-12)
                trim = len(future_ret)
                pl_t, ps_t = sub_pl[:trim], sub_ps[:trim]
                p_flat = np.clip(1.0 - pl_t - ps_t, 0.0, 1.0)
                direction = np.argmax(np.stack([p_flat, pl_t, ps_t], axis=1), axis=1)
                direction = np.where(direction == 1, 1, np.where(direction == 2, -1, 0))
                acc = (np.sign(future_ret) == direction).mean() if (direction != 0).any() else np.nan
                sp = pd.Series(pl_t - ps_t).corr(pd.Series(future_ret), method="spearman") if trim > 10 else np.nan
                sd = (np.maximum(np.maximum(pl_t, ps_t), p_flat) >= 0.60).mean()
            rows.append({"period_days": last_d, "model_id": name, "cost_on": res.get("total_return") if res else np.nan, "MDD": res.get("max_drawdown") if res else np.nan, "trades": res.get("total_trades", 0) if res else 0, "win_rate": res.get("win_rate") if res else np.nan, "direction_accuracy": acc, "long_short_spearman": sp, "signal_density": sd})
    pd.DataFrame(rows).to_csv(OUT_DIR / "recent_oos_check.csv", index=False)
    print("[WF] Part D: recent_oos_check.csv", flush=True)


# ---------- Part E: Final table + Summary ----------
def part_e_final_and_summary(rows_360, rows_540, summary_list):
    # final table: section, metric, baseline_value, fr1_value, delta, status, note
    fin = []
    # walkforward
    wf_360_f = [r for r in rows_360 if r["model_id"] == "h15_extsafe_v1"]
    wf_360_b = [r for r in rows_360 if r["model_id"] == "h15_t0p004"]
    fr1_wins_360 = sum(1 for r in wf_360_f if r.get("fr1_beats_baseline"))
    fin.append({"section": "walkforward", "metric": "fr1_better_splits_ratio_360", "baseline_value": 4 - fr1_wins_360, "fr1_value": fr1_wins_360, "delta": fr1_wins_360 - (4 - fr1_wins_360), "status": "FR1_WIN" if fr1_wins_360 >= 3 else "BASELINE_WIN", "note": "4 splits"})
    wf_540_f = [r for r in rows_540 if r["model_id"] == "h15_extsafe_v1"]
    fr1_wins_540 = sum(1 for r in wf_540_f if r.get("fr1_beats_baseline"))
    fin.append({"section": "walkforward", "metric": "fr1_better_splits_ratio_540", "baseline_value": 4 - fr1_wins_540, "fr1_value": fr1_wins_540, "delta": fr1_wins_540 - (4 - fr1_wins_540), "status": "FR1_WIN" if fr1_wins_540 >= 3 else "BASELINE_WIN", "note": "4 splits"})
    # recent oos
    rec = pd.read_csv(OUT_DIR / "recent_oos_check.csv") if (OUT_DIR / "recent_oos_check.csv").exists() else pd.DataFrame()
    if not rec.empty:
        for last_d in [180, 90, 60]:
            r_b = rec[(rec["period_days"] == last_d) & (rec["model_id"] == "h15_t0p004")]
            r_f = rec[(rec["period_days"] == last_d) & (rec["model_id"] == "h15_extsafe_v1")]
            if len(r_b) and len(r_f):
                cb, cf = r_b["cost_on"].iloc[0], r_f["cost_on"].iloc[0]
                st = "FR1_WIN" if cf > cb else "BASELINE_WIN"
                fin.append({"section": "recent_oos", "metric": f"cost_on_last_{last_d}d", "baseline_value": cb, "fr1_value": cf, "delta": cf - cb, "status": st, "note": ""})
    pd.DataFrame(fin).to_csv(OUT_DIR / "fr1_walkforward_final_table.csv", index=False)

    # verdict
    r180f = rec[(rec["period_days"] == 180) & (rec["model_id"] == "h15_extsafe_v1")]
    r180b = rec[(rec["period_days"] == 180) & (rec["model_id"] == "h15_t0p004")]
    recent_180_f = float(r180f["cost_on"].iloc[0]) if len(r180f) else np.nan
    recent_180_b = float(r180b["cost_on"].iloc[0]) if len(r180b) else np.nan
    if fr1_wins_360 >= 3 and fr1_wins_540 >= 3 and (pd.isna(recent_180_f) or recent_180_f >= recent_180_b):
        verdict = "FR1_WALKFORWARD_CONFIRMED_CHALLENGER"
    else:
        verdict = "FR1_RECENT_BREAKDOWN_REQUIRES_CAUTION"

    lines = [
        "# FR1 Walk-Forward Summary",
        "",
        "## 1. Models",
        "- h15_t0p004 (baseline)",
        "- h15_extsafe_v1 (FR1)",
        "",
        "## 2. Method",
        "- **Pseudo-walk-forward**: fixed trained models; OOS test windows only (no retrain).",
        "",
        "## 3. Walk-forward",
        f"- 360/90/90: FR1 better in {fr1_wins_360}/4 splits.",
        f"- 540/90/30: FR1 better in {fr1_wins_540}/4 splits.",
        "",
        "## 4. Recent OOS",
        f"- Last 180d: baseline cost_on={recent_180_b}, FR1 cost_on={recent_180_f}.",
        "",
        "## 5. Verdict",
        f"**{verdict}**",
        "",
        "## 6. Next step",
        "- FR1 as challenger; recent-only tuning." if verdict == "FR1_WALKFORWARD_CONFIRMED_CHALLENGER" else "- FR1 caution; baseline main; consider walk-forward with retrain or feature round.",
    ]
    (OUT_DIR / "fr1_walkforward_summary.md").write_text("\n".join(lines), encoding="utf-8")
    with open(OUT_DIR / "fr1_walkforward_summary.json", "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "fr1_wins_360": fr1_wins_360, "fr1_wins_540": fr1_wins_540, "recent_180_baseline": recent_180_b, "recent_180_fr1": recent_180_f}, f, indent=2)
    print(f"[WF] Part E: verdict={verdict}", flush=True)


def main():
    print("[WF] Loading 720d base and FR1...", flush=True)
    triple_b, err_b = get_ohlcv_and_proba(DAYS, BASE_PT, "base")
    triple_f, err_f = get_ohlcv_and_proba(DAYS, FR1_PT, "extended_safe_v1")
    if err_b or err_f:
        print(f"[WF] Load failed: base={err_b}, fr1={err_f}", flush=True)
        return 1
    df_bt_base, pl_base, ps_base = triple_b
    df_bt_fr1, pl_fr1, ps_fr1 = triple_f
    print("[WF] Part A: Walk-forward...", flush=True)
    rows_360, rows_540, summary_list = part_a_walkforward(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1)
    print("[WF] Part B: Drift...", flush=True)
    part_b_drift(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1)
    print("[WF] Part C: Regime...", flush=True)
    part_c_regime(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1)
    print("[WF] Part D: Recent OOS...", flush=True)
    part_d_recent_oos(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1)
    print("[WF] Part E: Final table + Summary...", flush=True)
    part_e_final_and_summary(rows_360, rows_540, summary_list)
    print("[WF] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
