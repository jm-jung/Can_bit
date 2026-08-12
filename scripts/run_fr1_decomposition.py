#!/usr/bin/env python3
"""
FR1 (h15_extsafe_v1) 구간 분해 + regime dependency 해부.
목적: 720d 성과가 어디서/왜 나왔는지 해부. period/regime robust 여부 판정.
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
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr1_decomposition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
FR1_PT = MODELS_DIR / "tcn_h15_extsafe_v1.pt"


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


def _ensure_timestamp(d: pd.DataFrame):
    if "timestamp" not in d.columns and isinstance(d.index, pd.DatetimeIndex):
        d = d.copy()
        d["timestamp"] = d.index
    return d


def part_a_time_blocks(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1):
    """6-way equal time blocks. 720d / 6 = 120d per block."""
    start_ts = pd.Timestamp(END_DATE).tz_localize("UTC") - pd.Timedelta(days=DAYS)
    block_days = 120
    rows = []
    for block_id in range(6):
        b_start = start_ts + pd.Timedelta(days=block_id * block_days)
        b_end = start_ts + pd.Timedelta(days=(block_id + 1) * block_days)
        for name, df_bt, pl, ps in [("h15_t0p004", df_bt_base, pl_base, ps_base), ("h15_extsafe_v1", df_bt_fr1, pl_fr1, ps_fr1)]:
            df_bt = _ensure_timestamp(df_bt)
            ts = pd.to_datetime(df_bt["timestamp"], utc=True)
            if ts.dt.tz is None:
                b_start_cmp = b_start.tz_localize(None) if b_start.tz else b_start
                b_end_cmp = b_end.tz_localize(None) if b_end.tz else b_end
            else:
                b_start_cmp, b_end_cmp = b_start, b_end
            mask = (ts >= b_start_cmp) & (ts < b_end_cmp)
            if mask.sum() < 50:
                rows.append({"block_id": block_id, "start_date": str(b_start.date()), "end_date": str(b_end.date()), "model_id": name, "cost_on": np.nan, "total_return": np.nan, "MDD": np.nan, "trades": 0, "win_rate": np.nan, "mean_hold": np.nan, "entries_attempted": np.nan, "entries_executed": np.nan})
                continue
            sub_bt = df_bt.loc[mask].reset_index(drop=True)
            sub_pl = pl[mask.values]
            sub_ps = ps[mask.values]
            res = run_backtest(sub_bt, sub_pl, sub_ps)
            if res is None:
                rows.append({"block_id": block_id, "start_date": str(b_start.date()), "end_date": str(b_end.date()), "model_id": name, "cost_on": np.nan, "total_return": np.nan, "MDD": np.nan, "trades": 0, "win_rate": np.nan, "mean_hold": np.nan, "entries_attempted": np.nan, "entries_executed": np.nan})
                continue
            cap = res.get("cap_trigger_stats") or {}
            mean_hold = np.nan
            if res.get("trade_events"):
                exits = [e.get("bars_held", 0) for e in res["trade_events"] if e.get("event", "").startswith("EXIT")]
                if exits:
                    mean_hold = float(np.mean(exits))
            rows.append({
                "block_id": block_id, "start_date": str(b_start.date()), "end_date": str(b_end.date()), "model_id": name,
                "cost_on": res.get("total_return"), "total_return": res.get("total_return"), "MDD": res.get("max_drawdown"),
                "trades": res.get("total_trades", 0), "win_rate": res.get("win_rate"), "mean_hold": mean_hold,
                "entries_attempted": res.get("entries_attempted"), "entries_executed": cap.get("entries_executed"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "time_blocks_6way.csv", index=False)
    print(f"[FR1-D] Part A: time_blocks_6way.csv ({len(rows)} rows)", flush=True)
    return df


def part_a_rolling(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1, window_days=90, step_days=30):
    """Rolling window backtest."""
    start_ts = pd.Timestamp(END_DATE).tz_localize("UTC") - pd.Timedelta(days=DAYS)
    rows = []
    w = 0
    while w * step_days + window_days <= DAYS:
        w_start = start_ts + pd.Timedelta(days=w * step_days)
        w_end = start_ts + pd.Timedelta(days=w * step_days + window_days)
        for name, df_bt, pl, ps in [("h15_t0p004", df_bt_base, pl_base, ps_base), ("h15_extsafe_v1", df_bt_fr1, pl_fr1, ps_fr1)]:
            df_bt = _ensure_timestamp(df_bt)
            ts = pd.to_datetime(df_bt["timestamp"], utc=True)
            if ts.dt.tz is None:
                w_start_cmp = w_start.tz_localize(None) if w_start.tz else w_start
                w_end_cmp = w_end.tz_localize(None) if w_end.tz else w_end
            else:
                w_start_cmp, w_end_cmp = w_start, w_end
            mask = (ts >= w_start_cmp) & (ts < w_end_cmp)
            if mask.sum() < 50:
                rows.append({"window_start": str(w_start.date()), "window_end": str(w_end.date()), "model_id": name, "cost_on": np.nan, "MDD": np.nan, "trades": 0, "win_rate": np.nan})
                continue
            res = run_backtest(df_bt.loc[mask].reset_index(drop=True), pl[mask.values], ps[mask.values])
            if res is None:
                rows.append({"window_start": str(w_start.date()), "window_end": str(w_end.date()), "model_id": name, "cost_on": np.nan, "MDD": np.nan, "trades": 0, "win_rate": np.nan})
            else:
                rows.append({"window_start": str(w_start.date()), "window_end": str(w_end.date()), "model_id": name, "cost_on": res.get("total_return"), "MDD": res.get("max_drawdown"), "trades": res.get("total_trades", 0), "win_rate": res.get("win_rate")})
        w += 1
    df = pd.DataFrame(rows)
    return df


def part_a_summary(time_blocks_df, rolling_df):
    """Delta vs baseline and summary stats."""
    # time blocks: per block, delta cost_on = FR1 - base
    block_base = time_blocks_df[time_blocks_df["model_id"] == "h15_t0p004"].set_index("block_id")
    block_fr1 = time_blocks_df[time_blocks_df["model_id"] == "h15_extsafe_v1"].set_index("block_id")
    rows = []
    for bid in block_base.index.intersection(block_fr1.index):
        cb = block_base.loc[bid, "cost_on"]
        cf = block_fr1.loc[bid, "cost_on"]
        if pd.isna(cb) or pd.isna(cf):
            continue
        rows.append({"block_id": bid, "baseline_cost_on": cb, "fr1_cost_on": cf, "delta_cost_on_vs_baseline": cf - cb, "fr1_beats_baseline": cf > cb})
    block_comp = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["block_id", "baseline_cost_on", "fr1_cost_on", "delta_cost_on_vs_baseline", "fr1_beats_baseline"])

    # rolling: same
    roll_base = rolling_df[rolling_df["model_id"] == "h15_t0p004"].copy()
    roll_fr1 = rolling_df[rolling_df["model_id"] == "h15_extsafe_v1"].copy()
    roll_base["key"] = roll_base["window_start"] + "_" + roll_base["window_end"]
    roll_fr1["key"] = roll_fr1["window_start"] + "_" + roll_fr1["window_end"]
    merged = roll_base.merge(roll_fr1, on="key", suffixes=("_base", "_fr1"))
    merged["delta_cost_on_vs_baseline"] = merged["cost_on_fr1"] - merged["cost_on_base"]
    merged["fr1_beats_baseline"] = merged["cost_on_fr1"] > merged["cost_on_base"]
    total_windows = len(merged)
    fr1_better = merged["fr1_beats_baseline"].sum()
    deltas = merged["delta_cost_on_vs_baseline"].dropna()
    summary = [{
        "metric": "total_windows", "value": total_windows,
    }, {"metric": "fr1_better_windows_count", "value": int(fr1_better)},
    {"metric": "fr1_better_windows_ratio", "value": fr1_better / total_windows if total_windows else 0},
    {"metric": "median_delta_cost_on", "value": deltas.median() if len(deltas) else np.nan},
    {"metric": "mean_delta_cost_on", "value": deltas.mean() if len(deltas) else np.nan},
    {"metric": "worst_delta_cost_on", "value": deltas.min() if len(deltas) else np.nan},
    {"metric": "best_delta_cost_on", "value": deltas.max() if len(deltas) else np.nan},
    ]
    pd.DataFrame(summary).to_csv(OUT_DIR / "time_slice_summary.csv", index=False)
    print(f"[FR1-D] time_slice_summary.csv (fr1_better_ratio={fr1_better/total_windows if total_windows else 0:.2f})", flush=True)
    return block_comp, merged, summary


def add_regime(df_bt: pd.DataFrame):
    """Add trend_q and vol_q (0,1,2) from close. In-place."""
    close = df_bt["close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    rolling_std_20 = close.rolling(20, min_periods=1).std()
    df_bt["trend_ratio"] = np.abs(close - ema20) / np.maximum(ema20, 1e-12)
    df_bt["vol"] = rolling_std_20.replace(0, np.nan).bfill().fillna(1e-12)
    try:
        df_bt["trend_q"] = pd.qcut(df_bt["trend_ratio"], q=3, labels=[0, 1, 2], duplicates="drop")
    except Exception:
        df_bt["trend_q"] = 0
    try:
        df_bt["vol_q"] = pd.qcut(df_bt["vol"], q=3, labels=[0, 1, 2], duplicates="drop")
    except Exception:
        df_bt["vol_q"] = 0
    return df_bt


def part_b_regime_basic(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1):
    """Regime tertiles backtest (trend_q0/1/2, vol_q0/1/2) per model."""
    rows = []
    for name, df_bt, pl, ps in [("h15_t0p004", df_bt_base.copy(), pl_base, ps_base), ("h15_extsafe_v1", df_bt_fr1.copy(), pl_fr1, ps_fr1)]:
        add_regime(df_bt)
        # full
        res = run_backtest(df_bt, pl, ps)
        r = {"regime_type": "full", "regime_name": "full", "row_count": len(df_bt), "model_id": name}
        if res:
            cap = res.get("cap_trigger_stats") or {}
            r.update({"cost_on": res.get("total_return"), "MDD": res.get("max_drawdown"), "trades": res.get("total_trades"), "entries_attempted": res.get("entries_attempted"), "entries_executed": cap.get("entries_executed"), "win_rate": res.get("win_rate"), "mean_hold": np.nan})
        else:
            r.update({"cost_on": np.nan, "MDD": np.nan, "trades": 0, "entries_attempted": np.nan, "entries_executed": np.nan, "win_rate": np.nan, "mean_hold": np.nan})
        rows.append(r)
        for regime_type, col in [("trend", "trend_q"), ("vol", "vol_q")]:
            for q in [0, 1, 2]:
                mask = (df_bt[col] == q).fillna(False).values
                if mask.sum() < 100:
                    rows.append({"regime_type": regime_type, "regime_name": f"{regime_type}_q{q}", "row_count": int(mask.sum()), "model_id": name, "cost_on": np.nan, "MDD": np.nan, "trades": 0, "entries_attempted": np.nan, "entries_executed": np.nan, "win_rate": np.nan, "mean_hold": np.nan})
                    continue
                res = run_backtest(df_bt.iloc[mask].reset_index(drop=True), pl[mask], ps[mask])
                r = {"regime_type": regime_type, "regime_name": f"{regime_type}_q{q}", "row_count": int(mask.sum()), "model_id": name}
                if res:
                    cap = res.get("cap_trigger_stats") or {}
                    r.update({"cost_on": res.get("total_return"), "MDD": res.get("max_drawdown"), "trades": res.get("total_trades"), "entries_attempted": res.get("entries_attempted"), "entries_executed": cap.get("entries_executed"), "win_rate": res.get("win_rate"), "mean_hold": np.nan})
                else:
                    r.update({"cost_on": np.nan, "MDD": np.nan, "trades": 0, "entries_attempted": np.nan, "entries_executed": np.nan, "win_rate": np.nan, "mean_hold": np.nan})
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "regime_backtest_basic.csv", index=False)
    print(f"[FR1-D] Part B: regime_backtest_basic.csv ({len(rows)} rows)", flush=True)
    return df


def part_b_intersection(df_bt_fr1, pl_fr1, ps_fr1, df_bt_base, pl_base, ps_base):
    """Trend x Vol intersection backtest; heatmap delta."""
    df_f = df_bt_fr1.copy()
    df_b = df_bt_base.copy()
    add_regime(df_f)
    add_regime(df_b)
    rows = []
    for tq in [0, 1, 2]:
        for vq in [0, 1, 2]:
            mask_f = ((df_f["trend_q"] == tq) & (df_f["vol_q"] == vq)).fillna(False).values
            mask_b = ((df_b["trend_q"] == tq) & (df_b["vol_q"] == vq)).fillna(False).values
            if mask_f.sum() < 80 or mask_b.sum() < 80:
                rows.append({"trend_regime": f"q{tq}", "vol_regime": f"q{vq}", "row_count": int(mask_f.sum()), "model_id": "h15_extsafe_v1", "cost_on": np.nan, "MDD": np.nan, "trades": 0, "win_rate": np.nan, "delta_cost_on_vs_baseline": np.nan, "fr1_beats_baseline": False})
                rows.append({"trend_regime": f"q{tq}", "vol_regime": f"q{vq}", "row_count": int(mask_b.sum()), "model_id": "h15_t0p004", "cost_on": np.nan, "MDD": np.nan, "trades": 0, "win_rate": np.nan, "delta_cost_on_vs_baseline": np.nan, "fr1_beats_baseline": False})
                continue
            res_b = run_backtest(df_b.iloc[mask_b].reset_index(drop=True), pl_base[mask_b], ps_base[mask_b])
            res_f = run_backtest(df_f.iloc[mask_f].reset_index(drop=True), pl_fr1[mask_f], ps_fr1[mask_f])
            cb = res_b.get("total_return") if res_b else np.nan
            cf = res_f.get("total_return") if res_f else np.nan
            delta = cf - cb if pd.notna(cf) and pd.notna(cb) else np.nan
            rows.append({"trend_regime": f"q{tq}", "vol_regime": f"q{vq}", "row_count": int(mask_b.sum()), "model_id": "h15_t0p004", "cost_on": cb, "MDD": res_b.get("max_drawdown") if res_b else np.nan, "trades": res_b.get("total_trades", 0) if res_b else 0, "win_rate": res_b.get("win_rate") if res_b else np.nan, "delta_cost_on_vs_baseline": np.nan, "fr1_beats_baseline": False})
            rows.append({"trend_regime": f"q{tq}", "vol_regime": f"q{vq}", "row_count": int(mask_f.sum()), "model_id": "h15_extsafe_v1", "cost_on": cf, "MDD": res_f.get("max_drawdown") if res_f else np.nan, "trades": res_f.get("total_trades", 0) if res_f else 0, "win_rate": res_f.get("win_rate") if res_f else np.nan, "delta_cost_on_vs_baseline": delta, "fr1_beats_baseline": (delta > 0) if pd.notna(delta) else False})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "regime_backtest_intersection.csv", index=False)
    # heatmap: pivot delta_cost_on for FR1 rows only
    fr1_only = df[df["model_id"] == "h15_extsafe_v1"][["trend_regime", "vol_regime", "delta_cost_on_vs_baseline"]].drop_duplicates()
    pivot = fr1_only.pivot(index="trend_regime", columns="vol_regime", values="delta_cost_on_vs_baseline")
    pivot.to_csv(OUT_DIR / "regime_delta_heatmap.csv")
    print(f"[FR1-D] regime_backtest_intersection.csv, regime_delta_heatmap.csv", flush=True)
    return df


def part_b_signal_quality(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1):
    """Direction accuracy, spearman, signal_density per regime (and intersection) per model."""
    # need future_return: from close at t and t+HORIZON
    def _metrics(df_bt, pl, ps):
        close = df_bt["close"].values.astype(float)
        if len(close) <= HORIZON:
            return {}
        future_ret = (close[HORIZON:] - close[:-HORIZON]) / np.maximum(close[:-HORIZON], 1e-12)
        trim = len(future_ret)
        pl_t = pl[:trim]
        ps_t = ps[:trim]
        p_flat = np.clip(1.0 - pl_t - ps_t, 0.0, 1.0)
        argmax_class = np.argmax(np.stack([p_flat, pl_t, ps_t], axis=1), axis=1)
        direction = np.where(argmax_class == 1, 1, np.where(argmax_class == 2, -1, 0))
        sign_y = np.sign(future_ret)
        mask = direction != 0
        acc = (np.sign(sign_y[mask]) == direction[mask]).mean() if mask.any() else np.nan
        long_edge = pl_t - ps_t
        spearman = pd.Series(long_edge).corr(pd.Series(future_ret), method="spearman") if trim > 10 else np.nan
        max_proba = np.maximum(np.maximum(pl_t, ps_t), p_flat)
        signal_density = (max_proba >= 0.60).mean() if trim else np.nan
        mean_ret_sig = np.nanmean(future_ret[max_proba >= 0.60]) if (max_proba >= 0.60).any() else np.nan
        return {"direction_accuracy": acc, "long_short_spearman": spearman, "signal_density": signal_density, "mean_return_signal": mean_ret_sig}

    rows = []
    for name, df_bt, pl, ps in [("h15_t0p004", df_bt_base.copy(), pl_base, ps_base), ("h15_extsafe_v1", df_bt_fr1.copy(), pl_fr1, ps_fr1)]:
        add_regime(df_bt)
        # full
        m = _metrics(df_bt, pl, ps)
        rows.append({"regime_type": "full", "regime_name": "full", "model_id": name, "row_count": len(df_bt), **m})
        for regime_type, col in [("trend", "trend_q"), ("vol", "vol_q")]:
            for q in [0, 1, 2]:
                mask = (df_bt[col] == q).fillna(False).values
                if mask.sum() < 100:
                    rows.append({"regime_type": regime_type, "regime_name": f"{regime_type}_q{q}", "model_id": name, "row_count": int(mask.sum()), "direction_accuracy": np.nan, "long_short_spearman": np.nan, "signal_density": np.nan, "mean_return_signal": np.nan})
                    continue
                m = _metrics(df_bt.iloc[mask].reset_index(drop=True), pl[mask], ps[mask])
                rows.append({"regime_type": regime_type, "regime_name": f"{regime_type}_q{q}", "model_id": name, "row_count": int(mask.sum()), **m})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "regime_signal_quality.csv", index=False)
    print(f"[FR1-D] regime_signal_quality.csv", flush=True)
    return df


def part_c_concentration(rolling_90_df):
    """FR1 rolling window contribution: top 10%, top 3, top 5 share."""
    fr1 = rolling_90_df[rolling_90_df["model_id"] == "h15_extsafe_v1"].copy()
    fr1["cost_on"] = pd.to_numeric(fr1["cost_on"], errors="coerce")
    fr1 = fr1.dropna(subset=["cost_on"])
    if len(fr1) == 0:
        pd.DataFrame([{"metric": "top_pct10_contribution_share", "value": np.nan}, {"metric": "top3_windows_contribution_share", "value": np.nan}, {"metric": "top5_windows_contribution_share", "value": np.nan}]).to_csv(OUT_DIR / "performance_concentration.csv", index=False)
        return
    total = fr1["cost_on"].sum()
    if total == 0:
        total = 1e-12
    sorted_ = fr1.sort_values("cost_on", ascending=False)
    n = len(sorted_)
    top10_cnt = max(1, int(n * 0.10))
    top10_sum = sorted_["cost_on"].head(top10_cnt).sum()
    top3_sum = sorted_["cost_on"].head(3).sum()
    top5_sum = sorted_["cost_on"].head(5).sum()
    rows = [{"metric": "top_pct10_contribution_share", "value": top10_sum / total}, {"metric": "top3_windows_contribution_share", "value": top3_sum / total}, {"metric": "top5_windows_contribution_share", "value": top5_sum / total}, {"metric": "total_windows", "value": n}, {"metric": "total_return_sum", "value": total}]
    pd.DataFrame(rows).to_csv(OUT_DIR / "performance_concentration.csv", index=False)
    print(f"[FR1-D] performance_concentration.csv (top10% share={top10_sum/total:.2f})", flush=True)


def write_summary(time_blocks_df, rolling_90_df, time_slice_summary, regime_basic_df, regime_intersection_df, concentration_df):
    """Final summary md + json and verdict."""
    # Count FR1 wins
    block_fr1 = time_blocks_df[time_blocks_df["model_id"] == "h15_extsafe_v1"]
    block_base = time_blocks_df[time_blocks_df["model_id"] == "h15_t0p004"]
    fr1_blocks_win = 0
    for bid in range(6):
        bf = block_fr1[block_fr1["block_id"] == bid]
        bb = block_base[block_base["block_id"] == bid]
        if len(bf) and len(bb) and pd.notna(bf["cost_on"].iloc[0]) and pd.notna(bb["cost_on"].iloc[0]):
            if bf["cost_on"].iloc[0] > bb["cost_on"].iloc[0]:
                fr1_blocks_win += 1
    total_windows = 0
    fr1_better_ratio = 0.0
    if isinstance(time_slice_summary, list):
        for s in time_slice_summary:
            if s.get("metric") == "total_windows":
                total_windows = int(s.get("value", 0))
            elif s.get("metric") == "fr1_better_windows_ratio":
                fr1_better_ratio = float(s.get("value", 0))
    elif isinstance(time_slice_summary, pd.DataFrame) and not time_slice_summary.empty:
        r = time_slice_summary[time_slice_summary["metric"] == "total_windows"]
        if len(r):
            total_windows = int(r["value"].iloc[0])
        r = time_slice_summary[time_slice_summary["metric"] == "fr1_better_windows_ratio"]
        if len(r):
            fr1_better_ratio = float(r["value"].iloc[0])

    # Regime: where FR1 beats baseline
    regime_fr1 = regime_basic_df[regime_basic_df["model_id"] == "h15_extsafe_v1"]
    regime_base = regime_basic_df[regime_basic_df["model_id"] == "h15_t0p004"]
    regime_wins = []
    for rn in regime_fr1["regime_name"].unique():
        rf = regime_fr1[regime_fr1["regime_name"] == rn]
        rb = regime_base[regime_base["regime_name"] == rn]
        if len(rf) and len(rb) and pd.notna(rf["cost_on"].iloc[0]) and pd.notna(rb["cost_on"].iloc[0]):
            if rf["cost_on"].iloc[0] > rb["cost_on"].iloc[0]:
                regime_wins.append(rn)

    # Concentration
    top10_share = np.nan
    if concentration_df is not None and len(concentration_df):
        c = concentration_df[concentration_df["metric"] == "top_pct10_contribution_share"]
        if len(c):
            top10_share = c["value"].iloc[0]

    # Verdict
    if fr1_blocks_win >= 4 and fr1_better_ratio >= 0.5 and len(regime_wins) >= 3 and (pd.isna(top10_share) or top10_share < 0.7):
        verdict = "FR1_REGIME_ROBUST_CANDIDATE"
    else:
        verdict = "FR1_PERIOD_DEPENDENT_NEEDS_CAUTION"

    lines = [
        "# FR1 Decomposition Summary",
        "",
        "## 1. Models",
        "- h15_t0p004 (baseline)",
        "- h15_extsafe_v1 (FR1)",
        "",
        "## 2. Time decomposition",
        f"- 6 blocks: FR1 wins {fr1_blocks_win}/6 blocks.",
        f"- Rolling 90d step 30d: FR1 better ratio = {fr1_better_ratio:.2f}.",
        "",
        "## 3. Regime",
        f"- Regimes where FR1 beats baseline: {regime_wins}.",
        "",
        "## 4. Concentration",
        f"- Top 10% windows contribution share (FR1): {top10_share:.2f}" if pd.notna(top10_share) else "- (N/A)",
        "",
        "## 5. Verdict",
        f"**{verdict}**",
        "",
        "## 6. Next step",
        "- FR1 as strong challenger; regime-aware threshold tuning." if verdict == "FR1_REGIME_ROBUST_CANDIDATE" else "- FR1 caution; out-of-period walk-forward or more regime checks.",
    ]
    (OUT_DIR / "fr1_decomposition_summary.md").write_text("\n".join(lines), encoding="utf-8")
    with open(OUT_DIR / "fr1_decomposition_summary.json", "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "fr1_blocks_win": fr1_blocks_win, "fr1_better_windows_ratio": fr1_better_ratio, "regime_wins": regime_wins, "top10_contribution_share": top10_share}, f, indent=2)
    print(f"[FR1-D] Verdict: {verdict}", flush=True)


def main():
    print("[FR1-D] Loading 720d base and FR1...", flush=True)
    triple_b, err_b = get_ohlcv_and_proba(DAYS, BASE_PT, "base")
    triple_f, err_f = get_ohlcv_and_proba(DAYS, FR1_PT, "extended_safe_v1")
    if err_b:
        print(f"[FR1-D] Base load failed: {err_b}", flush=True)
        return 1
    if err_f:
        print(f"[FR1-D] FR1 load failed: {err_f}", flush=True)
        return 1
    df_bt_base, pl_base, ps_base = triple_b
    df_bt_fr1, pl_fr1, ps_fr1 = triple_f
    print(f"[FR1-D] Base rows={len(df_bt_base)}, FR1 rows={len(df_bt_fr1)}", flush=True)

    print("[FR1-D] Part A: time blocks 6-way...", flush=True)
    time_blocks_df = part_a_time_blocks(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1)
    print("[FR1-D] Part A: rolling 90d step 30d...", flush=True)
    rolling_90 = part_a_rolling(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1, 90, 30)
    rolling_90.to_csv(OUT_DIR / "rolling_90d_step30.csv", index=False)
    rolling_120 = part_a_rolling(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1, 120, 30)
    rolling_120.to_csv(OUT_DIR / "rolling_120d_step30.csv", index=False)
    block_comp, merged, time_slice_summary = part_a_summary(time_blocks_df, rolling_90)
    # Add delta columns to time_blocks for output
    tb = pd.read_csv(OUT_DIR / "time_blocks_6way.csv")
    base_tb = tb[tb["model_id"] == "h15_t0p004"].set_index("block_id")
    fr1_tb = tb[tb["model_id"] == "h15_extsafe_v1"].set_index("block_id")
    tb = tb.copy()
    tb["delta_cost_on_vs_baseline"] = np.nan
    tb["fr1_beats_baseline"] = False
    for bid in base_tb.index.intersection(fr1_tb.index):
        cb, cf = base_tb.loc[bid, "cost_on"], fr1_tb.loc[bid, "cost_on"]
        if pd.notna(cb) and pd.notna(cf):
            mask = (tb["block_id"] == bid) & (tb["model_id"] == "h15_extsafe_v1")
            tb.loc[mask, "delta_cost_on_vs_baseline"] = cf - cb
            tb.loc[mask, "fr1_beats_baseline"] = cf > cb
    tb.to_csv(OUT_DIR / "time_blocks_6way.csv", index=False)

    print("[FR1-D] Part B: regime basic...", flush=True)
    regime_basic_df = part_b_regime_basic(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1)
    print("[FR1-D] Part B: regime intersection...", flush=True)
    regime_intersection_df = part_b_intersection(df_bt_fr1, pl_fr1, ps_fr1, df_bt_base, pl_base, ps_base)
    print("[FR1-D] Part B: regime signal quality...", flush=True)
    part_b_signal_quality(df_bt_base, pl_base, ps_base, df_bt_fr1, pl_fr1, ps_fr1)

    print("[FR1-D] Part C: concentration...", flush=True)
    concentration_df = pd.read_csv(OUT_DIR / "performance_concentration.csv") if (OUT_DIR / "performance_concentration.csv").exists() else None
    part_c_concentration(rolling_90)
    concentration_df = pd.read_csv(OUT_DIR / "performance_concentration.csv") if (OUT_DIR / "performance_concentration.csv").exists() else None

    print("[FR1-D] Summary + verdict...", flush=True)
    write_summary(time_blocks_df, rolling_90, time_slice_summary, regime_basic_df, regime_intersection_df, concentration_df)
    print("[FR1-D] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
