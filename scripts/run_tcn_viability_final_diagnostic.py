#!/usr/bin/env python3
"""
TCN Viability Final Diagnostic.

목적: h15_t0p004 + BTCUSDT 5m horizon=15가 더 밀어볼 가치가 있는지 최종 진단.
- Part A: Permutation importance (5 metrics)
- Part B: Prediction decile return + top bucket summary
- Part C: Regime-filtered backtest
- 최종 판정: TCN_STILL_WORTH_PUSHING vs MOVE_TO_FEATURE_RESEARCH
"""
from __future__ import annotations

import argparse
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
POS_THRESHOLD = 0.004
NEG_THRESHOLD = 0.004
COST = 0.001
MODEL_ID = "h15_t0p004"
MODELS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "models"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "tcn_viability"

# Baseline backtest params (match retrain round 1)
COMMISSION = 0.0009
SLIPPAGE = 0.0001
MIN_MAX_PROBA = 0.575
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8


def _load_ohlcv_range_5m(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame | None:
    csv_path = PROJECT_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv"
    if symbol.upper() != "BTCUSDT" or not csv_path.exists():
        return None
    chunks = []
    for chunk in pd.read_csv(csv_path, parse_dates=["timestamp"], chunksize=60_000):
        if chunk["timestamp"].min() > end_ts:
            break
        if chunk["timestamp"].max() < start_ts:
            continue
        chunk = chunk[(chunk["timestamp"] >= start_ts) & (chunk["timestamp"] < end_ts)]
        if len(chunk) > 0:
            chunks.append(chunk)
    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def load_data_and_baseline_inference(
    symbol: str,
    timeframe: str,
    days: int,
    end_date: str,
    model_path: Path,
    horizon: int,
    window_size: int,
    cost: float,
):
    """Load OHLCV, build features (base), run TCN inference, return raw aligned + features + model + df_bt for backtest."""
    from src.features.ml_feature_config import MLFeatureConfig
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel

    start_ts = pd.Timestamp(end_date).tz_localize("UTC") - pd.Timedelta(days=days)
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
    start_naive = start_ts.tz_localize(None) if start_ts.tz else start_ts
    end_naive = end_ts.tz_localize(None) if end_ts.tz else end_ts
    df = _load_ohlcv_range_5m(symbol, start_naive, end_naive)
    if df is None or len(df) < 500:
        from src.services.ohlcv_service import load_ohlcv_df
        df = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
        else:
            df = df.loc[(df["timestamp"] >= start_naive) & (df["timestamp"] < end_naive)].copy()
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if len(df) < 100:
        raise ValueError(f"Not enough OHLCV rows: {len(df)}")
    config = MLFeatureConfig.from_preset("base")
    features = build_feature_frame(df, symbol=symbol, timeframe=timeframe, feature_config=config)
    features = features.dropna()
    if len(features) < window_size + horizon:
        raise ValueError(f"Not enough feature rows: {len(features)}")
    model = TCNSignalModel(model_path=model_path, use_events=True)
    if not model.is_loaded():
        raise ValueError("TCN model failed to load")
    pl, ps = model.predict_proba_batch(features=features, symbol=symbol, timeframe=timeframe, batch_size=512)
    pl = np.asarray(pl, dtype=np.float32)
    ps = np.asarray(ps, dtype=np.float32)
    N = len(features)
    valid_len = N - window_size - horizon
    if valid_len <= 0:
        raise ValueError(f"valid_len={valid_len}")
    close = np.asarray(features["close"].values, dtype=np.float64)
    timestamps, future_returns, p_longs, p_shorts = [], [], [], []
    for k in range(valid_len):
        row_idx = window_size + k - 1
        ts = features.index[row_idx] if isinstance(features.index, pd.DatetimeIndex) else row_idx
        c = close[row_idx]
        c_future = close[row_idx + horizon]
        fr = (c_future - c) / c if c > 0 else 0.0
        timestamps.append(ts)
        future_returns.append(fr)
        p_longs.append(pl[k])
        p_shorts.append(ps[k])
    p_long_arr = np.asarray(p_longs, dtype=np.float32)
    p_short_arr = np.asarray(p_shorts, dtype=np.float32)
    p_flat_arr = np.clip(1.0 - p_long_arr - p_short_arr, 0.0, 1.0)
    max_proba = np.maximum(np.maximum(p_long_arr, p_short_arr), p_flat_arr)
    entropy = np.zeros(len(p_long_arr), dtype=np.float64)
    for i in range(len(p_long_arr)):
        for p in (p_long_arr[i], p_flat_arr[i], p_short_arr[i]):
            if p > 1e-12:
                entropy[i] -= p * math.log2(p)
    long_edge = p_long_arr - p_short_arr
    argmax_class = np.argmax(np.stack([p_flat_arr, p_long_arr, p_short_arr], axis=1), axis=1)
    argmax_label = np.where(argmax_class == 0, "FLAT", np.where(argmax_class == 1, "LONG", "SHORT"))
    future_return_arr = np.asarray(future_returns, dtype=np.float64)
    long_side_adj = future_return_arr
    short_side_adj = -future_return_arr
    argmax_side_adj = np.where(argmax_class == 1, long_side_adj, np.where(argmax_class == 2, short_side_adj, np.nan))
    cost_adj_argmax = np.where(argmax_class == 1, long_side_adj - cost, np.where(argmax_class == 2, short_side_adj - cost, np.nan))
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "future_return_15bar": future_return_arr,
        "p_flat": p_flat_arr,
        "p_long": p_long_arr,
        "p_short": p_short_arr,
        "max_proba": max_proba,
        "entropy": entropy,
        "argmax_class": argmax_class,
        "argmax_label": argmax_label,
        "long_edge": long_edge,
        "short_edge": p_short_arr - p_long_arr,
        "long_side_adjusted_return": long_side_adj,
        "short_side_adjusted_return": short_side_adj,
        "argmax_side_adjusted_return": argmax_side_adj,
        "cost_adjusted_argmax_return": cost_adj_argmax,
    })
    pl = pl[:valid_len]
    ps = ps[:valid_len]
    feature_at_pred = features.iloc[window_size - 1 : window_size - 1 + valid_len].copy()
    if isinstance(feature_at_pred.index, pd.DatetimeIndex):
        feature_at_pred = feature_at_pred.reset_index(drop=True)
    idx = slice(window_size, window_size + valid_len)
    df_bt = features[["close", "high", "low"]].iloc[idx].copy()
    if isinstance(features.index, pd.DatetimeIndex):
        df_bt["timestamp"] = features.index[idx]
    else:
        df_bt["timestamp"] = features["timestamp"].values[idx]
    df_bt = df_bt.reset_index(drop=True)
    return raw, features, feature_at_pred, pl, ps, model, df_bt


def run_permutation_importance(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    feature_at_pred: pd.DataFrame,
    model,
    symbol: str,
    timeframe: str,
    window_size: int,
    horizon: int,
    out_dir: Path,
) -> None:
    """Part A: Permutation importance with log_loss, entropy_mean, max_proba_mean, accuracy, alignment (Spearman) deltas."""
    from src.dl.data.labels import create_3class_labels

    N = len(features)
    valid_len = N - window_size - horizon
    y_true = create_3class_labels(raw["future_return_15bar"].values, pos_threshold=POS_THRESHOLD, neg_threshold=NEG_THRESHOLD)
    pl_b = raw["p_long"].values
    ps_b = raw["p_short"].values
    pf_b = np.clip(1.0 - pl_b - ps_b, 0.0, 1.0)
    probs_b = np.stack([pf_b, pl_b, ps_b], axis=1)
    p_correct_b = np.clip(np.array([probs_b[i, y_true[i]] for i in range(len(y_true))]), 1e-15, 1.0)
    baseline_log_loss = -np.mean(np.log(p_correct_b))
    baseline_entropy_mean = float(raw["entropy"].mean())
    baseline_max_proba_mean = float(raw["max_proba"].mean())
    baseline_accuracy = (np.argmax(probs_b, axis=1) == y_true).mean()
    baseline_spearman = raw["p_long"].corr(raw["future_return_15bar"], method="spearman")
    if pd.isna(baseline_spearman):
        baseline_spearman = 0.0

    feature_cols = [c for c in features.columns if c in feature_at_pred.columns]
    rows = []
    for idx, col in enumerate(feature_cols):
        feat_shuffled = features.copy()
        feat_shuffled[col] = feat_shuffled[col].sample(frac=1, random_state=42).values
        try:
            pl, ps = model.predict_proba_batch(features=feat_shuffled, symbol=symbol, timeframe=timeframe, batch_size=512)
        except Exception:
            rows.append({
                "feature_name": col,
                "importance_log_loss_delta": np.nan,
                "importance_entropy_delta": np.nan,
                "importance_max_proba_delta": np.nan,
                "importance_accuracy_delta": np.nan,
                "importance_alignment_delta": np.nan,
                "rank": np.nan,
            })
            continue
        pl = np.asarray(pl, dtype=np.float32)[:valid_len]
        ps = np.asarray(ps, dtype=np.float32)[:valid_len]
        pf = np.clip(1.0 - pl - ps, 0.0, 1.0)
        probs = np.stack([pf, pl, ps], axis=1)
        p_corr = np.clip(np.array([probs[i, y_true[i]] for i in range(len(y_true))]), 1e-15, 1.0)
        shuf_log_loss = -np.mean(np.log(p_corr))
        max_p = np.maximum(np.maximum(pl, ps), pf)
        ent = np.zeros(len(pl), dtype=np.float64)
        for i in range(len(pl)):
            for p in (pl[i], pf[i], ps[i]):
                if p > 1e-12:
                    ent[i] -= p * math.log2(p)
        shuf_entropy_mean = float(np.mean(ent))
        shuf_max_proba_mean = float(np.mean(max_p))
        shuf_accuracy = (np.argmax(probs, axis=1) == y_true).mean()
        shuf_spearman = pd.Series(pl).corr(pd.Series(raw["future_return_15bar"].values), method="spearman")
        if pd.isna(shuf_spearman):
            shuf_spearman = 0.0
        rows.append({
            "feature_name": col,
            "importance_log_loss_delta": shuf_log_loss - baseline_log_loss,
            "importance_entropy_delta": shuf_entropy_mean - baseline_entropy_mean,
            "importance_max_proba_delta": baseline_max_proba_mean - shuf_max_proba_mean,
            "importance_accuracy_delta": shuf_accuracy - baseline_accuracy,
            "importance_alignment_delta": shuf_spearman - baseline_spearman,
            "rank": np.nan,
        })
        if (idx + 1) % 8 == 0:
            print(f"  [Viability] Permutation {idx+1}/{len(feature_cols)}", flush=True)
    res = pd.DataFrame(rows)
    if not res.empty:
        res["rank"] = res["importance_log_loss_delta"].rank(ascending=False, method="min").fillna(0).astype(int)
        res = res.sort_values("rank")
    res.to_csv(out_dir / "permutation_importance.csv", index=False)
    print(f"[Viability] Wrote permutation_importance.csv", flush=True)


def run_prediction_decile_return(raw: pd.DataFrame, out_dir: Path) -> None:
    """Part B (A): Decile return by p_long, p_short, max_proba, entropy, long_edge."""
    score_cols = [
        ("p_long", "p_long", False),
        ("p_short", "p_short", False),
        ("max_proba", "max_proba", False),
        ("entropy", "entropy", True),   # lower is better -> decile 9 = bottom 10% entropy
        ("long_edge", "long_edge", False),
    ]
    cost = COST
    all_rows = []
    for score_name, col, lower_better in score_cols:
        raw_c = raw.copy()
        try:
            raw_c["decile"] = pd.qcut(raw_c[col], q=10, labels=False, duplicates="drop")
        except Exception:
            continue
        if lower_better:
            raw_c["decile"] = 9 - raw_c["decile"].astype(float)  # so high decile = "top" (low entropy)
        for d in raw_c["decile"].dropna().unique():
            sub = raw_c[raw_c["decile"] == d]
            if len(sub) < 10:
                continue
            mean_fr = sub["future_return_15bar"].mean()
            median_fr = sub["future_return_15bar"].median()
            mean_long = sub["long_side_adjusted_return"].mean()
            mean_short = sub["short_side_adjusted_return"].mean()
            mean_argmax = sub["argmax_side_adjusted_return"].mean()
            mean_cost_adj = sub["cost_adjusted_argmax_return"].mean()
            pred_side = np.where(sub["argmax_class"] == 1, 1, np.where(sub["argmax_class"] == 2, -1, 0))
            hit = np.mean((np.sign(sub["future_return_15bar"].values) == pred_side) & (pred_side != 0))
            all_rows.append({
                "score_type": score_name,
                "decile": int(d),
                "count": len(sub),
                "mean_future_return": mean_fr,
                "median_future_return": median_fr,
                "mean_long_side_adjusted_return": mean_long,
                "mean_short_side_adjusted_return": mean_short,
                "mean_argmax_side_adjusted_return": mean_argmax,
                "mean_cost_adjusted_argmax_return": mean_cost_adj,
                "hit_rate_direction": hit,
                "avg_max_proba": sub["max_proba"].mean(),
                "avg_entropy": sub["entropy"].mean(),
            })
    pd.DataFrame(all_rows).to_csv(out_dir / "prediction_decile_return.csv", index=False)
    print(f"[Viability] Wrote prediction_decile_return.csv", flush=True)


def run_prediction_top_bucket(raw: pd.DataFrame, out_dir: Path) -> None:
    """Part B (B): Top bucket / extreme percentile summary."""
    percentiles = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    cost = COST
    rows = []
    n = len(raw)
    for pct in percentiles:
        k = max(1, int(n * pct / 100))
        # max_proba top
        top_proba = raw.nlargest(k, "max_proba")
        mean_adj = top_proba["argmax_side_adjusted_return"].mean()
        mean_cost = top_proba["cost_adjusted_argmax_return"].mean()
        pred_side = np.where(top_proba["argmax_class"] == 1, 1, np.where(top_proba["argmax_class"] == 2, -1, 0))
        hit = np.mean((np.sign(top_proba["future_return_15bar"].values) == pred_side) & (pred_side != 0))
        rows.append({
            "bucket_type": "max_proba_top",
            "percentile": pct,
            "count": len(top_proba),
            "mean_argmax_side_adjusted_return": mean_adj,
            "mean_cost_adjusted_argmax_return": mean_cost,
            "median_future_return": top_proba["future_return_15bar"].median(),
            "std_future_return": top_proba["future_return_15bar"].std(),
            "hit_rate_sign_correct": hit,
            "avg_max_proba": top_proba["max_proba"].mean(),
            "avg_entropy": top_proba["entropy"].mean(),
        })
        # entropy bottom
        bot_ent = raw.nsmallest(k, "entropy")
        mean_adj = bot_ent["argmax_side_adjusted_return"].mean()
        mean_cost = bot_ent["cost_adjusted_argmax_return"].mean()
        pred_side = np.where(bot_ent["argmax_class"] == 1, 1, np.where(bot_ent["argmax_class"] == 2, -1, 0))
        hit = np.mean((np.sign(bot_ent["future_return_15bar"].values) == pred_side) & (pred_side != 0))
        rows.append({
            "bucket_type": "entropy_bottom",
            "percentile": pct,
            "count": len(bot_ent),
            "mean_argmax_side_adjusted_return": mean_adj,
            "mean_cost_adjusted_argmax_return": mean_cost,
            "median_future_return": bot_ent["future_return_15bar"].median(),
            "std_future_return": bot_ent["future_return_15bar"].std(),
            "hit_rate_sign_correct": hit,
            "avg_max_proba": bot_ent["max_proba"].mean(),
            "avg_entropy": bot_ent["entropy"].mean(),
        })
        # long_edge top
        top_edge = raw.nlargest(k, "long_edge")
        mean_adj = top_edge["argmax_side_adjusted_return"].mean()
        mean_cost = top_edge["cost_adjusted_argmax_return"].mean()
        pred_side = np.where(top_edge["argmax_class"] == 1, 1, np.where(top_edge["argmax_class"] == 2, -1, 0))
        hit = np.mean((np.sign(top_edge["future_return_15bar"].values) == pred_side) & (pred_side != 0))
        rows.append({
            "bucket_type": "long_edge_top",
            "percentile": pct,
            "count": len(top_edge),
            "mean_argmax_side_adjusted_return": mean_adj,
            "mean_cost_adjusted_argmax_return": mean_cost,
            "median_future_return": top_edge["future_return_15bar"].median(),
            "std_future_return": top_edge["future_return_15bar"].std(),
            "hit_rate_sign_correct": hit,
            "avg_max_proba": top_edge["max_proba"].mean(),
            "avg_entropy": top_edge["entropy"].mean(),
        })
    pd.DataFrame(rows).to_csv(out_dir / "prediction_top_bucket_summary.csv", index=False)
    print(f"[Viability] Wrote prediction_top_bucket_summary.csv", flush=True)


def run_regime_filtered_backtest(
    raw: pd.DataFrame,
    feature_at_pred: pd.DataFrame,
    df_bt: pd.DataFrame,
    pl: np.ndarray,
    ps: np.ndarray,
    out_dir: Path,
) -> None:
    """Part C: Regime tertiles (trend, vol), run backtest on each subset + full baseline."""
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    raw = raw.copy()
    raw["close"] = feature_at_pred["close"].values
    raw["ema_20"] = feature_at_pred["ema_20"].values
    raw["rolling_std_20"] = feature_at_pred["rolling_std_20"].values
    raw["trend_ratio"] = np.abs(raw["close"] - raw["ema_20"]) / np.maximum(raw["ema_20"], 1e-12)
    raw["vol"] = raw["rolling_std_20"]
    try:
        raw["trend_q"] = pd.qcut(raw["trend_ratio"], q=3, labels=[0, 1, 2], duplicates="drop")
    except Exception:
        raw["trend_q"] = 0
    try:
        raw["vol_q"] = pd.qcut(raw["vol"], q=3, labels=[0, 1, 2], duplicates="drop")
    except Exception:
        raw["vol_q"] = 0

    def run_one(df_bt_sub, pl_sub, ps_sub, regime_name: str, count: int) -> dict:
        if len(df_bt_sub) < 100:
            return {"regime_name": regime_name, "count": count, "cost_on": np.nan, "MDD": np.nan, "trades": 0, "entries_attempted": np.nan, "entries_executed": np.nan, "win_rate": np.nan, "mean_hold": np.nan, "notes": "subset_too_small"}
        res, err = run_backtest_7d(
            SYMBOL, TIMEFRAME, df_bt_sub, pl_sub, ps_sub, COMMISSION, SLIPPAGE,
            min_max_proba=MIN_MAX_PROBA, max_entropy=MAX_ENTROPY,
            decision_mode="argmax",
            min_hold=MIN_HOLD, cooldown=COOLDOWN,
            early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
            time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
            regime_filter_enabled=False, position_scaling_enabled=False,
        )
        if err:
            return {"regime_name": regime_name, "count": count, "cost_on": np.nan, "MDD": np.nan, "trades": 0, "entries_attempted": np.nan, "entries_executed": np.nan, "win_rate": np.nan, "mean_hold": np.nan, "notes": err}
        cap = res.get("cap_trigger_stats") or {}
        mean_hold = np.nan
        if res.get("trade_events"):
            exits = [e.get("bars_held", 0) for e in res["trade_events"] if e.get("event", "").startswith("EXIT")]
            if exits:
                mean_hold = float(np.mean(exits))
        return {
            "regime_name": regime_name,
            "count": count,
            "cost_on": res.get("total_return"),
            "MDD": res.get("max_drawdown"),
            "trades": res.get("total_trades"),
            "entries_attempted": res.get("entries_attempted"),
            "entries_executed": cap.get("entries_executed"),
            "win_rate": res.get("win_rate"),
            "mean_hold": mean_hold,
            "notes": "",
        }

    rows = []
    # Full baseline
    full = run_one(df_bt, pl, ps, "full", len(df_bt))
    rows.append(full)
    # Trend tertiles
    for q in [0, 1, 2]:
        mask = (raw["trend_q"] == q).fillna(False)
        if mask.sum() < 100:
            rows.append({"regime_name": f"trend_q{q}", "count": int(mask.sum()), "cost_on": np.nan, "MDD": np.nan, "trades": 0, "entries_attempted": np.nan, "entries_executed": np.nan, "win_rate": np.nan, "mean_hold": np.nan, "notes": "subset_too_small"})
            continue
        rows.append(run_one(df_bt.iloc[mask.values].reset_index(drop=True), pl[mask.values], ps[mask.values], f"trend_q{q}", int(mask.sum())))
    # Vol tertiles
    for q in [0, 1, 2]:
        mask = (raw["vol_q"] == q).fillna(False)
        if mask.sum() < 100:
            rows.append({"regime_name": f"vol_q{q}", "count": int(mask.sum()), "cost_on": np.nan, "MDD": np.nan, "trades": 0, "entries_attempted": np.nan, "entries_executed": np.nan, "win_rate": np.nan, "mean_hold": np.nan, "notes": "subset_too_small"})
            continue
        rows.append(run_one(df_bt.iloc[mask.values].reset_index(drop=True), pl[mask.values], ps[mask.values], f"vol_q{q}", int(mask.sum())))
    pd.DataFrame(rows).to_csv(out_dir / "regime_filtered_backtest.csv", index=False)
    print(f"[Viability] Wrote regime_filtered_backtest.csv", flush=True)


def write_summary(out_dir: Path) -> None:
    """Write tcn_viability_summary.md and .json with verdict."""
    md_path = out_dir / "tcn_viability_summary.md"
    json_path = out_dir / "tcn_viability_summary.json"

    pi = pd.read_csv(out_dir / "permutation_importance.csv") if (out_dir / "permutation_importance.csv").exists() else pd.DataFrame()
    dec = pd.read_csv(out_dir / "prediction_decile_return.csv") if (out_dir / "prediction_decile_return.csv").exists() else pd.DataFrame()
    top = pd.read_csv(out_dir / "prediction_top_bucket_summary.csv") if (out_dir / "prediction_top_bucket_summary.csv").exists() else pd.DataFrame()
    reg = pd.read_csv(out_dir / "regime_filtered_backtest.csv") if (out_dir / "regime_filtered_backtest.csv").exists() else pd.DataFrame()

    # Verdict logic
    perm_meaningful = False
    if not pi.empty and pi["importance_log_loss_delta"].notna().any():
        top_delta = pi["importance_log_loss_delta"].max()
        if top_delta > 0.001:  # some feature clearly matters
            perm_meaningful = True
    event_importance = 0.0
    if not pi.empty:
        ev = pi[pi["feature_name"].str.startswith("event_")]
        if not ev.empty and ev["importance_log_loss_delta"].notna().any():
            event_importance = ev["importance_log_loss_delta"].max()

    usable_signal_top = False
    if not top.empty:
        cost_adj = top["mean_cost_adjusted_argmax_return"]
        if cost_adj.notna().any() and (cost_adj > 0.0002).any():
            usable_signal_top = True
    if not top.empty and not usable_signal_top:
        # check smallest percentiles
        small = top[top["percentile"] <= 1.0]
        if not small.empty and small["mean_cost_adjusted_argmax_return"].notna().any():
            if small["mean_cost_adjusted_argmax_return"].max() > 0:
                usable_signal_top = True

    regime_better = False
    if not reg.empty:
        full_row = reg[reg["regime_name"] == "full"]
        if len(full_row):
            full_cost = full_row["cost_on"].iloc[0]
            if pd.notna(full_cost):
                others = reg[reg["regime_name"] != "full"]
                for _, r in others.iterrows():
                    if pd.notna(r.get("cost_on")) and r["count"] >= 500 and r["cost_on"] > full_cost + 0.02:
                        regime_better = True
                        break

    if perm_meaningful or usable_signal_top or regime_better:
        verdict = "TCN_STILL_WORTH_PUSHING"
        reason = "perm_meaningful=%s, usable_signal_top=%s, regime_better=%s" % (perm_meaningful, usable_signal_top, regime_better)
    else:
        verdict = "MOVE_TO_FEATURE_RESEARCH"
        reason = "perm weak, no usable signal in top buckets, no regime improvement"

    lines = [
        "# TCN Viability Final Diagnostic: 요약",
        "",
        "## 1. 사용 모델",
        f"- model_id: {MODEL_ID}",
        f"- seq_len: {WINDOW_SIZE}, horizon: {HORIZON}, threshold: ±{POS_THRESHOLD}",
        f"- symbol: {SYMBOL}, timeframe: {TIMEFRAME}, days: {DAYS}",
        "",
        "## 2. Feature importance 요약",
        ""
    ]
    if not pi.empty:
        top10 = pi.nlargest(10, "importance_log_loss_delta")
        for _, r in top10.iterrows():
            lines.append(f"- {r['feature_name']}: log_loss_delta={r['importance_log_loss_delta']:.4f}, rank={r['rank']}")
        lines.append("- Event feature 기여: " + ("일부 있음" if event_importance > 0.001 else "거의 없음"))
    else:
        lines.append("- (데이터 없음)")
    lines.extend(["", "## 3. Decile / Top bucket 요약", ""])
    if not top.empty:
        for bucket in ["max_proba_top", "entropy_bottom", "long_edge_top"]:
            sub = top[top["bucket_type"] == bucket]
            if sub.empty:
                continue
            best = sub.loc[sub["mean_cost_adjusted_argmax_return"].idxmax()] if sub["mean_cost_adjusted_argmax_return"].notna().any() else None
            if best is not None and pd.notna(best.get("mean_cost_adjusted_argmax_return")):
                lines.append(f"- {bucket}: best pct={best['percentile']}, count={best['count']}, mean_cost_adj={best['mean_cost_adjusted_argmax_return']:.6f}")
        lines.append("- usable signal (상위 극소수 구간 양수): " + ("있음" if usable_signal_top else "없음"))
    else:
        lines.append("- (데이터 없음)")
    lines.extend(["", "## 4. Regime-filtered backtest 요약", ""])
    if not reg.empty:
        full_row = reg[reg["regime_name"] == "full"]
        if len(full_row):
            lines.append(f"- full baseline: cost_on={full_row['cost_on'].iloc[0]:.4f}, MDD={full_row['MDD'].iloc[0]:.4f}, trades={full_row['trades'].iloc[0]}")
        for _, r in reg.iterrows():
            if r["regime_name"] == "full":
                continue
            lines.append(f"- {r['regime_name']}: count={r['count']}, cost_on={r.get('cost_on', np.nan):.4f}, MDD={r.get('MDD', np.nan):.4f}")
        lines.append("- 특정 regime에서 baseline 대비 개선: " + ("있음" if regime_better else "없음"))
    else:
        lines.append("- (데이터 없음)")
    lines.extend([
        "",
        "## 5. 최종 판정",
        f"- **{verdict}**",
        "",
        "## 6. 근거",
        f"- {reason}",
        "- permutation importance 상위 feature 사용 여부, decile/top bucket cost_adj 양수 여부, regime subset cost_on 개선 여부로 판정.",
        "",
        "## 7. 바로 다음 step",
        "- " + ("특정 regime-only 전략 실험 또는 threshold 미세 조정" if verdict == "TCN_STILL_WORTH_PUSHING" else "feature research branch 시작 (orderbook/funding/volatility 등 signal 탐색)"),
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Viability] Wrote {md_path}", flush=True)

    payload = {
        "model_id": MODEL_ID,
        "seq_len": WINDOW_SIZE,
        "horizon": HORIZON,
        "verdict": verdict,
        "reason": reason,
        "perm_meaningful": perm_meaningful,
        "usable_signal_top": usable_signal_top,
        "regime_better": regime_better,
        "top10_features": pi.nlargest(10, "importance_log_loss_delta")["feature_name"].tolist() if not pi.empty else [],
        "bottom10_features": pi.nsmallest(10, "importance_log_loss_delta")["feature_name"].tolist() if not pi.empty and len(pi) >= 10 else [],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[Viability] Wrote {json_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="TCN Viability Final Diagnostic")
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--skip-permutation", action="store_true", help="Skip permutation (slow)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"tcn_{MODEL_ID}.pt"
    if not model_path.exists():
        print(f"ERROR: model not found {model_path}", file=sys.stderr)
        return 1

    print("[Viability] Loading data and baseline inference ...", flush=True)
    raw, features, feature_at_pred, pl, ps, model, df_bt = load_data_and_baseline_inference(
        SYMBOL, TIMEFRAME, args.days, END_DATE, model_path, HORIZON, WINDOW_SIZE, COST,
    )
    print(f"[Viability] Aligned rows: {len(raw)}", flush=True)

    if not args.skip_permutation:
        run_permutation_importance(raw, features, feature_at_pred, model, SYMBOL, TIMEFRAME, WINDOW_SIZE, HORIZON, out_dir)
    else:
        pd.DataFrame(columns=["feature_name", "importance_log_loss_delta", "importance_entropy_delta", "importance_max_proba_delta", "importance_accuracy_delta", "importance_alignment_delta", "rank"]).to_csv(out_dir / "permutation_importance.csv", index=False)
        print("[Viability] Skip permutation (--skip-permutation)", flush=True)

    run_prediction_decile_return(raw, out_dir)
    run_prediction_top_bucket(raw, out_dir)
    run_regime_filtered_backtest(raw, feature_at_pred, df_bt, pl, ps, out_dir)
    write_summary(out_dir)
    print("[Viability] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
