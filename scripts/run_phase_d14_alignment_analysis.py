#!/usr/bin/env python3
"""
Phase D14-Prep: TCN 출력(probability/edge/entropy)과 실현 수익(realized forward return) 정렬도 분석.

목적: tcn_h15_t0p004.pt 등 현재 모델의 출력이 실제 트레이딩 edge와 어떻게 연결되는지 정밀 분석.
- Raw aligned dataset 생성 (row alignment 오프바이원 금지)
- Decile/quantile 분석
- Top-percentile 전략 분석
- 2D 조합/heatmap 분석
- Correlation/monotonicity 및 baseline subset 분석
- d14_summary.md / d14_summary.json 생성
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
END_DATE = "2026-03-03"
DAYS_LIST = [180, 365, 720]
WINDOW_SIZE = 60
HORIZON_DEFAULT = 15  # h15
COST_DEFAULT = 0.001
MODELS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "models"
OUT_DIR_DEFAULT = PROJECT_ROOT / "data" / "diagnostics" / "d14_alignment"
BASELINE_MIN_MAX_PROBA = 0.575
BASELINE_MAX_ENTROPY = 1.30


def _parse_model_id(model_id: str) -> tuple[int, float]:
    """e.g. h15_t0p004 -> (15, 0.004)."""
    parts = model_id.strip().split("_")
    if len(parts) < 2 or not parts[0].startswith("h") or not parts[1].startswith("t"):
        return (HORIZON_DEFAULT, 0.004)
    try:
        h = int(parts[0][1:])
        t_str = parts[1][1:].replace("p", ".")
        thr = float(t_str)
        return (h, thr)
    except (ValueError, IndexError):
        return (HORIZON_DEFAULT, 0.004)


def _load_ohlcv_range_5m(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame | None:
    """Load 5m OHLCV only for the requested range from CSV to avoid loading full 500k+ rows."""
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


def _load_ohlcv_and_features(symbol: str, timeframe: str, days: int, end_date: str, feature_config):
    from src.indicators.basic import add_basic_indicators
    from src.ml.features import build_feature_frame

    start_ts = pd.Timestamp(end_date).tz_localize("UTC") - pd.Timedelta(days=days)
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
    start_naive = start_ts.tz_localize(None) if start_ts.tz else start_ts
    end_naive = end_ts.tz_localize(None) if end_ts.tz else end_ts
    if timeframe == "5m" and symbol.upper() == "BTCUSDT":
        df = _load_ohlcv_range_5m(symbol, start_naive, end_naive)
        if df is None:
            from src.services.ohlcv_service import load_ohlcv_df
            df = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
            if df["timestamp"].dt.tz is None:
                start_ts, end_ts = start_naive, end_naive
            else:
                start_ts, end_ts = start_ts, end_ts
            df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    else:
        from src.services.ohlcv_service import load_ohlcv_df
        df = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            start_ts, end_ts = start_naive, end_naive
        else:
            start_ts, end_ts = start_ts, end_ts
        df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if len(df) < 100:
        return None, f"rows={len(df)} < 100"
    features = build_feature_frame(df, symbol=symbol, timeframe=timeframe, feature_config=feature_config)
    features = features.dropna()
    if len(features) < WINDOW_SIZE + 1:
        return None, f"features rows={len(features)} < {WINDOW_SIZE + 1}"
    if "close" not in features.columns:
        return None, "features missing close"
    return (df, features), None


def _build_raw_aligned(
    features: pd.DataFrame,
    proba_long: np.ndarray,
    proba_short: np.ndarray,
    horizon: int,
    cost: float,
    window_size: int,
) -> pd.DataFrame:
    """
    Align TCN predictions with future return. No off-by-one.
    - Prediction index k corresponds to feature row (window_size + k - 1).
    - Valid k: 0 .. (N - window_size - horizon) - 1 so that row_idx+horizon exists.
    """
    N = len(features)
    num_pred = len(proba_long)
    assert num_pred == N - window_size, f"num_pred={num_pred} vs N-window={N - window_size}"
    valid_len = N - window_size - horizon
    if valid_len <= 0:
        raise ValueError(f"valid_len={valid_len} (N={N}, window={window_size}, horizon={horizon})")
    close = np.asarray(features["close"].values, dtype=np.float64)
    timestamps = []
    closes = []
    future_returns = []
    p_longs = []
    p_shorts = []
    for k in range(valid_len):
        row_idx = window_size + k - 1
        ts = features.index[row_idx] if isinstance(features.index, pd.DatetimeIndex) else row_idx
        c = close[row_idx]
        c_future = close[row_idx + horizon]
        fr = (c_future - c) / c if c > 0 else 0.0
        timestamps.append(ts)
        closes.append(c)
        future_returns.append(fr)
        p_longs.append(proba_long[k])
        p_shorts.append(proba_short[k])
    p_long_arr = np.asarray(p_longs, dtype=np.float32)
    p_short_arr = np.asarray(p_shorts, dtype=np.float32)
    p_flat_arr = np.clip(1.0 - p_long_arr - p_short_arr, 0.0, 1.0)
    future_return_arr = np.asarray(future_returns, dtype=np.float64)
    abs_fr = np.abs(future_return_arr)
    argmax_class = np.argmax(np.stack([p_flat_arr, p_long_arr, p_short_arr], axis=1), axis=1)
    max_proba = np.maximum(np.maximum(p_long_arr, p_short_arr), p_flat_arr)
    entropy = np.zeros(len(p_long_arr), dtype=np.float64)
    for i in range(len(p_long_arr)):
        for p in (p_long_arr[i], p_flat_arr[i], p_short_arr[i]):
            if p > 1e-12:
                entropy[i] -= p * math.log2(p)
    long_edge = p_long_arr - p_short_arr
    short_edge = p_short_arr - p_long_arr
    side_over_flat_long = p_long_arr - p_flat_arr
    side_over_flat_short = p_short_arr - p_flat_arr
    future_direction_sign = np.sign(future_return_arr)
    predicted_side_sign = np.where(argmax_class == 1, 1, np.where(argmax_class == 2, -1, 0))
    correct_direction_binary = (future_direction_sign == predicted_side_sign) & (predicted_side_sign != 0)
    long_side_adjusted = future_return_arr
    short_side_adjusted = -future_return_arr
    cost_adj_long = long_side_adjusted - cost
    cost_adj_short = short_side_adjusted - cost
    argmax_side_adjusted = np.where(argmax_class == 1, long_side_adjusted, np.where(argmax_class == 2, short_side_adjusted, np.nan))
    cost_adj_argmax = np.where(argmax_class == 1, cost_adj_long, np.where(argmax_class == 2, cost_adj_short, np.nan))
    argmax_label = np.where(argmax_class == 0, "FLAT", np.where(argmax_class == 1, "LONG", "SHORT"))
    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": closes,
        "future_return_15bar": future_return_arr,
        "abs_future_return": abs_fr,
        "p_flat": p_flat_arr,
        "p_long": p_long_arr,
        "p_short": p_short_arr,
        "argmax_class": argmax_class,
        "argmax_label": argmax_label,
        "max_proba": max_proba,
        "entropy": entropy,
        "long_edge": long_edge,
        "short_edge": short_edge,
        "side_over_flat_long": side_over_flat_long,
        "side_over_flat_short": side_over_flat_short,
        "future_direction_sign": future_direction_sign,
        "predicted_side_sign": predicted_side_sign,
        "correct_direction_binary": correct_direction_binary,
        "long_side_adjusted_return": long_side_adjusted,
        "short_side_adjusted_return": short_side_adjusted,
        "cost_adjusted_long_return": cost_adj_long,
        "cost_adjusted_short_return": cost_adj_short,
        "argmax_side_adjusted_return": argmax_side_adjusted,
        "cost_adjusted_argmax_side_return": cost_adj_argmax,
    })
    return df


def _run_inference_and_build_raw(
    symbol: str,
    timeframe: str,
    days: int,
    end_date: str,
    model_path: Path,
    horizon: int,
    cost: float,
    feature_config,
    use_events: bool,
) -> tuple[pd.DataFrame | None, str | None]:
    (df, features), err = _load_ohlcv_and_features(symbol, timeframe, days, end_date, feature_config)
    if err:
        return None, err
    print(f"  [D14] OHLCV/features loaded: rows={len(features)}", flush=True)
    from src.dl.tcn_model import TCNSignalModel
    model = TCNSignalModel(model_path=model_path, use_events=use_events)
    if not model.is_loaded():
        return None, "TCN model failed to load"
    print(f"  [D14] Running TCN inference ...", flush=True)
    pl, ps = model.predict_proba_batch(features=features, symbol=symbol, timeframe=timeframe, batch_size=512)
    pl = np.asarray(pl, dtype=np.float32)
    ps = np.asarray(ps, dtype=np.float32)
    window_size = getattr(model, "window_size", WINDOW_SIZE)
    raw = _build_raw_aligned(features, pl, ps, horizon, cost, window_size)
    return raw, None


def _decile_analysis(df: pd.DataFrame, cost: float, quantiles: int = 10) -> dict:
    out = {}
    df = df.copy()
    # A. p_long decile
    df["p_long_decile"] = pd.qcut(df["p_long"], q=quantiles, labels=False, duplicates="drop")
    g = df.groupby("p_long_decile", dropna=True)
    agg = g.agg(
        count=("p_long", "count"),
        mean_future_return=("future_return_15bar", "mean"),
        median_future_return=("future_return_15bar", "median"),
        mean_long_side_adjusted=("long_side_adjusted_return", "mean"),
        mean_cost_adj_long=("cost_adjusted_long_return", "mean"),
        win_rate_long=("future_return_15bar", lambda x: (x > 0).mean()),
        hit_rate_threshold=("future_return_15bar", lambda x: (x > 0.004).mean()),
        avg_p_long=("p_long", "mean"),
        avg_p_flat=("p_flat", "mean"),
        avg_entropy=("entropy", "mean"),
    )
    out["p_long_decile"] = agg
    # B. p_short decile
    df["p_short_decile"] = pd.qcut(df["p_short"], q=quantiles, labels=False, duplicates="drop")
    g = df.groupby("p_short_decile", dropna=True)
    out["p_short_decile"] = g.agg(
        count=("p_short", "count"),
        mean_future_return=("future_return_15bar", "mean"),
        median_future_return=("future_return_15bar", "median"),
        mean_short_side_adjusted=("short_side_adjusted_return", "mean"),
        mean_cost_adj_short=("cost_adjusted_short_return", "mean"),
        win_rate_short=("future_return_15bar", lambda x: (x < 0).mean()),
        hit_rate_threshold_short=("future_return_15bar", lambda x: (x < -0.004).mean()),
        avg_p_short=("p_short", "mean"),
        avg_p_flat=("p_flat", "mean"),
        avg_entropy=("entropy", "mean"),
    )
    # C. p_flat decile
    df["p_flat_decile"] = pd.qcut(df["p_flat"], q=quantiles, labels=False, duplicates="drop")
    g = df.groupby("p_flat_decile", dropna=True)
    out["p_flat_decile"] = g.agg(
        count=("p_flat", "count"),
        mean_abs_future_return=("abs_future_return", "mean"),
        median_abs_future_return=("abs_future_return", "median"),
        prop_small_move=("future_return_15bar", lambda x: (np.abs(x) <= 0.004).mean()),
        avg_entropy=("entropy", "mean"),
        avg_max_proba=("max_proba", "mean"),
    )
    # D. long_edge decile
    df["long_edge_decile"] = pd.qcut(df["long_edge"], q=quantiles, labels=False, duplicates="drop")
    g = df.groupby("long_edge_decile", dropna=True)
    out["long_edge_decile"] = g.agg(
        count=("long_edge", "count"),
        mean_future_return=("future_return_15bar", "mean"),
        mean_cost_adj_long=("cost_adjusted_long_return", "mean"),
        hit_rate_long=("future_return_15bar", lambda x: (x > 0.004).mean()),
        avg_p_long=("p_long", "mean"),
        avg_p_short=("p_short", "mean"),
        avg_p_flat=("p_flat", "mean"),
    )
    # E. short_edge decile
    df["short_edge_decile"] = pd.qcut(df["short_edge"], q=quantiles, labels=False, duplicates="drop")
    g = df.groupby("short_edge_decile", dropna=True)
    out["short_edge_decile"] = g.agg(
        count=("short_edge", "count"),
        mean_short_side_adjusted=("short_side_adjusted_return", "mean"),
        mean_cost_adj_short=("cost_adjusted_short_return", "mean"),
        hit_rate_short=("future_return_15bar", lambda x: (x < -0.004).mean()),
    )
    # F. max_proba decile (LONG/SHORT rows only)
    non_flat = df[df["argmax_class"] != 0]
    if len(non_flat) > quantiles:
        non_flat = non_flat.copy()
        non_flat["max_proba_decile"] = pd.qcut(non_flat["max_proba"], q=quantiles, labels=False, duplicates="drop")
        g = non_flat.groupby("max_proba_decile", dropna=True)
        out["max_proba_decile"] = g.agg(
            count=("max_proba", "count"),
            mean_argmax_side_adjusted=("argmax_side_adjusted_return", "mean"),
            mean_cost_adj_argmax=("cost_adjusted_argmax_side_return", "mean"),
            avg_entropy=("entropy", "mean"),
            avg_abs_future_return=("abs_future_return", "mean"),
        )
    # G. entropy decile
    df["entropy_decile"] = pd.qcut(df["entropy"], q=quantiles, labels=False, duplicates="drop")
    g = df.groupby("entropy_decile", dropna=True)
    out["entropy_decile"] = g.agg(
        count=("entropy", "count"),
        mean_argmax_side_adjusted=("argmax_side_adjusted_return", "mean"),
        mean_cost_adj_argmax=("cost_adjusted_argmax_side_return", "mean"),
        avg_max_proba=("max_proba", "mean"),
        avg_abs_future_return=("abs_future_return", "mean"),
    )
    return out


def _top_percentile_analysis(df: pd.DataFrame, cost: float, top_pcts: list[float]) -> dict:
    out = {}
    for pct in top_pcts:
        q = 100 - pct
        # p_long top
        th = df["p_long"].quantile(q / 100.0)
        sub = df[df["p_long"] >= th]
        if len(sub) >= 10:
            out[f"p_long_top_{pct}pct"] = {
                "count": len(sub),
                "mean_future_return": sub["future_return_15bar"].mean(),
                "mean_long_side_adjusted": sub["long_side_adjusted_return"].mean(),
                "mean_cost_adj_long": sub["cost_adjusted_long_return"].mean(),
                "median": sub["future_return_15bar"].median(),
                "std": sub["future_return_15bar"].std(),
                "win_rate": (sub["future_return_15bar"] > 0).mean(),
                "hit_rate_004": (sub["future_return_15bar"] > 0.004).mean(),
            }
        # p_short top
        th = df["p_short"].quantile(q / 100.0)
        sub = df[df["p_short"] >= th]
        if len(sub) >= 10:
            out[f"p_short_top_{pct}pct"] = {
                "count": len(sub),
                "mean_future_return": sub["future_return_15bar"].mean(),
                "mean_short_side_adjusted": sub["short_side_adjusted_return"].mean(),
                "mean_cost_adj_short": sub["cost_adjusted_short_return"].mean(),
                "median": sub["future_return_15bar"].median(),
                "std": sub["future_return_15bar"].std(),
                "win_rate": (sub["future_return_15bar"] < 0).mean(),
                "hit_rate_004": (sub["future_return_15bar"] < -0.004).mean(),
            }
        # long_edge top
        th = df["long_edge"].quantile(q / 100.0)
        sub = df[df["long_edge"] >= th]
        if len(sub) >= 10:
            out[f"long_edge_top_{pct}pct"] = {
                "count": len(sub),
                "mean_future_return": sub["future_return_15bar"].mean(),
                "mean_cost_adj_long": sub["cost_adjusted_long_return"].mean(),
            }
        # short_edge top
        th = df["short_edge"].quantile(q / 100.0)
        sub = df[df["short_edge"] >= th]
        if len(sub) >= 10:
            out[f"short_edge_top_{pct}pct"] = {
                "count": len(sub),
                "mean_short_side_adjusted": sub["short_side_adjusted_return"].mean(),
                "mean_cost_adj_short": sub["cost_adjusted_short_return"].mean(),
            }
        # entropy bottom (low entropy = high confidence)
        q_lo = pct
        th = df["entropy"].quantile(q_lo / 100.0)
        sub = df[df["entropy"] <= th]
        if len(sub) >= 10:
            out[f"entropy_bottom_{pct}pct"] = {
                "count": len(sub),
                "mean_argmax_side_adjusted": sub["argmax_side_adjusted_return"].mean(),
                "mean_cost_adj_argmax": sub["cost_adjusted_argmax_side_return"].mean(),
            }
    return out


def _correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["p_long", "p_short", "p_flat", "long_edge", "short_edge", "max_proba", "entropy"]
    targets = ["future_return_15bar", "long_side_adjusted_return", "short_side_adjusted_return", "abs_future_return", "cost_adjusted_long_return", "cost_adjusted_short_return"]
    rows = []
    for t in targets:
        if t not in df.columns:
            continue
        for c in cols:
            if c not in df.columns:
                continue
            valid = df[[c, t]].dropna()
            if len(valid) < 30:
                continue
            sp = valid[c].corr(valid[t], method="spearman")
            pe = valid[c].corr(valid[t], method="pearson")
            rows.append({"target": t, "variable": c, "spearman": sp, "pearson": pe})
    return pd.DataFrame(rows)


def _combo_heatmap(df: pd.DataFrame, cost: float, n_bins: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2D: long_edge x p_long and short_edge x p_short; value = mean cost_adj pnl."""
    df = df.copy()
    df["le_bin"] = pd.qcut(df["long_edge"], q=n_bins, labels=False, duplicates="drop")
    df["pl_bin"] = pd.qcut(df["p_long"], q=n_bins, labels=False, duplicates="drop")
    pivot_long = df.pivot_table(values="cost_adjusted_long_return", index="le_bin", columns="pl_bin", aggfunc="mean")
    df["se_bin"] = pd.qcut(df["short_edge"], q=n_bins, labels=False, duplicates="drop")
    df["ps_bin"] = pd.qcut(df["p_short"], q=n_bins, labels=False, duplicates="drop")
    pivot_short = df.pivot_table(values="cost_adjusted_short_return", index="se_bin", columns="ps_bin", aggfunc="mean")
    return pivot_long, pivot_short


def _baseline_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that pass baseline filters: min_max_proba, max_entropy."""
    return df[(df["max_proba"] >= BASELINE_MIN_MAX_PROBA) & (df["entropy"] <= BASELINE_MAX_ENTROPY)].copy()


def main() -> int:
    from src.features.ml_feature_config import MLFeatureConfig

    parser = argparse.ArgumentParser(description="Phase D14-Prep: TCN output vs realized return alignment analysis")
    parser.add_argument("--model-id", type=str, default="h15_t0p004", help="Model id e.g. h15_t0p004")
    parser.add_argument("--symbol", type=str, default=SYMBOL)
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME)
    parser.add_argument("--end-date", type=str, default=END_DATE)
    parser.add_argument("--days", type=int, nargs="+", default=DAYS_LIST, help="e.g. 180 365 720")
    parser.add_argument("--cost", type=float, default=COST_DEFAULT, help="Cost for cost-adjusted return proxy")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--include-baseline-filter-subset", action="store_true")
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument("--top-percentiles", type=str, default="1,2,5,10,20")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"tcn_{args.model_id}.pt"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1
    horizon, _ = _parse_model_id(args.model_id)
    top_pcts = [int(x) for x in args.top_percentiles.split(",")]
    config = MLFeatureConfig.from_preset("base")
    config.use_event_features = True
    use_events = True

    all_raw = {}
    all_deciles = {}
    all_top_pct = {}
    all_corr = {}
    all_heatmap_long = {}
    all_heatmap_short = {}

    for days in args.days:
        print(f"[D14] Building raw aligned dataset for {days}d ...", flush=True)
        raw, err = _run_inference_and_build_raw(
            args.symbol, args.timeframe, days, args.end_date,
            model_path, horizon, args.cost, config, use_events,
        )
        if err:
            print(f"[D14] ERROR {days}d: {err}", file=sys.stderr)
            continue
        raw_path = out_dir / f"d14_alignment_raw_{days}d.parquet"
        raw.to_parquet(raw_path, index=False)
        # Alignment NaN 디버깅용 4개 지표 (반드시 출력)
        valid_rows_after_merge = len(raw)
        future_return_col = "future_return_15bar"
        future_return_non_na_count = int(raw[future_return_col].notna().sum()) if future_return_col in raw.columns else 0
        p_long_std = float(raw["p_long"].std()) if "p_long" in raw.columns else float("nan")
        long_edge_std = float(raw["long_edge"].std()) if "long_edge" in raw.columns else float("nan")
        print(f"[D14] DEBUG {days}d: valid_rows_after_merge={valid_rows_after_merge}, future_return_non_na_count={future_return_non_na_count}, p_long_std={p_long_std:.6f}, long_edge_std={long_edge_std:.6f}", flush=True)
        debug_path = out_dir / f"d14_alignment_debug_{days}d.csv"
        pd.DataFrame([{
            "model_id": args.model_id,
            "horizon": horizon,
            "days": days,
            "valid_rows_after_merge": valid_rows_after_merge,
            "future_return_non_na_count": future_return_non_na_count,
            "p_long_std": p_long_std,
            "long_edge_std": long_edge_std,
        }]).to_csv(debug_path, index=False)
        print(f"[D14] Wrote {raw_path} rows={len(raw)}", flush=True)
        all_raw[days] = raw

        print(f"[D14] Decile analysis {days}d ...", flush=True)
        dec = _decile_analysis(raw, args.cost, quantiles=args.quantiles)
        for name, tbl in dec.items():
            if isinstance(tbl, pd.DataFrame):
                tbl.to_csv(out_dir / f"d14_deciles_{name}_{days}d.csv")
        all_deciles[days] = {k: v.to_dict(orient="index") if isinstance(v, pd.DataFrame) else v for k, v in dec.items()}

        print(f"[D14] Top-percentile analysis {days}d ...", flush=True)
        top = _top_percentile_analysis(raw, args.cost, top_pcts)
        all_top_pct[days] = top

        print(f"[D14] Correlation analysis {days}d ...", flush=True)
        corr = _correlation_analysis(raw)
        corr_path = out_dir / f"d14_correlations_{days}d.csv"
        corr.to_csv(corr_path, index=False)
        all_corr[days] = corr

        print(f"[D14] 2D heatmap {days}d ...", flush=True)
        hm_long, hm_short = _combo_heatmap(raw, args.cost, n_bins=5)
        hm_long.to_csv(out_dir / f"d14_combo_heatmap_long_{days}d.csv")
        hm_short.to_csv(out_dir / f"d14_combo_heatmap_short_{days}d.csv")
        all_heatmap_long[days] = hm_long
        all_heatmap_short[days] = hm_short

        if args.include_baseline_filter_subset:
            sub = _baseline_subset(raw)
            if len(sub) >= 100:
                dec_sub = _decile_analysis(sub, args.cost, quantiles=args.quantiles)
                for name, tbl in dec_sub.items():
                    if isinstance(tbl, pd.DataFrame):
                        tbl.to_csv(out_dir / f"d14_deciles_baseline_subset_{name}_{days}d.csv")

    # Top percentiles summary table (720d if available)
    if 720 in all_top_pct:
        rows = []
        for k, v in all_top_pct[720].items():
            if isinstance(v, dict):
                v_copy = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv) for kk, vv in v.items()}
                v_copy["segment"] = k
                rows.append(v_copy)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "d14_top_percentiles_720d.csv", index=False)

    # Summary md
    summary_md = out_dir / "d14_summary.md"
    summary_json = out_dir / "d14_summary.json"
    _write_summary(
        out_dir=out_dir,
        model_path=str(model_path),
        model_id=args.model_id,
        horizon=horizon,
        days_list=list(all_raw.keys()),
        all_raw=all_raw,
        all_deciles=all_deciles,
        all_top_pct=all_top_pct,
        all_corr=all_corr,
        cost=args.cost,
        summary_md_path=summary_md,
        summary_json_path=summary_json,
    )
    print(f"[D14] Wrote {summary_md} and {summary_json}", flush=True)
    print("[D14] Done.", flush=True)
    return 0


def _write_summary(
    out_dir: Path,
    model_path: str,
    model_id: str,
    horizon: int,
    days_list: list[int],
    all_raw: dict,
    all_deciles: dict,
    all_top_pct: dict,
    all_corr: dict,
    cost: float,
    summary_md_path: Path,
    summary_json_path: Path,
):
    # Include any days that have raw parquet in out_dir (e.g. from a previous run)
    extra_days = []
    for p in out_dir.glob("d14_alignment_raw_*d.parquet"):
        try:
            suffix = p.stem.replace("d14_alignment_raw_", "").replace("d", "")
            extra_days.append(int(suffix))
        except ValueError:
            pass
    days_list_display = sorted(set(days_list) | set(extra_days))
    lines = [
        "# Phase D14-Prep: TCN 출력 vs 실현 수익 정렬도 분석",
        "",
        "## 1. 사용 모델",
        f"- 경로: `{model_path}`",
        f"- model_id: `{model_id}`",
        f"- horizon: {horizon} bars",
        "",
        "## 2. 데이터 구간",
        f"- {', '.join(str(d) + 'd' for d in days_list_display)}",
        "",
        "## 3. 비용 가정",
        f"- cost-adjusted return proxy에 사용한 cost: {cost} (0.1%)",
        "- 참고: 엄밀한 백테스트 PnL이 아닌 forward return proxy임.",
        "",
        "## 4. 생성 파일",
    ]
    for d in days_list_display:
        lines.append(f"- `d14_alignment_raw_{d}d.parquet` — Raw aligned dataset")
        lines.append(f"- `d14_deciles_{d}d.csv` — Decile 분석")
        lines.append(f"- `d14_correlations_{d}d.csv` — Correlation")
        lines.append(f"- `d14_combo_heatmap_long_{d}d.csv`, `d14_combo_heatmap_short_{d}d.csv`")
    lines.extend([
        "",
        "## 5. 핵심 결론 요약",
    ])
    ref_days = 720 if 720 in all_raw else (365 if 365 in all_raw else next(iter(all_raw.keys()), None))
    if ref_days is not None and len(all_raw[ref_days]) > 0:
        df = all_raw[ref_days]
        sp = df["p_long"].corr(df["future_return_15bar"], method="spearman")
        lines.append(f"- p_long vs future_return ({ref_days}d) Spearman: {sp:.4f}")
        sp_le = df["long_edge"].corr(df["future_return_15bar"], method="spearman")
        lines.append(f"- long_edge vs future_return ({ref_days}d) Spearman: {sp_le:.4f}")
        if ref_days in all_top_pct and "long_edge_top_5pct" in all_top_pct[ref_days]:
            v = all_top_pct[ref_days]["long_edge_top_5pct"]
            lines.append(f"- long_edge top 5% ({ref_days}d): count={v.get('count')}, mean_cost_adj_long={v.get('mean_cost_adj_long', 0):.4f}")
        lines.append("")
    lines.extend([
        "## 6. 유망 후보 rule 제안 (실제 분석 결과 기반으로 보완)",
        "- p_long >= 0.55 AND p_long > p_flat",
        "- long_edge 상위 percentile만 진입",
        "- max_proba 높음 AND entropy 낮음 구간만 사용",
        "",
        "## 7. 다음 단계 권고",
        "- D14 threshold 전략 실험 또는 재학습/calibration 검토",
        "",
    ])
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    # JSON: serializable summary
    payload = {
        "model_path": model_path,
        "model_id": model_id,
        "horizon": horizon,
        "days": days_list,
        "cost": cost,
        "row_counts": {d: len(all_raw[d]) for d in all_raw},
    }
    ref_corr = 720 if 720 in all_corr else (365 if 365 in all_corr else next(iter(all_corr.keys()), None))
    if ref_corr is not None and not all_corr[ref_corr].empty:
        payload[f"correlations_{ref_corr}d"] = all_corr[ref_corr].to_dict(orient="records")
    def _json_serializer(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_serializer)


if __name__ == "__main__":
    sys.exit(main())
