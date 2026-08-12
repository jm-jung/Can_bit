#!/usr/bin/env python3
"""
Model Understanding & Predictability Diagnostics Phase.

목적: 현재 TCN이 무엇을 학습하는지, BTCUSDT 5m horizon=15bar 예측이 가능한 문제인지 확인.
- 새 모델 학습/전략 변경/rule 실험 금지. 현재 모델·데이터만 사용.
"""
from __future__ import annotations

import argparse
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
MODEL_ID = "h15_t0p004"
MODELS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "models"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "model_understanding"


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


def load_data_and_run_baseline_inference(
    symbol: str,
    timeframe: str,
    days: int,
    end_date: str,
    model_path: Path,
    horizon: int,
    window_size: int,
):
    """Load OHLCV, build features (base), run TCN inference, return (raw_aligned_df, features_df, feature_at_pred, proba_long, proba_short)."""
    from src.features.ml_feature_config import MLFeatureConfig
    from src.indicators.basic import add_basic_indicators
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
    timestamps = []
    future_returns = []
    p_longs, p_shorts = [], []
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
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "future_return_15bar": np.asarray(future_returns, dtype=np.float64),
        "p_long": p_long_arr,
        "p_short": p_short_arr,
        "p_flat": p_flat_arr,
        "max_proba": max_proba,
        "entropy": entropy,
        "long_edge": long_edge,
    })
    feature_at_pred = features.iloc[window_size - 1 : window_size - 1 + valid_len].copy()
    if isinstance(feature_at_pred.index, pd.DatetimeIndex):
        feature_at_pred = feature_at_pred.reset_index(drop=True)
    return raw, features, feature_at_pred, pl, ps, model, config


def run_feature_vs_future_return(raw: pd.DataFrame, feature_at_pred: pd.DataFrame, out_dir: Path) -> None:
    """[1] Feature vs future_return_15bar: Pearson, Spearman, abs_spearman, rank."""
    merged = raw[["future_return_15bar"]].copy()
    for c in feature_at_pred.columns:
        merged[c] = feature_at_pred[c].values
    rows = []
    for col in feature_at_pred.columns:
        x = merged[col].replace([np.inf, -np.inf], np.nan)
        y = merged["future_return_15bar"]
        valid = x.notna()
        if valid.sum() < 100:
            continue
        x = x.loc[valid].values
        y = y.loc[valid].values
        pearson = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
        spearman = pd.Series(x).corr(pd.Series(y), method="spearman")
        if pd.isna(spearman):
            spearman = np.nan
        rows.append({
            "feature_name": col,
            "pearson_corr": pearson,
            "spearman_corr": spearman,
            "abs_spearman": abs(spearman) if not pd.isna(spearman) else np.nan,
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        rank_ser = res["abs_spearman"].rank(ascending=False, method="min")
        res["rank"] = rank_ser.fillna(0).astype(int)  # NaN (constant feature) -> rank 0
        res = res.sort_values("rank")
    res.to_csv(out_dir / "feature_vs_future_return.csv", index=False)
    print(f"[MU] Wrote feature_vs_future_return.csv ({len(res)} features)", flush=True)


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
    """[2] Permutation feature importance: shuffle each feature, measure log_loss change."""
    from src.dl.data.labels import LstmClassIndex, create_3class_labels

    N = len(features)
    valid_len = N - window_size - horizon
    y_true = create_3class_labels(
        raw["future_return_15bar"].values,
        pos_threshold=POS_THRESHOLD,
        neg_threshold=NEG_THRESHOLD,
    )
    p_long_baseline = raw["p_long"].values
    p_short_baseline = raw["p_short"].values
    p_flat_baseline = np.clip(1.0 - p_long_baseline - p_short_baseline, 0.0, 1.0)
    probs_baseline = np.stack([p_flat_baseline, p_long_baseline, p_short_baseline], axis=1)
    p_correct = np.array([probs_baseline[i, y_true[i]] for i in range(len(y_true))])
    p_correct = np.clip(p_correct, 1e-15, 1.0)
    baseline_log_loss = -np.mean(np.log(p_correct))

    feature_cols = [c for c in features.columns if c in feature_at_pred.columns]
    rows = []
    for idx, col in enumerate(feature_cols):
        feat_shuffled = features.copy()
        feat_shuffled[col] = feat_shuffled[col].sample(frac=1).values
        try:
            pl, ps = model.predict_proba_batch(features=feat_shuffled, symbol=symbol, timeframe=timeframe, batch_size=512)
        except Exception:
            rows.append({"feature_name": col, "importance_score": np.nan, "rank": np.nan})
            continue
        pl = np.asarray(pl, dtype=np.float32)[:valid_len]
        ps = np.asarray(ps, dtype=np.float32)[:valid_len]
        p_flat = np.clip(1.0 - pl - ps, 0.0, 1.0)
        probs = np.stack([p_flat, pl, ps], axis=1)
        p_corr = np.clip(np.array([probs[i, y_true[i]] for i in range(len(y_true))]), 1e-15, 1.0)
        shuffled_log_loss = -np.mean(np.log(p_corr))
        importance = shuffled_log_loss - baseline_log_loss
        rows.append({"feature_name": col, "importance_score": importance})
        if (idx + 1) % 10 == 0:
            print(f"  [MU] Permutation {idx+1}/{len(feature_cols)}", flush=True)
    res = pd.DataFrame(rows)
    if not res.empty:
        res["rank"] = res["importance_score"].rank(ascending=False, method="min").astype(int)
        res = res.sort_values("rank")
    res.to_csv(out_dir / "permutation_importance.csv", index=False)
    print(f"[MU] Wrote permutation_importance.csv", flush=True)


def run_event_feature_ablation(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    model,
    symbol: str,
    timeframe: str,
    window_size: int,
    horizon: int,
    out_dir: Path,
) -> None:
    """[3] Event feature ablation: full vs event-zeroed."""
    event_cols = [c for c in features.columns if c.startswith("event_")]
    valid_len = len(raw)
    # Experiment A: full (already have raw)
    p_long_spearman_a = raw["p_long"].corr(raw["future_return_15bar"], method="spearman")
    long_edge_spearman_a = raw["long_edge"].corr(raw["future_return_15bar"], method="spearman")
    entropy_mean_a = raw["entropy"].mean()
    max_proba_mean_a = raw["max_proba"].mean()
    # Experiment B: event columns zeroed
    features_no_ev = features.copy()
    for c in event_cols:
        features_no_ev[c] = 0.0
    pl_b, ps_b = model.predict_proba_batch(features=features_no_ev, symbol=symbol, timeframe=timeframe, batch_size=512)
    pl_b = np.asarray(pl_b, dtype=np.float32)[:valid_len]
    ps_b = np.asarray(ps_b, dtype=np.float32)[:valid_len]
    p_flat_b = np.clip(1.0 - pl_b - ps_b, 0.0, 1.0)
    long_edge_b = pl_b - ps_b
    p_long_spearman_b = pd.Series(pl_b).corr(raw["future_return_15bar"], method="spearman")
    long_edge_spearman_b = pd.Series(long_edge_b).corr(raw["future_return_15bar"], method="spearman")
    max_proba_b = np.maximum(np.maximum(pl_b, ps_b), p_flat_b)
    entropy_b = np.zeros(len(pl_b), dtype=np.float64)
    for i in range(len(pl_b)):
        for p in (pl_b[i], p_flat_b[i], ps_b[i]):
            if p > 1e-12:
                entropy_b[i] -= p * math.log2(p)
    entropy_mean_b = float(np.mean(entropy_b))
    max_proba_mean_b = float(np.mean(max_proba_b))
    res = pd.DataFrame([
        {"experiment": "A_full", "p_long_spearman": p_long_spearman_a, "long_edge_spearman": long_edge_spearman_a, "entropy_mean": entropy_mean_a, "max_proba_mean": max_proba_mean_a},
        {"experiment": "B_no_event", "p_long_spearman": p_long_spearman_b, "long_edge_spearman": long_edge_spearman_b, "entropy_mean": entropy_mean_b, "max_proba_mean": max_proba_mean_b},
    ])
    res.to_csv(out_dir / "event_feature_ablation.csv", index=False)
    print(f"[MU] Wrote event_feature_ablation.csv", flush=True)


def run_prediction_confidence_summary(raw: pd.DataFrame, out_dir: Path) -> None:
    """[4] max_proba, entropy distribution stats."""
    rows = []
    for name, ser in [("max_proba", raw["max_proba"]), ("entropy", raw["entropy"])]:
        rows.append({
            "metric": name,
            "mean": ser.mean(),
            "std": ser.std(),
            "p50": ser.quantile(0.50),
            "p75": ser.quantile(0.75),
            "p90": ser.quantile(0.90),
            "p95": ser.quantile(0.95),
            "p99": ser.quantile(0.99),
        })
    pd.DataFrame(rows).to_csv(out_dir / "prediction_confidence_summary.csv", index=False)
    print(f"[MU] Wrote prediction_confidence_summary.csv", flush=True)


def run_regime_analysis(raw: pd.DataFrame, feature_at_pred: pd.DataFrame, out_dir: Path) -> None:
    """[5] Regime: trend / volatility bins, p_long_spearman per regime."""
    if "ema_20" not in feature_at_pred.columns or "rolling_std_20" not in feature_at_pred.columns:
        pd.DataFrame(columns=["regime_type", "regime_name", "count", "p_long_spearman", "long_edge_spearman"]).to_csv(out_dir / "regime_alignment.csv", index=False)
        print(f"[MU] Wrote regime_alignment.csv (no ema/rolling)", flush=True)
        return
    raw = raw.copy()
    raw["close"] = feature_at_pred["close"].values
    raw["ema_20"] = feature_at_pred["ema_20"].values
    raw["rolling_std_20"] = feature_at_pred["rolling_std_20"].values
    raw["trend_ratio"] = np.abs(raw["close"] - raw["ema_20"]) / np.maximum(raw["ema_20"], 1e-12)
    raw["vol"] = raw["rolling_std_20"]
    rows = []
    for regime_type, col, q in [("trend", "trend_ratio", 3), ("volatility", "vol", 3)]:
        raw["_q"] = pd.qcut(raw[col], q=q, labels=False, duplicates="drop")
        for qid in raw["_q"].dropna().unique():
            sub = raw[raw["_q"] == qid]
            if len(sub) < 50:
                continue
            pl_sp = sub["p_long"].corr(sub["future_return_15bar"], method="spearman")
            le_sp = sub["long_edge"].corr(sub["future_return_15bar"], method="spearman")
            rows.append({"regime_type": regime_type, "regime_name": f"{regime_type}_q{qid}", "count": len(sub), "p_long_spearman": pl_sp, "long_edge_spearman": le_sp})
    pd.DataFrame(rows).to_csv(out_dir / "regime_alignment.csv", index=False)
    print(f"[MU] Wrote regime_alignment.csv", flush=True)


def run_probability_calibration(raw: pd.DataFrame, out_dir: Path) -> None:
    """[6] p_long decile -> future_return_mean, long_frequency, count."""
    raw = raw.copy()
    raw["p_long_decile"] = pd.qcut(raw["p_long"], q=10, labels=False, duplicates="drop")
    g = raw.groupby("p_long_decile", dropna=True).agg(
        p_long_mean=("p_long", "mean"),
        future_return_mean=("future_return_15bar", "mean"),
        count=("p_long", "count"),
    )
    g["long_frequency"] = raw.groupby("p_long_decile", dropna=True)["future_return_15bar"].apply(lambda x: (x > POS_THRESHOLD).mean())
    g = g.reset_index()
    g.to_csv(out_dir / "probability_calibration.csv", index=False)
    print(f"[MU] Wrote probability_calibration.csv", flush=True)


def run_predictability_test(raw: pd.DataFrame, out_dir: Path) -> None:
    """[7] TCN vs Random, Always FLAT, Always LONG, Always SHORT."""
    y = raw["future_return_15bar"].values
    n = len(y)
    # True labels for accuracy (3-class)
    from src.dl.data.labels import create_3class_labels
    y_class = create_3class_labels(y, pos_threshold=POS_THRESHOLD, neg_threshold=NEG_THRESHOLD)
    # TCN
    le_tcn = raw["long_edge"].values
    argmax_tcn = np.argmax(np.stack([raw["p_flat"].values, raw["p_long"].values, raw["p_short"].values], axis=1), axis=1)
    sp_tcn = pd.Series(le_tcn).corr(pd.Series(y), method="spearman")
    acc_tcn = (argmax_tcn == y_class).mean()
    ret_tcn = np.where(argmax_tcn == 1, y, np.where(argmax_tcn == 2, -y, 0.0)).mean()
    # Random (uniform prob -> expected long_edge 0, or sample random class)
    np.random.seed(42)
    argmax_rnd = np.random.randint(0, 3, size=n)
    le_rnd = np.where(argmax_rnd == 1, 1.0, np.where(argmax_rnd == 2, -1.0, 0.0))
    sp_rnd = pd.Series(le_rnd).corr(pd.Series(y), method="spearman")
    acc_rnd = (argmax_rnd == y_class).mean()
    ret_rnd = np.where(argmax_rnd == 1, y, np.where(argmax_rnd == 2, -y, 0.0)).mean()
    # Always FLAT
    le_flat = np.zeros(n)
    sp_flat = 0.0
    acc_flat = (y_class == 0).mean()
    ret_flat = 0.0
    # Always LONG
    le_long = np.ones(n)
    sp_long = 0.0  # constant
    acc_long = (y_class == 1).mean()
    ret_long = np.mean(y)
    # Always SHORT
    le_short = np.full(n, -1.0)
    sp_short = 0.0
    acc_short = (y_class == 2).mean()
    ret_short = np.mean(-y)
    res = pd.DataFrame([
        {"model": "TCN", "spearman": sp_tcn, "accuracy": acc_tcn, "mean_return": ret_tcn, "count": n},
        {"model": "Random", "spearman": sp_rnd, "accuracy": acc_rnd, "mean_return": ret_rnd, "count": n},
        {"model": "Always_FLAT", "spearman": sp_flat, "accuracy": acc_flat, "mean_return": ret_flat, "count": n},
        {"model": "Always_LONG", "spearman": sp_long, "accuracy": acc_long, "mean_return": ret_long, "count": n},
        {"model": "Always_SHORT", "spearman": sp_short, "accuracy": acc_short, "mean_return": ret_short, "count": n},
    ])
    res.to_csv(out_dir / "predictability_test.csv", index=False)
    print(f"[MU] Wrote predictability_test.csv", flush=True)


def run_summary_report(out_dir: Path) -> None:
    """[8] model_understanding_summary.md answering the 8 questions."""
    lines = [
        "# Model Understanding & Predictability Diagnostics: 요약",
        "",
        "## 데이터 및 모델",
        f"- 심볼: {SYMBOL}, 타임프레임: {TIMEFRAME}, 기간: {DAYS}d",
        f"- 모델: {MODEL_ID}, horizon={HORIZON}bar, threshold=±{POS_THRESHOLD}",
        "",
        "## 1. 어떤 feature가 future_return과 가장 상관이 높은가",
        ""
    ]
    fv_path = out_dir / "feature_vs_future_return.csv"
    if fv_path.exists():
        fv = pd.read_csv(fv_path)
        if not fv.empty:
            top = fv.nlargest(5, "abs_spearman")
            for _, r in top.iterrows():
                lines.append(f"- {r['feature_name']}: Spearman={r['spearman_corr']:.4f}, |Spearman|={r['abs_spearman']:.4f}")
            max_abs = fv["abs_spearman"].max()
            lines.append(f"- 대부분 |Spearman| < 0.02 이면 feature 정보가 부족할 가능성 있음. 최대 |Spearman| = {max_abs:.4f}")
        else:
            lines.append("- (데이터 없음)")
    else:
        lines.append("- (파일 없음)")
    lines.extend(["", "## 2. 모델이 실제로 사용하는 feature는 무엇인가", ""])
    pi_path = out_dir / "permutation_importance.csv"
    if pi_path.exists():
        pi = pd.read_csv(pi_path)
        if not pi.empty:
            top_pi = pi.nlargest(5, "importance_score")
            for _, r in top_pi.iterrows():
                lines.append(f"- {r['feature_name']}: importance={r['importance_score']:.4f}")
        else:
            lines.append("- (데이터 없음)")
    else:
        lines.append("- (파일 없음)")
    lines.extend(["", "## 3. event feature는 signal인가 noise인가", ""])
    ab_path = out_dir / "event_feature_ablation.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        if len(ab) >= 2:
            a = ab[ab["experiment"] == "A_full"].iloc[0]
            b = ab[ab["experiment"] == "B_no_event"].iloc[0]
            lines.append(f"- Full: p_long_spearman={a['p_long_spearman']:.4f}, long_edge_spearman={a['long_edge_spearman']:.4f}")
            lines.append(f"- No event: p_long_spearman={b['p_long_spearman']:.4f}, long_edge_spearman={b['long_edge_spearman']:.4f}")
            lines.append("- 두 값이 비슷하면 event는 signal이 아니고 noise에 가깝다고 해석 가능.")
        else:
            lines.append("- (행 부족)")
    else:
        lines.append("- (파일 없음)")
    lines.extend(["", "## 4. 모델 prediction confidence 분포는 어떤가", ""])
    conf_path = out_dir / "prediction_confidence_summary.csv"
    if conf_path.exists():
        conf = pd.read_csv(conf_path)
        for _, r in conf.iterrows():
            lines.append(f"- {r['metric']}: mean={r['mean']:.4f}, std={r['std']:.4f}, p50={r['p50']:.4f}, p95={r['p95']:.4f}")
        lines.append("- max_proba ≈ 0.33~0.4 이면 모델이 거의 항상 불확실한 상태.")
    else:
        lines.append("- (파일 없음)")
    lines.extend(["", "## 5. 특정 regime에서만 모델이 작동하는가", ""])
    reg_path = out_dir / "regime_alignment.csv"
    if reg_path.exists():
        reg = pd.read_csv(reg_path)
        if not reg.empty:
            for _, r in reg.iterrows():
                lines.append(f"- {r['regime_name']}: count={r['count']}, p_long_spearman={r['p_long_spearman']:.4f}")
        else:
            lines.append("- (데이터 없음)")
    else:
        lines.append("- (파일 없음)")
    lines.extend(["", "## 6. probability calibration은 정상인가", ""])
    cal_path = out_dir / "probability_calibration.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        if not cal.empty:
            lines.append("- p_long decile별 future_return_mean, long_frequency 확인. 상위 decile에서 return이 더 높고 long 비율이 높으면 정렬됨.")
        else:
            lines.append("- (데이터 없음)")
    else:
        lines.append("- (파일 없음)")
    lines.extend(["", "## 7. TCN 모델이 random predictor보다 실제로 나은가", ""])
    pred_path = out_dir / "predictability_test.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        tcn = pred[pred["model"] == "TCN"]
        rnd = pred[pred["model"] == "Random"]
        if len(tcn) and len(rnd):
            lines.append(f"- TCN: spearman={tcn['spearman'].iloc[0]:.4f}, accuracy={tcn['accuracy'].iloc[0]:.4f}, mean_return={tcn['mean_return'].iloc[0]:.6f}")
            lines.append(f"- Random: spearman={rnd['spearman'].iloc[0]:.4f}, accuracy={rnd['accuracy'].iloc[0]:.4f}, mean_return={rnd['mean_return'].iloc[0]:.6f}")
            lines.append("- TCN ≈ Random 이면 문제는 모델이 아니라 예측 불가능성일 가능성.")
        else:
            lines.append("- (행 없음)")
    else:
        lines.append("- (파일 없음)")
    lines.extend(["", "## 8. 현재 feature set으로 예측 가능한 문제인가", ""])
    lines.append("- 위 1~7 종합: feature 상관이 전반적으로 약하고, TCN이 random과 유의미하게 차이 나지 않으면, 현재 feature set으로는 5m horizon 15bar 예측이 사실상 불가능한 문제로 해석할 수 있음.")
    lines.append("")
    (out_dir / "model_understanding_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[MU] Wrote model_understanding_summary.md", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Model Understanding & Predictability Diagnostics")
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

    print("[MU] Loading data and running baseline inference ...", flush=True)
    raw, features, feature_at_pred, pl, ps, model, config = load_data_and_run_baseline_inference(
        SYMBOL, TIMEFRAME, args.days, END_DATE, model_path, HORIZON, WINDOW_SIZE,
    )
    print(f"[MU] Aligned rows: {len(raw)}", flush=True)

    run_feature_vs_future_return(raw, feature_at_pred, out_dir)
    if not args.skip_permutation:
        run_permutation_importance(raw, features, feature_at_pred, model, SYMBOL, TIMEFRAME, WINDOW_SIZE, HORIZON, out_dir)
    else:
        pd.DataFrame(columns=["feature_name", "importance_score", "rank"]).to_csv(out_dir / "permutation_importance.csv", index=False)
        print("[MU] Skip permutation_importance (--skip-permutation)", flush=True)
    run_event_feature_ablation(raw, features, model, SYMBOL, TIMEFRAME, WINDOW_SIZE, HORIZON, out_dir)
    run_prediction_confidence_summary(raw, out_dir)
    run_regime_analysis(raw, feature_at_pred, out_dir)
    run_probability_calibration(raw, out_dir)
    run_predictability_test(raw, out_dir)
    run_summary_report(out_dir)
    print("[MU] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
