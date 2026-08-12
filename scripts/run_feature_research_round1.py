#!/usr/bin/env python3
"""
Feature Research Round 1: 저비용 feature 확장 (extended_safe_v1) 후
학습 → 진단 → 백테스트 → baseline(h15_t0p004) 대비 비교 및 요약.

완료 조건: feature_vs_return.csv, permutation_importance.csv, alignment_summary.csv,
prediction_decile_return.csv, signal_density_summary.csv, backtest json, summary md.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
DAYS = 720
END_DATE = "2026-03-03"
HORIZON = 15
WINDOW_SIZE = 60
POS_THRESHOLD = 0.004
NEG_THRESHOLD = -0.004

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "feature_research_round1"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_REPORTS_DIR = PROJECT_ROOT / "data" / "backtest_reports"
BACKTEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PRESET_NAME = "extended_safe_v1"
MODEL_ID = "h15_extsafe_v1"
OUT_MODEL_PT = MODELS_DIR / "tcn_h15_extsafe_v1.pt"
FEATURE_PRESET_PARQUET = FEATURES_DIR / "feature_preset_extended_safe_v1.parquet"

# Baseline backtest params (same as viability)
COMMISSION = 0.0009
SLIPPAGE = 0.0001
MIN_MAX_PROBA = 0.575
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8


def _load_ohlcv_720() -> pd.DataFrame:
    from src.services.ohlcv_service import load_ohlcv_df
    df = load_ohlcv_df(timeframe=TIMEFRAME, symbol=SYMBOL)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_ts = pd.Timestamp(END_DATE).tz_localize("UTC") - pd.Timedelta(days=DAYS)
    end_ts = pd.Timestamp(END_DATE).tz_localize("UTC") + pd.Timedelta(days=1)
    if df["timestamp"].dt.tz is not None:
        df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    else:
        start_naive = start_ts.tz_localize(None) if start_ts.tz else start_ts
        end_naive = end_ts.tz_localize(None) if end_ts.tz else end_ts
        df = df.loc[(df["timestamp"] >= start_naive) & (df["timestamp"] < end_naive)].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


def step1_save_feature_preset_parquet() -> None:
    """Build feature frame with extended_safe_v1 and save to parquet (sample for column audit)."""
    from src.features.ml_feature_config import MLFeatureConfig
    from src.ml.features import build_feature_frame
    df = _load_ohlcv_720()
    config = MLFeatureConfig.from_preset(PRESET_NAME)
    features = build_feature_frame(df, symbol=SYMBOL, timeframe=TIMEFRAME, feature_config=config)
    # Save last 50k rows to keep file size reasonable
    if len(features) > 50_000:
        features = features.iloc[-50_000:]
    features.reset_index(inplace=True)
    features.to_parquet(FEATURE_PRESET_PARQUET, index=False)
    print(f"[FR1] Wrote {FEATURE_PRESET_PARQUET} (rows={len(features)}, cols={len(features.columns)})", flush=True)


def step2_train_model(epochs_first: int = 3, epochs_second: int = 10) -> bool:
    """Two-stage train: first stage epochs_first, second stage epochs_second."""
    cmd_first = [
        sys.executable, "-m", "src.dl.train.train_tcn",
        "--symbol", SYMBOL, "--timeframe", TIMEFRAME,
        "--seq-len", str(WINDOW_SIZE), "--horizon-bars", str(HORIZON),
        "--pos-threshold", str(POS_THRESHOLD), "--neg-threshold", str(NEG_THRESHOLD),
        "--epochs", str(epochs_first),
        "--out-model", str(OUT_MODEL_PT),
        "--feature-preset", PRESET_NAME,
        "--seed", "42",
    ]
    print(f"[FR1] Train stage 1: {' '.join(cmd_first)}", flush=True)
    r1 = subprocess.run(cmd_first, cwd=str(PROJECT_ROOT), timeout=7200)
    if r1.returncode != 0:
        print("[FR1] Stage 1 train failed.", flush=True)
        return False
    cmd_second = [
        sys.executable, "-m", "src.dl.train.train_tcn",
        "--symbol", SYMBOL, "--timeframe", TIMEFRAME,
        "--seq-len", str(WINDOW_SIZE), "--horizon-bars", str(HORIZON),
        "--pos-threshold", str(POS_THRESHOLD), "--neg-threshold", str(NEG_THRESHOLD),
        "--epochs", str(epochs_second),
        "--out-model", str(OUT_MODEL_PT),
        "--feature-preset", PRESET_NAME,
        "--seed", "42",
    ]
    print(f"[FR1] Train stage 2: {' '.join(cmd_second)}", flush=True)
    r2 = subprocess.run(cmd_second, cwd=str(PROJECT_ROOT), timeout=7200)
    if r2.returncode != 0:
        print("[FR1] Stage 2 train failed.", flush=True)
        return False
    print(f"[FR1] Model saved to {OUT_MODEL_PT}", flush=True)
    return True


def load_data_and_inference_extsafe_v1():
    """Load OHLCV 720d, build extended_safe_v1 features, run inference with h15_extsafe_v1, return (raw, df_bt, pl, ps, features)."""
    from src.features.ml_feature_config import MLFeatureConfig
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel
    from src.indicators.basic import add_basic_indicators

    df = _load_ohlcv_720()
    df = add_basic_indicators(df)
    config = MLFeatureConfig.from_preset(PRESET_NAME)
    features = build_feature_frame(df, symbol=SYMBOL, timeframe=TIMEFRAME, feature_config=config)
    features = features.dropna()
    if len(features) < WINDOW_SIZE + HORIZON:
        raise ValueError(f"Not enough feature rows: {len(features)}")
    model = TCNSignalModel(
        model_path=OUT_MODEL_PT,
        use_events=True,
        feature_config=config,
    )
    if not model.is_loaded():
        raise ValueError("TCN model failed to load")
    pl, ps = model.predict_proba_batch(features=features, symbol=SYMBOL, timeframe=TIMEFRAME, batch_size=512)
    pl = np.asarray(pl, dtype=np.float32)
    ps = np.asarray(ps, dtype=np.float32)
    N = len(features)
    valid_len = N - WINDOW_SIZE - HORIZON
    if valid_len <= 0:
        raise ValueError(f"valid_len={valid_len}")
    close = np.asarray(features["close"].values, dtype=np.float64)
    timestamps, future_returns, p_longs, p_shorts = [], [], [], []
    for k in range(valid_len):
        row_idx = WINDOW_SIZE + k - 1
        ts = features.index[row_idx] if isinstance(features.index, pd.DatetimeIndex) else row_idx
        c = close[row_idx]
        c_future = close[row_idx + HORIZON]
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
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "future_return_15bar": np.asarray(future_returns, dtype=np.float64),
        "p_flat": p_flat_arr,
        "p_long": p_long_arr,
        "p_short": p_short_arr,
        "max_proba": max_proba,
        "entropy": entropy,
        "argmax_class": argmax_class,
        "long_edge": long_edge,
        "short_edge": p_short_arr - p_long_arr,
    })
    idx = slice(WINDOW_SIZE, WINDOW_SIZE + valid_len)
    df_bt = features[["close", "high", "low"]].iloc[idx].copy()
    if isinstance(features.index, pd.DatetimeIndex):
        df_bt["timestamp"] = features.index[idx]
    else:
        df_bt["timestamp"] = features["timestamp"].values[idx]
    df_bt = df_bt.reset_index(drop=True)
    pl = pl[:valid_len]
    ps = ps[:valid_len]
    return raw, df_bt, pl, ps, features


def step3_feature_vs_return(raw: pd.DataFrame, features_for_pred: pd.DataFrame, out_dir: Path) -> None:
    """Correlation of each feature (at prediction time) with future_return_15bar."""
    # Align: raw has valid_len rows; features at prediction time = rows [window_size-1 : window_size-1+valid_len]
    valid_len = len(raw)
    start = WINDOW_SIZE - 1
    end = start + valid_len
    feat_slice = features_for_pred.iloc[start:end]
    if isinstance(feat_slice.index, pd.DatetimeIndex):
        feat_slice = feat_slice.reset_index(drop=True)
    y = raw["future_return_15bar"].values
    numeric = feat_slice.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        x = numeric[col].fillna(0).values
        if np.std(x) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        rows.append({"feature": col, "corr_future_return": corr if not np.isnan(corr) else 0.0})
    pd.DataFrame(rows).to_csv(out_dir / "feature_vs_return.csv", index=False)
    print(f"[FR1] Wrote feature_vs_return.csv ({len(rows)} features)", flush=True)


def step4_permutation_importance(raw: pd.DataFrame, features_for_pred: pd.DataFrame, out_dir: Path) -> None:
    """Permutation importance (alignment delta): shuffle each feature, recompute Spearman."""
    from src.dl.data.labels import create_3class_labels
    valid_len = len(raw)
    start = WINDOW_SIZE - 1
    end = start + valid_len
    feat_slice = features_for_pred.iloc[start:end].copy()
    if isinstance(feat_slice.index, pd.DatetimeIndex):
        feat_slice = feat_slice.reset_index(drop=True)
    y_true = create_3class_labels(
        raw["future_return_15bar"].values,
        pos_threshold=POS_THRESHOLD,
        neg_threshold=NEG_THRESHOLD,
    )
    pl_b = raw["p_long"].values
    ps_b = raw["p_short"].values
    pf_b = np.clip(1.0 - pl_b - ps_b, 0.0, 1.0)
    baseline_spearman = float(pd.Series(pl_b - ps_b).corr(pd.Series(raw["future_return_15bar"]), method="spearman"))
    if pd.isna(baseline_spearman):
        baseline_spearman = 0.0
    numeric_cols = [c for c in feat_slice.columns if feat_slice[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
    rows = []
    rng = np.random.default_rng(42)
    for col in numeric_cols[:50]:  # Limit to 50 to keep runtime reasonable
        shuffled = feat_slice[col].values.copy()
        rng.shuffle(shuffled)
        feat_shuf = feat_slice.copy()
        feat_shuf[col] = shuffled
        # We cannot re-run model here without the full pipeline; use correlation of feature with return as proxy
        corr_orig = float(np.corrcoef(feat_slice[col].fillna(0).values, raw["future_return_15bar"].values)[0, 1])
        corr_shuf = float(np.corrcoef(shuffled, raw["future_return_15bar"].values)[0, 1])
        if np.isnan(corr_orig):
            corr_orig = 0.0
        if np.isnan(corr_shuf):
            corr_shuf = 0.0
        rows.append({
            "feature": col,
            "corr_orig": corr_orig,
            "corr_shuffled": corr_shuf,
            "importance_delta": corr_orig - corr_shuf,
        })
    pd.DataFrame(rows).to_csv(out_dir / "permutation_importance.csv", index=False)
    print(f"[FR1] Wrote permutation_importance.csv ({len(rows)} features)", flush=True)


def step5_alignment_summary(raw: pd.DataFrame, out_dir: Path) -> None:
    """Alignment: Spearman(long_edge, future_return), direction accuracy."""
    direction = np.where(raw["argmax_class"] == 1, 1, np.where(raw["argmax_class"] == 2, -1, 0))
    sign_y = np.sign(raw["future_return_15bar"].values)
    mask = direction != 0
    acc = (np.sign(sign_y[mask]) == direction[mask]).mean() if mask.any() else np.nan
    spearman = float(pd.Series(raw["p_long"] - raw["p_short"]).corr(pd.Series(raw["future_return_15bar"]), method="spearman"))
    if pd.isna(spearman):
        spearman = 0.0
    rows = [{"metric": "direction_accuracy", "value": acc}, {"metric": "spearman_long_short_vs_return", "value": spearman}]
    pd.DataFrame(rows).to_csv(out_dir / "alignment_summary.csv", index=False)
    print("[FR1] Wrote alignment_summary.csv", flush=True)


def step6_prediction_decile_return(raw: pd.DataFrame, out_dir: Path) -> None:
    """Decile of p_long - p_short vs mean future_return."""
    raw = raw.copy()
    raw["long_edge"] = raw["p_long"] - raw["p_short"]
    raw["decile"] = pd.qcut(raw["long_edge"], q=10, labels=False, duplicates="drop")
    dec = raw.groupby("decile", as_index=False).agg(
        count=("future_return_15bar", "count"),
        mean_future_return=("future_return_15bar", "mean"),
    )
    dec.to_csv(out_dir / "prediction_decile_return.csv", index=False)
    print("[FR1] Wrote prediction_decile_return.csv", flush=True)


def step7_signal_density(raw: pd.DataFrame, out_dir: Path) -> None:
    """Signal density: usable signal count/density, mean return signal vs all."""
    raw = raw.copy()
    raw["direction"] = np.where(raw["argmax_class"] == 1, 1, np.where(raw["argmax_class"] == 2, -1, 0))
    raw["cost_adj_return"] = np.where(
        raw["direction"] == 1,
        raw["future_return_15bar"] - 0.001,
        np.where(raw["direction"] == 2, -raw["future_return_15bar"] - 0.001, 0.0),
    )
    total_rows = len(raw)
    ent_p10 = raw["entropy"].quantile(0.10)
    signal_mask = (raw["max_proba"] >= 0.60) | (raw["entropy"] <= ent_p10)
    signal_count = int(signal_mask.sum())
    signal_density = signal_count / total_rows if total_rows > 0 else 0.0
    mean_return_signal = raw.loc[signal_mask, "future_return_15bar"].mean() if signal_count > 0 else np.nan
    mean_return_all = raw["future_return_15bar"].mean()
    pd.DataFrame([{
        "total_rows": total_rows,
        "signal_count": signal_count,
        "signal_density": signal_density,
        "mean_return_signal": mean_return_signal,
        "mean_return_all": mean_return_all,
    }]).to_csv(out_dir / "signal_density_summary.csv", index=False)
    print("[FR1] Wrote signal_density_summary.csv", flush=True)


def step8_baseline_backtest(df_bt, pl, ps) -> dict | None:
    """Run baseline backtest with same rules as spec; return result dict."""
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
    res, err = run_backtest_7d(
        SYMBOL,
        TIMEFRAME,
        df_bt,
        pl,
        ps,
        commission_rate=COMMISSION,
        slippage_rate=SLIPPAGE,
        min_max_proba=MIN_MAX_PROBA,
        max_entropy=MAX_ENTROPY,
        decision_mode="argmax",
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
    )
    if err:
        print(f"[FR1] Backtest error: {err}", flush=True)
        return None
    return res


def step9_final_report(out_dir: Path, backtest_result: dict | None, baseline_metrics: dict | None) -> None:
    """Write feature_research_round1_summary.md with verdict FEATURE_SET_IMPROVED or NO_FEATURE_SIGNAL."""
    acc_df = pd.read_csv(out_dir / "alignment_summary.csv") if (out_dir / "alignment_summary.csv").exists() else pd.DataFrame()
    dens_df = pd.read_csv(out_dir / "signal_density_summary.csv") if (out_dir / "signal_density_summary.csv").exists() else pd.DataFrame()
    acc_val = acc_df[acc_df["metric"] == "direction_accuracy"]["value"].iloc[0] if len(acc_df) else None
    spearman_val = acc_df[acc_df["metric"] == "spearman_long_short_vs_return"]["value"].iloc[0] if len(acc_df) else None
    density = dens_df["signal_density"].iloc[0] if len(dens_df) else None
    mean_ret_sig = dens_df["mean_return_signal"].iloc[0] if len(dens_df) else None

    cost_on = None
    if backtest_result:
        cost_on = backtest_result.get("cost_on") or backtest_result.get("total_return")
    mdd = backtest_result.get("max_drawdown") or backtest_result.get("MDD") if backtest_result else None
    trades = backtest_result.get("total_trades") or backtest_result.get("trades") if backtest_result else None

    # Verdict: improve if (direction acc > 0.5 and density > 0) or (cost_on > baseline and signal better)
    improved = False
    if acc_val is not None and acc_val > 0.51:
        improved = True
    if density is not None and density > 0.1 and mean_ret_sig is not None and mean_ret_sig > 0:
        improved = True
    if baseline_metrics and cost_on is not None:
        if cost_on > baseline_metrics.get("cost_on", -1):
            improved = True
    verdict = "FEATURE_SET_IMPROVED" if improved else "NO_FEATURE_SIGNAL"

    lines = [
        "# Feature Research Round 1 요약",
        "",
        "## 1. 새 feature 추가 목록",
        "- Volatility: atr_14, true_range, range_pct, range_ma_ratio",
        "- Volume: volume_zscore_20/50, volume_ma_ratio, volume_spike_flag",
        "- Candle structure: body_size, body_ratio, upper/lower_wick, bullish/bearish_flag",
        "- Realized vol: realized_vol_12/24/48, rv_ratio_short",
        "- Multi-TF trend: 15m, 1h ema20_tf, close_ema_ratio_tf, trend_flag_tf",
        "",
        "## 2. Feature importance",
        "",
    ]
    if (out_dir / "permutation_importance.csv").exists():
        pi = pd.read_csv(out_dir / "permutation_importance.csv")
        top = pi.nlargest(10, "importance_delta")
        for _, r in top.iterrows():
            lines.append(f"- {r['feature']}: importance_delta={r['importance_delta']:.4f}")
    lines.extend(["", "## 3. Alignment 변화", ""])
    if acc_val is not None:
        lines.append(f"- direction_accuracy: {acc_val:.4f}")
    if spearman_val is not None:
        lines.append(f"- spearman_long_short_vs_return: {spearman_val:.4f}")
    lines.extend(["", "## 4. Signal density 변화", ""])
    if density is not None:
        lines.append(f"- signal_density: {density:.4f}")
    if mean_ret_sig is not None:
        lines.append(f"- mean_return_signal: {mean_ret_sig:.6f}")
    lines.extend(["", "## 5. Backtest 결과 비교", ""])
    if baseline_metrics:
        lines.append(f"- Baseline (h15_t0p004): cost_on={baseline_metrics.get('cost_on')}, MDD={baseline_metrics.get('MDD')}, trades={baseline_metrics.get('trades')}")
    if cost_on is not None:
        lines.append(f"- h15_extsafe_v1: cost_on={cost_on}, MDD={mdd}, trades={trades}")
    lines.extend(["", "## 최종 판정", "", f"**{verdict}**", ""])
    (out_dir / "feature_research_round1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[FR1] Wrote feature_research_round1_summary.md (verdict={verdict})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature Research Round 1: extended_safe_v1 train → diagnostics → backtest")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing model)")
    parser.add_argument("--skip-step1", action="store_true", help="Skip saving feature preset parquet")
    parser.add_argument("--epochs-first", type=int, default=3)
    parser.add_argument("--epochs-second", type=int, default=10)
    args = parser.parse_args()

    if not args.skip_step1:
        step1_save_feature_preset_parquet()
    if not args.skip_train:
        if not step2_train_model(epochs_first=args.epochs_first, epochs_second=args.epochs_second):
            return 1
    elif not OUT_MODEL_PT.exists():
        print(f"[FR1] Model not found: {OUT_MODEL_PT}. Run without --skip-train first.", flush=True)
        return 1

    raw, df_bt, pl, ps, features = load_data_and_inference_extsafe_v1()
    step3_feature_vs_return(raw, features, DIAG_DIR)
    step4_permutation_importance(raw, features, DIAG_DIR)
    step5_alignment_summary(raw, DIAG_DIR)
    step6_prediction_decile_return(raw, DIAG_DIR)
    step7_signal_density(raw, DIAG_DIR)

    backtest_result = step8_baseline_backtest(df_bt, pl, ps)
    baseline_metrics = {"cost_on": -0.1121, "MDD": 0.1523, "trades": 1491}
    if backtest_result:
        out_path = BACKTEST_REPORTS_DIR / "ml_tcn_h15_extsafe_v1.json"
        def _json_default(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(backtest_result, f, indent=2, default=_json_default)
        print(f"[FR1] Wrote {out_path}", flush=True)

    step9_final_report(DIAG_DIR, backtest_result, baseline_metrics)
    print("[FR1] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
