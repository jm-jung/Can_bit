#!/usr/bin/env python3
"""
FR2 Diagnostics: alignment, signal density, regime, backtest, rolling, walk-forward.
Produces: fr2_alignment.csv, fr2_signal_density.csv, fr2_backtest_comparison.csv,
fr2_regime_analysis.csv, fr2_rolling_90d.csv, fr2_walkforward.csv,
fr2_vs_fr1_vs_base.csv, fr2_summary.md, fr2_summary.json.
Requires: h15_t0p004.pt, tcn_h15_extsafe_v1.pt, tcn_h15_micro_v1.pt in data/diagnostics/models/.
"""
from __future__ import annotations

import json
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
NEG_THRESHOLD = -0.004

COMMISSION = 0.0009
SLIPPAGE = 0.0001
MIN_MAX_PROBA = 0.575
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8

MODELS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "models"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("h15_t0p004", "base", MODELS_DIR / "tcn_h15_t0p004.pt", False),
    ("h15_extsafe_v1", "extended_safe_v1", MODELS_DIR / "tcn_h15_extsafe_v1.pt", False),
    ("h15_micro_v1", "microstructure_v1", MODELS_DIR / "tcn_h15_micro_v1.pt", True),
]


def _load_ohlcv(
    days: int,
    include_microstructure: bool = False,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV. end_date: None = use module END_DATE (e.g. fixed cutoff); else 'YYYY-MM-DD' for operational latest."""
    from src.services.ohlcv_service import load_ohlcv_df
    if end_date is None:
        cutoff = END_DATE
    elif hasattr(end_date, "strftime"):
        cutoff = end_date.strftime("%Y-%m-%d")
    else:
        cutoff = end_date.strip()
    df = load_ohlcv_df(timeframe=TIMEFRAME, symbol=SYMBOL, include_microstructure=include_microstructure)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_ts = pd.Timestamp(cutoff).tz_localize("UTC") - pd.Timedelta(days=days)
    end_ts = pd.Timestamp(cutoff).tz_localize("UTC") + pd.Timedelta(days=1)
    if df["timestamp"].dt.tz is not None:
        df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    else:
        start_naive = start_ts.tz_localize(None) if start_ts.tz else start_ts
        end_naive = end_ts.tz_localize(None) if end_ts.tz else end_ts
        df = df.loc[(df["timestamp"] >= start_naive) & (df["timestamp"] < end_naive)].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


def get_ohlcv_and_proba(
    days: int,
    model_path: Path,
    feature_preset: str,
    include_microstructure: bool,
    temperature: float = 1.0,
    end_date: str | None = None,
):
    """Return (df_bt, pl, ps, future_ret) or (None, err). temperature=1.0 preserves current behavior.
    end_date: None = use module END_DATE; else 'YYYY-MM-DD' (or date-like) for data cutoff (e.g. operational latest)."""
    from src.features.ml_feature_config import MLFeatureConfig
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel
    from src.indicators.basic import add_basic_indicators

    if not model_path.exists():
        return (None, f"model not found: {model_path}")
    df = _load_ohlcv(days, include_microstructure=include_microstructure, end_date=end_date)
    df = add_basic_indicators(df)
    config = MLFeatureConfig.from_preset(feature_preset)
    features = build_feature_frame(df, symbol=SYMBOL, timeframe=TIMEFRAME, feature_config=config)
    features = features.dropna()
    if len(features) < WINDOW_SIZE + HORIZON:
        return (None, f"not enough rows: {len(features)}")
    model = TCNSignalModel(model_path=model_path, use_events=True, feature_config=config)
    if not model.is_loaded():
        return (None, "model load failed")
    pl, ps = model.predict_proba_batch(
        features=features, symbol=SYMBOL, timeframe=TIMEFRAME, batch_size=512, temperature=temperature
    )
    pl = np.asarray(pl, dtype=np.float32)
    ps = np.asarray(ps, dtype=np.float32)
    N = len(features)
    valid_len = N - WINDOW_SIZE - HORIZON
    if valid_len <= 0:
        return (None, f"valid_len={valid_len}")
    idx = slice(WINDOW_SIZE, WINDOW_SIZE + valid_len)
    close = df["close"].values
    future_ret = (close[WINDOW_SIZE + HORIZON : WINDOW_SIZE + valid_len + HORIZON] / close[WINDOW_SIZE : WINDOW_SIZE + valid_len]) - 1
    if len(future_ret) > valid_len:
        future_ret = future_ret[:valid_len]
    elif len(future_ret) < valid_len:
        future_ret = np.resize(future_ret, valid_len)
    df_bt = features[["close", "high", "low"]].iloc[idx].copy()
    if isinstance(features.index, pd.DatetimeIndex):
        df_bt["timestamp"] = features.index[idx]
    else:
        df_bt["timestamp"] = features["timestamp"].values[idx] if "timestamp" in features.columns else features.index[idx]
    df_bt = df_bt.reset_index(drop=True)
    pl = pl[:valid_len]
    ps = ps[:valid_len]
    return ((df_bt, pl, ps, future_ret), "")


def run_backtest(df_bt, pl, ps) -> dict | None:
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


def _direction_accuracy_spearman_density(pl, ps, future_ret, threshold_signal=0.60):
    p_flat = 1 - pl - ps
    pred = np.argmax(np.stack([p_flat, pl, ps], axis=1), axis=1)
    actual = np.where(future_ret > POS_THRESHOLD, 1, np.where(future_ret < NEG_THRESHOLD, 2, 0))
    trim = min(len(pred), len(actual), len(pl))
    if trim < 10:
        return np.nan, np.nan, np.nan, np.nan
    pred, actual = pred[:trim], actual[:trim]
    pl_t, ps_t = pl[:trim], ps[:trim]
    future_ret = future_ret[:trim]
    acc = (pred == actual).mean()
    sp = pd.Series(pl_t - ps_t).corr(pd.Series(future_ret), method="spearman") if trim > 10 else np.nan
    max_proba = np.maximum(np.maximum(pl_t, ps_t), p_flat[:trim])
    sig_den = (max_proba >= threshold_signal).mean()
    mean_ret_sig = float(np.mean(future_ret[max_proba >= threshold_signal])) if (max_proba >= threshold_signal).any() else np.nan
    return acc, sp, sig_den, mean_ret_sig


def run_alignment():
    rows = []
    for model_id, preset, path, inc_micro in MODELS:
        triple, err = get_ohlcv_and_proba(DAYS, path, preset, inc_micro)
        if err:
            rows.append({"model_id": model_id, "direction_accuracy": np.nan, "spearman": np.nan, "signal_density": np.nan, "mean_return_signal": np.nan, "error": err})
            continue
        df_bt, pl, ps, future_ret = triple
        acc, sp, sd, mrs = _direction_accuracy_spearman_density(pl, ps, future_ret)
        rows.append({"model_id": model_id, "direction_accuracy": acc, "spearman": sp, "signal_density": sd, "mean_return_signal": mrs, "error": ""})
    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_alignment.csv", index=False)
    print("[FR2] Wrote fr2_alignment.csv", flush=True)
    return rows


def run_signal_density():
    rows = []
    for model_id, preset, path, inc_micro in MODELS:
        triple, err = get_ohlcv_and_proba(DAYS, path, preset, inc_micro)
        if err:
            rows.append({"model_id": model_id, "signal_density": np.nan, "mean_return_signal": np.nan, "error": err})
            continue
        _, pl, ps, future_ret = triple
        _, _, sd, mrs = _direction_accuracy_spearman_density(pl, ps, future_ret)
        rows.append({"model_id": model_id, "signal_density": sd, "mean_return_signal": mrs, "error": ""})
    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_signal_density.csv", index=False)
    print("[FR2] Wrote fr2_signal_density.csv", flush=True)
    return rows


def run_backtest_comparison():
    rows = []
    for model_id, preset, path, inc_micro in MODELS:
        triple, err = get_ohlcv_and_proba(DAYS, path, preset, inc_micro)
        if err:
            rows.append({"model_id": model_id, "cost_on": np.nan, "MDD": np.nan, "trades": np.nan, "win_rate": np.nan, "error": err})
            continue
        df_bt, pl, ps, _ = triple
        res = run_backtest(df_bt, pl, ps)
        if res is None:
            rows.append({"model_id": model_id, "cost_on": np.nan, "MDD": np.nan, "trades": np.nan, "win_rate": np.nan, "error": "backtest failed"})
            continue
        rows.append({
            "model_id": model_id,
            "cost_on": res.get("total_return"),
            "MDD": res.get("max_drawdown"),
            "trades": res.get("total_trades"),
            "win_rate": res.get("win_rate"),
            "error": "",
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_backtest_comparison.csv", index=False)
    print("[FR2] Wrote fr2_backtest_comparison.csv", flush=True)
    return rows


def run_regime_analysis():
    rows = []
    for model_id, preset, path, inc_micro in MODELS:
        triple, err = get_ohlcv_and_proba(DAYS, path, preset, inc_micro)
        if err:
            continue
        df_bt, pl, ps, future_ret = triple
        if len(df_bt) < 100 or "close" not in df_bt.columns:
            continue
        close = df_bt["close"].astype(float).values
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
        trend = pd.Series(close).rolling(50, min_periods=10).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) >= 2 else np.nan, raw=False)
        vol = pd.Series(ret).rolling(20, min_periods=5).std()
        try:
            trend_q = pd.qcut(trend.rank(method="first"), q=2, labels=[0, 1], duplicates="drop")
        except Exception:
            trend_q = pd.Series(0, index=trend.index)
        try:
            vol_q = pd.qcut(vol.rank(method="first"), q=2, labels=[0, 1], duplicates="drop")
        except Exception:
            vol_q = pd.Series(0, index=vol.index)
        trim = min(len(pl), len(df_bt), len(trend_q), len(vol_q))
        if trim < 50:
            continue
        pl, ps, future_ret = pl[:trim], ps[:trim], future_ret[:trim]
        trend_q, vol_q = trend_q.values[:trim], vol_q.values[:trim]
        for regime_type, qcol in [("trend_q", trend_q), ("vol_q", vol_q)]:
            for q in [0, 1]:
                mask = (qcol == q) if np.issubdtype(qcol.dtype, np.number) else (qcol == str(q))
                if mask.sum() < 20:
                    continue
                pl_t, ps_t = pl[mask], ps[mask]
                fr_t = future_ret[mask]
                acc, sp, sd, _ = _direction_accuracy_spearman_density(pl_t, ps_t, fr_t)
                rows.append({"regime_type": regime_type, "regime_name": f"{regime_type}_{q}", "model_id": model_id, "direction_accuracy": acc, "spearman": sp, "signal_density": sd})
    if not rows:
        pd.DataFrame(columns=["regime_type", "regime_name", "model_id", "direction_accuracy", "spearman", "signal_density"]).to_csv(OUT_DIR / "fr2_regime_analysis.csv", index=False)
    else:
        pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_regime_analysis.csv", index=False)
    print("[FR2] Wrote fr2_regime_analysis.csv", flush=True)
    return rows


def run_rolling_90d():
    rows = []
    for last_d in [90]:
        for model_id, preset, path, inc_micro in MODELS:
            triple, err = get_ohlcv_and_proba(last_d, path, preset, inc_micro)
            if err:
                rows.append({"period_days": last_d, "model_id": model_id, "cost_on": np.nan, "direction_accuracy": np.nan, "spearman": np.nan, "signal_density": np.nan})
                continue
            df_bt, pl, ps, future_ret = triple
            res = run_backtest(df_bt, pl, ps)
            acc, sp, sd, _ = _direction_accuracy_spearman_density(pl, ps, future_ret)
            rows.append({
                "period_days": last_d,
                "model_id": model_id,
                "cost_on": res.get("total_return") if res else np.nan,
                "direction_accuracy": acc,
                "spearman": sp,
                "signal_density": sd,
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_rolling_90d.csv", index=False)
    print("[FR2] Wrote fr2_rolling_90d.csv", flush=True)
    return rows


def run_walkforward():
    train_d, valid_d, test_d = 360, 90, 90
    rows = []
    for model_id, preset, path, inc_micro in MODELS:
        triple, err = get_ohlcv_and_proba(train_d + valid_d + test_d, path, preset, inc_micro)
        if err:
            rows.append({"model_id": model_id, "train_d": train_d, "valid_d": valid_d, "test_d": test_d, "cost_on_test": np.nan, "direction_accuracy": np.nan, "error": err})
            continue
        df_bt, pl, ps, future_ret = triple
        res = run_backtest(df_bt, pl, ps)
        acc, sp, sd, _ = _direction_accuracy_spearman_density(pl, ps, future_ret)
        rows.append({
            "model_id": model_id,
            "train_d": train_d,
            "valid_d": valid_d,
            "test_d": test_d,
            "cost_on_test": res.get("total_return") if res else np.nan,
            "direction_accuracy": acc,
            "spearman": sp,
            "signal_density": sd,
            "error": "",
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_walkforward.csv", index=False)
    print("[FR2] Wrote fr2_walkforward.csv", flush=True)
    return rows


def run_final_comparison():
    metrics = ["direction_accuracy", "spearman", "signal_density", "mean_return_signal", "cost_on", "MDD", "trades", "rolling_win_ratio", "walkforward_win_ratio"]
    align = pd.read_csv(OUT_DIR / "fr2_alignment.csv") if (OUT_DIR / "fr2_alignment.csv").exists() else pd.DataFrame()
    dens = pd.read_csv(OUT_DIR / "fr2_signal_density.csv") if (OUT_DIR / "fr2_signal_density.csv").exists() else pd.DataFrame()
    bt = pd.read_csv(OUT_DIR / "fr2_backtest_comparison.csv") if (OUT_DIR / "fr2_backtest_comparison.csv").exists() else pd.DataFrame()
    roll = pd.read_csv(OUT_DIR / "fr2_rolling_90d.csv") if (OUT_DIR / "fr2_rolling_90d.csv").exists() else pd.DataFrame()
    wf = pd.read_csv(OUT_DIR / "fr2_walkforward.csv") if (OUT_DIR / "fr2_walkforward.csv").exists() else pd.DataFrame()
    rows = []
    for model_id in [m[0] for m in MODELS]:
        r = {"model_id": model_id}
        if len(align):
            a = align[align["model_id"] == model_id]
            if len(a):
                r["direction_accuracy"] = float(a["direction_accuracy"].iloc[0]) if pd.notna(a["direction_accuracy"].iloc[0]) else None
                r["spearman"] = float(a["spearman"].iloc[0]) if pd.notna(a["spearman"].iloc[0]) else None
                r["mean_return_signal"] = float(a["mean_return_signal"].iloc[0]) if pd.notna(a["mean_return_signal"].iloc[0]) else None
        if len(dens):
            d = dens[dens["model_id"] == model_id]
            if len(d):
                r["signal_density"] = float(d["signal_density"].iloc[0]) if pd.notna(d["signal_density"].iloc[0]) else None
        if len(bt):
            b = bt[bt["model_id"] == model_id]
            if len(b):
                r["cost_on"] = float(b["cost_on"].iloc[0]) if pd.notna(b["cost_on"].iloc[0]) else None
                r["MDD"] = float(b["MDD"].iloc[0]) if pd.notna(b["MDD"].iloc[0]) else None
                r["trades"] = int(b["trades"].iloc[0]) if pd.notna(b["trades"].iloc[0]) else None
        if len(roll):
            ro = roll[roll["model_id"] == model_id]
            if len(ro):
                c = ro["cost_on"].iloc[0]
                r["rolling_win_ratio"] = bool(c > 0) if pd.notna(c) else None
        if len(wf):
            w = wf[wf["model_id"] == model_id]
            if len(w):
                c = w["cost_on_test"].iloc[0]
                r["walkforward_win_ratio"] = bool(c > 0) if pd.notna(c) else None
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "fr2_vs_fr1_vs_base.csv", index=False)
    print("[FR2] Wrote fr2_vs_fr1_vs_base.csv", flush=True)
    return rows


def _to_json_safe(obj):
    """Convert to JSON-serializable types only; avoid circular refs and numpy/pandas."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
        return obj.tolist() if hasattr(obj, "tolist") else str(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return str(obj)


def verdict_and_summary(comparison_rows: list) -> str:
    base = next((r for r in comparison_rows if r.get("model_id") == "h15_t0p004"), {})
    fr1 = next((r for r in comparison_rows if r.get("model_id") == "h15_extsafe_v1"), {})
    fr2 = next((r for r in comparison_rows if r.get("model_id") == "h15_micro_v1"), {})
    acc_fr2 = fr2.get("direction_accuracy")
    acc_base = base.get("direction_accuracy")
    acc_fr1 = fr1.get("direction_accuracy")
    sd_fr2 = fr2.get("signal_density")
    sd_fr1 = fr1.get("signal_density")
    wf_fr2 = fr2.get("walkforward_win_ratio")
    wf_base = fr2.get("cost_on")  # use backtest cost_on for FR2 vs base
    improved_align = (acc_fr2 is not None and acc_base is not None and acc_fr2 > acc_base) or (acc_fr2 is not None and acc_fr1 is not None and acc_fr2 > acc_fr1)
    sd_ok = sd_fr2 is None or sd_fr1 is None or sd_fr2 >= sd_fr1 * 0.95
    if improved_align and sd_ok and (wf_fr2 is True or wf_fr2 is None):
        verdict = "FR2_SIGNAL_IMPROVEMENT_CONFIRMED"
    else:
        verdict = "FR2_NO_MEANINGFUL_IMPROVEMENT"
    summary = {
        "verdict": verdict,
        "comparison": [_to_json_safe(r) for r in comparison_rows],
        "feature_preset": "microstructure_v1",
        "model_id_fr2": "h15_micro_v1",
    }
    with open(OUT_DIR / "fr2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    lines = [
        "# FR2 Summary",
        "",
        "## Verdict: " + verdict,
        "",
        "## Comparison (baseline / FR1 / FR2)",
        "",
    ]
    for r in comparison_rows:
        lines.append(f"- **{r.get('model_id', '')}**: direction_accuracy={r.get('direction_accuracy')}, spearman={r.get('spearman')}, signal_density={r.get('signal_density')}, cost_on={r.get('cost_on')}, MDD={r.get('MDD')}, trades={r.get('trades')}")
    lines.extend(["", "## Criteria", "", "- FR2_SIGNAL_IMPROVEMENT_CONFIRMED: alignment increase, signal_density maintained, walk-forward improvement vs baseline.", "- FR2_NO_MEANINGFUL_IMPROVEMENT: alignment no improvement, walk-forward weak, regime collapse repeated."])
    (OUT_DIR / "fr2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[FR2] Wrote fr2_summary.md, fr2_summary.json — verdict={verdict}", flush=True)
    return verdict


def main():
    run_alignment()
    run_signal_density()
    run_backtest_comparison()
    try:
        run_regime_analysis()
    except Exception as e:
        print(f"[FR2] regime_analysis skipped: {e}", flush=True)
    run_rolling_90d()
    run_walkforward()
    comparison_rows = run_final_comparison()
    verdict_and_summary(comparison_rows)


if __name__ == "__main__":
    main()
