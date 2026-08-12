#!/usr/bin/env python3
"""
FR2 최근 OOS 약화 원인 진단.

실험 1: Threshold sweep (0.60~0.80) × 720d/180d/90d
실험 2: Signal quality drift (early/mid/late 240d)
실험 3: Recent regime decomposition (180d, 90d, 2×2 trend×vol)
실험 4: Feature importance (correlation-based, full vs recent 180d)

사용:
  .venv/bin/python scripts/run_fr2_oos_diagnosis.py
  .venv/bin/python scripts/run_fr2_oos_diagnosis.py --quick   # 90d only, 2 thresholds, skip drift/regime/importance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
END_DATE = "2026-03-03"
WINDOW_SIZE = 60
HORIZON = 15
POS_THRESHOLD = 0.004
NEG_THRESHOLD = -0.004

COMMISSION = 0.0009
SLIPPAGE = 0.0001
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8

MODELS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "models"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"
FR2_PRESET = "microstructure_v1"


def _load_ohlcv(days: int, include_microstructure: bool = True) -> pd.DataFrame:
    from src.services.ohlcv_service import load_ohlcv_df
    df = load_ohlcv_df(timeframe=TIMEFRAME, symbol=SYMBOL, include_microstructure=include_microstructure)
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


def get_fr2_proba(days: int) -> tuple[tuple | None, str]:
    """Return ((df_bt, pl, ps, future_ret), '') or (None, err)."""
    from src.features.ml_feature_config import MLFeatureConfig
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel
    from src.indicators.basic import add_basic_indicators

    if not FR2_PT.exists():
        return (None, f"model not found: {FR2_PT}")
    df = _load_ohlcv(days, include_microstructure=True)
    df = add_basic_indicators(df)
    config = MLFeatureConfig.from_preset(FR2_PRESET)
    features = build_feature_frame(df, symbol=SYMBOL, timeframe=TIMEFRAME, feature_config=config)
    features = features.dropna()
    if len(features) < WINDOW_SIZE + HORIZON:
        return (None, f"not enough rows: {len(features)}")
    model = TCNSignalModel(model_path=FR2_PT, use_events=True, feature_config=config)
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


def run_bt(min_max_proba: float, df_bt, pl, ps) -> dict | None:
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
    res, err = run_backtest_7d(
        SYMBOL, TIMEFRAME, df_bt, pl, ps,
        commission_rate=COMMISSION, slippage_rate=SLIPPAGE,
        min_max_proba=min_max_proba, max_entropy=MAX_ENTROPY,
        decision_mode="argmax", min_hold=MIN_HOLD, cooldown=COOLDOWN,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
    )
    if err:
        return None
    return res


def _metrics(pl, ps, future_ret, threshold_signal: float = 0.60):
    p_flat = np.clip(1.0 - pl - ps, 0.0, 1.0)
    pred = np.argmax(np.stack([p_flat, pl, ps], axis=1), axis=1)
    actual = np.where(future_ret > POS_THRESHOLD, 1, np.where(future_ret < NEG_THRESHOLD, 2, 0))
    trim = min(len(pred), len(actual), len(pl))
    if trim < 10:
        return {"direction_accuracy": np.nan, "spearman": np.nan, "signal_density": np.nan, "mean_return_signal": np.nan}
    pl_t, ps_t = pl[:trim], ps[:trim]
    future_ret = future_ret[:trim]
    acc = (pred[:trim] == actual[:trim]).mean()
    sp = pd.Series(pl_t - ps_t).corr(pd.Series(future_ret), method="spearman") if trim > 10 else np.nan
    max_proba = np.maximum(np.maximum(pl_t, ps_t), p_flat[:trim])
    sig_den = (max_proba >= threshold_signal).mean()
    mrs = float(np.mean(future_ret[max_proba >= threshold_signal])) if (max_proba >= threshold_signal).any() else np.nan
    return {"direction_accuracy": acc, "spearman": sp, "signal_density": sig_den, "mean_return_signal": mrs}


# ---------- Experiment 1: Threshold sweep ----------
def run_threshold_sweep(quick: bool = False):
    thresholds = [0.60, 0.70] if quick else [0.60, 0.65, 0.70, 0.75, 0.80]
    windows = [90] if quick else [720, 180, 90]
    rows = []
    for days in windows:
        triple, err = get_fr2_proba(days)
        if err:
            for th in thresholds:
                rows.append({"window_days": days, "threshold": th, "trades": np.nan, "signal_density": np.nan, "direction_accuracy": np.nan, "spearman": np.nan, "mean_return_signal": np.nan, "cost_on": np.nan, "MDD": np.nan, "error": err})
            continue
        df_bt, pl, ps, future_ret = triple
        for th in thresholds:
            met = _metrics(pl, ps, future_ret, threshold_signal=th)
            res = run_bt(th, df_bt, pl, ps)
            row = {
                "window_days": days,
                "threshold": th,
                "trades": res.get("total_trades") if res else np.nan,
                "signal_density": met["signal_density"],
                "direction_accuracy": met["direction_accuracy"],
                "spearman": met["spearman"],
                "mean_return_signal": met["mean_return_signal"],
                "cost_on": res.get("total_return") if res else np.nan,
                "MDD": res.get("max_drawdown") if res else np.nan,
                "error": "",
            }
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_threshold_sweep.csv", index=False)
    print("[FR2-OOS] Wrote fr2_threshold_sweep.csv", flush=True)
    recent90 = [r for r in rows if r["window_days"] == 90]
    pd.DataFrame(recent90).to_csv(OUT_DIR / "fr2_threshold_sweep_recent90.csv", index=False)
    print("[FR2-OOS] Wrote fr2_threshold_sweep_recent90.csv", flush=True)
    return rows


# ---------- Experiment 2: Drift (early/mid/late) ----------
def run_drift():
    triple, err = get_fr2_proba(720)
    if err:
        pd.DataFrame(columns=["segment", "start_days", "end_days", "trades", "signal_density", "direction_accuracy", "spearman", "mean_return_signal", "cost_on", "MDD"]).to_csv(OUT_DIR / "fr2_drift_by_period.csv", index=False)
        print("[FR2-OOS] drift skipped:", err, flush=True)
        return []
    df_bt, pl, ps, future_ret = triple
    n = len(df_bt)
    segs = [
        ("early", 0, n // 3),
        ("mid", n // 3, 2 * n // 3),
        ("late", 2 * n // 3, n),
    ]
    rows = []
    for name, i0, i1 in segs:
        if i1 - i0 < 50:
            continue
        d = df_bt.iloc[i0:i1].reset_index(drop=True)
        pl_s = pl[i0:i1]
        ps_s = ps[i0:i1]
        fr_s = future_ret[i0:i1]
        met = _metrics(pl_s, ps_s, fr_s)
        res = run_bt(0.60, d, pl_s, ps_s)
        start_days = int(720 * i0 / n)
        end_days = int(720 * i1 / n)
        rows.append({
            "segment": name,
            "start_days": start_days,
            "end_days": end_days,
            "trades": res.get("total_trades") if res else np.nan,
            "signal_density": met["signal_density"],
            "direction_accuracy": met["direction_accuracy"],
            "spearman": met["spearman"],
            "mean_return_signal": met["mean_return_signal"],
            "cost_on": res.get("total_return") if res else np.nan,
            "MDD": res.get("max_drawdown") if res else np.nan,
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_drift_by_period.csv", index=False)
    print("[FR2-OOS] Wrote fr2_drift_by_period.csv", flush=True)
    return rows


def add_regime_2x2(df_bt: pd.DataFrame) -> None:
    close = df_bt["close"].astype(float)
    ema = close.ewm(span=min(50, len(close)//4), adjust=False).mean()
    trend = (close > ema).astype(int)
    ret = close.pct_change().fillna(0)
    vol = ret.rolling(20, min_periods=5).std().fillna(ret.std() or 1e-12)
    vol_med = vol.median()
    df_bt["trend_2"] = trend
    df_bt["vol_2"] = (vol >= vol_med).astype(int)


# ---------- Experiment 3: Recent regime decomposition ----------
def run_regime_recent():
    rows = []
    for days in [180, 90]:
        triple, err = get_fr2_proba(days)
        if err:
            continue
        df_bt, pl, ps, future_ret = triple
        add_regime_2x2(df_bt)
        for t in [0, 1]:
            for v in [0, 1]:
                mask = ((df_bt["trend_2"] == t) & (df_bt["vol_2"] == v)).values
                if mask.sum() < 30:
                    rows.append({"window_days": days, "regime": f"trend{t}_vol{v}", "samples": int(mask.sum()), "trades": np.nan, "signal_density": np.nan, "direction_accuracy": np.nan, "spearman": np.nan, "mean_return_signal": np.nan, "cost_on": np.nan, "MDD": np.nan, "sample_size_warning": "too_few"})
                    continue
                d = df_bt.iloc[mask].reset_index(drop=True)
                pl_s = pl[mask]
                ps_s = ps[mask]
                fr_s = future_ret[mask]
                met = _metrics(pl_s, ps_s, fr_s)
                res = run_bt(0.60, d, pl_s, ps_s)
                rows.append({
                    "window_days": days,
                    "regime": f"trend{t}_vol{v}",
                    "samples": int(mask.sum()),
                    "trades": res.get("total_trades") if res else np.nan,
                    "signal_density": met["signal_density"],
                    "direction_accuracy": met["direction_accuracy"],
                    "spearman": met["spearman"],
                    "mean_return_signal": met["mean_return_signal"],
                    "cost_on": res.get("total_return") if res else np.nan,
                    "MDD": res.get("max_drawdown") if res else np.nan,
                    "sample_size_warning": "ok" if mask.sum() >= 80 else "caution",
                })
    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_regime_decomposition_recent.csv", index=False)
        print("[FR2-OOS] Wrote fr2_regime_decomposition_recent.csv", flush=True)
    return rows


# ---------- Experiment 4: Feature importance (correlation-based) ----------
def run_feature_importance():
    from src.features.ml_feature_config import MLFeatureConfig
    from src.ml.features import build_feature_frame
    from src.indicators.basic import add_basic_indicators

    def get_feature_and_return(days: int):
        df = _load_ohlcv(days, include_microstructure=True)
        df = add_basic_indicators(df)
        config = MLFeatureConfig.from_preset(FR2_PRESET)
        features = build_feature_frame(df, symbol=SYMBOL, timeframe=TIMEFRAME, feature_config=config)
        features = features.dropna()
        if len(features) < WINDOW_SIZE + HORIZON:
            return None, None, None
        close = df["close"].values
        valid_len = len(features) - WINDOW_SIZE - HORIZON
        future_ret = (close[WINDOW_SIZE + HORIZON : WINDOW_SIZE + valid_len + HORIZON] / close[WINDOW_SIZE : WINDOW_SIZE + valid_len]) - 1
        if len(future_ret) > valid_len:
            future_ret = future_ret[:valid_len]
        feat_slice = features.iloc[WINDOW_SIZE : WINDOW_SIZE + valid_len].copy()
        if isinstance(feat_slice.index, pd.DatetimeIndex):
            feat_slice = feat_slice.reset_index(drop=True)
        return feat_slice, future_ret, config

    micro_keys = ("ofi_", "buy_pressure", "sell_pressure", "funding_", "oi_", "liq_", "vol_spike", "taker_buy_ratio", "taker_sell_ratio", "vol_imbalance", "vol_momentum")
    def group(fname: str) -> str:
        if any(fname.startswith(k) or k in fname for k in micro_keys):
            return "microstructure"
        if "feat_vol" in fname or "atr" in fname or "volatility" in fname or "realized_vol" in fname:
            return "volatility"
        if "feat_trend" in fname or "ema" in fname or "sma" in fname or "trend" in fname:
            return "trend"
        if "volu" in fname or "volume" in fname:
            return "volume"
        if "struct" in fname or "body" in fname or "wick" in fname:
            return "structure"
        return "technical"

    out_rows = []
    for label, days in [("full", 720), ("recent", 180)]:
        feat_slice, future_ret, _ = get_feature_and_return(days)
        if feat_slice is None or future_ret is None or len(future_ret) < 50:
            continue
        future_ret = np.asarray(future_ret[: len(feat_slice)])
        numeric_cols = [c for c in feat_slice.columns if feat_slice[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
        imp = []
        for col in numeric_cols:
            x = feat_slice[col].fillna(0).values
            if np.std(x) < 1e-12:
                imp.append((col, 0.0, group(col)))
                continue
            c = np.corrcoef(x, future_ret)[0, 1]
            imp.append((col, float(c) if not np.isnan(c) else 0.0, group(col)))
        imp.sort(key=lambda t: abs(t[1]), reverse=True)
        for rank, (col, score, grp) in enumerate(imp, 1):
            out_rows.append({"window": label, "window_days": days, "feature": col, "importance_abs_corr": abs(score), "importance_signed": score, "rank": rank, "group": grp})
    if out_rows:
        pd.DataFrame(out_rows).to_csv(OUT_DIR / "fr2_feature_importance_full.csv", index=False)
        print("[FR2-OOS] Wrote fr2_feature_importance_full.csv", flush=True)
        recent_only = [r for r in out_rows if r["window"] == "recent"]
        if recent_only:
            pd.DataFrame(recent_only).to_csv(OUT_DIR / "fr2_feature_importance_recent.csv", index=False)
            print("[FR2-OOS] Wrote fr2_feature_importance_recent.csv", flush=True)
    return out_rows


def write_reports():
    """Write markdown reports from generated CSVs."""
    # Threshold sweep report
    p = OUT_DIR / "fr2_threshold_sweep.csv"
    if p.exists():
        df = pd.read_csv(p)
        lines = ["# FR2 Threshold Sweep Report\n", "| window_days | threshold | trades | signal_density | direction_accuracy | spearman | cost_on | MDD |", "|-------------|-----------|--------|----------------|--------------------|----------|---------|-----|"]
        for _, r in df.iterrows():
            lines.append(f"| {r['window_days']} | {r['threshold']} | {r['trades']} | {float(r['signal_density']):.4f} | {float(r['direction_accuracy']):.4f} | {float(r['spearman']):.4f} | {float(r['cost_on']):.4f} | {float(r['MDD']):.4f} |")
        (OUT_DIR / "FR2_THRESHOLD_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # Drift report
    p = OUT_DIR / "fr2_drift_by_period.csv"
    if p.exists():
        df = pd.read_csv(p)
        lines = ["# FR2 Signal Quality Drift Report\n", "| segment | start_days | end_days | trades | signal_density | direction_accuracy | spearman | cost_on | MDD |", "|---------|------------|----------|--------|----------------|--------------------|----------|---------|-----|"]
        for _, r in df.iterrows():
            lines.append(f"| {r['segment']} | {r['start_days']} | {r['end_days']} | {r['trades']} | {float(r['signal_density']):.4f} | {float(r['direction_accuracy']):.4f} | {float(r['spearman']):.4f} | {float(r['cost_on']):.4f} | {float(r['MDD']):.4f} |")
        (OUT_DIR / "FR2_DRIFT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # Regime report
    p = OUT_DIR / "fr2_regime_decomposition_recent.csv"
    if p.exists():
        df = pd.read_csv(p)
        lines = ["# FR2 Recent Regime Decomposition Report\n", "| window_days | regime | samples | trades | signal_density | direction_accuracy | spearman | cost_on | MDD | sample_size_warning |", "|-------------|--------|---------|--------|----------------|--------------------|----------|---------|-----|---------------------|"]
        for _, r in df.iterrows():
            sd = float(r["signal_density"]) if pd.notna(r["signal_density"]) else 0
            acc = float(r["direction_accuracy"]) if pd.notna(r["direction_accuracy"]) else 0
            sp = float(r["spearman"]) if pd.notna(r["spearman"]) else 0
            co = float(r["cost_on"]) if pd.notna(r["cost_on"]) else 0
            mdd = float(r["MDD"]) if pd.notna(r["MDD"]) else 0
            lines.append(f"| {r['window_days']} | {r['regime']} | {r['samples']} | {r['trades']} | {sd:.4f} | {acc:.4f} | {sp:.4f} | {co:.4f} | {mdd:.4f} | {r.get('sample_size_warning','')} |")
        (OUT_DIR / "FR2_REGIME_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # Importance report (top 20 full + recent)
    p = OUT_DIR / "fr2_feature_importance_full.csv"
    if p.exists():
        df = pd.read_csv(p)
        lines = ["# FR2 Feature Importance Report (|corr| with future return)\n"]
        for win in ["full", "recent"]:
            sub = df[df["window"] == win].head(20)
            lines.append(f"\n## Top 20 ({win})\n| feature | importance_abs_corr | group | rank |")
            lines.append("|---------|---------------------|-------|------|")
            for _, r in sub.iterrows():
                lines.append(f"| {r['feature']} | {float(r['importance_abs_corr']):.4f} | {r['group']} | {r['rank']} |")
        (OUT_DIR / "FR2_IMPORTANCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("[FR2-OOS] Wrote FR2_*_REPORT.md", flush=True)


def write_final_decision_report():
    """Synthesize FR2_FINAL_DECISION_REPORT.md from experiments."""
    sections = []
    sections.append("# FR2 Final Decision Report: Recent OOS Weakness Diagnosis\n")
    sections.append("## Executive summary\n")
    sections.append("FR2 (h15_micro_v1) is strong on 720d but underperforms baseline on recent 90d (cost_on baseline +0.006, FR2 -0.196). This report summarizes threshold sweep, drift, regime decomposition, and feature importance to diagnose cause.\n")
    sections.append("## What was run\n")
    sections.append("- Experiment 1: Threshold sweep (0.60–0.80) × windows 720d, 180d, 90d.")
    sections.append("- Experiment 2: Drift by period (early/mid/late 240d over 720d).")
    sections.append("- Experiment 3: Recent regime decomposition (180d, 90d; 2×2 trend×vol).")
    sections.append("- Experiment 4: Feature importance (|corr| with future return, full vs recent 180d).\n")
    sections.append("## Key tables\n")
    sections.append("See: fr2_threshold_sweep.csv, fr2_drift_by_period.csv, fr2_regime_decomposition_recent.csv, fr2_feature_importance_full.csv.\n")
    sections.append("## Recent OOS interpretation\n")
    sections.append("- If threshold sweep shows recent 90d cost_on improving with higher threshold → threshold/entry quality issue.")
    sections.append("- If drift shows late segment accuracy/spearman collapse → signal quality drift.")
    sections.append("- If regime decomposition shows one cell dominating loss → regime concentration.")
    sections.append("- If importance shows microstructure features drop in recent → feature relevance decay.\n")
    sections.append("## Statistical cautions\n")
    sections.append("- Recent 90d and regime cells with very few trades are not conclusive.")
    sections.append("- 720d strength does not imply recent OOS is safe for production.")
    sections.append("- Importance is predictive correlation, not trading causality.\n")
    sections.append("## Recommended next action\n")
    sections.append("Based on experiments: if threshold lift improves recent 90d and regime is not concentrated in one bad cell → CONDITIONAL_CANDIDATE_REQUIRES_FILTERING. If drift is clear and importance shifts → STRONG_CHALLENGER_NOT_YET_ROBUST. See verdict below.\n")
    verdict = "STRONG_CHALLENGER_NOT_YET_ROBUST"
    p90 = OUT_DIR / "fr2_threshold_sweep_recent90.csv"
    if p90.exists():
        try:
            df = pd.read_csv(p90)
            best = df.loc[df["cost_on"].idxmax()] if len(df) and pd.notna(df["cost_on"]).any() else None
            if best is not None and float(best["cost_on"]) > 0.006:
                verdict = "CONDITIONAL_CANDIDATE_REQUIRES_FILTERING"
            elif best is not None and float(best["cost_on"]) <= 0:
                verdict = "STRONG_CHALLENGER_NOT_YET_ROBUST"
        except Exception:
            pass
    sections.append("## Final verdict label\n")
    sections.append(f"**{verdict}**\n")
    sections.append("(One of: ROBUST_CANDIDATE | CONDITIONAL_CANDIDATE_REQUIRES_FILTERING | STRONG_CHALLENGER_NOT_YET_ROBUST | LIKELY_PERIOD_SPECIFIC_EDGE | DO_NOT_ADVANCE_YET.)\n")
    (OUT_DIR / "FR2_FINAL_DECISION_REPORT.md").write_text("\n".join(sections), encoding="utf-8")
    print("[FR2-OOS] Wrote FR2_FINAL_DECISION_REPORT.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description="FR2 recent OOS weakness diagnosis")
    parser.add_argument("--quick", action="store_true", help="90d only, 2 thresholds, skip drift/regime/importance")
    parser.add_argument("--reports-only", action="store_true", help="Only regenerate markdown reports from existing CSVs")
    args = parser.parse_args()
    quick = getattr(args, "quick", False)
    reports_only = getattr(args, "reports_only", False)

    if reports_only:
        write_reports()
        write_final_decision_report()
        print("[FR2-OOS] Reports only: done.", flush=True)
        return

    print("[FR2-OOS] Experiment 1: Threshold sweep", flush=True)
    run_threshold_sweep(quick=quick)
    if not quick:
        print("[FR2-OOS] Experiment 2: Drift", flush=True)
        run_drift()
        print("[FR2-OOS] Experiment 3: Regime recent", flush=True)
        run_regime_recent()
        print("[FR2-OOS] Experiment 4: Feature importance", flush=True)
        run_feature_importance()
    print("[FR2-OOS] Writing reports", flush=True)
    write_reports()
    write_final_decision_report()
    print("[FR2-OOS] Done.", flush=True)


if __name__ == "__main__":
    main()
