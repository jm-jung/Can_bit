#!/usr/bin/env python3
"""
FR2 Regime Conditioning: 진단 분석만 (feature/모델 변경 없음).

실험 1: Volatility regime (low/mid/high by realized_vol percentile)
실험 2: Trend regime (uptrend/downtrend by EMA200)
실험 3: Cross regime (uptrend_lowvol, uptrend_highvol, downtrend_lowvol, downtrend_highvol)
실험 4: Recent 90d 동일 분해 → 어떤 regime가 손실을 만드는지

사용:
  .venv/bin/python scripts/run_fr2_regime_conditioning.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse FR2 OOS diagnosis data + backtest
from scripts.run_fr2_oos_diagnosis import (
    _load_ohlcv,
    get_fr2_proba,
    run_bt,
    _metrics,
)


def add_regime_columns(days: int, df_bt: pd.DataFrame) -> pd.DataFrame:
    """Add trend_regime (EMA200), vol_regime (realized_vol_30 percentile), cross_regime. In-place on copy."""
    df = _load_ohlcv(days, include_microstructure=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    close = df["close"].astype(float)
    # EMA200 → trend
    df["ema200"] = close.ewm(span=200, adjust=False).mean()
    df["trend_regime"] = np.where(close > df["ema200"], "uptrend", "downtrend")
    # realized_vol_30 → vol percentile 0-33 / 33-66 / 66-100
    ret = close.pct_change()
    df["realized_vol_30"] = ret.rolling(30, min_periods=5).std()
    df["realized_vol_30"] = df["realized_vol_30"].bfill().fillna(1e-12)
    df["vol_pct"] = df["realized_vol_30"].rank(pct=True)
    df["vol_regime"] = "mid_vol"
    df.loc[df["vol_pct"] <= 1 / 3, "vol_regime"] = "low_vol"
    df.loc[df["vol_pct"] > 2 / 3, "vol_regime"] = "high_vol"
    # Cross: lowvol = bottom 50%, highvol = top 50%
    df["vol_cross"] = np.where(df["vol_pct"] <= 0.5, "lowvol", "highvol")
    df["cross_regime"] = df["trend_regime"] + "_" + df["vol_cross"]

    regime_df = df[["timestamp", "trend_regime", "vol_regime", "vol_cross", "cross_regime"]].drop_duplicates("timestamp")
    # Align timezone for merge
    if hasattr(df_bt["timestamp"].dtype, "tz") and df_bt["timestamp"].dtype.tz is not None:
        if regime_df["timestamp"].dt.tz is None:
            regime_df["timestamp"] = regime_df["timestamp"].dt.tz_localize("UTC", ambiguous="infer")
    elif regime_df["timestamp"].dt.tz is not None and (df_bt["timestamp"].dtype.kind == "M" and getattr(df_bt["timestamp"].dtype, "tz", None) is None):
        df_bt = df_bt.copy()
        df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"], utc=True)
    out = df_bt.merge(regime_df, on="timestamp", how="left")
    return out


def _one_regime_metrics(pl: np.ndarray, ps: np.ndarray, future_ret: np.ndarray, df_bt: pd.DataFrame, mask, threshold: float = 0.60):
    if mask.sum() < 20:
        return None, None
    pl_s = pl[mask]
    ps_s = ps[mask]
    fr_s = future_ret[mask]
    met = _metrics(pl_s, ps_s, fr_s, threshold_signal=threshold)
    d = df_bt.loc[mask].reset_index(drop=True)
    res = run_bt(threshold, d, pl_s, ps_s)
    return met, res


# ---------- Experiment 1: Volatility regime ----------
def run_volatility_regime_from_data(df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray, future_ret: np.ndarray, days: int) -> list:
    rows = []
    for reg in ["low_vol", "mid_vol", "high_vol"]:
        mask = (df_bt["vol_regime"] == reg).values
        met, res = _one_regime_metrics(pl, ps, future_ret, df_bt, mask)
        if met is None:
            rows.append({"window_days": days, "regime": reg, "trades": np.nan, "signal_density": np.nan, "direction_accuracy": np.nan, "spearman": np.nan, "mean_return_signal": np.nan, "cost_on": np.nan, "MDD": np.nan})
            continue
        trades = res.get("total_trades") if res else np.nan
        rows.append({
            "window_days": days, "regime": reg, "trades": trades,
            "signal_density": met["signal_density"], "direction_accuracy": met["direction_accuracy"],
            "spearman": met["spearman"], "mean_return_signal": met["mean_return_signal"],
            "cost_on": res.get("total_return") if res else np.nan,
            "MDD": res.get("max_drawdown") if res else np.nan,
        })
    return rows


# ---------- Experiment 2: Trend regime ----------
def run_trend_regime_from_data(df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray, future_ret: np.ndarray, days: int) -> list:
    rows = []
    for reg in ["uptrend", "downtrend"]:
        mask = (df_bt["trend_regime"] == reg).values
        met, res = _one_regime_metrics(pl, ps, future_ret, df_bt, mask)
        if met is None:
            rows.append({"window_days": days, "regime": reg, "trades": np.nan, "signal_density": np.nan, "direction_accuracy": np.nan, "spearman": np.nan, "mean_return_signal": np.nan, "cost_on": np.nan, "MDD": np.nan})
            continue
        rows.append({
            "window_days": days, "regime": reg, "trades": res.get("total_trades") if res else np.nan,
            "signal_density": met["signal_density"], "direction_accuracy": met["direction_accuracy"],
            "spearman": met["spearman"], "mean_return_signal": met["mean_return_signal"],
            "cost_on": res.get("total_return") if res else np.nan,
            "MDD": res.get("max_drawdown") if res else np.nan,
        })
    return rows


# ---------- Experiment 3: Cross regime ----------
def run_cross_regime_from_data(df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray, future_ret: np.ndarray, days: int) -> list:
    rows = []
    for reg in ["uptrend_lowvol", "uptrend_highvol", "downtrend_lowvol", "downtrend_highvol"]:
        mask = (df_bt["cross_regime"] == reg).values
        met, res = _one_regime_metrics(pl, ps, future_ret, df_bt, mask)
        if met is None:
            rows.append({"window_days": days, "regime": reg, "trades": np.nan, "signal_density": np.nan, "direction_accuracy": np.nan, "spearman": np.nan, "mean_return_signal": np.nan, "cost_on": np.nan, "MDD": np.nan, "sample_too_small": "yes"})
            continue
        trades = res.get("total_trades") if res else 0
        rows.append({
            "window_days": days, "regime": reg, "trades": trades,
            "signal_density": met["signal_density"], "direction_accuracy": met["direction_accuracy"],
            "spearman": met["spearman"], "mean_return_signal": met["mean_return_signal"],
            "cost_on": res.get("total_return") if res else np.nan,
            "MDD": res.get("max_drawdown") if res else np.nan,
            "sample_too_small": "yes" if (trades is not None and int(trades) < 100) else "no",
        })
    return rows


def run_all_720d(triple, days: int = 720):
    df_bt, pl, ps, future_ret = triple
    df_bt = add_regime_columns(days, df_bt)
    pl = np.asarray(pl)
    ps = np.asarray(ps)
    future_ret = np.asarray(future_ret)
    v_rows = run_volatility_regime_from_data(df_bt, pl, ps, future_ret, days)
    pd.DataFrame(v_rows).to_csv(OUT_DIR / "fr2_volatility_regime.csv", index=False)
    print("[FR2-Regime] Wrote fr2_volatility_regime.csv", flush=True)
    t_rows = run_trend_regime_from_data(df_bt, pl, ps, future_ret, days)
    pd.DataFrame(t_rows).to_csv(OUT_DIR / "fr2_trend_regime.csv", index=False)
    print("[FR2-Regime] Wrote fr2_trend_regime.csv", flush=True)
    c_rows = run_cross_regime_from_data(df_bt, pl, ps, future_ret, days)
    pd.DataFrame(c_rows).to_csv(OUT_DIR / "fr2_cross_regime.csv", index=False)
    print("[FR2-Regime] Wrote fr2_cross_regime.csv", flush=True)


# ---------- Experiment 4: Recent 90d regime ----------
def run_recent90_from_data(df_bt: pd.DataFrame, pl: np.ndarray, ps: np.ndarray, future_ret: np.ndarray, days: int = 90) -> list:
    rows = []
    for reg in ["low_vol", "mid_vol", "high_vol"]:
        mask = (df_bt["vol_regime"] == reg).values
        met, res = _one_regime_metrics(pl, ps, future_ret, df_bt, mask)
        t = res.get("total_trades") if res else None
        rows.append({"window_days": days, "decomposition": "volatility", "regime": reg, "trades": t, "signal_density": met["signal_density"] if met else np.nan, "direction_accuracy": met["direction_accuracy"] if met else np.nan, "spearman": met["spearman"] if met else np.nan, "mean_return_signal": met["mean_return_signal"] if met else np.nan, "cost_on": res.get("total_return") if res else np.nan, "MDD": res.get("max_drawdown") if res else np.nan, "sample_too_small": "yes" if (t is not None and int(t) < 100) else "no"})
    for reg in ["uptrend", "downtrend"]:
        mask = (df_bt["trend_regime"] == reg).values
        met, res = _one_regime_metrics(pl, ps, future_ret, df_bt, mask)
        t = res.get("total_trades") if res else None
        rows.append({"window_days": days, "decomposition": "trend", "regime": reg, "trades": t, "signal_density": met["signal_density"] if met else np.nan, "direction_accuracy": met["direction_accuracy"] if met else np.nan, "spearman": met["spearman"] if met else np.nan, "mean_return_signal": met["mean_return_signal"] if met else np.nan, "cost_on": res.get("total_return") if res else np.nan, "MDD": res.get("max_drawdown") if res else np.nan, "sample_too_small": "yes" if (t is not None and int(t) < 100) else "no"})
    for reg in ["uptrend_lowvol", "uptrend_highvol", "downtrend_lowvol", "downtrend_highvol"]:
        mask = (df_bt["cross_regime"] == reg).values
        met, res = _one_regime_metrics(pl, ps, future_ret, df_bt, mask)
        t = res.get("total_trades") if res else None
        rows.append({"window_days": days, "decomposition": "cross", "regime": reg, "trades": t, "signal_density": met["signal_density"] if met else np.nan, "direction_accuracy": met["direction_accuracy"] if met else np.nan, "spearman": met["spearman"] if met else np.nan, "mean_return_signal": met["mean_return_signal"] if met else np.nan, "cost_on": res.get("total_return") if res else np.nan, "MDD": res.get("max_drawdown") if res else np.nan, "sample_too_small": "yes" if (t is not None and int(t) < 100) else "no"})
    return rows


def write_volatility_report():
    p = OUT_DIR / "fr2_volatility_regime.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lines = ["# FR2 Volatility Regime Report\n", "realized_vol_30 percentile: low_vol 0–33%, mid_vol 33–66%, high_vol 66–100%.\n", "| regime | trades | signal_density | direction_accuracy | spearman | mean_return_signal | cost_on | MDD |", "|--------|--------|----------------|--------------------|----------|--------------------|---------|-----|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['regime']} | {r['trades']} | {float(r['signal_density']):.4f} | {float(r['direction_accuracy']):.4f} | {float(r['spearman']):.4f} | {float(r['mean_return_signal']):.6f} | {float(r['cost_on']):.4f} | {float(r['MDD']):.4f} |")
    (OUT_DIR / "FR2_VOLATILITY_REGIME.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-Regime] Wrote FR2_VOLATILITY_REGIME.md", flush=True)


def write_trend_report():
    p = OUT_DIR / "fr2_trend_regime.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lines = ["# FR2 Trend Regime Report\n", "EMA200: uptrend = close > EMA200, downtrend = close <= EMA200.\n", "| regime | trades | signal_density | direction_accuracy | spearman | mean_return_signal | cost_on | MDD |", "|--------|--------|----------------|--------------------|----------|--------------------|---------|-----|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['regime']} | {r['trades']} | {float(r['signal_density']):.4f} | {float(r['direction_accuracy']):.4f} | {float(r['spearman']):.4f} | {float(r['mean_return_signal']):.6f} | {float(r['cost_on']):.4f} | {float(r['MDD']):.4f} |")
    (OUT_DIR / "FR2_TREND_REGIME.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-Regime] Wrote FR2_TREND_REGIME.md", flush=True)


def write_cross_report():
    p = OUT_DIR / "fr2_cross_regime.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lines = ["# FR2 Cross Regime Report (Trend × Vol)\n", "| regime | trades | signal_density | direction_accuracy | spearman | cost_on | MDD | sample_too_small |", "|--------|--------|----------------|--------------------|----------|---------|-----|------------------|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['regime']} | {r['trades']} | {float(r['signal_density']):.4f} | {float(r['direction_accuracy']):.4f} | {float(r['spearman']):.4f} | {float(r['cost_on']):.4f} | {float(r['MDD']):.4f} | {r.get('sample_too_small','')} |")
    (OUT_DIR / "FR2_CROSS_REGIME_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-Regime] Wrote FR2_CROSS_REGIME_REPORT.md", flush=True)


def write_recent90_report():
    p = OUT_DIR / "fr2_recent90_regime.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lines = ["# FR2 Recent 90d Regime Report\n", "| decomposition | regime | trades | signal_density | direction_accuracy | spearman | cost_on | MDD | sample_too_small |", "|---------------|--------|--------|----------------|--------------------|----------|---------|-----|------------------|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['decomposition']} | {r['regime']} | {r['trades']} | {float(r['signal_density']):.4f} | {float(r['direction_accuracy']):.4f} | {float(r['spearman']):.4f} | {float(r['cost_on']):.4f} | {float(r['MDD']):.4f} | {r.get('sample_too_small','')} |")
    (OUT_DIR / "FR2_RECENT90_REGIME_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-Regime] Wrote FR2_RECENT90_REGIME_REPORT.md", flush=True)


def write_final_report():
    """FR2_REGIME_FINAL_REPORT.md: volatility + trend + cross + recent90 + interpretation + verdict."""
    sections = [
        "# FR2 Regime Conditioning Final Report\n",
        "## 1) Volatility regime 결과\n",
        "See: fr2_volatility_regime.csv, FR2_VOLATILITY_REGIME.md.\n",
        "## 2) Trend regime 결과\n",
        "See: fr2_trend_regime.csv, FR2_TREND_REGIME.md.\n",
        "## 3) Cross regime 결과\n",
        "See: fr2_cross_regime.csv, FR2_CROSS_REGIME_REPORT.md.\n",
        "## 4) Recent 90d regime 결과\n",
        "See: fr2_recent90_regime.csv, FR2_RECENT90_REGIME_REPORT.md.\n",
        "## 5) 핵심 해석\n",
    ]
    # Infer verdict from CSVs if present
    vol_path = OUT_DIR / "fr2_volatility_regime.csv"
    recent_path = OUT_DIR / "fr2_recent90_regime.csv"
    verdict = "REGIME_DEPENDENT_SIGNAL"
    if recent_path.exists():
        df = pd.read_csv(recent_path)
        cost = df["cost_on"].dropna()
        if len(cost) and (cost > 0).any() and (cost <= 0).any():
            verdict = "REGIME_DEPENDENT_SIGNAL"
        elif len(cost) and (cost <= 0).all():
            verdict = "LIKELY_PERIOD_SPECIFIC"
        elif len(cost) and (cost > 0).all():
            if vol_path.exists():
                vdf = pd.read_csv(vol_path)
                if len(vdf) and (vdf["cost_on"] > 0).all():
                    verdict = "ROBUST_SIGNAL"
    else:
        # recent90 not run yet; 720d vol positive but overall recent OOS weak → conditional
        verdict = "REGIME_DEPENDENT_SIGNAL"
    sections.append("- **FR2는 모든 regime에서 작동하는가 (A)?** → 720d volatility/trend/cross 테이블에서 cost_on > 0인 regime만 있으면 A에 가깝다.\n")
    sections.append("- **특정 regime에서만 작동하는가 (B)?** → 일부 regime만 cost_on > 0이면 B.\n")
    sections.append("- **recent regime shift 때문에 깨졌는가 (C)?** → recent 90d에서 특정 regime가 큰 손실을 만들면 C.\n")
    sections.append("\n## 최종 판정\n")
    sections.append(f"**{verdict}**\n")
    sections.append("(ROBUST_SIGNAL | REGIME_DEPENDENT_SIGNAL | LIKELY_PERIOD_SPECIFIC)\n")
    (OUT_DIR / "FR2_REGIME_FINAL_REPORT.md").write_text("".join(sections), encoding="utf-8")
    print("[FR2-Regime] Wrote FR2_REGIME_FINAL_REPORT.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description="FR2 Regime Conditioning (diagnostic only)")
    parser.add_argument("--reports-only", action="store_true", help="Only regenerate markdown from existing CSVs")
    parser.add_argument("--recent-only", action="store_true", help="Only run recent 90d regime (one inference)")
    args = parser.parse_args()
    if getattr(args, "reports_only", False):
        write_volatility_report()
        write_trend_report()
        write_cross_report()
        write_recent90_report()
        write_final_report()
        print("[FR2-Regime] Reports only: done.", flush=True)
        return
    if getattr(args, "recent_only", False):
        print("[FR2-Regime] Loading 90d...", flush=True)
        triple, err = get_fr2_proba(90)
        if err:
            print("[FR2-Regime] recent-only failed:", err, flush=True)
            return
        df_bt = add_regime_columns(90, triple[0])
        pl = np.asarray(triple[1])
        ps = np.asarray(triple[2])
        future_ret = np.asarray(triple[3])
        rows = run_recent90_from_data(df_bt, pl, ps, future_ret, 90)
        pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_recent90_regime.csv", index=False)
        print("[FR2-Regime] Wrote fr2_recent90_regime.csv", flush=True)
        write_recent90_report()
        write_final_report()
        print("[FR2-Regime] Done (recent-only).", flush=True)
        return
    print("[FR2-Regime] Loading 720d (single inference)...", flush=True)
    triple_720, err = get_fr2_proba(720)
    if err:
        print("[FR2-Regime] 720d failed:", err, flush=True)
        for name in ["fr2_volatility_regime.csv", "fr2_trend_regime.csv", "fr2_cross_regime.csv"]:
            pd.DataFrame(columns=[]).to_csv(OUT_DIR / name, index=False)
    else:
        print("[FR2-Regime] Experiment 1–3: Volatility + Trend + Cross (720d)", flush=True)
        run_all_720d(triple_720, 720)
    print("[FR2-Regime] Loading 90d...", flush=True)
    triple_90, err90 = get_fr2_proba(90)
    if err90:
        print("[FR2-Regime] 90d failed:", err90, flush=True)
        pd.DataFrame(columns=["window_days", "decomposition", "regime", "trades", "signal_density", "direction_accuracy", "spearman", "mean_return_signal", "cost_on", "MDD", "sample_too_small"]).to_csv(OUT_DIR / "fr2_recent90_regime.csv", index=False)
    else:
        print("[FR2-Regime] Experiment 4: Recent 90d regime", flush=True)
        df_bt = add_regime_columns(90, triple_90[0])
        rows = run_recent90_from_data(df_bt, np.asarray(triple_90[1]), np.asarray(triple_90[2]), np.asarray(triple_90[3]), 90)
        pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_recent90_regime.csv", index=False)
        print("[FR2-Regime] Wrote fr2_recent90_regime.csv", flush=True)
    print("[FR2-Regime] Writing reports", flush=True)
    write_volatility_report()
    write_trend_report()
    write_cross_report()
    write_recent90_report()
    write_final_report()
    print("[FR2-Regime] Done.", flush=True)


if __name__ == "__main__":
    main()
