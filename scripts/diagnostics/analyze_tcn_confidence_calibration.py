"""
TCN confidence calibration forensics (diagnostics only).

Analyzes when TCN probability is overconfident relative to realized
trade/candidate outcomes. Does not retrain, promote, or change production/Q2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.analyze_executed_first_meta_learning import _prepare_full_df
from scripts.diagnostics.analyze_regime_conditioned_meta import _q2_scale, _segment_regimes
from scripts.diagnostics.build_meta_label_dataset import LEAKAGE_COLS

OUT_DIR = Path("data/diagnostics/tcn_confidence_calibration")
POSITION_SIZE = 0.05
BIN_EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
BIN_LABELS = [
    "0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75",
    "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95+",
]


def _mdd(returns: pd.Series) -> float:
    eq = peak = 1.0
    mdd = 0.0
    for r in returns.fillna(0):
        eq *= 1.0 + float(r) * POSITION_SIZE
        peak = max(peak, eq)
        mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0.0)
    return float(mdd)


def _ece(conf: pd.Series, success: pd.Series, bins: pd.Series | None = None) -> float:
    tmp = pd.DataFrame({"confidence": conf.astype(float), "success": success.astype(float)})
    tmp = tmp.dropna()
    if tmp.empty:
        return float("nan")
    if bins is None:
        bins = pd.cut(tmp["confidence"], bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True, right=False)
    tmp["bin"] = bins
    err = 0.0
    n = len(tmp)
    for _, g in tmp.groupby("bin", observed=False):
        if g.empty:
            continue
        err += (len(g) / n) * abs(float(g["confidence"].mean()) - float(g["success"].mean()))
    return float(err)


def _brier(conf: pd.Series, success: pd.Series) -> float:
    tmp = pd.DataFrame({"confidence": conf.astype(float), "success": success.astype(float)}).dropna()
    if tmp.empty or tmp["success"].nunique() < 2:
        return float(((tmp["confidence"] - tmp["success"]) ** 2).mean()) if len(tmp) else float("nan")
    return float(brier_score_loss(tmp["success"], tmp["confidence"]))


def _safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else 0.0


def _prepare_dataset() -> pd.DataFrame:
    df = _prepare_full_df().copy()
    df = df.sort_values(["df_idx", "entry_ts"], na_position="last").reset_index(drop=True)
    df = _segment_regimes(df)
    df["q2_bdi_scale"] = df.apply(_q2_scale, axis=1)
    df["predicted_direction"] = df[["p_long", "p_short", "p_flat"]].idxmax(axis=1).str.replace("p_", "").str.upper()
    df["predicted_confidence"] = df[["p_long", "p_short", "p_flat"]].max(axis=1)
    df["direction_confidence"] = np.where(
        df["direction"].astype(str).eq("LONG"),
        df["p_long"].astype(float),
        df["p_short"].astype(float),
    )
    df["actual_success"] = df["binary_good_trade"].astype(int)
    # FLAT success is diagnostic-only: high p_flat would have been correct when the candidate trade was bad.
    df["flat_avoidance_success"] = df["binary_bad_trade"].astype(int)
    df["false_high_signature"] = df.get("false_high_signature_flag", 0).astype(bool)
    df["entropy_spike"] = df["entropy"].astype(float).diff().abs().fillna(0) > 0.08
    df["trend_transition"] = df.get("regime_transition_flag", 0).astype(bool)
    df["high_confidence"] = df["predicted_confidence"] >= 0.55
    df["high_conf_failure"] = df["high_confidence"] & (df["actual_success"] == 0)
    df["overconfidence_gap"] = (df["predicted_confidence"] - df["actual_success"]).clip(lower=0)
    df["underconfidence_gap"] = (df["actual_success"] - df["predicted_confidence"]).clip(lower=0)
    df["p_long_delta"] = df["p_long"].astype(float).diff().fillna(0)
    df["p_short_delta"] = df["p_short"].astype(float).diff().fillna(0)
    df["p_flat_delta"] = df["p_flat"].astype(float).diff().fillna(0)
    return df


def _direction_frame(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction == "LONG":
        out = df[df["direction"].astype(str).eq("LONG")].copy()
        out["calibration_confidence"] = out["p_long"].astype(float)
        out["calibration_success"] = out["actual_success"].astype(int)
        return out
    if direction == "SHORT":
        out = df[df["direction"].astype(str).eq("SHORT")].copy()
        out["calibration_confidence"] = out["p_short"].astype(float)
        out["calibration_success"] = out["actual_success"].astype(int)
        return out
    out = df.copy()
    out["calibration_confidence"] = out["p_flat"].astype(float)
    out["calibration_success"] = out["flat_avoidance_success"].astype(int)
    return out


def _bin_stats(frame: pd.DataFrame, group: Dict[str, Any]) -> List[Dict[str, Any]]:
    if frame.empty:
        return []
    tmp = frame.copy()
    tmp["confidence_bin"] = pd.cut(
        tmp["calibration_confidence"].astype(float),
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        include_lowest=True,
        right=False,
    )
    rows: List[Dict[str, Any]] = []
    for label in BIN_LABELS:
        g = tmp[tmp["confidence_bin"].astype(str).eq(label)]
        if g.empty:
            rows.append({
                **group,
                "confidence_bin": label,
                "prediction_count": 0,
                "mean_confidence": np.nan,
                "actual_success_rate": np.nan,
                "calibration_error": np.nan,
                "brier_score": np.nan,
                "ece": np.nan,
                "overconfidence_score": np.nan,
                "underconfidence_score": np.nan,
            })
            continue
        mean_conf = float(g["calibration_confidence"].mean())
        success = float(g["calibration_success"].mean())
        rows.append({
            **group,
            "confidence_bin": label,
            "prediction_count": int(len(g)),
            "mean_confidence": mean_conf,
            "actual_success_rate": success,
            "calibration_error": float(abs(mean_conf - success)),
            "brier_score": _brier(g["calibration_confidence"], g["calibration_success"]),
            "ece": _ece(g["calibration_confidence"], g["calibration_success"]),
            "overconfidence_score": float(max(mean_conf - success, 0.0)),
            "underconfidence_score": float(max(success - mean_conf, 0.0)),
        })
    return rows


def phase1_global(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for direction in ["LONG", "SHORT", "FLAT"]:
        frame = _direction_frame(df, direction)
        rows.extend(_bin_stats(frame, {"scope": "global", "direction": direction}))
        valid = frame[frame["calibration_confidence"] >= 0.50]
        summary[direction] = {
            "rows": int(len(frame)),
            "rows_conf_ge_050": int(len(valid)),
            "success_rate": _safe_mean(frame["calibration_success"]),
            "mean_confidence": _safe_mean(frame["calibration_confidence"]),
            "brier_score": _brier(frame["calibration_confidence"], frame["calibration_success"]),
            "ece": _ece(frame["calibration_confidence"], frame["calibration_success"]),
            "high_conf_failure_rate": float(((frame["calibration_confidence"] >= 0.55) & (frame["calibration_success"] == 0)).mean()) if len(frame) else 0.0,
        }
        for r in _bin_stats(frame, {"scope": "global", "direction": direction}):
            curves.append({
                "scope": "global",
                "direction": direction,
                "confidence_bin": r["confidence_bin"],
                "predicted_confidence": r["mean_confidence"],
                "actual_success": r["actual_success_rate"],
                "count": r["prediction_count"],
            })
    return pd.DataFrame(rows), pd.DataFrame(curves), summary


def _regime_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "high_vol": df["vol_bucket"].astype(str).eq("high"),
        "low_vol": df["vol_bucket"].astype(str).eq("low"),
        "vol_expansion": df["vol_regime"].astype(str).eq("vol_expansion"),
        "strong_uptrend": df["trend_regime"].astype(str).eq("strong_uptrend"),
        "strong_downtrend": df["trend_regime"].astype(str).eq("strong_downtrend"),
        "sideways": df["trend_regime"].astype(str).eq("sideways"),
        "false_high_signature": df["false_high_signature"].astype(bool),
        "entropy_spike": df["entropy_spike"].astype(bool),
        "confidence_overextension": df["confidence_overextension"].astype(float) > 0,
        "trend_transition": df["trend_transition"].astype(bool),
        "low_entropy": df["confidence_regime"].astype(str).eq("low_entropy"),
    }


def _calibration_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "mean_confidence": np.nan,
            "success_rate": np.nan,
            "calibration_error": np.nan,
            "brier_score": np.nan,
            "ece": np.nan,
            "overconfidence_frequency": np.nan,
            "high_confidence_failure_rate": np.nan,
            "confidence_reliability": np.nan,
        }
    conf = frame["predicted_confidence"].astype(float)
    success = frame["actual_success"].astype(int)
    cal_err = abs(float(conf.mean()) - float(success.mean()))
    return {
        "rows": int(len(frame)),
        "mean_confidence": float(conf.mean()),
        "success_rate": float(success.mean()),
        "calibration_error": cal_err,
        "brier_score": _brier(conf, success),
        "ece": _ece(conf, success),
        "overconfidence_frequency": float((conf > success).mean()),
        "high_confidence_failure_rate": float(((conf >= 0.55) & (success == 0)).mean()),
        "confidence_reliability": float(max(0.0, 1.0 - cal_err)),
    }


def phase2_regimes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for regime, mask in _regime_masks(df).items():
        sub = df[mask].copy()
        met = _calibration_metrics(sub)
        rows.append({"regime": regime, **met})
        sub2 = sub.copy()
        sub2["calibration_confidence"] = sub2["predicted_confidence"]
        sub2["calibration_success"] = sub2["actual_success"]
        for b in _bin_stats(sub2, {"scope": f"regime:{regime}", "direction": "ARGMAX"}):
            rows.append({
                "regime": f"{regime}:{b['confidence_bin']}",
                "rows": b["prediction_count"],
                "mean_confidence": b["mean_confidence"],
                "success_rate": b["actual_success_rate"],
                "calibration_error": b["calibration_error"],
                "brier_score": b["brier_score"],
                "ece": b["ece"],
                "overconfidence_frequency": np.nan,
                "high_confidence_failure_rate": np.nan,
                "confidence_reliability": np.nan,
            })
    base = pd.DataFrame(rows)
    top = (
        base[~base["regime"].str.contains(":", regex=False)]
        .sort_values(["high_confidence_failure_rate", "calibration_error"], ascending=False)
        .head(8)
        .to_dict(orient="records")
    )
    return base, {"worst_overconfidence_regimes": top}


def phase3_false_high(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    fh = df[df["false_high_signature"].astype(bool)].copy()
    fh["calibration_confidence"] = fh["p_long"].astype(float)
    fh["calibration_success"] = fh["actual_success"].astype(int)
    dist = pd.DataFrame(_bin_stats(fh, {"scope": "false_high", "direction": "LONG"}))
    report = {
        "rows": int(len(fh)),
        "confidence_distribution": fh["p_long"].describe().to_dict() if len(fh) else {},
        "actual_outcome_distribution": fh["actual_success"].value_counts().to_dict() if len(fh) else {},
        "mae_mean": _safe_mean(fh["mae"]) if len(fh) else 0.0,
        "mfe_mean": _safe_mean(fh["mfe"]) if len(fh) else 0.0,
        "rfe_rate": _safe_mean(fh["rfe_flag"].astype(float)) if len(fh) else 0.0,
        "drawdown_contribution_mdd": _mdd(fh["engine_ret"]) if len(fh) else 0.0,
        "confidence_vs_actual_success_gap": float(fh["p_long"].mean() - fh["actual_success"].mean()) if len(fh) else np.nan,
        "confidence_saturation_rate": float((fh["p_long"] >= 0.60).mean()) if len(fh) else 0.0,
        "margin_mean": _safe_mean(fh["margin"]) if len(fh) else 0.0,
        "margin_delta_mean": _safe_mean(fh["margin_delta"]) if len(fh) else 0.0,
        "probability_collapse_timing_proxy": {
            "p_long_delta_mean": _safe_mean(fh["p_long_delta"]) if len(fh) else 0.0,
            "entropy_delta_mean": _safe_mean(fh["entropy_delta"]) if len(fh) else 0.0,
            "negative_p_long_delta_rate": float((fh["p_long_delta"] < 0).mean()) if len(fh) else 0.0,
        },
    }
    return dist, report


def phase4_reliability(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scopes = {
        "global": pd.Series(True, index=df.index),
        "high_vol": df["vol_bucket"].astype(str).eq("high"),
        "false_high": df["false_high_signature"].astype(bool),
        "low_entropy": df["confidence_regime"].astype(str).eq("low_entropy"),
        "trend_transition": df["trend_transition"].astype(bool),
    }
    for scope, mask in scopes.items():
        sub = df[mask].copy()
        for direction in ["LONG", "SHORT", "FLAT"]:
            frame = _direction_frame(sub, direction)
            for r in _bin_stats(frame, {"scope": scope, "direction": direction}):
                rows.append({
                    "scope": scope,
                    "direction": direction,
                    "confidence_bin": r["confidence_bin"],
                    "predicted_confidence": r["mean_confidence"],
                    "actual_success": r["actual_success_rate"],
                    "count": r["prediction_count"],
                    "calibration_error": r["calibration_error"],
                })
    out = pd.DataFrame(rows)
    summary = (
        out.groupby(["scope", "direction"], dropna=False)
        .agg(total_count=("count", "sum"), mean_abs_calibration_error=("calibration_error", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    return out, {"reliability_summary": summary}


def _cluster_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "high_confidence_negative_expectancy": (df["predicted_confidence"] >= 0.55) & (df["engine_ret"] < 0),
        "low_entropy_high_mae": (df["entropy"] <= 0.90) & (df["mae"] <= -0.005),
        "strong_trend_reversal_collapse": df["trend_regime"].isin(["strong_uptrend", "strong_downtrend"]) & ((df["engine_ret"] < -0.003) | df["rfe_flag"].astype(bool)),
        "confidence_overextension": df["confidence_overextension"].astype(float) > 0,
        "volatility_expansion_trap": df["vol_regime"].eq("vol_expansion") & (df["engine_ret"] < 0),
        "long_bias_cluster": df["direction"].astype(str).eq("LONG") & (df["p_long"] >= 0.55) & (df["engine_ret"] < 0),
        "false_high_signature": df["false_high_signature"].astype(bool),
    }


def _persistence(mask: pd.Series) -> Dict[str, Any]:
    max_run = run = 0
    runs = []
    for val in mask.astype(bool).tolist():
        if val:
            run += 1
            max_run = max(max_run, run)
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return {"max_persistence": int(max_run), "mean_persistence": float(np.mean(runs)) if runs else 0.0}


def phase5_clusters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, mask in _cluster_masks(df).items():
        sub = df[mask].copy()
        pers = _persistence(mask)
        rows.append({
            "cluster": name,
            "rows": int(len(sub)),
            "frequency": float(len(sub) / max(len(df), 1)),
            "expectancy": _safe_mean(sub["engine_ret"]) if len(sub) else 0.0,
            "mdd_contribution": _mdd(sub["engine_ret"]) if len(sub) else 0.0,
            "rfe_rate": _safe_mean(sub["rfe_flag"].astype(float)) if len(sub) else 0.0,
            "transition_relationship": float((sub["trend_transition"].astype(bool) | (sub["vol_regime"] == "vol_expansion")).mean()) if len(sub) else 0.0,
            **pers,
            "mean_confidence": _safe_mean(sub["predicted_confidence"]) if len(sub) else 0.0,
            "success_rate": _safe_mean(sub["actual_success"]) if len(sub) else 0.0,
        })
    out = pd.DataFrame(rows).sort_values(["mdd_contribution", "frequency"], ascending=[True, False])
    report = {"top_failure_clusters": out.head(8).to_dict(orient="records")}
    return out, report


def _subset_calibration(df: pd.DataFrame, name: str, mask: pd.Series) -> Dict[str, Any]:
    sub = df[mask].copy()
    met = _calibration_metrics(sub)
    return {"subset": name, **met}


def phase6_executed_q2(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    q2_accepted = df["q2_bdi_scale"].astype(float) >= 0.40
    fh = df["false_high_signature"].astype(bool)
    high_conf = df["predicted_confidence"] >= 0.55
    rows = [
        _subset_calibration(df, "executed", df["is_executed"].astype(bool)),
        _subset_calibration(df, "non_executed_candidate", ~df["is_executed"].astype(bool)),
        _subset_calibration(df, "q2_filtered_accepted", q2_accepted),
        _subset_calibration(df, "q2_filtered_rejected", ~q2_accepted),
        _subset_calibration(df, "false_high_accepted", fh & q2_accepted),
        _subset_calibration(df, "false_high_rejected", fh & ~q2_accepted),
        _subset_calibration(df, "high_confidence_good_trades", high_conf & df["binary_good_trade"].astype(bool)),
        _subset_calibration(df, "high_confidence_catastrophic_trades", high_conf & ((df["mae"] <= -0.008) | df["rfe_flag"].astype(bool))),
    ]
    table = pd.DataFrame(rows)
    executed = table[table["subset"] == "executed"].iloc[0].to_dict()
    non_exec = table[table["subset"] == "non_executed_candidate"].iloc[0].to_dict()
    accepted = table[table["subset"] == "q2_filtered_accepted"].iloc[0].to_dict()
    rejected = table[table["subset"] == "q2_filtered_rejected"].iloc[0].to_dict()
    return (
        {
            "table": table.to_dict(orient="records"),
            "executed_vs_non_executed": {
                "executed_ece": executed.get("ece"),
                "non_executed_ece": non_exec.get("ece"),
                "executed_high_conf_failure_rate": executed.get("high_confidence_failure_rate"),
                "non_executed_high_conf_failure_rate": non_exec.get("high_confidence_failure_rate"),
            },
        },
        {
            "table": table.to_dict(orient="records"),
            "q2_reduces_calibration_error": bool(
                pd.notna(accepted.get("ece")) and pd.notna(rejected.get("ece")) and accepted.get("ece") < rejected.get("ece")
            ),
            "accepted_ece": accepted.get("ece"),
            "rejected_ece": rejected.get("ece"),
            "accepted_high_conf_failure_rate": accepted.get("high_confidence_failure_rate"),
            "rejected_high_conf_failure_rate": rejected.get("high_confidence_failure_rate"),
        },
    )


def phase7_loss_mapping(cluster_df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "high_confidence_negative_expectancy": ("confidence_calibration_loss", "Penalize confident wrong directional calls."),
        "low_entropy_high_mae": ("entropy_aware_loss", "Low entropy should not imply low drawdown risk."),
        "strong_trend_reversal_collapse": ("cost_sensitive_loss", "Charge more for reversal collapses in trend regimes."),
        "confidence_overextension": ("focal_loss", "Focus gradient on hard overextended confidence errors."),
        "volatility_expansion_trap": ("drawdown_weighted_loss", "Weight volatility expansion losses by MAE/MDD."),
        "long_bias_cluster": ("cost_sensitive_loss", "Counter LONG bias under negative expectancy."),
        "false_high_signature": ("false_high_penalty_loss", "Direct penalty for LONG/up/high-vol low-entropy failures."),
    }
    rows = []
    for _, r in cluster_df.iterrows():
        loss, rationale = mapping.get(r["cluster"], ("risk_weighted_loss", "Generic risk-aware weighting."))
        severity = float(abs(r["expectancy"]) * 1000 + r["rfe_rate"] + r["frequency"])
        rows.append({
            "cluster": r["cluster"],
            "recommended_loss": loss,
            "severity_score": severity,
            "cluster_rows": int(r["rows"]),
            "expectancy": r["expectancy"],
            "mdd_contribution": r["mdd_contribution"],
            "rfe_rate": r["rfe_rate"],
            "rationale": rationale,
        })
    return pd.DataFrame(rows).sort_values("severity_score", ascending=False)


def _readiness_verdict(
    global_summary: Dict[str, Any],
    regime_df: pd.DataFrame,
    false_high_report: Dict[str, Any],
    cluster_df: pd.DataFrame,
    q2_report: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    long_ece = global_summary.get("LONG", {}).get("ece", np.nan)
    fh_gap = false_high_report.get("confidence_vs_actual_success_gap", np.nan)
    clusters_defined = int((cluster_df["rows"] >= 20).sum()) if len(cluster_df) else 0
    worst_hcf = float(regime_df[~regime_df["regime"].str.contains(":", regex=False)]["high_confidence_failure_rate"].max())
    readiness = {
        "failure_cluster_sufficiently_defined": clusters_defined >= 3,
        "regime_calibration_understood": len(regime_df) >= 10,
        "loss_direction_decidable": clusters_defined >= 3 and (pd.notna(fh_gap) and fh_gap > 0),
        "label_contamination_understood": True,
        "q2_forensic_intelligence_transferable": bool(q2_report.get("q2_reduces_calibration_error", False)),
        "long_ece": long_ece,
        "false_high_confidence_gap": fh_gap,
        "worst_regime_high_conf_failure": worst_hcf,
        "promotion_ready": False,
    }
    if pd.notna(fh_gap) and fh_gap > 0.10:
        return "false_high_overconfidence_confirmed", readiness
    if pd.notna(long_ece) and long_ece > 0.10:
        return "tcn_confidence_calibration_failure_confirmed", readiness
    if readiness["loss_direction_decidable"] and readiness["q2_forensic_intelligence_transferable"]:
        return "risk_aware_retraining_ready", readiness
    if readiness["q2_forensic_intelligence_transferable"]:
        return "Q2_reduces_calibration_error", readiness
    if worst_hcf > 0.30:
        return "confidence_signal_not_reliable", readiness
    return "retraining_not_ready", readiness


def _write_md(path: Path, title: str, payload: Dict[str, Any]) -> None:
    path.write_text(
        f"# {title}\n\n"
        "Diagnostics only. Production/Q2_BDI unchanged; retraining not executed.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )


def run_calibration_forensics() -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _prepare_dataset()

    bin_df, curve_df, global_summary = phase1_global(df)
    bin_df.to_csv(OUT_DIR / "confidence_bin_statistics.csv", index=False)
    curve_df.to_csv(OUT_DIR / "calibration_curve.csv", index=False)
    _write_md(OUT_DIR / "global_confidence_calibration.md", "Global Confidence Calibration", global_summary)

    regime_df, regime_report = phase2_regimes(df)
    regime_df.to_csv(OUT_DIR / "regime_specific_calibration.csv", index=False)
    _write_md(OUT_DIR / "regime_overconfidence_analysis.md", "Regime Overconfidence Analysis", regime_report)

    fh_dist, fh_report = phase3_false_high(df)
    fh_dist.to_csv(OUT_DIR / "false_high_confidence_distribution.csv", index=False)
    _write_md(OUT_DIR / "false_high_calibration_forensics.md", "False High Calibration Forensics", fh_report)

    rel_df, rel_summary = phase4_reliability(df)
    rel_df.to_csv(OUT_DIR / "reliability_diagram_data.csv", index=False)
    _write_md(OUT_DIR / "reliability_summary.md", "Reliability Summary", rel_summary)

    cluster_df, cluster_report = phase5_clusters(df)
    cluster_df.to_csv(OUT_DIR / "confidence_failure_clusters.csv", index=False)
    _write_md(OUT_DIR / "cluster_forensics_report.md", "Cluster Forensics Report", cluster_report)

    exec_report, q2_report = phase6_executed_q2(df)
    _write_md(OUT_DIR / "executed_vs_candidate_calibration.md", "Executed vs Candidate Calibration", exec_report)
    _write_md(OUT_DIR / "q2_calibration_impact_analysis.md", "Q2 Calibration Impact Analysis", q2_report)

    loss_df = phase7_loss_mapping(cluster_df)
    loss_df.to_csv(OUT_DIR / "failure_cluster_loss_mapping.csv", index=False)
    _write_md(
        OUT_DIR / "risk_aware_loss_design_report.md",
        "Risk-Aware Loss Design Report",
        {
            "recommended_losses": loss_df.head(10).to_dict(orient="records"),
            "retraining_executed": False,
        },
    )

    verdict, readiness = _readiness_verdict(global_summary, regime_df, fh_report, cluster_df, q2_report)
    _write_md(
        OUT_DIR / "tcn_retraining_readiness.md",
        "TCN Retraining Readiness",
        {
            "final_verdict": verdict,
            "readiness": readiness,
            "allowed_state": "diagnostic_only",
            "promotion_ready": False,
        },
    )
    return {
        "verdict": verdict,
        "rows": int(len(df)),
        "long_ece": global_summary.get("LONG", {}).get("ece"),
        "false_high_rows": fh_report.get("rows"),
        "false_high_gap": fh_report.get("confidence_vs_actual_success_gap"),
        "q2_reduces_calibration_error": q2_report.get("q2_reduces_calibration_error"),
        "promotion_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TCN confidence calibration forensics")
    parser.parse_args()
    result = run_calibration_forensics()
    print(f"verdict: {result['verdict']}")
    print(f"rows: {result['rows']}")
    print(f"long_ece: {result['long_ece']:.4f}")
    print(f"false_high_rows: {result['false_high_rows']}")


if __name__ == "__main__":
    main()
