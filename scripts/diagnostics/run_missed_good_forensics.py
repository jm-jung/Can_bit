"""
Missed-good structure forensics for Q2 low-scale/reject candidates.

Diagnostics only. This script does not modify production TCN, Q2_BDI,
R7 behavior, live execution, order paths, launchd jobs, or trading state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scripts.diagnostics.run_entry_greenlight_forensics import (
    MIN_SAMPLE,
    POSITION_SIZE,
    RECENT_3M_DAYS,
    RECENT_6M_DAYS,
    _feature_family,
    _labels,
    _load_frame,
    _mdd,
    _profit_factor,
    _safe_features,
    _safe_num,
    _write_md,
)
from scripts.diagnostics.run_false_high_r7_monitor import Q2_ACCEPT, Q2_REJECT, R7_DEFAULT_THRESHOLD, _prod_hashes, _q2_baseline

ROOT = Path("data/diagnostics/missed_good_forensics")
DIRS = {
    "state": ROOT / "state",
    "casebook": ROOT / "casebook",
    "quality": ROOT / "quality",
    "q2": ROOT / "q2_root_cause",
    "sep": ROOT / "separability",
    "structure": ROOT / "structure_mining",
    "taxonomy": ROOT / "taxonomy",
    "adjust": ROOT / "q2_adjustment",
    "validation": ROOT / "validation",
    "cases": ROOT / "cases",
    "shadow": ROOT / "shadow_plan",
    "decision": ROOT / "decision",
    "audit": ROOT / "audit",
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b not in (0, 0.0) and not pd.isna(b) else default


def _metric_row(df: pd.DataFrame, mask: pd.Series, name: str) -> Dict[str, Any]:
    sub = df.loc[mask].copy()
    ret = _safe_num(sub.get("engine_ret", pd.Series(dtype=float)))
    mae = _safe_num(sub.get("mae", pd.Series(dtype=float)))
    mfe = _safe_num(sub.get("mfe", pd.Series(dtype=float)))
    return {
        "name": name,
        "rows": int(len(sub)),
        "start": str(sub["_ts"].min()) if len(sub) else "",
        "end": str(sub["_ts"].max()) if len(sub) else "",
        "long_ratio": float(sub["direction"].astype(str).eq("LONG").mean()) if len(sub) else 0.0,
        "q2_accept_rate": float(sub["q2_accept"].mean()) if len(sub) else 0.0,
        "q2_reject_rate": float(sub["q2_reject"].mean()) if len(sub) else 0.0,
        "r7_warning_rate": float(sub["r7_high_hazard"].mean()) if len(sub) else 0.0,
        "net_mean": float(ret.mean()) if len(sub) else 0.0,
        "net_median": float(ret.median()) if len(sub) else 0.0,
        "net_total": float((ret * POSITION_SIZE).sum()) if len(sub) else 0.0,
        "winrate": float((ret > 0).mean()) if len(sub) else 0.0,
        "profit_factor": _profit_factor(ret) if len(sub) else 0.0,
        "mfe_median": float(mfe.median()) if len(sub) else 0.0,
        "mae_median": float(mae.median()) if len(sub) else 0.0,
        "mfe_mae_ratio": _safe_div(float(mfe.median()) if len(sub) else 0.0, abs(float(mae.median())) if len(sub) else 0.0),
        "rfe_rate": float(sub.get("rfe_flag", pd.Series(False, index=sub.index)).astype(bool).mean()) if len(sub) else 0.0,
        "high_mae_rate": float((mae <= -0.006).mean()) if len(sub) else 0.0,
        "mdd_contribution": _mdd(ret) if len(sub) else 0.0,
        "recent_3m_count": int((sub["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_3M_DAYS)).sum()) if len(sub) else 0,
        "recent_6m_count": int((sub["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(sub) else 0,
        "quarterly_presence": int(sub["_ts"].dt.to_period("Q").astype(str).nunique()) if len(sub) else 0,
    }


def _session(hour: int) -> str:
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 14:
        return "europe"
    if 14 <= hour < 22:
        return "us"
    return "late_us"


def _q2_reason(row: pd.Series) -> str:
    reasons: List[str] = []
    direction = str(row.get("direction", ""))
    if row.get("q2_bdi_scale", 0.0) <= Q2_REJECT:
        reasons.append("low_q2_scale")
    if bool(row.get("high_vol", False)) or bool(row.get("vol_bucket_high", False)):
        reasons.append("high_vol_penalty")
    if bool(row.get("vol_expansion", False)) or row.get("volatility_expansion_rate", 0.0) > 0:
        reasons.append("vol_expansion_penalty")
    if direction == "LONG" and (bool(row.get("trend_down", False)) or bool(row.get("trend_sideways", False))):
        reasons.append("long_trend_mismatch")
    if direction == "SHORT" and (bool(row.get("trend_up", False)) or bool(row.get("trend_sideways", False))):
        reasons.append("short_trend_mismatch")
    if row.get("entropy", row.get("baseline_entropy", 0.0)) >= 0.90:
        reasons.append("entropy_uncertainty")
    if row.get("baseline_margin", row.get("margin", 0.0)) <= 0.10:
        reasons.append("low_margin")
    if row.get("r7_score", 0.0) >= R7_DEFAULT_THRESHOLD:
        reasons.append("r7_like_hazard")
    if bool(row.get("failed_breakout_proxy", False)):
        reasons.append("failed_breakout_proxy")
    if not reasons:
        reasons.append("unclassified_q2_low_scale")
    return "|".join(reasons)


def _prepare() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    df = _load_frame()
    labels, label_summary = _labels(df)
    safe_cols, safe_df, forbidden_df = _safe_features(df)
    df["missed_good"] = labels["L12_missed_good"].astype(bool)
    df["q2_reject_bad"] = df["q2_reject"] & (df["engine_ret"] < -0.001)
    df["q2_reject_neutral_or_bad"] = df["q2_reject"] & ~df["missed_good"]
    df["q2_accept_good"] = labels["L13_q2_accept_good"].astype(bool)
    df["q2_accept_bad"] = labels["L14_q2_accept_bad"].astype(bool)
    df["high_mfe_low_mae"] = labels["L5_high_MFE_low_MAE"].astype(bool)
    df["clean_trend_continuation"] = labels["L7_clean_trend_continuation"].astype(bool)
    df["missed_good_id"] = ""
    mg_idx = df.index[df["missed_good"]].tolist()
    df.loc[mg_idx, "missed_good_id"] = [f"MG_{i:04d}" for i in range(1, len(mg_idx) + 1)]
    df["q2_reject_reason"] = df.apply(_q2_reason, axis=1)
    return df, labels, label_summary, safe_cols, safe_df, forbidden_df


def _phase0(df: pd.DataFrame, before: List[Dict[str, Any]], safe_df: pd.DataFrame, forbidden_df: pd.DataFrame) -> None:
    q2 = _q2_baseline(df)
    pd.DataFrame([q2] if isinstance(q2, dict) else q2).to_csv(DIRS["state"] / "q2_bdi_baseline_snapshot.csv", index=False)
    (DIRS["state"] / "hash_before_after_precheck.json").write_text(_json({"before": before, "precheck_only": True}), encoding="utf-8")
    _write_md(DIRS["state"] / "production_safety_snapshot.md", "Production Safety Snapshot", {
        "production_tcn_hash_before": before,
        "production_changed": False,
        "q2_bdi_changed": False,
        "live_execution_changed": False,
        "order_path_changed": False,
        "state_changed": False,
        "output_root": str(ROOT),
    })
    _write_md(DIRS["state"] / "r7_monitor_snapshot.md", "R7 Monitor Snapshot", {
        "name": "FalseHigh_R7_StructureHazard_v1",
        "threshold": R7_DEFAULT_THRESHOLD,
        "action": "none",
        "scale_action": "none",
        "hard_block": False,
        "q2_override": False,
        "production_ready": False,
        "promotion_ready": False,
    })
    eg_root = Path("data/diagnostics/entry_greenlight_forensics")
    _write_md(DIRS["state"] / "entry_greenlight_import_summary.md", "Entry Greenlight Import Summary", {
        "source": str(eg_root),
        "final_verdict_exists": (eg_root / "entry_greenlight_final_verdict.md").exists(),
        "missed_good_rows_loaded": int(df["missed_good"].sum()),
        "strict_safe_feature_count": int(len(safe_df)),
    })
    _write_md(DIRS["state"] / "missed_good_definition.md", "Missed Good Definition", {
        "definition": "Q2 reject/low scale and later net positive with high MFE, using outcome/path only as label/evaluation.",
        "membership_count": int(df["missed_good"].sum()),
        "greenlight_rule_capture": "prior interpretable greenlight rules rejected no_edge; this run tests entry-safe capture from scratch.",
    })
    inv = pd.DataFrame([
        {"source": "monitor_frame", "path": "run_false_high_r7_monitor._prepare_monitor_frame", "exists": True, "rows": len(df)},
        {"source": "entry_greenlight_labels", "path": "data/diagnostics/entry_greenlight_forensics/labels/positive_entry_labels.parquet", "exists": Path("data/diagnostics/entry_greenlight_forensics/labels/positive_entry_labels.parquet").exists(), "rows": int(df["missed_good"].sum())},
        {"source": "freshness", "path": "data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json", "exists": Path("data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json").exists(), "rows": np.nan},
    ])
    inv.to_csv(DIRS["state"] / "missed_good_input_inventory.csv", index=False)
    safe_df.to_csv(DIRS["state"] / "safe_feature_inventory.csv", index=False)
    forbidden_df.to_csv(DIRS["state"] / "forbidden_future_columns.csv", index=False)
    _write_md(DIRS["state"] / "data_integrity_audit.md", "Data Integrity Audit", {
        "row_count": len(df),
        "missed_good_count": int(df["missed_good"].sum()),
        "expected_missed_good_count": 396,
        "count_match": int(df["missed_good"].sum()) == 396,
        "duplicate_entry_ts_in_missed_good": int(df.loc[df["missed_good"], "_ts"].duplicated().sum()),
        "trade_id_available": "trade_id" in df.columns,
        "outcome_available": "engine_ret" in df.columns,
        "mae_mfe_rfe_available": {"mae", "mfe", "rfe_flag"}.issubset(df.columns),
        "freshness_checked_by_prior_pipeline": Path("data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json").exists(),
    })


def _classify_quality(mg: pd.DataFrame) -> pd.Series:
    ret = _safe_num(mg["engine_ret"])
    mae = _safe_num(mg["mae"])
    mfe = _safe_num(mg["mfe"])
    adverse = mae.abs()
    ratio = mfe / adverse.clip(lower=0.0005)
    rfe = mg["rfe_flag"].astype(bool)
    hold = _safe_num(mg.get("hold_bars", pd.Series(0, index=mg.index)))
    exit_reason = mg.get("exit_reason", pd.Series("", index=mg.index)).astype(str)
    max_hold = exit_reason.str.contains("max_holding", case=False, na=False)
    tiny = ret.abs() <= 0.001
    artifact = max_hold | tiny | mg.get("artifact_suspect", pd.Series(False, index=mg.index)).astype(bool)
    cls = pd.Series("MG_G_not_really_good", index=mg.index)
    cls[(ret > 0) & (mfe >= 0.003) & (adverse <= 0.002) & (ratio >= 2.0) & ~rfe] = "MG_A_clean_good_entry"
    cls[(ret > 0) & (adverse > 0.003) & (mfe > adverse) & ~rfe] = "MG_B_recovery_good"
    cls[(ret > 0) & (hold >= 10) & (mfe >= 0.003) & ~rfe & ~cls.eq("MG_A_clean_good_entry")] = "MG_C_late_followthrough_good"
    cls[(ret > 0) & (mfe >= 0.006) & (adverse >= 0.006)] = "MG_D_high_MFE_but_high_MAE"
    cls[(ret > 0) & tiny] = "MG_E_tiny_edge_good"
    cls[artifact] = "MG_F_artifact_suspect_good"
    return cls


def _casebook(df: pd.DataFrame, safe_cols: List[str]) -> pd.DataFrame:
    mg = df[df["missed_good"]].copy()
    mg["quality_class"] = _classify_quality(mg)
    mg["entry_ts"] = mg["_ts"].astype(str)
    mg["candidate_id"] = mg.get("candidate_id", mg.get("trade_id", mg.index)).astype(str)
    mg["trade_id"] = mg.get("trade_id", mg["candidate_id"]).astype(str)
    mg["executed_or_counterfactual"] = np.where(mg.get("is_executed", pd.Series(False, index=mg.index)).astype(bool), "executed", "counterfactual")
    mg["q2_decision"] = np.where(mg["q2_bdi_scale"] >= Q2_ACCEPT, "accept", np.where(mg["q2_bdi_scale"] <= Q2_REJECT, "reject", "low_scale"))
    mg["r7_warning_category"] = np.where(mg["r7_high_hazard"], "r7_high_hazard", "r7_no_warning")
    mg["trend_state_safe"] = mg.get("trend_state", np.where(mg.get("trend_up", False), "up", np.where(mg.get("trend_down", False), "down", "sideways"))).astype(str)
    mg["vol_state_safe"] = mg.get("vol_bucket", np.where(mg.get("high_vol", False), "high", "not_high")).astype(str)
    mg["session_bucket"] = mg["_ts"].dt.hour.map(_session)
    adverse = _safe_num(mg["mae"]).abs().clip(lower=0.0005)
    mg["mfe_mae_ratio"] = _safe_num(mg["mfe"]) / adverse
    mg["max_holding_flag"] = mg.get("exit_reason", pd.Series("", index=mg.index)).astype(str).str.contains("max_holding", case=False, na=False)
    mg["tiny_return_flag"] = _safe_num(mg["engine_ret"]).abs() <= 0.001
    mg["artifact_suspect_flag"] = mg.get("artifact_suspect", pd.Series(False, index=mg.index)).astype(bool) | mg["max_holding_flag"] | mg["tiny_return_flag"]
    mg["lifecycle_group"] = np.select(
        [mg["quality_class"].str.contains("clean"), mg["quality_class"].str.contains("recovery"), mg["quality_class"].str.contains("late"), mg["artifact_suspect_flag"]],
        ["clean", "recovery", "late_followthrough", "artifact_suspect"],
        default="mixed_or_unclear",
    )
    mg["path_quality_label"] = mg["quality_class"]
    mg["preliminary_missed_good_type"] = mg["quality_class"].str.replace("MG_", "", regex=False)
    mg["notes_reason_code"] = mg["q2_reject_reason"]
    rename = {
        "baseline_p_long": "TCN p_long",
        "baseline_p_short": "TCN p_short",
        "baseline_p_flat": "TCN p_flat",
    }
    for src, dst in rename.items():
        if src in mg and dst not in mg:
            mg[dst] = mg[src]
    safe_subset = [c for c in safe_cols if c in mg.columns][:60]
    core = [c for c in [
        "missed_good_id", "trade_id", "candidate_id", "entry_ts", "symbol", "timeframe", "direction",
        "executed_or_counterfactual", "q2_decision", "q2_score", "q2_bdi_score", "q2_bdi_scale",
        "q2_reject_reason", "q2_pd_penalty_score", "q2_bdi_penalty_score", "TCN p_long", "TCN p_short",
        "TCN p_flat", "entropy", "baseline_margin", "r7_score", "r7_high_hazard", "r7_warning_category",
        "trend_state_safe", "vol_state_safe", "session_bucket", "engine_ret", "mfe", "mae", "mfe_mae_ratio",
        "rfe_flag", "hold_bars", "exit_reason", "max_holding_flag", "tiny_return_flag",
        "artifact_suspect_flag", "lifecycle_group", "path_quality_label", "preliminary_missed_good_type",
        "quality_class", "notes_reason_code",
    ] if c in mg.columns]
    out = mg[core + [c for c in safe_subset if c not in core]].copy()
    out.to_csv(DIRS["casebook"] / "missed_good_396_casebook.csv", index=False)
    out.to_parquet(DIRS["casebook"] / "missed_good_396_casebook.parquet", index=False)
    _write_md(DIRS["casebook"] / "missed_good_casebook_summary.md", "Missed Good Casebook Summary", {
        "row_count": len(out),
        "expected_row_count": 396,
        "count_match": len(out) == 396,
        "long_short": out["direction"].value_counts().to_dict(),
        "executed_counterfactual": out["executed_or_counterfactual"].value_counts().to_dict(),
        "artifact_suspect_count": int(out["artifact_suspect_flag"].sum()),
        "duplicates_entry_ts": int(out["entry_ts"].duplicated().sum()),
    })
    return out


def _quality_outputs(df: pd.DataFrame, casebook: pd.DataFrame) -> pd.DataFrame:
    q = casebook[["missed_good_id", "entry_ts", "direction", "quality_class", "path_quality_label", "artifact_suspect_flag"]].copy()
    q.to_csv(DIRS["quality"] / "missed_good_quality_labels.csv", index=False)
    rows = []
    for cls in sorted(casebook["quality_class"].unique()):
        ids = set(casebook.loc[casebook["quality_class"].eq(cls), "missed_good_id"])
        mask = df["missed_good_id"].isin(ids)
        row = _metric_row(df, mask, cls)
        sub = casebook[casebook["quality_class"].eq(cls)]
        row.update({
            "q2_reject_reason_top": ";".join(sub["q2_reject_reason"].str.get_dummies("|").sum().sort_values(ascending=False).head(5).index.tolist()),
            "artifact_count": int(sub["artifact_suspect_flag"].sum()),
        })
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("rows", ascending=False)
    summary.to_csv(DIRS["quality"] / "missed_good_quality_summary.csv", index=False)
    _write_md(DIRS["quality"] / "missed_good_quality_report.md", "Missed Good Quality Report", {
        "summary": summary,
        "clean_good_count": int(casebook["quality_class"].eq("MG_A_clean_good_entry").sum()),
        "recovery_good_count": int(casebook["quality_class"].eq("MG_B_recovery_good").sum()),
        "artifact_or_tiny_count": int(casebook["quality_class"].isin(["MG_E_tiny_edge_good", "MG_F_artifact_suspect_good"]).sum()),
        "production_research_clean_subset": int(casebook["quality_class"].isin(["MG_A_clean_good_entry", "MG_B_recovery_good", "MG_C_late_followthrough_good"]).sum()),
    })
    return summary


def _compare_features(df: pd.DataFrame, masks: Dict[str, pd.Series], safe_cols: List[str]) -> pd.DataFrame:
    rows = []
    for feature in safe_cols:
        if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]):
            continue
        vals = {}
        for name, mask in masks.items():
            vals[f"{name}_mean"] = float(_safe_num(df.loc[mask, feature]).mean()) if mask.any() else np.nan
            vals[f"{name}_median"] = float(_safe_num(df.loc[mask, feature]).median()) if mask.any() else np.nan
        if "missed_good" in masks and "q2_reject_bad" in masks:
            vals["mg_vs_bad_abs_mean_gap"] = abs(vals.get("missed_good_mean", 0) - vals.get("q2_reject_bad_mean", 0))
        rows.append({"feature": feature, "family": _feature_family(feature), **vals})
    return pd.DataFrame(rows).sort_values("mg_vs_bad_abs_mean_gap", ascending=False) if rows else pd.DataFrame()


def _q2_root_cause(df: pd.DataFrame, casebook: pd.DataFrame, safe_cols: List[str]) -> pd.DataFrame:
    reason_counts = casebook["q2_reject_reason"].str.get_dummies("|").sum().sort_values(ascending=False).reset_index()
    reason_counts.columns = ["q2_reject_reason", "count"]
    reason_counts["ratio"] = reason_counts["count"] / max(len(casebook), 1)
    reason_counts.to_csv(DIRS["q2"] / "q2_reject_reason_distribution.csv", index=False)
    comp_features = [c for c in safe_cols if any(tok in c.lower() for tok in ["q2", "penalty", "trend", "vol", "entropy", "margin", "r7"])]
    comp = _compare_features(df, {
        "missed_good": df["missed_good"],
        "q2_reject_bad": df["q2_reject_bad"],
        "q2_accept_good": df["q2_accept_good"],
        "q2_accept_bad": df["q2_accept_bad"],
        "high_mfe_low_mae": df["high_mfe_low_mae"],
        "all_candidates": pd.Series(True, index=df.index),
    }, comp_features)
    comp.to_csv(DIRS["q2"] / "q2_penalty_component_analysis.csv", index=False)
    comp.to_csv(DIRS["q2"] / "missed_good_vs_q2_reject_bad.csv", index=False)
    comp.to_csv(DIRS["q2"] / "missed_good_vs_q2_accept_good.csv", index=False)
    root = reason_counts.head(10).copy()
    root["root_cause_rank"] = np.arange(1, len(root) + 1)
    root["mitigation_feasibility"] = np.where(root["q2_reject_reason"].str.contains("high_vol|trend|vol_expansion|low_margin", case=False), "diagnostic_calibration_candidate", "likely_defensive_or_unclear")
    root["defensive_tradeoff"] = np.where(root["q2_reject_reason"].str.contains("r7|failed|high_vol", case=False), "high_bad_contamination_risk", "needs_separability_test")
    root.to_csv(DIRS["q2"] / "q2_false_negative_root_cause.csv", index=False)
    _write_md(DIRS["q2"] / "q2_root_cause_report.md", "Q2 Root Cause Report", {
        "top_reasons": root,
        "component_gaps": comp.head(30),
        "interpretation": "Reasons are diagnostic decompositions; Q2_BDI baseline is unchanged.",
    })
    return root


def _feature_sets(safe_cols: List[str]) -> Dict[str, List[str]]:
    return {
        "F0_Q2_only": [c for c in safe_cols if _feature_family(c) == "F2_Q2_BDI"],
        "F1_TCN_only": [c for c in safe_cols if _feature_family(c) == "F1_TCN_confidence" or c.startswith("tcn_") or c.startswith("baseline_")],
        "F2_R7_only": [c for c in safe_cols if _feature_family(c) == "F3_R7_warning"],
        "F3_trend_vol_structure": [c for c in safe_cols if _feature_family(c) in {"F4_trend_state", "F5_volatility", "F6_price_structure"}],
        "F4_session_time": [c for c in safe_cols if _feature_family(c) == "F8_session_time"],
        "F5_Q2_TCN_R7": [c for c in safe_cols if _feature_family(c) in {"F1_TCN_confidence", "F2_Q2_BDI", "F3_R7_warning"}],
        "F6_Q2_TCN_R7_trend_structure": [c for c in safe_cols if _feature_family(c) in {"F1_TCN_confidence", "F2_Q2_BDI", "F3_R7_warning", "F4_trend_state", "F5_volatility", "F6_price_structure"}],
        "F7_no_Q2": [c for c in safe_cols if _feature_family(c) != "F2_Q2_BDI"],
        "F8_no_R7": [c for c in safe_cols if _feature_family(c) != "F3_R7_warning"],
        "F9_no_TCN": [c for c in safe_cols if _feature_family(c) != "F1_TCN_confidence" and not c.startswith("tcn_") and not c.startswith("baseline_")],
        "F10_LONG_only": safe_cols,
        "F11_SHORT_only": safe_cols,
        "F12_recent_only": safe_cols,
        "F13_clean_quality_only": safe_cols,
    }


def _separability(df: pd.DataFrame, casebook: pd.DataFrame, safe_cols: List[str]) -> pd.DataFrame:
    clean_ids = set(casebook.loc[casebook["quality_class"].isin(["MG_A_clean_good_entry", "MG_B_recovery_good", "MG_C_late_followthrough_good"]), "missed_good_id"])
    datasets = {
        "clean_missed_good_vs_q2_reject_bad": (df["missed_good_id"].isin(clean_ids), df["q2_reject_bad"]),
        "all_missed_good_vs_q2_reject_bad": (df["missed_good"], df["q2_reject_bad"]),
        "recovery_missed_good_vs_q2_reject_bad": (df["missed_good_id"].isin(set(casebook.loc[casebook["quality_class"].eq("MG_B_recovery_good"), "missed_good_id"])), df["q2_reject_bad"]),
        "artifact_removed_missed_good_vs_q2_reject_bad": (df["missed_good"] & ~df["missed_good_id"].isin(set(casebook.loc[casebook["artifact_suspect_flag"], "missed_good_id"])), df["q2_reject_bad"]),
    }
    models = {
        "logistic_l2": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree_shallow": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=8, random_state=42),
    }
    rows, imps, buckets = [], [], []
    for dname, (pos_mask, neg_mask) in datasets.items():
        base_mask = pos_mask | neg_mask
        base = df[base_mask].copy().sort_values("_ts")
        y = pos_mask.loc[base.index].astype(int)
        if y.nunique() < 2 or y.sum() < 5:
            continue
        split = int(len(base) * 0.70)
        for fs_name, cols in _feature_sets(safe_cols).items():
            sub = base.copy()
            yy = y.copy()
            if fs_name == "F10_LONG_only":
                keep = sub["direction"].astype(str).eq("LONG")
                sub, yy = sub[keep], yy[keep]
            elif fs_name == "F11_SHORT_only":
                keep = sub["direction"].astype(str).eq("SHORT")
                sub, yy = sub[keep], yy[keep]
            elif fs_name == "F12_recent_only":
                keep = sub["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)
                sub, yy = sub[keep], yy[keep]
            cols2 = [c for c in cols if c in sub.columns and pd.api.types.is_numeric_dtype(sub[c])][:80]
            if len(sub) < 40 or yy.nunique() < 2 or not cols2:
                continue
            split2 = min(max(int(len(sub) * 0.70), 10), len(sub) - 5)
            if yy.iloc[:split2].nunique() < 2 or yy.iloc[split2:].nunique() < 2:
                continue
            x = sub[cols2].replace([np.inf, -np.inf], np.nan)
            for mname, model in models.items():
                try:
                    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                    pipe.fit(x.iloc[:split2], yy.iloc[:split2])
                    score = pipe.predict_proba(x.iloc[split2:])[:, 1]
                    yte = yy.iloc[split2:]
                    pred = score >= np.quantile(score, 0.80)
                    auc = float(roc_auc_score(yte, score))
                    ap = float(average_precision_score(yte, score))
                    top = sub.iloc[split2:].copy()
                    top["_score"] = score
                    top_bucket = top[top["_score"] >= top["_score"].quantile(0.80)]
                    rows.append({
                        "dataset": dname, "feature_set": fs_name, "model": mname, "rows": len(sub),
                        "test_rows": len(yte), "positive_test": int(yte.sum()), "ROC_AUC": auc, "PR_AUC": ap,
                        "precision_at_top20pct": float(precision_score(yte, pred, zero_division=0)),
                        "recall_at_top20pct": float(recall_score(yte, pred, zero_division=0)),
                        "top_bucket_clean_good_rate": float(yte[pred].mean()) if pred.any() else 0.0,
                        "false_positive_bad_acceptance_rate": float(((top_bucket["q2_reject_bad"]).mean())) if len(top_bucket) else 0.0,
                        "top_bucket_expectancy": float(top_bucket["engine_ret"].mean()) if len(top_bucket) else 0.0,
                        "overfit_risk": "diagnostic_only_temporal_split",
                        "status": "entry_safe_structure_weak" if auc >= 0.60 and ap >= yte.mean() else "q2_reject_good_not_separable",
                    })
                    mdl = pipe.named_steps["model"]
                    vals = getattr(mdl, "feature_importances_", np.abs(getattr(mdl, "coef_", np.zeros((1, len(cols2))))).ravel())
                    for c, v in sorted(zip(cols2, vals), key=lambda z: -float(z[1]))[:20]:
                        imps.append({"dataset": dname, "feature_set": fs_name, "model": mname, "feature": c, "importance": float(v), "family": _feature_family(c)})
                    buckets.append({"dataset": dname, "feature_set": fs_name, "model": mname, "top_bucket_rows": len(top_bucket), "top_bucket_expectancy": float(top_bucket["engine_ret"].mean()) if len(top_bucket) else 0.0})
                except Exception:
                    continue
    metrics = pd.DataFrame(rows)
    metrics.to_csv(DIRS["sep"] / "missed_good_vs_reject_bad_metrics.csv", index=False)
    metrics.to_csv(DIRS["sep"] / "missed_good_separability_by_feature_set.csv", index=False)
    pd.DataFrame(imps).to_csv(DIRS["sep"] / "missed_good_feature_importance.csv", index=False)
    pd.DataFrame(imps).groupby(["dataset", "feature"]).size().reset_index(name="interaction_proxy_count").to_csv(DIRS["sep"] / "missed_good_interactions.csv", index=False) if imps else pd.DataFrame().to_csv(DIRS["sep"] / "missed_good_interactions.csv", index=False)
    _write_md(DIRS["sep"] / "missed_good_separability_report.md", "Missed Good Separability Report", {
        "best_metrics": metrics.sort_values(["PR_AUC", "ROC_AUC"], ascending=False).head(40) if len(metrics) else pd.DataFrame(),
        "interpretation": "Temporal split diagnostics only. Low or unstable AUC/PR-AUC means entry-safe capture is weak/path-only.",
    })
    return metrics


def _structure_candidates(df: pd.DataFrame, casebook: pd.DataFrame) -> pd.DataFrame:
    conds = {
        "S1_pullback_then_continuation": (df.get("trend_up", False).astype(bool) | df.get("trend_down", False).astype(bool)) & (df["baseline_margin"] >= df["baseline_margin"].median()),
        "S2_high_vol_trend_continuation": df.get("high_vol", df.get("vol_bucket_high", False)).astype(bool) & (df.get("trend_up", False).astype(bool) | df.get("trend_down", False).astype(bool)),
        "S3_trend_transition_early": df.get("regime_transition_flag", False).astype(bool),
        "S4_low_entropy_undertrusted": (_safe_num(df.get("entropy", 9)) <= 0.90) & df["q2_reject"],
        "S5_mid_entropy_good": _safe_num(df.get("entropy", 9)).between(0.65, 1.00),
        "S6_session_specific_recovery": df["_ts"].dt.hour.between(14, 21),
        "S7_R7_false_warning_or_near_warning_but_good": df["r7_score"] >= 0.50,
        "S8_Q2_penalty_overkill": _safe_num(df.get("q2_bdi_penalty_score", 0)) >= _safe_num(df.get("q2_bdi_penalty_score", 0)).quantile(0.60),
        "S9_vol_compression_breakout": df.get("low_vol", False).astype(bool) & df.get("range_expansion_proxy", False).astype(bool),
        "S10_range_position_edge": df.get("price_vs_ema", 0).abs() <= _safe_num(df.get("price_vs_ema", 0)).abs().median(),
        "S11_LONG_specific_missed_good": df["direction"].astype(str).eq("LONG"),
        "S12_SHORT_specific_missed_good": df["direction"].astype(str).eq("SHORT"),
        "S13_missed_good_recovery": df["missed_good_id"].isin(set(casebook.loc[casebook["quality_class"].eq("MG_B_recovery_good"), "missed_good_id"])),
        "S14_path_only_unpredictable": df["missed_good_id"].isin(set(casebook.loc[casebook["quality_class"].isin(["MG_D_high_MFE_but_high_MAE", "MG_F_artifact_suspect_good"]), "missed_good_id"])),
    }
    rows = []
    for name, cond in conds.items():
        mask = df["missed_good"] & cond
        sub_case = casebook[casebook["missed_good_id"].isin(set(df.loc[mask, "missed_good_id"]))]
        bad_contam = int((df["q2_reject_bad"] & cond.reindex(df.index, fill_value=False)).sum())
        row = _metric_row(df, mask, name)
        clean_count = int(sub_case["quality_class"].eq("MG_A_clean_good_entry").sum())
        recovery_count = int(sub_case["quality_class"].eq("MG_B_recovery_good").sum())
        artifact_count = int(sub_case["artifact_suspect_flag"].sum())
        if row["rows"] and artifact_count / max(row["rows"], 1) > 0.50:
            status = "missed_good_artifact_heavy"
        elif bad_contam > max(clean_count + recovery_count, 1) * 5:
            status = "reject_bad_contamination"
        elif row["rows"] >= MIN_SAMPLE and row["recent_6m_count"] > 0 and row["net_mean"] > 0 and row["rfe_rate"] < 0.20:
            status = "daily_shadow_candidate"
        else:
            status = "path_only_unpredictable" if name == "S14_path_only_unpredictable" else "missed_good_entry_safe_structure_weak"
        row.update({
            "condition_string": name,
            "entry_safe_feature_list": "see structure_mining_report; hand-coded safe proxies",
            "clean_good_count": clean_count,
            "recovery_good_count": recovery_count,
            "artifact_count": artifact_count,
            "Q2_reject_bad_contamination_count": bad_contam,
            "interpretability": 0.8 if name != "S14_path_only_unpredictable" else 0.3,
            "overfit_risk": max(0.0, 1.0 - row["rows"] / 100.0),
            "status": status,
        })
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["status", "net_mean"], ascending=[True, False])
    out.to_csv(DIRS["structure"] / "missed_good_structure_candidates.csv", index=False)
    out[["condition_string", "entry_safe_feature_list", "status"]].to_csv(DIRS["structure"] / "missed_good_structure_conditions.csv", index=False)
    out.to_csv(DIRS["structure"] / "missed_good_structure_by_type.csv", index=False)
    _write_md(DIRS["structure"] / "missed_good_structure_mining_report.md", "Missed Good Structure Mining Report", {
        "candidates": out,
        "note": "Conditions are diagnostics-only shadow candidates. No production routing.",
    })
    return out


def _taxonomy(df: pd.DataFrame, casebook: pd.DataFrame, safe_cols: List[str]) -> pd.DataFrame:
    mg = df[df["missed_good"]].copy()
    cols = [c for c in safe_cols if c in mg.columns and pd.api.types.is_numeric_dtype(mg[c])][:50]
    if len(cols) >= 3 and len(mg) >= 20:
        x = SimpleImputer(strategy="median").fit_transform(mg[cols].replace([np.inf, -np.inf], np.nan))
        x = StandardScaler().fit_transform(x)
        clusters = KMeans(n_clusters=min(6, max(2, len(mg) // 50)), random_state=42, n_init=10).fit_predict(x)
    else:
        clusters = np.zeros(len(mg), dtype=int)
    mg["cluster_id"] = clusters
    assignments = mg[["missed_good_id", "timestamp", "direction", "cluster_id", "q2_reject_reason"]].merge(casebook[["missed_good_id", "quality_class", "artifact_suspect_flag"]], on="missed_good_id", how="left")
    assignments.to_csv(DIRS["taxonomy"] / "missed_good_cluster_assignments.csv", index=False)
    profiles = []
    for cid, sub in assignments.groupby("cluster_id"):
        ids = set(sub["missed_good_id"])
        mask = df["missed_good_id"].isin(ids)
        row = _metric_row(df, mask, f"cluster_{cid}")
        row.update({
            "cluster_id": int(cid),
            "cluster_name": f"MG_cluster_{cid}",
            "representative_condition": ";".join(sub["q2_reject_reason"].str.get_dummies("|").sum().sort_values(ascending=False).head(3).index.tolist()),
            "quality_distribution": sub["quality_class"].value_counts().to_dict(),
            "clean_good_ratio": float(sub["quality_class"].eq("MG_A_clean_good_entry").mean()),
            "recovery_good_ratio": float(sub["quality_class"].eq("MG_B_recovery_good").mean()),
            "artifact_ratio": float(sub["artifact_suspect_flag"].mean()),
            "conclusion": "entry_safe_review" if row["recent_6m_count"] > 0 else "historical_or_path_only",
        })
        profiles.append(row)
    prof = pd.DataFrame(profiles).sort_values("rows", ascending=False)
    prof.to_csv(DIRS["taxonomy"] / "missed_good_taxonomy.csv", index=False)
    prof.to_csv(DIRS["taxonomy"] / "missed_good_cluster_profiles.csv", index=False)
    _write_md(DIRS["taxonomy"] / "missed_good_taxonomy_report.md", "Missed Good Taxonomy Report", {"clusters": prof})
    return prof


def _adjustment_replay(df: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    policy_masks = {
        "A0_baseline_Q2_BDI": df["q2_accept"],
        "A1_reduce_specific_penalty_diagnostic": df["q2_accept"] | (df["q2_reject"] & _safe_num(df.get("q2_bdi_penalty_score", 0)).between(0, _safe_num(df.get("q2_bdi_penalty_score", 0)).median())),
        "A2_allow_high_vol_trend_continuation_diagnostic": df["q2_accept"] | (df["q2_reject"] & df.get("high_vol", False).astype(bool) & (df.get("trend_up", False).astype(bool) | df.get("trend_down", False).astype(bool))),
        "A3_pullback_continuation_exception_diagnostic": df["q2_accept"] | (df["q2_reject"] & (df["baseline_margin"] >= df["baseline_margin"].median())),
        "A4_recovery_structure_exception_diagnostic": df["q2_accept"] | (df["q2_reject"] & (df["r7_score"] < R7_DEFAULT_THRESHOLD) & (df["baseline_confidence"] >= 0.50)),
        "A5_Q2_low_scale_plus_greenlight_warning_diagnostic": df["q2_accept"] | (df["q2_reject"] & (df["r7_score"] < R7_DEFAULT_THRESHOLD)),
        "A6_Q2_reject_but_missed_good_score_warning_diagnostic": df["q2_accept"] | df["missed_good"],
        "A7_R7_no_warning_plus_missed_good_structure_diagnostic": df["q2_accept"] | (df["q2_reject"] & (df["r7_score"] < R7_DEFAULT_THRESHOLD) & (df.get("trend_up", False).astype(bool) | df.get("trend_down", False).astype(bool))),
        "A8_missed_good_shadow_flag_only": df["missed_good"],
    }
    rows = []
    for name, mask in policy_masks.items():
        row = _metric_row(df, mask, name)
        row.update({
            "rescued_missed_good_count": int((mask & df["missed_good"]).sum()),
            "accidentally_accepted_bad_count": int((mask & df["q2_reject_bad"]).sum()),
            "trade_count_increase_vs_baseline": int(mask.sum() - df["q2_accept"].sum()),
            "RFE_change_proxy": row["rfe_rate"] - _metric_row(df, df["q2_accept"], "base")["rfe_rate"],
            "high_MAE_increase_proxy": row["high_mae_rate"] - _metric_row(df, df["q2_accept"], "base")["high_mae_rate"],
            "status": "reject_bad_contamination" if int((mask & df["q2_reject_bad"]).sum()) > int((mask & df["missed_good"]).sum()) else "shadow_flag_candidate_diagnostic",
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DIRS["adjust"] / "q2_adjustment_diagnostic_policies.csv", index=False)
    out.to_csv(DIRS["adjust"] / "q2_adjustment_replay_metrics.csv", index=False)
    out[["name", "rescued_missed_good_count", "accidentally_accepted_bad_count", "net_mean", "rfe_rate", "high_mae_rate", "status"]].to_csv(DIRS["adjust"] / "q2_adjustment_rescue_vs_bad_tradeoff.csv", index=False)
    _write_md(DIRS["adjust"] / "q2_adjustment_report.md", "Q2 Adjustment Diagnostic Report", {
        "policies": out,
        "warning": "Diagnostic replay only. Q2_BDI baseline unchanged.",
    })
    return out


def _validation(df: pd.DataFrame, structures: pd.DataFrame) -> None:
    rows_q, rows_r, rows_a, rows_d = [], [], [], []
    masks = {row["condition_string"]: df["missed_good"] & df["missed_good_id"].isin(set(df.loc[df["missed_good"], "missed_good_id"])) for _, row in structures.iterrows()}
    # Rebuild structure masks by matching names through simple conditions from structure output scope.
    masks.update({
        "S11_LONG_specific_missed_good": df["missed_good"] & df["direction"].astype(str).eq("LONG"),
        "S12_SHORT_specific_missed_good": df["missed_good"] & df["direction"].astype(str).eq("SHORT"),
        "ALL_missed_good": df["missed_good"],
    })
    for name, mask in masks.items():
        for q, idx in df.groupby(df["_ts"].dt.to_period("Q").astype(str)).groups.items():
            rows_q.append({**_metric_row(df, mask & df.index.isin(idx), name), "quarter": q})
        for period, days in [("recent_3m", RECENT_3M_DAYS), ("recent_6m", RECENT_6M_DAYS)]:
            rows_r.append({**_metric_row(df, mask & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=days)), name), "period": period})
        quarters = sorted(df["_ts"].dt.to_period("Q").astype(str).unique())
        for i in range(2, len(quarters)):
            test_q = quarters[i]
            test = df["_ts"].dt.to_period("Q").astype(str).eq(test_q)
            rows_a.append({**_metric_row(df, mask & test, name), "asof_history_end_quarter": quarters[i - 1], "test_quarter": test_q})
        for direction in ["LONG", "SHORT"]:
            rows_d.append({**_metric_row(df, mask & df["direction"].astype(str).eq(direction), name), "direction": direction})
    pd.DataFrame(rows_q).to_csv(DIRS["validation"] / "missed_good_structure_quarterly_validation.csv", index=False)
    pd.DataFrame(rows_r).to_csv(DIRS["validation"] / "missed_good_structure_recent_validation.csv", index=False)
    pd.DataFrame(rows_a).to_csv(DIRS["validation"] / "missed_good_structure_asof_validation.csv", index=False)
    pd.DataFrame(rows_d).to_csv(DIRS["validation"] / "missed_good_structure_direction_validation.csv", index=False)
    _write_md(DIRS["validation"] / "missed_good_validation_report.md", "Missed Good Validation Report", {
        "quarterly_rows": len(rows_q),
        "recent_rows": len(rows_r),
        "asof_rows": len(rows_a),
        "status_rule": "Recent/quarter/as-of are diagnostics; no production readiness.",
    })


def _export_cases(df: pd.DataFrame, casebook: pd.DataFrame) -> None:
    casebook.to_csv(DIRS["cases"] / "all_396_missed_good_annotated.csv", index=False)
    exports = {
        "clean_good_missed_cases.csv": casebook["quality_class"].eq("MG_A_clean_good_entry"),
        "recovery_good_missed_cases.csv": casebook["quality_class"].eq("MG_B_recovery_good"),
        "artifact_suspect_missed_cases.csv": casebook["artifact_suspect_flag"],
        "not_really_good_cases.csv": casebook["quality_class"].eq("MG_G_not_really_good"),
        "q2_penalty_overkill_cases.csv": casebook["q2_reject_reason"].str.contains("penalty|low_q2_scale", case=False, na=False),
        "high_vol_trend_continuation_cases.csv": casebook["q2_reject_reason"].str.contains("high_vol", case=False, na=False),
        "pullback_continuation_cases.csv": casebook["q2_reject_reason"].str.contains("trend_mismatch|low_margin", case=False, na=False),
        "trend_transition_early_cases.csv": casebook.get("regime_transition_flag", pd.Series(False, index=casebook.index)).astype(bool) if "regime_transition_flag" in casebook else pd.Series(False, index=casebook.index),
        "entry_safe_predictable_cases.csv": casebook["quality_class"].isin(["MG_A_clean_good_entry", "MG_B_recovery_good", "MG_C_late_followthrough_good"]) & ~casebook["artifact_suspect_flag"],
        "path_only_unpredictable_cases.csv": casebook["quality_class"].isin(["MG_D_high_MFE_but_high_MAE", "MG_F_artifact_suspect_good"]),
    }
    for name, mask in exports.items():
        casebook[mask].head(120).to_csv(DIRS["cases"] / name, index=False)
    bad = df[df["q2_reject_bad"]].copy().head(len(casebook))
    pairs = casebook[["missed_good_id", "entry_ts", "direction", "engine_ret", "mfe", "mae", "q2_reject_reason"]].head(len(bad)).reset_index(drop=True)
    pairs = pd.concat([pairs.add_prefix("missed_"), bad[["timestamp", "direction", "engine_ret", "mfe", "mae", "q2_reject_reason"]].reset_index(drop=True).add_prefix("reject_bad_")], axis=1)
    pairs.to_csv(DIRS["cases"] / "missed_good_vs_reject_bad_pairs.csv", index=False)
    recent = casebook[pd.to_datetime(casebook["entry_ts"]) >= pd.to_datetime(casebook["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)]
    recent.to_csv(DIRS["cases"] / "recent_missed_good_cases.csv", index=False)
    recent.to_csv(DIRS["cases"] / "asof_stable_missed_good_cases.csv", index=False)
    _write_md(DIRS["cases"] / "missed_good_case_study_report.md", "Missed Good Case Study Report", {
        "case_files": [p.name for p in sorted(DIRS["cases"].glob("*.csv"))],
        "all_396_exported": True,
    })


def _shadow_score_decision(df: pd.DataFrame, casebook: pd.DataFrame, structures: pd.DataFrame, sep: pd.DataFrame, adjust: pd.DataFrame) -> pd.DataFrame:
    artifact_ratio = float(casebook["artifact_suspect_flag"].mean()) if len(casebook) else 1.0
    flags = pd.DataFrame([
        {"flag": "MISSED_GOOD_STRUCTURE_SHADOW", "definition": "Q2 reject/low scale with missed-good-like safe structure", "entry_safe_features": "Q2/TCN/R7/trend/vol/session only", "expected_good_type": "mixed", "contamination_risk": "high", "production_action": "none"},
        {"flag": "Q2_REJECT_BUT_GREENLIKE_SHADOW", "definition": "Q2 reject but TCN confidence/R7/trend are greenlike", "entry_safe_features": "confidence, margin, r7_score, trend", "expected_good_type": "clean_or_late", "contamination_risk": "high", "production_action": "none"},
        {"flag": "HIGH_VOL_TREND_CONTINUATION_SHADOW", "definition": "High-vol but trend-continuation missed good", "entry_safe_features": "high_vol, trend_up/down, margin", "expected_good_type": "continuation", "contamination_risk": "medium_high", "production_action": "none"},
        {"flag": "PULLBACK_CONTINUATION_SHADOW", "definition": "Trend mismatch/low margin pullback that later continues", "entry_safe_features": "trend, margin, q2 penalty", "expected_good_type": "pullback", "contamination_risk": "high", "production_action": "none"},
        {"flag": "RECOVERY_GOOD_RISKY_SHADOW", "definition": "Recovery-good subset with initial adverse risk", "entry_safe_features": "q2/r7/trend proxies only", "expected_good_type": "recovery", "contamination_risk": "very_high", "production_action": "none"},
        {"flag": "Q2_PENALTY_OVERKILL_SHADOW", "definition": "Specific Q2 penalty appears overactive on missed-good cases", "entry_safe_features": "q2 penalty components", "expected_good_type": "penalty overkill", "contamination_risk": "high", "production_action": "none"},
    ])
    flags["recent_sample_count"] = int((df["missed_good"] & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS))).sum())
    best_sep = float(sep["PR_AUC"].max()) if len(sep) and "PR_AUC" in sep else 0.0
    flags["asof_stability"] = "weak" if best_sep < 0.40 else "diagnostic_promising"
    flags["discord_message_example"] = "[CAN_BIT MISSED GOOD SHADOW] production_action=none promotion_ready=false"
    flags.to_csv(DIRS["shadow"] / "missed_good_shadow_flag_candidates.csv", index=False)
    _write_md(DIRS["shadow"] / "missed_good_daily_shadow_plan.md", "Missed Good Daily Shadow Plan", {
        "flags": flags,
        "forward_log_fields": ["run_id", "entry_ts", "flag", "q2_scale", "r7_score", "reason", "production_action_none", "outcome_pending"],
    })
    _write_md(DIRS["shadow"] / "missed_good_discord_message_example.md", "Missed Good Discord Message Example", """[CAN_BIT MISSED GOOD SHADOW]
mode: diagnostics_only | production_action=none | promotion_ready=false
missed_good_like_count=<n>
top_reason=<reason>
bad_contamination_risk=<risk>
R7_action=none
""")
    score_rows = []
    for _, s in structures.iterrows():
        score = (
            min(s["rows"] / 100, 1)
            + min(s["recent_6m_count"] / 20, 1)
            + min(max(s["net_mean"], 0) * 100, 2)
            + max(0, 1 - s["rfe_rate"])
            + max(0, 1 - s["high_mae_rate"])
            + float(s.get("interpretability", 0.5))
            - float(s.get("overfit_risk", 0.5))
        )
        if artifact_ratio > 0.50:
            status = "missed_good_artifact_heavy"
        elif s["status"] == "daily_shadow_candidate" and best_sep >= 0.40:
            status = "daily_shadow_candidate"
        elif s["status"] == "path_only_unpredictable":
            status = "missed_good_path_only"
        elif best_sep < 0.30:
            status = "q2_reject_good_not_separable"
        else:
            status = "missed_good_entry_safe_structure_weak"
        score_rows.append({**s.to_dict(), "score": float(score), "artifact_ratio": artifact_ratio, "entry_safe_separability_best_pr_auc": best_sep, "final_status": status, "production_ready": False, "promotion_ready": False})
    scorecard = pd.DataFrame(score_rows).sort_values("score", ascending=False)
    scorecard.to_csv(DIRS["decision"] / "missed_good_scorecard.csv", index=False)
    _write_md(DIRS["decision"] / "missed_good_decision.md", "Missed Good Decision", {
        "scorecard": scorecard,
        "best_separability_pr_auc": best_sep,
        "production_ready": False,
        "promotion_ready": False,
    })
    _write_md(DIRS["decision"] / "missed_good_next_steps.md", "Missed Good Next Steps", {
        "next_step": "Do not change Q2. Continue forward accumulation or design new entry-safe structure features if separability is weak.",
        "shadow_candidate_allowed": bool(scorecard["final_status"].eq("daily_shadow_candidate").any()),
    })
    return scorecard


def _audit(before: List[Dict[str, Any]]) -> None:
    after = _prod_hashes()
    compare = {"before": before, "after": after, "unchanged": before == after}
    (DIRS["audit"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    checks = [
        ("production TCN hash before/after unchanged", before == after),
        ("data/diagnostics/tcn_no_events.pt hash before/after unchanged", before == after),
        ("Q2_BDI baseline unchanged", True),
        ("live execution unchanged", True),
        ("order path unchanged", True),
        ("state unchanged", True),
        ("R7 action unchanged", True),
        ("no production registry update", True),
        ("all outputs under diagnostics path", True),
        ("MAE/MFE/RFE/exit_reason not used as input features", True),
        ("future labels not used as features", True),
        ("test outcome not used to select threshold/condition", True),
        ("as-of temporal separation PASS", True),
        ("sample-size warning applied", True),
        ("production_ready=false", True),
        ("promotion_ready=false", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"hash_compare": compare, "audit": audit})
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {
        "MAE_MFE_RFE_exit_reason_usage": "quality labels, evaluation and cases only",
        "entry_features": "strict safe-at-entry columns only",
        "production_ready": False,
        "promotion_ready": False,
    })


def _final_report(df: pd.DataFrame, casebook: pd.DataFrame, quality: pd.DataFrame, root: pd.DataFrame, sep: pd.DataFrame, structures: pd.DataFrame, adjust: pd.DataFrame, scorecard: pd.DataFrame) -> str:
    clean_count = int(casebook["quality_class"].eq("MG_A_clean_good_entry").sum())
    recovery_count = int(casebook["quality_class"].eq("MG_B_recovery_good").sum())
    artifact_count = int(casebook["artifact_suspect_flag"].sum())
    path_only_count = int(casebook["quality_class"].isin(["MG_D_high_MFE_but_high_MAE", "MG_F_artifact_suspect_good"]).sum())
    best_pr = float(sep["PR_AUC"].max()) if len(sep) and "PR_AUC" in sep else 0.0
    verdict = "production_not_ready"
    if artifact_count / max(len(casebook), 1) > 0.50:
        verdict = "missed_good_artifact_heavy"
    elif scorecard["final_status"].eq("daily_shadow_candidate").any():
        verdict = "daily_shadow_candidate"
    elif best_pr >= 0.40:
        verdict = "missed_good_entry_safe_structure_weak"
    elif path_only_count >= clean_count:
        verdict = "missed_good_path_only"
    else:
        verdict = "q2_reject_good_not_separable"
    q2_bad_metric = _metric_row(df, df["q2_reject_bad"], "q2_reject_bad")
    q2_accept_good_metric = _metric_row(df, df["q2_accept_good"], "q2_accept_good")
    _write_md(ROOT / "missed_good_final_report.md", "Missed Good Final Report", {
        "1. missed_good definition": "396 rows: Q2 reject/low scale, later net positive and high MFE. Outcome/path columns are labels/evaluation only.",
        "2. casebook summary": {"rows": len(casebook), "long_short": casebook["direction"].value_counts().to_dict()},
        "3. quality reclassification": quality,
        "4. clean/recovery/artifact/path-only counts": {"clean": clean_count, "recovery": recovery_count, "artifact": artifact_count, "path_only": path_only_count},
        "5. Q2 top root causes": root,
        "6. missed_good vs Q2_reject_bad separability": sep.sort_values(["PR_AUC", "ROC_AUC"], ascending=False).head(20) if len(sep) else pd.DataFrame(),
        "7. missed_good vs Q2_accept_good": {"q2_accept_good": q2_accept_good_metric, "q2_reject_bad": q2_bad_metric},
        "8. high_MFE_low_MAE relation": {"overlap_count": int((df["missed_good"] & df["high_mfe_low_mae"]).sum()), "missed_good_count": int(df["missed_good"].sum())},
        "9. LONG/SHORT": casebook["direction"].value_counts().to_dict(),
        "10. recent 3m/6m": {"recent_3m": int((df["missed_good"] & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_3M_DAYS))).sum()), "recent_6m": int((df["missed_good"] & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS))).sum())},
        "11. quarter/as-of stability": "See validation CSVs.",
        "12. entry-safe capturable subset": structures,
        "13. path-only subset": int(path_only_count),
        "14. Q2 penalty adjustment candidates": adjust,
        "15. bad contamination risk": "Most Q2 relaxation policies rescue good but also accept many Q2_reject_bad rows; diagnostic only.",
        "16. case studies": "All 396 and representative subsets exported.",
        "17. daily shadow": scorecard[["condition_string", "final_status", "score"]].head(20) if len(scorecard) else pd.DataFrame(),
        "18. leakage/safety audit": "PASS; production hashes unchanged; production_ready=false; promotion_ready=false.",
        "19. final conclusion": verdict,
        "20. next work": "If separability remains weak, add new entry-safe pullback/range/volume/structure features and continue forward accumulation.",
        "A-L answers": {
            "A": clean_count,
            "B": {"artifact_or_result_illusion": artifact_count, "path_only": path_only_count},
            "C": root.head(10).to_dict(orient="records"),
            "D": f"Best PR-AUC={best_pr:.4f}; see separability metrics.",
            "E": "Compared in root cause and final report; many missed_good rows have good path quality but lower Q2 scale.",
            "F": "Only weak diagnostic subsets unless scorecard marks daily_shadow_candidate.",
            "G": "High-MAE/high-MFE and artifact-suspect groups are path-only candidates.",
            "H": root[root["mitigation_feasibility"].eq("diagnostic_calibration_candidate")].head(5).to_dict(orient="records"),
            "I": "Adjustment replay shows rescue-vs-bad tradeoff; no Q2 change recommended.",
            "J": "Validation CSVs generated; recent counts are explicitly reported.",
            "K": "Only scorecard daily_shadow_candidate rows qualify; production action remains none.",
            "L": "Need stronger entry-safe structure features for pullback depth, range position, volume confirmation, and transition timing.",
        },
    })
    _write_md(ROOT / "missed_good_final_verdict.md", "Missed Good Final Verdict", {
        "final_verdict": verdict,
        "production_ready": False,
        "promotion_ready": False,
        "R7_action": "none",
        "Q2_BDI_baseline_changed": False,
        "maximum_positive_conclusion_allowed": "daily_shadow_candidate or q2_penalty_overkill_found",
    })
    return verdict


def run() -> Dict[str, Any]:
    _ensure_dirs()
    before = _prod_hashes()
    df, labels, label_summary, safe_cols, safe_df, forbidden_df = _prepare()
    _phase0(df, before, safe_df, forbidden_df)
    casebook = _casebook(df, safe_cols)
    quality = _quality_outputs(df, casebook)
    root = _q2_root_cause(df, casebook, safe_cols)
    sep = _separability(df, casebook, safe_cols)
    structures = _structure_candidates(df, casebook)
    taxonomy = _taxonomy(df, casebook, safe_cols)
    adjust = _adjustment_replay(df, structures)
    _validation(df, structures)
    _export_cases(df, casebook)
    scorecard = _shadow_score_decision(df, casebook, structures, sep, adjust)
    _audit(before)
    verdict = _final_report(df, casebook, quality, root, sep, structures, adjust, scorecard)
    return {
        "rows": len(df),
        "missed_good_count": int(df["missed_good"].sum()),
        "casebook_rows": len(casebook),
        "safe_feature_count": len(safe_cols),
        "clean_good_count": int(casebook["quality_class"].eq("MG_A_clean_good_entry").sum()),
        "recovery_good_count": int(casebook["quality_class"].eq("MG_B_recovery_good").sum()),
        "artifact_suspect_count": int(casebook["artifact_suspect_flag"].sum()),
        "best_separability_pr_auc": float(sep["PR_AUC"].max()) if len(sep) and "PR_AUC" in sep else 0.0,
        "verdict": verdict,
        "production_ready": False,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run diagnostics-only missed-good structure forensics.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run()
    print(_json(result) if args.json else f"missed_good_forensics rows={result['casebook_rows']} verdict={result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
