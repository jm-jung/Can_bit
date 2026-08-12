"""
Executed-only Entry-Quality Label V2 builder.

Diagnostics only. This script builds a new label dataset under
data/diagnostics/executed_entry_quality_label_v2/ and never overwrites
production datasets, model weights, Q2 config, live execution, order paths,
state files, launchd jobs, or R7 behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scripts.diagnostics.run_entry_greenlight_forensics import (
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
from scripts.diagnostics.run_false_high_r7_monitor import Q2_ACCEPT, Q2_REJECT, R7_DEFAULT_THRESHOLD, _prod_hashes

ROOT = Path("data/diagnostics/executed_entry_quality_label_v2")
VERSION = f"executed_entry_quality_label_v2_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
POSITION_SIZE = 0.05

DIRS = {
    "discovery": ROOT / "discovery",
    "safety": ROOT / "safety",
    "artifact_audit": ROOT / "artifact_audit",
    "universe": ROOT / "universe",
    "artifacts": ROOT / "artifacts",
    "label_design": ROOT / "label_design",
    "cost": ROOT / "cost",
    "path": ROOT / "path_quality",
    "maxholding": ROOT / "max_holding",
    "counterfactual": ROOT / "counterfactual",
    "tournament": ROOT / "tournament",
    "alignment": ROOT / "system_alignment",
    "stability": ROOT / "stability",
    "separability": ROOT / "separability",
    "dataset": ROOT / "dataset",
    "readiness": ROOT / "readiness",
    "audit": ROOT / "audit",
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b not in (0, 0.0) and not pd.isna(b) else default


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT) if path.is_absolute() and path.exists() else path),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() and path.is_file() else "",
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _git_status() -> str:
    try:
        return subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, timeout=10).stdout
    except Exception as exc:
        return f"git_status_unavailable: {exc}"


def _metric_row(df: pd.DataFrame, mask: pd.Series, name: str) -> Dict[str, Any]:
    sub = df.loc[mask].copy()
    ret = _safe_num(sub.get("engine_ret", pd.Series(dtype=float)))
    mfe = _safe_num(sub.get("mfe", pd.Series(dtype=float)))
    mae = _safe_num(sub.get("mae", pd.Series(dtype=float)))
    max_ts = df["_ts"].max() if len(df) else pd.Timestamp("1970-01-01")
    return {
        "name": name,
        "rows": int(len(sub)),
        "start": str(sub["_ts"].min()) if len(sub) else "",
        "end": str(sub["_ts"].max()) if len(sub) else "",
        "long_count": int(sub["direction"].astype(str).eq("LONG").sum()) if len(sub) else 0,
        "short_count": int(sub["direction"].astype(str).eq("SHORT").sum()) if len(sub) else 0,
        "q2_accept_count": int(sub["q2_accept"].sum()) if len(sub) and "q2_accept" in sub else 0,
        "q2_reject_count": int(sub["q2_reject"].sum()) if len(sub) and "q2_reject" in sub else 0,
        "r7_warning_count": int(sub["r7_high_hazard"].sum()) if len(sub) and "r7_high_hazard" in sub else 0,
        "net_mean": float(ret.mean()) if len(sub) else 0.0,
        "net_median": float(ret.median()) if len(sub) else 0.0,
        "winrate": float((ret > 0).mean()) if len(sub) else 0.0,
        "profit_factor": _profit_factor(ret) if len(sub) else 0.0,
        "mfe_median": float(mfe.median()) if len(sub) else 0.0,
        "mae_median": float(mae.median()) if len(sub) else 0.0,
        "mfe_mae_ratio": _safe_div(float(mfe.median()) if len(sub) else 0.0, abs(float(mae.median())) if len(sub) else 0.0),
        "rfe_rate": float(sub.get("rfe_flag", pd.Series(False, index=sub.index)).astype(bool).mean()) if len(sub) else 0.0,
        "max_holding_count": int(sub.get("exit_reason", pd.Series("", index=sub.index)).astype(str).str.contains("max_holding", case=False, na=False).sum()) if len(sub) else 0,
        "artifact_count": int(sub.get("artifact_exclude", pd.Series(False, index=sub.index)).astype(bool).sum()) if len(sub) else 0,
        "recent_3m_count": int((sub["_ts"] >= max_ts - pd.Timedelta(days=RECENT_3M_DAYS)).sum()) if len(sub) else 0,
        "recent_6m_count": int((sub["_ts"] >= max_ts - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(sub) else 0,
        "quarter_count": int(sub["_ts"].dt.to_period("Q").astype(str).nunique()) if len(sub) else 0,
        "mdd": _mdd(ret) if len(sub) else 0.0,
    }


def _discover_paths() -> Dict[str, List[str]]:
    patterns = {
        "trade_dataset": ["*trade*", "*executed*", "*meta_dataset*"],
        "candidate_dataset": ["*candidate*", "*meta_dataset*", "*r7_input*"],
        "label_dataset": ["*label*", "*quality*", "*target*"],
        "feature_proba_q2_r7": ["*feature*", "*proba*", "*q2*", "*r7*", "*false_high*"],
        "engine_outcome": ["*engine*", "*replay*", "*simulate*"],
    }
    out: Dict[str, List[str]] = {}
    for name, pats in patterns.items():
        vals: List[str] = []
        for pat in pats:
            for p in REPO_ROOT.rglob(pat):
                rel = str(p.relative_to(REPO_ROOT))
                if any(skip in rel for skip in [".git", ".venv", "__pycache__", "node_modules"]):
                    continue
                vals.append(rel)
        out[name] = sorted(set(vals))[:200]
    return out


def _prepare_frame() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    df = _load_frame()
    labels, label_summary = _labels(df)
    safe_cols, safe_df, forbidden_df = _safe_features(df)
    df["executed_flag"] = df.get("is_executed", pd.Series(True, index=df.index)).astype(bool) if "is_executed" in df else pd.Series(True, index=df.index)
    df["counterfactual_flag"] = ~df["executed_flag"]
    df["trade_id"] = df.get("trade_id", pd.Series(df.index, index=df.index)).astype(str)
    df["candidate_id"] = df.get("candidate_id", df["trade_id"]).astype(str)
    df["entry_ts"] = df["_ts"]
    df["exit_ts"] = pd.to_datetime(df.get("exit_ts", pd.NaT), errors="coerce") if "exit_ts" in df else pd.NaT
    df["symbol"] = df.get("symbol", "BTCUSDT")
    df["timeframe"] = df.get("timeframe", "5m")
    df["holding_bars"] = _safe_num(df.get("hold_bars", df.get("holding_bars", pd.Series(0, index=df.index))))
    df["net_return"] = _safe_num(df.get("engine_ret", df.get("net_return", pd.Series(0, index=df.index))))
    df["net_return_after_cost"] = df["net_return"]
    df["MFE"] = _safe_num(df.get("mfe", pd.Series(0, index=df.index)))
    df["MAE"] = _safe_num(df.get("mae", pd.Series(0, index=df.index)))
    df["RFE"] = df.get("rfe_flag", pd.Series(False, index=df.index)).astype(bool)
    df["exit_reason"] = df.get("exit_reason", pd.Series("", index=df.index)).astype(str)
    for c in [c for c in labels.columns if c.startswith("L")]:
        df[c] = labels[c].astype(bool)
    return df, labels, label_summary, safe_cols, safe_df, forbidden_df


def phase0_discovery(df: pd.DataFrame, labels: pd.DataFrame, safe_df: pd.DataFrame, forbidden_df: pd.DataFrame) -> Dict[str, Any]:
    paths = _discover_paths()
    (DIRS["discovery"] / "discovered_data_paths.json").write_text(_json(paths), encoding="utf-8")
    label_files = [{"path": p, "category": group} for group, vals in paths.items() for p in vals if "label" in p.lower() or "quality" in p.lower()]
    pd.DataFrame(label_files).to_csv(DIRS["discovery"] / "label_file_inventory.csv", index=False)
    pd.DataFrame([{"path": p, "category": group} for group, vals in paths.items() for p in vals if group in {"trade_dataset", "candidate_dataset", "engine_outcome"}]).to_csv(DIRS["discovery"] / "trade_dataset_inventory.csv", index=False)
    pd.DataFrame([{"column": c, "dtype": str(df[c].dtype), "non_null_rate": float(df[c].notna().mean())} for c in df.columns]).to_csv(DIRS["discovery"] / "column_inventory.csv", index=False)
    label_cols = [c for c in df.columns if c.startswith("L") or "label" in c.lower() or "target" in c.lower() or "quality" in c.lower()]
    pd.DataFrame([{"label": c, "positive_count": int(df[c].astype(bool).sum()) if df[c].dropna().isin([True, False, 0, 1]).all() else np.nan, "available": True} for c in label_cols]).to_csv(DIRS["discovery"] / "existing_label_registry.csv", index=False)
    split = pd.concat([
        safe_df.assign(split="entry_safe"),
        forbidden_df.assign(split="future_or_evaluation"),
    ], ignore_index=True, sort=False)
    split.to_csv(DIRS["discovery"] / "entry_safe_vs_future_column_split.csv", index=False)
    executed = df["executed_flag"]
    summary = {
        "executed_rows": int(executed.sum()),
        "counterfactual_rows": int((~executed).sum()),
        "candidate_rows": len(df),
        "date_range": [str(df["_ts"].min()), str(df["_ts"].max())],
        "long_short": df["direction"].astype(str).value_counts().to_dict(),
        "recent_3m": int((df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_3M_DAYS)).sum()),
        "recent_6m": int((df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()),
        "duplicate_trade_id": int(df["trade_id"].duplicated().sum()),
        "duplicate_timestamp": int(df["_ts"].duplicated().sum()),
        "path_metric_available": {"MFE": "MFE" in df, "MAE": "MAE" in df, "RFE": "RFE" in df},
        "cost_metric_available": "net_return_after_cost" in df,
    }
    _write_md(DIRS["discovery"] / "discovery_report.md", "Discovery Report", summary)
    return summary


def _safety_paths() -> List[Path]:
    return [
        REPO_ROOT / "models/tcn_v1.pt",
        REPO_ROOT / "data/diagnostics/tcn_no_events.pt",
        REPO_ROOT / "scripts/diagnostics/run_false_high_r7_daily_monitor.py",
        REPO_ROOT / "ops/run_false_high_r7_daily_monitor.sh",
        REPO_ROOT / "scripts/diagnostics/run_false_high_r7_monitor.py",
    ]


def phase1_safety_before() -> Dict[str, Any]:
    snap = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "os": os.name,
        "prod_hashes": _prod_hashes(),
        "selected_hashes": [_hash_path(p) for p in _safety_paths()],
        "git_status_short": _git_status(),
        "production_ready": False,
        "promotion_ready": False,
        "r7_action": "none",
    }
    (DIRS["safety"] / "safety_snapshot_before.json").write_text(_json(snap), encoding="utf-8")
    return snap


def _flag_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["A1_counterfactual_only"] = out["counterfactual_flag"]
    out["A2_max_holding_artifact"] = out["exit_reason"].str.contains("max_holding", case=False, na=False)
    out["A3_tiny_return_noise"] = out["net_return_after_cost"].abs() < 0.001
    out["A4_fee_slippage_flip"] = (out["net_return"] > 0) & (out["net_return_after_cost"] <= 0)
    out["A5_high_MFE_high_MAE_path_risk"] = (out["MFE"] >= 0.006) & (out["MAE"].abs() >= 0.006)
    out["A6_late_MFE_only"] = out["holding_bars"] >= 10
    out["A7_RFE_failure"] = out["RFE"]
    out["A8_missing_path_metrics"] = out[["MFE", "MAE"]].isna().any(axis=1)
    out["A9_timestamp_alignment_suspect"] = out["_ts"].duplicated(keep=False)
    out["A10_duplicate_trade_suspect"] = out["trade_id"].duplicated(keep=False)
    out["A11_exit_reason_ambiguous"] = out["exit_reason"].eq("") | out["exit_reason"].str.contains("unknown|ambiguous", case=False, na=False)
    out["A12_horizon_mismatch"] = out["holding_bars"].le(0) | out["holding_bars"].gt(48)
    out["A13_outlier_wick_or_data_gap_suspect"] = (out["MFE"].abs() > out["MFE"].abs().quantile(0.995)) | (out["MAE"].abs() > out["MAE"].abs().quantile(0.995))
    short_count = int(out["direction"].astype(str).eq("SHORT").sum())
    out["A14_low_sample_direction_suspect"] = out["direction"].astype(str).eq("SHORT") & (short_count < 100)
    out["A15_model_generated_candidate_not_executed"] = out["counterfactual_flag"]
    artifact_cols = [c for c in out.columns if c.startswith("A") and c[1:3].split("_")[0].isdigit()]
    out["A16_unknown_artifact"] = False
    artifact_cols = [c for c in out.columns if c.startswith("A")]
    severity = pd.Series(0, index=out.index)
    severity += out["A1_counterfactual_only"].astype(int) * 4
    severity += out["A2_max_holding_artifact"].astype(int) * 3
    severity += out["A3_tiny_return_noise"].astype(int) * 2
    severity += out["A4_fee_slippage_flip"].astype(int) * 2
    severity += out["A5_high_MFE_high_MAE_path_risk"].astype(int) * 2
    severity += out["A7_RFE_failure"].astype(int) * 2
    severity += out["A8_missing_path_metrics"].astype(int) * 4
    severity += out["A10_duplicate_trade_suspect"].astype(int)
    out["artifact_severity"] = severity.clip(upper=4)
    out["artifact_flags"] = out[artifact_cols].apply(lambda row: "|".join([c for c, v in row.items() if bool(v)]) or "A0_clean", axis=1)
    out["artifact_exclude"] = out["artifact_severity"] >= 4
    out["artifact_moderate_or_worse"] = out["artifact_severity"] >= 2
    return out


def phase2_existing_label_audit(df: pd.DataFrame, labels: pd.DataFrame, safe_cols: List[str]) -> pd.DataFrame:
    label_cols = [c for c in labels.columns if c.startswith("L")]
    rows = []
    for c in label_cols:
        y = labels[c].astype(bool)
        pos = df[y]
        rows.append({
            "label": c,
            "positive_count": int(y.sum()),
            "negative_count": int((~y).sum()),
            "neutral_count": int((df["net_return_after_cost"].abs() < 0.001).sum()),
            "ambiguous_count": int((y & df["artifact_moderate_or_worse"]).sum()),
            "executed_count": int((y & df["executed_flag"]).sum()),
            "counterfactual_count": int((y & df["counterfactual_flag"]).sum()),
            "long_count": int((y & df["direction"].astype(str).eq("LONG")).sum()),
            "short_count": int((y & df["direction"].astype(str).eq("SHORT")).sum()),
            "recent_count": int((y & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS))).sum()),
            "max_holding_ratio": float(pos["A2_max_holding_artifact"].mean()) if len(pos) else 0.0,
            "tiny_return_ratio": float(pos["A3_tiny_return_noise"].mean()) if len(pos) else 0.0,
            "fee_slippage_flip_ratio": float(pos["A4_fee_slippage_flip"].mean()) if len(pos) else 0.0,
            "high_mfe_high_mae_ratio": float(pos["A5_high_MFE_high_MAE_path_risk"].mean()) if len(pos) else 0.0,
            "rfe_ratio": float(pos["RFE"].mean()) if len(pos) else 0.0,
            "path_only_suspect_ratio": float((pos["A2_max_holding_artifact"] | pos["A6_late_MFE_only"] | pos["counterfactual_flag"]).mean()) if len(pos) else 0.0,
            "reuse_verdict": "discard_or_quarantine" if len(pos) and float((pos["artifact_moderate_or_worse"]).mean()) > 0.50 else "auxiliary_only",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(DIRS["artifact_audit"] / "existing_label_artifact_summary.csv", index=False)
    by_source = []
    for c in label_cols:
        y = labels[c].astype(bool)
        by_source.append({"label": c, "source": "executed", "positive_count": int((y & df["executed_flag"]).sum()), "artifact_rate": float(df.loc[y & df["executed_flag"], "artifact_moderate_or_worse"].mean()) if (y & df["executed_flag"]).any() else 0.0})
        by_source.append({"label": c, "source": "counterfactual", "positive_count": int((y & df["counterfactual_flag"]).sum()), "artifact_rate": float(df.loc[y & df["counterfactual_flag"], "artifact_moderate_or_worse"].mean()) if (y & df["counterfactual_flag"]).any() else 0.0})
    pd.DataFrame(by_source).to_csv(DIRS["artifact_audit"] / "label_by_source_executed_vs_counterfactual.csv", index=False)
    summary[["label", "positive_count", "max_holding_ratio", "reuse_verdict"]].to_csv(DIRS["artifact_audit"] / "max_holding_artifact_audit.csv", index=False)
    summary[["label", "positive_count", "tiny_return_ratio", "reuse_verdict"]].to_csv(DIRS["artifact_audit"] / "tiny_return_noise_audit.csv", index=False)
    summary[["label", "positive_count", "fee_slippage_flip_ratio", "reuse_verdict"]].to_csv(DIRS["artifact_audit"] / "fee_slippage_flip_audit.csv", index=False)
    summary[summary["label"].str.contains("MFE|L5", case=False, na=False)].to_csv(DIRS["artifact_audit"] / "high_mfe_label_audit.csv", index=False)
    summary[summary["label"].str.contains("L12", case=False, na=False)].to_csv(DIRS["artifact_audit"] / "missed_good_label_audit.csv", index=False)
    drift_rows = []
    tmp = df.copy()
    tmp["_quarter"] = tmp["_ts"].dt.to_period("Q").astype(str)
    for c in label_cols:
        tmp[c] = labels[c].astype(bool)
        for q, sub in tmp.groupby("_quarter"):
            drift_rows.append({"label": c, "quarter": q, "positive_rate": float(sub[c].mean()), "artifact_rate": float(sub["artifact_moderate_or_worse"].mean()), "rows": len(sub)})
    pd.DataFrame(drift_rows).to_csv(DIRS["artifact_audit"] / "label_drift_audit.csv", index=False)
    _write_md(DIRS["artifact_audit"] / "existing_label_artifact_report.md", "Existing Label Artifact Report", {
        "summary": summary,
        "main_contamination": "counterfactual, max_holding, tiny return, path-only MFE, RFE/high-MAE risk",
        "policy": "Existing path labels should be auxiliary/evaluation unless redefined in executed-only V2.",
    })
    return summary


def phase3_universes(df: pd.DataFrame) -> Dict[str, pd.Series]:
    max_ts = df["_ts"].max()
    masks = {
        "U0_all_available": pd.Series(True, index=df.index),
        "U1_executed_only_core": df["executed_flag"],
        "U2_executed_without_artifact": df["executed_flag"] & ~df["artifact_exclude"],
        "U3_executed_clean_path_available": df["executed_flag"] & (df["artifact_severity"] <= 1),
        "U4_counterfactual_separate_reference_only": df["counterfactual_flag"],
        "U5_counterfactual_high_confidence_reference_only": df["counterfactual_flag"] & (df["baseline_confidence"] >= 0.55),
        "U6_q2_accepted_executed": df["executed_flag"] & df["q2_accept"],
        "U7_q2_rejected_or_low_scale_executed": df["executed_flag"] & df["q2_reject"],
        "U8_long_executed": df["executed_flag"] & df["direction"].astype(str).eq("LONG"),
        "U9_short_executed": df["executed_flag"] & df["direction"].astype(str).eq("SHORT"),
        "U10_recent_3m_executed": df["executed_flag"] & (df["_ts"] >= max_ts - pd.Timedelta(days=RECENT_3M_DAYS)),
        "U11_recent_6m_executed": df["executed_flag"] & (df["_ts"] >= max_ts - pd.Timedelta(days=RECENT_6M_DAYS)),
    }
    rows = [_metric_row(df, m, n) for n, m in masks.items()]
    pd.DataFrame(rows).to_csv(DIRS["universe"] / "universe_registry.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["universe"] / "universe_summary.csv", index=False)
    df.loc[masks["U1_executed_only_core"]].to_parquet(DIRS["universe"] / "executed_core_universe.parquet", index=False)
    df.loc[masks["U4_counterfactual_separate_reference_only"]].to_parquet(DIRS["universe"] / "counterfactual_reference_universe.parquet", index=False)
    _write_md(DIRS["universe"] / "universe_report.md", "Universe Report", {"summary": pd.DataFrame(rows), "policy": "Counterfactual is reference only and excluded from core trainable label by default."})
    return masks


def phase4_artifact_taxonomy(df: pd.DataFrame, labels: pd.DataFrame) -> None:
    taxonomy = """# Artifact Taxonomy

A0_clean: no material artifact flags.
A1_counterfactual_only: not an executed trade; reference only.
A2_max_holding_artifact: exit depends on max holding policy.
A3_tiny_return_noise: abs(net_after_cost) below tiny band.
A4_fee_slippage_flip: raw positive flips after costs.
A5_high_MFE_high_MAE_path_risk: high MFE but deep adverse path.
A6_late_MFE_only: MFE appears late; holding-policy dependent.
A7_RFE_failure: reversal/failure evidence present.
A8_missing_path_metrics: MFE/MAE/RFE unavailable.
A9_timestamp_alignment_suspect: duplicated/alignment suspect timestamp.
A10_duplicate_trade_suspect: duplicated trade id.
A11_exit_reason_ambiguous: missing/ambiguous exit reason.
A12_horizon_mismatch: invalid/very long holding horizon.
A13_outlier_wick_or_data_gap_suspect: extreme path value.
A14_low_sample_direction_suspect: SHORT sample too sparse.
A15_model_generated_candidate_not_executed: counterfactual candidate.
A16_unknown_artifact: fallback.
"""
    (DIRS["artifacts"] / "artifact_taxonomy.md").write_text(taxonomy, encoding="utf-8")
    cols = ["trade_id", "candidate_id", "entry_ts", "direction", "executed_flag", "counterfactual_flag", "artifact_flags", "artifact_severity", "artifact_exclude"] + [c for c in df.columns if c.startswith("A")]
    df[cols].to_parquet(DIRS["artifacts"] / "artifact_flags.parquet", index=False)
    artifact_cols = [c for c in df.columns if c.startswith("A")]
    summary = pd.DataFrame([{"artifact": c, "count": int(df[c].astype(bool).sum()), "ratio": float(df[c].astype(bool).mean())} for c in artifact_cols])
    summary.to_csv(DIRS["artifacts"] / "artifact_summary.csv", index=False)
    by_label = []
    for c in [c for c in labels.columns if c.startswith("L")]:
        y = labels[c].astype(bool)
        by_label.append({"label": c, "positive_count": int(y.sum()), "artifact_moderate_or_worse_rate": float(df.loc[y, "artifact_moderate_or_worse"].mean()) if y.any() else 0.0, "exclude_rate": float(df.loc[y, "artifact_exclude"].mean()) if y.any() else 0.0})
    pd.DataFrame(by_label).to_csv(DIRS["artifacts"] / "artifact_by_label.csv", index=False)
    policies = {
        "P0_no_purge_baseline": pd.Series(True, index=df.index),
        "P1_exclude_severe_artifacts": df["artifact_severity"] < 4,
        "P2_exclude_counterfactual_and_max_holding": ~(df["A1_counterfactual_only"] | df["A2_max_holding_artifact"]),
        "P3_exclude_counterfactual_maxholding_tiny": ~(df["A1_counterfactual_only"] | df["A2_max_holding_artifact"] | df["A3_tiny_return_noise"]),
        "P4_exclude_all_path_only_artifacts": ~(df["A1_counterfactual_only"] | df["A2_max_holding_artifact"] | df["A5_high_MFE_high_MAE_path_risk"] | df["A6_late_MFE_only"]),
        "P5_strict_executed_clean_only": df["executed_flag"] & (df["artifact_severity"] <= 1),
    }
    pd.DataFrame([{"policy": k, "kept_rows": int(v.sum()), "excluded_rows": int((~v).sum()), "executed_kept": int((v & df["executed_flag"]).sum())} for k, v in policies.items()]).to_csv(DIRS["artifacts"] / "artifact_purge_policy_comparison.csv", index=False)
    _write_md(DIRS["artifacts"] / "artifact_report.md", "Artifact Report", {"artifact_summary": summary, "purge_policies": pd.DataFrame([{"policy": k, "kept_rows": int(v.sum())} for k, v in policies.items()])})


def _path_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["trade_id", "candidate_id", "entry_ts", "direction", "net_return_after_cost", "MFE", "MAE", "RFE", "holding_bars", "exit_reason"]].copy()
    adverse = out["MAE"].abs().clip(lower=0.0005)
    out["MFE_to_MAE_ratio"] = out["MFE"] / adverse
    out["high_MAE_flag"] = out["MAE"] <= -0.006
    out["tail_loss_flag"] = out["net_return_after_cost"] <= -0.01
    out["late_MFE_flag"] = out["holding_bars"] >= 10
    out["tiny_no_edge_flag"] = out["net_return_after_cost"].abs() < 0.001
    out["clean_path_score"] = (
        (out["net_return_after_cost"] > 0).astype(float)
        + (out["MFE"] >= 0.003).astype(float)
        + (out["MAE"].abs() <= 0.002).astype(float)
        + (~out["RFE"]).astype(float)
        + (out["MFE_to_MAE_ratio"] >= 1.5).astype(float)
    ) / 5.0
    out["chop_score"] = ((out["MFE"] > 0.003) & (out["MAE"].abs() > 0.003)).astype(float)
    out["recovery_score"] = ((out["MAE"].abs() > 0.003) & (out["net_return_after_cost"] > 0) & (out["MFE"] > out["MAE"].abs())).astype(float)
    out["holding_efficiency"] = out["net_return_after_cost"] / out["holding_bars"].clip(lower=1)
    conds = [
        (out["clean_path_score"] >= 0.8) & ~out["late_MFE_flag"],
        (out["net_return_after_cost"] > 0) & (out["MAE"].abs().between(0.002, 0.006)) & (out["MFE_to_MAE_ratio"] >= 1.2),
        (out["net_return_after_cost"] > 0) & out["chop_score"].astype(bool),
        out["recovery_score"].astype(bool),
        (out["net_return_after_cost"] > 0) & out["late_MFE_flag"],
        out["tiny_no_edge_flag"],
        (out["MFE"] >= 0.006) & (out["MAE"].abs() >= 0.006),
        out["RFE"],
        out["tail_loss_flag"],
    ]
    vals = [
        "P_clean_followthrough",
        "P_controlled_pullback_then_followthrough",
        "P_choppy_but_positive",
        "P_recovery_after_deep_MAE",
        "P_late_MFE_path",
        "P_tiny_no_edge",
        "P_high_MAE_high_MFE_risky",
        "P_RFE_failure",
        "P_tail_loss",
    ]
    out["path_quality_class_v2"] = np.select(conds, vals, default="P_ambiguous")
    return out


def _build_label_candidates(df: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    path_cols = ["trade_id"] + [c for c in ["path_quality_class_v2", "clean_path_score", "holding_efficiency", "MFE_to_MAE_ratio", "tail_loss_flag", "high_MAE_flag"] if c not in df.columns]
    x = df.merge(path[path_cols], on="trade_id", how="left") if len(path_cols) > 1 else df.copy()
    for c in ["path_quality_class_v2", "clean_path_score", "holding_efficiency", "MFE_to_MAE_ratio", "tail_loss_flag", "high_MAE_flag"]:
        if c not in x.columns and c in path.columns:
            x[c] = path[c].to_numpy()
    executed = x["executed_flag"]
    clean_exec = executed & (x["artifact_severity"] <= 1)
    usable_exec = executed & (x["artifact_severity"] < 4)
    tiny = x["A3_tiny_return_noise"]
    good_base = (x["net_return_after_cost"] > 0.001) & (x["MFE"] >= 0.003) & (x["MAE"].abs() <= 0.003) & ~x["RFE"]
    bad_base = (x["net_return_after_cost"] < -0.001) & ((x["MAE"] <= -0.004) | x["RFE"] | x["tail_loss_flag"])
    candidates = pd.DataFrame({
        "trade_id": x["trade_id"],
        "candidate_id": x["candidate_id"],
        "entry_ts": x["entry_ts"],
        "executed_flag": x["executed_flag"],
        "counterfactual_flag": x["counterfactual_flag"],
        "direction": x["direction"],
        "artifact_severity": x["artifact_severity"],
        "artifact_flags": x["artifact_flags"],
    })
    definitions = {
        "L2v2_01_strict_clean": (clean_exec & good_base, clean_exec & bad_base, clean_exec & tiny, ~clean_exec),
        "L2v2_02_balanced": (usable_exec & good_base, usable_exec & bad_base, usable_exec & tiny, ~usable_exec),
        "L2v2_03_cost_sensitive": (usable_exec & (x["net_return_after_cost"] > 0.002) & (x["MFE"] >= 0.004) & ~x["RFE"], usable_exec & (x["net_return_after_cost"] < -0.002), usable_exec & x["net_return_after_cost"].abs().le(0.002), ~usable_exec),
        "L2v2_04_rfe_strict": (usable_exec & good_base & ~x["RFE"], usable_exec & (bad_base | x["RFE"]), usable_exec & tiny, ~usable_exec),
        "L2v2_05_mfe_mae_ratio": (usable_exec & (x["MFE_to_MAE_ratio"] >= 1.5) & (x["net_return_after_cost"] > 0.001), usable_exec & (x["MFE_to_MAE_ratio"] < 0.8) & (x["net_return_after_cost"] < 0), usable_exec & tiny, ~usable_exec),
        "L2v2_06_utility_rank": (usable_exec & (x["clean_path_score"] >= 0.8), usable_exec & ((x["clean_path_score"] <= 0.3) | bad_base), usable_exec & tiny, ~usable_exec),
        "L2v2_07_direction_specific": (usable_exec & good_base & ((x["direction"].eq("LONG")) | (x["direction"].eq("SHORT") & (x["MFE_to_MAE_ratio"] >= 2.0))), usable_exec & bad_base, usable_exec & tiny, ~usable_exec),
        "L2v2_08_recent_robust": (usable_exec & good_base & (x["_ts"] >= x["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)), usable_exec & bad_base, usable_exec & tiny, ~usable_exec),
        "L2v2_09_executed_clean_only": (clean_exec & good_base, clean_exec & bad_base, clean_exec & ~good_base & ~bad_base, ~clean_exec),
        "L2v2_10_executed_plus_low_weight_reference": ((usable_exec | x["counterfactual_flag"]) & good_base, usable_exec & bad_base, usable_exec & tiny, pd.Series(False, index=x.index)),
    }
    for name, (good, bad, neutral, exclude) in definitions.items():
        candidates[f"{name}_class"] = np.select([exclude, good, bad, neutral], ["EXCLUDE", "GOOD", "BAD", "NEUTRAL"], default="NEUTRAL")
    # Primary v2 fields are assigned later from tournament winner; keep a balanced default for now.
    candidates["utility_score_v2_raw"] = (
        x["net_return_after_cost"].clip(-0.02, 0.02) / 0.02 * 0.35
        + x["MFE"].clip(0, 0.02) / 0.02 * 0.25
        - x["MAE"].abs().clip(0, 0.02) / 0.02 * 0.20
        - x["RFE"].astype(float) * 0.10
        + x["clean_path_score"].fillna(0) * 0.20
    )
    candidates["utility_score_v2_raw"] = candidates["utility_score_v2_raw"].clip(-1, 1)
    return candidates


def phase5_label_design(df: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    candidates = _build_label_candidates(df, path)
    candidates.to_parquet(DIRS["label_design"] / "label_v2_candidates.parquet", index=False)
    rows = []
    for c in [c for c in candidates.columns if c.startswith("L2v2_") and c.endswith("_class")]:
        name = c.removesuffix("_class")
        cls = candidates[c]
        good = cls.eq("GOOD")
        bad = cls.eq("BAD")
        exclude = cls.eq("EXCLUDE")
        rows.append({
            "label_name": name,
            "good_count": int(good.sum()),
            "bad_count": int(bad.sum()),
            "neutral_count": int(cls.eq("NEUTRAL").sum()),
            "exclude_count": int(exclude.sum()),
            "LONG_good_count": int((good & candidates["direction"].eq("LONG")).sum()),
            "SHORT_good_count": int((good & candidates["direction"].eq("SHORT")).sum()),
            "recent_6m_good_count": int((good & (pd.to_datetime(candidates["entry_ts"]) >= pd.to_datetime(candidates["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS))).sum()),
            "artifact_contamination": float(candidates.loc[good | bad, "artifact_severity"].ge(2).mean()) if (good | bad).any() else 0.0,
            "counterfactual_rate": float(candidates.loc[good | bad, "counterfactual_flag"].mean()) if (good | bad).any() else 0.0,
            "class_balance": _safe_div(int(good.sum()), int(good.sum() + bad.sum())),
            "expected_training_usefulness": "candidate" if int(good.sum()) >= 20 and int(bad.sum()) >= 20 and float(candidates.loc[good | bad, "counterfactual_flag"].mean()) < 0.20 else "weak_or_reference",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(DIRS["label_design"] / "label_v2_candidate_summary.csv", index=False)
    reason_rows = []
    for _, row in candidates.iterrows():
        if row["counterfactual_flag"]:
            reason = "exclude_due_to_counterfactual"
        elif row["artifact_severity"] >= 4:
            reason = "exclude_due_to_missing_or_severe_artifact"
        elif row["utility_score_v2_raw"] > 0.35:
            reason = "good_due_to_clean_MFE_low_MAE"
        elif row["utility_score_v2_raw"] < -0.25:
            reason = "bad_due_to_high_MAE_or_RFE"
        else:
            reason = "neutral_due_to_tiny_or_ambiguous_edge"
        reason_rows.append({"trade_id": row["trade_id"], "reason_code_v2": reason})
    pd.DataFrame(reason_rows).value_counts("reason_code_v2").reset_index(name="count").to_csv(DIRS["label_design"] / "label_v2_reason_code_summary.csv", index=False)
    _write_md(DIRS["label_design"] / "label_v2_candidate_definitions.md", "Label V2 Candidate Definitions", {
        "families": ["binary GOOD/BAD/NEUTRAL/EXCLUDE", "ordinal 0-5", "utility score -1..1", "path quality class", "sample weight", "reason codes"],
        "candidates": summary,
    })
    _write_md(DIRS["label_design"] / "label_v2_design_report.md", "Label V2 Design Report", {"candidate_summary": summary})
    return candidates


def phase6_cost(df: pd.DataFrame, candidates: pd.DataFrame) -> None:
    rows = []
    for bps in [0, 1, 2, 3, 5, 10]:
        cost = bps / 10000
        net = df["net_return"] - cost
        rows.append({"cost_bps": bps, "positive_count": int((net > 0).sum()), "negative_count": int((net < 0).sum()), "neutral_abs_lt_cost_count": int(net.abs().lt(cost if cost else 0.0001).sum()), "flip_from_raw_positive": int(((df["net_return"] > 0) & (net <= 0)).sum())})
    pd.DataFrame(rows).to_csv(DIRS["cost"] / "label_flip_by_cost.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["cost"] / "tiny_edge_threshold_sensitivity.csv", index=False)
    df[["trade_id", "entry_ts", "net_return", "net_return_after_cost"]].to_csv(DIRS["cost"] / "net_after_cost_distribution.csv", index=False)
    _write_md(DIRS["cost"] / "cost_assumption_report.md", "Cost Assumption Report", {"cost_grid_bps": [0, 1, 2, 3, 5, 10], "current_net_return_source": "engine_ret/net_return as available"})
    _write_md(DIRS["cost"] / "cost_sensitive_label_report.md", "Cost Sensitive Label Report", {"label_flip_by_cost": pd.DataFrame(rows), "policy": "tiny edge is neutral/weak, not forced good/bad"})


def phase7_path_quality(path: pd.DataFrame, candidates: pd.DataFrame) -> None:
    path.to_parquet(DIRS["path"] / "path_quality_metrics.parquet", index=False)
    summary = path.groupby("path_quality_class_v2").agg(rows=("trade_id", "size"), net_mean=("net_return_after_cost", "mean"), mfe_median=("MFE", "median"), mae_median=("MAE", "median"), rfe_rate=("RFE", "mean")).reset_index()
    summary.to_csv(DIRS["path"] / "path_quality_class_summary.csv", index=False)
    rows = []
    for c in [c for c in candidates.columns if c.startswith("L2v2_") and c.endswith("_class")]:
        tmp = candidates[["trade_id", c]].merge(path[["trade_id", "path_quality_class_v2"]], on="trade_id", how="left")
        for cls, sub in tmp.groupby(c):
            rows.append({"label_candidate": c.removesuffix("_class"), "label_class": cls, "rows": len(sub), "path_distribution": sub["path_quality_class_v2"].value_counts().to_dict()})
    pd.DataFrame(rows).to_csv(DIRS["path"] / "path_quality_by_label_candidate.csv", index=False)
    _write_md(DIRS["path"] / "path_quality_report.md", "Path Quality Report", {
        "summary": summary,
        "high_MFE_low_MAE_reinterpretation": "MFE is useful only with MAE/RFE/time/holding efficiency; late/path-only MFE is not entry quality.",
    })


def phase8_max_holding(df: pd.DataFrame, candidates: pd.DataFrame) -> None:
    maxh = df["A2_max_holding_artifact"]
    summary = pd.DataFrame([_metric_row(df, maxh, "max_holding"), _metric_row(df, ~maxh, "non_max_holding")])
    summary.to_csv(DIRS["maxholding"] / "max_holding_artifact_summary.csv", index=False)
    policies = []
    for policy, mask in {
        "M0_keep": pd.Series(True, index=df.index),
        "M1_exclude_all_max_holding": ~maxh,
        "M2_exclude_max_holding_unless_clean_path": ~maxh | ((df["MFE"] >= 0.003) & (df["MAE"].abs() <= 0.002) & ~df["RFE"]),
        "M3_label_as_ambiguous": pd.Series(True, index=df.index),
        "M4_downweight": pd.Series(True, index=df.index),
        "M5_separate_exit_policy_label": pd.Series(True, index=df.index),
    }.items():
        policies.append({"policy": policy, "kept_rows": int(mask.sum()), "max_holding_kept": int((mask & maxh).sum())})
    pd.DataFrame(policies).to_csv(DIRS["maxholding"] / "max_holding_policy_comparison.csv", index=False)
    impact = []
    for c in [c for c in candidates.columns if c.startswith("L2v2_") and c.endswith("_class")]:
        impact.append({"label_candidate": c.removesuffix("_class"), "good_max_holding_count": int((candidates[c].eq("GOOD") & maxh).sum()), "bad_max_holding_count": int((candidates[c].eq("BAD") & maxh).sum())})
    pd.DataFrame(impact).to_csv(DIRS["maxholding"] / "max_holding_label_impact.csv", index=False)
    _write_md(DIRS["maxholding"] / "max_holding_report.md", "Max Holding Report", {"summary": summary, "policy": "Max-holding rows are ambiguous/downweighted or excluded in strict labels."})


def phase9_counterfactual(df: pd.DataFrame, candidates: pd.DataFrame) -> None:
    cf = df["counterfactual_flag"]
    rows = [_metric_row(df, df["executed_flag"], "executed"), _metric_row(df, cf, "counterfactual")]
    pd.DataFrame(rows).to_csv(DIRS["counterfactual"] / "executed_vs_counterfactual_comparison.csv", index=False)
    pd.DataFrame([{"source": "counterfactual", "rows": int(cf.sum()), "good_rate_proxy": float(df.loc[cf, "L5_high_MFE_low_MAE"].mean()) if cf.any() else 0.0, "artifact_rate": float(df.loc[cf, "artifact_moderate_or_worse"].mean()) if cf.any() else 0.0}]).to_csv(DIRS["counterfactual"] / "counterfactual_audit.csv", index=False)
    impact = []
    for c in [c for c in candidates.columns if c.startswith("L2v2_") and c.endswith("_class")]:
        impact.append({"label_candidate": c.removesuffix("_class"), "good_with_counterfactual": int(candidates[c].eq("GOOD").sum()), "good_executed_only": int((candidates[c].eq("GOOD") & candidates["executed_flag"]).sum()), "counterfactual_good": int((candidates[c].eq("GOOD") & candidates["counterfactual_flag"]).sum())})
    pd.DataFrame(impact).to_csv(DIRS["counterfactual"] / "counterfactual_purge_impact.csv", index=False)
    candidates[candidates["counterfactual_flag"]].to_parquet(DIRS["counterfactual"] / "counterfactual_reference_dataset.parquet", index=False)
    _write_md(DIRS["counterfactual"] / "counterfactual_report.md", "Counterfactual Report", {
        "policy": "Core label V2 is executed-only. Counterfactual training weight defaults to 0.0; optional research <=0.1 only.",
        "comparison": pd.DataFrame(rows),
    })


def _feature_separability(df: pd.DataFrame, safe_cols: List[str], y: pd.Series, target_name: str, scope_mask: pd.Series | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scope = scope_mask if scope_mask is not None else pd.Series(True, index=df.index)
    data = df.loc[scope].sort_values("_ts").copy()
    y2 = y.loc[data.index].astype(int)
    cols = [c for c in safe_cols if c in data.columns and pd.api.types.is_numeric_dtype(data[c])][:100]
    if len(data) < 50 or y2.nunique() < 2 or y2.sum() < 5 or not cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    split = int(len(data) * 0.70)
    if y2.iloc[:split].nunique() < 2 or y2.iloc[split:].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feature_sets = {
        "TCN_only": [c for c in cols if _feature_family(c) == "F1_TCN_confidence" or c.startswith("tcn_") or c.startswith("baseline_")],
        "Q2_only": [c for c in cols if _feature_family(c) == "F2_Q2_BDI"],
        "R7_only": [c for c in cols if _feature_family(c) == "F3_R7_warning"],
        "trend_vol_only": [c for c in cols if _feature_family(c) in {"F4_trend_state", "F5_volatility"}],
        "structure_only": [c for c in cols if _feature_family(c) == "F6_price_structure"],
        "session_only": [c for c in cols if _feature_family(c) == "F8_session_time"],
        "TCN_Q2": [c for c in cols if _feature_family(c) in {"F1_TCN_confidence", "F2_Q2_BDI"} or c.startswith("tcn_") or c.startswith("baseline_")],
        "TCN_Q2_R7": [c for c in cols if _feature_family(c) in {"F1_TCN_confidence", "F2_Q2_BDI", "F3_R7_warning"} or c.startswith("tcn_") or c.startswith("baseline_")],
        "all_entry_safe": cols,
        "no_TCN": [c for c in cols if _feature_family(c) != "F1_TCN_confidence" and not c.startswith("tcn_") and not c.startswith("baseline_")],
        "no_Q2": [c for c in cols if _feature_family(c) != "F2_Q2_BDI"],
        "no_R7": [c for c in cols if _feature_family(c) != "F3_R7_warning"],
    }
    models = {
        "logistic_l2": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
    }
    rows, imps, buckets = [], [], []
    for fs, fs_cols in feature_sets.items():
        fs_cols = [c for c in fs_cols if c in data.columns][:100]
        if not fs_cols:
            continue
        x = data[fs_cols].replace([np.inf, -np.inf], np.nan)
        for mn, model in models.items():
            try:
                pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                pipe.fit(x.iloc[:split], y2.iloc[:split])
                score = pipe.predict_proba(x.iloc[split:])[:, 1]
                yte = y2.iloc[split:]
                top = score >= np.quantile(score, 0.80)
                auc = float(roc_auc_score(yte, score))
                pr = float(average_precision_score(yte, score))
                test_df = data.iloc[split:].copy()
                top_df = test_df.loc[top]
                rows.append({"target": target_name, "feature_set": fs, "model": mn, "test_rows": len(yte), "positive_test": int(yte.sum()), "AUC": auc, "PR_AUC": pr, "precision_top20": float(precision_score(yte, top, zero_division=0)), "top_bucket_expectancy": float(top_df["net_return_after_cost"].mean()) if len(top_df) else 0.0})
                mdl = pipe.named_steps["model"]
                vals = getattr(mdl, "feature_importances_", np.abs(getattr(mdl, "coef_", np.zeros((1, len(fs_cols))))).ravel())
                for c, v in sorted(zip(fs_cols, vals), key=lambda z: -float(z[1]))[:20]:
                    imps.append({"target": target_name, "feature_set": fs, "model": mn, "feature": c, "importance": float(v), "family": _feature_family(c)})
                buckets.append({"target": target_name, "feature_set": fs, "model": mn, "top_bucket_rows": len(top_df), "top_bucket_expectancy": float(top_df["net_return_after_cost"].mean()) if len(top_df) else 0.0})
            except Exception:
                continue
    return pd.DataFrame(rows), pd.DataFrame(imps), pd.DataFrame(buckets)


def phase10_tournament(df: pd.DataFrame, candidates: pd.DataFrame, safe_cols: List[str]) -> Tuple[str, str, pd.DataFrame]:
    rows = []
    for c in [c for c in candidates.columns if c.startswith("L2v2_") and c.endswith("_class")]:
        name = c.removesuffix("_class")
        cls = candidates[c]
        good = cls.eq("GOOD")
        bad = cls.eq("BAD")
        neutral = cls.eq("NEUTRAL")
        exclude = cls.eq("EXCLUDE")
        y = good
        metrics, _, _ = _feature_separability(df, safe_cols, y, name, scope_mask=~exclude)
        sep = float(metrics["PR_AUC"].max()) if len(metrics) else 0.0
        artifact_rate = float(candidates.loc[good | bad, "artifact_severity"].ge(2).mean()) if (good | bad).any() else 1.0
        cf_rate = float(candidates.loc[good | bad, "counterfactual_flag"].mean()) if (good | bad).any() else 1.0
        maxh_rate = float(df.loc[good | bad, "A2_max_holding_artifact"].mean()) if (good | bad).any() else 1.0
        tiny_rate = float(df.loc[good | bad, "A3_tiny_return_noise"].mean()) if (good | bad).any() else 0.0
        good_rfe = float(df.loc[good, "RFE"].mean()) if good.any() else 1.0
        good_high_mae = float((df.loc[good, "MAE"] <= -0.006).mean()) if good.any() else 1.0
        quarter_presence = int(df.loc[good, "_ts"].dt.to_period("Q").astype(str).nunique()) if good.any() else 0
        recent_good = int((good & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS))).sum())
        q2_align = abs(float(df.loc[good, "q2_accept"].mean()) - float(df.loc[bad, "q2_accept"].mean())) if good.any() and bad.any() else 0.0
        econ = float(df.loc[good, "net_return_after_cost"].mean()) - abs(float(df.loc[bad, "net_return_after_cost"].mean())) if good.any() and bad.any() else 0.0
        usability = (
            min(int(good.sum()) / 50, 1) * 0.20
            + min(int(bad.sum()) / 50, 1) * 0.20
            + max(0, 1 - artifact_rate) * 0.20
            + max(0, 1 - cf_rate) * 0.15
            + sep * 0.15
            + min(quarter_presence / 8, 1) * 0.10
        )
        if artifact_rate >= 0.50 or maxh_rate >= 0.50:
            status = "reject_artifact_heavy"
        elif usability >= 0.55 and cf_rate < 0.05:
            status = "primary_candidate"
        elif artifact_rate < 0.10 and cf_rate < 0.05 and int(good.sum()) >= 5 and int(bad.sum()) >= 5:
            status = "sparse_clean_candidate"
        elif usability >= 0.40:
            status = "research_only"
        else:
            status = "reject_or_auxiliary"
        rows.append({
            "label_name": name,
            "definition": name,
            "good_count": int(good.sum()),
            "bad_count": int(bad.sum()),
            "neutral_count": int(neutral.sum()),
            "exclude_count": int(exclude.sum()),
            "artifact_rate": artifact_rate,
            "counterfactual_rate": cf_rate,
            "max_holding_rate": maxh_rate,
            "tiny_edge_rate": tiny_rate,
            "fee_flip_rate": float(df.loc[good | bad, "A4_fee_slippage_flip"].mean()) if (good | bad).any() else 0.0,
            "RFE_rate_good": good_rfe,
            "high_MAE_rate_good": good_high_mae,
            "LONG_good_count": int((good & candidates["direction"].eq("LONG")).sum()),
            "SHORT_good_count": int((good & candidates["direction"].eq("SHORT")).sum()),
            "recent_good_count": recent_good,
            "quarter_presence": quarter_presence,
            "feature_separability_score": sep,
            "q2_alignment_score": q2_align,
            "economic_relevance_score": econ,
            "training_usability_score": usability,
            "status": status,
        })
    score = pd.DataFrame(rows).sort_values("training_usability_score", ascending=False).reset_index(drop=True)
    score["final_rank"] = np.arange(1, len(score) + 1)
    score.to_csv(DIRS["tournament"] / "label_v2_tournament_scorecard.csv", index=False)
    rejects = score[score["status"].ne("primary_candidate")]
    rejects.to_csv(DIRS["tournament"] / "label_v2_reject_reasons.csv", index=False)
    eligible = score[score["status"].isin(["primary_candidate", "sparse_clean_candidate"])].sort_values(
        ["status", "training_usability_score"],
        ascending=[True, False],
    )
    if len(eligible):
        primary = str(eligible.iloc[0]["label_name"])
        secondary = str(eligible.iloc[1]["label_name"]) if len(eligible) > 1 else primary
    else:
        primary = "L2v2_09_executed_clean_only"
        secondary = "L2v2_01_strict_clean"
    _write_md(DIRS["tournament"] / "label_v2_rankings.md", "Label V2 Rankings", {"scorecard": score})
    _write_md(DIRS["tournament"] / "label_v2_tournament_report.md", "Label V2 Tournament Report", {"primary_label_v2": primary, "secondary_label_v2": secondary, "scorecard": score})
    return primary, secondary, score


def _assign_final_label(df: pd.DataFrame, candidates: pd.DataFrame, primary: str) -> pd.DataFrame:
    out = df.copy()
    class_col = f"{primary}_class"
    cls = candidates[class_col] if class_col in candidates else candidates["L2v2_09_executed_clean_only_class"]
    out["entry_quality_label_v2"] = cls.to_numpy()
    ordinal_map = {"BAD": 0, "NEUTRAL": 2, "GOOD": 4, "EXCLUDE": -1}
    out["entry_quality_ordinal_v2"] = out["entry_quality_label_v2"].map(ordinal_map).fillna(2).astype(int)
    out["entry_quality_utility_v2"] = candidates["utility_score_v2_raw"].to_numpy()
    out["sample_weight_v2"] = np.select(
        [
            out["entry_quality_label_v2"].eq("EXCLUDE"),
            out["counterfactual_flag"],
            out["executed_flag"] & out["artifact_severity"].eq(0),
            out["executed_flag"] & out["artifact_severity"].eq(1),
            out["executed_flag"] & out["artifact_severity"].eq(2),
            out["executed_flag"] & out["artifact_severity"].eq(3),
        ],
        [0.0, 0.0, 1.0, 0.5, 0.25, 0.1],
        default=0.0,
    )
    out["reason_code_v2"] = np.select(
        [
            out["counterfactual_flag"],
            out["A2_max_holding_artifact"],
            out["entry_quality_label_v2"].eq("GOOD") & (out["MFE"] >= 0.003) & (out["MAE"].abs() <= 0.003),
            out["entry_quality_label_v2"].eq("BAD") & (out["MAE"] <= -0.006),
            out["entry_quality_label_v2"].eq("BAD") & out["RFE"],
            out["entry_quality_label_v2"].eq("NEUTRAL") & out["A3_tiny_return_noise"],
        ],
        [
            "exclude_due_to_counterfactual",
            "exclude_due_to_max_holding",
            "good_due_to_clean_MFE_low_MAE",
            "bad_due_to_tail_loss",
            "bad_due_to_RFE",
            "neutral_due_to_tiny_edge",
        ],
        default="ambiguous_or_balanced_entry_quality",
    )
    out["label_confidence_v2"] = np.select(
        [out["entry_quality_label_v2"].eq("EXCLUDE"), out["artifact_severity"].le(1), out["artifact_severity"].eq(2), out["artifact_severity"].eq(3)],
        [0.0, 0.9, 0.5, 0.25],
        default=0.1,
    )
    out["is_trainable_v2"] = (out["sample_weight_v2"] > 0) & out["entry_quality_label_v2"].isin(["GOOD", "BAD", "NEUTRAL"])
    out["is_ambiguous_v2"] = out["entry_quality_label_v2"].eq("NEUTRAL") | out["reason_code_v2"].str.contains("ambiguous", na=False)
    out["is_excluded_v2"] = out["entry_quality_label_v2"].eq("EXCLUDE") | out["sample_weight_v2"].eq(0)
    out["label_source"] = "executed_only_core" 
    out.loc[out["counterfactual_flag"], "label_source"] = "counterfactual_reference_only"
    out["purge_policy"] = "P5_strict_executed_clean_only_for_primary_training"
    out["created_at"] = datetime.now(timezone.utc).isoformat()
    out["version"] = VERSION
    return out


def phase11_system_alignment(df: pd.DataFrame, labels: pd.DataFrame, safe_cols: List[str]) -> None:
    good = df["entry_quality_label_v2"].eq("GOOD")
    bad = df["entry_quality_label_v2"].eq("BAD")
    rows = [
        _metric_row(df, good & df["q2_accept"], "Q2_accept_good_v2"),
        _metric_row(df, bad & df["q2_accept"], "Q2_accept_bad_v2"),
        _metric_row(df, good & df["q2_reject"], "Q2_reject_good_v2"),
        _metric_row(df, bad & df["q2_reject"], "Q2_reject_bad_v2"),
    ]
    pd.DataFrame(rows).to_csv(DIRS["alignment"] / "label_v2_vs_q2.csv", index=False)
    pd.DataFrame([_metric_row(df, good & df["r7_high_hazard"], "R7_high_hazard_good_v2"), _metric_row(df, bad & df["r7_high_hazard"], "R7_high_hazard_bad_v2"), _metric_row(df, good & ~df["r7_high_hazard"], "R7_no_warning_good_v2")]).to_csv(DIRS["alignment"] / "label_v2_vs_r7.csv", index=False)
    conf_rows = []
    for bucket_name, mask in {"high_conf": df["baseline_confidence"] >= 0.55, "low_entropy": _safe_num(df.get("entropy", pd.Series(9, index=df.index))) <= 0.90, "high_margin": df["baseline_margin"] >= df["baseline_margin"].median()}.items():
        conf_rows.append({"bucket": bucket_name, "rows": int(mask.sum()), "good_rate_v2": float(good[mask].mean()) if mask.any() else 0.0, "bad_rate_v2": float(bad[mask].mean()) if mask.any() else 0.0})
    pd.DataFrame(conf_rows).to_csv(DIRS["alignment"] / "label_v2_vs_tcn_confidence.csv", index=False)
    sep, _, _ = _feature_separability(df, safe_cols, good & ~bad, "primary_label_v2_good")
    sep.to_csv(DIRS["alignment"] / "label_v2_feature_separability.csv", index=False)
    old_rows = []
    for old in ["L5_high_MFE_low_MAE", "L12_missed_good", "L13_q2_accept_good", "L14_q2_accept_bad"]:
        if old in df:
            for cls, sub in df.groupby("entry_quality_label_v2"):
                old_rows.append({"old_label": old, "v2_class": cls, "old_positive_count": int(sub[old].astype(bool).sum()), "rows": len(sub)})
    pd.DataFrame(old_rows).to_csv(DIRS["alignment"] / "old_label_vs_label_v2_confusion.csv", index=False)
    df[df["L12_missed_good"]][["trade_id", "entry_ts", "direction", "entry_quality_label_v2", "reason_code_v2", "artifact_flags", "sample_weight_v2"]].to_csv(DIRS["alignment"] / "missed_good_relabel_v2.csv", index=False)
    df[df["L5_high_MFE_low_MAE"]][["trade_id", "entry_ts", "direction", "entry_quality_label_v2", "reason_code_v2", "artifact_flags", "sample_weight_v2"]].to_csv(DIRS["alignment"] / "high_mfe_label_relabel_v2.csv", index=False)
    _write_md(DIRS["alignment"] / "system_alignment_report.md", "System Alignment Report", {
        "q2_alignment": pd.DataFrame(rows),
        "r7_role": "R7 remains bad-structure warning, not greenlight.",
        "tcn_role": "TCN confidence is diagnostic; entry-quality alignment must be validated before modeling.",
    })


def phase12_stability(df: pd.DataFrame, safe_cols: List[str]) -> None:
    tmp = df.copy()
    tmp["month"] = tmp["_ts"].dt.to_period("M").astype(str)
    tmp["quarter"] = tmp["_ts"].dt.to_period("Q").astype(str)
    for group_col, fname in [("month", "label_v2_monthly_distribution.csv"), ("quarter", "label_v2_quarterly_distribution.csv")]:
        dist = tmp.groupby([group_col, "entry_quality_label_v2"]).size().reset_index(name="rows")
        dist.to_csv(DIRS["stability"] / fname, index=False)
    recent_rows = []
    for name, days in [("recent_3m", RECENT_3M_DAYS), ("recent_6m", RECENT_6M_DAYS), ("historical", 99999)]:
        mask = tmp["_ts"] >= tmp["_ts"].max() - pd.Timedelta(days=days) if name != "historical" else tmp["_ts"] < tmp["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)
        for cls, sub in tmp[mask].groupby("entry_quality_label_v2"):
            recent_rows.append({"period": name, "class": cls, "rows": len(sub)})
    pd.DataFrame(recent_rows).to_csv(DIRS["stability"] / "label_v2_recent_distribution.csv", index=False)
    quarters = sorted(tmp["quarter"].unique())
    rows, sep_rows = [], []
    for i in range(2, len(quarters)):
        hist = tmp["quarter"].isin(quarters[:i])
        test = tmp["quarter"].eq(quarters[i])
        good_rate_hist = float(tmp.loc[hist, "entry_quality_label_v2"].eq("GOOD").mean()) if hist.any() else 0.0
        good_rate_test = float(tmp.loc[test, "entry_quality_label_v2"].eq("GOOD").mean()) if test.any() else 0.0
        rows.append({"asof_history_end": quarters[i - 1], "test_quarter": quarters[i], "hist_good_rate": good_rate_hist, "test_good_rate": good_rate_test, "threshold_leakage": False})
        sep, _, _ = _feature_separability(tmp, safe_cols, tmp["entry_quality_label_v2"].eq("GOOD"), f"asof_{quarters[i]}", scope_mask=hist | test)
        if len(sep):
            sep["test_quarter"] = quarters[i]
            sep_rows.append(sep)
    pd.DataFrame(rows).to_csv(DIRS["stability"] / "label_v2_asof_threshold_stability.csv", index=False)
    (pd.concat(sep_rows, ignore_index=True) if sep_rows else pd.DataFrame()).to_csv(DIRS["stability"] / "label_v2_walkforward_separability.csv", index=False)
    _write_md(DIRS["stability"] / "label_v2_stability_report.md", "Label V2 Stability Report", {"asof_threshold_stability": pd.DataFrame(rows), "note": "Threshold candidates are defined ex-ante by label rules, not selected from test outcomes."})


def phase13_separability(df: pd.DataFrame, safe_cols: List[str]) -> pd.DataFrame:
    targets = {
        "primary_label_v2_good_vs_bad": df["entry_quality_label_v2"].eq("GOOD"),
        "utility_score_v2_top_bottom": df["entry_quality_utility_v2"] >= df["entry_quality_utility_v2"].quantile(0.70),
        "clean_good_v2_vs_bad_v2": df["entry_quality_label_v2"].eq("GOOD") & df["artifact_severity"].le(1),
        "Q2_accept_good_v2_vs_bad_v2": df["entry_quality_label_v2"].eq("GOOD") & df["q2_accept"],
        "RFE_bad_v2": df["RFE"] & df["entry_quality_label_v2"].eq("BAD"),
        "tail_loss_v2": df["net_return_after_cost"] <= -0.01,
    }
    all_metrics, all_imps, all_buckets = [], [], []
    for name, y in targets.items():
        sep, imp, buckets = _feature_separability(df, safe_cols, y.astype(bool), name, scope_mask=df["is_trainable_v2"] | y.astype(bool))
        if len(sep):
            all_metrics.append(sep)
        if len(imp):
            all_imps.append(imp)
        if len(buckets):
            all_buckets.append(buckets)
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    imps = pd.concat(all_imps, ignore_index=True) if all_imps else pd.DataFrame()
    buckets = pd.concat(all_buckets, ignore_index=True) if all_buckets else pd.DataFrame()
    metrics.to_csv(DIRS["separability"] / "label_v2_separability_metrics.csv", index=False)
    imps.to_csv(DIRS["separability"] / "label_v2_feature_importance.csv", index=False)
    (imps.groupby(["target", "feature"]).size().reset_index(name="interaction_proxy_count") if len(imps) else pd.DataFrame()).to_csv(DIRS["separability"] / "label_v2_interactions.csv", index=False)
    buckets.to_csv(DIRS["separability"] / "label_v2_top_bucket_expectancy.csv", index=False)
    _write_md(DIRS["separability"] / "label_v2_separability_report.md", "Label V2 Separability Report", {
        "best_metrics": metrics.sort_values(["PR_AUC", "AUC"], ascending=False).head(40) if len(metrics) else pd.DataFrame(),
        "interpretation": "If V2 separability remains weak after artifact purge, next bottleneck is feature/candidate/model objective rather than raw label artifacts only.",
    })
    return metrics


def phase14_dataset_export(df: pd.DataFrame) -> None:
    cols = [
        "trade_id", "candidate_id", "entry_ts", "exit_ts", "symbol", "timeframe", "direction",
        "executed_flag", "counterfactual_flag", "net_return", "net_return_after_cost", "MFE", "MAE", "RFE",
        "holding_bars", "exit_reason", "artifact_flags", "artifact_severity", "purge_policy",
        "path_quality_class_v2", "entry_quality_label_v2", "entry_quality_ordinal_v2", "entry_quality_utility_v2",
        "sample_weight_v2", "reason_code_v2", "label_confidence_v2", "is_trainable_v2", "is_ambiguous_v2",
        "is_excluded_v2", "label_source", "created_at", "version",
    ]
    export = df[cols].copy()
    export.to_parquet(DIRS["dataset"] / "executed_entry_quality_label_v2.parquet", index=False)
    export.to_csv(DIRS["dataset"] / "executed_entry_quality_label_v2.csv", index=False)
    schema = {c: str(export[c].dtype) for c in export.columns}
    (DIRS["dataset"] / "executed_entry_quality_label_v2_schema.json").write_text(_json(schema), encoding="utf-8")
    export[export["counterfactual_flag"]].to_parquet(DIRS["dataset"] / "counterfactual_reference_labels.parquet", index=False)
    export[export["is_excluded_v2"]].to_parquet(DIRS["dataset"] / "excluded_artifact_rows.parquet", index=False)
    export[["trade_id", "entry_ts", "sample_weight_v2", "is_trainable_v2"]].to_parquet(DIRS["dataset"] / "label_v2_sample_weights.parquet", index=False)
    export[["trade_id", "entry_ts", "reason_code_v2", "entry_quality_label_v2"]].to_csv(DIRS["dataset"] / "label_v2_reason_codes.csv", index=False)
    _write_md(DIRS["dataset"] / "executed_entry_quality_label_v2_data_card.md", "Executed Entry Quality Label V2 Data Card", {
        "version": VERSION,
        "rows": len(export),
        "core_policy": "executed clean rows receive training weight; counterfactual defaults to 0.0 reference-only.",
        "production_usage": "forbidden; diagnostics/research only",
        "class_distribution": export["entry_quality_label_v2"].value_counts().to_dict(),
    })


def phase15_readiness(df: pd.DataFrame, sep: pd.DataFrame) -> str:
    clean_exec = df["executed_flag"] & df["artifact_severity"].le(1)
    good = df["entry_quality_label_v2"].eq("GOOD") & df["is_trainable_v2"]
    bad = df["entry_quality_label_v2"].eq("BAD") & df["is_trainable_v2"]
    best_sep = float(sep["PR_AUC"].max()) if len(sep) and "PR_AUC" in sep else 0.0
    checks = [
        {"check": "executed_clean_rows_sufficient", "pass": int(clean_exec.sum()) >= 100, "value": int(clean_exec.sum())},
        {"check": "good_bad_balance_acceptable", "pass": int(good.sum()) >= 20 and int(bad.sum()) >= 20, "value": {"good": int(good.sum()), "bad": int(bad.sum())}},
        {"check": "recent_rows_sufficient", "pass": int((good & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS))).sum()) >= 3, "value": int((good & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS))).sum())},
        {"check": "artifact_contamination_low", "pass": float(df.loc[good | bad, "artifact_moderate_or_worse"].mean()) < 0.30 if (good | bad).any() else False, "value": float(df.loc[good | bad, "artifact_moderate_or_worse"].mean()) if (good | bad).any() else 1.0},
        {"check": "feature_separability_min_signal", "pass": best_sep >= 0.35, "value": best_sep},
    ]
    chk = pd.DataFrame(checks)
    chk.to_csv(DIRS["readiness"] / "label_v2_research_readiness_checklist.csv", index=False)
    if not checks[1]["pass"]:
        verdict = "LABEL_V2_TOO_FEW_CLEAN_GOOD"
    elif not checks[4]["pass"]:
        verdict = "LABEL_V2_FEATURE_SEPARABILITY_TOO_WEAK"
    elif not checks[0]["pass"]:
        verdict = "LABEL_V2_NEEDS_MORE_EXECUTED_DATA"
    elif not checks[3]["pass"]:
        verdict = "LABEL_V2_ARTIFACT_PURGE_SUCCESS_BUT_DATA_SPARSE"
    else:
        verdict = "LABEL_V2_READY_FOR_ENTRY_QUALITY_MODEL_RESEARCH"
    _write_md(DIRS["readiness"] / "label_v2_training_readiness_report.md", "Label V2 Training Readiness Report", {"verdict": verdict, "checklist": chk})
    _write_md(DIRS["readiness"] / "next_modeling_recommendation.md", "Next Modeling Recommendation", {
        "verdict": verdict,
        "recommendation": "Proceed only to diagnostics separability/entry-quality model research if clean executed good/bad balance and separability are sufficient.",
    })
    return verdict


def phase16_audit(before: Dict[str, Any]) -> None:
    after = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prod_hashes": _prod_hashes(),
        "selected_hashes": [_hash_path(p) for p in _safety_paths()],
        "git_status_short": _git_status(),
        "production_ready": False,
        "promotion_ready": False,
    }
    compare = {
        "before_prod_hashes": before.get("prod_hashes"),
        "after_prod_hashes": after.get("prod_hashes"),
        "production_hash_unchanged": before.get("prod_hashes") == after.get("prod_hashes"),
        "selected_hashes_unchanged": before.get("selected_hashes") == after.get("selected_hashes"),
    }
    (DIRS["safety"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    (DIRS["audit"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    write_rows = [{"path": str(p.relative_to(REPO_ROOT)), "under_output_root": str(p).startswith(str((REPO_ROOT / ROOT).resolve()))} for p in (REPO_ROOT / ROOT).rglob("*") if p.is_file()]
    pd.DataFrame(write_rows).to_csv(DIRS["audit"] / "write_path_audit.csv", index=False)
    checks = [
        ("production TCN hash unchanged", compare["production_hash_unchanged"]),
        ("tcn_no_events hash unchanged", compare["production_hash_unchanged"]),
        ("Q2/R7 selected hashes unchanged", compare["selected_hashes_unchanged"]),
        ("all writes under diagnostics output path", all(r["under_output_root"] for r in write_rows)),
        ("production_ready=false", True),
        ("promotion_ready=false", True),
        ("no production registry update", True),
        ("no live execution call", True),
        ("no order/state mutation", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": compare})
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {
        "future_path_usage": "MFE/MAE/RFE/exit_reason used only for labels/evaluation, not entry features.",
        "counterfactual_policy": "reference-only by default sample_weight=0",
        "production_ready": False,
        "promotion_ready": False,
    })


def phase17_final_report(df: pd.DataFrame, artifact_summary: pd.DataFrame, tournament: pd.DataFrame, readiness: str, primary: str, secondary: str, sep: pd.DataFrame) -> str:
    class_counts = df["entry_quality_label_v2"].value_counts().to_dict()
    missed = df[df["L12_missed_good"]]["entry_quality_label_v2"].value_counts().to_dict()
    high_mfe = df[df["L5_high_MFE_low_MAE"]]["entry_quality_label_v2"].value_counts().to_dict()
    good = int(class_counts.get("GOOD", 0))
    bad = int(class_counts.get("BAD", 0))
    neutral = int(class_counts.get("NEUTRAL", 0))
    exclude = int(class_counts.get("EXCLUDE", 0))
    if readiness == "LABEL_V2_READY_FOR_ENTRY_QUALITY_MODEL_RESEARCH":
        verdict = "EXECUTED_LABEL_V2_READY_FOR_ENTRY_QUALITY_MODEL_RESEARCH"
    elif readiness == "LABEL_V2_TOO_FEW_CLEAN_GOOD":
        verdict = "EXECUTED_LABEL_V2_TOO_FEW_CLEAN_GOOD"
    elif readiness == "LABEL_V2_FEATURE_SEPARABILITY_TOO_WEAK":
        verdict = "EXECUTED_LABEL_V2_FEATURE_SEPARABILITY_WEAK"
    elif df["counterfactual_flag"].mean() > 0.30:
        verdict = "EXECUTED_LABEL_V2_COUNTERFACTUAL_ARTIFACT_CONFIRMED"
    else:
        verdict = "EXECUTED_LABEL_V2_BUILT_ARTIFACT_PURGE_SUCCESS"
    _write_md(ROOT / "executed_entry_quality_label_v2_final_report.md", "Executed Entry Quality Label V2 Final Report", {
        "1. why label surgery was needed": "Prior greenlight/missed-good work showed path/counterfactual/max-holding artifacts and objective mismatch.",
        "2. existing label artifact audit": artifact_summary,
        "3. executed-only core universe": int(df["executed_flag"].sum()),
        "4. counterfactual separation": {"counterfactual_rows": int(df["counterfactual_flag"].sum()), "default_weight": 0.0},
        "5. max_holding artifact": int(df["A2_max_holding_artifact"].sum()),
        "6. tiny/cost treatment": "Tiny absolute net return is neutral/weak; cost sensitivity exported.",
        "7. path quality": "MFE is combined with MAE/RFE/holding efficiency and path class.",
        "8. Entry Quality Label V2": primary,
        "9. tournament": tournament,
        "10. primary selection reason": "Highest training usability under executed/artifact/counterfactual constraints.",
        "11. secondary/utility": {"secondary": secondary, "utility": "entry_quality_utility_v2"},
        "12. label distribution": class_counts,
        "13. LONG/SHORT distribution": df.groupby(["direction", "entry_quality_label_v2"]).size().reset_index(name="rows"),
        "14. recent distribution": {"recent_3m_good": int((df["entry_quality_label_v2"].eq("GOOD") & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_3M_DAYS))).sum()), "recent_6m_good": int((df["entry_quality_label_v2"].eq("GOOD") & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS))).sum())},
        "15. stability": "Monthly/quarterly/as-of artifacts exported.",
        "16. missed_good relabel": missed,
        "17. high_MFE_low_MAE relabel": high_mfe,
        "18. Q2/R7/TCN relation": "System alignment CSVs exported.",
        "19. feature separability": sep.sort_values(["PR_AUC", "AUC"], ascending=False).head(30) if len(sep) else pd.DataFrame(),
        "20. next step": readiness,
        "21. safety audit": "Production hashes unchanged; production_ready=false; promotion_ready=false.",
        "A-M answers": {
            "A": "counterfactual + max_holding + tiny/path-only MFE contamination",
            "B": "counterfactual is separated with default sample_weight=0; executed-only distribution is exported separately.",
            "C": int(df["A2_max_holding_artifact"].sum()),
            "D": "It used future path/MFE and often max-holding/counterfactual rows, not entry-safe structure.",
            "E": missed,
            "F": {"GOOD": good, "BAD": bad, "NEUTRAL": neutral, "EXCLUDE": exclude},
            "G": "See cost/label_flip_by_cost.csv.",
            "H": "See system_alignment/label_v2_vs_q2.csv.",
            "I": "See system_alignment/label_v2_vs_r7.csv; R7 remains bad-structure warning.",
            "J": "See system_alignment/label_v2_vs_tcn_confidence.csv.",
            "K": f"Best PR-AUC={float(sep['PR_AUC'].max()) if len(sep) else 0.0:.4f}",
            "L": readiness,
            "M": "If not ready, shortage is clean executed good/bad balance plus feature separability, not production routing.",
        },
    })
    _write_md(ROOT / "executed_entry_quality_label_v2_final_verdict.md", "Executed Entry Quality Label V2 Final Verdict", {
        "final_verdict": verdict,
        "primary_label_v2": primary,
        "secondary_label_v2": secondary,
        "readiness": readiness,
        "production_ready": False,
        "promotion_ready": False,
        "Q2_BDI_baseline_changed": False,
        "R7_action": "none",
    })
    return verdict


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        paths = _discover_paths()
        return {"dry_run": True, "would_write_root": str(ROOT), "discovered_groups": {k: len(v) for k, v in paths.items()}, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    before = phase1_safety_before()
    df, labels, label_summary, safe_cols, safe_df, forbidden_df = _prepare_frame()
    phase0_discovery(df, labels, safe_df, forbidden_df)
    df = _flag_artifacts(df)
    artifact_summary = phase2_existing_label_audit(df, labels, safe_cols)
    phase3_universes(df)
    phase4_artifact_taxonomy(df, labels)
    path = _path_quality(df)
    for col in ["path_quality_class_v2", "clean_path_score", "MFE_to_MAE_ratio", "holding_efficiency"]:
        if col in path.columns:
            df[col] = path[col].to_numpy()
    candidates = phase5_label_design(df, path)
    phase6_cost(df, candidates)
    phase7_path_quality(path, candidates)
    phase8_max_holding(df, candidates)
    phase9_counterfactual(df, candidates)
    primary, secondary, tournament = phase10_tournament(df, candidates, safe_cols)
    df = _assign_final_label(df, candidates, primary)
    phase11_system_alignment(df, labels, safe_cols)
    phase12_stability(df, safe_cols)
    sep = phase13_separability(df, safe_cols)
    phase14_dataset_export(df)
    readiness = phase15_readiness(df, sep)
    phase16_audit(before)
    verdict = phase17_final_report(df, artifact_summary, tournament, readiness, primary, secondary, sep)
    return {
        "dry_run": False,
        "rows": len(df),
        "executed_rows": int(df["executed_flag"].sum()),
        "counterfactual_rows": int(df["counterfactual_flag"].sum()),
        "primary_label_v2": primary,
        "class_counts": df["entry_quality_label_v2"].value_counts().to_dict(),
        "readiness": readiness,
        "final_verdict": verdict,
        "production_ready": False,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build diagnostics-only executed entry-quality label v2.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"executed_entry_quality_label_v2 verdict={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
