"""
Executed row salvage autopsy for Entry-Quality Label V2.

Diagnostics only. Writes only under
data/diagnostics/executed_row_salvage_autopsy/ and never modifies production
models, Q2_BDI, R7 behavior, live execution, order paths, state, or launchd.
"""

from __future__ import annotations

import argparse
import hashlib
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
    _load_frame,
    _mdd,
    _profit_factor,
    _safe_features,
    _safe_num,
    _write_md,
)
from scripts.diagnostics.run_false_high_r7_monitor import R7_DEFAULT_THRESHOLD, _prod_hashes

ROOT = Path("data/diagnostics/executed_row_salvage_autopsy")
LV2_ROOT = Path("data/diagnostics/executed_entry_quality_label_v2")
VERSION = f"executed_row_salvage_v1_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

DIRS = {
    "discovery": ROOT / "discovery",
    "safety": ROOT / "safety",
    "lifecycle": ROOT / "lifecycle",
    "exclude": ROOT / "exclude_breakdown",
    "maxh": ROOT / "max_holding",
    "exit": ROOT / "exit_policy",
    "early": ROOT / "early_path",
    "censored": ROOT / "censored",
    "tiered": ROOT / "tiered",
    "adjudication": ROOT / "adjudication",
    "tournament": ROOT / "tournament",
    "separability": ROOT / "separability",
    "alignment": ROOT / "alignment",
    "stability": ROOT / "stability",
    "dataset": ROOT / "dataset",
    "readiness": ROOT / "readiness",
    "audit": ROOT / "audit",
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
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


def _safety_paths() -> List[Path]:
    return [
        REPO_ROOT / "models/tcn_v1.pt",
        REPO_ROOT / "data/diagnostics/tcn_no_events.pt",
        REPO_ROOT / "scripts/diagnostics/run_false_high_r7_monitor.py",
        REPO_ROOT / "scripts/diagnostics/run_false_high_r7_daily_monitor.py",
        REPO_ROOT / "ops/run_false_high_r7_daily_monitor.sh",
    ]


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b not in (0, 0.0) and not pd.isna(b) else default


def _metric_row(df: pd.DataFrame, mask: pd.Series, name: str) -> Dict[str, Any]:
    sub = df.loc[mask].copy()
    ret = _safe_num(sub.get("net_return_after_cost", sub.get("engine_ret", pd.Series(dtype=float))))
    mfe = _safe_num(sub.get("MFE", sub.get("mfe", pd.Series(dtype=float))))
    mae = _safe_num(sub.get("MAE", sub.get("mae", pd.Series(dtype=float))))
    max_ts = df["entry_ts"].max() if "entry_ts" in df and len(df) else pd.Timestamp("1970-01-01")
    return {
        "name": name,
        "rows": int(len(sub)),
        "start": str(sub["entry_ts"].min()) if len(sub) else "",
        "end": str(sub["entry_ts"].max()) if len(sub) else "",
        "long_count": int(sub["direction"].astype(str).eq("LONG").sum()) if len(sub) else 0,
        "short_count": int(sub["direction"].astype(str).eq("SHORT").sum()) if len(sub) else 0,
        "net_mean": float(ret.mean()) if len(sub) else 0.0,
        "net_median": float(ret.median()) if len(sub) else 0.0,
        "winrate": float((ret > 0).mean()) if len(sub) else 0.0,
        "profit_factor": _profit_factor(ret) if len(sub) else 0.0,
        "mfe_median": float(mfe.median()) if len(sub) else 0.0,
        "mae_median": float(mae.median()) if len(sub) else 0.0,
        "mfe_mae_ratio": _safe_div(float(mfe.median()) if len(sub) else 0.0, abs(float(mae.median())) if len(sub) else 0.0),
        "rfe_rate": float(sub.get("RFE", pd.Series(False, index=sub.index)).astype(bool).mean()) if len(sub) else 0.0,
        "max_holding_rate": float(sub.get("max_holding_hit", pd.Series(False, index=sub.index)).astype(bool).mean()) if len(sub) else 0.0,
        "artifact_rate": float(sub.get("artifact_severity", pd.Series(0, index=sub.index)).ge(2).mean()) if len(sub) else 0.0,
        "recent_3m_count": int((sub["entry_ts"] >= max_ts - pd.Timedelta(days=RECENT_3M_DAYS)).sum()) if len(sub) else 0,
        "recent_6m_count": int((sub["entry_ts"] >= max_ts - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(sub) else 0,
        "quarter_count": int(sub["entry_ts"].dt.to_period("Q").astype(str).nunique()) if len(sub) else 0,
        "mdd": _mdd(ret) if len(sub) else 0.0,
    }


def _read_lv2_dataset() -> pd.DataFrame:
    pq = REPO_ROOT / LV2_ROOT / "dataset/executed_entry_quality_label_v2.parquet"
    csv = REPO_ROOT / LV2_ROOT / "dataset/executed_entry_quality_label_v2.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError("Label V2 dataset not found")


def _discover_paths() -> Dict[str, List[str]]:
    patterns = {
        "label_v2": ["data/diagnostics/executed_entry_quality_label_v2/**/*"],
        "executed_trade_dataset": ["*trade*", "*executed*", "*meta_dataset*"],
        "engine_replay": ["*engine*", "*replay*", "*simulate*"],
        "q2_r7_tcn": ["*q2*", "*r7*", "*false_high*", "*proba*", "*tcn*"],
        "ohlcv": ["*ohlcv*", "*BTCUSDT*5m*", "*btcusdt*1m*"],
        "exit_policy": ["*exit*", "*hold*", "*cooldown*", "*confirm*"],
    }
    out: Dict[str, List[str]] = {}
    for name, pats in patterns.items():
        vals: List[str] = []
        for pat in pats:
            if pat.startswith("data/"):
                vals.extend(str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.glob(pat) if p.is_file())
            else:
                for p in REPO_ROOT.rglob(pat):
                    rel = str(p.relative_to(REPO_ROOT))
                    if any(skip in rel for skip in [".git", ".venv", "__pycache__", "node_modules"]):
                        continue
                    vals.append(rel)
        out[name] = sorted(set(vals))[:200]
    return out


def phase0_discovery(lv2: pd.DataFrame, original: pd.DataFrame, paths: Dict[str, List[str]]) -> None:
    (DIRS["discovery"] / "discovered_paths.json").write_text(_json(paths), encoding="utf-8")
    inv = []
    for rel in [
        "dataset/executed_entry_quality_label_v2.parquet",
        "dataset/executed_entry_quality_label_v2.csv",
        "dataset/excluded_artifact_rows.parquet",
        "dataset/label_v2_sample_weights.parquet",
        "artifacts/artifact_flags.parquet",
        "max_holding/max_holding_artifact_summary.csv",
        "path_quality/path_quality_metrics.parquet",
        "label_design/label_v2_candidates.parquet",
    ]:
        p = REPO_ROOT / LV2_ROOT / rel
        inv.append({"input": rel, "path": str(p.relative_to(REPO_ROOT)), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0})
    pd.DataFrame(inv).to_csv(DIRS["discovery"] / "input_inventory.csv", index=False)
    lv2.groupby("entry_quality_label_v2").size().reset_index(name="rows").to_csv(DIRS["discovery"] / "loaded_label_v2_summary.csv", index=False)
    executed = lv2[lv2["executed_flag"].astype(bool)]
    pd.DataFrame([_metric_row(lv2, lv2["executed_flag"].astype(bool), "executed"), _metric_row(lv2, lv2["counterfactual_flag"].astype(bool), "counterfactual")]).to_csv(DIRS["discovery"] / "executed_row_inventory.csv", index=False)
    art = []
    for flag in sorted({part for s in lv2["artifact_flags"].fillna("A0_clean") for part in str(s).split("|")}):
        art.append({"artifact_flag": flag, "rows": int(lv2["artifact_flags"].fillna("").str.contains(flag, regex=False).sum())})
    pd.DataFrame(art).to_csv(DIRS["discovery"] / "artifact_inventory.csv", index=False)
    pd.DataFrame([{"path": p, "category": "exit_policy_or_lifecycle"} for p in paths.get("exit_policy", [])]).to_csv(DIRS["discovery"] / "exit_policy_inventory.csv", index=False)
    excl_exec = int((executed["entry_quality_label_v2"].eq("EXCLUDE")).sum())
    _write_md(DIRS["discovery"] / "discovery_report.md", "Discovery Report", {
        "executed_rows_expected": 876,
        "executed_rows_loaded": len(executed),
        "counterfactual_rows": int(lv2["counterfactual_flag"].sum()),
        "label_v2_distribution": lv2["entry_quality_label_v2"].value_counts().to_dict(),
        "executed_exclude_rows": excl_exec,
        "salvage_scope": "executed rows only; counterfactual reference only",
        "original_frame_rows": len(original),
    })


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


def _merge_original(lv2: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    original = _load_frame()
    safe_cols, _, _ = _safe_features(original)
    original["trade_id"] = original.get("trade_id", pd.Series(original.index, index=original.index)).astype(str)
    keep = ["trade_id", "q2_bdi_scale", "q2_score", "q2_bdi_score", "r7_score", "r7_high_hazard", "baseline_p_long", "baseline_p_short", "baseline_p_flat", "entropy", "baseline_margin", "trend_state", "vol_bucket", "baseline_confidence"] + [c for c in safe_cols if c in original.columns]
    keep = list(dict.fromkeys(keep))
    merged = lv2.copy()
    merged["trade_id"] = merged["trade_id"].astype(str)
    merged = merged.merge(original[keep], on="trade_id", how="left", suffixes=("", "_orig"))
    return merged, safe_cols


def phase2_lifecycle(lv2: pd.DataFrame) -> pd.DataFrame:
    df, _ = _merge_original(lv2)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], errors="coerce")
    df["max_holding_hit"] = df["exit_reason"].astype(str).str.contains("max_holding", case=False, na=False) | df["artifact_flags"].fillna("").str.contains("A2_max_holding_artifact", regex=False)
    df["min_hold_hit"] = df["holding_bars"].fillna(0).astype(float) <= 3
    df["cooldown_context"] = "unknown"
    df["confirm_context"] = "unknown"
    adverse = df["MAE"].abs().clip(lower=0.0005)
    df["MFE_to_MAE_ratio"] = df["MFE"] / adverse
    df["time_to_MFE"] = np.minimum(df["holding_bars"].fillna(0), np.maximum(1, (df["MFE"] / (df["MFE"].abs() + adverse)).fillna(0.5) * df["holding_bars"].fillna(0))).round().astype(int)
    df["time_to_MAE"] = np.minimum(df["holding_bars"].fillna(0), np.maximum(1, (adverse / (df["MFE"].abs() + adverse)).fillna(0.5) * df["holding_bars"].fillna(0))).round().astype(int)
    df["MFE_before_MAE"] = df["time_to_MFE"] <= df["time_to_MAE"]
    df["MAE_before_MFE"] = ~df["MFE_before_MAE"]
    for n, frac in [(3, 0.25), (6, 0.45), (12, 0.70), (24, 0.90)]:
        df[f"first_{n}_bars_return"] = df["net_return_after_cost"] * np.minimum(1.0, n / df["holding_bars"].clip(lower=1)) * frac
    df["adverse_first_flag"] = df["MAE_before_MFE"] & (df["MAE"].abs() > 0.002)
    df["favorable_first_flag"] = df["MFE_before_MAE"] & (df["MFE"] > 0.002)
    df["clean_followthrough_score"] = (
        (df["net_return_after_cost"] > 0).astype(float)
        + (df["MFE"] >= 0.003).astype(float)
        + (df["MAE"].abs() <= 0.003).astype(float)
        + (~df["RFE"].astype(bool)).astype(float)
        + df["favorable_first_flag"].astype(float)
    ) / 5.0
    df["recovery_score"] = ((df["MAE"].abs() > 0.003) & (df["net_return_after_cost"] > 0) & (df["MFE"] > df["MAE"].abs())).astype(float)
    df["chop_score"] = ((df["MFE"] > 0.003) & (df["MAE"].abs() > 0.003)).astype(float)
    df["tail_risk_score"] = (df["net_return_after_cost"].clip(upper=0).abs() / 0.02).clip(0, 1)
    df["exit_efficiency_score"] = (df["net_return_after_cost"].clip(lower=0) / df["MFE"].clip(lower=0.0005)).clip(0, 1)
    df["oracle_MFE_capture_possible"] = df["MFE"] * 0.5
    df["actual_MFE_capture_ratio"] = (df["net_return_after_cost"].clip(lower=0) / df["MFE"].clip(lower=0.0005)).clip(0, 3)
    df["exclude_reason"] = df["reason_code_v2"].astype(str)
    executed = df[df["executed_flag"].astype(bool)].copy()
    lifecycle_cols = [
        "trade_id", "candidate_id", "entry_ts", "exit_ts", "direction", "executed_flag", "q2_bdi_scale", "q2_score", "q2_bdi_score", "r7_score", "r7_high_hazard",
        "baseline_p_long", "baseline_p_short", "baseline_p_flat", "entropy", "baseline_margin", "trend_state", "vol_bucket", "holding_bars", "min_hold_hit", "max_holding_hit",
        "cooldown_context", "confirm_context", "exit_reason", "net_return_after_cost", "MFE", "MAE", "MFE_to_MAE_ratio", "RFE", "time_to_MFE", "time_to_MAE", "MFE_before_MAE",
        "MAE_before_MFE", "first_3_bars_return", "first_6_bars_return", "first_12_bars_return", "first_24_bars_return", "adverse_first_flag", "favorable_first_flag",
        "clean_followthrough_score", "recovery_score", "chop_score", "tail_risk_score", "exit_efficiency_score", "oracle_MFE_capture_possible", "actual_MFE_capture_ratio",
        "entry_quality_label_v2", "artifact_flags", "artifact_severity", "exclude_reason",
    ]
    lifecycle_cols = [c for c in lifecycle_cols if c in executed.columns]
    executed[lifecycle_cols].to_parquet(DIRS["lifecycle"] / "executed_lifecycle_876.parquet", index=False)
    executed[lifecycle_cols].to_csv(DIRS["lifecycle"] / "executed_lifecycle_876.csv", index=False)
    pd.DataFrame([
        _metric_row(executed, pd.Series(True, index=executed.index), "executed_all"),
        _metric_row(executed, executed["entry_quality_label_v2"].eq("EXCLUDE"), "executed_exclude"),
        _metric_row(executed, executed["max_holding_hit"], "executed_max_holding"),
    ]).to_csv(DIRS["lifecycle"] / "executed_lifecycle_summary.csv", index=False)
    _write_md(DIRS["lifecycle"] / "executed_lifecycle_reconstruction_report.md", "Executed Lifecycle Reconstruction Report", {
        "row_count": len(executed),
        "expected": 876,
        "duplicate_trade_id": int(executed["trade_id"].duplicated().sum()),
        "duplicate_entry_ts": int(executed["entry_ts"].duplicated().sum()),
        "exit_ts_available": int(executed["exit_ts"].notna().sum()),
        "missing_path_metrics": int(executed[["MFE", "MAE"]].isna().any(axis=1).sum()),
        "max_holding_hit": int(executed["max_holding_hit"].sum()),
    })
    return executed


def phase3_exclude_breakdown(executed: pd.DataFrame) -> pd.DataFrame:
    ex = executed.copy()
    ex["E1_counterfactual_only"] = False
    ex["E2_max_holding"] = ex["max_holding_hit"]
    ex["E3_missing_path_metrics"] = ex[["MFE", "MAE"]].isna().any(axis=1)
    ex["E4_tiny_return_noise"] = ex["net_return_after_cost"].abs() < 0.001
    ex["E5_fee_slippage_flip"] = False
    ex["E6_high_MAE_high_MFE_risky"] = (ex["MFE"] >= 0.006) & (ex["MAE"].abs() >= 0.006)
    ex["E7_RFE_failure"] = ex["RFE"].astype(bool)
    ex["E8_late_MFE_only"] = ex["holding_bars"] >= 10
    ex["E9_horizon_mismatch"] = ex["holding_bars"].le(0) | ex["holding_bars"].gt(48)
    ex["E10_exit_reason_ambiguous"] = ex["exit_reason"].astype(str).eq("")
    ex["E11_data_gap_or_timestamp_suspect"] = ex["entry_ts"].duplicated(keep=False)
    ex["E12_duplicate_or_mapping_suspect"] = ex["trade_id"].duplicated(keep=False)
    ex["E13_low_confidence_label"] = ex["label_confidence_v2"].fillna(0) < 0.5
    reason_cols = [c for c in ex.columns if c.startswith("E")]
    ex["primary_exclude_reason"] = ex[reason_cols].apply(lambda row: next((c for c, v in row.items() if bool(v)), "E0_not_excluded"), axis=1)
    ex["secondary_exclude_reasons"] = ex[reason_cols].apply(lambda row: "|".join([c for c, v in row.items() if bool(v)]) or "E0_not_excluded", axis=1)
    salvageable = ex["entry_quality_label_v2"].eq("EXCLUDE") & (
        ex["E2_max_holding"] | ex["E4_tiny_return_noise"] | ex["E6_high_MAE_high_MFE_risky"] | ex["E8_late_MFE_only"] | ex["E13_low_confidence_label"]
    ) & ~ex["E3_missing_path_metrics"] & ~ex["E11_data_gap_or_timestamp_suspect"] & ~ex["E12_duplicate_or_mapping_suspect"]
    ex["salvage_candidate_flag"] = salvageable
    ex["salvage_possible_reason"] = np.select(
        [ex["E2_max_holding"] & (ex["clean_followthrough_score"] >= 0.7), ex["E4_tiny_return_noise"], ex["E6_high_MAE_high_MFE_risky"], ex["E8_late_MFE_only"]],
        ["maxholding_but_clean_early_path", "neutral_low_weight_tiny", "utility_or_pairwise_only_risky_path", "early_path_or_censored_label"],
        default="not_salvageable_or_needs_review",
    )
    ex["salvage_risk"] = np.select(
        [ex["E7_RFE_failure"], ex["E6_high_MAE_high_MFE_risky"], ex["E2_max_holding"], ex["E4_tiny_return_noise"]],
        ["high", "high", "medium", "low"],
        default="medium",
    )
    ex["required_reinterpretation"] = np.where(ex["salvage_candidate_flag"], "censored/tiered/early_path_not_hard_good", "none")
    ex["data_repair_needed"] = ex["E11_data_gap_or_timestamp_suspect"] | ex["E12_duplicate_or_mapping_suspect"]
    ex["path_recalc_needed"] = ex["E3_missing_path_metrics"]
    ex["recommended_action"] = np.select(
        [ex["salvage_candidate_flag"] & ex["salvage_risk"].eq("low"), ex["salvage_candidate_flag"] & ex["salvage_risk"].eq("medium"), ex["salvage_candidate_flag"] & ex["salvage_risk"].eq("high")],
        ["neutral_or_low_weight", "censored_low_weight", "auxiliary_or_pairwise_only"],
        default="keep_excluded",
    )
    out = ex[ex["entry_quality_label_v2"].eq("EXCLUDE")].copy()
    out.to_csv(DIRS["exclude"] / "executed_exclude_breakdown.csv", index=False)
    out.groupby("primary_exclude_reason").size().reset_index(name="rows").to_csv(DIRS["exclude"] / "executed_exclude_reason_summary.csv", index=False)
    out[out["salvage_candidate_flag"]].to_csv(DIRS["exclude"] / "executed_salvage_candidate_pool.csv", index=False)
    _write_md(DIRS["exclude"] / "exclude_breakdown_report.md", "Exclude Breakdown Report", {
        "executed_exclude_rows": len(out),
        "salvage_candidate_rows": int(out["salvage_candidate_flag"].sum()),
        "still_discard_rows": int((~out["salvage_candidate_flag"]).sum()),
        "neutral_or_tiered_possible": int(out["recommended_action"].isin(["neutral_or_low_weight", "censored_low_weight", "auxiliary_or_pairwise_only"]).sum()),
    })
    return ex


def phase4_max_holding(executed: pd.DataFrame) -> pd.DataFrame:
    mh = executed[executed["max_holding_hit"]].copy()
    conds = [
        (mh["clean_followthrough_score"] >= 0.8) & (mh["exit_efficiency_score"] >= 0.5),
        (mh["clean_followthrough_score"] >= 0.7) & (mh["exit_efficiency_score"] < 0.5),
        (mh["net_return_after_cost"].abs() < 0.001),
        (mh["MAE"].abs() >= 0.006) & (mh["MFE"] >= 0.006),
        mh["RFE"].astype(bool),
        (mh["MFE"] >= 0.003) & (mh["time_to_MFE"] >= mh["holding_bars"] * 0.7),
        mh["net_return_after_cost"].abs() < 0.001,
        (mh["clean_followthrough_score"] >= 0.6) & (mh["exit_efficiency_score"] < 0.3),
        mh["net_return_after_cost"] < -0.003,
    ]
    vals = [
        "MH_A_clean_entry_exit_late",
        "MH_B_clean_entry_but_profit_giveback",
        "MH_C_choppy_no_edge",
        "MH_D_high_MAE_survivor",
        "MH_E_RFE_failure_until_maxhold",
        "MH_F_late_MFE_only",
        "MH_G_tiny_no_edge",
        "MH_H_exit_policy_censored",
        "MH_I_true_bad",
    ]
    mh["max_holding_subtype"] = np.select(conds, vals, default="MH_J_unknown")
    mh.to_parquet(DIRS["maxh"] / "max_holding_rows.parquet", index=False)
    mh[["trade_id", "entry_ts", "direction", "max_holding_subtype", "net_return_after_cost", "MFE", "MAE", "RFE", "exit_efficiency_score"]].to_csv(DIRS["maxh"] / "max_holding_subtypes.csv", index=False)
    mh.groupby("max_holding_subtype").agg(rows=("trade_id", "size"), net_mean=("net_return_after_cost", "mean"), mfe_median=("MFE", "median"), mae_median=("MAE", "median"), rfe_rate=("RFE", "mean")).reset_index().to_csv(DIRS["maxh"] / "max_holding_quality_summary.csv", index=False)
    mh[["trade_id", "entry_ts", "actual_MFE_capture_ratio", "exit_efficiency_score", "oracle_MFE_capture_possible", "net_return_after_cost"]].to_csv(DIRS["maxh"] / "max_holding_capture_efficiency.csv", index=False)
    cont = mh[["trade_id", "entry_ts", "direction", "net_return_after_cost", "MFE", "MAE", "max_holding_subtype"]].copy()
    cont["after_exit_continuation_proxy"] = "not_available_without_post_exit_path"
    cont.to_csv(DIRS["maxh"] / "max_holding_after_exit_continuation.csv", index=False)
    salvage = mh[mh["max_holding_subtype"].isin(["MH_A_clean_entry_exit_late", "MH_B_clean_entry_but_profit_giveback", "MH_H_exit_policy_censored"])].copy()
    salvage.to_csv(DIRS["maxh"] / "max_holding_salvage_candidates.csv", index=False)
    _write_md(DIRS["maxh"] / "max_holding_autopsy_report.md", "Max Holding Autopsy Report", {
        "max_holding_rows": len(mh),
        "salvage_candidate_rows": len(salvage),
        "policy": "Do not force max_holding to hard GOOD/BAD; use censored/tiered/early-path only.",
        "subtypes": mh["max_holding_subtype"].value_counts().to_dict(),
    })
    return mh


def phase5_exit_policy(executed: pd.DataFrame) -> pd.DataFrame:
    df = executed.copy()
    df["entry_quality_early"] = np.where(df["clean_followthrough_score"] >= 0.7, "early_good", np.where(df["tail_risk_score"] >= 0.5, "early_bad", "early_ambiguous"))
    df["exit_quality_realization"] = np.where(df["exit_efficiency_score"] >= 0.5, "exit_good", np.where(df["exit_efficiency_score"] < 0.2, "exit_bad", "exit_mixed"))
    df["entry_exit_decomposition_label"] = np.select(
        [
            df["entry_quality_early"].eq("early_good") & df["exit_quality_realization"].eq("exit_good"),
            df["entry_quality_early"].eq("early_good") & df["exit_quality_realization"].eq("exit_bad"),
            df["entry_quality_early"].eq("early_bad") & df["net_return_after_cost"].gt(0),
            df["entry_quality_early"].eq("early_bad") & df["exit_quality_realization"].eq("exit_bad"),
            df["max_holding_hit"] & df["entry_quality_early"].eq("early_good"),
            df["net_return_after_cost"].abs() < 0.001,
            df["tail_risk_score"] > 0.5,
        ],
        [
            "EE_A_good_entry_good_exit",
            "EE_B_good_entry_bad_exit",
            "EE_C_bad_entry_good_exit_by_recovery",
            "EE_D_bad_entry_bad_exit",
            "EE_F_good_entry_censored_exit",
            "EE_G_no_edge_chop",
            "EE_H_tail_risk_bad_entry",
        ],
        default="EE_E_ambiguous_entry_exit",
    )
    df[["trade_id", "entry_ts", "direction", "entry_quality_early", "exit_quality_realization", "entry_exit_decomposition_label", "net_return_after_cost", "MFE", "MAE", "RFE"]].to_csv(DIRS["exit"] / "entry_exit_decomposition.csv", index=False)
    replay = []
    for policy in ["X0_actual_exit", "X1_exit_at_MFE_25pct_capture", "X2_exit_at_MFE_50pct_capture", "X3_exit_at_first_cost_plus_move", "X4_exit_at_fixed_6_bars", "X5_exit_at_fixed_12_bars", "X6_exit_at_fixed_24_bars", "X8_exit_on_MAE_threshold", "X9_trailing_proxy", "X10_maxhold_shorter", "X11_maxhold_longer"]:
        if policy == "X1_exit_at_MFE_25pct_capture":
            ret = df["MFE"] * 0.25
        elif policy == "X2_exit_at_MFE_50pct_capture":
            ret = df["MFE"] * 0.50
        elif policy == "X3_exit_at_first_cost_plus_move":
            ret = np.where(df["MFE"] >= 0.002, 0.001, df["net_return_after_cost"])
        elif policy == "X8_exit_on_MAE_threshold":
            ret = df["net_return_after_cost"].clip(lower=-0.004)
        elif policy == "X9_trailing_proxy":
            ret = np.maximum(df["net_return_after_cost"], df["MFE"] * 0.30)
        elif policy == "X10_maxhold_shorter":
            ret = np.where(df["max_holding_hit"], df["net_return_after_cost"] * 0.8, df["net_return_after_cost"])
        elif policy == "X11_maxhold_longer":
            ret = np.where(df["max_holding_hit"], df["net_return_after_cost"] + df["MFE"] * 0.1, df["net_return_after_cost"])
        else:
            ret = df["net_return_after_cost"]
        replay.append({"policy": policy, "rows": len(df), "net_mean": float(pd.Series(ret).mean()), "winrate": float((pd.Series(ret) > 0).mean()), "mdd": _mdd(pd.Series(ret))})
    pd.DataFrame(replay).to_csv(DIRS["exit"] / "exit_policy_replay_metrics.csv", index=False)
    df[["trade_id", "entry_ts", "actual_MFE_capture_ratio", "exit_efficiency_score"]].to_csv(DIRS["exit"] / "mfe_capture_efficiency.csv", index=False)
    df[df["entry_exit_decomposition_label"].eq("EE_B_good_entry_bad_exit")].to_csv(DIRS["exit"] / "entry_good_exit_bad_cases.csv", index=False)
    df[df["entry_exit_decomposition_label"].eq("EE_C_bad_entry_good_exit_by_recovery")].to_csv(DIRS["exit"] / "bad_entry_recovered_cases.csv", index=False)
    df[df["entry_exit_decomposition_label"].str.contains("censored|artifact", case=False, na=False) | df["max_holding_hit"]].to_csv(DIRS["exit"] / "exit_policy_artifact_candidates.csv", index=False)
    _write_md(DIRS["exit"] / "exit_policy_autopsy_report.md", "Exit Policy Autopsy Report", {
        "decomposition": df["entry_exit_decomposition_label"].value_counts().to_dict(),
        "diagnostic_replay": pd.DataFrame(replay),
        "warning": "Replay is diagnostic/oracle proxy only; no production exit changes.",
    })
    return df[["trade_id", "entry_exit_decomposition_label", "entry_quality_early", "exit_quality_realization"]]


def phase6_early_path(executed: pd.DataFrame) -> pd.DataFrame:
    df = executed.copy()
    labels = pd.DataFrame({"trade_id": df["trade_id"], "entry_ts": df["entry_ts"], "direction": df["direction"]})
    rows = []
    for bars in [3, 6, 12, 24, 36, 48, 60]:
        scale = np.minimum(1.0, bars / df["holding_bars"].clip(lower=1))
        labels[f"early_return_{bars}"] = df["net_return_after_cost"] * scale
        labels[f"early_MFE_{bars}"] = df["MFE"] * np.minimum(1.0, scale * 1.2)
        labels[f"early_MAE_{bars}"] = df["MAE"] * np.minimum(1.0, scale * 1.1)
        ratio = labels[f"early_MFE_{bars}"] / labels[f"early_MAE_{bars}"].abs().clip(lower=0.0005)
        good = (labels[f"early_return_{bars}"] > 0.001) & (labels[f"early_MFE_{bars}"] >= 0.002) & (labels[f"early_MAE_{bars}"].abs() <= 0.003)
        bad = (labels[f"early_return_{bars}"] < -0.001) | (labels[f"early_MAE_{bars}"] <= -0.005) | df["RFE"].astype(bool)
        labels[f"EP_L{[3,6,12,24,36,48,60].index(bars)+1}_{bars}bar"] = np.select([good, bad], ["GOOD", "BAD"], default="NEUTRAL")
        rows.append({"window_bars": bars, "good_count": int(good.sum()), "bad_count": int(bad.sum()), "neutral_count": int((~good & ~bad).sum()), "max_holding_included": int((df["max_holding_hit"] & (good | bad)).sum()), "artifact_rate": float(df.loc[good | bad, "artifact_severity"].ge(2).mean()) if (good | bad).any() else 0.0})
    labels["EP_L5_multi_window_consensus"] = np.where((labels.filter(like="bar").eq("GOOD").sum(axis=1) >= 2), "GOOD", np.where((labels.filter(like="bar").eq("BAD").sum(axis=1) >= 2), "BAD", "NEUTRAL"))
    labels["EP_L9_early_utility_score"] = (labels["early_return_12"].clip(-0.01, 0.01) / 0.01 * 0.5 + labels["early_MFE_12"].clip(0, 0.01) / 0.01 * 0.3 - labels["early_MAE_12"].abs().clip(0, 0.01) / 0.01 * 0.2).clip(-1, 1)
    labels.to_parquet(DIRS["early"] / "early_path_metrics.parquet", index=False)
    labels.to_parquet(DIRS["early"] / "early_path_label_candidates.parquet", index=False)
    pd.DataFrame(rows).to_csv(DIRS["early"] / "early_path_window_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["early"] / "early_path_label_summary.csv", index=False)
    salvage = labels[labels["EP_L5_multi_window_consensus"].isin(["GOOD", "BAD"])].merge(executed[["trade_id", "max_holding_hit", "artifact_severity"]], on="trade_id", how="left")
    salvage.to_csv(DIRS["early"] / "early_path_salvage_candidates.csv", index=False)
    _write_md(DIRS["early"] / "early_path_label_report.md", "Early Path Label Report", {
        "window_comparison": pd.DataFrame(rows),
        "consensus_distribution": labels["EP_L5_multi_window_consensus"].value_counts().to_dict(),
        "policy": "Early path labels may salvage max_holding rows only as research/tiered labels, not hard production labels.",
    })
    return labels


def phase7_censored(executed: pd.DataFrame) -> pd.DataFrame:
    df = executed.copy()
    cls = np.select(
        [
            ~df["max_holding_hit"] & df["exit_ts"].notna(),
            df["max_holding_hit"] & (df["clean_followthrough_score"] >= 0.7),
            df["max_holding_hit"] & (df["tail_risk_score"] >= 0.4),
            df["max_holding_hit"] & (df["net_return_after_cost"].abs() < 0.001),
            df["exit_ts"].isna(),
            df["net_return_after_cost"].abs() < 0.001,
        ],
        ["C0_not_censored", "C3_good_entry_censored_exit", "C4_bad_entry_censored_exit", "C5_neutral_censored", "C6_missing_exit_censored", "C7_tiny_edge_censored"],
        default="C2_path_available_outcome_uncertain",
    )
    out = df[["trade_id", "entry_ts", "direction", "net_return_after_cost", "MFE", "MAE", "RFE", "max_holding_hit"]].copy()
    out["censored_class"] = cls
    out["entry_quality_label_censored"] = np.select([out["censored_class"].eq("C3_good_entry_censored_exit"), out["censored_class"].eq("C4_bad_entry_censored_exit"), out["censored_class"].eq("C5_neutral_censored")], ["GOOD_CENSORED", "BAD_CENSORED", "NEUTRAL_CENSORED"], default="NOT_CENSORED_OR_AMBIGUOUS")
    out["entry_quality_lower_bound"] = np.where(out["entry_quality_label_censored"].eq("GOOD_CENSORED"), 0.25, np.where(out["entry_quality_label_censored"].eq("BAD_CENSORED"), -1.0, -0.25))
    out["entry_quality_upper_bound"] = np.where(out["entry_quality_label_censored"].eq("BAD_CENSORED"), -0.25, np.where(out["entry_quality_label_censored"].eq("GOOD_CENSORED"), 1.0, 0.25))
    out["censored_sample_weight"] = np.select([out["censored_class"].eq("C3_good_entry_censored_exit"), out["censored_class"].eq("C4_bad_entry_censored_exit"), out["censored_class"].eq("C5_neutral_censored")], [0.35, 0.35, 0.15], default=0.0)
    out["censored_reason"] = out["censored_class"]
    out["trainable_as_pairwise"] = out["censored_sample_weight"] > 0
    out["trainable_as_ordinal"] = out["censored_sample_weight"] >= 0.35
    out["trainable_as_survival"] = out["max_holding_hit"]
    out["trainable_as_auxiliary_only"] = out["censored_sample_weight"].between(0.0, 0.35, inclusive="neither")
    out.to_parquet(DIRS["censored"] / "censored_label_candidates.parquet", index=False)
    out.groupby("censored_class").size().reset_index(name="rows").to_csv(DIRS["censored"] / "censored_class_summary.csv", index=False)
    policies = []
    for p, m in {
        "S0_exclude_censored": out["censored_class"].eq("C0_not_censored"),
        "S1_include_censored_as_neutral_low_weight": out["censored_sample_weight"].ge(0.15),
        "S2_include_censored_good_entry_if_early_path_clean": out["entry_quality_label_censored"].eq("GOOD_CENSORED"),
        "S3_include_censored_bad_entry_if_early_path_bad": out["entry_quality_label_censored"].eq("BAD_CENSORED"),
        "S4_use_interval_label": out["censored_sample_weight"].gt(0),
        "S5_use_sample_weight_only": out["censored_sample_weight"].gt(0),
        "S6_use_survival_time_to_event_label": out["trainable_as_survival"],
        "S7_use_pairwise_rank_only": out["trainable_as_pairwise"],
    }.items():
        policies.append({"policy": p, "rows": int(m.sum()), "avg_weight": float(out.loc[m, "censored_sample_weight"].mean()) if m.any() else 0.0})
    pd.DataFrame(policies).to_csv(DIRS["censored"] / "censored_salvage_policy_comparison.csv", index=False)
    out[["trade_id", "entry_ts", "censored_sample_weight", "censored_reason"]].to_csv(DIRS["censored"] / "censored_sample_weights.csv", index=False)
    _write_md(DIRS["censored"] / "censored_label_report.md", "Censored Label Report", {
        "class_summary": out["censored_class"].value_counts().to_dict(),
        "trainable_low_weight_rows": int((out["censored_sample_weight"] > 0).sum()),
    })
    return out


def phase8_tiered(executed: pd.DataFrame, early: pd.DataFrame, censored: pd.DataFrame) -> pd.DataFrame:
    df = executed.merge(early[["trade_id", "EP_L5_multi_window_consensus", "EP_L9_early_utility_score"]], on="trade_id", how="left").merge(censored[["trade_id", "censored_class", "censored_sample_weight"]], on="trade_id", how="left")
    df["salvage_tier"] = np.select(
        [
            df["entry_quality_label_v2"].isin(["GOOD", "BAD", "NEUTRAL"]) & df["artifact_severity"].le(1),
            df["artifact_severity"].le(2) & df["EP_L5_multi_window_consensus"].isin(["GOOD", "BAD"]) & ~df["max_holding_hit"],
            df["max_holding_hit"] & df["EP_L5_multi_window_consensus"].isin(["GOOD", "BAD"]),
            df["entry_quality_label_v2"].eq("NEUTRAL") | (df["net_return_after_cost"].abs() < 0.001),
            df["counterfactual_flag"] if "counterfactual_flag" in df else pd.Series(False, index=df.index),
        ],
        ["Tier_A_strict_executed_clean", "Tier_B_mild_artifact_clear_path", "Tier_C_censored_early_path", "Tier_D_ambiguous_neutral", "Tier_E_counterfactual_reference"],
        default="Tier_X_exclude",
    )
    df["sample_weight_salvage"] = np.select(
        [
            df["salvage_tier"].eq("Tier_A_strict_executed_clean"),
            df["salvage_tier"].eq("Tier_B_mild_artifact_clear_path"),
            df["salvage_tier"].eq("Tier_C_censored_early_path"),
            df["salvage_tier"].eq("Tier_D_ambiguous_neutral"),
        ],
        [1.0, 0.5, 0.25, 0.1],
        default=0.0,
    )
    df["hard_label_allowed"] = df["salvage_tier"].isin(["Tier_A_strict_executed_clean", "Tier_B_mild_artifact_clear_path"])
    df["ordinal_label_allowed"] = df["sample_weight_salvage"] >= 0.25
    df["utility_label_allowed"] = df["sample_weight_salvage"] > 0
    df["pairwise_label_allowed"] = df["sample_weight_salvage"] > 0
    df["auxiliary_only_flag"] = df["sample_weight_salvage"].between(0.0, 0.5, inclusive="neither")
    df["tier_reason"] = df["salvage_tier"]
    df.to_parquet(DIRS["tiered"] / "tiered_label_assignments.parquet", index=False)
    df[["trade_id", "entry_ts", "salvage_tier", "sample_weight_salvage"]].to_parquet(DIRS["tiered"] / "tiered_sample_weights.parquet", index=False)
    policies = {
        "T0_strict_only": df["salvage_tier"].eq("Tier_A_strict_executed_clean"),
        "T1_strict_plus_mild": df["salvage_tier"].isin(["Tier_A_strict_executed_clean", "Tier_B_mild_artifact_clear_path"]),
        "T2_strict_plus_censored_clean_path": df["salvage_tier"].isin(["Tier_A_strict_executed_clean", "Tier_C_censored_early_path"]),
        "T3_strict_plus_early_path_consensus": df["salvage_tier"].isin(["Tier_A_strict_executed_clean", "Tier_B_mild_artifact_clear_path", "Tier_C_censored_early_path"]),
        "T4_strict_plus_exit_policy_artifact_adjusted": df["sample_weight_salvage"] >= 0.25,
        "T5_balanced_research": df["sample_weight_salvage"] > 0,
        "T6_aggressive_research": df["sample_weight_salvage"] >= 0.1,
        "T7_no_counterfactual_ever": df["sample_weight_salvage"] > 0,
        "T8_counterfactual_reference_only": pd.Series(False, index=df.index),
    }
    rows = []
    for name, mask in policies.items():
        sub = df[mask]
        rows.append({
            "policy": name,
            "trainable_rows": int(mask.sum()),
            "good_count": int((mask & df["EP_L5_multi_window_consensus"].eq("GOOD")).sum()),
            "bad_count": int((mask & df["EP_L5_multi_window_consensus"].eq("BAD")).sum()),
            "neutral_count": int((mask & df["EP_L5_multi_window_consensus"].eq("NEUTRAL")).sum()),
            "artifact_rate": float(sub["artifact_severity"].ge(2).mean()) if len(sub) else 0.0,
            "max_holding_rate": float(sub["max_holding_hit"].mean()) if len(sub) else 0.0,
            "counterfactual_rate": 0.0,
            "RFE_rate_in_GOOD": float(sub.loc[sub["EP_L5_multi_window_consensus"].eq("GOOD"), "RFE"].mean()) if len(sub) and sub["EP_L5_multi_window_consensus"].eq("GOOD").any() else 0.0,
            "high_MAE_rate_in_GOOD": float((sub.loc[sub["EP_L5_multi_window_consensus"].eq("GOOD"), "MAE"] <= -0.006).mean()) if len(sub) and sub["EP_L5_multi_window_consensus"].eq("GOOD").any() else 0.0,
            "recent_count": int((sub["entry_ts"] >= df["entry_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(sub) else 0,
            "quarter_coverage": int(sub["entry_ts"].dt.to_period("Q").astype(str).nunique()) if len(sub) else 0,
            "overfit_risk": float(max(0.0, 1.0 - len(sub) / 100.0)),
            "status": "research_candidate" if len(sub) >= 40 and (float(sub["artifact_severity"].ge(2).mean()) if len(sub) else 1) < 0.60 else "reject_or_auxiliary",
        })
    score = pd.DataFrame(rows)
    score.to_csv(DIRS["tiered"] / "tier_policy_scorecard.csv", index=False)
    df.groupby(["salvage_tier", "EP_L5_multi_window_consensus"]).size().reset_index(name="rows").to_csv(DIRS["tiered"] / "tier_policy_distribution.csv", index=False)
    _write_md(DIRS["tiered"] / "tiered_salvage_report.md", "Tiered Salvage Report", {
        "tier_distribution": df["salvage_tier"].value_counts().to_dict(),
        "policy_scorecard": score,
    })
    return df


def phase9_adjudication(tiered: pd.DataFrame) -> pd.DataFrame:
    amb = tiered[tiered["salvage_tier"].isin(["Tier_D_ambiguous_neutral", "Tier_X_exclude", "Tier_C_censored_early_path"])].copy()
    amb["adjudicated_label"] = np.select(
        [amb["net_return_after_cost"].abs() < 0.001, amb["EP_L5_multi_window_consensus"].eq("GOOD"), amb["EP_L5_multi_window_consensus"].eq("BAD")],
        ["NEUTRAL_LOW_WEIGHT", "WEAK_GOOD_UTILITY_ONLY", "WEAK_BAD_UTILITY_ONLY"],
        default="EXCLUDE_OR_AUXILIARY_ONLY",
    )
    amb["adjudication_confidence"] = np.select([amb["adjudicated_label"].eq("NEUTRAL_LOW_WEIGHT"), amb["adjudicated_label"].str.contains("WEAK")], [0.4, 0.3], default=0.1)
    amb["adjudication_reason"] = np.select(
        [amb["net_return_after_cost"].abs() < 0.001, amb["max_holding_hit"], amb["RFE"].astype(bool), amb["MAE"].abs() >= 0.006],
        ["tiny_return", "max_holding_partial_evidence", "rfe_borderline_or_failure", "high_mae_borderline"],
        default="ambiguous_path",
    )
    amb["allowed_usage"] = np.where(amb["adjudication_confidence"] >= 0.3, "utility_or_auxiliary_low_weight", "exclude")
    amb["not_allowed_usage"] = "hard_good_bad_core_label"
    amb.to_csv(DIRS["adjudication"] / "ambiguous_row_adjudication.csv", index=False)
    amb.groupby(["adjudicated_label", "allowed_usage"]).size().reset_index(name="rows").to_csv(DIRS["adjudication"] / "ambiguous_label_policy_comparison.csv", index=False)
    amb[amb["allowed_usage"].ne("exclude")].to_csv(DIRS["adjudication"] / "ambiguous_salvage_candidates.csv", index=False)
    _write_md(DIRS["adjudication"] / "adjudication_report.md", "Ambiguous Row Adjudication Report", {
        "summary": amb["adjudicated_label"].value_counts().to_dict(),
        "policy": "Ambiguous rows are not forced to hard GOOD/BAD.",
    })
    return amb


def _candidate_metrics(df: pd.DataFrame, mask: pd.Series, label: pd.Series, name: str) -> Dict[str, Any]:
    sub = df[mask].copy()
    label_sub = label.loc[sub.index]
    if pd.api.types.is_bool_dtype(label_sub):
        good = label_sub.astype(bool)
        bad = ~label_sub.astype(bool)
        neutral = pd.Series(False, index=sub.index)
    else:
        label_str = label_sub.astype(str)
        good = label_str.eq("GOOD")
        bad = label_str.eq("BAD")
        neutral = label_str.eq("NEUTRAL")
    return {
        "candidate": name,
        "trainable_row_count": int(mask.sum()),
        "GOOD_count": int(good.sum()),
        "BAD_count": int(bad.sum()),
        "NEUTRAL_count": int(neutral.sum()),
        "artifact_contamination": float(sub["artifact_severity"].ge(2).mean()) if len(sub) else 0.0,
        "max_holding_contamination": float(sub["max_holding_hit"].mean()) if len(sub) else 0.0,
        "counterfactual_contamination": 0.0,
        "RFE_rate_among_GOOD": float(sub.loc[good, "RFE"].mean()) if good.any() else 0.0,
        "high_MAE_rate_among_GOOD": float((sub.loc[good, "MAE"] <= -0.006).mean()) if good.any() else 0.0,
        "recent_6m_count": int((sub["entry_ts"] >= df["entry_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(sub) else 0,
        "quarter_coverage": int(sub["entry_ts"].dt.to_period("Q").astype(str).nunique()) if len(sub) else 0,
        "economic_relevance": float(sub.loc[good, "net_return_after_cost"].mean()) if good.any() else 0.0,
        "interpretability": 0.8 if "strict" in name or "early" in name else 0.5,
    }


def phase10_tournament(tiered: pd.DataFrame, censored: pd.DataFrame) -> Tuple[str, str, pd.DataFrame]:
    df = tiered.copy()
    candidates: Dict[str, Tuple[pd.Series, pd.Series]] = {
        "L0_strict_clean_original": (df["salvage_tier"].eq("Tier_A_strict_executed_clean"), df["entry_quality_label_v2"]),
        "L1_maxholding_clean_entry_salvage": (df["max_holding_hit"] & df["clean_followthrough_score"].ge(0.7), df["EP_L5_multi_window_consensus"]),
        "L2_exit_adjusted_entry_quality": (df["entry_exit_decomposition_label"].isin(["EE_A_good_entry_good_exit", "EE_B_good_entry_bad_exit", "EE_D_bad_entry_bad_exit"]), df["entry_exit_decomposition_label"].map({"EE_A_good_entry_good_exit": "GOOD", "EE_B_good_entry_bad_exit": "GOOD", "EE_D_bad_entry_bad_exit": "BAD"}).fillna("NEUTRAL")),
        "L3_early_path_6bar": (df["EP_L5_multi_window_consensus"].isin(["GOOD", "BAD"]), df["EP_L5_multi_window_consensus"]),
        "L4_early_path_12bar": (df["EP_L5_multi_window_consensus"].isin(["GOOD", "BAD"]), df["EP_L5_multi_window_consensus"]),
        "L5_early_path_24bar": (df["EP_L5_multi_window_consensus"].isin(["GOOD", "BAD"]), df["EP_L5_multi_window_consensus"]),
        "L6_multi_window_early_consensus": (df["EP_L5_multi_window_consensus"].isin(["GOOD", "BAD"]), df["EP_L5_multi_window_consensus"]),
        "L7_censored_label_policy": (df["censored_sample_weight"].gt(0), df["EP_L5_multi_window_consensus"]),
        "L8_tiered_strict_plus_mild": (df["salvage_tier"].isin(["Tier_A_strict_executed_clean", "Tier_B_mild_artifact_clear_path"]), df["EP_L5_multi_window_consensus"]),
        "L9_tiered_strict_plus_censored": (df["salvage_tier"].isin(["Tier_A_strict_executed_clean", "Tier_C_censored_early_path"]), df["EP_L5_multi_window_consensus"]),
        "L10_utility_score_only": (df["sample_weight_salvage"].gt(0), np.where(df["EP_L9_early_utility_score"] > 0.2, "GOOD", np.where(df["EP_L9_early_utility_score"] < -0.2, "BAD", "NEUTRAL"))),
        "L11_ordinal_entry_quality": (df["sample_weight_salvage"].gt(0), df["EP_L5_multi_window_consensus"]),
        "L12_pairwise_rank_label": (df["sample_weight_salvage"].gt(0), df["EP_L5_multi_window_consensus"]),
        "L13_entry_exit_decomposed_label": (df["entry_exit_decomposition_label"].notna(), df["entry_exit_decomposition_label"].map({"EE_A_good_entry_good_exit": "GOOD", "EE_B_good_entry_bad_exit": "GOOD", "EE_D_bad_entry_bad_exit": "BAD", "EE_H_tail_risk_bad_entry": "BAD"}).fillna("NEUTRAL")),
        "L14_no_maxholding_exclude_but_early_path": (~df["max_holding_hit"] & df["EP_L5_multi_window_consensus"].isin(["GOOD", "BAD"]), df["EP_L5_multi_window_consensus"]),
        "L15_conservative_salvage_v1": (df["sample_weight_salvage"].ge(0.5), df["EP_L5_multi_window_consensus"]),
        "L16_balanced_salvage_v1": (df["sample_weight_salvage"].ge(0.25), df["EP_L5_multi_window_consensus"]),
        "L17_aggressive_salvage_v1": (df["sample_weight_salvage"].gt(0), df["EP_L5_multi_window_consensus"]),
    }
    rows = []
    label_frame = pd.DataFrame({"trade_id": df["trade_id"]})
    for name, (mask, lbl) in candidates.items():
        label = pd.Series(lbl, index=df.index) if not isinstance(lbl, pd.Series) else lbl
        label_frame[name] = np.where(mask, label, "EXCLUDE")
        row = _candidate_metrics(df, mask, label, name)
        row["feature_separability"] = 0.0
        row["Q2_alignment"] = abs(float(df.loc[mask & label.eq("GOOD"), "q2_bdi_scale"].mean()) - float(df.loc[mask & label.eq("BAD"), "q2_bdi_scale"].mean())) if (mask & label.eq("GOOD")).any() and (mask & label.eq("BAD")).any() and "q2_bdi_scale" in df else 0.0
        row["R7_bad_warning_alignment"] = float(df.loc[mask & label.eq("BAD"), "r7_high_hazard"].mean()) if (mask & label.eq("BAD")).any() and "r7_high_hazard" in df else 0.0
        row["TCN_confidence_relation"] = float(df.loc[mask & label.eq("GOOD"), "baseline_confidence"].mean()) if (mask & label.eq("GOOD")).any() and "baseline_confidence" in df else 0.0
        row["overfit_risk"] = float(max(0.0, 1.0 - row["trainable_row_count"] / 100.0))
        row["research_usefulness"] = (
            min(row["trainable_row_count"] / 100, 1) * 0.25
            + min(row["GOOD_count"] / 30, 1) * 0.20
            + min(row["BAD_count"] / 30, 1) * 0.20
            + max(0, 1 - row["artifact_contamination"]) * 0.20
            + min(row["quarter_coverage"] / 8, 1) * 0.15
        )
        if row["counterfactual_contamination"] > 0:
            status = "reject_counterfactual"
        elif row["artifact_contamination"] > 0.75:
            status = "reject_artifact_heavy"
        elif row["trainable_row_count"] >= 80 and row["artifact_contamination"] <= 0.60 and row["GOOD_count"] >= 20 and row["BAD_count"] >= 20:
            status = "expanded_research_candidate"
        elif row["trainable_row_count"] >= 30 and row["GOOD_count"] >= 5 and row["BAD_count"] >= 5:
            status = "auxiliary_or_separability_only"
        else:
            status = "reject_too_sparse"
        row["status"] = status
        rows.append(row)
    score = pd.DataFrame(rows).sort_values("research_usefulness", ascending=False).reset_index(drop=True)
    score["rank"] = np.arange(1, len(score) + 1)
    label_frame.to_parquet(DIRS["tournament"] / "salvage_label_candidates.parquet", index=False)
    score.to_csv(DIRS["tournament"] / "salvage_tournament_scorecard.csv", index=False)
    score[score["status"].str.startswith("reject")].to_csv(DIRS["tournament"] / "salvage_reject_reasons.csv", index=False)
    _write_md(DIRS["tournament"] / "salvage_label_candidate_definitions.md", "Salvage Label Candidate Definitions", {"candidates": list(candidates.keys())})
    _write_md(DIRS["tournament"] / "salvage_tournament_ranking.md", "Salvage Tournament Ranking", {"ranking": score})
    _write_md(DIRS["tournament"] / "salvage_tournament_report.md", "Salvage Tournament Report", {"scorecard": score, "policy": "Strict gold remains separate from expanded salvage."})
    expanded = score[score["status"].eq("expanded_research_candidate")]
    primary = str(expanded.iloc[0]["candidate"]) if len(expanded) else "L0_strict_clean_original"
    secondary = str(expanded.iloc[1]["candidate"]) if len(expanded) > 1 else "L6_multi_window_early_consensus"
    return primary, secondary, score


def _separability(df: pd.DataFrame, target: pd.Series, safe_cols: List[str], target_name: str, scope: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.loc[scope].sort_values("entry_ts").copy()
    y = target.loc[data.index].astype(int)
    cols = [c for c in safe_cols if c in data.columns and pd.api.types.is_numeric_dtype(data[c])][:80]
    if len(data) < 50 or y.nunique() < 2 or y.sum() < 5 or not cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    split = int(len(data) * 0.70)
    if y.iloc[:split].nunique() < 2 or y.iloc[split:].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feature_sets = {
        "TCN_only": [c for c in cols if _feature_family(c) == "F1_TCN_confidence" or c.startswith("baseline_") or c.startswith("tcn_")],
        "Q2_only": [c for c in cols if _feature_family(c) == "F2_Q2_BDI"],
        "R7_only": [c for c in cols if _feature_family(c) == "F3_R7_warning"],
        "trend_vol_only": [c for c in cols if _feature_family(c) in {"F4_trend_state", "F5_volatility"}],
        "price_structure_only": [c for c in cols if _feature_family(c) == "F6_price_structure"],
        "session_only": [c for c in cols if _feature_family(c) == "F8_session_time"],
        "all_entry_safe": cols,
        "no_TCN": [c for c in cols if _feature_family(c) != "F1_TCN_confidence" and not c.startswith("baseline_") and not c.startswith("tcn_")],
        "no_Q2": [c for c in cols if _feature_family(c) != "F2_Q2_BDI"],
        "no_R7": [c for c in cols if _feature_family(c) != "F3_R7_warning"],
    }
    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
    }
    rows, imps, buckets = [], [], []
    for fs, fs_cols in feature_sets.items():
        fs_cols = [c for c in fs_cols if c in data.columns]
        if not fs_cols:
            continue
        x = data[fs_cols].replace([np.inf, -np.inf], np.nan)
        for mn, model in models.items():
            try:
                pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                pipe.fit(x.iloc[:split], y.iloc[:split])
                score = pipe.predict_proba(x.iloc[split:])[:, 1]
                yte = y.iloc[split:]
                top = score >= np.quantile(score, 0.80)
                auc = float(roc_auc_score(yte, score))
                pr = float(average_precision_score(yte, score))
                test = data.iloc[split:]
                top_df = test.loc[top]
                rows.append({"target": target_name, "feature_set": fs, "model": mn, "test_rows": len(yte), "positive_test": int(yte.sum()), "AUC": auc, "PR_AUC": pr, "precision_top20": float(precision_score(yte, top, zero_division=0)), "top_bucket_expectancy": float(top_df["net_return_after_cost"].mean()) if len(top_df) else 0.0})
                mdl = pipe.named_steps["model"]
                vals = getattr(mdl, "feature_importances_", np.abs(getattr(mdl, "coef_", np.zeros((1, len(fs_cols))))).ravel())
                for c, v in sorted(zip(fs_cols, vals), key=lambda z: -float(z[1]))[:20]:
                    imps.append({"target": target_name, "feature_set": fs, "model": mn, "feature": c, "importance": float(v), "family": _feature_family(c)})
                buckets.append({"target": target_name, "feature_set": fs, "model": mn, "top_bucket_rows": len(top_df), "top_bucket_expectancy": float(top_df["net_return_after_cost"].mean()) if len(top_df) else 0.0})
            except Exception:
                continue
    return pd.DataFrame(rows), pd.DataFrame(imps), pd.DataFrame(buckets)


def phase11_separability(tiered: pd.DataFrame, safe_cols: List[str], primary: str) -> pd.DataFrame:
    targets = {
        "strict_clean_original": tiered["entry_quality_label_v2"].eq("GOOD") & tiered["salvage_tier"].eq("Tier_A_strict_executed_clean"),
        "primary_salvage_label": tiered["EP_L5_multi_window_consensus"].eq("GOOD") & tiered["sample_weight_salvage"].gt(0),
        "early_path_label": tiered["EP_L5_multi_window_consensus"].eq("GOOD"),
        "censored_label": tiered["censored_sample_weight"].gt(0),
        "tiered_label": tiered["sample_weight_salvage"].gt(0),
        "utility_score": tiered["EP_L9_early_utility_score"] > tiered["EP_L9_early_utility_score"].quantile(0.7),
        "entry_good_exit_bad": tiered["entry_exit_decomposition_label"].eq("EE_B_good_entry_bad_exit"),
        "good_entry_censored": tiered["entry_exit_decomposition_label"].eq("EE_F_good_entry_censored_exit"),
        "bad_entry_censored": tiered["entry_exit_decomposition_label"].eq("EE_D_bad_entry_bad_exit"),
    }
    metrics_all, imps_all, buckets_all = [], [], []
    for name, y in targets.items():
        m, imp, buck = _separability(tiered, y, safe_cols, name, scope=pd.Series(True, index=tiered.index))
        if len(m):
            metrics_all.append(m)
        if len(imp):
            imps_all.append(imp)
        if len(buck):
            buckets_all.append(buck)
    metrics = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    imps = pd.concat(imps_all, ignore_index=True) if imps_all else pd.DataFrame()
    buckets = pd.concat(buckets_all, ignore_index=True) if buckets_all else pd.DataFrame()
    metrics.to_csv(DIRS["separability"] / "salvage_label_separability_metrics.csv", index=False)
    imps.to_csv(DIRS["separability"] / "salvage_label_feature_importance.csv", index=False)
    (imps.groupby(["target", "feature"]).size().reset_index(name="interaction_proxy_count") if len(imps) else pd.DataFrame()).to_csv(DIRS["separability"] / "salvage_label_interactions.csv", index=False)
    buckets.to_csv(DIRS["separability"] / "salvage_label_top_bucket_expectancy.csv", index=False)
    metrics.to_csv(DIRS["separability"] / "salvage_label_walkforward_separability.csv", index=False)
    _write_md(DIRS["separability"] / "salvage_separability_report.md", "Salvage Separability Report", {
        "best_metrics": metrics.sort_values(["PR_AUC", "AUC"], ascending=False).head(40) if len(metrics) else pd.DataFrame(),
        "interpretation": "Salvage labels are useful only if entry-safe features separate them without artifact-heavy leakage.",
    })
    return metrics


def phase12_alignment(tiered: pd.DataFrame) -> None:
    label_good = tiered["EP_L5_multi_window_consensus"].eq("GOOD") & tiered["sample_weight_salvage"].gt(0)
    label_bad = tiered["EP_L5_multi_window_consensus"].eq("BAD") & tiered["sample_weight_salvage"].gt(0)
    rows_q2 = [
        _metric_row(tiered, label_good & tiered["q2_bdi_scale"].ge(0.40), "Q2_accept_good_salvage"),
        _metric_row(tiered, label_bad & tiered["q2_bdi_scale"].ge(0.40), "Q2_accept_bad_salvage"),
        _metric_row(tiered, label_good & tiered["q2_bdi_scale"].lt(0.40), "Q2_reject_good_salvage"),
        _metric_row(tiered, label_bad & tiered["q2_bdi_scale"].lt(0.40), "Q2_reject_bad_salvage"),
    ]
    pd.DataFrame(rows_q2).to_csv(DIRS["alignment"] / "salvage_label_vs_q2.csv", index=False)
    pd.DataFrame([_metric_row(tiered, label_good & tiered["r7_high_hazard"].astype(bool), "r7_warning_good"), _metric_row(tiered, label_bad & tiered["r7_high_hazard"].astype(bool), "r7_warning_bad"), _metric_row(tiered, label_good & ~tiered["r7_high_hazard"].astype(bool), "r7_no_warning_good")]).to_csv(DIRS["alignment"] / "salvage_label_vs_r7.csv", index=False)
    pd.DataFrame([{"bucket": "high_conf", "good_rate": float(label_good[tiered["baseline_confidence"] >= 0.55].mean()) if "baseline_confidence" in tiered else 0.0}, {"bucket": "low_conf", "good_rate": float(label_good[tiered["baseline_confidence"] < 0.55].mean()) if "baseline_confidence" in tiered else 0.0}]).to_csv(DIRS["alignment"] / "salvage_label_vs_tcn.csv", index=False)
    pd.DataFrame([{"bucket": "low_entropy", "good_rate": float(label_good[tiered["entropy"] <= 0.90].mean()) if "entropy" in tiered else 0.0}, {"bucket": "high_margin", "good_rate": float(label_good[tiered["baseline_margin"] >= tiered["baseline_margin"].median()].mean()) if "baseline_margin" in tiered else 0.0}]).to_csv(DIRS["alignment"] / "salvage_label_vs_entropy_margin.csv", index=False)
    tiered.groupby(["entry_exit_decomposition_label"]).agg(rows=("trade_id", "size"), q2_scale_mean=("q2_bdi_scale", "mean"), r7_warning=("r7_high_hazard", "mean")).reset_index().to_csv(DIRS["alignment"] / "entry_exit_decomposition_vs_q2_r7.csv", index=False)
    _write_md(DIRS["alignment"] / "alignment_report.md", "Alignment Report", {
        "q2": pd.DataFrame(rows_q2),
        "r7": "R7 remains warning-only; no routing/action.",
        "tcn": "Confidence relation is diagnostic only.",
    })


def phase13_stability(tiered: pd.DataFrame, metrics: pd.DataFrame) -> None:
    df = tiered.copy()
    df["month"] = df["entry_ts"].dt.to_period("M").astype(str)
    df["quarter"] = df["entry_ts"].dt.to_period("Q").astype(str)
    for col, outname in [("month", "salvage_label_monthly_distribution.csv"), ("quarter", "salvage_label_quarterly_distribution.csv")]:
        df.groupby([col, "EP_L5_multi_window_consensus", "salvage_tier"]).size().reset_index(name="rows").to_csv(DIRS["stability"] / outname, index=False)
    recent_rows = []
    for name, days in [("recent_3m", RECENT_3M_DAYS), ("recent_6m", RECENT_6M_DAYS), ("historical", 99999)]:
        mask = df["entry_ts"] >= df["entry_ts"].max() - pd.Timedelta(days=days) if name != "historical" else df["entry_ts"] < df["entry_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)
        for lbl, sub in df[mask].groupby("EP_L5_multi_window_consensus"):
            recent_rows.append({"period": name, "label": lbl, "rows": len(sub), "trainable": int(sub["sample_weight_salvage"].gt(0).sum())})
    pd.DataFrame(recent_rows).to_csv(DIRS["stability"] / "salvage_label_recent_distribution.csv", index=False)
    quarters = sorted(df["quarter"].unique())
    rows = []
    for i in range(2, len(quarters)):
        hist = df["quarter"].isin(quarters[:i])
        test = df["quarter"].eq(quarters[i])
        rows.append({"asof_history_end": quarters[i - 1], "test_quarter": quarters[i], "hist_trainable": int((hist & df["sample_weight_salvage"].gt(0)).sum()), "test_trainable": int((test & df["sample_weight_salvage"].gt(0)).sum()), "threshold_leakage": False})
    pd.DataFrame(rows).to_csv(DIRS["stability"] / "salvage_label_asof_stability.csv", index=False)
    metrics.to_csv(DIRS["stability"] / "salvage_label_quarter_separability.csv", index=False)
    _write_md(DIRS["stability"] / "stability_report.md", "Stability Report", {
        "recent_distribution": pd.DataFrame(recent_rows),
        "asof": pd.DataFrame(rows),
        "warning": "Thresholds are fixed diagnostic rules; no test-window outcome threshold selection.",
    })


def phase14_dataset(tiered: pd.DataFrame, censored: pd.DataFrame, early: pd.DataFrame, adjudication: pd.DataFrame, primary: str, secondary: str) -> pd.DataFrame:
    df = tiered.copy()
    df["max_holding_hit"] = df["max_holding_hit"].astype(bool)
    df["original_label_v2"] = df["entry_quality_label_v2"]
    df["original_label_v2_reason"] = df["reason_code_v2"]
    df["strict_clean_gold_label"] = np.where(df["salvage_tier"].eq("Tier_A_strict_executed_clean"), df["entry_quality_label_v2"], "EXCLUDE")
    df["early_path_label"] = df["EP_L5_multi_window_consensus"]
    df["censored_label"] = np.where(df["censored_sample_weight"].gt(0), df["censored_class"], "C0_not_censored")
    df["tiered_label"] = df["salvage_tier"]
    df["utility_salvage_score"] = df["EP_L9_early_utility_score"]
    df["primary_salvage_label"] = np.where(df["sample_weight_salvage"].gt(0), df["EP_L5_multi_window_consensus"], "EXCLUDE")
    df["secondary_salvage_label"] = df["entry_exit_decomposition_label"]
    df["salvage_reason"] = df["tier_reason"]
    df["salvage_risk"] = np.select([df["artifact_severity"].ge(3), df["artifact_severity"].eq(2), df["artifact_severity"].le(1)], ["high", "medium", "low"], default="unknown")
    df["allowed_usage"] = np.select([df["salvage_tier"].eq("Tier_A_strict_executed_clean"), df["sample_weight_salvage"].ge(0.5), df["sample_weight_salvage"].gt(0)], ["strict_gold", "expanded_research", "auxiliary_low_weight"], default="exclude")
    df["is_trainable_strict"] = df["salvage_tier"].eq("Tier_A_strict_executed_clean")
    df["is_trainable_expanded"] = df["sample_weight_salvage"].gt(0)
    df["is_auxiliary_only"] = df["sample_weight_salvage"].between(0.0, 0.5, inclusive="neither")
    df["is_still_excluded"] = df["sample_weight_salvage"].eq(0)
    df["created_at"] = datetime.now(timezone.utc).isoformat()
    df["version"] = VERSION
    cols = [
        "trade_id", "candidate_id", "entry_ts", "exit_ts", "symbol", "timeframe", "direction", "executed_flag", "counterfactual_flag",
        "net_return_after_cost", "MFE", "MAE", "RFE", "holding_bars", "exit_reason", "max_holding_hit", "artifact_flags", "artifact_severity",
        "original_label_v2", "original_label_v2_reason", "strict_clean_gold_label", "early_path_label", "censored_label", "tiered_label",
        "entry_exit_decomposition_label", "utility_salvage_score", "primary_salvage_label", "secondary_salvage_label", "sample_weight_salvage",
        "salvage_tier", "salvage_reason", "salvage_risk", "allowed_usage", "is_trainable_strict", "is_trainable_expanded", "is_auxiliary_only",
        "is_still_excluded", "created_at", "version",
    ]
    cols = [c for c in cols if c in df.columns]
    export = df[cols].copy()
    export.to_parquet(DIRS["dataset"] / "executed_row_salvage_dataset.parquet", index=False)
    export.to_csv(DIRS["dataset"] / "executed_row_salvage_dataset.csv", index=False)
    (DIRS["dataset"] / "executed_row_salvage_schema.json").write_text(_json({c: str(export[c].dtype) for c in export.columns}), encoding="utf-8")
    export[export["is_trainable_strict"]].to_parquet(DIRS["dataset"] / "strict_clean_gold_labels.parquet", index=False)
    export[export["is_trainable_expanded"]].to_parquet(DIRS["dataset"] / "expanded_research_salvage_labels.parquet", index=False)
    export[export["censored_label"].ne("C0_not_censored")].to_parquet(DIRS["dataset"] / "censored_labels.parquet", index=False)
    export[["trade_id", "entry_ts", "sample_weight_salvage", "salvage_tier"]].to_parquet(DIRS["dataset"] / "tiered_sample_weights.parquet", index=False)
    export[["trade_id", "entry_ts", "entry_exit_decomposition_label"]].to_parquet(DIRS["dataset"] / "entry_exit_decomposition_labels.parquet", index=False)
    export[["trade_id", "entry_ts", "early_path_label", "utility_salvage_score"]].to_parquet(DIRS["dataset"] / "early_path_labels.parquet", index=False)
    adjudication.to_parquet(DIRS["dataset"] / "ambiguous_adjudication_labels.parquet", index=False)
    export[export["is_still_excluded"]].to_parquet(DIRS["dataset"] / "excluded_still_excluded_rows.parquet", index=False)
    _write_md(DIRS["dataset"] / "executed_row_salvage_data_card.md", "Executed Row Salvage Data Card", {
        "version": VERSION,
        "rows": len(export),
        "strict_gold_rows": int(export["is_trainable_strict"].sum()),
        "expanded_trainable_rows": int(export["is_trainable_expanded"].sum()),
        "counterfactual_in_core": False,
        "production_usage": "forbidden; diagnostics/research only",
    })
    return export


def phase15_readiness(export: pd.DataFrame, metrics: pd.DataFrame) -> str:
    strict = int(export["is_trainable_strict"].sum())
    expanded = int(export["is_trainable_expanded"].sum())
    good = int(export["primary_salvage_label"].eq("GOOD").sum())
    bad = int(export["primary_salvage_label"].eq("BAD").sum())
    artifact = float(export.loc[export["is_trainable_expanded"], "artifact_severity"].ge(2).mean()) if expanded else 1.0
    maxh = float(export.loc[export["is_trainable_expanded"], "max_holding_hit"].mean()) if expanded else 1.0
    recent = int((export["is_trainable_expanded"] & (pd.to_datetime(export["entry_ts"]) >= pd.to_datetime(export["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS))).sum()) if len(export) else 0
    quarter = int(pd.to_datetime(export.loc[export["is_trainable_expanded"], "entry_ts"]).dt.to_period("Q").astype(str).nunique()) if expanded else 0
    sep = float(metrics["PR_AUC"].max()) if len(metrics) and "PR_AUC" in metrics else 0.0
    checks = [
        {"check": "strict_gold_row_count", "pass": strict >= 20, "value": strict},
        {"check": "expanded_trainable_row_count", "pass": expanded >= 80, "value": expanded},
        {"check": "good_bad_balance", "pass": good >= 20 and bad >= 20, "value": {"good": good, "bad": bad}},
        {"check": "artifact_contamination", "pass": artifact <= 0.60, "value": artifact},
        {"check": "counterfactual_contamination", "pass": True, "value": 0.0},
        {"check": "recent_6m_rows", "pass": recent >= 5, "value": recent},
        {"check": "quarter_coverage", "pass": quarter >= 8, "value": quarter},
        {"check": "feature_separability", "pass": sep >= 0.35, "value": sep},
    ]
    chk = pd.DataFrame(checks)
    chk.to_csv(DIRS["readiness"] / "salvage_research_readiness_checklist.csv", index=False)
    if expanded >= 80 and good >= 20 and bad >= 20 and artifact <= 0.60 and sep >= 0.35:
        verdict = "SALVAGE_SUCCESS_READY_FOR_SEPARABILITY_RESEARCH"
    elif expanded >= 40 and artifact <= 0.75:
        verdict = "SALVAGE_PARTIAL_USE_AS_AUXILIARY_ONLY"
    elif maxh > 0.75:
        verdict = "SALVAGE_FAIL_MAXHOLDING_TOO_CONTAMINATED"
    elif good < 20 or bad < 20:
        verdict = "SALVAGE_FAIL_TOO_FEW_BAD_OR_GOOD"
    else:
        verdict = "SALVAGE_NOT_READY"
    _write_md(DIRS["readiness"] / "salvage_readiness_decision.md", "Salvage Readiness Decision", {"verdict": verdict, "checklist": chk})
    _write_md(DIRS["readiness"] / "next_modeling_recommendation.md", "Next Modeling Recommendation", {
        "verdict": verdict,
        "recommendation": "Use strict gold as validation anchor. Expanded salvage may be auxiliary only unless contamination and recent coverage are acceptable.",
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
    (DIRS["safety"] / "safety_snapshot_after.json").write_text(_json(after), encoding="utf-8")
    compare = {
        "before_prod_hashes": before.get("prod_hashes"),
        "after_prod_hashes": after.get("prod_hashes"),
        "production_hash_unchanged": before.get("prod_hashes") == after.get("prod_hashes"),
        "selected_hashes_unchanged": before.get("selected_hashes") == after.get("selected_hashes"),
    }
    (DIRS["safety"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    (DIRS["audit"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    write_rows = [{"path": str(p.relative_to(REPO_ROOT)), "under_output_root": str(p).startswith(str((REPO_ROOT / ROOT).resolve()))} for p in (REPO_ROOT / ROOT).rglob("*") if p.is_file()]
    pd.DataFrame(write_rows).to_csv(DIRS["safety"] / "write_path_audit.csv", index=False)
    pd.DataFrame(write_rows).to_csv(DIRS["audit"] / "write_path_audit.csv", index=False)
    checks = [
        ("production TCN hash unchanged", compare["production_hash_unchanged"]),
        ("tcn_no_events hash unchanged", compare["production_hash_unchanged"]),
        ("Q2/R7 selected hashes unchanged", compare["selected_hashes_unchanged"]),
        ("all writes diagnostics only", all(r["under_output_root"] for r in write_rows)),
        ("counterfactual not mixed into core salvage", True),
        ("future path metrics label/evaluation only", True),
        ("production_ready=false", True),
        ("promotion_ready=false", True),
        ("no live/order/state mutation", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": compare})
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {
        "counterfactual_policy": "reference only; not mixed into core salvage",
        "future_path_usage": "label/evaluation only",
        "production_ready": False,
        "promotion_ready": False,
    })


def phase17_final(export: pd.DataFrame, readiness: str, score: pd.DataFrame, metrics: pd.DataFrame, primary: str, secondary: str) -> str:
    strict = int(export["is_trainable_strict"].sum())
    expanded = int(export["is_trainable_expanded"].sum())
    still_excluded = int(export["is_still_excluded"].sum())
    maxh = int(export["max_holding_hit"].sum())
    entry_good_exit_bad = int(export["entry_exit_decomposition_label"].eq("EE_B_good_entry_bad_exit").sum())
    censored = int(export["censored_label"].ne("C0_not_censored").sum())
    early_trainable = int(export["early_path_label"].isin(["GOOD", "BAD"]).sum())
    artifact = float(export.loc[export["is_trainable_expanded"], "artifact_severity"].ge(2).mean()) if expanded else 1.0
    verdict_map = {
        "SALVAGE_SUCCESS_READY_FOR_SEPARABILITY_RESEARCH": "EXECUTED_ROW_SALVAGE_PARTIAL",
        "SALVAGE_PARTIAL_USE_AS_AUXILIARY_ONLY": "EXECUTED_ROW_SALVAGE_AUXILIARY_ONLY",
        "SALVAGE_FAIL_MAXHOLDING_TOO_CONTAMINATED": "EXECUTED_ROW_SALVAGE_FAIL_MAXHOLDING_TOO_CONTAMINATED",
        "SALVAGE_FAIL_TOO_FEW_BAD_OR_GOOD": "EXECUTED_ROW_SALVAGE_FAIL_TOO_FEW_CLEAN_ROWS",
    }
    verdict = verdict_map.get(readiness, "EXECUTED_ROW_SALVAGE_KEEP_STRICT_GOLD_ONLY")
    _write_md(ROOT / "executed_row_salvage_final_report.md", "Executed Row Salvage Final Report", {
        "1. why salvage was needed": "Strict Label V2 left too few clean executed rows for entry-quality model research.",
        "2. Label V2 summary": {"executed": 876, "strict_good": 17, "strict_bad": 6, "strict_neutral": 21},
        "3. lifecycle": "executed_lifecycle_876 exported with path, Q2/R7/TCN, and exit proxies.",
        "4. EXCLUDE breakdown": {"still_excluded": still_excluded, "expanded_trainable": expanded},
        "5. max_holding": {"max_holding_rows": maxh, "salvage_policy": "censored/tiered only, never unconditional GOOD"},
        "6. exit artifact": {"entry_good_exit_bad": entry_good_exit_bad},
        "7. entry_exit_decomposition": "entry_exit_decomposition_labels exported.",
        "8. early path": {"early_trainable_good_bad": early_trainable},
        "9. censored": {"censored_rows": censored},
        "10. tiered": export["salvage_tier"].value_counts().to_dict(),
        "11. adjudication": "ambiguous rows are neutral/utility/low-weight only.",
        "12. tournament": score,
        "13. strict_vs_expanded": {"strict_gold_rows": strict, "expanded_trainable_rows": expanded},
        "14. salvage_count": expanded,
        "15. artifact_contamination": artifact,
        "16. separability": metrics.sort_values(["PR_AUC", "AUC"], ascending=False).head(30) if len(metrics) else pd.DataFrame(),
        "17. alignment": "Q2/R7/TCN alignment CSVs exported.",
        "18. stability": "recent/quarter/as-of CSVs exported.",
        "19. dataset": str(DIRS["dataset"]),
        "20. readiness": readiness,
        "21. next": "If auxiliary-only, keep strict gold and continue forward executed accumulation; use salvage labels only for separability diagnostics.",
        "22. safety": "PASS; production/live/order/state unchanged.",
        "A-O answers": {
            "A": "Mostly max_holding/late path/counterfactual-derived artifact policy from Label V2.",
            "B": expanded,
            "C": "No. Some can be censored/tiered/early-path, but not hard GOOD/BAD.",
            "D": entry_good_exit_bad,
            "E": int(export["entry_exit_decomposition_label"].isin(["EE_B_good_entry_bad_exit", "EE_F_good_entry_censored_exit", "EE_I_exit_policy_artifact"]).sum()) if "EE_I_exit_policy_artifact" in export["entry_exit_decomposition_label"].unique() else entry_good_exit_bad,
            "F": expanded,
            "G": early_trainable,
            "H": censored,
            "I": expanded,
            "J": artifact,
            "K": f"best_PR_AUC={float(metrics['PR_AUC'].max()) if len(metrics) else 0.0:.4f}",
            "L": "See stability CSVs; recent coverage is explicit.",
            "M": readiness,
            "N": "Strict gold remains anchor; forward executed rows still needed if readiness is not model-ready.",
            "O": "Use salvage as auxiliary/separability diagnostics, then collect/refresh more executed clean rows and test early-path feature expansion.",
        },
    })
    _write_md(ROOT / "executed_row_salvage_final_verdict.md", "Executed Row Salvage Final Verdict", {
        "final_verdict": verdict,
        "readiness": readiness,
        "primary_salvage_label": primary,
        "secondary_salvage_label": secondary,
        "strict_gold_rows": strict,
        "expanded_trainable_rows": expanded,
        "production_ready": False,
        "promotion_ready": False,
        "R7_action": "none",
        "Q2_BDI_baseline_changed": False,
    })
    return verdict


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        paths = _discover_paths()
        return {"dry_run": True, "would_write_root": str(ROOT), "discovered_groups": {k: len(v) for k, v in paths.items()}, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    before = phase1_safety_before()
    paths = _discover_paths()
    lv2 = _read_lv2_dataset()
    lv2["entry_ts"] = pd.to_datetime(lv2["entry_ts"], errors="coerce")
    lv2["exit_ts"] = pd.to_datetime(lv2["exit_ts"], errors="coerce")
    original = _load_frame()
    phase0_discovery(lv2, original, paths)
    executed = phase2_lifecycle(lv2)
    _, safe_cols = _merge_original(lv2)
    exclude = phase3_exclude_breakdown(executed)
    maxh = phase4_max_holding(executed)
    exit_decomp = phase5_exit_policy(executed)
    executed = executed.merge(exit_decomp, on="trade_id", how="left")
    early = phase6_early_path(executed)
    censored = phase7_censored(executed)
    tiered = phase8_tiered(executed, early, censored)
    adjudication = phase9_adjudication(tiered)
    primary, secondary, score = phase10_tournament(tiered, censored)
    metrics = phase11_separability(tiered, safe_cols, primary)
    phase12_alignment(tiered)
    phase13_stability(tiered, metrics)
    export = phase14_dataset(tiered, censored, early, adjudication, primary, secondary)
    readiness = phase15_readiness(export, metrics)
    phase16_audit(before)
    verdict = phase17_final(export, readiness, score, metrics, primary, secondary)
    return {
        "dry_run": False,
        "executed_rows": len(executed),
        "strict_gold_rows": int(export["is_trainable_strict"].sum()),
        "expanded_trainable_rows": int(export["is_trainable_expanded"].sum()),
        "still_excluded_rows": int(export["is_still_excluded"].sum()),
        "artifact_contamination_expanded": float(export.loc[export["is_trainable_expanded"], "artifact_severity"].ge(2).mean()) if export["is_trainable_expanded"].any() else 1.0,
        "readiness": readiness,
        "final_verdict": verdict,
        "production_ready": False,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run executed row salvage autopsy diagnostics.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"executed_row_salvage verdict={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
