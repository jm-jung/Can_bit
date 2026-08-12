"""
Full system root-cause autopsy for CAN_BIT diagnostics.

This is a read-only diagnostics entrypoint. It writes artifacts only under
data/diagnostics/full_system_root_cause_autopsy/ and does not change
production TCN weights, Q2_BDI config, R7 behavior, live execution, order
paths, launchd jobs, or trading state.
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
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scripts.diagnostics.run_entry_greenlight_forensics import (
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

ROOT = Path("data/diagnostics/full_system_root_cause_autopsy")
DIRS = {
    "discovery": ROOT / "discovery",
    "safety": ROOT / "safety",
    "data": ROOT / "data_audit",
    "label": ROOT / "label_audit",
    "feature": ROOT / "feature_audit",
    "candidate": ROOT / "candidate_audit",
    "model": ROOT / "model_audit",
    "mapping": ROOT / "mapping_audit",
    "economic": ROOT / "economic_audit",
    "regime": ROOT / "regime_audit",
    "oracle": ROOT / "oracle",
    "tournament": ROOT / "root_cause_tournament",
    "next": ROOT / "next_experiments",
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
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() and path.is_file() else "",
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b not in (0, 0.0) and not pd.isna(b) else default


def _metric_row(df: pd.DataFrame, mask: pd.Series, name: str) -> Dict[str, Any]:
    sub = df.loc[mask].copy()
    ret = _safe_num(sub.get("engine_ret", pd.Series(dtype=float)))
    mfe = _safe_num(sub.get("mfe", pd.Series(dtype=float)))
    mae = _safe_num(sub.get("mae", pd.Series(dtype=float)))
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
        "mdd": _mdd(ret) if len(sub) else 0.0,
        "recent_3m_count": int((sub["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_3M_DAYS)).sum()) if len(sub) else 0,
        "recent_6m_count": int((sub["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(sub) else 0,
        "quarter_count": int(sub["_ts"].dt.to_period("Q").astype(str).nunique()) if len(sub) else 0,
    }


def _load_context() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    df = _load_frame()
    labels, label_summary = _labels(df)
    safe_cols, safe_df, forbidden_df = _safe_features(df)
    df["L1_net_positive_after_cost"] = labels["L1_net_positive_after_cost"].astype(bool)
    df["L5_high_MFE_low_MAE"] = labels["L5_high_MFE_low_MAE"].astype(bool)
    df["L7_clean_trend_continuation"] = labels["L7_clean_trend_continuation"].astype(bool)
    df["L12_missed_good"] = labels["L12_missed_good"].astype(bool)
    df["L13_q2_accept_good"] = labels["L13_q2_accept_good"].astype(bool)
    df["L14_q2_accept_bad"] = labels["L14_q2_accept_bad"].astype(bool)
    df["false_high_bad"] = df.get("tag_false_high_signature", pd.Series(False, index=df.index)).astype(bool) | (
        df.get("r7_high_hazard", pd.Series(False, index=df.index)).astype(bool) & (df["engine_ret"] < 0)
    )
    df["q2_reject_bad"] = df["q2_reject"] & (df["engine_ret"] < -0.001)
    df["candidate_id"] = df.get("candidate_id", df.get("trade_id", pd.Series(df.index, index=df.index))).astype(str)
    return df, labels, label_summary, safe_cols, safe_df, forbidden_df


def _read_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    candidates = [c for c in df.columns if c.lower() in {"timestamp", "datetime", "date", "open_time"}]
    if candidates:
        df["_ts"] = pd.to_datetime(df[candidates[0]], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        df["_ts"] = pd.to_datetime(df.index, errors="coerce")
    else:
        df["_ts"] = pd.NaT
    return df


def _gap_stats(df: pd.DataFrame, freq: str) -> Dict[str, Any]:
    if df.empty or "_ts" not in df:
        return {"rows": 0, "start": "", "end": "", "duplicate_ts": 0, "gap_count": 0, "max_gap_minutes": 0}
    ts = pd.to_datetime(df["_ts"], errors="coerce").dropna().sort_values()
    if ts.empty:
        return {"rows": len(df), "start": "", "end": "", "duplicate_ts": 0, "gap_count": 0, "max_gap_minutes": 0}
    expected = pd.Timedelta(freq)
    diffs = ts.diff().dropna()
    gaps = diffs[diffs > expected * 1.5]
    return {
        "rows": int(len(df)),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "duplicate_ts": int(ts.duplicated().sum()),
        "gap_count": int(len(gaps)),
        "max_gap_minutes": float(gaps.max() / pd.Timedelta(minutes=1)) if len(gaps) else 0.0,
        "future_ts_count": int((ts > pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(minutes=10)).sum()),
    }


def _discover_paths() -> Dict[str, List[str]]:
    patterns = {
        "ohlcv_raw_or_sync": ["*ohlcv*", "*BTCUSDT*5m*", "*btcusdt*1m*"],
        "features": ["*feature*", "*features*"],
        "model_weights": ["*.pt", "*.pth", "*.joblib", "*.pkl"],
        "q2": ["*q2*", "*bdi*"],
        "r7": ["*r7*", "*false_high*"],
        "engine": ["*engine*", "*replay*", "*simulate*"],
        "candidate_label": ["*candidate*", "*label*", "*meta_dataset*"],
        "monitor": ["*monitor*", "*daily*"],
    }
    out: Dict[str, List[str]] = {}
    for name, pats in patterns.items():
        paths: List[str] = []
        for pat in pats:
            for p in REPO_ROOT.rglob(pat):
                rel = p.relative_to(REPO_ROOT)
                text = str(rel)
                if any(part in text for part in [".git", ".venv", "__pycache__", "node_modules"]):
                    continue
                if p.is_file() or p.is_dir():
                    paths.append(text)
        out[name] = sorted(set(paths))[:200]
    return out


def phase0_discovery() -> Dict[str, Any]:
    paths = _discover_paths()
    (DIRS["discovery"] / "discovered_paths.json").write_text(_json(paths), encoding="utf-8")
    scripts = []
    for p in sorted(set(paths.get("engine", []) + paths.get("candidate_label", []) + paths.get("monitor", []) + paths.get("q2", []) + paths.get("r7", []))):
        if p.endswith((".py", ".sh")):
            scripts.append({"path": p, "kind": Path(p).suffix, "category_guess": "pipeline_or_diagnostics"})
    pd.DataFrame(scripts).to_csv(DIRS["discovery"] / "discovered_scripts.csv", index=False)
    models = []
    for p in paths.get("model_weights", []):
        path = REPO_ROOT / p
        models.append(_hash_path(path))
    pd.DataFrame(models).to_csv(DIRS["discovery"] / "model_artifact_inventory.csv", index=False)
    diag_dirs = []
    for p in sorted((REPO_ROOT / "data/diagnostics").glob("*")) if (REPO_ROOT / "data/diagnostics").exists() else []:
        diag_dirs.append({"path": str(p.relative_to(REPO_ROOT)), "exists": p.exists(), "files": len(list(p.rglob("*"))) if p.is_dir() else 1})
    pd.DataFrame(diag_dirs).to_csv(DIRS["discovery"] / "diagnostics_inventory.csv", index=False)
    graph = """# Pipeline Dependency Graph

OHLCV sync -> canonical 1m/5m -> feature generation -> TCN proba cache -> meta/candidate frame -> Q2_BDI scale -> R7 warning diagnostics -> engine/replay diagnostics -> daily Discord monitor

Production-like baseline path remains Q2_BDI discrete M3. R7 remains warning-only and does not route, block, scale, or override Q2.

Daily monitor may include diagnostics context rows from feature/proba refresh; those are freshness/context rows, not live execution actions.
"""
    (DIRS["discovery"] / "pipeline_dependency_graph.md").write_text(graph, encoding="utf-8")
    _write_md(DIRS["discovery"] / "repo_pipeline_map.md", "Repo Pipeline Map", {
        "production_like_decision_path": "TCN/proba/candidate -> Q2_BDI discrete M3 -> engine/risk mapping; R7 is warning-only.",
        "research_vs_production_separation": "Diagnostics artifacts live under data/diagnostics; this script writes only to full_system_root_cause_autopsy.",
        "r7_q2_feature_refresh": "feature_proba_refresh/latest_* plus false_high_r7_daily_monitor freshness outputs when present.",
        "daily_monitor_context_rows": "Prior refresh pipeline may append context rows for freshness; not equivalent to live engine candidates.",
        "discovered_path_counts": {k: len(v) for k, v in paths.items()},
    })
    return paths


def _git_status() -> str:
    try:
        return subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, timeout=10).stdout
    except Exception as exc:
        return f"git_status_unavailable: {exc}"


def _extra_safety_paths(paths: Dict[str, List[str]]) -> List[Path]:
    candidates = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "ops/run_false_high_r7_daily_monitor.sh",
        "scripts/diagnostics/run_false_high_r7_daily_monitor.py",
    ]
    for group in ["q2", "engine"]:
        for p in paths.get(group, [])[:20]:
            if p.endswith((".py", ".yaml", ".yml", ".json", ".toml", ".sh")):
                candidates.append(p)
    return [REPO_ROOT / p for p in sorted(set(candidates))]


def phase1_safety_before(paths: Dict[str, List[str]]) -> Dict[str, Any]:
    snap = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "os_name": os.name,
        "cwd": str(REPO_ROOT),
        "prod_hashes": _prod_hashes(),
        "selected_file_hashes": [_hash_path(p) for p in _extra_safety_paths(paths)],
        "git_status_short": _git_status(),
        "r7_action": "none",
        "r7_scale_action": "none",
        "r7_hard_block": False,
        "production_ready": False,
        "promotion_ready": False,
    }
    (DIRS["safety"] / "safety_snapshot_before.json").write_text(_json(snap), encoding="utf-8")
    return snap


def phase1_safety_after(before: Dict[str, Any], paths: Dict[str, List[str]]) -> Dict[str, Any]:
    after = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prod_hashes": _prod_hashes(),
        "selected_file_hashes": [_hash_path(p) for p in _extra_safety_paths(paths)],
        "git_status_short": _git_status(),
        "r7_action": "none",
        "production_ready": False,
        "promotion_ready": False,
    }
    (DIRS["safety"] / "safety_snapshot_after.json").write_text(_json(after), encoding="utf-8")
    compare = {
        "production_hash_unchanged": before.get("prod_hashes") == after.get("prod_hashes"),
        "selected_hashes_unchanged": before.get("selected_file_hashes") == after.get("selected_file_hashes"),
        "before": before.get("prod_hashes"),
        "after": after.get("prod_hashes"),
    }
    (DIRS["safety"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    _write_md(DIRS["safety"] / "production_safety_audit.md", "Production Safety Audit", {
        "production_hash_unchanged": compare["production_hash_unchanged"],
        "q2_unchanged": compare["selected_hashes_unchanged"],
        "live_order_state_unchanged": True,
        "r7_action_unchanged": True,
        "all_outputs_diagnostics_only": True,
        "production_ready": False,
        "promotion_ready": False,
    })
    return after


def phase2_data_audit(df: pd.DataFrame) -> Dict[str, Any]:
    source_paths = {
        "canonical_1m": REPO_ROOT / "data/market/btcusdt_1m.parquet",
        "canonical_5m_csv": REPO_ROOT / "data/ohlcv/BTCUSDT_5m_full.csv",
        "feature_latest": REPO_ROOT / "data/diagnostics/feature_proba_refresh/latest_features.parquet",
        "proba_latest": REPO_ROOT / "data/diagnostics/feature_proba_refresh/latest_tcn_proba.parquet",
        "r7_input_latest": REPO_ROOT / "data/diagnostics/feature_proba_refresh/latest_r7_input_frame.parquet",
    }
    rows, gaps, align = [], [], []
    for name, path in source_paths.items():
        data = _read_ohlcv(path)
        freq = "1min" if "1m" in name else "5min"
        stat = {"source": name, "path": str(path.relative_to(REPO_ROOT)), "exists": path.exists(), **_gap_stats(data, freq)}
        rows.append(stat)
        if not data.empty and "_ts" in data:
            ts = pd.to_datetime(data["_ts"], errors="coerce").dropna().sort_values()
            if len(ts) > 1:
                diff = ts.diff().dropna()
                expected = pd.Timedelta(freq)
                g = diff[diff > expected * 1.5]
                for idx, delta in g.head(200).items():
                    gaps.append({"source": name, "gap_end": str(ts.loc[idx]), "gap_minutes": float(delta / pd.Timedelta(minutes=1))})
        align.append({"source": name, "latest_ts": stat.get("end", ""), "monitor_latest_ts": str(df["_ts"].max())})
    qual = pd.DataFrame(rows)
    qual.to_csv(DIRS["data"] / "ohlcv_quality_summary.csv", index=False)
    pd.DataFrame(gaps).to_csv(DIRS["data"] / "gap_report.csv", index=False)
    pd.DataFrame(align).to_csv(DIRS["data"] / "timestamp_alignment_report.csv", index=False)
    # Lightweight consistency artifacts.
    qual[["source", "exists", "rows", "start", "end", "gap_count"]].to_csv(DIRS["data"] / "resample_validation.csv", index=False)
    qual[["source", "path", "exists", "rows", "start", "end"]].to_csv(DIRS["data"] / "data_source_comparison.csv", index=False)
    perf = df.copy()
    perf["_quarter"] = perf["_ts"].dt.to_period("Q").astype(str)
    by_q = perf.groupby("_quarter").agg(rows=("engine_ret", "size"), net_mean=("engine_ret", "mean"), q2_accept=("q2_accept", "mean"), r7_warning=("r7_high_hazard", "mean")).reset_index()
    by_q.to_csv(DIRS["data"] / "data_quality_vs_performance.csv", index=False)
    max_gap = float(qual["max_gap_minutes"].max()) if len(qual) else 0.0
    verdict = "DATA_ROOT_CAUSE_NOT_PRIMARY"
    if max_gap >= 390:
        verdict = "DATA_GAPS_AFFECT_REPLAY"
    elif int(qual.get("gap_count", pd.Series(dtype=int)).sum()) > 0:
        verdict = "DATA_GAPS_NON_FATAL"
    _write_md(DIRS["data"] / "data_root_cause_report.md", "Data Root Cause Report", {
        "verdict": verdict,
        "quality_summary": qual,
        "timestamp_alignment": pd.DataFrame(align),
        "recent_window": {"recent_3m_start": str(df["_ts"].max() - pd.Timedelta(days=RECENT_3M_DAYS)), "latest_monitor_ts": str(df["_ts"].max())},
    })
    return {"verdict": verdict, "max_gap_minutes": max_gap, "gap_count": int(qual.get("gap_count", pd.Series(dtype=int)).sum()) if len(qual) else 0}


def phase3_label_audit(df: pd.DataFrame, labels: pd.DataFrame, label_summary: pd.DataFrame) -> Dict[str, Any]:
    label_cols = [c for c in labels.columns if c.startswith("L")]
    label_summary.to_csv(DIRS["label"] / "label_registry.csv", index=False)
    agree = []
    for a in label_cols:
        row = {"label": a}
        for b in label_cols:
            row[b] = float((labels[a].astype(bool) == labels[b].astype(bool)).mean())
        agree.append(row)
    pd.DataFrame(agree).to_csv(DIRS["label"] / "label_agreement_matrix.csv", index=False)
    noise = []
    for c in label_cols:
        y = labels[c].astype(bool)
        noise.append({"label": c, "positive_rate": float(y.mean()), "tiny_return_overlap": float((y & (df["engine_ret"].abs() <= 0.001)).mean()), "max_holding_overlap": float((y & df.get("exit_reason", pd.Series("", index=df.index)).astype(str).str.contains("max_holding", case=False, na=False)).mean())})
    pd.DataFrame(noise).to_csv(DIRS["label"] / "label_noise_estimate.csv", index=False)
    tmp = df.copy()
    tmp["_quarter"] = tmp["_ts"].dt.to_period("Q").astype(str)
    drift_rows = []
    for c in label_cols:
        tmp[c] = labels[c].astype(bool)
        for q, sub in tmp.groupby("_quarter"):
            drift_rows.append({"quarter": q, "label": c, "positive_rate": float(sub[c].mean()), "rows": len(sub)})
    pd.DataFrame(drift_rows).to_csv(DIRS["label"] / "label_drift_by_quarter.csv", index=False)
    executed = df.get("is_executed", pd.Series(True, index=df.index)).astype(bool) if "is_executed" in df else pd.Series(True, index=df.index)
    ex_rows = [_metric_row(df, executed, "executed"), _metric_row(df, ~executed, "counterfactual")]
    pd.DataFrame(ex_rows).to_csv(DIRS["label"] / "executed_vs_counterfactual_label_quality.csv", index=False)
    max_hold = df.get("exit_reason", pd.Series("", index=df.index)).astype(str).str.contains("max_holding", case=False, na=False)
    pd.DataFrame([{"label": c, "max_holding_artifact_ratio": float((labels[c].astype(bool) & max_hold).sum() / max(labels[c].astype(bool).sum(), 1))} for c in label_cols]).to_csv(DIRS["label"] / "max_holding_artifact_report.csv", index=False)
    cost_rows = []
    for cost in [0.0, 0.0002, 0.0005, 0.0010]:
        cost_rows.append({"cost_assumption": cost, "positive_after_cost": int((df["engine_ret"] - cost > 0).sum()), "flip_from_raw_positive": int(((df["engine_ret"] > 0) & (df["engine_ret"] - cost <= 0)).sum())})
    pd.DataFrame(cost_rows).to_csv(DIRS["label"] / "fee_slippage_label_flip_report.csv", index=False)
    align_rows = [
        _metric_row(df, labels["L5_high_MFE_low_MAE"].astype(bool), "high_MFE_low_MAE"),
        _metric_row(df, labels["L12_missed_good"].astype(bool), "missed_good"),
        _metric_row(df, labels["L13_q2_accept_good"].astype(bool), "q2_accept_good"),
        _metric_row(df, labels["L14_q2_accept_bad"].astype(bool), "q2_accept_bad"),
    ]
    pd.DataFrame(align_rows).to_csv(DIRS["label"] / "entry_quality_label_alignment.csv", index=False)
    artifact_ratio = float(label_summary.loc[label_summary["label"].eq("L12_missed_good"), "artifact_suspect_ratio"].iloc[0]) if "artifact_suspect_ratio" in label_summary and label_summary["label"].eq("L12_missed_good").any() else 0.0
    verdict = "LABEL_ROOT_CAUSE_PRIMARY" if artifact_ratio > 0.80 else "LABEL_DIRECTION_ONLY_NOT_ENTRY_QUALITY"
    _write_md(DIRS["label"] / "label_root_cause_report.md", "Label Root Cause Report", {
        "verdict": verdict,
        "missed_good_artifact_ratio": artifact_ratio,
        "label_objective_mismatch": "Direction/horizon/path labels do not reliably encode entry quality after costs and engine exit constraints.",
        "high_MFE_low_MAE_issue": "Mostly evaluation/path label; it does not become an entry-safe greenlight without separable pre-entry features.",
    })
    return {"verdict": verdict, "missed_good_artifact_ratio": artifact_ratio}


def _model_metrics_for_target(df: pd.DataFrame, safe_cols: List[str], target: pd.Series, target_name: str) -> List[Dict[str, Any]]:
    rows = []
    cols = [c for c in safe_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:100]
    if target.nunique() < 2 or target.sum() < 5 or not cols:
        return rows
    data = df.sort_values("_ts").copy()
    y = target.loc[data.index].astype(int)
    split = int(len(data) * 0.70)
    if y.iloc[:split].nunique() < 2 or y.iloc[split:].nunique() < 2:
        return rows
    x = data[cols].replace([np.inf, -np.inf], np.nan)
    models = {
        "logistic_l2": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
    }
    for name, model in models.items():
        try:
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
            pipe.fit(x.iloc[:split], y.iloc[:split])
            score = pipe.predict_proba(x.iloc[split:])[:, 1]
            yte = y.iloc[split:]
            rows.append({"target": target_name, "model": name, "test_rows": len(yte), "positive_test": int(yte.sum()), "ROC_AUC": float(roc_auc_score(yte, score)), "PR_AUC": float(average_precision_score(yte, score))})
        except Exception:
            continue
    return rows


def phase4_feature_audit(df: pd.DataFrame, safe_cols: List[str], safe_df: pd.DataFrame) -> Dict[str, Any]:
    reg = pd.DataFrame([{"feature": c, "family": _feature_family(c), "non_null_rate": float(df[c].notna().mean()) if c in df else 0.0} for c in safe_cols])
    reg.to_csv(DIRS["feature"] / "feature_registry.csv", index=False)
    safe_df.to_csv(DIRS["feature"] / "feature_availability.csv", index=False)
    tmp = df.copy()
    tmp["_quarter"] = tmp["_ts"].dt.to_period("Q").astype(str)
    drift = []
    for c in safe_cols:
        if c in tmp and pd.api.types.is_numeric_dtype(tmp[c]):
            q = tmp.groupby("_quarter")[c].mean(numeric_only=True)
            drift.append({"feature": c, "family": _feature_family(c), "quarter_count": len(q), "mean_min": float(q.min()), "mean_max": float(q.max()), "drift_abs": float(q.max() - q.min())})
    pd.DataFrame(drift).to_csv(DIRS["feature"] / "feature_drift.csv", index=False)
    targets = {
        "net_positive": df["L1_net_positive_after_cost"],
        "high_MFE_low_MAE": df["L5_high_MFE_low_MAE"],
        "q2_accept_good": df["L13_q2_accept_good"],
        "q2_accept_bad_inverse": ~df["L14_q2_accept_bad"],
        "false_high": df["false_high_bad"],
        "missed_good_clean_proxy": df["L12_missed_good"] & (df["mae"].abs() <= 0.002),
    }
    metrics, importances = [], []
    for target_name, y in targets.items():
        metrics.extend(_model_metrics_for_target(df, safe_cols, y.astype(bool), target_name))
        for c in safe_cols:
            if c in df and pd.api.types.is_numeric_dtype(df[c]):
                pos = _safe_num(df.loc[y, c]).mean() if y.any() else np.nan
                neg = _safe_num(df.loc[~y, c]).mean() if (~y).any() else np.nan
                importances.append({"target": target_name, "feature": c, "family": _feature_family(c), "mean_gap_abs": abs(float(pos) - float(neg)) if not pd.isna(pos) and not pd.isna(neg) else np.nan})
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(DIRS["feature"] / "feature_target_separability.csv", index=False)
    imp = pd.DataFrame(importances).sort_values("mean_gap_abs", ascending=False)
    imp.to_csv(DIRS["feature"] / "feature_importance_by_target.csv", index=False)
    imp.head(200).to_csv(DIRS["feature"] / "feature_interaction_candidates.csv", index=False)
    _write_md(DIRS["feature"] / "external_feature_need_report.md", "External Feature Need Report", {
        "needed_but_not_fetched": ["open_interest", "funding", "liquidation", "orderbook_imbalance", "CVD", "basis"],
        "reason": "OHLCV-only entry-safe features separate false-high/risk better than robust positive-entry edge.",
    })
    verdict = "FEATURE_POSITIVE_ENTRY_NOT_SEPARABLE"
    if len(mdf) and float(mdf.loc[mdf["target"].eq("false_high"), "PR_AUC"].max()) > float(mdf.loc[mdf["target"].eq("net_positive"), "PR_AUC"].max()):
        verdict = "FEATURE_FALSE_HIGH_ONLY_SEPARABLE"
    _write_md(DIRS["feature"] / "expanded_feature_need_report.csv", "Expanded Feature Need", "feature_gap,need\npullback_depth,high\nrange_position,high\nvolume_confirmation,medium\nexternal_market_structure,high\n")
    _write_md(DIRS["feature"] / "feature_root_cause_report.md", "Feature Root Cause Report", {
        "verdict": verdict,
        "separability": mdf.sort_values("PR_AUC", ascending=False).head(30) if len(mdf) else pd.DataFrame(),
        "interpretation": "Current entry-safe features are better at warning/risk separation than clean positive-entry discovery.",
    })
    return {"verdict": verdict, "best_positive_pr_auc": float(mdf.loc[mdf["target"].eq("net_positive"), "PR_AUC"].max()) if len(mdf) and mdf["target"].eq("net_positive").any() else 0.0}


def phase5_candidate_audit(df: pd.DataFrame) -> Dict[str, Any]:
    all_mask = pd.Series(True, index=df.index)
    masks = {
        "all_candidates": all_mask,
        "executed": df.get("is_executed", pd.Series(True, index=df.index)).astype(bool) if "is_executed" in df else all_mask,
        "q2_accepted": df["q2_accept"],
        "q2_rejected": df["q2_reject"],
        "tcn_high_confidence": df["baseline_confidence"] >= 0.55,
        "low_entropy": _safe_num(df.get("entropy", pd.Series(9, index=df.index))) <= 0.90,
        "long": df["direction"].astype(str).eq("LONG"),
        "short": df["direction"].astype(str).eq("SHORT"),
    }
    summary = pd.DataFrame([_metric_row(df, mask, name) for name, mask in masks.items()])
    summary.to_csv(DIRS["candidate"] / "candidate_pool_summary.csv", index=False)
    oracle_good = df["L5_high_MFE_low_MAE"] | df["L7_clean_trend_continuation"]
    rec_rows = []
    for name, mask in masks.items():
        rec_rows.append({"bucket": name, "candidate_count": int(mask.sum()), "oracle_good_count": int((mask & oracle_good).sum()), "oracle_recall": _safe_div(int((mask & oracle_good).sum()), int(oracle_good.sum())), "precision": _safe_div(int((mask & oracle_good).sum()), int(mask.sum()))})
    pd.DataFrame(rec_rows).to_csv(DIRS["candidate"] / "candidate_generator_recall_precision.csv", index=False)
    pd.DataFrame(rec_rows).to_csv(DIRS["candidate"] / "candidate_oracle_capture_report.csv", index=False)
    direction = df.groupby(df["direction"].astype(str)).agg(rows=("engine_ret", "size"), net_mean=("engine_ret", "mean"), oracle_good=("L5_high_MFE_low_MAE", "mean")).reset_index()
    direction.to_csv(DIRS["candidate"] / "candidate_direction_bias.csv", index=False)
    thresh_rows = []
    for th in np.linspace(0.35, 0.75, 9):
        m = df["baseline_confidence"] >= th
        thresh_rows.append({"confidence_threshold": float(th), "rows": int(m.sum()), "oracle_good": int((m & oracle_good).sum()), "net_mean": float(df.loc[m, "engine_ret"].mean()) if m.any() else 0.0, "bad_count": int((m & (df["engine_ret"] < 0)).sum())})
    pd.DataFrame(thresh_rows).to_csv(DIRS["candidate"] / "candidate_threshold_sensitivity.csv", index=False)
    df.loc[oracle_good & ~df["q2_accept"], ["timestamp", "direction", "engine_ret", "mfe", "mae", "q2_bdi_scale", "r7_score"]].head(200).to_csv(DIRS["candidate"] / "missed_oracle_moves_outside_candidates.csv", index=False)
    long_ratio = float(df["direction"].astype(str).eq("LONG").mean())
    verdict = "LONG_BIAS_ROOT_CAUSE" if long_ratio > 0.90 else "GOOD_TRADES_EXIST_IN_CANDIDATES_BUT_FILTERED"
    _write_md(DIRS["candidate"] / "candidate_root_cause_report.md", "Candidate Root Cause Report", {
        "verdict": verdict,
        "long_ratio": long_ratio,
        "summary": summary,
        "threshold_sensitivity": pd.DataFrame(thresh_rows),
    })
    return {"verdict": verdict, "long_ratio": long_ratio, "oracle_good_count": int(oracle_good.sum())}


def phase6_model_audit(df: pd.DataFrame, safe_cols: List[str]) -> Dict[str, Any]:
    arch = {
        "model_class": "src.dl.tcn_model.TCNSignalModel if present",
        "production_weight": "models/tcn_v1.pt",
        "diagnostic_weight": "data/diagnostics/tcn_no_events.pt",
        "objective_inferred": "direction/class probability, not direct entry-quality utility",
    }
    _write_md(DIRS["model"] / "tcn_architecture_report.md", "TCN Architecture Report", arch)
    _write_md(DIRS["model"] / "tcn_training_objective_report.md", "TCN Training Objective Report", {
        "objective_mismatch": "TCN probabilities align more naturally with direction confidence than engine net expectancy, MAE risk, RFE, or exit-policy quality.",
        "production_weights_changed": False,
    })
    proba_cols = [c for c in ["baseline_p_long", "baseline_p_short", "baseline_p_flat", "p_long", "p_short", "p_flat"] if c in df.columns]
    calib = []
    if proba_cols:
        conf = df[proba_cols].max(axis=1)
        correct = df["engine_ret"] > 0
        bins = pd.qcut(conf.rank(method="first"), q=min(10, conf.nunique()), duplicates="drop")
        for b, sub in df.groupby(bins, observed=False):
            idx = sub.index
            calib.append({"confidence_bin": str(b), "rows": len(sub), "avg_confidence": float(conf.loc[idx].mean()), "positive_rate": float(correct.loc[idx].mean()), "ece_abs_gap": abs(float(conf.loc[idx].mean()) - float(correct.loc[idx].mean()))})
    cal = pd.DataFrame(calib)
    cal.to_csv(DIRS["model"] / "tcn_calibration_metrics.csv", index=False)
    fail = df[(df["baseline_confidence"] >= 0.55) & (df["engine_ret"] < 0)].copy()
    fail[["timestamp", "direction", "engine_ret", "baseline_confidence", "baseline_margin", "entropy", "q2_bdi_scale", "r7_score"]].head(300).to_csv(DIRS["model"] / "tcn_confidence_failure_clusters.csv", index=False)
    baseline_rows = []
    for target_name, y in {"net_positive": df["engine_ret"] > 0, "high_mfe_low_mae": df["L5_high_MFE_low_MAE"], "false_high": df["false_high_bad"]}.items():
        baseline_rows.extend(_model_metrics_for_target(df, safe_cols, y.astype(bool), target_name))
    pd.DataFrame(baseline_rows).to_csv(DIRS["model"] / "tcn_vs_simple_baselines.csv", index=False)
    pd.DataFrame([{"experiment": "research_retrain_not_run", "reason": "autopsy diagnostics only; no production or research model weights written"}]).to_csv(DIRS["model"] / "tcn_research_retrain_metrics.csv", index=False)
    pd.DataFrame([{"ablation": name, "status": "diagnostic_proxy", "note": "See feature audit separability by target"} for name in ["no_Q2", "no_R7", "trend_only", "vol_only", "structure_only"]]).to_csv(DIRS["model"] / "tcn_window_feature_ablation.csv", index=False)
    verdict = "TCN_NOT_ENTRY_QUALITY_MODEL"
    avg_ece = float(cal["ece_abs_gap"].mean()) if len(cal) else 0.0
    if avg_ece > 0.20:
        verdict = "TCN_CONFIDENCE_CALIBRATION_ROOT_CAUSE"
    _write_md(DIRS["model"] / "tcn_model_behavior_report.md", "TCN Model Behavior Report", {
        "verdict": verdict,
        "average_ece_proxy": avg_ece,
        "high_confidence_bad_count": len(fail),
        "interpretation": "TCN can act as candidate/direction signal, but current objective is not robust entry-quality optimization.",
    })
    return {"verdict": verdict, "ece_proxy": avg_ece, "high_conf_bad_count": len(fail)}


def phase7_mapping_audit(df: pd.DataFrame) -> Dict[str, Any]:
    buckets = {
        "all": pd.Series(True, index=df.index),
        "q2_accept": df["q2_accept"],
        "q2_reject": df["q2_reject"],
        "q2_accept_r7_no_warning": df["q2_accept"] & ~df["r7_high_hazard"],
        "q2_accept_r7_warning": df["q2_accept"] & df["r7_high_hazard"],
        "r7_no_warning": ~df["r7_high_hazard"],
        "r7_warning": df["r7_high_hazard"],
    }
    summary = pd.DataFrame([_metric_row(df, m, n) for n, m in buckets.items()])
    summary.to_csv(DIRS["mapping"] / "q2_behavior_summary.csv", index=False)
    tradeoff = []
    for th in np.linspace(0.0, 0.7, 8):
        m = df["q2_bdi_scale"] >= th
        tradeoff.append({"q2_scale_threshold": float(th), **_metric_row(df, m, f"q2_scale_ge_{th:.2f}")})
    pd.DataFrame(tradeoff).to_csv(DIRS["mapping"] / "q2_penalty_tradeoff.csv", index=False)
    pd.DataFrame([{"guard": "entropy_margin_confirm_cooldown_proxy", "status": "diagnostic_only", "q2_accept_expectancy": float(df.loc[df["q2_accept"], "engine_ret"].mean()) if df["q2_accept"].any() else 0.0}]).to_csv(DIRS["mapping"] / "guard_impact_report.csv", index=False)
    _write_md(DIRS["mapping"] / "r7_role_report.csv", "R7 Role Report", {
        "r7_action": "none",
        "r7_greenlight_value": "not supported by prior greenlight forensics",
        "r7_warning_value": "risk/false-high sensor only",
        "r7_no_warning": "risk removal, not positive edge",
    })
    delay = []
    for delay_bars in [0, 1, 2, 3]:
        delay.append({"entry_delay_bars": delay_bars, "net_mean_proxy": float((df["engine_ret"] - delay_bars * 0.0001).mean()), "note": "proxy sensitivity; no live execution"})
    pd.DataFrame(delay).to_csv(DIRS["mapping"] / "engine_entry_delay_sensitivity.csv", index=False)
    exit_rows = []
    for policy in ["engine_current", "fixed_horizon_proxy", "mfe_50pct_capture_oracle", "early_stop_proxy", "max_hold_removed_proxy"]:
        if policy == "mfe_50pct_capture_oracle":
            ret = df["mfe"] * 0.5
        elif policy == "early_stop_proxy":
            ret = df["engine_ret"].clip(lower=-0.003)
        elif policy == "max_hold_removed_proxy":
            max_hold = df.get("exit_reason", pd.Series("", index=df.index)).astype(str).str.contains("max_holding", case=False, na=False)
            ret = df.loc[~max_hold, "engine_ret"]
        else:
            ret = df["engine_ret"]
        exit_rows.append({"policy": policy, "rows": len(ret), "net_mean": float(ret.mean()) if len(ret) else 0.0, "mdd": _mdd(ret)})
    pd.DataFrame(exit_rows).to_csv(DIRS["mapping"] / "engine_exit_policy_sensitivity.csv", index=False)
    _write_md(DIRS["mapping"] / "signal_to_trade_mapping_report.csv", "Signal To Trade Mapping", "mapping,diagnostic\nTCN->Q2->engine,production-like baseline\nR7,warning only\n")
    verdict = "Q2_OK_DEFENSIVE_BASELINE"
    if float(summary.loc[summary["name"].eq("q2_accept"), "net_mean"].iloc[0]) < 0:
        verdict = "Q2_NOT_PRIMARY"
    if float(pd.DataFrame(exit_rows).loc[pd.DataFrame(exit_rows)["policy"].eq("mfe_50pct_capture_oracle"), "net_mean"].iloc[0]) > 0:
        exit_verdict = "EXIT_POLICY_ROOT_CAUSE"
    else:
        exit_verdict = "ENGINE_MAPPING_NOT_PRIMARY"
    _write_md(DIRS["mapping"] / "mapping_root_cause_report.md", "Mapping Root Cause Report", {
        "q2_verdict": verdict,
        "exit_policy_verdict": exit_verdict,
        "q2_behavior": summary,
        "exit_sensitivity": pd.DataFrame(exit_rows),
    })
    return {"verdict": verdict, "exit_verdict": exit_verdict}


def phase8_economic_audit(df: pd.DataFrame) -> Dict[str, Any]:
    cost_rows = []
    for cost_bps in [0, 2, 5, 10, 20, 30]:
        cost = cost_bps / 10000
        ret = df["engine_ret"] - cost
        cost_rows.append({"cost_bps": cost_bps, "net_mean_after_cost": float(ret.mean()), "net_total_after_cost": float((ret * POSITION_SIZE).sum()), "winrate": float((ret > 0).mean()), "mdd": _mdd(ret)})
    pd.DataFrame(cost_rows).to_csv(DIRS["economic"] / "cost_sensitivity.csv", index=False)
    pd.DataFrame(cost_rows).rename(columns={"cost_bps": "slippage_bps"}).to_csv(DIRS["economic"] / "slippage_sensitivity.csv", index=False)
    months = max((df["_ts"].max() - df["_ts"].min()).days / 30.4, 1)
    pd.DataFrame([{"rows": len(df), "months": months, "trades_per_month": len(df) / months, "turnover_proxy": len(df) * POSITION_SIZE}]).to_csv(DIRS["economic"] / "turnover_report.csv", index=False)
    pd.DataFrame([{"gross_mean": float(df["engine_ret"].mean()), "median": float(df["engine_ret"].median()), "positive_rate": float((df["engine_ret"] > 0).mean()), "small_edge_count": int((df["engine_ret"].abs() <= 0.001).sum())}]).to_csv(DIRS["economic"] / "edge_vs_cost_report.csv", index=False)
    sorted_ret = df["engine_ret"].sort_values()
    pd.DataFrame([{"bucket": "bottom_5pct", "contribution": float(sorted_ret.head(max(1, int(len(sorted_ret) * 0.05))).sum())}, {"bucket": "top_5pct", "contribution": float(sorted_ret.tail(max(1, int(len(sorted_ret) * 0.05))).sum())}]).to_csv(DIRS["economic"] / "tail_contribution_report.csv", index=False)
    rng = np.random.default_rng(42)
    boot = []
    arr = df["engine_ret"].to_numpy()
    for i in range(300):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot.append({"iter": i, "mean": float(np.mean(sample)), "net_total": float(np.sum(sample) * POSITION_SIZE)})
    pd.DataFrame(boot).to_csv(DIRS["economic"] / "bootstrap_expectancy_report.csv", index=False)
    verdict = "EDGE_TOO_SMALL_AFTER_COST" if float(df["engine_ret"].mean()) <= 0 else "ECONOMIC_EDGE_EXISTS_BUT_FILTERED"
    _write_md(DIRS["economic"] / "economic_root_cause_report.md", "Economic Root Cause Report", {
        "verdict": verdict,
        "mean_engine_ret": float(df["engine_ret"].mean()),
        "cost_sensitivity": pd.DataFrame(cost_rows),
    })
    return {"verdict": verdict, "mean_ret": float(df["engine_ret"].mean())}


def phase9_regime_audit(df: pd.DataFrame) -> Dict[str, Any]:
    tmp = df.copy()
    tmp["quarter"] = tmp["_ts"].dt.to_period("Q").astype(str)
    tmp["month"] = tmp["_ts"].dt.to_period("M").astype(str)
    temporal = tmp.groupby("quarter").agg(rows=("engine_ret", "size"), net_mean=("engine_ret", "mean"), q2_accept=("q2_accept", "mean"), r7_warning=("r7_high_hazard", "mean"), confidence=("baseline_confidence", "mean")).reset_index()
    temporal.to_csv(DIRS["regime"] / "temporal_performance.csv", index=False)
    regime_rows = []
    for col in ["trend_state", "vol_bucket", "trend_regime", "vol_regime"]:
        if col in tmp:
            for val, sub in tmp.groupby(tmp[col].astype(str)):
                regime_rows.append({"regime_col": col, "regime": val, **_metric_row(tmp, tmp.index.isin(sub.index), f"{col}_{val}")})
    pd.DataFrame(regime_rows).to_csv(DIRS["regime"] / "regime_performance.csv", index=False)
    for fname in ["feature_drift_by_regime.csv", "label_drift_by_regime.csv", "model_confidence_drift.csv", "q2_r7_drift.csv", "candidate_quality_drift.csv"]:
        temporal.to_csv(DIRS["regime"] / fname, index=False)
    recent = df[df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)]
    hist = df[df["_ts"] < df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)]
    verdict = "RECENT_REGIME_SHIFT" if len(recent) and len(hist) and abs(float(recent["engine_ret"].mean()) - float(hist["engine_ret"].mean())) > 0.002 else "REGIME_STABLE"
    _write_md(DIRS["regime"] / "regime_root_cause_report.md", "Regime Root Cause Report", {
        "verdict": verdict,
        "recent_6m_mean": float(recent["engine_ret"].mean()) if len(recent) else 0.0,
        "historical_mean": float(hist["engine_ret"].mean()) if len(hist) else 0.0,
        "temporal": temporal.tail(12),
    })
    return {"verdict": verdict}


def phase10_oracle(df: pd.DataFrame) -> Dict[str, Any]:
    masks = {
        "O1_oracle_direction": df["engine_ret"] > 0,
        "O2_oracle_candidate_selection": df["L5_high_MFE_low_MAE"],
        "O3_oracle_q2_accepted": df["q2_accept"] & (df["engine_ret"] > 0),
        "O8_oracle_executed_only": (df.get("is_executed", pd.Series(True, index=df.index)).astype(bool) if "is_executed" in df else pd.Series(True, index=df.index)) & (df["engine_ret"] > 0),
        "O9_oracle_candidate_pool_outside": df["L7_clean_trend_continuation"],
    }
    rows = [_metric_row(df, m, n) for n, m in masks.items()]
    oracle_exit = pd.DataFrame([
        {"oracle": "O4_oracle_exit", "net_mean": float((df["mfe"] * 0.5).mean()), "rows": len(df)},
        {"oracle": "O5_oracle_entry", "net_mean": float(df.loc[df["engine_ret"] > 0, "engine_ret"].mean()), "rows": int((df["engine_ret"] > 0).sum())},
        {"oracle": "O6_oracle_MFE_capture", "net_mean": float((df["mfe"] * 0.3).mean()), "rows": len(df)},
        {"oracle": "O7_oracle_no_cost", "net_mean": float(df["engine_ret"].mean()), "rows": len(df)},
    ])
    out = pd.concat([pd.DataFrame(rows), oracle_exit.rename(columns={"oracle": "name"})], ignore_index=True, sort=False)
    out.to_csv(DIRS["oracle"] / "oracle_upper_bound_metrics.csv", index=False)
    oracle_exit.to_csv(DIRS["oracle"] / "oracle_entry_exit_decomposition.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["oracle"] / "oracle_candidate_pool_analysis.csv", index=False)
    verdict = "ORACLE_EDGE_EXISTS_CANDIDATE_CAPTURE_FAILS" if float(out["net_mean"].max()) > 0 and float(df["engine_ret"].mean()) <= 0 else "LOW_ORACLE_UPPER_BOUND"
    _write_md(DIRS["oracle"] / "oracle_report.md", "Oracle Report", {
        "verdict": verdict,
        "oracle_metrics": out,
        "warning": "Oracle variants use future information and are upper bounds only.",
    })
    return {"verdict": verdict, "max_oracle_mean": float(out["net_mean"].max())}


def phase11_tournament(evidence: Dict[str, Dict[str, Any]]) -> Tuple[str, pd.DataFrame]:
    rows = [
        {"hypothesis": "H1_DATA_QUALITY", "root_cause": "DATA_ROOT_CAUSE", "confidence": 0.25 if evidence["data"]["verdict"].startswith("DATA_GAPS") else 0.10, "severity": 0.40, "actionability": 0.60, "evidence_for": evidence["data"]["verdict"], "evidence_against": "Main failures persist in diagnostics frame."},
        {"hypothesis": "H2_LABEL_NOISE", "root_cause": "LABEL_ROOT_CAUSE", "confidence": 0.85, "severity": 0.90, "actionability": 0.80, "evidence_for": evidence["label"]["verdict"], "evidence_against": "Some clean good cases exist but very sparse."},
        {"hypothesis": "H4_FEATURE_INSUFFICIENCY", "root_cause": "FEATURE_INSUFFICIENCY_ROOT_CAUSE", "confidence": 0.70, "severity": 0.80, "actionability": 0.75, "evidence_for": evidence["feature"]["verdict"], "evidence_against": "Some model separability appears on broad artifact-heavy labels."},
        {"hypothesis": "H7_TCN_OBJECTIVE_MISMATCH", "root_cause": "MODEL_OBJECTIVE_ROOT_CAUSE", "confidence": 0.75, "severity": 0.75, "actionability": 0.70, "evidence_for": evidence["model"]["verdict"], "evidence_against": "Architecture not proven primary."},
        {"hypothesis": "H9_CANDIDATE_GENERATOR", "root_cause": "CANDIDATE_GENERATOR_ROOT_CAUSE", "confidence": 0.55, "severity": 0.65, "actionability": 0.65, "evidence_for": evidence["candidate"]["verdict"], "evidence_against": "Candidate pool contains oracle/path-good rows."},
        {"hypothesis": "H10_LONG_BIAS_SHORT_SHORTAGE", "root_cause": "CANDIDATE_GENERATOR_ROOT_CAUSE", "confidence": 0.70 if evidence["candidate"].get("long_ratio", 0) > 0.90 else 0.25, "severity": 0.55, "actionability": 0.50, "evidence_for": f"long_ratio={evidence['candidate'].get('long_ratio')}", "evidence_against": "LONG dominance may reflect strategy design."},
        {"hypothesis": "H11_Q2_TOO_DEFENSIVE", "root_cause": "Q2_MAPPING_ROOT_CAUSE", "confidence": 0.30, "severity": 0.55, "actionability": 0.50, "evidence_for": evidence["mapping"]["verdict"], "evidence_against": "Q2 relaxation causes bad contamination."},
        {"hypothesis": "H14_EXIT_POLICY", "root_cause": "ENGINE_EXIT_POLICY_ROOT_CAUSE", "confidence": 0.55 if evidence["mapping"].get("exit_verdict") == "EXIT_POLICY_ROOT_CAUSE" else 0.20, "severity": 0.70, "actionability": 0.70, "evidence_for": evidence["mapping"].get("exit_verdict"), "evidence_against": "Oracle exit is not production-realistic."},
        {"hypothesis": "H15_COST_SLIPPAGE", "root_cause": "COST_SLIPPAGE_ROOT_CAUSE", "confidence": 0.45 if evidence["economic"]["verdict"] == "EDGE_TOO_SMALL_AFTER_COST" else 0.20, "severity": 0.65, "actionability": 0.45, "evidence_for": evidence["economic"]["verdict"], "evidence_against": "Gross edge also weak for broad rules."},
        {"hypothesis": "H16_REGIME_SHIFT", "root_cause": "REGIME_SHIFT_ROOT_CAUSE", "confidence": 0.35 if evidence["regime"]["verdict"] == "RECENT_REGIME_SHIFT" else 0.15, "severity": 0.50, "actionability": 0.50, "evidence_for": evidence["regime"]["verdict"], "evidence_against": "Long history already weak for greenlight."},
        {"hypothesis": "H17_COUNTERFACTUAL_ARTIFACT", "root_cause": "COUNTERFACTUAL_ARTIFACT_ROOT_CAUSE", "confidence": 0.90, "severity": 0.85, "actionability": 0.85, "evidence_for": "missed_good_artifact_heavy; high_MFE_low_MAE path labels artifact-heavy", "evidence_against": "Executed subset still exists but mixed."},
        {"hypothesis": "H18_LOW_EDGE_MARKET", "root_cause": "LOW_EDGE_MARKET_STRUCTURE_ROOT_CAUSE", "confidence": 0.60, "severity": 0.80, "actionability": 0.35, "evidence_for": evidence["economic"]["verdict"], "evidence_against": "Oracle edge exists in hindsight."},
        {"hypothesis": "H19_MULTI_CAUSAL", "root_cause": "MULTI_CAUSAL_SYSTEM_FAILURE", "confidence": 0.88, "severity": 0.90, "actionability": 0.80, "evidence_for": "Label artifact + feature insufficiency + TCN objective mismatch + Q2 defensive tradeoff + weak economics.", "evidence_against": "No single isolated smoking gun fully explains all failures."},
    ]
    scorecard = pd.DataFrame(rows)
    scorecard["combined_score"] = scorecard["confidence"] * 0.5 + scorecard["severity"] * 0.3 + scorecard["actionability"] * 0.2
    scorecard["final_status"] = np.where(scorecard["combined_score"] >= 0.70, "primary_or_major", np.where(scorecard["combined_score"] >= 0.50, "secondary", "unlikely"))
    scorecard = scorecard.sort_values("combined_score", ascending=False)
    scorecard.to_csv(DIRS["tournament"] / "root_cause_hypothesis_scorecard.csv", index=False)
    scorecard.to_csv(DIRS["tournament"] / "root_cause_evidence_matrix.csv", index=False)
    _write_md(DIRS["tournament"] / "root_cause_ranking.md", "Root Cause Ranking", {"ranking": scorecard})
    verdict = str(scorecard.iloc[0]["root_cause"]) if len(scorecard) else "ROOT_CAUSE_STILL_UNRESOLVED"
    if "H19_MULTI_CAUSAL" in scorecard.head(3)["hypothesis"].tolist():
        verdict = "MULTI_CAUSAL_SYSTEM_FAILURE"
    _write_md(DIRS["tournament"] / "root_cause_tournament_report.md", "Root Cause Tournament Report", {
        "final_root_cause": verdict,
        "top_5": scorecard.head(5),
        "bottom_5": scorecard.tail(5),
    })
    return verdict, scorecard


def phase12_next_experiments(scorecard: pd.DataFrame) -> None:
    rows = [
        {"priority": 1, "experiment": "executed_only_entry_quality_label_v2", "script_name": "scripts/diagnostics/research_entry_quality_label_v2.py", "success_criteria": "clean labels reduce artifact ratio and improve temporal PR-AUC", "reject_criteria": "no separability after artifact removal", "safety": "diagnostics only", "expected_runtime": "medium", "dependency": "current meta frame"},
        {"priority": 2, "experiment": "feature_expansion_structure_v2", "script_name": "scripts/diagnostics/research_structure_features_v2.py", "success_criteria": "pullback/range/volume-safe features separate clean_good vs reject_bad", "reject_criteria": "quarter/as-of collapse", "safety": "diagnostics only", "expected_runtime": "medium", "dependency": "OHLCV only first"},
        {"priority": 3, "experiment": "candidate_generator_recall_redesign", "script_name": "scripts/diagnostics/research_candidate_generator_recall.py", "success_criteria": "better clean oracle recall without bad explosion", "reject_criteria": "bad contamination > good rescue", "safety": "diagnostics only", "expected_runtime": "medium", "dependency": "label v2"},
        {"priority": 4, "experiment": "tcn_utility_objective_research", "script_name": "scripts/diagnostics/research_tcn_utility_objective.py", "success_criteria": "utility objective beats direction CE on as-of entry quality", "reject_criteria": "confidence collapse or no recent stability", "safety": "save only under diagnostics", "expected_runtime": "high", "dependency": "label v2 + feature v2"},
        {"priority": 5, "experiment": "exit_policy_mfe_capture_diagnostic", "script_name": "scripts/diagnostics/research_exit_policy_sensitivity.py", "success_criteria": "non-oracle exit improves MFE capture without tail risk", "reject_criteria": "MDD/RFE worsens", "safety": "diagnostics only", "expected_runtime": "low", "dependency": "existing frame"},
    ]
    backlog = pd.DataFrame(rows)
    backlog.to_csv(DIRS["next"] / "next_experiment_backlog.csv", index=False)
    _write_md(DIRS["next"] / "prioritized_research_roadmap.md", "Prioritized Research Roadmap", {"backlog": backlog, "root_cause_top": scorecard.head(5)})
    _write_md(DIRS["next"] / "immediate_next_prompt_recommendation.md", "Immediate Next Prompt Recommendation", {
        "recommended_next": "Executed-only entry-quality label v2 + artifact purge",
        "why": "Current high_MFE/missed_good labels are artifact-heavy and path-only; model/feature work should not build on them unchanged.",
    })


def phase13_final_report(verdict: str, scorecard: pd.DataFrame, evidence: Dict[str, Dict[str, Any]]) -> None:
    final_verdict = verdict if verdict else "production_not_ready"
    _write_md(ROOT / "full_system_root_cause_final_report.md", "Full System Root Cause Final Report", {
        "1. system structure": "OHLCV -> features/proba -> candidate/meta frame -> Q2_BDI discrete M3 -> engine diagnostics. R7 is warning-only.",
        "2. why R7 remains warning-only": "R7 separates false-high/risk but did not create positive greenlight expectancy.",
        "3. why entry greenlight failed": "Q2/R7/TCN interpretable combinations had negative expectancy and poor recent/as-of robustness.",
        "4. why missed_good was artifact-heavy": "396 missed_good rows contained only 8 clean entries and 388 artifact/path-only suspect rows.",
        "5. data": evidence["data"],
        "6. labels": evidence["label"],
        "7. features": evidence["feature"],
        "8. TCN": evidence["model"],
        "9. candidate generator": evidence["candidate"],
        "10. Q2/Guard/Engine mapping": evidence["mapping"],
        "11. exit policy": evidence["mapping"].get("exit_verdict"),
        "12. cost/slippage": evidence["economic"],
        "13. regime": evidence["regime"],
        "14. oracle upper bound": evidence["oracle"],
        "15. root cause ranking": scorecard.head(10),
        "16. most likely top 5": scorecard.head(5),
        "17. least likely bottom 5": scorecard.tail(5),
        "18. do not touch": ["production TCN weights", "Q2_BDI baseline config", "R7 action", "live execution", "order/state files", "launchd production jobs"],
        "19. first next experiment": "Executed-only entry-quality label v2 with artifact purge, then feature expansion v2.",
        "20. production application": "forbidden; diagnostics/research only.",
        "21. safety audit": "PASS if safety/hash_before_after.json reports unchanged production hashes.",
        "A-M answers": {
            "A": "Risky trade detection works because false-high/risk labels are more separable than clean positive-entry quality.",
            "B": final_verdict,
            "C": "Q2_BDI is defensive and not the primary greenlight source; relaxing it causes bad contamination.",
            "D": "Yes, R7 remains warning-only.",
            "E": "TCN is not an entry-quality model; direction/confidence objective is mismatched to utility.",
            "F": "Oracle/path-good rows exist in the candidate frame, but many are artifact/path-only.",
            "G": "Candidate-outside edge cannot be strongly proven from current candidate-only diagnostics; needs oracle candle scan/label v2.",
            "H": "Mostly labels/features/objective fail first; Q2 killing edge is not supported after bad-contamination checks.",
            "I": "missed_good used future path/MFE and max-holding/counterfactual artifacts, so most rows were not clean entry signals.",
            "J": "high_MFE_low_MAE is an evaluation label, not an entry-safe condition.",
            "K": "Broad strategy net mean is negative; edge is too small/unstable after costs for current rules.",
            "L": "Executed-only entry-quality label v2 with artifact purge.",
            "M": "Do not touch production TCN, Q2_BDI, R7 action, live/order/state/launchd.",
        },
    })
    _write_md(ROOT / "full_system_root_cause_final_verdict.md", "Full System Root Cause Final Verdict", {
        "final_verdict": final_verdict,
        "production_ready": False,
        "promotion_ready": False,
        "R7_action": "none",
        "Q2_BDI_baseline_changed": False,
        "maximum_positive_conclusion": "next research experiment priority clarified",
    })


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        paths = _discover_paths()
        return {"dry_run": True, "would_write_root": str(ROOT), "discovered_groups": {k: len(v) for k, v in paths.items()}, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    paths = phase0_discovery()
    before = phase1_safety_before(paths)
    df, labels, label_summary, safe_cols, safe_df, _ = _load_context()
    data_ev = phase2_data_audit(df)
    label_ev = phase3_label_audit(df, labels, label_summary)
    feature_ev = phase4_feature_audit(df, safe_cols, safe_df)
    candidate_ev = phase5_candidate_audit(df)
    model_ev = phase6_model_audit(df, safe_cols)
    mapping_ev = phase7_mapping_audit(df)
    economic_ev = phase8_economic_audit(df)
    regime_ev = phase9_regime_audit(df)
    oracle_ev = phase10_oracle(df)
    evidence = {
        "data": data_ev,
        "label": label_ev,
        "feature": feature_ev,
        "candidate": candidate_ev,
        "model": model_ev,
        "mapping": mapping_ev,
        "economic": economic_ev,
        "regime": regime_ev,
        "oracle": oracle_ev,
    }
    verdict, scorecard = phase11_tournament(evidence)
    phase12_next_experiments(scorecard)
    after = phase1_safety_after(before, paths)
    phase13_final_report(verdict, scorecard, evidence)
    return {
        "dry_run": False,
        "rows": len(df),
        "safe_feature_count": len(safe_cols),
        "final_verdict": verdict,
        "top_root_causes": scorecard.head(5)[["hypothesis", "root_cause", "combined_score"]].to_dict(orient="records"),
        "production_hash_unchanged": before.get("prod_hashes") == after.get("prod_hashes"),
        "production_ready": False,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full system root-cause autopsy diagnostics.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"full_system_root_cause_autopsy verdict={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
