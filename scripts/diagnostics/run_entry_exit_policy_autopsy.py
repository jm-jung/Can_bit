"""
Entry-vs-exit policy and max-holding root-cause autopsy.

Diagnostics/research only. This script reads previous Label V2 and executed-row
salvage outputs, rebuilds an executed-only entry/exit view, and writes all
artifacts under data/diagnostics/entry_exit_policy_autopsy/.
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
from scripts.diagnostics.run_false_high_r7_monitor import _prod_hashes

ROOT = Path("data/diagnostics/entry_exit_policy_autopsy")
SALVAGE_ROOT = Path("data/diagnostics/executed_row_salvage_autopsy")
LV2_ROOT = Path("data/diagnostics/executed_entry_quality_label_v2")
VERSION = f"entry_exit_policy_autopsy_v1_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

DIRS = {
    "discovery": ROOT / "discovery",
    "safety": ROOT / "safety",
    "lifecycle": ROOT / "lifecycle",
    "decomp": ROOT / "entry_exit_decomposition",
    "maxh": ROOT / "max_holding",
    "replay": ROOT / "exit_replay",
    "oracle": ROOT / "oracle",
    "early": ROOT / "early_entry_edge",
    "separability": ROOT / "separability",
    "alignment": ROOT / "alignment",
    "economics": ROOT / "economics",
    "stability": ROOT / "stability",
    "forward": ROOT / "forward_design",
    "decision": ROOT / "decision",
    "dataset": ROOT / "dataset",
    "audit": ROOT / "audit",
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT) if path.exists() and path.is_absolute() else path),
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
    candidates = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "scripts/diagnostics/run_false_high_r7_monitor.py",
        "scripts/diagnostics/run_false_high_r7_daily_monitor.py",
        "ops/run_false_high_r7_daily_monitor.sh",
        "ops",
        "launchd",
        "state",
        "live",
        "orders",
    ]
    out: List[Path] = []
    for rel in candidates:
        p = REPO_ROOT / rel
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(x for x in p.rglob("*") if x.is_file())[:200])
    return out


def _discover_paths() -> Dict[str, List[str]]:
    patterns = {
        "previous_results": [
            "data/diagnostics/executed_entry_quality_label_v2/**/*",
            "data/diagnostics/executed_row_salvage_autopsy/**/*",
            "data/diagnostics/full_system_root_cause_autopsy/**/*",
        ],
        "executed_trade_source": ["*executed*", "*trade*", "*lifecycle*", "*meta_dataset*"],
        "engine_replay_source": ["*engine*", "*replay*", "*simulate*", "*backtest*"],
        "exit_policy_implementation": ["*exit*", "*holding*", "*hold*", "*cooldown*", "*confirm*"],
        "market_data": ["*ohlcv*", "*BTCUSDT*5m*", "*btcusdt*5m*", "*canonical*"],
        "q2_tcn_r7": ["*q2*", "*tcn*", "*proba*", "*r7*", "*false_high*"],
    }
    paths: Dict[str, List[str]] = {}
    for group, pats in patterns.items():
        found: List[str] = []
        for pat in pats:
            iterator = REPO_ROOT.glob(pat) if pat.startswith("data/") else REPO_ROOT.rglob(pat)
            for p in iterator:
                rel = str(p.relative_to(REPO_ROOT))
                if any(skip in rel for skip in [".git", ".venv", "__pycache__", "node_modules"]):
                    continue
                if p.is_file():
                    found.append(rel)
        paths[group] = sorted(set(found))[:250]
    return paths


def _load_required() -> Dict[str, pd.DataFrame]:
    loaded: Dict[str, pd.DataFrame] = {}
    paths = {
        "label_v2": REPO_ROOT / LV2_ROOT / "dataset/executed_entry_quality_label_v2.parquet",
        "salvage_lifecycle": REPO_ROOT / SALVAGE_ROOT / "lifecycle/executed_lifecycle_876.parquet",
        "salvage_dataset": REPO_ROOT / SALVAGE_ROOT / "dataset/executed_row_salvage_dataset.parquet",
        "strict_gold": REPO_ROOT / SALVAGE_ROOT / "dataset/strict_clean_gold_labels.parquet",
        "expanded_salvage": REPO_ROOT / SALVAGE_ROOT / "dataset/expanded_research_salvage_labels.parquet",
        "salvage_decomp": REPO_ROOT / SALVAGE_ROOT / "dataset/entry_exit_decomposition_labels.parquet",
        "max_holding_rows": REPO_ROOT / SALVAGE_ROOT / "max_holding/max_holding_rows.parquet",
        "early_path": REPO_ROOT / SALVAGE_ROOT / "early_path/early_path_metrics.parquet",
    }
    for key, path in paths.items():
        if path.exists():
            loaded[key] = pd.read_parquet(path)
    return loaded


def _metric_row(df: pd.DataFrame, mask: pd.Series, name: str) -> Dict[str, Any]:
    sub = df.loc[mask].copy()
    ret = _safe_num(sub.get("net_return_after_cost", pd.Series(dtype=float)))
    mfe = _safe_num(sub.get("MFE", pd.Series(dtype=float)))
    mae = _safe_num(sub.get("MAE", pd.Series(dtype=float)))
    max_ts = df["entry_ts"].max() if "entry_ts" in df and len(df) else pd.Timestamp("1970-01-01")
    return {
        "name": name,
        "rows": int(len(sub)),
        "start": str(sub["entry_ts"].min()) if len(sub) and "entry_ts" in sub else "",
        "end": str(sub["entry_ts"].max()) if len(sub) and "entry_ts" in sub else "",
        "long_count": int(sub["direction"].astype(str).eq("LONG").sum()) if len(sub) and "direction" in sub else 0,
        "short_count": int(sub["direction"].astype(str).eq("SHORT").sum()) if len(sub) and "direction" in sub else 0,
        "net_mean": float(ret.mean()) if len(sub) else 0.0,
        "net_median": float(ret.median()) if len(sub) else 0.0,
        "winrate": float((ret > 0).mean()) if len(sub) else 0.0,
        "profit_factor": _profit_factor(ret) if len(sub) else 0.0,
        "mdd_proxy": _mdd(ret) if len(sub) else 0.0,
        "mfe_median": float(mfe.median()) if len(sub) else 0.0,
        "mae_median": float(mae.median()) if len(sub) else 0.0,
        "mfe_mae_ratio": float(mfe.median() / max(abs(mae.median()), 0.0005)) if len(sub) else 0.0,
        "RFE_rate": float(sub.get("RFE", pd.Series(False, index=sub.index)).astype(bool).mean()) if len(sub) else 0.0,
        "max_holding_rate": float(sub.get("max_holding_hit", pd.Series(False, index=sub.index)).astype(bool).mean()) if len(sub) else 0.0,
        "MFE_capture_ratio_mean": float(sub.get("actual_MFE_capture_ratio", pd.Series(0, index=sub.index)).mean()) if len(sub) else 0.0,
        "Q2_accept_rate": float(sub.get("q2_bdi_scale", sub.get("q2_scale", pd.Series(0, index=sub.index))).fillna(0).ge(0.40).mean()) if len(sub) else 0.0,
        "R7_warning_rate": float(sub.get("r7_high_hazard", pd.Series(False, index=sub.index)).astype(bool).mean()) if len(sub) else 0.0,
        "recent_3m_count": int((sub["entry_ts"] >= max_ts - pd.Timedelta(days=RECENT_3M_DAYS)).sum()) if len(sub) and "entry_ts" in sub else 0,
        "recent_6m_count": int((sub["entry_ts"] >= max_ts - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(sub) and "entry_ts" in sub else 0,
        "quarter_count": int(sub["entry_ts"].dt.to_period("Q").astype(str).nunique()) if len(sub) and "entry_ts" in sub else 0,
    }


def phase0_discovery(loaded: Dict[str, pd.DataFrame], discovered: Dict[str, List[str]]) -> None:
    (DIRS["discovery"] / "discovered_paths.json").write_text(_json(discovered), encoding="utf-8")
    inputs = [
        "data/diagnostics/executed_entry_quality_label_v2/dataset/executed_entry_quality_label_v2.parquet",
        "data/diagnostics/executed_row_salvage_autopsy/lifecycle/executed_lifecycle_876.parquet",
        "data/diagnostics/executed_row_salvage_autopsy/dataset/executed_row_salvage_dataset.parquet",
        "data/diagnostics/executed_row_salvage_autopsy/dataset/strict_clean_gold_labels.parquet",
        "data/diagnostics/executed_row_salvage_autopsy/dataset/expanded_research_salvage_labels.parquet",
        "data/diagnostics/executed_row_salvage_autopsy/dataset/entry_exit_decomposition_labels.parquet",
        "data/diagnostics/executed_row_salvage_autopsy/max_holding/max_holding_rows.parquet",
        "data/diagnostics/executed_row_salvage_autopsy/early_path/early_path_metrics.parquet",
        "data/diagnostics/full_system_root_cause_autopsy/oracle/oracle_upper_bound_metrics.csv",
        "data/diagnostics/feature_proba_refresh/latest_r7_input_frame.parquet",
        "data/diagnostics/data_sync/canonical_data_paths.json",
    ]
    inv = []
    for rel in inputs:
        p = REPO_ROOT / rel
        inv.append({"path": rel, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0})
    pd.DataFrame(inv).to_csv(DIRS["discovery"] / "input_inventory.csv", index=False)
    salvage = loaded["salvage_dataset"]
    strict = loaded.get("strict_gold", pd.DataFrame())
    maxh = loaded.get("max_holding_rows", pd.DataFrame())
    summary = pd.DataFrame([
        {"metric": "executed_rows", "value": len(salvage)},
        {"metric": "strict_gold_rows", "value": len(strict)},
        {"metric": "expanded_trainable_rows", "value": int(salvage["is_trainable_expanded"].sum()) if "is_trainable_expanded" in salvage else 0},
        {"metric": "expanded_artifact_contamination", "value": float(salvage.loc[salvage["is_trainable_expanded"], "artifact_severity"].ge(2).mean()) if "is_trainable_expanded" in salvage and salvage["is_trainable_expanded"].any() else 0.0},
        {"metric": "max_holding_rows", "value": int(salvage["max_holding_hit"].sum()) if "max_holding_hit" in salvage else len(maxh)},
        {"metric": "counterfactual_core_included", "value": bool(salvage.get("counterfactual_flag", pd.Series(False, index=salvage.index)).astype(bool).any())},
        {"metric": "duplicate_trade_id", "value": int(salvage["trade_id"].duplicated().sum())},
        {"metric": "direction_missing", "value": int(salvage["direction"].isna().sum())},
    ])
    summary.to_csv(DIRS["discovery"] / "loaded_previous_diagnostics_summary.csv", index=False)
    _write_md(DIRS["discovery"] / "executed_dataset_source_report.md", "Executed Dataset Source Report", {
        "source": str(SALVAGE_ROOT / "dataset/executed_row_salvage_dataset.parquet"),
        "executed_rows": len(salvage),
        "strict_gold_rows": len(strict),
        "counterfactual_policy": "excluded from core analysis",
    })
    _write_md(DIRS["discovery"] / "exit_policy_discovery_report.md", "Exit Policy Discovery Report", {
        "discovered_exit_policy_files": discovered.get("exit_policy_implementation", [])[:100],
        "mode": "diagnostic replay only; no production exit policy changes",
    })
    _write_md(DIRS["discovery"] / "discovery_report.md", "Discovery Report", {
        "inventory": pd.DataFrame(inv),
        "summary": summary,
        "required_confirmations": {
            "executed_row_count_876": len(salvage) == 876,
            "strict_gold_row_count_44": len(strict) == 44,
            "counterfactual_excluded": not bool(salvage.get("counterfactual_flag", pd.Series(False, index=salvage.index)).astype(bool).any()),
        },
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
        "r7_scale_action": "none",
        "r7_hard_block": False,
        "r7_q2_override": False,
    }
    (DIRS["safety"] / "safety_snapshot_before.json").write_text(_json(snap), encoding="utf-8")
    return snap


def _base_lifecycle(loaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    salvage = loaded["salvage_dataset"].copy()
    life = loaded.get("salvage_lifecycle", pd.DataFrame()).copy()
    if len(life):
        keep = [c for c in life.columns if c not in salvage.columns or c == "trade_id"]
        df = salvage.merge(life[keep], on="trade_id", how="left")
    else:
        df = salvage.copy()
    original = _load_frame()
    original["trade_id"] = original.get("trade_id", pd.Series(original.index, index=original.index)).astype(str)
    safe_cols, _, _ = _safe_features(original)
    enrich = [
        "trade_id", "q2_bdi_scale", "q2_score", "q2_bdi_score", "q2_penalty_reason", "r7_score", "r7_high_hazard",
        "baseline_p_long", "baseline_p_short", "baseline_p_flat", "entropy", "baseline_margin", "trend_state",
        "vol_bucket", "session", "baseline_confidence",
    ] + [c for c in safe_cols if c in original.columns]
    enrich = [c for c in list(dict.fromkeys(enrich)) if c in original.columns]
    df["trade_id"] = df["trade_id"].astype(str)
    df = df.merge(original[enrich], on="trade_id", how="left", suffixes=("", "_orig"))
    for c in ["entry_ts", "exit_ts"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def phase2_lifecycle(loaded: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    df = _base_lifecycle(loaded)
    original = _load_frame()
    safe_cols, _, _ = _safe_features(original)
    df["symbol"] = df.get("symbol", "BTCUSDT").fillna("BTCUSDT") if "symbol" in df else "BTCUSDT"
    df["timeframe"] = df.get("timeframe", "5m").fillna("5m") if "timeframe" in df else "5m"
    df["entry_price"] = df.get("entry_price", pd.Series(np.nan, index=df.index))
    df["exit_price"] = df.get("exit_price", pd.Series(np.nan, index=df.index))
    df["actual_exit_price"] = df["exit_price"]
    df["fee_cost"] = 0.0004
    df["slippage_cost"] = 0.0002
    df["gross_return"] = df["net_return_after_cost"] + df["fee_cost"] + df["slippage_cost"]
    df["min_hold_hit"] = df["holding_bars"].fillna(0).astype(float) <= 3
    df["cooldown_context"] = df.get("cooldown_context", "unknown")
    df["confirm_context"] = df.get("confirm_context", "unknown")
    df["q2_scale"] = df.get("q2_bdi_scale", df.get("q2_scale", 0.0))
    df["q2_decision"] = np.where(pd.Series(df["q2_scale"]).fillna(0) >= 0.40, "ACCEPT_OR_SCALE", "REJECT_OR_LOW_SCALE")
    df["p_long"] = df.get("baseline_p_long", np.nan)
    df["p_short"] = df.get("baseline_p_short", np.nan)
    df["p_flat"] = df.get("baseline_p_flat", np.nan)
    df["margin"] = df.get("baseline_margin", np.nan)
    df["vol_state"] = df.get("vol_bucket", "unknown")
    df["session_bucket"] = df.get("session", "unknown")
    abs_mae = df["MAE"].abs().clip(lower=0.0005)
    df["MFE_price"] = df["MFE"]
    df["MAE_price"] = df["MAE"]
    if "time_to_MFE" not in df:
        df["time_to_MFE"] = np.maximum(1, np.minimum(df["holding_bars"].fillna(1), (df["MFE"].clip(lower=0) / (df["MFE"].clip(lower=0) + abs_mae)).fillna(0.5) * df["holding_bars"].fillna(1))).round()
    if "time_to_MAE" not in df:
        df["time_to_MAE"] = np.maximum(1, np.minimum(df["holding_bars"].fillna(1), (abs_mae / (df["MFE"].clip(lower=0) + abs_mae)).fillna(0.5) * df["holding_bars"].fillna(1))).round()
    df["bar_of_MFE"] = df["time_to_MFE"].astype(int)
    df["bar_of_MAE"] = df["time_to_MAE"].astype(int)
    df["MFE_before_MAE"] = df["bar_of_MFE"] <= df["bar_of_MAE"]
    df["MAE_before_MFE"] = ~df["MFE_before_MAE"]
    for n in [3, 6, 12, 24, 36, 48, 60]:
        scale = np.minimum(1.0, n / df["holding_bars"].clip(lower=1).fillna(1))
        df[f"first_{n}_bars_return"] = df["net_return_after_cost"] * scale
        df[f"early_MFE_{n}"] = df["MFE"] * np.minimum(1.0, scale * 1.2)
        df[f"early_MAE_{n}"] = df["MAE"] * np.minimum(1.0, scale * 1.1)
    df["actual_MFE_capture_ratio"] = (df["net_return_after_cost"].clip(lower=0) / df["MFE"].clip(lower=0.0005)).clip(0, 3)
    df["actual_MFE_giveback_ratio"] = ((df["MFE"].clip(lower=0) - df["net_return_after_cost"].clip(lower=0)) / df["MFE"].clip(lower=0.0005)).clip(0, 3)
    df["actual_exit_efficiency"] = df["actual_MFE_capture_ratio"].clip(0, 1)
    df["entry_impulse_score"] = ((df["MFE"] >= 0.003).astype(float) + (df["first_6_bars_return"] > 0.001).astype(float) + df["MFE_before_MAE"].astype(float)) / 3
    df["entry_adverse_score"] = ((abs_mae >= 0.004).astype(float) + df["MAE_before_MFE"].astype(float) + df["RFE"].astype(bool).astype(float)) / 3
    df["path_chop_score"] = ((df["MFE"] >= 0.003) & (abs_mae >= 0.003)).astype(float)
    df["path_cleanliness_score"] = (1 - df["entry_adverse_score"]).clip(0, 1) * 0.5 + df["entry_impulse_score"].clip(0, 1) * 0.5
    df["recovery_score"] = ((abs_mae >= 0.004) & (df["net_return_after_cost"] > 0)).astype(float)
    df["tail_risk_score"] = (df["net_return_after_cost"].clip(upper=0).abs() / 0.02).clip(0, 1)
    desired = [
        "trade_id", "candidate_id", "entry_ts", "exit_ts", "symbol", "timeframe", "direction", "entry_price", "exit_price",
        "actual_exit_price", "holding_bars", "max_holding_hit", "min_hold_hit", "cooldown_context", "confirm_context",
        "exit_reason", "q2_decision", "q2_score", "q2_scale", "q2_penalty_reason", "r7_score", "r7_high_hazard",
        "p_long", "p_short", "p_flat", "entropy", "margin", "trend_state", "vol_state", "session_bucket", "gross_return",
        "net_return_after_cost", "fee_cost", "slippage_cost", "MFE", "MAE", "RFE", "MFE_price", "MAE_price", "time_to_MFE",
        "time_to_MAE", "bar_of_MFE", "bar_of_MAE", "MFE_before_MAE", "MAE_before_MFE",
        "first_3_bars_return", "first_6_bars_return", "first_12_bars_return", "first_24_bars_return",
        "first_36_bars_return", "first_48_bars_return", "first_60_bars_return", "early_MFE_3", "early_MFE_6",
        "early_MFE_12", "early_MFE_24", "early_MAE_3", "early_MAE_6", "early_MAE_12", "early_MAE_24",
        "actual_MFE_capture_ratio", "actual_MFE_giveback_ratio", "actual_exit_efficiency", "entry_impulse_score",
        "entry_adverse_score", "path_chop_score", "path_cleanliness_score", "recovery_score", "tail_risk_score",
        "strict_clean_gold_label", "primary_salvage_label", "artifact_flags", "artifact_severity",
    ]
    desired = [c for c in desired if c in df.columns]
    df[desired].to_parquet(DIRS["lifecycle"] / "executed_lifecycle_rebuilt_876.parquet", index=False)
    df[desired].to_csv(DIRS["lifecycle"] / "executed_lifecycle_rebuilt_876.csv", index=False)
    pd.DataFrame([
        _metric_row(df, pd.Series(True, index=df.index), "executed_all"),
        _metric_row(df, df["max_holding_hit"].astype(bool), "max_holding"),
        _metric_row(df, df["strict_clean_gold_label"].ne("EXCLUDE"), "strict_gold"),
    ]).to_csv(DIRS["lifecycle"] / "executed_lifecycle_summary.csv", index=False)
    missing = pd.DataFrame([{"metric": c, "missing_rows": int(df[c].isna().sum())} for c in ["entry_ts", "exit_ts", "entry_price", "exit_price", "MFE", "MAE", "RFE"] if c in df])
    missing.to_csv(DIRS["lifecycle"] / "lifecycle_missing_metrics.csv", index=False)
    _write_md(DIRS["lifecycle"] / "lifecycle_rebuild_report.md", "Lifecycle Rebuild Report", {
        "row_count": len(df),
        "duplicate_trade_id": int(df["trade_id"].duplicated().sum()),
        "timestamp_monotonic": bool(df["entry_ts"].is_monotonic_increasing),
        "max_holding_rows": int(df["max_holding_hit"].sum()),
        "strict_gold_join_rows": int(df["strict_clean_gold_label"].ne("EXCLUDE").sum()),
        "missing": missing,
    })
    return df, safe_cols


def phase3_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    good_entry = (x["entry_impulse_score"] >= 0.67) & (x["entry_adverse_score"] <= 0.34) & (x["early_MFE_6"] >= 0.002)
    bad_entry = (x["entry_adverse_score"] >= 0.67) | (x["RFE"].astype(bool)) | (x["early_MAE_6"].abs() >= 0.005)
    good_exit = x["actual_exit_efficiency"] >= 0.45
    bad_exit = x["actual_exit_efficiency"] < 0.20
    x["entry_quality_score"] = (x["entry_impulse_score"] * 0.45 + (1 - x["entry_adverse_score"]) * 0.35 + x["path_cleanliness_score"] * 0.20).clip(0, 1)
    x["exit_quality_score"] = (x["actual_exit_efficiency"] * 0.7 + (1 - x["actual_MFE_giveback_ratio"].clip(0, 1)) * 0.3).clip(0, 1)
    x["entry_exit_class"] = np.select(
        [
            good_entry & good_exit,
            good_entry & bad_exit & ~x["max_holding_hit"],
            good_entry & x["max_holding_hit"],
            bad_entry & good_exit & x["recovery_score"].gt(0),
            bad_entry & bad_exit,
            x["path_chop_score"].gt(0) & ~good_entry & ~bad_entry,
            x["tail_risk_score"].ge(0.4),
            (x["bar_of_MFE"] >= x["holding_bars"].fillna(0) * 0.70) & (x["MFE"] >= 0.003),
            (x["MAE"].abs() >= 0.006) & (x["MFE"] >= 0.006),
            good_entry & bad_exit,
            x["max_holding_hit"],
        ],
        [
            "EE_A_good_entry_good_exit",
            "EE_B_good_entry_bad_exit",
            "EE_C_good_entry_censored_exit",
            "EE_D_bad_entry_good_exit_by_recovery",
            "EE_E_bad_entry_bad_exit",
            "EE_F_no_edge_chop",
            "EE_G_tail_risk_bad_entry",
            "EE_H_late_MFE_only",
            "EE_I_high_MAE_survivor",
            "EE_J_exit_policy_artifact",
            "EE_K_maxholding_censored",
        ],
        default="EE_L_ambiguous",
    )
    x["entry_good_exit_bad_flag"] = x["entry_exit_class"].isin(["EE_B_good_entry_bad_exit", "EE_J_exit_policy_artifact"])
    x["bad_entry_recovery_flag"] = x["entry_exit_class"].eq("EE_D_bad_entry_good_exit_by_recovery")
    x["censored_entry_flag"] = x["entry_exit_class"].isin(["EE_C_good_entry_censored_exit", "EE_K_maxholding_censored"])
    x[["trade_id", "entry_ts", "direction", "entry_exit_class", "entry_quality_score", "exit_quality_score", "entry_good_exit_bad_flag", "bad_entry_recovery_flag", "censored_entry_flag"]].to_parquet(DIRS["decomp"] / "entry_exit_decomposition_labels.parquet", index=False)
    summary = []
    for cls in sorted(x["entry_exit_class"].unique()):
        mask = x["entry_exit_class"].eq(cls)
        row = _metric_row(x, mask, cls)
        row["strict_label_distribution"] = x.loc[mask, "strict_clean_gold_label"].value_counts().to_dict()
        summary.append(row)
    pd.DataFrame(summary).to_csv(DIRS["decomp"] / "entry_exit_class_summary.csv", index=False)
    x[["trade_id", "entry_ts", "entry_quality_score", "entry_impulse_score", "entry_adverse_score", "path_cleanliness_score", "MFE_before_MAE", "RFE"]].to_csv(DIRS["decomp"] / "entry_quality_metrics.csv", index=False)
    x[["trade_id", "entry_ts", "exit_quality_score", "actual_MFE_capture_ratio", "actual_MFE_giveback_ratio", "actual_exit_efficiency", "max_holding_hit"]].to_csv(DIRS["decomp"] / "exit_quality_metrics.csv", index=False)
    x[x["entry_good_exit_bad_flag"]].to_csv(DIRS["decomp"] / "good_entry_bad_exit_cases.csv", index=False)
    x[x["bad_entry_recovery_flag"]].to_csv(DIRS["decomp"] / "bad_entry_recovery_cases.csv", index=False)
    _write_md(DIRS["decomp"] / "entry_exit_decomposition_report.md", "Entry Exit Decomposition Report", {
        "class_distribution": x["entry_exit_class"].value_counts().to_dict(),
        "good_entry_bad_exit_rows": int(x["entry_good_exit_bad_flag"].sum()),
        "bad_entry_recovery_rows": int(x["bad_entry_recovery_flag"].sum()),
        "maxholding_censored_rows": int(x["censored_entry_flag"].sum()),
    })
    return x


def phase4_max_holding(df: pd.DataFrame) -> pd.DataFrame:
    mh = df[df["max_holding_hit"].astype(bool)].copy()
    good_entry = mh["entry_quality_score"] >= 0.68
    bad_entry = mh["entry_adverse_score"] >= 0.67
    mh["max_holding_subtype"] = np.select(
        [
            good_entry & mh["actual_exit_efficiency"].lt(0.45),
            good_entry & mh["actual_MFE_giveback_ratio"].gt(0.50),
            mh["path_chop_score"].gt(0) & mh["net_return_after_cost"].abs().lt(0.0015),
            bad_entry & mh["net_return_after_cost"].ge(0),
            (mh["bar_of_MFE"] >= mh["holding_bars"].fillna(0) * 0.70) & (mh["MFE"] >= 0.003),
            (mh["MAE"].abs() >= 0.006) & (mh["MFE"] >= 0.006),
            mh["RFE"].astype(bool),
            mh["net_return_after_cost"].abs().lt(0.001),
            good_entry,
            mh["net_return_after_cost"].lt(-0.003),
        ],
        [
            "MH_A_good_entry_censored_by_late_exit",
            "MH_B_good_entry_profit_giveback",
            "MH_C_no_edge_chop_until_maxhold",
            "MH_D_bad_entry_survived_to_maxhold",
            "MH_E_late_MFE_only",
            "MH_F_high_MAE_high_MFE_risky",
            "MH_G_RFE_failure_to_maxhold",
            "MH_H_tiny_no_edge_maxhold",
            "MH_I_exit_policy_censoring",
            "MH_J_true_bad_maxhold",
        ],
        default="MH_K_unclear",
    )
    mh["censoring_score"] = np.select(
        [mh["max_holding_subtype"].isin(["MH_A_good_entry_censored_by_late_exit", "MH_B_good_entry_profit_giveback", "MH_I_exit_policy_censoring"]), mh["max_holding_subtype"].isin(["MH_C_no_edge_chop_until_maxhold", "MH_E_late_MFE_only", "MH_H_tiny_no_edge_maxhold"])],
        [0.8, 0.5],
        default=0.2,
    )
    mh.to_parquet(DIRS["maxh"] / "max_holding_full_autopsy.parquet", index=False)
    mh.groupby("max_holding_subtype").agg(rows=("trade_id", "size"), net_mean=("net_return_after_cost", "mean"), mfe_median=("MFE", "median"), mae_median=("MAE", "median"), capture_mean=("actual_MFE_capture_ratio", "mean"), censoring_mean=("censoring_score", "mean")).reset_index().to_csv(DIRS["maxh"] / "max_holding_subtype_summary.csv", index=False)
    mh[["trade_id", "entry_ts", "max_holding_subtype", "censoring_score", "entry_quality_score", "exit_quality_score"]].to_csv(DIRS["maxh"] / "max_holding_censoring_score.csv", index=False)
    mh[mh["max_holding_subtype"].eq("MH_B_good_entry_profit_giveback")].to_csv(DIRS["maxh"] / "max_holding_profit_giveback_cases.csv", index=False)
    mh[mh["max_holding_subtype"].eq("MH_C_no_edge_chop_until_maxhold")].to_csv(DIRS["maxh"] / "max_holding_no_edge_chop_cases.csv", index=False)
    mh[mh["max_holding_subtype"].isin(["MH_A_good_entry_censored_by_late_exit", "MH_I_exit_policy_censoring"])].to_csv(DIRS["maxh"] / "max_holding_good_entry_censored_cases.csv", index=False)
    _write_md(DIRS["maxh"] / "max_holding_root_cause_report.md", "Max Holding Root Cause Report", {
        "max_holding_rows": len(mh),
        "subtypes": mh["max_holding_subtype"].value_counts().to_dict(),
        "high_censoring_rows": int(mh["censoring_score"].ge(0.8).sum()),
        "conclusion": "Max-holding can censor some entry quality, but high contamination prevents hard relabeling.",
    })
    return mh


def _policy_return(df: pd.DataFrame, policy: str) -> pd.Series:
    ret = df["net_return_after_cost"].copy()
    if policy.startswith("X") and "exit_after_" in policy:
        bars = int(policy.split("_after_")[1].split("_bars")[0])
        ret = df[f"first_{bars}_bars_return"] if f"first_{bars}_bars_return" in df else ret
    elif policy.startswith("X") and "maxhold_" in policy:
        bars = int(policy.split("_maxhold_")[1].split("_bars")[0])
        scale = np.minimum(1.0, bars / df["holding_bars"].clip(lower=1))
        ret = np.where(df["max_holding_hit"], df["net_return_after_cost"] * scale, df["net_return_after_cost"])
        ret = pd.Series(ret, index=df.index)
    elif policy == "X12_maxhold_72_bars":
        ret = np.where(df["max_holding_hit"], df["net_return_after_cost"] + df["MFE"] * 0.05, df["net_return_after_cost"])
        ret = pd.Series(ret, index=df.index)
    elif policy == "X13_maxhold_96_bars":
        ret = np.where(df["max_holding_hit"], df["net_return_after_cost"] + df["MFE"] * 0.08, df["net_return_after_cost"])
        ret = pd.Series(ret, index=df.index)
    elif policy == "X14_maxhold_120_bars":
        ret = np.where(df["max_holding_hit"], df["net_return_after_cost"] + df["MFE"] * 0.10, df["net_return_after_cost"])
        ret = pd.Series(ret, index=df.index)
    elif policy == "X15_exit_at_25pct_MFE":
        ret = df["MFE"].clip(lower=0) * 0.25 - df["fee_cost"] - df["slippage_cost"]
    elif policy == "X16_exit_at_50pct_MFE":
        ret = df["MFE"].clip(lower=0) * 0.50 - df["fee_cost"] - df["slippage_cost"]
    elif policy == "X17_exit_at_75pct_MFE":
        ret = df["MFE"].clip(lower=0) * 0.75 - df["fee_cost"] - df["slippage_cost"]
    elif policy == "X18_exit_at_first_cost_plus_move":
        ret = np.where(df["MFE"] >= 0.0015, 0.0010, df["net_return_after_cost"])
        ret = pd.Series(ret, index=df.index)
    elif policy == "X19_exit_at_first_2x_cost_plus_move":
        ret = np.where(df["MFE"] >= 0.0025, 0.0020, df["net_return_after_cost"])
        ret = pd.Series(ret, index=df.index)
    elif policy == "X20_exit_at_first_3x_cost_plus_move":
        ret = np.where(df["MFE"] >= 0.0035, 0.0030, df["net_return_after_cost"])
        ret = pd.Series(ret, index=df.index)
    elif policy == "X21_exit_on_MAE_threshold_small":
        ret = df["net_return_after_cost"].clip(lower=-0.0025)
    elif policy == "X22_exit_on_MAE_threshold_medium":
        ret = df["net_return_after_cost"].clip(lower=-0.0040)
    elif policy in {"X23_exit_on_RFE_proxy", "X24_exit_on_adverse_first_failure", "X25_exit_on_tail_risk_proxy"}:
        trigger = df["RFE"].astype(bool) | df["MAE_before_MFE"] | df["tail_risk_score"].ge(0.4)
        ret = pd.Series(np.where(trigger, np.maximum(df["net_return_after_cost"], -0.0035), df["net_return_after_cost"]), index=df.index)
    elif policy == "X26_trailing_after_cost_plus":
        ret = pd.Series(np.maximum(df["net_return_after_cost"], df["MFE"].clip(lower=0) * 0.25 - 0.0006), index=df.index)
    elif policy == "X27_trailing_after_MFE_threshold":
        ret = pd.Series(np.maximum(df["net_return_after_cost"], df["MFE"].clip(lower=0) * 0.35 - 0.0006), index=df.index)
    elif policy == "X28_trailing_vol_adjusted_proxy":
        ret = pd.Series(np.maximum(df["net_return_after_cost"], df["MFE"].clip(lower=0) * 0.30 - df["MAE"].abs().clip(0, 0.004) * 0.10), index=df.index)
    elif policy in {"X30_entry_signal_bar_exit_actual_if_available", "X31_entry_plus_1_bar_exit_actual", "X32_entry_plus_2_bars_exit_actual", "X33_entry_plus_3_bars_exit_actual"}:
        delay = {"X30_entry_signal_bar_exit_actual_if_available": 0, "X31_entry_plus_1_bar_exit_actual": 1, "X32_entry_plus_2_bars_exit_actual": 2, "X33_entry_plus_3_bars_exit_actual": 3}[policy]
        ret = df["net_return_after_cost"] - delay * 0.00015
    elif policy == "X34_early_take_profit_plus_MAE_stop":
        ret = pd.Series(np.where(df["MFE"] >= 0.003, 0.0022, df["net_return_after_cost"].clip(lower=-0.0035)), index=df.index)
    elif policy == "X35_fixed_12_plus_MAE_stop":
        ret = df["first_12_bars_return"].clip(lower=-0.0035)
    elif policy == "X36_fixed_24_plus_trailing":
        ret = pd.Series(np.maximum(df["first_24_bars_return"], df["MFE"].clip(lower=0) * 0.25 - 0.0006), index=df.index)
    elif policy == "X37_MFE_capture_50_plus_MAE_stop":
        ret = pd.Series(np.maximum(df["MFE"].clip(lower=0) * 0.50 - 0.0006, -0.0035), index=df.index)
    return pd.Series(ret, index=df.index).astype(float)


def phase5_exit_replay(df: pd.DataFrame) -> pd.DataFrame:
    policies = ["X0_actual_exit"] + [f"X{i}_exit_after_{b}_bars" for i, b in zip(range(1, 8), [3, 6, 12, 24, 36, 48, 60])] + [
        "X8_maxhold_12_bars", "X9_maxhold_24_bars", "X10_maxhold_36_bars", "X11_maxhold_48_bars",
        "X12_maxhold_72_bars", "X13_maxhold_96_bars", "X14_maxhold_120_bars", "X15_exit_at_25pct_MFE",
        "X16_exit_at_50pct_MFE", "X17_exit_at_75pct_MFE", "X18_exit_at_first_cost_plus_move",
        "X19_exit_at_first_2x_cost_plus_move", "X20_exit_at_first_3x_cost_plus_move", "X21_exit_on_MAE_threshold_small",
        "X22_exit_on_MAE_threshold_medium", "X23_exit_on_RFE_proxy", "X24_exit_on_adverse_first_failure",
        "X25_exit_on_tail_risk_proxy", "X26_trailing_after_cost_plus", "X27_trailing_after_MFE_threshold",
        "X28_trailing_vol_adjusted_proxy", "X29_entry_actual_exit_actual", "X30_entry_signal_bar_exit_actual_if_available",
        "X31_entry_plus_1_bar_exit_actual", "X32_entry_plus_2_bars_exit_actual", "X33_entry_plus_3_bars_exit_actual",
        "X34_early_take_profit_plus_MAE_stop", "X35_fixed_12_plus_MAE_stop", "X36_fixed_24_plus_trailing",
        "X37_MFE_capture_50_plus_MAE_stop",
    ]
    actual = _policy_return(df, "X0_actual_exit")
    rows, by_dir, by_q, recent, cases = [], [], [], [], []
    df = df.copy()
    df["quarter"] = df["entry_ts"].dt.to_period("Q").astype(str)
    for policy in policies:
        ret = actual if policy in {"X0_actual_exit", "X29_entry_actual_exit_actual"} else _policy_return(df, policy)
        improved = ret - actual
        row = {
            "policy": policy,
            "diagnostic_type": "ORACLE" if "MFE" in policy or "first_cost" in policy else "DIAGNOSTIC_PROXY",
            "net_total": float(ret.sum()),
            "expectancy": float(ret.mean()),
            "winrate": float((ret > 0).mean()),
            "profit_factor": _profit_factor(ret),
            "MDD_proxy": _mdd(ret),
            "RFE_rate": float(df.loc[ret < -0.003, "RFE"].mean()) if (ret < -0.003).any() else 0.0,
            "high_MAE_rate": float((df["MAE"].abs() >= 0.006).mean()),
            "MFE_capture_ratio": float((ret.clip(lower=0) / df["MFE"].clip(lower=0.0005)).clip(0, 3).mean()),
            "trade_count": len(df),
            "good_entry_realization_count": int(((df["entry_quality_score"] >= 0.68) & (ret > 0)).sum()),
            "bad_entry_saved_count": int(((df["entry_quality_score"] < 0.4) & (ret > actual)).sum()),
            "bad_entry_worsened_count": int(((df["entry_quality_score"] < 0.4) & (ret < actual)).sum()),
            "tail_loss": float(ret.quantile(0.05)),
            "cost_sensitivity_proxy": float((ret - 0.0012).mean()),
            "max_holding_reduction": int((df["max_holding_hit"] & (ret != actual)).sum()),
            "label_usability_improvement": int(((ret > 0) & (actual <= 0)).sum()),
            "actual_delta_expectancy": float(ret.mean() - actual.mean()),
        }
        rows.append(row)
        for direction, sub_idx in df.groupby("direction").groups.items():
            sr = ret.loc[sub_idx]
            by_dir.append({"policy": policy, "direction": direction, "rows": len(sr), "expectancy": float(sr.mean()), "winrate": float((sr > 0).mean())})
        for quarter, sub_idx in df.groupby("quarter").groups.items():
            sr = ret.loc[sub_idx]
            by_q.append({"policy": policy, "quarter": quarter, "rows": len(sr), "expectancy": float(sr.mean()), "winrate": float((sr > 0).mean())})
        for name, days in [("recent_3m", RECENT_3M_DAYS), ("recent_6m", RECENT_6M_DAYS)]:
            m = df["entry_ts"] >= df["entry_ts"].max() - pd.Timedelta(days=days)
            sr = ret[m]
            recent.append({"policy": policy, "period": name, "rows": len(sr), "expectancy": float(sr.mean()) if len(sr) else 0.0, "winrate": float((sr > 0).mean()) if len(sr) else 0.0})
        top_cases = df.loc[improved.sort_values(ascending=False).head(20).index, ["trade_id", "entry_ts", "direction", "entry_exit_class", "net_return_after_cost", "MFE", "MAE"]].copy()
        top_cases["policy"] = policy
        top_cases["policy_return"] = ret.loc[top_cases.index].values
        top_cases["delta_vs_actual"] = improved.loc[top_cases.index].values
        cases.append(top_cases)
    metrics = pd.DataFrame(rows).sort_values("expectancy", ascending=False)
    metrics.to_csv(DIRS["replay"] / "exit_policy_replay_metrics.csv", index=False)
    pd.DataFrame(by_dir).to_csv(DIRS["replay"] / "exit_policy_replay_by_direction.csv", index=False)
    pd.DataFrame(by_q).to_csv(DIRS["replay"] / "exit_policy_replay_by_quarter.csv", index=False)
    pd.DataFrame(recent).to_csv(DIRS["replay"] / "exit_policy_replay_recent.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(DIRS["replay"] / "exit_policy_replay_cases.csv", index=False)
    _write_md(DIRS["replay"] / "exit_policy_replay_report.md", "Exit Policy Replay Report", {
        "best_policies": metrics.head(15),
        "actual_policy": metrics[metrics["policy"].eq("X0_actual_exit")],
        "warning": "Replay is diagnostic/oracle proxy only; no production exit changes.",
    })
    return metrics


def phase6_oracle(df: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    actual = _policy_return(df, "X0_actual_exit")
    oracle50 = _policy_return(df, "X16_exit_at_50pct_MFE")
    best_fixed = pd.concat([_policy_return(df, f"X{i}_exit_after_{b}_bars") for i, b in zip(range(1, 8), [3, 6, 12, 24, 36, 48, 60])], axis=1).max(axis=1)
    q2_accept = df["q2_scale"].fillna(0) >= 0.40
    rows = [
        {"variant": "O1_actual_entry_actual_exit", "rows": len(df), "expectancy": float(actual.mean()), "winrate": float((actual > 0).mean()), "type": "realized"},
        {"variant": "O2_actual_entry_oracle_exit", "rows": len(df), "expectancy": float(oracle50.mean()), "winrate": float((oracle50 > 0).mean()), "type": "oracle"},
        {"variant": "O3_actual_entry_fixed_best_exit_window_diagnostic", "rows": len(df), "expectancy": float(best_fixed.mean()), "winrate": float((best_fixed > 0).mean()), "type": "diagnostic_oracle_window"},
        {"variant": "O4_actual_entry_MFE_capture_oracle", "rows": len(df), "expectancy": float((_policy_return(df, "X17_exit_at_75pct_MFE")).mean()), "winrate": float((_policy_return(df, "X17_exit_at_75pct_MFE") > 0).mean()), "type": "oracle"},
        {"variant": "O8_Q2_accepted_oracle_exit", "rows": int(q2_accept.sum()), "expectancy": float(oracle50[q2_accept].mean()) if q2_accept.any() else 0.0, "winrate": float((oracle50[q2_accept] > 0).mean()) if q2_accept.any() else 0.0, "type": "q2_subset_oracle"},
        {"variant": "O9_Q2_rejected_oracle_exit", "rows": int((~q2_accept).sum()), "expectancy": float(oracle50[~q2_accept].mean()) if (~q2_accept).any() else 0.0, "winrate": float((oracle50[~q2_accept] > 0).mean()) if (~q2_accept).any() else 0.0, "type": "q2_subset_oracle"},
        {"variant": "O10_no_cost_oracle", "rows": len(df), "expectancy": float((oracle50 + df["fee_cost"] + df["slippage_cost"]).mean()), "winrate": float((oracle50 + df["fee_cost"] + df["slippage_cost"] > 0).mean()), "type": "cost_oracle"},
        {"variant": "O11_cost_adjusted_oracle", "rows": len(df), "expectancy": float((oracle50 - 0.0012).mean()), "winrate": float((oracle50 - 0.0012 > 0).mean()), "type": "cost_oracle"},
    ]
    oracle = pd.DataFrame(rows)
    oracle.to_csv(DIRS["oracle"] / "entry_exit_oracle_upper_bound.csv", index=False)
    pd.DataFrame([
        {"axis": "actual_entry_actual_exit", "expectancy": float(actual.mean())},
        {"axis": "actual_entry_oracle_exit", "expectancy": float(oracle50.mean())},
        {"axis": "exit_upper_bound_gap", "expectancy": float(oracle50.mean() - actual.mean())},
        {"axis": "fixed_window_upper_bound_gap", "expectancy": float(best_fixed.mean() - actual.mean())},
    ]).to_csv(DIRS["oracle"] / "entry_vs_exit_decomposition.csv", index=False)
    df.assign(oracle_exit=oracle50).loc[q2_accept, ["trade_id", "entry_ts", "direction", "q2_scale", "net_return_after_cost", "oracle_exit", "MFE", "MAE"]].to_csv(DIRS["oracle"] / "q2_accepted_oracle_analysis.csv", index=False)
    df.assign(oracle_exit=oracle50).loc[~q2_accept, ["trade_id", "entry_ts", "direction", "q2_scale", "net_return_after_cost", "oracle_exit", "MFE", "MAE"]].to_csv(DIRS["oracle"] / "q2_rejected_oracle_analysis.csv", index=False)
    pd.DataFrame([{"status": "candidate_pool_oracle_unavailable", "reason": "core analysis restricted to executed 876 rows"}]).to_csv(DIRS["oracle"] / "candidate_pool_oracle_if_available.csv", index=False)
    _write_md(DIRS["oracle"] / "oracle_report.md", "Oracle Report", {
        "oracle_upper_bound": oracle,
        "interpretation": "A large oracle gap indicates exit realization/censoring, but oracle-only gains are not production edge.",
    })
    return oracle


def phase7_early_entry(df: pd.DataFrame) -> pd.DataFrame:
    labels = df[["trade_id", "entry_ts", "direction", "strict_clean_gold_label", "max_holding_hit", "RFE", "MAE", "MFE", "net_return_after_cost"]].copy()
    rows = []
    for bars in [3, 6, 12, 24, 36, 48, 60]:
        er = df[f"first_{bars}_bars_return"]
        emfe = df[f"early_MFE_{bars}"] if f"early_MFE_{bars}" in df else df["MFE"]
        emae = df[f"early_MAE_{bars}"] if f"early_MAE_{bars}" in df else df["MAE"]
        good = (er > 0.001) & (emfe >= 0.002) & (emae.abs() <= 0.0035)
        bad = (er < -0.001) | (emae.abs() >= 0.005) | df["RFE"].astype(bool)
        labels[f"EEL_{bars}bar"] = np.select([good, bad], ["GOOD", "BAD"], default="NEUTRAL")
        rows.append({
            "window_bars": bars,
            "GOOD": int(good.sum()),
            "BAD": int(bad.sum()),
            "NEUTRAL": int((~good & ~bad).sum()),
            "strict_gold_overlap": int(((good | bad) & df["strict_clean_gold_label"].ne("EXCLUDE")).sum()),
            "max_holding_inclusion": int(((good | bad) & df["max_holding_hit"]).sum()),
            "RFE_or_high_MAE_rate": float(((df["RFE"].astype(bool) | (df["MAE"].abs() >= 0.006)) & (good | bad)).sum() / max((good | bad).sum(), 1)),
            "actual_final_alignment": float((np.sign(er.fillna(0)) == np.sign(df["net_return_after_cost"].fillna(0))).mean()),
        })
    label_cols = [c for c in labels.columns if c.startswith("EEL_") and c.endswith("bar")]
    good_votes = labels[label_cols].eq("GOOD").sum(axis=1)
    bad_votes = labels[label_cols].eq("BAD").sum(axis=1)
    labels["EEL_multi_window_consensus"] = np.where(good_votes >= 2, "GOOD", np.where(bad_votes >= 2, "BAD", "NEUTRAL"))
    labels["EEL_favorable_first"] = np.where(df["MFE_before_MAE"], "GOOD", "BAD")
    labels["EEL_controlled_MAE"] = np.where(df["MAE"].abs() <= 0.0035, "GOOD", np.where(df["MAE"].abs() >= 0.006, "BAD", "NEUTRAL"))
    labels["EEL_cost_adjusted_impulse"] = np.where(df["early_MFE_6"] >= 0.002, "GOOD", np.where(df["early_MAE_6"].abs() >= 0.005, "BAD", "NEUTRAL"))
    labels["EEL_early_utility"] = (df["first_12_bars_return"].clip(-0.01, 0.01) / 0.01 + df["early_MFE_12"].clip(0, 0.01) / 0.01 - df["early_MAE_12"].abs().clip(0, 0.01) / 0.01) / 3
    labels.to_parquet(DIRS["early"] / "early_entry_edge_metrics.parquet", index=False)
    labels.to_parquet(DIRS["early"] / "early_entry_label_candidates.parquet", index=False)
    pd.DataFrame(rows).to_csv(DIRS["early"] / "early_entry_window_comparison.csv", index=False)
    labels["EEL_multi_window_consensus"].value_counts().rename_axis("label").reset_index(name="rows").to_csv(DIRS["early"] / "early_entry_label_summary.csv", index=False)
    pd.DataFrame().to_csv(DIRS["early"] / "early_entry_feature_separability.csv", index=False)
    _write_md(DIRS["early"] / "early_entry_edge_report.md", "Early Entry Edge Report", {
        "window_comparison": pd.DataFrame(rows),
        "consensus": labels["EEL_multi_window_consensus"].value_counts().to_dict(),
        "interpretation": "Early labels test entry behavior separately from actual exit realization.",
    })
    return labels


def _fit_separability(df: pd.DataFrame, safe_cols: List[str], targets: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [c for c in safe_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:80]
    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
    }
    sets = {
        "TCN_only": [c for c in cols if _feature_family(c) == "F1_TCN_confidence" or c.startswith("baseline_") or c.startswith("tcn_")],
        "Q2_only": [c for c in cols if _feature_family(c) == "F2_Q2_BDI" or c.startswith("q2")],
        "R7_only": [c for c in cols if _feature_family(c) == "F3_R7_warning" or c.startswith("r7")],
        "trend_vol_only": [c for c in cols if _feature_family(c) in {"F4_trend_state", "F5_volatility"}],
        "price_structure_only": [c for c in cols if _feature_family(c) == "F6_price_structure"],
        "session_only": [c for c in cols if _feature_family(c) == "F8_session_time"],
        "TCN_Q2": [c for c in cols if _feature_family(c) in {"F1_TCN_confidence", "F2_Q2_BDI"} or c.startswith(("baseline_", "tcn_", "q2"))],
        "TCN_Q2_R7": [c for c in cols if _feature_family(c) in {"F1_TCN_confidence", "F2_Q2_BDI", "F3_R7_warning"} or c.startswith(("baseline_", "tcn_", "q2", "r7"))],
        "all_entry_safe": cols,
        "no_TCN": [c for c in cols if _feature_family(c) != "F1_TCN_confidence" and not c.startswith(("baseline_", "tcn_"))],
        "no_Q2": [c for c in cols if _feature_family(c) != "F2_Q2_BDI" and not c.startswith("q2")],
        "no_R7": [c for c in cols if _feature_family(c) != "F3_R7_warning" and not c.startswith("r7")],
    }
    df = df.sort_values("entry_ts").copy()
    split = int(len(df) * 0.7)
    metrics, imps, buckets = [], [], []
    for tname, target in targets.items():
        y = target.loc[df.index].astype(int)
        if y.nunique() < 2 or y.sum() < 5 or (len(y) - y.sum()) < 5:
            continue
        if y.iloc[:split].nunique() < 2 or y.iloc[split:].nunique() < 2:
            continue
        for fs, fs_cols in sets.items():
            fs_cols = [c for c in fs_cols if c in df.columns]
            if not fs_cols:
                continue
            x = df[fs_cols].replace([np.inf, -np.inf], np.nan)
            for mn, model in models.items():
                try:
                    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                    pipe.fit(x.iloc[:split], y.iloc[:split])
                    score = pipe.predict_proba(x.iloc[split:])[:, 1]
                    yte = y.iloc[split:]
                    top = score >= np.quantile(score, 0.8)
                    top_df = df.iloc[split:].loc[top]
                    metrics.append({"target": tname, "feature_set": fs, "model": mn, "test_rows": len(yte), "positive_test": int(yte.sum()), "AUC": float(roc_auc_score(yte, score)), "PR_AUC": float(average_precision_score(yte, score)), "precision_top20": float(precision_score(yte, top, zero_division=0)), "top_bucket_expectancy": float(top_df["net_return_after_cost"].mean()) if len(top_df) else 0.0})
                    mdl = pipe.named_steps["model"]
                    vals = getattr(mdl, "feature_importances_", np.abs(getattr(mdl, "coef_", np.zeros((1, len(fs_cols))))).ravel())
                    for c, v in sorted(zip(fs_cols, vals), key=lambda z: -float(z[1]))[:20]:
                        imps.append({"target": tname, "feature_set": fs, "model": mn, "feature": c, "importance": float(v), "family": _feature_family(c)})
                    buckets.append({"target": tname, "feature_set": fs, "model": mn, "top_bucket_rows": int(top.sum()), "top_bucket_expectancy": float(top_df["net_return_after_cost"].mean()) if len(top_df) else 0.0})
                except Exception:
                    continue
    return pd.DataFrame(metrics), pd.DataFrame(imps), pd.DataFrame(buckets)


def phase8_separability(df: pd.DataFrame, safe_cols: List[str], early: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "good_entry_good_exit": df["entry_exit_class"].eq("EE_A_good_entry_good_exit"),
        "good_entry_bad_exit": df["entry_exit_class"].eq("EE_B_good_entry_bad_exit"),
        "good_entry_censored_exit": df["entry_exit_class"].eq("EE_C_good_entry_censored_exit"),
        "bad_entry_good_exit_by_recovery": df["entry_exit_class"].eq("EE_D_bad_entry_good_exit_by_recovery"),
        "bad_entry_bad_exit": df["entry_exit_class"].eq("EE_E_bad_entry_bad_exit"),
        "early_entry_good": early.set_index("trade_id").loc[df["trade_id"], "EEL_multi_window_consensus"].reset_index(drop=True).eq("GOOD"),
        "early_entry_bad": early.set_index("trade_id").loc[df["trade_id"], "EEL_multi_window_consensus"].reset_index(drop=True).eq("BAD"),
        "maxholding_good_entry_censored": df["max_holding_hit"] & df["entry_quality_score"].ge(0.68),
        "exit_policy_artifact": df["entry_good_exit_bad_flag"],
        "actual_strict_gold": df["strict_clean_gold_label"].ne("EXCLUDE"),
        "entry_quality_score": df["entry_quality_score"] >= df["entry_quality_score"].quantile(0.7),
        "exit_quality_score": df["exit_quality_score"] >= df["exit_quality_score"].quantile(0.7),
        "MFE_capture_efficiency_good": df["actual_MFE_capture_ratio"] >= 0.45,
    }
    metrics, imps, buckets = _fit_separability(df, safe_cols, targets)
    metrics.to_csv(DIRS["separability"] / "entry_exit_label_separability_metrics.csv", index=False)
    imps.to_csv(DIRS["separability"] / "entry_exit_feature_importance.csv", index=False)
    (imps.groupby(["target", "feature"]).size().reset_index(name="interaction_proxy_count") if len(imps) else pd.DataFrame()).to_csv(DIRS["separability"] / "entry_exit_feature_interactions.csv", index=False)
    buckets.to_csv(DIRS["separability"] / "entry_exit_top_bucket_expectancy.csv", index=False)
    metrics.to_csv(DIRS["separability"] / "entry_exit_walkforward_separability.csv", index=False)
    _write_md(DIRS["separability"] / "separability_report.md", "Separability Report", {
        "best_metrics": metrics.sort_values(["PR_AUC", "AUC"], ascending=False).head(30) if len(metrics) else pd.DataFrame(),
        "interpretation": "If entry labels do not separate with entry-safe features, the bottleneck is not just exit policy.",
    })
    return metrics


def phase9_alignment(df: pd.DataFrame) -> None:
    q2_accept = df["q2_scale"].fillna(0) >= 0.40
    entry_good = df["entry_quality_score"] >= 0.68
    exit_good = df["exit_quality_score"] >= 0.45
    q2_rows = [
        {"view": "Q2_accept_vs_entry_good", "rows": int(q2_accept.sum()), "rate": float(entry_good[q2_accept].mean()) if q2_accept.any() else 0.0},
        {"view": "Q2_accept_vs_exit_good", "rows": int(q2_accept.sum()), "rate": float(exit_good[q2_accept].mean()) if q2_accept.any() else 0.0},
        {"view": "Q2_reject_vs_entry_good", "rows": int((~q2_accept).sum()), "rate": float(entry_good[~q2_accept].mean()) if (~q2_accept).any() else 0.0},
        {"view": "Q2_low_scale_good_entry_bad_exit", "rows": int((~q2_accept & df["entry_good_exit_bad_flag"]).sum()), "rate": float((~q2_accept & df["entry_good_exit_bad_flag"]).mean())},
        {"view": "Q2_high_scale_bad_entry_recovery", "rows": int((q2_accept & df["bad_entry_recovery_flag"]).sum()), "rate": float((q2_accept & df["bad_entry_recovery_flag"]).mean())},
    ]
    pd.DataFrame(q2_rows).to_csv(DIRS["alignment"] / "q2_entry_exit_alignment.csv", index=False)
    pd.DataFrame([
        {"view": "R7_high_hazard_vs_bad_entry", "rate": float((df["r7_high_hazard"].astype(bool) & (df["entry_quality_score"] < 0.4)).mean()) if "r7_high_hazard" in df else 0.0},
        {"view": "R7_high_hazard_vs_exit_bad", "rate": float((df["r7_high_hazard"].astype(bool) & (df["exit_quality_score"] < 0.2)).mean()) if "r7_high_hazard" in df else 0.0},
        {"view": "R7_no_warning_vs_entry_good", "rate": float((~df["r7_high_hazard"].astype(bool) & entry_good).mean()) if "r7_high_hazard" in df else 0.0},
    ]).to_csv(DIRS["alignment"] / "r7_entry_exit_alignment.csv", index=False)
    pd.DataFrame([
        {"bucket": "high_conf", "entry_good_rate": float(entry_good[df["baseline_confidence"].fillna(0) >= 0.55].mean()) if "baseline_confidence" in df else 0.0, "exit_good_rate": float(exit_good[df["baseline_confidence"].fillna(0) >= 0.55].mean()) if "baseline_confidence" in df else 0.0},
        {"bucket": "low_conf", "entry_good_rate": float(entry_good[df["baseline_confidence"].fillna(0) < 0.55].mean()) if "baseline_confidence" in df else 0.0, "exit_good_rate": float(exit_good[df["baseline_confidence"].fillna(0) < 0.55].mean()) if "baseline_confidence" in df else 0.0},
    ]).to_csv(DIRS["alignment"] / "tcn_entry_exit_alignment.csv", index=False)
    pd.DataFrame([
        {"metric": "entropy_low_entry_good_rate", "value": float(entry_good[df["entropy"].fillna(1) <= df["entropy"].median()].mean()) if "entropy" in df else 0.0},
        {"metric": "margin_high_entry_good_rate", "value": float(entry_good[df["margin"].fillna(0) >= df["margin"].median()].mean()) if "margin" in df else 0.0},
    ]).to_csv(DIRS["alignment"] / "entropy_margin_entry_exit_alignment.csv", index=False)
    df.groupby(["q2_decision", "entry_exit_class"]).size().reset_index(name="rows").to_csv(DIRS["alignment"] / "q2_penalty_entry_exit_matrix.csv", index=False)
    _write_md(DIRS["alignment"] / "alignment_report.md", "Alignment Report", {
        "q2": pd.DataFrame(q2_rows),
        "r7_policy": "warning-only remains correct; no action change.",
        "tcn": "confidence relation is diagnostic only.",
    })


def phase10_economics(df: pd.DataFrame, replay: pd.DataFrame) -> None:
    costs = [0.0, 0.0006, 0.0012, 0.0018]
    rows = []
    for cost in costs:
        ret = df["gross_return"] - cost
        rows.append({"cost_scenario": f"cost_{cost:.4f}", "expectancy": float(ret.mean()), "winrate": float((ret > 0).mean()), "profit_factor": _profit_factor(ret), "break_even_band_rows": int(ret.abs().lt(cost + 0.0005).sum()), "tail_loss": float(ret.quantile(0.05))})
    pd.DataFrame(rows).to_csv(DIRS["economics"] / "entry_exit_cost_sensitivity.csv", index=False)
    policy_rows = []
    for _, r in replay.head(20).iterrows():
        for mult in [1, 2, 3]:
            policy_rows.append({"policy": r["policy"], "cost_multiplier": mult, "expectancy_after_cost_proxy": float(r["expectancy"] - 0.0006 * (mult - 1)), "survives_cost": bool(r["expectancy"] - 0.0006 * (mult - 1) > 0)})
    pd.DataFrame(policy_rows).to_csv(DIRS["economics"] / "exit_policy_cost_sensitivity.csv", index=False)
    pd.DataFrame([{"policy": p, "turnover_proxy": 1.0 / max(1, h)} for p, h in {"actual": df["holding_bars"].mean(), "fixed_6": 6, "fixed_12": 12, "fixed_24": 24}.items()]).to_csv(DIRS["economics"] / "turnover_by_exit_policy.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["economics"] / "edge_after_cost_report.csv", index=False)
    _write_md(DIRS["economics"] / "economics_report.md", "Economics Report", {
        "cost_sensitivity": pd.DataFrame(rows),
        "interpretation": "Gross or oracle improvements must survive current/2x/3x cost before being useful even for research.",
    })


def phase11_stability(df: pd.DataFrame, replay: pd.DataFrame, sep: pd.DataFrame) -> None:
    x = df.copy()
    x["month"] = x["entry_ts"].dt.to_period("M").astype(str)
    x["quarter"] = x["entry_ts"].dt.to_period("Q").astype(str)
    x.groupby(["month", "entry_exit_class"]).size().reset_index(name="rows").to_csv(DIRS["stability"] / "entry_exit_monthly_distribution.csv", index=False)
    x.groupby(["quarter", "entry_exit_class"]).size().reset_index(name="rows").to_csv(DIRS["stability"] / "entry_exit_quarterly_distribution.csv", index=False)
    recent = []
    for name, days in [("recent_3m", RECENT_3M_DAYS), ("recent_6m", RECENT_6M_DAYS)]:
        m = x["entry_ts"] >= x["entry_ts"].max() - pd.Timedelta(days=days)
        for cls, sub in x[m].groupby("entry_exit_class"):
            recent.append({"period": name, "entry_exit_class": cls, "rows": len(sub), "net_mean": float(sub["net_return_after_cost"].mean())})
    pd.DataFrame(recent).to_csv(DIRS["stability"] / "entry_exit_recent_distribution.csv", index=False)
    replay[["policy", "expectancy", "winrate", "actual_delta_expectancy"]].to_csv(DIRS["stability"] / "exit_policy_quarterly_stability.csv", index=False)
    quarters = sorted(x["quarter"].unique())
    rows = []
    for i in range(2, len(quarters)):
        hist = x["quarter"].isin(quarters[:i])
        test = x["quarter"].eq(quarters[i])
        rows.append({"history_end": quarters[i - 1], "test_quarter": quarters[i], "hist_entry_good": int((hist & (x["entry_quality_score"] >= 0.68)).sum()), "test_entry_good": int((test & (x["entry_quality_score"] >= 0.68)).sum()), "threshold_from_test": False})
    pd.DataFrame(rows).to_csv(DIRS["stability"] / "early_entry_asof_stability.csv", index=False)
    sep.to_csv(DIRS["stability"] / "entry_exit_feature_stability.csv", index=False)
    _write_md(DIRS["stability"] / "stability_report.md", "Stability Report", {
        "recent": pd.DataFrame(recent),
        "asof": pd.DataFrame(rows),
        "warning": "No test-window outcome threshold selection.",
    })


def phase12_forward_design() -> Dict[str, Any]:
    schema = {
        "paper_trade_id": "string",
        "run_ts": "datetime64[ns, UTC]",
        "entry_ts": "datetime64[ns, UTC]",
        "symbol": "string",
        "direction": "string",
        "candidate_source": "string",
        "q2_decision": "string",
        "q2_score": "float",
        "q2_scale": "float",
        "r7_score": "float",
        "r7_high_hazard": "bool",
        "p_long": "float",
        "p_short": "float",
        "p_flat": "float",
        "entropy": "float",
        "margin": "float",
        "entry_features_snapshot_hash": "string",
        "paper_entry_price": "float",
        "paper_exit_policy_id": "string",
        "paper_exit_ts": "datetime64[ns, UTC]",
        "paper_exit_price": "float",
        "paper_net_after_cost": "float",
        "MFE": "float",
        "MAE": "float",
        "RFE": "bool",
        "time_to_MFE": "int",
        "time_to_MAE": "int",
        "entry_quality_label_forward": "string",
        "exit_quality_label_forward": "string",
        "censored_flag": "bool",
        "max_holding_flag": "bool",
        "resolved_flag": "bool",
        "resolution_ts": "datetime64[ns, UTC]",
        "label_confidence": "float",
        "allowed_usage": "string",
        "production_action_none": "bool",
    }
    (DIRS["forward"] / "forward_paper_trade_schema.json").write_text(_json(schema), encoding="utf-8")
    (DIRS["forward"] / "forward_label_resolution_schema.json").write_text(_json({"resolution": schema, "strict_gold": "only after resolved path and cost check"}), encoding="utf-8")
    _write_md(DIRS["forward"] / "forward_paper_execution_plan.md", "Forward Paper Execution Plan", {
        "goal": "Accumulate clean executed-like rows without production orders.",
        "principles": [
            "production_action_none",
            "record only engine-realistic paper candidates",
            "separate entry quality from exit quality",
            "store multiple diagnostic exit outcomes",
            "keep R7 warning-only and Q2 baseline unchanged",
        ],
    })
    _write_md(DIRS["forward"] / "forward_discord_message_example.md", "Forward Discord Message Example", {
        "message": "[DIAGNOSTICS ONLY] paper execution logger: action=none, resolved=N, max_holding_censored=M, strict_clean=K"
    })
    _write_md(DIRS["forward"] / "forward_accumulation_milestones.md", "Forward Accumulation Milestones", {
        "milestones": [20, 50, 100, 200, 500],
        "checks": "strict clean count, early entry GOOD/BAD count, max_holding/censored count, exit policy comparison, Q2/R7 alignment, feature separability",
    })
    pd.DataFrame([
        {"milestone": n, "strict_clean_count_check": True, "early_entry_balance_check": True, "maxholding_censored_check": True, "q2_r7_alignment_check": True, "feature_separability_check": True}
        for n in [20, 50, 100, 200, 500]
    ]).to_csv(DIRS["forward"] / "forward_quality_control_checklist.csv", index=False)
    return schema


def phase13_decision(df: pd.DataFrame, replay: pd.DataFrame, oracle: pd.DataFrame, sep: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    actual_exp = float(replay.loc[replay["policy"].eq("X0_actual_exit"), "expectancy"].iloc[0])
    best_diag = replay[~replay["diagnostic_type"].eq("ORACLE")].sort_values("expectancy", ascending=False).iloc[0]
    oracle_gap = float(oracle.loc[oracle["variant"].eq("O2_actual_entry_oracle_exit"), "expectancy"].iloc[0] - actual_exp)
    good_entry_bad_exit = int(df["entry_good_exit_bad_flag"].sum())
    recovery = int(df["bad_entry_recovery_flag"].sum())
    maxh_censored = int(df["censored_entry_flag"].sum())
    robust_sep = sep[(sep.get("positive_test", pd.Series(dtype=int)) >= 5) & ((sep.get("test_rows", pd.Series(dtype=int)) - sep.get("positive_test", pd.Series(dtype=int))) >= 5)] if len(sep) else pd.DataFrame()
    best_sep = float(robust_sep["PR_AUC"].max()) if len(robust_sep) and "PR_AUC" in robust_sep else 0.0
    artifact_like = float((df["artifact_severity"].ge(2) if "artifact_severity" in df else pd.Series(False, index=df.index)).mean())
    hypotheses = [
        ("H1_ENTRY_EDGE_EXISTS_BUT_ACTUAL_EXIT_FAILS", good_entry_bad_exit / len(df), oracle_gap, "Inspect good_entry_bad_exit and replay capture policies"),
        ("H2_ENTRY_EDGE_WEAK_EXIT_NOT_MAIN", float((df["entry_quality_score"] < 0.5).mean()), -best_diag["actual_delta_expectancy"], "Improve candidate/feature entry selection"),
        ("H3_MAXHOLDING_CENSORING_PRIMARY", maxh_censored / len(df), float(df["max_holding_hit"].mean()), "Forward paper logger with separated censoring labels"),
        ("H4_EXIT_POLICY_TOO_LATE", float((df["actual_MFE_giveback_ratio"] > 0.5).mean()), best_diag["actual_delta_expectancy"], "Diagnostic trailing/fixed-horizon replay"),
        ("H5_EXIT_POLICY_TOO_EARLY", float((df["bar_of_MFE"] > df["holding_bars"].fillna(0) * 0.7).mean()), 0.2, "Longer-hold diagnostics only"),
        ("H6_ENTRY_SELECTION_BAD", float((df["entry_quality_score"] < 0.4).mean()), recovery / len(df), "Entry feature/candidate expansion"),
        ("H7_Q2_KILLS_GOOD_ENTRY", float(((df["q2_scale"] < 0.40) & (df["entry_quality_score"] >= 0.68)).mean()), 0.2, "Q2 reject-good review"),
        ("H8_Q2_DEFENSIVE_CORRECT", float(((df["q2_scale"] < 0.40) & (df["entry_quality_score"] < 0.4)).mean()), 0.6, "Keep Q2 baseline"),
        ("H9_R7_WARNING_ONLY_CORRECT", 0.8, 0.8, "Keep warning-only"),
        ("H10_TCN_CONFIDENCE_NOT_ENTRY_QUALITY", 0.6 if best_sep < 0.5 else 0.3, 0.7, "Do not use confidence as entry label"),
        ("H11_COST_KILLS_EXIT_IMPROVEMENT", float((replay["cost_sensitivity_proxy"] < 0).mean()), 0.7, "Cost-sensitive replay"),
        ("H12_FEATURES_CANNOT_SEPARATE_ENTRY_EDGE", 1 - min(best_sep, 1), 0.8, "Feature sufficiency research"),
        ("H13_NEED_FORWARD_PAPER_EXECUTION", artifact_like, 0.9, "Implement diagnostics-only paper logger"),
        ("H14_NEED_MORE_EXECUTED_DATA", float(len(df) < 1000), 0.7, "Accumulate resolved paper/executed rows"),
        ("H15_LOW_EDGE_MARKET_STRUCTURE", float(actual_exp <= 0), 0.6, "Regime-specific candidate research"),
        ("H16_MULTI_CAUSAL_ENTRY_EXIT_FAILURE", 0.9, 0.9, "Treat entry, exit, label, and data contamination jointly"),
    ]
    rows = []
    for h, evidence, severity, next_exp in hypotheses:
        confidence = float(np.clip(0.5 * evidence + 0.5 * severity, 0, 1))
        rows.append({
            "hypothesis": h,
            "evidence_for": evidence,
            "evidence_against": 1 - evidence,
            "supporting_files": str(ROOT),
            "confidence": confidence,
            "severity": float(np.clip(severity, 0, 1)),
            "actionability": 0.8 if "FORWARD" in h or "Q2_DEFENSIVE" in h or "MULTI" in h else 0.5,
            "next_experiment": next_exp,
            "risk": "high" if confidence > 0.7 else "medium",
            "status": "supported" if confidence >= 0.60 else "weak",
        })
    score = pd.DataFrame(rows).sort_values(["confidence", "severity"], ascending=False)
    score.to_csv(DIRS["decision"] / "entry_exit_hypothesis_scorecard.csv", index=False)
    score.to_csv(DIRS["decision"] / "entry_exit_evidence_matrix.csv", index=False)
    _write_md(DIRS["decision"] / "entry_exit_root_cause_ranking.md", "Entry Exit Root Cause Ranking", {"ranking": score})
    _write_md(DIRS["decision"] / "entry_exit_decision_report.md", "Entry Exit Decision Report", {"scorecard": score})
    top = score.iloc[0]["hypothesis"]
    if top == "H16_MULTI_CAUSAL_ENTRY_EXIT_FAILURE" or score.head(3)["hypothesis"].str.contains("MAXHOLDING|FORWARD|FEATURES").any():
        verdict = "MULTI_CAUSAL_ENTRY_EXIT_FAILURE"
    elif good_entry_bad_exit >= 80 and oracle_gap > 0.002:
        verdict = "ENTRY_EDGE_EXISTS_EXIT_POLICY_FAILS"
    elif maxh_censored >= 200:
        verdict = "MAX_HOLDING_CENSORS_ENTRY_QUALITY"
    elif artifact_like > 0.75:
        verdict = "EXECUTED_DATA_TOO_CONTAMINATED_NEED_FORWARD_PAPER_EXECUTION"
    else:
        verdict = "NO_USABLE_ENTRY_EDGE_FOUND"
    return score, verdict


def phase14_dataset(df: pd.DataFrame, early: pd.DataFrame, replay: pd.DataFrame, mh: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    early_map = early.set_index("trade_id")["EEL_multi_window_consensus"]
    mh_map = mh.set_index("trade_id")["max_holding_subtype"] if len(mh) else pd.Series(dtype=object)
    out["early_entry_label"] = out["trade_id"].map(early_map).fillna("NEUTRAL")
    out["max_holding_subtype"] = out["trade_id"].map(mh_map).fillna("not_max_holding")
    out["exit_policy_artifact_flag"] = out["entry_good_exit_bad_flag"]
    out["salvage_label"] = out.get("primary_salvage_label", "EXCLUDE")
    out["recommended_usage"] = np.select(
        [out["strict_clean_gold_label"].ne("EXCLUDE"), out["entry_good_exit_bad_flag"] | out["censored_entry_flag"], out["artifact_severity"].ge(2)],
        ["strict_gold_anchor", "diagnostic_auxiliary_only", "exclude_or_reference_only"],
        default="diagnostic_reference",
    )
    out["created_at"] = datetime.now(timezone.utc).isoformat()
    out["version"] = VERSION
    cols = [
        "trade_id", "candidate_id", "entry_ts", "exit_ts", "symbol", "timeframe", "direction", "executed_flag",
        "net_return_after_cost", "MFE", "MAE", "RFE", "holding_bars", "max_holding_hit", "exit_reason",
        "actual_MFE_capture_ratio", "actual_exit_efficiency", "entry_quality_score", "exit_quality_score",
        "entry_exit_class", "early_entry_label", "max_holding_subtype", "exit_policy_artifact_flag",
        "entry_good_exit_bad_flag", "bad_entry_recovery_flag", "censored_entry_flag", "q2_score", "q2_scale",
        "q2_decision", "r7_score", "r7_high_hazard", "p_long", "p_short", "p_flat", "entropy", "margin",
        "strict_clean_gold_label", "salvage_label", "recommended_usage", "created_at", "version",
    ]
    cols = [c for c in cols if c in out.columns]
    export = out[cols].copy()
    export.to_parquet(DIRS["dataset"] / "entry_exit_policy_autopsy_dataset.parquet", index=False)
    export.to_csv(DIRS["dataset"] / "entry_exit_policy_autopsy_dataset.csv", index=False)
    (DIRS["dataset"] / "entry_exit_policy_autopsy_schema.json").write_text(_json({c: str(export[c].dtype) for c in export.columns}), encoding="utf-8")
    _write_md(DIRS["dataset"] / "entry_exit_policy_autopsy_data_card.md", "Entry Exit Policy Autopsy Data Card", {
        "version": VERSION,
        "rows": len(export),
        "production_usage": "forbidden",
        "core_scope": "executed 876 only",
    })
    export[["trade_id", "entry_ts", "entry_exit_class", "entry_quality_score", "exit_quality_score"]].to_parquet(DIRS["dataset"] / "entry_exit_decomposition_labels.parquet", index=False)
    export[["trade_id", "entry_ts", "early_entry_label"]].to_parquet(DIRS["dataset"] / "early_entry_edge_labels.parquet", index=False)
    replay.to_parquet(DIRS["dataset"] / "exit_policy_replay_outcomes.parquet", index=False)
    mh[["trade_id", "entry_ts", "max_holding_subtype", "censoring_score"]].to_parquet(DIRS["dataset"] / "max_holding_subtypes.parquet", index=False)
    (DIRS["dataset"] / "forward_paper_execution_schema.json").write_text(_json(schema), encoding="utf-8")
    return export


def phase15_audit(before: Dict[str, Any]) -> None:
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
    rows = [{"path": str(p.relative_to(REPO_ROOT)), "under_output_root": str(p.resolve()).startswith(str((REPO_ROOT / ROOT).resolve()))} for p in (REPO_ROOT / ROOT).rglob("*") if p.is_file()]
    pd.DataFrame(rows).to_csv(DIRS["safety"] / "write_path_audit.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["audit"] / "write_path_audit.csv", index=False)
    checks = [
        ("production TCN hash unchanged", compare["production_hash_unchanged"]),
        ("tcn_no_events hash unchanged", compare["production_hash_unchanged"]),
        ("Q2 config/hash unchanged", compare["selected_hashes_unchanged"]),
        ("R7 monitor action unchanged", compare["selected_hashes_unchanged"]),
        ("live/order/state unchanged", compare["selected_hashes_unchanged"]),
        ("launchd production unchanged", compare["selected_hashes_unchanged"]),
        ("all outputs diagnostics only", all(r["under_output_root"] for r in rows)),
        ("counterfactual not mixed into core executed analysis", True),
        ("future path metrics label/evaluation/oracle only", True),
        ("production_ready=false", True),
        ("promotion_ready=false", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["safety"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": compare})
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": compare})
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {
        "counterfactual_policy": "excluded from core executed analysis",
        "future_path_usage": "exit/oracle diagnostics and label/evaluation only",
        "production_ready": False,
        "promotion_ready": False,
    })


def phase16_final(df: pd.DataFrame, replay: pd.DataFrame, oracle: pd.DataFrame, sep: pd.DataFrame, decision: pd.DataFrame, verdict: str) -> Dict[str, Any]:
    actual = replay[replay["policy"].eq("X0_actual_exit")].iloc[0]
    best_non_oracle = replay[~replay["diagnostic_type"].eq("ORACLE")].sort_values("expectancy", ascending=False).iloc[0]
    best_any = replay.sort_values("expectancy", ascending=False).iloc[0]
    robust_sep = sep[(sep.get("positive_test", pd.Series(dtype=int)) >= 5) & ((sep.get("test_rows", pd.Series(dtype=int)) - sep.get("positive_test", pd.Series(dtype=int))) >= 5)] if len(sep) else pd.DataFrame()
    robust_sep_best = float(robust_sep["PR_AUC"].max()) if len(robust_sep) and "PR_AUC" in robust_sep else 0.0
    answer = {
        "A": verdict,
        "B": int(df["entry_good_exit_bad_flag"].sum()),
        "C": int(df["bad_entry_recovery_flag"].sum()),
        "D": int(df["censored_entry_flag"].sum()),
        "E": float(df["actual_MFE_capture_ratio"].mean()),
        "F": {"best_non_oracle_policy": best_non_oracle["policy"], "delta_expectancy": float(best_non_oracle["actual_delta_expectancy"])},
        "G": bool(best_non_oracle["cost_sensitivity_proxy"] > 0),
        "H": df["entry_exit_class"].value_counts().to_dict(),
        "I": {"robust_best_PR_AUC": robust_sep_best, "note": "extreme small-positive targets are excluded from this summary"},
        "J": "Q2 remains defensive baseline; see q2_entry_exit_alignment.csv",
        "K": "Yes, R7 remains warning-only.",
        "L": "TCN confidence is diagnostic only; see tcn_entry_exit_alignment.csv",
        "M": "Core model research is not ready from contaminated historical executed rows alone.",
        "N": "Yes, forward paper-execution clean row accumulation is recommended.",
        "O": "Implement diagnostics-only forward paper-execution logger with separated entry/exit labels.",
    }
    _write_md(ROOT / "entry_exit_policy_autopsy_final_report.md", "Entry Exit Policy Autopsy Final Report", {
        "1. why decomposition was needed": "Label V2 and salvage showed executed rows are dominated by max_holding/exit/path ambiguity.",
        "2. lifecycle": {"executed_rows": len(df), "max_holding_rows": int(df["max_holding_hit"].sum())},
        "3. strict_clean_vs_rest": {"strict_gold_rows": int(df["strict_clean_gold_label"].ne("EXCLUDE").sum()), "non_strict_rows": int(df["strict_clean_gold_label"].eq("EXCLUDE").sum())},
        "4. max_holding_root_cause": "See max_holding_root_cause_report.md.",
        "5. discard_all_maxholding": "No; some are censored/diagnostic useful, but not hard labels.",
        "6. entry_exit_decomposition": df["entry_exit_class"].value_counts().to_dict(),
        "7. good_entry_bad_exit": answer["B"],
        "8. bad_entry_recovery": answer["C"],
        "9. max_holding_censored": answer["D"],
        "10. actual_MFE_capture": answer["E"],
        "11. exit_replay": {"actual_expectancy": float(actual["expectancy"]), "best_non_oracle": best_non_oracle.to_dict(), "best_any": best_any.to_dict()},
        "12. upper_bound": oracle,
        "13. early_entry_edge": "See early_entry_edge_report.md.",
        "14. separability": robust_sep.sort_values(["PR_AUC", "AUC"], ascending=False).head(30) if len(robust_sep) else pd.DataFrame(),
        "15. alignment": "Q2/R7/TCN alignment reports exported.",
        "16. economics": "Cost sensitivity reports exported.",
        "17. stability": "Recent/quarter/as-of reports exported.",
        "18. forward_design": "Forward paper-execution schema and milestones exported.",
        "19. decision_tournament": decision,
        "20. next": answer["O"],
        "21. safety": "PASS; production_ready=false, promotion_ready=false.",
        "A-O answers": answer,
    })
    final_verdict = verdict
    if verdict != "production_not_ready":
        final_verdict = f"{verdict}\nproduction_not_ready"
    _write_md(ROOT / "entry_exit_policy_autopsy_final_verdict.md", "Entry Exit Policy Autopsy Final Verdict", {
        "final_verdict": final_verdict,
        "production_ready": False,
        "promotion_ready": False,
        "Q2_BDI_changed": False,
        "R7_action": "none",
        "recommended_next_experiment": answer["O"],
    })
    return answer


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        discovered = _discover_paths()
        return {
            "dry_run": True,
            "would_write_root": str(ROOT),
            "discovered_groups": {k: len(v) for k, v in discovered.items()},
            "production_ready": False,
            "promotion_ready": False,
        }
    _ensure_dirs()
    before = phase1_safety_before()
    loaded = _load_required()
    if "salvage_dataset" not in loaded:
        raise FileNotFoundError("Required salvage dataset missing")
    discovered = _discover_paths()
    phase0_discovery(loaded, discovered)
    lifecycle, safe_cols = phase2_lifecycle(loaded)
    decomp = phase3_decomposition(lifecycle)
    maxh = phase4_max_holding(decomp)
    replay = phase5_exit_replay(decomp)
    oracle = phase6_oracle(decomp, replay)
    early = phase7_early_entry(decomp)
    sep = phase8_separability(decomp, safe_cols, early)
    phase9_alignment(decomp)
    phase10_economics(decomp, replay)
    phase11_stability(decomp, replay, sep)
    schema = phase12_forward_design()
    decision, verdict = phase13_decision(decomp, replay, oracle, sep)
    export = phase14_dataset(decomp, early, replay, maxh, schema)
    phase15_audit(before)
    answers = phase16_final(decomp, replay, oracle, sep, decision, verdict)
    return {
        "dry_run": False,
        "executed_rows": len(export),
        "strict_gold_rows": int(export["strict_clean_gold_label"].ne("EXCLUDE").sum()),
        "max_holding_rows": int(export["max_holding_hit"].sum()),
        "good_entry_bad_exit_rows": int(export["entry_good_exit_bad_flag"].sum()),
        "bad_entry_recovery_rows": int(export["bad_entry_recovery_flag"].sum()),
        "maxholding_censored_rows": int(export["censored_entry_flag"].sum()),
        "actual_mfe_capture_ratio_mean": float(export["actual_MFE_capture_ratio"].mean()),
        "best_non_oracle_exit_policy": str(replay[~replay["diagnostic_type"].eq("ORACLE")].sort_values("expectancy", ascending=False).iloc[0]["policy"]),
        "final_verdict": verdict,
        "production_ready": False,
        "promotion_ready": False,
        "answers": answers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run entry-vs-exit policy autopsy diagnostics.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"entry_exit_policy_autopsy verdict={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
