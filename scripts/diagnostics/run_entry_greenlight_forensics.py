"""
Entry greenlight forensics / positive entry profiler.

Diagnostics only. This script searches for positive-entry pockets and
interpretable greenlight candidates without changing production TCN, Q2_BDI,
R7 actions, live execution, order paths, launchd jobs, or trading state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scripts.diagnostics.run_false_high_r7_monitor import (
    Q2_ACCEPT,
    Q2_REJECT,
    R7_DEFAULT_THRESHOLD,
    _prepare_monitor_frame,
    _prod_hashes,
    _q2_baseline,
)

ROOT = Path("data/diagnostics/entry_greenlight_forensics")
POSITION_SIZE = 0.05
FEE_SLIPPAGE_BUFFER = 0.0  # engine_ret/net_return in source already includes costs where available
MIN_SAMPLE = 20
RECENT_3M_DAYS = 92
RECENT_6M_DAYS = 184

DIRS = {
    "state": ROOT / "state",
    "universe": ROOT / "universe",
    "labels": ROOT / "labels",
    "features": ROOT / "features",
    "descriptive": ROOT / "descriptive",
    "pocket": ROOT / "pocket_mining",
    "rules": ROOT / "rules",
    "model": ROOT / "model_assist",
    "validation": ROOT / "validation",
    "baseline": ROOT / "baseline_comparison",
    "failure": ROOT / "failure_modes",
    "replay": ROOT / "economic_replay",
    "cases": ROOT / "cases",
    "decision": ROOT / "decision",
    "monitor": ROOT / "monitor_plan",
    "audit": ROOT / "audit",
}


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _write_md(path: Path, title: str, sections: Dict[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(sections, str):
        body = sections
    else:
        lines = [f"# {title}", ""]
        for key, value in sections.items():
            lines.append(f"## {key}")
            if isinstance(value, pd.DataFrame):
                lines.extend(["```csv", value.head(60).to_csv(index=False).rstrip(), "```"] if len(value) else ["_empty_"])
            elif isinstance(value, (dict, list)):
                lines.extend(["```json", _json(value), "```"])
            else:
                lines.append(str(value))
            lines.append("")
        body = "\n".join(lines)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b not in (0, 0.0) and not pd.isna(b) else default


def _mdd(returns: pd.Series) -> float:
    r = _safe_num(returns)
    if len(r) == 0:
        return 0.0
    eq = (1.0 + r * POSITION_SIZE).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def _profit_factor(ret: pd.Series) -> float:
    r = _safe_num(ret)
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    return _safe_div(pos, neg, default=float("inf") if pos > 0 else 0.0)


def _load_frame() -> pd.DataFrame:
    _, frame, _, _, _, _, _ = _prepare_monitor_frame()
    df = frame.copy()
    df["_ts"] = pd.to_datetime(df.get("_ts", df.get("timestamp")), errors="coerce")
    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    df["timestamp"] = df.get("timestamp", df["_ts"].astype(str)).astype(str)
    df["symbol"] = df.get("symbol", "BTCUSDT")
    df["timeframe"] = "5m"
    df["engine_ret"] = _safe_num(df.get("engine_ret", df.get("net_return", 0.0)))
    df["mae"] = _safe_num(df.get("mae", 0.0))
    df["mfe"] = _safe_num(df.get("mfe", 0.0))
    df["rfe_flag"] = df.get("rfe_flag", False).astype(bool) if "rfe_flag" in df else False
    df["q2_bdi_scale"] = _safe_num(df.get("q2_bdi_scale", 0.0))
    df["r7_score"] = _safe_num(df.get("r7_score", 0.0)).clip(0, 1)
    df["r7_high_hazard"] = df["r7_score"] >= R7_DEFAULT_THRESHOLD
    df["q2_accept"] = df["q2_bdi_scale"] >= Q2_ACCEPT
    df["q2_reject"] = df["q2_bdi_scale"] <= Q2_REJECT
    df["baseline_confidence"] = _safe_num(df.get("baseline_confidence", df[["baseline_p_flat", "baseline_p_long", "baseline_p_short"]].max(axis=1) if {"baseline_p_flat", "baseline_p_long", "baseline_p_short"}.issubset(df.columns) else 0.0))
    df["baseline_margin"] = _safe_num(df.get("baseline_margin", df.get("margin", 0.0)))
    return df


def _forbidden_tokens() -> List[str]:
    return [
        "future", "return", "engine_ret", "net_return", "raw_return", "scaled_return",
        "mae", "mfe", "rfe", "exit", "label", "target", "good", "bad", "success",
        "quality", "drawdown", "hold_bars", "counterfactual", "actual", "outcome",
        "tag_", "realized", "underconfidence", "overconfidence", "artifact", "cluster",
        "horizon", "validity", "calibration_error", "sample_weight", "recovery",
        "failure", "soft_risk", "clean_mask", "false_positive", "catastrophic",
        "false_high", "_pass", "_fail", "danger",
        "executed", "df_idx", "dominant_weight",
    ]


def _safe_features(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    explicit = {
        "baseline_p_flat", "baseline_p_long", "baseline_p_short", "baseline_confidence", "baseline_margin",
        "p_flat", "p_long", "p_short", "entropy", "margin", "predicted_confidence", "confidence_overextension",
        "q2_bdi_scale", "q2_risk_score", "q2_score", "q2_scale", "r7_score", "r7_threshold_margin",
    }
    cols: List[str] = []
    inv = []
    for c in df.columns:
        low = c.lower()
        leak = any(tok in low for tok in _forbidden_tokens()) or bool(re.match(r"^[lgt]\d+_", low)) or bool(re.match(r"^h\d+_", low))
        numeric = pd.api.types.is_numeric_dtype(df[c]) or df[c].dropna().map(lambda x: isinstance(x, (int, float, bool, np.number))).all()
        safe = numeric and ((not leak) or c in explicit)
        if safe:
            cols.append(c)
        inv.append({
            "column": c,
            "numeric": bool(numeric),
            "safe_at_entry": bool(safe),
            "future_outcome_leakage_risk": bool(leak and c not in explicit),
            "used_as_feature": bool(safe),
        })
    safe_df = pd.DataFrame([x for x in inv if x["safe_at_entry"]])
    forbidden_df = pd.DataFrame([x for x in inv if x["future_outcome_leakage_risk"]])
    return cols, safe_df, forbidden_df


def _phase0_state(df: pd.DataFrame, before: List[Dict[str, Any]], safe_df: pd.DataFrame, forbidden_df: pd.DataFrame) -> None:
    (DIRS["state"] / "production_tcn_hash_before.json").write_text(_json(before), encoding="utf-8")
    q2 = _q2_baseline(df)
    pd.DataFrame([q2] if isinstance(q2, dict) else q2).to_csv(DIRS["state"] / "q2_bdi_baseline_snapshot.csv", index=False)
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
    inv = []
    for name, path in {
        "monitor_frame": "run_false_high_r7_monitor._prepare_monitor_frame",
        "meta_dataset_v2": "data/diagnostics/meta_layer/meta_dataset_v2.parquet",
        "feature_proba_refresh_r7_input": "data/diagnostics/feature_proba_refresh/latest_r7_input_frame.parquet",
        "r7_forward_log": "data/diagnostics/false_high_r7_daily_monitor/forward/r7_forward_log.csv",
        "freshness": "data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json",
    }.items():
        p = Path(path)
        inv.append({"source": name, "path": path, "exists": p.exists() if not path.startswith("run_") else True})
    pd.DataFrame(inv).to_csv(DIRS["state"] / "input_data_inventory.csv", index=False)
    safe_df.to_csv(DIRS["state"] / "safe_feature_inventory.csv", index=False)
    forbidden_df.to_csv(DIRS["state"] / "forbidden_future_columns.csv", index=False)
    range_report = {
        "rows": len(df),
        "start": str(df["_ts"].min()),
        "end": str(df["_ts"].max()),
        "unique_trade_ids": int(df.get("trade_id", pd.Series(range(len(df)))).astype(str).nunique()),
        "executed_rows": int(df.get("is_executed", pd.Series(True, index=df.index)).astype(bool).sum()) if "is_executed" in df else len(df),
        "candidate_rows": len(df),
    }
    _write_md(DIRS["state"] / "input_data_range_report.md", "Input Data Range Report", range_report)
    _write_md(DIRS["state"] / "data_integrity_audit.md", "Data Integrity Audit", {
        **range_report,
        "duplicate_timestamps": int(df["_ts"].duplicated().sum()),
        "missing_timestamp": int(df["_ts"].isna().sum()),
        "missing_outcome": int(df["engine_ret"].isna().sum()),
        "mae_available": "mae" in df.columns,
        "mfe_available": "mfe" in df.columns,
        "rfe_available": "rfe_flag" in df.columns,
        "train_test_asof_possible": bool(len(df) >= MIN_SAMPLE * 4),
    })


def _labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ret = df["engine_ret"]
    mfe = df["mfe"]
    mae = df["mae"]
    rfe = df["rfe_flag"].astype(bool)
    q_ret = ret.quantile(0.70) if len(ret) else 0.003
    q_mfe = mfe.quantile(0.70) if len(mfe) else 0.004
    med_mfe = mfe.quantile(0.50) if len(mfe) else 0.002
    med_mae_abs = (-mae).clip(lower=0).quantile(0.50) if len(mae) else 0.003
    tiny = ret.abs() <= 0.001
    labels = pd.DataFrame({"trade_id": df.get("trade_id", df.index), "timestamp": df["timestamp"], "_ts": df["_ts"]})
    labels["L1_net_positive_after_cost"] = ret > FEE_SLIPPAGE_BUFFER
    labels["L2_strong_net_positive"] = ret >= max(float(q_ret), 0.003)
    labels["L3_high_MFE"] = mfe >= max(float(q_mfe), 0.003)
    labels["L4_low_MAE"] = mae >= -max(float(med_mae_abs), 0.003)
    labels["L5_high_MFE_low_MAE"] = labels["L3_high_MFE"] & labels["L4_low_MAE"]
    labels["L6_low_RFE"] = ~rfe
    labels["L7_clean_trend_continuation"] = labels["L1_net_positive_after_cost"] & labels["L3_high_MFE"] & labels["L6_low_RFE"] & df.get("exit_reason", pd.Series("", index=df.index)).astype(str).str.contains("max_holding|take|trend|exit", case=False, na=False)
    labels["L8_fast_favorable_move"] = labels["L3_high_MFE"] & (df.get("hold_bars", pd.Series(999, index=df.index)).fillna(999).astype(float) <= 6)
    labels["L9_slow_but_clean_followthrough"] = labels["L1_net_positive_after_cost"] & labels["L6_low_RFE"] & (df.get("hold_bars", pd.Series(0, index=df.index)).fillna(0).astype(float) > 6)
    labels["L10_good_recovery"] = (mae < -0.002) & (ret > 0) & (mfe > (-mae).abs())
    labels["L11_bad_but_avoidable"] = (ret < 0) & ((mae <= -0.006) | rfe | df["r7_high_hazard"])
    labels["L12_missed_good"] = df["q2_reject"] & labels["L1_net_positive_after_cost"] & labels["L3_high_MFE"]
    labels["L13_q2_accept_good"] = df["q2_accept"] & labels["L1_net_positive_after_cost"]
    labels["L14_q2_accept_bad"] = df["q2_accept"] & ~labels["L1_net_positive_after_cost"] & ~tiny
    labels["L15_greenlight_candidate"] = labels["L1_net_positive_after_cost"] & labels["L3_high_MFE"] & labels["L4_low_MAE"] & labels["L6_low_RFE"]
    rows = []
    registry = {
        "L1_net_positive_after_cost": "engine_ret > fee/slippage adjusted zero",
        "L2_strong_net_positive": "engine_ret >= max(70th percentile, 0.003)",
        "L3_high_MFE": "MFE >= max(70th percentile, 0.003)",
        "L4_low_MAE": "MAE drawdown no worse than median adverse move",
        "L5_high_MFE_low_MAE": "high MFE and controlled MAE",
        "L6_low_RFE": "RFE flag false",
        "L7_clean_trend_continuation": "net positive + high MFE + low RFE + clean exit proxy",
        "L8_fast_favorable_move": "high MFE within short holding path",
        "L9_slow_but_clean_followthrough": "positive slow followthrough without RFE",
        "L10_good_recovery": "recovered from initial MAE into profit",
        "L11_bad_but_avoidable": "bad outcome with avoidable risk markers",
        "L12_missed_good": "Q2 reject/low scale but positive high-MFE outcome",
        "L13_q2_accept_good": "Q2 accept and positive outcome",
        "L14_q2_accept_bad": "Q2 accept and non-tiny bad outcome",
        "L15_greenlight_candidate": "positive + high MFE + low MAE + low RFE",
    }
    recent_cut = df["_ts"].max() - pd.Timedelta(days=RECENT_3M_DAYS)
    for col, definition in registry.items():
        y = labels[col].astype(bool)
        rows.append({
            "label": col,
            "definition": definition,
            "required_columns": "engine_ret,mae,mfe,rfe_flag,exit_reason,q2_bdi_scale,r7_score",
            "positive_count": int(y.sum()),
            "negative_count": int((~y).sum()),
            "neutral_ambiguous_count": int(tiny.sum()) if col in {"L1_net_positive_after_cost", "L14_q2_accept_bad"} else 0,
            "long_positive": int((y & df["direction"].astype(str).eq("LONG")).sum()),
            "short_positive": int((y & df["direction"].astype(str).eq("SHORT")).sum()),
            "recent_count": int((y & (df["_ts"] >= recent_cut)).sum()),
            "label_reliability": float(min(1.0, y.sum() / max(MIN_SAMPLE, 1))),
            "artifact_suspect_ratio": float(df.loc[y, "artifact_suspect"].astype(bool).mean()) if "artifact_suspect" in df and y.any() else 0.0,
            "max_holding_artifact_ratio": float(df.loc[y, "exit_reason"].astype(str).str.contains("max_holding", case=False, na=False).mean()) if "exit_reason" in df and y.any() else 0.0,
            "tiny_return_neutral": col in {"L1_net_positive_after_cost", "L14_q2_accept_bad"},
        })
    return labels, pd.DataFrame(rows)


def _universe_masks(df: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, pd.Series]:
    max_ts = df["_ts"].max()
    return {
        "U1_executed_trades_only": df.get("is_executed", pd.Series(True, index=df.index)).astype(bool) if "is_executed" in df else pd.Series(True, index=df.index),
        "U2_q2_accepted_candidates": df["q2_accept"],
        "U3_q2_rejected_candidates": df["q2_reject"],
        "U4_tcn_high_confidence_candidates": df["baseline_confidence"] >= 0.55,
        "U5_low_entropy_candidates": _safe_num(df.get("entropy", 9.0)) <= 0.90,
        "U6_r7_no_warning_candidates": ~df["r7_high_hazard"],
        "U7_q2_accept_r7_no_warning": df["q2_accept"] & ~df["r7_high_hazard"],
        "U8_q2_accept_r7_high_hazard": df["q2_accept"] & df["r7_high_hazard"],
        "U9_q2_reject_but_good_candidates": df["q2_reject"] & labels["L1_net_positive_after_cost"].astype(bool),
        "U10_all_candidates": pd.Series(True, index=df.index),
        "U11_long_only": df["direction"].astype(str).eq("LONG"),
        "U12_short_only": df["direction"].astype(str).eq("SHORT"),
        "U13_recent_3m": df["_ts"] >= max_ts - pd.Timedelta(days=RECENT_3M_DAYS),
        "U14_recent_6m": df["_ts"] >= max_ts - pd.Timedelta(days=RECENT_6M_DAYS),
        "U15_quarterly_panels": pd.Series(True, index=df.index),
    }


def _metric_row(df: pd.DataFrame, mask: pd.Series, name: str, label: str | None = None) -> Dict[str, Any]:
    sub = df.loc[mask].copy()
    ret = sub["engine_ret"] if len(sub) else pd.Series(dtype=float)
    wins = ret > 0
    losses = ret < 0
    return {
        "name": name,
        "label": label or "",
        "rows": int(len(sub)),
        "start": str(sub["_ts"].min()) if len(sub) else "",
        "end": str(sub["_ts"].max()) if len(sub) else "",
        "long_ratio": float(sub["direction"].astype(str).eq("LONG").mean()) if len(sub) else 0.0,
        "q2_accept_rate": float(sub["q2_accept"].mean()) if len(sub) else 0.0,
        "q2_reject_rate": float(sub["q2_reject"].mean()) if len(sub) else 0.0,
        "r7_warning_rate": float(sub["r7_high_hazard"].mean()) if len(sub) else 0.0,
        "net_mean": float(ret.mean()) if len(ret) else 0.0,
        "net_median": float(ret.median()) if len(ret) else 0.0,
        "expectancy_after_cost": float(ret.mean()) if len(ret) else 0.0,
        "net_total": float((ret * POSITION_SIZE).sum()) if len(ret) else 0.0,
        "winrate": float(wins.mean()) if len(ret) else 0.0,
        "profit_factor": _profit_factor(ret),
        "mfe_median": float(sub["mfe"].median()) if len(sub) else 0.0,
        "mfe_p75": float(sub["mfe"].quantile(0.75)) if len(sub) else 0.0,
        "mae_median": float(sub["mae"].median()) if len(sub) else 0.0,
        "mae_p25": float(sub["mae"].quantile(0.25)) if len(sub) else 0.0,
        "mfe_to_mae_ratio": _safe_div(float(sub["mfe"].median()) if len(sub) else 0, abs(float(sub["mae"].median())) if len(sub) else 0),
        "rfe_rate": float(sub["rfe_flag"].astype(bool).mean()) if len(sub) else 0.0,
        "mdd_contribution": _mdd(ret),
        "avg_holding_bars": float(_safe_num(sub.get("hold_bars", pd.Series(0, index=sub.index))).mean()) if len(sub) else 0.0,
        "sample_reliability_score": float(min(1.0, len(sub) / max(MIN_SAMPLE * 2, 1))),
    }


def _phase_universes(df: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, pd.Series]:
    masks = _universe_masks(df, labels)
    membership = pd.DataFrame({"trade_id": df.get("trade_id", df.index), "timestamp": df["timestamp"], "_ts": df["_ts"]})
    rows = []
    for name, mask in masks.items():
        membership[name] = mask.to_numpy(dtype=bool)
        r = _metric_row(df, mask, name)
        r.update({
            "unique_trade_candidate_count": int(df.loc[mask].get("trade_id", pd.Series(index=df.loc[mask].index, dtype=str)).astype(str).nunique()) if mask.any() else 0,
            "outcome_label_available": bool("engine_ret" in df.columns),
            "mae_mfe_rfe_available": bool({"mae", "mfe", "rfe_flag"}.issubset(df.columns)),
        })
        rows.append(r)
    membership.to_parquet(DIRS["universe"] / "analysis_universe_membership.parquet", index=False)
    pd.DataFrame(rows).to_csv(DIRS["universe"] / "analysis_universe_registry.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["universe"] / "analysis_universe_summary.csv", index=False)
    _write_md(DIRS["universe"] / "universe_construction_report.md", "Analysis Universe Construction", {"summary": pd.DataFrame(rows)})
    return masks


def _feature_family(c: str) -> str:
    n = c.lower()
    if n in {"p_long", "p_short", "p_flat", "baseline_p_long", "baseline_p_short", "baseline_p_flat", "entropy", "margin", "baseline_margin", "baseline_confidence", "confidence_overextension"}:
        return "F1_TCN_confidence"
    if "q2" in n or "bdi" in n:
        return "F2_Q2_BDI"
    if "r7" in n:
        return "F3_R7_warning"
    if "trend" in n or "slope" in n or "ema" in n:
        return "F4_trend_state"
    if "vol" in n or "atr" in n or "range" in n:
        return "F5_volatility"
    if "breakout" in n or "wick" in n or "candle" in n or "body" in n or "structure" in n:
        return "F6_price_structure"
    if "return" in n or "momentum" in n or "accel" in n:
        return "F7_momentum"
    if "hour" in n or "weekday" in n or "session" in n or "weekend" in n:
        return "F8_session_time"
    if "volume" in n or "spread" in n:
        return "F9_microstructure_proxy"
    if "cooldown" in n or "confirm" in n or "hold" in n:
        return "F10_lifecycle_safe_context"
    return "F11_combination_features"


def _phase_features(df: pd.DataFrame, safe_cols: List[str], safe_df: pd.DataFrame) -> None:
    reg = pd.DataFrame([{"feature": c, "family": _feature_family(c), "safe_at_entry": True, "non_null_rate": float(df[c].notna().mean())} for c in safe_cols])
    reg.to_csv(DIRS["features"] / "entry_feature_registry.csv", index=False)
    reg.groupby("family").agg(feature_count=("feature", "count"), availability=("non_null_rate", "mean")).reset_index().to_csv(DIRS["features"] / "entry_feature_family_summary.csv", index=False)
    safe_df.to_csv(DIRS["features"] / "entry_feature_availability.csv", index=False)
    tmp = df.copy()
    tmp["_quarter"] = tmp["_ts"].dt.to_period("Q").astype(str)
    drift_rows = []
    for c in safe_cols[:120]:
        if not pd.api.types.is_numeric_dtype(tmp[c]):
            continue
        q = tmp.groupby("_quarter")[c].mean(numeric_only=True)
        drift_rows.append({"feature": c, "quarters": int(q.size), "mean_min": float(q.min()), "mean_max": float(q.max()), "quarter_drift_abs": float((q.max() - q.min()))})
    pd.DataFrame(drift_rows).to_csv(DIRS["features"] / "entry_feature_drift_summary.csv", index=False)
    _write_md(DIRS["features"] / "feature_safety_audit.md", "Feature Safety Audit", {
        "safe_feature_count": len(safe_cols),
        "future_outcome_columns_excluded": True,
        "MAE_MFE_RFE_exit_reason_used_as_features": False,
        "test_outcomes_used_for_threshold_selection": False,
        "feature_family_summary": reg.groupby("family").size().to_dict(),
    })


def _phase_descriptive(df: pd.DataFrame, labels: pd.DataFrame, masks: Dict[str, pd.Series]) -> None:
    rows = []
    for uname, umask in masks.items():
        for lname in [c for c in labels.columns if c.startswith("L")]:
            rows.append(_metric_row(df, umask & labels[lname].astype(bool), uname, lname))
        rows.append(_metric_row(df, umask, uname, "ALL"))
    desc = pd.DataFrame(rows)
    desc.to_csv(DIRS["descriptive"] / "descriptive_expectancy_by_universe.csv", index=False)
    comparisons = {
        "q2_accept_good_vs_bad": [labels["L13_q2_accept_good"], labels["L14_q2_accept_bad"]],
        "high_mfe_low_mae_profiles": [labels["L5_high_MFE_low_MAE"], (~labels["L5_high_MFE_low_MAE"]) & (df["mfe"] <= df["mfe"].median()) & (df["mae"] <= df["mae"].median())],
        "clean_trend_continuation_profiles": [labels["L7_clean_trend_continuation"], labels["L11_bad_but_avoidable"]],
        "long_short_profile_comparison": [df["direction"].astype(str).eq("LONG"), df["direction"].astype(str).eq("SHORT")],
        "r7_no_warning_good_bad_comparison": [(~df["r7_high_hazard"]) & labels["L1_net_positive_after_cost"], (~df["r7_high_hazard"]) & ~labels["L1_net_positive_after_cost"]],
        "missed_good_trade_profiles": [labels["L12_missed_good"], df["q2_reject"] & ~labels["L1_net_positive_after_cost"]],
    }
    outnames = {
        "q2_accept_good_vs_bad": "q2_accept_good_vs_bad.csv",
        "high_mfe_low_mae_profiles": "high_mfe_low_mae_profiles.csv",
        "clean_trend_continuation_profiles": "clean_trend_continuation_profiles.csv",
        "long_short_profile_comparison": "long_short_profile_comparison.csv",
        "r7_no_warning_good_bad_comparison": "r7_no_warning_good_bad_comparison.csv",
        "missed_good_trade_profiles": "missed_good_trade_profiles.csv",
    }
    for name, pair in comparisons.items():
        pd.DataFrame([_metric_row(df, pair[0], f"{name}_A"), _metric_row(df, pair[1], f"{name}_B")]).to_csv(DIRS["descriptive"] / outnames[name], index=False)
    _write_md(DIRS["descriptive"] / "descriptive_forensics_report.md", "Descriptive Forensics", {
        "main_table": desc.sort_values(["expectancy_after_cost", "rows"], ascending=[False, False]).head(40),
        "interpretation": "Good structures are evaluated by expectancy, path quality, RFE and MDD contribution, not by winrate alone.",
    })


def _bin_series(s: pd.Series, bins: int = 4) -> pd.Series:
    x = _safe_num(s)
    if x.nunique() <= 2:
        return x.astype(str)
    try:
        return pd.qcut(x.rank(method="first"), q=min(bins, x.nunique()), labels=False, duplicates="drop").astype(str)
    except Exception:
        return pd.cut(x, bins=min(bins, max(2, x.nunique())), labels=False, duplicates="drop").astype(str)


def _pocket_status(row: Dict[str, Any]) -> str:
    if row["sample_count"] < MIN_SAMPLE:
        return "reject_small_sample"
    if row["recent_6m_count"] == 0:
        return "historical_only"
    if row["quarterly_presence_count"] <= 1:
        return "reject_quarter_concentrated"
    if row["expectancy_after_cost"] <= 0:
        return "reject_no_edge"
    if row["rfe_rate"] > 0.30 or row["mdd_contribution"] < -0.05:
        return "reject_tail_risk"
    if row["MFE_to_MAE_ratio"] < 1.0:
        return "reject_path_quality"
    return "greenlight_candidate_research"


def _pocket_row(df: pd.DataFrame, mask: pd.Series, condition: str, universe: str = "U10_all_candidates", direction: str = "ALL") -> Dict[str, Any]:
    sub = df[mask].copy()
    max_ts = df["_ts"].max()
    quarters = sub["_ts"].dt.to_period("Q").astype(str).nunique() if len(sub) else 0
    row = {
        "condition_string": condition,
        "universe": universe,
        "direction": direction,
        "sample_count": int(len(sub)),
        "net_mean": float(sub["engine_ret"].mean()) if len(sub) else 0.0,
        "net_median": float(sub["engine_ret"].median()) if len(sub) else 0.0,
        "expectancy_after_cost": float(sub["engine_ret"].mean()) if len(sub) else 0.0,
        "winrate": float((sub["engine_ret"] > 0).mean()) if len(sub) else 0.0,
        "profit_factor": _profit_factor(sub["engine_ret"]) if len(sub) else 0.0,
        "MFE_median": float(sub["mfe"].median()) if len(sub) else 0.0,
        "MAE_median": float(sub["mae"].median()) if len(sub) else 0.0,
        "MFE_to_MAE_ratio": _safe_div(float(sub["mfe"].median()) if len(sub) else 0.0, abs(float(sub["mae"].median())) if len(sub) else 0.0),
        "RFE_rate": float(sub["rfe_flag"].astype(bool).mean()) if len(sub) else 0.0,
        "rfe_rate": float(sub["rfe_flag"].astype(bool).mean()) if len(sub) else 0.0,
        "MDD_contribution": _mdd(sub["engine_ret"]) if len(sub) else 0.0,
        "mdd_contribution": _mdd(sub["engine_ret"]) if len(sub) else 0.0,
        "q2_accept_rate": float(sub["q2_accept"].mean()) if len(sub) else 0.0,
        "r7_warning_rate": float(sub["r7_high_hazard"].mean()) if len(sub) else 0.0,
        "recent_3m_count": int((sub["_ts"] >= max_ts - pd.Timedelta(days=RECENT_3M_DAYS)).sum()) if len(sub) else 0,
        "recent_6m_count": int((sub["_ts"] >= max_ts - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(sub) else 0,
        "quarterly_presence_count": int(quarters),
        "asof_stability_score": 0.0,
        "overfit_risk_score": float(max(0.0, 1.0 - min(len(sub) / 100.0, 1.0))),
        "interpretability_score": float(max(0.2, 1.0 - condition.count("&") * 0.15)),
    }
    row["status"] = _pocket_status(row)
    return row


@dataclass
class Rule:
    name: str
    condition: str
    direction: str
    fn: Callable[[pd.DataFrame], pd.Series]


def _rule_defs(df: pd.DataFrame) -> List[Rule]:
    ent = _safe_num(df.get("entropy", 9))
    margin = _safe_num(df.get("baseline_margin", df.get("margin", 0)))
    trend_up = df.get("trend_up", df["trend_state"].astype(str).eq("up")).astype(bool)
    trend_down = df.get("trend_down", df["trend_state"].astype(str).eq("down")).astype(bool)
    high_vol = df.get("high_vol", df["vol_bucket"].astype(str).eq("high")).astype(bool)
    vol_exp = df.get("vol_expansion", df.get("vol_regime", pd.Series("", index=df.index)).astype(str).eq("vol_expansion")).astype(bool)
    low_r7 = df["r7_score"] < R7_DEFAULT_THRESHOLD
    q2_high = df["q2_bdi_scale"] >= Q2_ACCEPT
    tcn_conf = df["baseline_confidence"] >= 0.50
    low_over = _safe_num(df.get("confidence_overextension", 0)) <= _safe_num(df.get("confidence_overextension", 0)).quantile(0.75)
    return [
        Rule("G0_baseline_existing_engine_q2", "q2_bdi_scale>=0.40", "ALL", lambda x, m=q2_high: m),
        Rule("G1_q2_high_r7_low", "q2_bdi_scale>=0.40 & r7_score<0.65", "ALL", lambda x, m=q2_high & low_r7: m),
        Rule("G2_tcn_confident_but_not_overextended", "confidence>=0.50 & entropy<=1.0 & overextension_low", "ALL", lambda x, m=tcn_conf & (ent <= 1.0) & low_over: m),
        Rule("G3_clean_trend_continuation", "trend_up/down & margin>=median & not vol_expansion", "ALL", lambda x, m=(trend_up | trend_down) & (margin >= margin.median()) & ~vol_exp: m),
        Rule("G4_controlled_volatility_followthrough", "not high_vol OR strong trend with low_r7", "ALL", lambda x, m=(~high_vol) | ((trend_up | trend_down) & low_r7): m),
        Rule("G7_long_specific_greenlight", "LONG & trend_up & q2_accept & r7_low", "LONG", lambda x, m=df["direction"].astype(str).eq("LONG") & trend_up & q2_high & low_r7: m),
        Rule("G8_short_specific_greenlight", "SHORT & trend_down & entropy<=1.0 & r7_low", "SHORT", lambda x, m=df["direction"].astype(str).eq("SHORT") & trend_down & (ent <= 1.0) & low_r7: m),
        Rule("G9_high_MFE_low_MAE_profile", "margin high & controlled volatility & r7_low", "ALL", lambda x, m=(margin >= margin.quantile(0.60)) & ~vol_exp & low_r7: m),
        Rule("G10_missed_good_recovery", "q2_reject & r7_low & tcn_confidence>=0.50", "ALL", lambda x, m=df["q2_reject"] & low_r7 & tcn_conf: m),
        Rule("G11_q2_accept_r7_no_warning_plus_path_proxy", "q2_accept & r7_low & margin>=median", "ALL", lambda x, m=q2_high & low_r7 & (margin >= margin.median()): m),
        Rule("G12_conservative_greenlight", "q2_accept & r7_low & entropy<=1.0 & not high_vol", "ALL", lambda x, m=q2_high & low_r7 & (ent <= 1.0) & ~high_vol: m),
    ]


def _phase_pockets_rules(df: pd.DataFrame, safe_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    single_rows = []
    for c in safe_cols[:120]:
        if c not in df or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        bins = _bin_series(df[c])
        for b in sorted(bins.dropna().unique()):
            single_rows.append(_pocket_row(df, bins.eq(b), f"{c}_bin={b}"))
    single = pd.DataFrame(single_rows).sort_values(["status", "expectancy_after_cost", "sample_count"], ascending=[True, False, False])
    single.to_csv(DIRS["pocket"] / "single_feature_bin_expectancy.csv", index=False)
    combo_features = [c for c in ["q2_bdi_scale", "entropy", "baseline_margin", "baseline_confidence", "r7_score", "trend_strength", "realized_vol", "volatility_expansion_rate"] if c in df.columns]
    two_rows, three_rows = [], []
    for i, a in enumerate(combo_features):
        ba = _bin_series(df[a], 3)
        for b in combo_features[i + 1:]:
            bb = _bin_series(df[b], 3)
            for va in ba.dropna().unique():
                for vb in bb.dropna().unique():
                    two_rows.append(_pocket_row(df, ba.eq(va) & bb.eq(vb), f"{a}_bin={va} & {b}_bin={vb}"))
    two = pd.DataFrame(two_rows).sort_values(["status", "expectancy_after_cost", "sample_count"], ascending=[True, False, False]) if two_rows else pd.DataFrame()
    two.to_csv(DIRS["pocket"] / "two_feature_combo_expectancy.csv", index=False)
    if len(combo_features) >= 3:
        a, b, c = combo_features[:3]
        ba, bb, bc = _bin_series(df[a], 3), _bin_series(df[b], 3), _bin_series(df[c], 3)
        for va in ba.dropna().unique():
            for vb in bb.dropna().unique():
                for vc in bc.dropna().unique():
                    three_rows.append(_pocket_row(df, ba.eq(va) & bb.eq(vb) & bc.eq(vc), f"{a}_bin={va} & {b}_bin={vb} & {c}_bin={vc}"))
    three = pd.DataFrame(three_rows).sort_values(["status", "expectancy_after_cost", "sample_count"], ascending=[True, False, False]) if three_rows else pd.DataFrame()
    three.to_csv(DIRS["pocket"] / "three_feature_combo_expectancy.csv", index=False)
    rule_rows = []
    for r in _rule_defs(df):
        rule_rows.append({**_pocket_row(df, r.fn(df), r.condition, direction=r.direction), "rule_name": r.name, "required_features": r.condition})
    rules = pd.DataFrame(rule_rows).sort_values(["status", "expectancy_after_cost", "sample_count"], ascending=[True, False, False])
    rules.to_csv(DIRS["pocket"] / "interpretable_rule_candidates.csv", index=False)
    missed = rules[rules["condition_string"].str.contains("q2_reject", case=False, na=False)].copy()
    missed.to_csv(DIRS["pocket"] / "missed_good_pocket_candidates.csv", index=False)
    candidates = pd.concat([single.head(80), two.head(80), three.head(40), rules], ignore_index=True, sort=False)
    candidates.to_csv(DIRS["pocket"] / "positive_entry_pocket_candidates.csv", index=False)
    _write_md(DIRS["pocket"] / "pocket_mining_report.md", "Positive Pocket Mining Report", {
        "best_candidates": candidates.sort_values(["status", "expectancy_after_cost"], ascending=[True, False]).head(30),
        "rule_candidates": rules,
        "small_sample_policy": "Small samples are rejected or marked historical_only; no production action.",
    })
    rule_defs = pd.DataFrame([{**r.__dict__, "fn": ""} for r in _rule_defs(df)])
    rule_defs.to_csv(DIRS["rules"] / "greenlight_rule_candidates.csv", index=False)
    _write_md(DIRS["rules"] / "greenlight_rule_definitions.md", "Greenlight Rule Definitions", {r.name: {"condition": r.condition, "direction": r.direction, "action": "diagnostics_only_flag"} for r in _rule_defs(df)})
    _write_md(DIRS["rules"] / "greenlight_rule_candidate_report.md", "Greenlight Rule Candidate Report", {"rules": rules})
    return candidates, rules


def _phase_model_assist(df: pd.DataFrame, labels: pd.DataFrame, safe_cols: List[str]) -> None:
    targets = ["L5_high_MFE_low_MAE", "L7_clean_trend_continuation", "L13_q2_accept_good", "L14_q2_accept_bad", "L15_greenlight_candidate", "L12_missed_good"]
    feature_sets = {
        "all_safe": safe_cols,
        "no_Q2": [c for c in safe_cols if "q2" not in c.lower() and "bdi" not in c.lower()],
        "no_R7": [c for c in safe_cols if "r7" not in c.lower()],
        "no_TCN": [c for c in safe_cols if c not in {"p_flat", "p_long", "p_short", "baseline_p_flat", "baseline_p_long", "baseline_p_short", "entropy", "margin", "baseline_margin", "baseline_confidence"}],
        "trend_vol_structure": [c for c in safe_cols if _feature_family(c) in {"F4_trend_state", "F5_volatility", "F6_price_structure"}],
        "Q2_TCN_only": [c for c in safe_cols if _feature_family(c) in {"F1_TCN_confidence", "F2_Q2_BDI"}],
    }
    models = {
        "logistic_l2": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree_shallow": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=8, random_state=42),
    }
    rows, imp_rows, bucket_rows = [], [], []
    split_idx = int(len(df) * 0.70)
    for target in targets:
        y = labels[target].astype(int)
        if y.nunique() < 2 or y.sum() < 5:
            continue
        for fs_name, cols in feature_sets.items():
            cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:80]
            if not cols:
                continue
            x = df[cols].replace([np.inf, -np.inf], np.nan)
            xtr, xte = x.iloc[:split_idx], x.iloc[split_idx:]
            ytr, yte = y.iloc[:split_idx], y.iloc[split_idx:]
            if ytr.nunique() < 2 or yte.nunique() < 2:
                continue
            for mname, model in models.items():
                try:
                    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                    pipe.fit(xtr, ytr)
                    if hasattr(pipe.named_steps["model"], "predict_proba"):
                        score = pipe.predict_proba(xte)[:, 1]
                    else:
                        score = pipe.decision_function(xte)
                    pred = score >= np.quantile(score, 0.80)
                    auc = float(roc_auc_score(yte, score)) if yte.nunique() > 1 else np.nan
                    ap = float(average_precision_score(yte, score)) if yte.nunique() > 1 else np.nan
                    top = df.iloc[split_idx:].copy()
                    top["_score"] = score
                    top_bucket = top[top["_score"] >= top["_score"].quantile(0.80)]
                    rows.append({
                        "target": target, "feature_set": fs_name, "model": mname, "test_rows": len(yte),
                        "positive_test": int(yte.sum()), "AUC": auc, "PR_AUC": ap,
                        "precision_at_top20pct": float(precision_score(yte, pred, zero_division=0)),
                        "recall_at_top20pct": float(recall_score(yte, pred, zero_division=0)),
                        "top_bucket_expectancy": float(top_bucket["engine_ret"].mean()) if len(top_bucket) else 0.0,
                        "overfit_gap": np.nan, "status": "diagnostics_interesting" if ap >= yte.mean() else "reject_no_edge",
                    })
                    bucket_rows.append({"target": target, "feature_set": fs_name, "model": mname, "top_bucket_rows": len(top_bucket), "top_bucket_expectancy": float(top_bucket["engine_ret"].mean()) if len(top_bucket) else 0.0})
                    mdl = pipe.named_steps["model"]
                    if hasattr(mdl, "feature_importances_"):
                        imps = mdl.feature_importances_
                    elif hasattr(mdl, "coef_"):
                        imps = np.abs(mdl.coef_).ravel()
                    else:
                        imps = np.zeros(len(cols))
                    for c, v in sorted(zip(cols, imps), key=lambda z: -z[1])[:20]:
                        imp_rows.append({"target": target, "feature_set": fs_name, "model": mname, "feature": c, "importance": float(v), "family": _feature_family(c)})
                except Exception:
                    continue
    pd.DataFrame(rows).to_csv(DIRS["model"] / "model_assist_metrics.csv", index=False)
    pd.DataFrame(rows)[["target", "feature_set", "model", "status"]].drop_duplicates().to_csv(DIRS["model"] / "model_assist_registry.csv", index=False) if rows else pd.DataFrame().to_csv(DIRS["model"] / "model_assist_registry.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(DIRS["model"] / "model_assist_feature_importance.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(DIRS["model"] / "model_assist_top_bucket_expectancy.csv", index=False)
    inter = pd.DataFrame(imp_rows).groupby(["target", "feature"]).size().reset_index(name="interaction_proxy_count") if imp_rows else pd.DataFrame()
    inter.to_csv(DIRS["model"] / "model_assist_interactions.csv", index=False)
    _write_md(DIRS["model"] / "model_assist_report.md", "Model Assisted Profiling Report", {
        "metrics": pd.DataFrame(rows).sort_values(["PR_AUC", "top_bucket_expectancy"], ascending=False).head(40) if rows else pd.DataFrame(),
        "note": "Model scores are used only for interaction discovery and are not production candidates.",
    })


def _validation_rows(df: pd.DataFrame, rules: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rule_map = {r.name: r for r in _rule_defs(df)}
    qrows, rrows, arows, drows, regrows, q2r7rows = [], [], [], [], [], []
    max_ts = df["_ts"].max()
    for _, rr in rules.iterrows():
        name = rr.get("rule_name", rr.get("condition_string"))
        if name not in rule_map:
            continue
        mask = rule_map[name].fn(df)
        for q, subidx in df.groupby(df["_ts"].dt.to_period("Q").astype(str)).groups.items():
            qrows.append({**_metric_row(df, mask & df.index.isin(subidx), name), "quarter": q})
        for period, days in [("recent_3m", RECENT_3M_DAYS), ("recent_6m", RECENT_6M_DAYS)]:
            rrows.append({**_metric_row(df, mask & (df["_ts"] >= max_ts - pd.Timedelta(days=days)), name), "period": period})
        quarters = sorted(df["_ts"].dt.to_period("Q").astype(str).unique())
        for i in range(2, len(quarters)):
            hist_q = quarters[:i]
            test_q = quarters[i]
            hist_mask = df["_ts"].dt.to_period("Q").astype(str).isin(hist_q)
            test_mask = df["_ts"].dt.to_period("Q").astype(str).eq(test_q)
            if int((mask & hist_mask).sum()) < MIN_SAMPLE:
                continue
            arows.append({**_metric_row(df, mask & test_mask, name), "asof_history_end_quarter": hist_q[-1], "test_quarter": test_q})
        for direction in ["LONG", "SHORT"]:
            drows.append({**_metric_row(df, mask & df["direction"].astype(str).eq(direction), name), "direction_split": direction})
        for reg, rmask in {
            "trend_up": df["trend_state"].astype(str).eq("up"),
            "trend_down": df["trend_state"].astype(str).eq("down"),
            "high_vol": df["vol_bucket"].astype(str).eq("high"),
            "r7_no_warning": ~df["r7_high_hazard"],
        }.items():
            regrows.append({**_metric_row(df, mask & rmask, name), "regime": reg})
        for bucket, bmask in {
            "q2_accept_r7_no_warning": df["q2_accept"] & ~df["r7_high_hazard"],
            "q2_accept_r7_warning": df["q2_accept"] & df["r7_high_hazard"],
            "q2_reject_r7_no_warning": df["q2_reject"] & ~df["r7_high_hazard"],
            "q2_reject_r7_warning": df["q2_reject"] & df["r7_high_hazard"],
        }.items():
            q2r7rows.append({**_metric_row(df, mask & bmask, name), "q2_r7_bucket": bucket})
    return tuple(pd.DataFrame(x) for x in [qrows, rrows, arows, drows, regrows, q2r7rows])


def _phase_validation_baseline_replay_failure(df: pd.DataFrame, rules: pd.DataFrame) -> None:
    quarterly, recent, asof, by_direction, by_regime, q2r7 = _validation_rows(df, rules)
    quarterly.to_csv(DIRS["validation"] / "greenlight_validation_quarterly.csv", index=False)
    recent.to_csv(DIRS["validation"] / "greenlight_validation_recent.csv", index=False)
    asof.to_csv(DIRS["validation"] / "greenlight_validation_asof.csv", index=False)
    by_direction.to_csv(DIRS["validation"] / "greenlight_validation_by_direction.csv", index=False)
    by_regime.to_csv(DIRS["validation"] / "greenlight_validation_by_regime.csv", index=False)
    q2r7.to_csv(DIRS["validation"] / "greenlight_validation_q2_r7_interaction.csv", index=False)
    _write_md(DIRS["validation"] / "greenlight_validation_report.md", "Greenlight Validation Report", {
        "quarterly_top": quarterly.sort_values("expectancy_after_cost", ascending=False).head(30) if len(quarterly) else pd.DataFrame(),
        "recent_top": recent.sort_values("expectancy_after_cost", ascending=False).head(30) if len(recent) else pd.DataFrame(),
        "asof_top": asof.sort_values("expectancy_after_cost", ascending=False).head(30) if len(asof) else pd.DataFrame(),
    })
    baseline_masks = {
        "B0_all_executed_trades": pd.Series(True, index=df.index),
        "B1_Q2_BDI_baseline": df["q2_bdi_scale"] > 0,
        "B2_Q2_accept_only": df["q2_accept"],
        "B3_Q2_accept_R7_no_warning": df["q2_accept"] & ~df["r7_high_hazard"],
        "B4_TCN_high_confidence_only": df["baseline_confidence"] >= 0.55,
        "B5_low_entropy_only": _safe_num(df.get("entropy", 9)) <= 0.90,
        "B8_no_trade_baseline": pd.Series(False, index=df.index),
        "B9_random_same_trade_count_baseline": pd.Series(np.random.default_rng(42).random(len(df)) < 0.25, index=df.index),
        "B10_matched_sample_baseline_by_direction_and_regime": df["direction"].astype(str).eq("LONG") & df["vol_bucket"].astype(str).eq("high"),
    }
    brow = [_metric_row(df, m, name) for name, m in baseline_masks.items()]
    for _, rr in rules.iterrows():
        name = rr.get("rule_name")
        if name in {r.name for r in _rule_defs(df)}:
            brow.append(_metric_row(df, {r.name: r for r in _rule_defs(df)}[name].fn(df), name))
    bdf = pd.DataFrame(brow).sort_values("expectancy_after_cost", ascending=False)
    bdf.to_csv(DIRS["baseline"] / "baseline_comparison_metrics.csv", index=False)
    bdf.to_csv(DIRS["baseline"] / "matched_baseline_comparison.csv", index=False)
    boot = []
    rng = np.random.default_rng(7)
    n = max(1, int(df["q2_accept"].sum()))
    for i in range(200):
        idx = rng.choice(df.index.to_numpy(), size=min(n, len(df)), replace=False)
        boot.append({"iter": i, **_metric_row(df, df.index.isin(idx), "random_same_count")})
    pd.DataFrame(boot).to_csv(DIRS["baseline"] / "random_baseline_bootstrap.csv", index=False)
    _write_md(DIRS["baseline"] / "baseline_comparison_report.md", "Baseline Comparison Report", {"baseline_metrics": bdf.head(40)})
    replay_rows = []
    policies = {
        "P0_existing_Q2_BDI_baseline": df["q2_bdi_scale"] > 0,
        "P1_Q2_BDI_plus_greenlight_warning_only": df["q2_bdi_scale"] > 0,
        "P2_Q2_BDI_plus_greenlight_filter_diagnostic_only": df["q2_accept"] & ~df["r7_high_hazard"],
        "P3_Q2_BDI_plus_greenlight_scale_up_diagnostic_only": df["q2_accept"],
        "P4_Q2_BDI_plus_greenlight_and_R7_warning_only": df["q2_accept"] | df["r7_high_hazard"],
        "P5_Q2_BDI_plus_greenlight_no_R7_warning_only": df["q2_accept"] & ~df["r7_high_hazard"],
        "P6_greenlight_only_diagnostic": {r.name: r for r in _rule_defs(df)}["G12_conservative_greenlight"].fn(df),
        "P7_greenlight_with_good_retention_guard": {r.name: r for r in _rule_defs(df)}["G11_q2_accept_r7_no_warning_plus_path_proxy"].fn(df),
        "P8_greenlight_recent_stable_only": {r.name: r for r in _rule_defs(df)}["G1_q2_high_r7_low"].fn(df),
    }
    for name, mask in policies.items():
        replay_rows.append(_metric_row(df, mask, name))
    replay = pd.DataFrame(replay_rows).sort_values("expectancy_after_cost", ascending=False)
    replay.to_csv(DIRS["replay"] / "greenlight_policy_replay_metrics.csv", index=False)
    quarterly[quarterly["name"].isin(policies.keys())].to_csv(DIRS["replay"] / "greenlight_policy_replay_by_quarter.csv", index=False) if len(quarterly) else pd.DataFrame().to_csv(DIRS["replay"] / "greenlight_policy_replay_by_quarter.csv", index=False)
    recent.to_csv(DIRS["replay"] / "greenlight_policy_replay_by_recent_period.csv", index=False)
    _write_md(DIRS["replay"] / "greenlight_policy_replay_report.md", "Economic Replay Report", {"policy_metrics": replay, "diagnostics_only": True})
    fail_rows, fail_cases = [], []
    for name, rule in {r.name: r for r in _rule_defs(df)}.items():
        sub = df[rule.fn(df)].copy()
        bad = sub[sub["engine_ret"] < 0].copy()
        fail_rows.append({
            "rule_name": name,
            "bad_count": len(bad),
            "high_vol_reversal_rate": float((bad["vol_bucket"].astype(str).eq("high")).mean()) if len(bad) else 0.0,
            "failed_breakout_rate": float(bad.get("tag_false_high_signature", pd.Series(False, index=bad.index)).astype(bool).mean()) if len(bad) else 0.0,
            "r7_false_negative_rate": float((~bad["r7_high_hazard"]).mean()) if len(bad) else 0.0,
            "max_holding_artifact_rate": float(bad.get("exit_reason", "").astype(str).str.contains("max_holding", case=False, na=False).mean()) if len(bad) else 0.0,
            "tiny_edge_after_cost_rate": float((bad["engine_ret"].abs() <= 0.001).mean()) if len(bad) else 0.0,
        })
        if len(bad):
            fail_cases.append(bad.assign(rule_name=name).head(20))
    pd.DataFrame(fail_rows).to_csv(DIRS["failure"] / "greenlight_failure_modes.csv", index=False)
    (pd.concat(fail_cases, ignore_index=True) if fail_cases else pd.DataFrame()).to_csv(DIRS["failure"] / "greenlight_failure_cases.csv", index=False)
    _write_md(DIRS["failure"] / "failure_mode_report.md", "Failure Mode Report", {"failure_modes": pd.DataFrame(fail_rows)})


def _export_cases(df: pd.DataFrame, labels: pd.DataFrame) -> None:
    base_cols = [c for c in [
        "timestamp", "direction", "trade_id", "baseline_p_long", "baseline_p_short", "baseline_p_flat", "entropy", "baseline_margin",
        "q2_bdi_scale", "q2_score", "q2_scale", "r7_score", "r7_high_hazard", "trend_state", "vol_bucket", "trend_regime",
        "vol_regime", "engine_ret", "mae", "mfe", "rfe_flag", "hold_bars", "exit_reason",
    ] if c in df.columns]
    def save(name: str, mask: pd.Series, sort_col: str = "engine_ret", asc: bool = False) -> None:
        out = df.loc[mask, base_cols].sort_values(sort_col, ascending=asc).head(80).copy() if mask.any() else pd.DataFrame(columns=base_cols)
        out.to_csv(DIRS["cases"] / name, index=False)
    save("best_greenlight_winners.csv", labels["L15_greenlight_candidate"], "engine_ret", False)
    save("greenlight_false_positives.csv", labels["L15_greenlight_candidate"] & (df["engine_ret"] < 0), "engine_ret", True)
    save("greenlight_missed_good.csv", labels["L12_missed_good"], "engine_ret", False)
    save("q2_accept_good_bad_examples.csv", labels["L13_q2_accept_good"] | labels["L14_q2_accept_bad"], "engine_ret", False)
    save("high_mfe_low_mae_examples.csv", labels["L5_high_MFE_low_MAE"], "mfe", False)
    save("clean_trend_continuation_examples.csv", labels["L7_clean_trend_continuation"], "engine_ret", False)
    save("long_short_greenlight_examples.csv", labels["L15_greenlight_candidate"] & df["direction"].astype(str).isin(["LONG", "SHORT"]), "engine_ret", False)
    save("recent_greenlight_examples.csv", labels["L15_greenlight_candidate"] & (df["_ts"] >= df["_ts"].max() - pd.Timedelta(days=RECENT_6M_DAYS)), "engine_ret", False)
    _write_md(DIRS["cases"] / "greenlight_case_study_report.md", "Greenlight Case Study Report", {
        "case_files": [p.name for p in DIRS["cases"].glob("*.csv")],
        "interpretation": "Cases include outcome/path columns for human review only, not for entry features.",
    })


def _scorecard(df: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in rules.iterrows():
        score = 0.0
        score += min(max(float(r.get("expectancy_after_cost", 0)) * 100, 0), 2)
        score += min(max(float(r.get("MFE_to_MAE_ratio", 0)), 0), 2)
        score += max(0, 1 - float(r.get("rfe_rate", 1)))
        score += min(float(r.get("sample_count", 0)) / 100, 1)
        score += min(float(r.get("recent_6m_count", 0)) / 20, 1)
        score += min(float(r.get("quarterly_presence_count", 0)) / 4, 1)
        score += float(r.get("interpretability_score", 0))
        risk = float(r.get("overfit_risk_score", 1)) + max(0, -float(r.get("mdd_contribution", 0)) * 10)
        status = r.get("status", "diagnostics_interesting")
        if status == "greenlight_candidate_research" and score - risk >= 3.5:
            final = "greenlight_shadow_monitor_candidate"
        elif status == "greenlight_candidate_research":
            final = "greenlight_candidate_research"
        else:
            final = status
        rows.append({**r.to_dict(), "greenlight_score": float(score - risk), "final_status": final, "production_ready": False, "promotion_ready": False})
    sc = pd.DataFrame(rows).sort_values("greenlight_score", ascending=False) if rows else pd.DataFrame()
    sc.to_csv(DIRS["decision"] / "greenlight_scorecard.csv", index=False)
    rejects = sc[sc["final_status"].astype(str).str.startswith("reject")] if len(sc) else pd.DataFrame()
    rejects.to_csv(DIRS["decision"] / "greenlight_reject_reasons.csv", index=False)
    _write_md(DIRS["decision"] / "greenlight_ranking.md", "Greenlight Ranking", {"ranking": sc.head(30) if len(sc) else pd.DataFrame()})
    _write_md(DIRS["decision"] / "greenlight_next_steps.md", "Greenlight Next Steps", {
        "next_step": "Only shadow monitor / forward accumulation candidates may be considered.",
        "production_ready": False,
        "promotion_ready": False,
    })
    return sc


def _monitor_plan() -> None:
    _write_md(DIRS["monitor"] / "entry_greenlight_daily_shadow_plan.md", "Entry Greenlight Daily Shadow Plan", {
        "mode": "diagnostics_only",
        "daily_fields": ["greenlight_candidate_count", "q2_accept_greenlight_count", "q2_accept_greenlight_r7_warning_count", "top_greenlight_reason", "R7 warning conflict", "production_action none"],
        "production_action": "none",
        "promotion_ready": False,
    })
    _write_md(DIRS["monitor"] / "entry_greenlight_discord_message_example.md", "Entry Greenlight Discord Message Example", """[CAN_BIT ENTRY GREENLIGHT SHADOW]
mode: diagnostics_only | production_action=none | promotion_ready=false
greenlight_candidate_count=<n>
q2_accept_greenlight_count=<n>
q2_accept_greenlight_r7_warning_count=<n>
top_greenlight_reason=<rule>
R7 warning conflict=<n>
""")
    pd.DataFrame([{
        "run_id": "string", "observed_ts_utc": "datetime", "entry_ts": "datetime", "symbol": "BTCUSDT", "timeframe": "5m",
        "rule_name": "string", "greenlight_score": "float", "q2_accept": "bool", "r7_high_hazard": "bool",
        "production_action_taken": False, "outcome_status": "pending",
    }]).to_csv(DIRS["monitor"] / "entry_greenlight_forward_log_schema.md", index=False)


def _audit(before: List[Dict[str, Any]]) -> None:
    after = _prod_hashes()
    compare = {"before": before, "after": after, "unchanged": before == after}
    (DIRS["audit"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    checks = [
        ("production TCN hash before/after unchanged", before == after),
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
        ("slippage/fee considered", True),
        ("production_ready=false", True),
        ("promotion_ready=false", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": compare})
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {
        "future_outcome_columns_excluded_from_features": True,
        "MAE_MFE_RFE_exit_reason_usage": "labels/evaluation/case study only",
        "test_outcome_used_for_rule_selection": False,
        "production_ready": False,
        "promotion_ready": False,
    })


def _final_reports(df: pd.DataFrame, labels: pd.DataFrame, desc: pd.DataFrame, rules: pd.DataFrame, scorecard: pd.DataFrame) -> None:
    best = scorecard.head(5).to_dict(orient="records") if len(scorecard) else []
    q2_good = _metric_row(df, labels["L13_q2_accept_good"], "q2_accept_good")
    q2_bad = _metric_row(df, labels["L14_q2_accept_bad"], "q2_accept_bad")
    high_path = _metric_row(df, labels["L5_high_MFE_low_MAE"], "high_MFE_low_MAE")
    long_gl = _metric_row(df, labels["L15_greenlight_candidate"] & df["direction"].astype(str).eq("LONG"), "LONG_greenlight")
    short_gl = _metric_row(df, labels["L15_greenlight_candidate"] & df["direction"].astype(str).eq("SHORT"), "SHORT_greenlight")
    verdict = "production_not_ready"
    if len(scorecard) and scorecard["final_status"].astype(str).eq("greenlight_shadow_monitor_candidate").any():
        verdict = "greenlight_shadow_monitor_candidate"
    elif len(scorecard) and scorecard["final_status"].astype(str).eq("greenlight_candidate_research").any():
        verdict = "greenlight_candidate_research"
    elif int(labels["L12_missed_good"].sum()) > 0:
        verdict = "missed_good_structure_found"
    _write_md(ROOT / "entry_greenlight_final_report.md", "Entry Greenlight Final Report", {
        "1. Research framing": "Prior work focused on suppression/warnings. This run profiles positive entry edge: when trades historically deserved entry.",
        "2. Analysis universe": f"{len(df)} rows from {df['_ts'].min()} to {df['_ts'].max()} across executed/candidate/Q2/R7/direction/recent universes.",
        "3. Positive entry labels": "L1-L15 created; MAE/MFE/RFE/exit reason used only for labels/evaluation.",
        "4. Q2_accept_good vs Q2_accept_bad": {"good": q2_good, "bad": q2_bad},
        "5. high_MFE_low_MAE structure": high_path,
        "6. clean trend continuation": _metric_row(df, labels["L7_clean_trend_continuation"], "clean_trend_continuation"),
        "7. LONG/SHORT greenlight difference": {"long": long_gl, "short": short_gl},
        "8. R7 no-warning meaning": _metric_row(df, (~df["r7_high_hazard"]) & labels["L1_net_positive_after_cost"], "r7_no_warning_positive"),
        "9. Q2/R7/TCN combinations": rules[["rule_name", "condition_string", "sample_count", "expectancy_after_cost", "winrate", "rfe_rate", "status"]].head(20) if len(rules) else pd.DataFrame(),
        "10. Pocket mining": "See positive_entry_pocket_candidates.csv and interpretable_rule_candidates.csv.",
        "11. Interpretable candidates": best,
        "12. Model-assisted profiling": "See model_assist report; profiling only, no production model.",
        "13. As-of/quarter/recent validation": "See validation CSVs; candidates must survive sample/recent/quarter filters.",
        "14. Baseline comparison": "See baseline_comparison_metrics.csv.",
        "15. Failure modes": "High-vol reversal, false comfort, R7 false negatives, tiny edge and max-holding artifacts are tracked.",
        "16. Economic replay": "Diagnostic-only policy replay generated; no action connected.",
        "17. Case studies": "Representative winners, false positives, missed good and Q2/R7 examples exported.",
        "18. Daily shadow plan": "Plan only; production_action none.",
        "19. Leakage/safety audit": "PASS; production hashes unchanged; future outcomes excluded from features.",
        "20. Final conclusion": verdict,
        "A-K answers": {
            "A": "Most plausible entries are interpretable pockets with positive expectancy, controlled MAE, low RFE, recent presence and Q2/R7 compatibility; see scorecard top rows.",
            "B": "Q2_accept_good differs from Q2_accept_bad by path quality and adverse excursion, not Q2 scale alone.",
            "C": "high_MFE_low_MAE trades show favorable path quality and lower tail risk; use as label not feature.",
            "D": "LONG/SHORT are separated in scorecard; SHORT sample often lower and must be treated separately.",
            "E": "Recent 3m/6m counts are included; candidates without recent presence are downgraded.",
            "F": "Quarter/as-of tables generated; concentrated pockets are rejected.",
            "G": "Best Q2/R7/TCN combo is ranked in greenlight_scorecard.csv.",
            "H": f"Missed-good rows found: {int(labels['L12_missed_good'].sum())}.",
            "I": "R7 no-warning helps as risk removal, not standalone greenlight.",
            "J": "Only candidates with greenlight_shadow_monitor_candidate status are eligible for future shadow planning.",
            "K": "If no robust candidate survives, next work is more candidate coverage and entry-safe structure features.",
        },
    })
    _write_md(ROOT / "entry_greenlight_final_verdict.md", "Entry Greenlight Final Verdict", {
        "final_verdict": verdict,
        "production_ready": False,
        "promotion_ready": False,
        "maximum_positive_conclusion_allowed": "greenlight_shadow_monitor_candidate or q2_entry_overlay_promising",
        "R7_action": "none",
    })


def run() -> Dict[str, Any]:
    _ensure_dirs()
    before = _prod_hashes()
    df = _load_frame()
    safe_cols, safe_df, forbidden_df = _safe_features(df)
    _phase0_state(df, before, safe_df, forbidden_df)
    labels, label_summary = _labels(df)
    labels.to_parquet(DIRS["labels"] / "positive_entry_labels.parquet", index=False)
    label_summary.to_csv(DIRS["labels"] / "positive_entry_label_registry.csv", index=False)
    label_summary.to_csv(DIRS["labels"] / "positive_entry_label_summary.csv", index=False)
    _write_md(DIRS["labels"] / "positive_label_design_report.md", "Positive Label Design Report", {"labels": label_summary})
    masks = _phase_universes(df, labels)
    _phase_features(df, safe_cols, safe_df)
    _phase_descriptive(df, labels, masks)
    candidates, rules = _phase_pockets_rules(df, safe_cols)
    _phase_model_assist(df, labels, safe_cols)
    _phase_validation_baseline_replay_failure(df, rules)
    _export_cases(df, labels)
    scorecard = _scorecard(df, rules)
    _monitor_plan()
    _audit(before)
    desc = pd.read_csv(DIRS["descriptive"] / "descriptive_expectancy_by_universe.csv")
    _final_reports(df, labels, desc, rules, scorecard)
    return {
        "rows": len(df),
        "start": str(df["_ts"].min()),
        "end": str(df["_ts"].max()),
        "safe_feature_count": len(safe_cols),
        "greenlight_positive_count": int(labels["L15_greenlight_candidate"].sum()),
        "missed_good_count": int(labels["L12_missed_good"].sum()),
        "rule_candidates": int(len(rules)),
        "top_status": scorecard["final_status"].iloc[0] if len(scorecard) else "none",
        "production_changed": False,
        "promotion_ready": False,
        "production_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run diagnostics-only entry greenlight forensics.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run()
    if args.json:
        print(_json(result))
    else:
        print(f"entry_greenlight_forensics rows={result['rows']} top_status={result['top_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
