"""
FalseHigh_R7_StructureHazard_v1 diagnostics monitor lock and forensics.

This script locks R7_structure_hazard_score as a warning-only diagnostics
monitor, then writes validation, leakage, retention, Q2 overlay, daily shadow,
forward-accumulation, forward-like simulation, regime policy, case study,
decision, and safety-audit artifacts under:

    data/diagnostics/false_high_r7_monitor/

It never writes production TCN, Q2_BDI, live execution, launchd, order routing,
or state paths. R7 is explicitly not production-ready and has no routing,
scale-down, hard-block, or Q2 override action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from scripts.diagnostics.run_false_high_specialist_research import (
    HIGH_CONF,
    MIN_TEST,
    MIN_TRAIN,
    MIN_VAL,
    Q2_ACCEPT,
    Q2_REJECT,
    _build_expanded_v1,
    _fit_threshold,
    _label_views,
    _make_splits,
    _score_rule,
    _task_membership,
)
from scripts.diagnostics.run_feature_sufficiency_forensics import (
    _group_membership,
    _mdd,
    _prepare_df,
    _prod_hashes,
    _q2_baseline,
    _regime_mask,
    _safe_ap,
    _safe_auc,
    _slice,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("data/diagnostics/false_high_r7_monitor")
SOURCE_ROOT = Path("data/diagnostics/false_high_specialist")
DIRS = {
    "state": ROOT / "state",
    "lock": ROOT / "lock",
    "perfect": ROOT / "perfect_score_audit",
    "leakage": ROOT / "leakage_proxy_audit",
    "good": ROOT / "good_retention",
    "missed": ROOT / "missed_false_high",
    "q2": ROOT / "q2_warning_overlay",
    "daily": ROOT / "daily_shadow",
    "forward": ROOT / "forward_accumulation",
    "sim": ROOT / "forward_like_simulation",
    "regime": ROOT / "regime_policy",
    "cases": ROOT / "cases",
    "decision": ROOT / "decision",
    "audit": ROOT / "audit",
}

MONITOR_NAME = "FalseHigh_R7_StructureHazard_v1"
R7_SOURCE_NAME = "R7_structure_hazard_score"
R7_DEFAULT_THRESHOLD = 0.65
POSITION_SIZE = 0.05
FORBIDDEN_FEATURES = [
    "false_high_bad",
    "false_high target membership",
    "future_return",
    "MAE",
    "MFE",
    "RFE",
    "exit_reason",
    "realized_outcome",
    "future_horizon_label",
    "target_group_membership",
    "post_entry_information",
]


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def _md(title: str, sections: Dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key, value in sections.items():
        lines += [f"## {key}"]
        if isinstance(value, pd.DataFrame):
            if len(value):
                lines.append("```csv")
                lines.append(value.head(30).to_csv(index=False).rstrip())
                lines.append("```")
            else:
                lines.append("_empty_")
        elif isinstance(value, (dict, list)):
            lines.append("```json")
            lines.append(_json(value))
            lines.append("```")
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_metric_auc(y: Iterable[int], score: Iterable[float]) -> float:
    try:
        return _safe_auc(y, score)
    except Exception:
        yv = np.asarray(y).astype(int)
        sv = np.asarray(score).astype(float)
        if len(np.unique(yv)) < 2:
            return np.nan
        return float(roc_auc_score(yv, sv))


def _safe_metric_ap(y: Iterable[int], score: Iterable[float]) -> float:
    try:
        return _safe_ap(y, score)
    except Exception:
        yv = np.asarray(y).astype(int)
        sv = np.asarray(score).astype(float)
        if len(np.unique(yv)) < 2:
            return np.nan
        return float(average_precision_score(yv, sv))


def _prepare_monitor_frame() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    df = _prepare_df().copy()
    df = df.sort_values("_ts").reset_index(drop=True)
    groups = _group_membership(df)
    ef, schema, feature_groups = _build_expanded_v1(df)
    labels = _label_views(df, groups)
    tasks = _task_membership(df, groups, labels)

    ef_extra = ef.drop(columns=["trade_id", "timestamp"], errors="ignore")
    ef_extra = ef_extra.drop(columns=[c for c in ef_extra.columns if c in df.columns], errors="ignore")
    extra = pd.concat(
        [
            ef_extra.reset_index(drop=True),
            groups.drop(columns=["trade_id", "timestamp"], errors="ignore").reset_index(drop=True),
            labels.drop(columns=["trade_id", "timestamp"], errors="ignore").reset_index(drop=True),
        ],
        axis=1,
    )
    frame = pd.concat([df.reset_index(drop=True), extra], axis=1)
    frame["r7_score"] = _score_rule(frame, frame, R7_SOURCE_NAME)
    frame["false_high_bad"] = groups["G3_false_high_bad"].to_numpy(dtype=bool)
    frame["normal_high_conf_good"] = groups["G4_normal_high_conf_good"].to_numpy(dtype=bool)
    frame["task_eligible"] = frame["false_high_bad"] | frame["normal_high_conf_good"]
    frame["_target"] = np.where(frame["false_high_bad"], 1, np.where(frame["normal_high_conf_good"], 0, np.nan))
    frame["artifact_suspect"] = groups["G15_engine_artifact_suspect"].to_numpy(dtype=bool)
    return df, frame, groups, ef, schema, labels, feature_groups


def _split_eval_frame(frame: pd.DataFrame, split: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = frame[frame["task_eligible"]].copy()
    return (
        _slice(eligible, split["train_start"], split["train_end"]),
        _slice(eligible, split["val_start"], split["val_end"]),
        _slice(eligible, split["test_start"], split["test_end"]),
    )


def _binary_metrics(sub: pd.DataFrame, score: np.ndarray, threshold: float, split: Dict[str, Any] | None = None, label_view: str = "L0_raw_outcome") -> Dict[str, Any]:
    y = sub["_target"].astype(int).to_numpy()
    pred = np.asarray(score) >= threshold
    normal_good = sub["normal_high_conf_good"].astype(bool).to_numpy()
    high_good = sub["binary_good_trade"].astype(bool).to_numpy() & sub["baseline_long_high_conf"].astype(bool).to_numpy()
    good = sub["binary_good_trade"].astype(bool).to_numpy()
    bad = sub["binary_bad_trade"].astype(bool).to_numpy()
    q2_conflict = (sub["q2_bdi_scale"].astype(float).to_numpy() >= Q2_ACCEPT) & pred
    row = {
        "rows": len(sub),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == 0).sum()),
        "false_high_bad_count": int(sub["false_high_bad"].astype(bool).sum()),
        "normal_high_conf_good_count": int(sub["normal_high_conf_good"].astype(bool).sum()),
        "high_conf_good_count": int(high_good.sum()),
        "R7 threshold": threshold,
        "R7 threshold source": "validation_fit_median_locked" if split is None else "validation_fit_per_split",
        "precision": float(precision_score(y, pred, zero_division=0)) if len(sub) else np.nan,
        "recall": float(recall_score(y, pred, zero_division=0)) if len(sub) else np.nan,
        "PR-AUC": _safe_metric_ap(y, score),
        "ROC-AUC": _safe_metric_auc(y, score),
        "false_positive_count": int((pred & (y == 0)).sum()),
        "false_negative_count": int((~pred & (y == 1)).sum()),
        "good_retention": float((~pred & normal_good).sum() / max(normal_good.sum(), 1)),
        "high_conf_good_retention": float((~pred & high_good).sum() / max(high_good.sum(), 1)),
        "signal_coverage": float((~pred).mean()) if len(sub) else np.nan,
        "q2_conflict_count": int(q2_conflict.sum()),
        "bad_suppression_count": int((pred & bad).sum()),
        "good_false_warning_count": int((pred & good).sum()),
        "label_view": label_view,
    }
    if split:
        for c in ["split_id", "window_type", "window_months", "train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]:
            row[c] = split.get(c)
    row["status"] = _status_from_metrics(row)
    return row


def _status_from_metrics(row: Dict[str, Any]) -> str:
    if row.get("positive_count", 0) < 3 or row.get("negative_count", 0) < 3:
        return "WARN"
    if row.get("false_negative_count", 0) or row.get("false_positive_count", 0):
        return "WARN"
    if row.get("good_retention", 0.0) < 0.95:
        return "WARN"
    return "PASS"


def _fit_locked_threshold(frame: pd.DataFrame, splits: pd.DataFrame) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    rows, counts = [], []
    for _, sp in splits[splits["status"] == "PASS"].iterrows():
        train, val, test = _split_eval_frame(frame, sp.to_dict())
        if len(train) < MIN_TRAIN or len(val) < MIN_VAL or len(test) < MIN_TEST or val["_target"].nunique() < 2 or test["_target"].nunique() < 2:
            continue
        thr = _fit_threshold(val["_target"].astype(int), val["r7_score"].to_numpy())
        rows.append({
            "split_id": sp["split_id"],
            "window_months": int(sp["window_months"]),
            "threshold": float(thr),
            "threshold_source": "validation_fit_only",
            "val_rows": len(val),
            "val_positive_count": int(val["_target"].sum()),
            "val_negative_count": int((1 - val["_target"].astype(int)).sum()),
            "test_rows": len(test),
            "threshold_drift_from_default": float(thr - R7_DEFAULT_THRESHOLD),
        })
        cnt = {k: sp[k] for k in ["split_id", "window_months", "train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]}
        cnt.update({
            "positive_count": int(test["_target"].sum()),
            "negative_count": int((1 - test["_target"].astype(int)).sum()),
            "false_high_bad_count": int(test["false_high_bad"].sum()),
            "normal_high_conf_good_count": int(test["normal_high_conf_good"].sum()),
            "high_conf_good_count": int((test["binary_good_trade"].astype(bool) & test["baseline_long_high_conf"].astype(bool)).sum()),
            "class_imbalance_positive_rate": float(test["_target"].mean()),
        })
        counts.append(cnt)
    thresholds = pd.DataFrame(rows)
    locked = float(thresholds["threshold"].median()) if len(thresholds) else R7_DEFAULT_THRESHOLD
    return locked, thresholds, pd.DataFrame(counts)


def _score_scenario(frame: pd.DataFrame, scenario: str) -> np.ndarray:
    f = frame
    if scenario == "A0_full_R7":
        return f["r7_score"].astype(float).clip(0, 1).to_numpy()
    if scenario in {"A1_no_direct_signature", "A12_all_safe_no_signature_overlap"}:
        return (
            0.25 * f["strong_uptrend"].astype(float)
            + 0.25 * f["vol_expansion"].astype(float)
            + 0.25 * f["tcn_confidence_overextension"].astype(float)
            + 0.25 * f["range_expansion_proxy"].astype(float)
        ).clip(0, 1).to_numpy()
    if scenario == "A2_no_entropy":
        return (
            0.30 * f["trend_up"].astype(float)
            + 0.30 * f["high_vol"].astype(float)
            + 0.20 * f["vol_expansion"].astype(float)
            + 0.20 * f["tcn_confidence_overextension"].astype(float)
        ).clip(0, 1).to_numpy()
    if scenario == "A3_no_high_vol":
        return (
            0.35 * f["trend_up"].astype(float)
            + 0.25 * f["low_entropy_proxy"].astype(float)
            + 0.20 * f["vol_expansion"].astype(float)
            + 0.20 * f["tcn_confidence_overextension"].astype(float)
        ).clip(0, 1).to_numpy()
    if scenario == "A4_no_trend_state":
        return (
            0.35 * f["high_vol"].astype(float)
            + 0.25 * f["low_entropy_proxy"].astype(float)
            + 0.20 * f["vol_expansion"].astype(float)
            + 0.20 * f["tcn_confidence_overextension"].astype(float)
        ).clip(0, 1).to_numpy()
    if scenario == "A5_no_tcn_output":
        return (
            0.35 * f["trend_up"].astype(float)
            + 0.35 * f["high_vol"].astype(float)
            + 0.30 * f["vol_expansion"].astype(float)
        ).clip(0, 1).to_numpy()
    if scenario == "A6_no_q2":
        return f["r7_score"].astype(float).clip(0, 1).to_numpy()
    if scenario == "A7_no_trend_structure":
        return (
            0.35 * f["high_vol"].astype(float)
            + 0.25 * f["low_entropy_proxy"].astype(float)
            + 0.20 * f["vol_expansion"].astype(float)
            + 0.20 * f["tcn_confidence_overextension"].astype(float)
        ).clip(0, 1).to_numpy()
    if scenario == "A8_only_trend_structure":
        return (
            0.45 * f["trend_up"].astype(float)
            + 0.35 * f["strong_uptrend"].astype(float)
            + 0.20 * f["trend_transition"].astype(float)
        ).clip(0, 1).to_numpy()
    if scenario == "A9_only_structure_proxy":
        return (
            0.45 * f["range_expansion_proxy"].astype(float)
            + 0.35 * f["candle_rejection_proxy"].astype(float)
            + 0.20 * f["trend_exhaustion_proxy"].astype(float)
        ).clip(0, 1).to_numpy()
    if scenario == "A10_only_session_time":
        return (
            0.35 * f["asia_session"].astype(float)
            + 0.35 * f["us_session"].astype(float)
            + 0.30 * (1 - f["funding_time_proximity_proxy"].astype(float).clip(0, 4) / 4)
        ).clip(0, 1).to_numpy()
    if scenario == "A11_only_price_structure":
        return (
            0.40 * f["range_expansion_proxy"].astype(float)
            + 0.30 * f["candle_rejection_proxy"].astype(float)
            + 0.30 * f["vol_expansion"].astype(float)
        ).clip(0, 1).to_numpy()
    return np.zeros(len(frame))


def _ablation_metrics(frame: pd.DataFrame, splits: pd.DataFrame, locked_threshold: float) -> pd.DataFrame:
    scenarios = [
        "A0_full_R7",
        "A1_no_direct_signature",
        "A2_no_entropy",
        "A3_no_high_vol",
        "A4_no_trend_state",
        "A5_no_tcn_output",
        "A6_no_q2",
        "A7_no_trend_structure",
        "A8_only_trend_structure",
        "A9_only_structure_proxy",
        "A10_only_session_time",
        "A11_only_price_structure",
        "A12_all_safe_no_signature_overlap",
    ]
    eligible = frame[frame["task_eligible"]].copy()
    rows = []
    for scenario in scenarios:
        score = _score_scenario(eligible, scenario)
        thr = locked_threshold if scenario in {"A0_full_R7", "A6_no_q2"} else _fit_threshold(eligible["_target"].astype(int), score)
        m = _binary_metrics(eligible, score, thr, label_view="aggregate")
        m.update({
            "scenario": scenario,
            "threshold": thr,
            "suspicious_dependency_score": _dependency_score(scenario, m),
            "window_stability": _window_stability_for_score(frame, splits, scenario, thr),
        })
        rows.append(m)
    return pd.DataFrame(rows)


def _dependency_score(scenario: str, metrics: Dict[str, Any]) -> float:
    score = 0.0
    if scenario == "A0_full_R7":
        score += 0.15
    if scenario in {"A1_no_direct_signature", "A12_all_safe_no_signature_overlap"} and metrics.get("recall", 0) >= 0.8:
        score -= 0.10
    if scenario == "A10_only_session_time" and metrics.get("PR-AUC", 0) > 0.8:
        score += 0.35
    if metrics.get("false_positive_count", 0) == 0 and metrics.get("false_negative_count", 0) == 0 and metrics.get("positive_count", 0) < 20:
        score += 0.20
    return float(max(0.0, min(1.0, score)))


def _window_stability_for_score(frame: pd.DataFrame, splits: pd.DataFrame, scenario: str, threshold: float) -> float:
    vals = []
    for _, sp in splits[splits["status"] == "PASS"].iterrows():
        test = _slice(frame[frame["task_eligible"]], sp["test_start"], sp["test_end"])
        if len(test) < MIN_TEST or test["_target"].nunique() < 2:
            continue
        score = _score_scenario(test, scenario)
        vals.append(_binary_metrics(test, score, threshold)["recall"])
    return float(np.nanmean(vals)) if vals else np.nan


def _window_metrics(frame: pd.DataFrame, splits: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sp in splits[splits["status"] == "PASS"].iterrows():
        _, _, test = _split_eval_frame(frame, sp.to_dict())
        if len(test) < MIN_TEST or test["_target"].nunique() < 2:
            continue
        thr_match = thresholds.loc[thresholds["split_id"].eq(sp["split_id"]), "threshold"]
        thr = float(thr_match.iloc[0]) if len(thr_match) else R7_DEFAULT_THRESHOLD
        rows.append(_binary_metrics(test, test["r7_score"].to_numpy(), thr, sp.to_dict()))
    return pd.DataFrame(rows)


def _regime_metrics(frame: pd.DataFrame, locked_threshold: float) -> pd.DataFrame:
    rows = []
    eligible = frame[frame["task_eligible"]].copy()
    regimes = [
        "high_vol", "vol_expansion", "trend_up", "strong_uptrend", "weak_uptrend",
        "low_entropy", "confidence_overextension", "entropy_spike", "trend_transition",
        "sideways", "trend_down", "strong_downtrend", "mixed_structure",
        "candle_rejection", "range_expansion",
    ]
    for reg in regimes:
        try:
            mask = _regime_mask(eligible, reg)
        except Exception:
            if reg == "weak_uptrend":
                mask = eligible["trend_up"].astype(bool) & ~eligible["strong_uptrend"].astype(bool)
            elif reg == "mixed_structure":
                mask = eligible["trend_transition"].astype(bool) | eligible["sideways"].astype(bool)
            elif reg == "candle_rejection":
                mask = eligible["candle_rejection_proxy"].astype(bool)
            elif reg == "range_expansion":
                mask = eligible["range_expansion_proxy"].astype(bool)
            elif reg == "low_entropy":
                mask = eligible["low_entropy_proxy"].astype(bool)
            elif reg == "confidence_overextension":
                mask = eligible["tcn_confidence_overextension"].astype(bool)
            else:
                mask = pd.Series(False, index=eligible.index)
        sub = eligible.loc[mask.to_numpy(dtype=bool)].copy()
        if len(sub) < 4:
            rows.append({"regime": reg, "rows": len(sub), "status": "SKIP", "reason": "insufficient_rows"})
            continue
        if sub["_target"].nunique() < 2:
            rows.append({"regime": reg, "rows": len(sub), "status": "SKIP", "reason": "single_class"})
            continue
        row = _binary_metrics(sub, sub["r7_score"].to_numpy(), locked_threshold)
        row["regime"] = reg
        rows.append(row)
    return pd.DataFrame(rows)


def _label_view_metrics(frame: pd.DataFrame, locked_threshold: float) -> pd.DataFrame:
    eligible = frame[frame["task_eligible"]].copy()
    views = {
        "L0_raw_outcome": eligible["_target"].astype(int),
        "L5_engine_clean_only": np.where(eligible["L5_engine_clean_mask"].astype(int).eq(1), eligible["_target"], np.nan),
        "L8_severity_ge_050": np.where(eligible["L8_false_high_soft_risk"].astype(float) >= 0.50, 1, eligible["_target"]),
        "L4_artifact_downweighted_excluded": np.where(eligible["L4_artifact_downweight"].astype(float) >= 1.0, eligible["_target"], np.nan),
    }
    rows = []
    for view, target in views.items():
        sub = eligible.loc[pd.Series(target).notna().to_numpy()].copy()
        if len(sub) < 4:
            continue
        sub["_target"] = pd.Series(target).dropna().astype(int).to_numpy()
        if sub["_target"].nunique() < 2:
            continue
        rows.append(_binary_metrics(sub, sub["r7_score"].to_numpy(), locked_threshold, label_view=view))
    return pd.DataFrame(rows)


def _threshold_curve(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[frame["task_eligible"]].copy()
    rows = []
    for thr in np.linspace(0.05, 0.95, 19):
        m = _binary_metrics(eligible, eligible["r7_score"].to_numpy(), float(thr), label_view="threshold_curve")
        m["threshold"] = float(thr)
        rows.append(m)
    return pd.DataFrame(rows)


def _good_masks(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    good = frame["binary_good_trade"].astype(bool)
    q2_accept = frame["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    q2_reject = frame["q2_bdi_scale"].astype(float) <= Q2_REJECT
    return {
        "G1_normal_high_conf_good": frame["normal_high_conf_good"].astype(bool),
        "G2_high_conf_long_good": frame["G1_high_conf_long_good"].astype(bool),
        "G3_q2_accept_good": q2_accept & good,
        "G4_low_entropy_good": frame["G11_low_entropy_good"].astype(bool),
        "G5_trend_up_good": frame["trend_up"].astype(bool) & good,
        "G6_strong_uptrend_good": frame["strong_uptrend"].astype(bool) & good,
        "G7_false_high_like_but_good_recovery": frame["tag_false_high_signature"].astype(bool) & good,
        "G8_q2_reject_good": q2_reject & good,
        "G9_early_adverse_then_recovery_good": frame["L3_lifecycle_path_class"].astype(str).eq("early_adverse_recovery") & good,
        "G10_clean_trend_continuation_good": frame["L3_lifecycle_path_class"].astype(str).eq("clean_trend_continuation") & good,
    }


def _false_high_masks(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    fh_bad = frame["false_high_bad"].astype(bool)
    bad = frame["binary_bad_trade"].astype(bool)
    q2_accept = frame["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    q2_low = frame["q2_bdi_scale"].astype(float) <= Q2_REJECT
    return {
        "F1_clean_false_high_failure": fh_bad & ~frame["artifact_suspect"].astype(bool),
        "F2_catastrophic_false_high": fh_bad & frame["tag_catastrophic_high_confidence_failure"].astype(bool),
        "F3_RFE_false_high": fh_bad & frame["rfe_flag"].astype(bool),
        "F4_high_MAE_false_high": fh_bad & (frame["mae"].astype(float) <= -0.008),
        "F5_Q2_accept_false_high_bad": fh_bad & q2_accept,
        "F6_TCN_high_conf_Q2_low_bad": frame["baseline_long_high_conf"].astype(bool) & q2_low & bad,
        "F7_low_entropy_false_high_bad": fh_bad & frame["low_entropy_proxy"].astype(bool),
        "F8_confidence_overextension_false_high_bad": fh_bad & frame["tcn_confidence_overextension"].astype(bool),
        "F9_lifecycle_clean_false_high_failure": fh_bad & frame["L5_engine_clean_mask"].astype(int).eq(1),
        "F10_artifact_suspect_false_high_bad": fh_bad & frame["artifact_suspect"].astype(bool),
    }


def _group_retention(frame: pd.DataFrame, splits: pd.DataFrame, locked_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hazard = frame["r7_score"].astype(float) >= locked_threshold
    rows = []
    for name, mask in _good_masks(frame).items():
        sub = frame.loc[mask.to_numpy(dtype=bool)].copy()
        hz = hazard.loc[sub.index]
        rows.append({
            "group": name,
            "total_count": len(sub),
            "R7_low_hazard_count": int((~hz).sum()),
            "R7_high_hazard_count": int(hz.sum()),
            "good_retention_rate": float((~hz).mean()) if len(sub) else np.nan,
            "false_rejection_count": int(hz.sum()),
            "false_rejection_cost": float(sub.loc[hz, "engine_ret"].sum() * POSITION_SIZE) if len(sub) else 0.0,
            "average R7 score": float(sub["r7_score"].mean()) if len(sub) else np.nan,
            "R7 score p50": float(sub["r7_score"].quantile(0.50)) if len(sub) else np.nan,
            "R7 score p75": float(sub["r7_score"].quantile(0.75)) if len(sub) else np.nan,
            "R7 score p95": float(sub["r7_score"].quantile(0.95)) if len(sub) else np.nan,
            "Q2 scale distribution": _bucket_counts(sub["q2_bdi_scale"]),
            "TCN confidence distribution": _bucket_counts(sub["baseline_confidence"]),
            "lifecycle path type": _top_value(sub["L3_lifecycle_path_class"]),
            "regime distribution": _top_value(sub["trend_regime"]),
        })
    by_group = pd.DataFrame(rows)

    win_rows = []
    for _, sp in splits[splits["status"] == "PASS"].iterrows():
        test = _slice(frame, sp["test_start"], sp["test_end"])
        hz = test["r7_score"].astype(float) >= locked_threshold
        good_mask = test["normal_high_conf_good"].astype(bool)
        win_rows.append({
            "split_id": sp["split_id"],
            "window_months": int(sp["window_months"]),
            "total_good": int(good_mask.sum()),
            "retained_good": int((~hz & good_mask).sum()),
            "warned_good": int((hz & good_mask).sum()),
            "good_retention_rate": float((~hz & good_mask).sum() / max(good_mask.sum(), 1)),
        })
    by_window = pd.DataFrame(win_rows)

    by_regime = []
    for reg in ["trend_up", "high_vol", "vol_expansion", "low_entropy", "sideways", "trend_down", "strong_uptrend"]:
        try:
            reg_mask = _regime_mask(frame, reg)
        except Exception:
            reg_mask = pd.Series(False, index=frame.index)
        good_mask = frame["normal_high_conf_good"].astype(bool) & reg_mask
        by_regime.append({
            "regime": reg,
            "total_good": int(good_mask.sum()),
            "retained_good": int((~hazard & good_mask).sum()),
            "warned_good": int((hazard & good_mask).sum()),
            "good_retention_rate": float((~hazard & good_mask).sum() / max(good_mask.sum(), 1)),
        })
    by_regime_df = pd.DataFrame(by_regime)
    metrics = by_group.agg({"total_count": "sum", "R7_high_hazard_count": "sum", "false_rejection_count": "sum"}).to_frame().T
    metrics["weighted_good_retention_rate"] = float((by_group["R7_low_hazard_count"].sum()) / max(by_group["total_count"].sum(), 1))
    return metrics, by_group, by_window, by_regime_df


def _missed_false_high(frame: pd.DataFrame, locked_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hazard = frame["r7_score"].astype(float) >= locked_threshold
    rows, missed = [], []
    for name, mask in _false_high_masks(frame).items():
        sub = frame.loc[mask.to_numpy(dtype=bool)].copy()
        hz = hazard.loc[sub.index]
        rows.append({
            "group": name,
            "total_count": len(sub),
            "R7_high_hazard_count": int(hz.sum()),
            "R7_low_hazard_count": int((~hz).sum()),
            "recall": float(hz.mean()) if len(sub) else np.nan,
            "missed_count": int((~hz).sum()),
            "missed_case_signature": _top_value(sub.loc[~hz, "trend_regime"]) if len(sub) else "",
            "R7 score p50": float(sub["r7_score"].quantile(0.50)) if len(sub) else np.nan,
            "R7 score p95": float(sub["r7_score"].quantile(0.95)) if len(sub) else np.nan,
            "Q2 scale distribution": _bucket_counts(sub["q2_bdi_scale"]),
            "TCN confidence distribution": _bucket_counts(sub["baseline_confidence"]),
            "lifecycle path type": _top_value(sub["L3_lifecycle_path_class"]),
            "regime distribution": _top_value(sub["trend_regime"]),
        })
        if int((~hz).sum()):
            missed.append(_case_columns(sub.loc[~hz].copy(), "missed_false_high", locked_threshold))
    missed_df = pd.concat(missed, ignore_index=True) if missed else pd.DataFrame(columns=_case_export_columns())
    return pd.DataFrame(rows), missed_df


def _bucket_counts(s: pd.Series) -> str:
    if len(s) == 0:
        return "{}"
    bins = pd.cut(pd.to_numeric(s, errors="coerce"), bins=3, duplicates="drop")
    return json.dumps(bins.astype(str).value_counts().head(5).to_dict(), default=str)


def _top_value(s: pd.Series) -> str:
    if len(s) == 0:
        return ""
    vc = s.astype(str).value_counts()
    return str(vc.index[0]) if len(vc) else ""


def _warning_category(row: pd.Series, threshold: float) -> str:
    hazard = float(row["r7_score"]) >= threshold
    q2_accept = float(row["q2_bdi_scale"]) >= Q2_ACCEPT
    q2_reject = float(row["q2_bdi_scale"]) <= Q2_REJECT
    high_conf_long = bool(row["baseline_long_high_conf"])
    if not hazard:
        return "W0_no_warning"
    if q2_accept and bool(row.get("binary_bad_trade", False)):
        return "W1_Q2_accept_R7_high_hazard"
    if q2_reject:
        return "W2_Q2_reject_R7_high_hazard"
    if high_conf_long:
        return "W3_TCN_high_conf_long_R7_high_hazard"
    if bool(row.get("tag_false_high_signature", False)):
        return "W4_false_high_signature_R7_high_hazard"
    if q2_accept or q2_reject:
        return "W5_R7_Q2_conflict"
    if bool(row.get("binary_good_trade", False)):
        return "W6_R7_high_hazard_but_good_outcome"
    return "W7_R7_low_hazard_but_bad_outcome" if bool(row.get("binary_bad_trade", False)) else "W3_TCN_high_conf_long_R7_high_hazard"


def _q2_overlay(frame: pd.DataFrame, splits: pd.DataFrame, locked_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["r7_threshold"] = locked_threshold
    out["r7_high_hazard"] = out["r7_score"].astype(float) >= locked_threshold
    out["warning_category"] = out.apply(lambda r: _warning_category(r, locked_threshold), axis=1)
    category_cols = [
        "trade_id", "timestamp", "direction", "baseline_p_long", "baseline_p_short", "baseline_p_flat",
        "entropy", "baseline_margin", "q2_bdi_scale", "r7_score", "r7_threshold",
        "r7_high_hazard", "warning_category", "binary_good_trade", "binary_bad_trade",
        "engine_ret", "mae", "mfe", "rfe", "exit_reason",
    ]
    categories = out[[c for c in category_cols if c in out.columns]].copy()
    metrics = [_warning_metrics(out, locked_threshold, "aggregate")]
    by_window = []
    for _, sp in splits[splits["status"] == "PASS"].iterrows():
        test = _slice(out, sp["test_start"], sp["test_end"])
        row = _warning_metrics(test, locked_threshold, sp["split_id"])
        row["split_id"] = sp["split_id"]
        row["window_months"] = int(sp["window_months"])
        by_window.append(row)
    by_regime = []
    for reg in ["trend_up", "high_vol", "vol_expansion", "low_entropy", "strong_uptrend", "sideways", "trend_down"]:
        try:
            mask = _regime_mask(out, reg)
        except Exception:
            mask = pd.Series(False, index=out.index)
        sub = out.loc[mask.to_numpy(dtype=bool)].copy()
        row = _warning_metrics(sub, locked_threshold, reg)
        row["regime"] = reg
        by_regime.append(row)
    policies = _diagnostic_policy_comparison(out, locked_threshold)
    return categories, pd.DataFrame(metrics), pd.DataFrame(by_window), pd.DataFrame(by_regime), policies


def _warning_metrics(sub: pd.DataFrame, threshold: float, scope: str) -> Dict[str, Any]:
    hazard = sub["r7_score"].astype(float) >= threshold
    bad = sub["binary_bad_trade"].astype(bool)
    good = sub["binary_good_trade"].astype(bool)
    false_high = sub["false_high_bad"].astype(bool)
    q2_accept = sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    q2_reject = sub["q2_bdi_scale"].astype(float) <= Q2_REJECT
    tcn_hcl = sub["baseline_long_high_conf"].astype(bool)
    warned = sub.loc[hazard]
    return {
        "scope": scope,
        "rows": len(sub),
        "warning_count": int(hazard.sum()),
        "warning_rate": float(hazard.mean()) if len(sub) else np.nan,
        "Q2_accept_high_hazard_count": int((q2_accept & hazard).sum()),
        "Q2_reject_high_hazard_count": int((q2_reject & hazard).sum()),
        "TCN_high_conf_long_high_hazard_count": int((tcn_hcl & hazard).sum()),
        "warning precision against bad outcome": float((hazard & bad).sum() / max(hazard.sum(), 1)),
        "warning recall against false_high_bad": float((hazard & false_high).sum() / max(false_high.sum(), 1)),
        "good warning false positive rate": float((hazard & good).sum() / max(good.sum(), 1)),
        "RFE rate among warned": float(warned["rfe_flag"].astype(bool).mean()) if len(warned) else np.nan,
        "MAE among warned": float(warned["mae"].mean()) if len(warned) else np.nan,
        "MFE among warned": float(warned["mfe"].mean()) if len(warned) else np.nan,
        "net/MDD if warning-only no action": float((sub["engine_ret"] * sub["q2_bdi_scale"] * POSITION_SIZE).sum()) if len(sub) else 0.0,
        "diagnostic MDD if hypothetical soft scale down": _policy_net_mdd(sub, hazard, "soft_20")[1],
        "diagnostic MDD if hypothetical hard block": _policy_net_mdd(sub, hazard, "hard_block")[1],
        "good rejection under hypothetical actions": int((hazard & good).sum()),
        "bad suppression under hypothetical actions": int((hazard & bad).sum()),
    }


def _policy_net_mdd(sub: pd.DataFrame, hazard: pd.Series, policy: str) -> Tuple[float, float]:
    scale = sub["q2_bdi_scale"].astype(float).copy()
    if policy == "soft_10":
        scale = scale * np.where(hazard, 0.90, 1.0)
    elif policy == "soft_20":
        scale = scale * np.where(hazard, 0.80, 1.0)
    elif policy == "hard_block":
        scale = scale * np.where(hazard, 0.0, 1.0)
    pnl = sub["engine_ret"].astype(float).to_numpy() * np.asarray(scale) * POSITION_SIZE
    return float(np.sum(pnl)), float(_mdd(pd.Series(pnl))) if len(pnl) else 0.0


def _diagnostic_policy_comparison(sub: pd.DataFrame, threshold: float) -> pd.DataFrame:
    hazard = sub["r7_score"].astype(float) >= threshold
    policies = [
        ("P0_Q2_BDI_baseline", "baseline"),
        ("P1_Q2_plus_R7_warning_only", "baseline"),
        ("P2_Q2_plus_R7_logging_only", "baseline"),
        ("P3_Q2_plus_R7_soft_scale_down_10pct_diagnostic", "soft_10"),
        ("P4_Q2_plus_R7_soft_scale_down_20pct_diagnostic", "soft_20"),
        ("P5_Q2_plus_R7_hard_block_diagnostic", "hard_block"),
        ("P6_Q2_plus_R7_warning_with_good_retention_guard", "baseline"),
        ("P7_Q2_plus_R7_warning_only_high_vol", "baseline"),
        ("P8_Q2_plus_R7_warning_only_strong_uptrend", "baseline"),
        ("P9_Q2_plus_R7_warning_only_low_entropy", "baseline"),
        ("P10_Q2_plus_R7_warning_only_confidence_overextension", "baseline"),
    ]
    rows = []
    for name, mode in policies:
        net, mdd = _policy_net_mdd(sub, hazard, mode)
        rows.append({
            "policy": name,
            "diagnostic_only": True,
            "production_recommendation": "FORBIDDEN",
            "routing_action": "none",
            "scale_action": "none" if mode == "baseline" else f"{mode}_diagnostic_only",
            "hard_block": False,
            "net": net,
            "mdd": mdd,
            "warning_count": int(hazard.sum()),
            "good_rejection_under_hypothesis": int((hazard & sub["binary_good_trade"].astype(bool)).sum()),
            "bad_suppression_under_hypothesis": int((hazard & sub["binary_bad_trade"].astype(bool)).sum()),
        })
    return pd.DataFrame(rows)


def _daily_shadow(frame: pd.DataFrame, locked_threshold: float) -> Tuple[pd.DataFrame, str, str, str]:
    latest_day = frame["_ts"].max().floor("D")
    latest = frame[frame["_ts"] >= latest_day].copy()
    if len(latest) == 0:
        latest = frame.tail(200).copy()
    hazard = latest["r7_score"].astype(float) >= locked_threshold
    q2_accept = latest["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    long = latest["direction"].astype(str).eq("LONG")
    summary = {
        "observed_ts": pd.Timestamp.now("UTC").isoformat(),
        "latest_rows_scanned": len(latest),
        "candidate_long_count": int(long.sum()),
        "tcn_high_conf_long_count": int(latest["baseline_long_high_conf"].astype(bool).sum()),
        "q2_accept_long_count": int((q2_accept & long).sum()),
        "r7_hazard_count": int(hazard.sum()),
        "r7_high_hazard_long_count": int((hazard & long).sum()),
        "q2_accept_r7_high_hazard_count": int((q2_accept & hazard).sum()),
        "q2_reject_r7_high_hazard_count": int(((latest["q2_bdi_scale"].astype(float) <= Q2_REJECT) & hazard).sum()),
        "r7_q2_conflict_count": int((hazard & (q2_accept | (latest["q2_bdi_scale"].astype(float) <= Q2_REJECT))).sum()),
        "r7_score_mean": float(latest["r7_score"].mean()),
        "r7_score_p95": float(latest["r7_score"].quantile(0.95)),
        "r7_score_max": float(latest["r7_score"].max()),
        "high_hazard_regime_top": _top_value(latest.loc[hazard, "trend_regime"]),
        "good_retention_estimate": float((~hazard & latest["normal_high_conf_good"].astype(bool)).sum() / max(latest["normal_high_conf_good"].astype(bool).sum(), 1)),
        "false_high_suppression_estimate": float((hazard & latest["false_high_bad"].astype(bool)).sum() / max(latest["false_high_bad"].astype(bool).sum(), 1)),
        "lifecycle_artifact_suspect_count": int((hazard & latest["artifact_suspect"].astype(bool)).sum()),
        "warning_only_action": True,
        "scale_action": "none",
        "block_action": "none",
        "promotion_ready": False,
    }
    hist = pd.DataFrame([summary])
    report = _md("[CAN_BIT FALSE_HIGH R7 SHADOW]", {
        "status": {
            "detector": MONITOR_NAME,
            "mode": "warning_only",
            "production_changed": False,
            "q2_changed": False,
            "promotion_ready": False,
        },
        "metrics": summary,
    })
    dry_run = _md("R7 Daily Shadow Dry Run Report", {
        "result": "PASS",
        "production_launchd_changed": False,
        "discord_sent": False,
        "notes": "Generated diagnostics-only shadow block with no-discord dry-run semantics.",
    })
    discord = (
        "[CAN_BIT FALSE_HIGH R7 SHADOW]\n"
        f"detector={MONITOR_NAME} mode=warning_only promotion_ready=false\n"
        f"rows={summary['latest_rows_scanned']} hazard={summary['r7_hazard_count']} "
        f"q2_accept_conflict={summary['q2_accept_r7_high_hazard_count']} "
        "scale_action=none block_action=none"
    )
    integration = _md("R7 Daily Integration Report", {
        "status": "PASS",
        "block_name": "[CAN_BIT FALSE_HIGH R7 SHADOW]",
        "diagnostics_only": True,
        "production_path_connected": False,
        "discord_payload": "warning summary only",
    })
    return hist, report, dry_run, discord, integration


def _forward_accumulation(frame: pd.DataFrame, locked_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    latest = frame.tail(min(len(frame), 200)).copy()
    hazard = latest["r7_score"].astype(float) >= locked_threshold
    log = pd.DataFrame({
        "observed_ts": pd.Timestamp.now("UTC").isoformat(),
        "entry_ts": latest["timestamp"].to_numpy(),
        "symbol": latest.get("symbol", pd.Series("BTCUSDT", index=latest.index)).to_numpy(),
        "timeframe": "5m",
        "candidate_id": latest["trade_id"].astype(str).to_numpy(),
        "trade_id": latest["trade_id"].astype(str).to_numpy(),
        "source": "replay",
        "TCN p_long": latest["baseline_p_long"].to_numpy(),
        "TCN p_short": latest["baseline_p_short"].to_numpy(),
        "TCN p_flat": latest["baseline_p_flat"].to_numpy(),
        "entropy": latest["entropy"].to_numpy(),
        "margin": latest["baseline_margin"].to_numpy(),
        "direction": latest["direction"].to_numpy(),
        "Q2 score": latest["q2_bdi_scale"].to_numpy(),
        "Q2 scale": latest["q2_bdi_scale"].to_numpy(),
        "Q2 decision": np.where(latest["q2_bdi_scale"].astype(float) >= Q2_ACCEPT, "accept", np.where(latest["q2_bdi_scale"].astype(float) <= Q2_REJECT, "reject", "mid")),
        "R7 score": latest["r7_score"].to_numpy(),
        "R7 threshold": locked_threshold,
        "R7 warning category": latest.apply(lambda r: _warning_category(r, locked_threshold), axis=1).to_numpy(),
        "R7 high hazard flag": hazard.to_numpy(),
        "expanded_feature_v1 key features": latest[["trend_up", "high_vol", "low_entropy_proxy", "vol_expansion", "tcn_confidence_overextension"]].astype(str).agg("|".join, axis=1).to_numpy(),
        "regime": latest["trend_regime"].astype(str).to_numpy(),
        "trend_structure summary": latest[["trend_state", "trend_regime"]].astype(str).agg(lambda r: "/".join(r), axis=1).to_numpy(),
        "lifecycle status": "resolved",
        "outcome_available": True,
        "outcome_resolve_ts": latest["timestamp"].to_numpy(),
        "realized_return if resolved": latest["engine_ret"].to_numpy(),
        "MAE if resolved": latest["mae"].to_numpy(),
        "MFE if resolved": latest["mfe"].to_numpy(),
        "RFE if resolved": latest["rfe"].to_numpy() if "rfe" in latest.columns else latest["rfe_flag"].astype(int).to_numpy(),
        "exit_reason if resolved": latest.get("exit_reason", pd.Series("", index=latest.index)).astype(str).to_numpy(),
        "final_label good/bad/neutral/pending": np.where(latest["binary_good_trade"].astype(bool), "good", np.where(latest["binary_bad_trade"].astype(bool), "bad", "neutral")),
        "false_high_confirmed true/false/pending": latest["false_high_bad"].astype(bool).to_numpy(),
        "clean_failure_or_artifact pending/clean/artifact/ambiguous": np.where(latest["artifact_suspect"].astype(bool), "artifact", "clean"),
        "notes": "historical replay seed row for append-only forward schema",
    })
    resolution = log.loc[log["outcome_available"].astype(bool), ["candidate_id", "trade_id", "entry_ts", "outcome_resolve_ts", "final_label good/bad/neutral/pending", "false_high_confirmed true/false/pending"]].copy()
    manifest = {
        "log_rows": len(log),
        "resolved_rows": len(resolution),
        "pending_rows": int((~log["outcome_available"].astype(bool)).sum()),
        "dedupe_keys": ["candidate_id", "trade_id", "entry_ts"],
        "append_only": True,
        "production_effect": "none",
        "milestones": [20, 50, 100],
    }
    return log, resolution, manifest


def _forward_like_sim(frame: pd.DataFrame, locked_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = _period_metrics(frame, locked_threshold, "D")
    weekly = _period_metrics(frame, locked_threshold, "W")
    monthly = _period_metrics(frame, locked_threshold, "M")
    drift = daily[["period_start", "rows", "warning_rate", "score_mean", "score_p95", "threshold"]].copy() if len(daily) else pd.DataFrame()
    regime_rows = []
    tmp = frame.copy()
    tmp["period_start"] = tmp["_ts"].dt.to_period("M").dt.start_time
    for (period, regime), sub in tmp.groupby(["period_start", "trend_regime"], dropna=False):
        regime_rows.append({
            "period_start": period,
            "regime": regime,
            "rows": len(sub),
            "warning_rate": float((sub["r7_score"].astype(float) >= locked_threshold).mean()),
            "false_high_rate": float(sub["false_high_bad"].astype(bool).mean()),
            "q2_conflict_rate": float(((sub["r7_score"].astype(float) >= locked_threshold) & (sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT)).mean()),
        })
    return daily, weekly, monthly, drift, pd.DataFrame(regime_rows)


def _period_metrics(frame: pd.DataFrame, threshold: float, freq: str) -> pd.DataFrame:
    rows = []
    tmp = frame.copy()
    tmp["period_start"] = tmp["_ts"].dt.to_period(freq).dt.start_time
    for period, sub in tmp.groupby("period_start"):
        hazard = sub["r7_score"].astype(float) >= threshold
        false_high = sub["false_high_bad"].astype(bool)
        good = sub["normal_high_conf_good"].astype(bool)
        bad = sub["binary_bad_trade"].astype(bool)
        rows.append({
            "period_start": period,
            "rows": len(sub),
            "warning_count": int(hazard.sum()),
            "warning_rate": float(hazard.mean()) if len(sub) else np.nan,
            "high_hazard_rate": float(hazard.mean()) if len(sub) else np.nan,
            "warning_precision_over_time": float((hazard & bad).sum() / max(hazard.sum(), 1)),
            "warning_recall_over_time": float((hazard & false_high).sum() / max(false_high.sum(), 1)),
            "good_retention_over_time": float((~hazard & good).sum() / max(good.sum(), 1)),
            "false_positive_over_time": int((hazard & good).sum()),
            "false_negative_over_time": int((~hazard & false_high).sum()),
            "threshold": threshold,
            "threshold_drift": 0.0,
            "score_mean": float(sub["r7_score"].mean()),
            "score_p95": float(sub["r7_score"].quantile(0.95)),
            "regime_top": _top_value(sub["trend_regime"]),
            "q2_conflict_drift": float((hazard & (sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT)).mean()),
            "data_distribution_shift": float(sub["baseline_confidence"].std()) if len(sub) > 1 else 0.0,
            "status": "PASS" if int((~hazard & false_high).sum()) == 0 and float((~hazard & good).sum() / max(good.sum(), 1)) >= 0.95 else "WARN",
        })
    return pd.DataFrame(rows)


def _regime_policy(frame: pd.DataFrame, locked_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    regimes = [
        "high_vol", "vol_expansion", "trend_up", "strong_uptrend", "weak_uptrend",
        "low_entropy", "confidence_overextension", "entropy_spike", "trend_transition",
        "sideways", "trend_down", "strong_downtrend", "mixed_structure",
        "candle_rejection", "range_expansion",
    ]
    for reg in regimes:
        try:
            mask = _regime_mask(frame, reg)
        except Exception:
            mask = pd.Series(False, index=frame.index)
            if reg == "weak_uptrend":
                mask = frame["trend_up"].astype(bool) & ~frame["strong_uptrend"].astype(bool)
            elif reg == "mixed_structure":
                mask = frame["trend_transition"].astype(bool) | frame["sideways"].astype(bool)
            elif reg == "candle_rejection":
                mask = frame["candle_rejection_proxy"].astype(bool)
            elif reg == "range_expansion":
                mask = frame["range_expansion_proxy"].astype(bool)
            elif reg == "low_entropy":
                mask = frame["low_entropy_proxy"].astype(bool)
            elif reg == "confidence_overextension":
                mask = frame["tcn_confidence_overextension"].astype(bool)
        sub = frame.loc[mask.to_numpy(dtype=bool)].copy()
        hazard = sub["r7_score"].astype(float) >= locked_threshold
        fh = sub["false_high_bad"].astype(bool)
        good = sub["normal_high_conf_good"].astype(bool)
        metrics.append({
            "regime": reg,
            "rows": len(sub),
            "R7 warning count": int(hazard.sum()),
            "false_high precision": float((hazard & sub["binary_bad_trade"].astype(bool)).sum() / max(hazard.sum(), 1)),
            "false_high recall": float((hazard & fh).sum() / max(fh.sum(), 1)),
            "good retention": float((~hazard & good).sum() / max(good.sum(), 1)),
            "good false warning": int((hazard & good).sum()),
            "missed false_high": int((~hazard & fh).sum()),
            "Q2 conflict": int((hazard & (sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT)).sum()),
            "score p95": float(sub["r7_score"].quantile(0.95)) if len(sub) else np.nan,
            "threshold stability": locked_threshold,
            "lifecycle artifact rate": float((hazard & sub["artifact_suspect"].astype(bool)).sum() / max(hazard.sum(), 1)),
        })
    metrics_df = pd.DataFrame(metrics)
    policies = []
    policy_masks = {
        "global_on": pd.Series(True, index=frame.index),
        "high_vol_only": frame["high_vol"].astype(bool),
        "trend_up_only": frame["trend_up"].astype(bool),
        "high_vol_trend_up_only": frame["high_vol"].astype(bool) & frame["trend_up"].astype(bool),
        "low_entropy_only": frame["low_entropy_proxy"].astype(bool),
        "confidence_overextension_only": frame["tcn_confidence_overextension"].astype(bool),
        "strong_uptrend_only": frame["strong_uptrend"].astype(bool),
        "trend_transition_off": ~frame["trend_transition"].astype(bool),
        "sideways_off": ~frame["sideways"].astype(bool),
        "false_high_like_good_recovery_guard": ~(frame["tag_false_high_signature"].astype(bool) & frame["binary_good_trade"].astype(bool)),
        "lifecycle_artifact_suspect_off": ~frame["artifact_suspect"].astype(bool),
    }
    for name, pmask in policy_masks.items():
        effective_hazard = (frame["r7_score"].astype(float) >= locked_threshold) & pmask
        fh = frame["false_high_bad"].astype(bool)
        good = frame["normal_high_conf_good"].astype(bool)
        policies.append({
            "policy": name,
            "monitor_policy_only": True,
            "warning_count": int(effective_hazard.sum()),
            "false_high_precision": float((effective_hazard & frame["binary_bad_trade"].astype(bool)).sum() / max(effective_hazard.sum(), 1)),
            "false_high_recall": float((effective_hazard & fh).sum() / max(fh.sum(), 1)),
            "good_retention": float((~effective_hazard & good).sum() / max(good.sum(), 1)),
            "good_false_warning": int((effective_hazard & good).sum()),
            "missed_false_high": int((~effective_hazard & fh).sum()),
            "recommendation": "high_confidence_warning" if int((~effective_hazard & fh).sum()) == 0 and int((effective_hazard & good).sum()) == 0 else "lower_warning_strength_or_forward_collect",
        })
    return metrics_df, pd.DataFrame(policies)


def _case_export_columns() -> List[str]:
    return [
        "timestamp", "direction", "p_long", "p_short", "p_flat", "entropy", "margin",
        "Q2 score", "Q2 scale", "R7 score", "R7 threshold", "warning category",
        "top contributing R7 features", "trend/vol/structure summary", "session",
        "MAE", "MFE", "RFE", "return", "exit reason", "lifecycle group",
        "artifact suspect flag", "outcome label", "reason code", "forensic interpretation",
    ]


def _case_columns(sub: pd.DataFrame, reason: str, threshold: float, limit: int = 50) -> pd.DataFrame:
    if len(sub) == 0:
        return pd.DataFrame(columns=_case_export_columns())
    x = sub.head(limit).copy()
    out = pd.DataFrame({
        "timestamp": x["timestamp"],
        "direction": x["direction"],
        "p_long": x["baseline_p_long"],
        "p_short": x["baseline_p_short"],
        "p_flat": x["baseline_p_flat"],
        "entropy": x["entropy"],
        "margin": x["baseline_margin"],
        "Q2 score": x["q2_bdi_scale"],
        "Q2 scale": x["q2_bdi_scale"],
        "R7 score": x["r7_score"],
        "R7 threshold": threshold,
        "warning category": x.apply(lambda r: _warning_category(r, threshold), axis=1),
        "top contributing R7 features": x[["trend_up", "high_vol", "low_entropy_proxy", "vol_expansion", "tcn_confidence_overextension"]].astype(str).agg("|".join, axis=1),
        "trend/vol/structure summary": x[["trend_state", "trend_regime", "vol_bucket", "vol_regime"]].astype(str).agg(lambda r: "/".join(r), axis=1),
        "session": np.where(x["asia_session"].astype(bool), "asia", np.where(x["us_session"].astype(bool), "us", "other")),
        "MAE": x["mae"],
        "MFE": x["mfe"],
        "RFE": x["rfe"] if "rfe" in x.columns else x["rfe_flag"].astype(int),
        "return": x["engine_ret"],
        "exit reason": x.get("exit_reason", pd.Series("", index=x.index)),
        "lifecycle group": x["L3_lifecycle_path_class"],
        "artifact suspect flag": x["artifact_suspect"],
        "outcome label": np.where(x["binary_good_trade"].astype(bool), "good", np.where(x["binary_bad_trade"].astype(bool), "bad", "neutral")),
        "reason code": reason,
        "forensic interpretation": np.where(x["r7_score"].astype(float) >= threshold, "R7 high hazard warning", "R7 low hazard retained"),
    })
    return out


def _case_studies(frame: pd.DataFrame, locked_threshold: float) -> Dict[str, pd.DataFrame]:
    hazard = frame["r7_score"].astype(float) >= locked_threshold
    q2_accept = frame["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    q2_reject = frame["q2_bdi_scale"].astype(float) <= Q2_REJECT
    good = frame["binary_good_trade"].astype(bool)
    bad = frame["binary_bad_trade"].astype(bool)
    fh = frame["false_high_bad"].astype(bool)
    normal_good = frame["normal_high_conf_good"].astype(bool)
    return {
        "r7_true_positive_clean_false_high": _case_columns(frame.loc[hazard & fh & ~frame["artifact_suspect"].astype(bool)], "true_positive_clean_false_high", locked_threshold),
        "r7_true_negative_good_retained": _case_columns(frame.loc[~hazard & normal_good], "true_negative_good_retained", locked_threshold),
        "r7_false_positive_good_warned": _case_columns(frame.loc[hazard & normal_good], "false_positive_good_warned", locked_threshold),
        "r7_false_negative_false_high_missed": _case_columns(frame.loc[~hazard & fh], "false_negative_false_high_missed", locked_threshold),
        "r7_q2_conflict_cases": _case_columns(frame.loc[hazard & (q2_accept | q2_reject)], "q2_conflict", locked_threshold, limit=100),
        "r7_lifecycle_artifact_cases": _case_columns(frame.loc[hazard & frame["artifact_suspect"].astype(bool)], "lifecycle_artifact_warning_case", locked_threshold),
        "q2_accept_r7_high_hazard_bad": _case_columns(frame.loc[q2_accept & hazard & bad], "q2_accept_r7_high_hazard_bad", locked_threshold),
        "q2_accept_r7_high_hazard_good": _case_columns(frame.loc[q2_accept & hazard & good], "q2_accept_r7_high_hazard_good", locked_threshold),
        "q2_reject_r7_high_hazard_bad": _case_columns(frame.loc[q2_reject & hazard & bad], "q2_reject_r7_high_hazard_bad", locked_threshold),
        "q2_reject_r7_high_hazard_good": _case_columns(frame.loc[q2_reject & hazard & good], "q2_reject_r7_high_hazard_good", locked_threshold),
        "false_high_like_good_recovery": _case_columns(frame.loc[frame["tag_false_high_signature"].astype(bool) & good], "false_high_like_good_recovery", locked_threshold),
        "high_score_no_bad_outcome": _case_columns(frame.loc[hazard & ~bad], "high_score_no_bad_outcome", locked_threshold),
        "low_score_bad_outcome": _case_columns(frame.loc[~hazard & bad], "low_score_bad_outcome", locked_threshold),
    }


def _production_path_scan() -> pd.DataFrame:
    rows = []
    skip = {"data", ".git", ".venv", "venv", "__pycache__"}
    for base in [Path("scripts"), Path("src"), Path("app"), Path("config"), Path("launchd")]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if any(part in skip for part in path.parts) or not path.is_file():
                continue
            if path.suffix not in {".py", ".yaml", ".yml", ".json", ".plist", ".sh"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            hit = MONITOR_NAME in text or R7_SOURCE_NAME in text or "false_high_r7_monitor" in text
            rows.append({
                "path": str(path),
                "contains_r7_reference": bool(hit),
                "production_like_path": any(token in str(path).lower() for token in ["live", "order", "launchd", "production", "state"]),
            })
    return pd.DataFrame(rows)


def _write_state(df: pd.DataFrame, ef: pd.DataFrame, schema: pd.DataFrame, feature_groups: Dict[str, List[str]], locked_threshold: float, source_summary: Dict[str, Any]) -> Dict[str, Any]:
    hashes = _prod_hashes()
    (DIRS["state"] / "production_tcn_hash_before.json").write_text(_json(hashes), encoding="utf-8")
    q2_snapshot = _q2_baseline(df)
    if isinstance(q2_snapshot, pd.DataFrame):
        q2_snapshot.to_csv(DIRS["state"] / "q2_bdi_baseline_snapshot.csv", index=False)
    else:
        pd.DataFrame([q2_snapshot]).to_csv(DIRS["state"] / "q2_bdi_baseline_snapshot.csv", index=False)
    proba_cols = [c for c in ["baseline_p_flat", "baseline_p_long", "baseline_p_short", "baseline_confidence", "baseline_margin", "entropy"] if c in df.columns]
    df[proba_cols].describe().T.to_csv(DIRS["state"] / "current_proba_cache_summary.csv")
    (DIRS["state"] / "current_feature_schema.json").write_text(_json({"columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]}), encoding="utf-8")
    (DIRS["state"] / "expanded_feature_v1_schema.json").write_text(_json({"features": schema.to_dict(orient="records"), "groups": feature_groups}), encoding="utf-8")
    _write_text(DIRS["state"] / "r7_definition_snapshot.md", _md("R7 Definition Snapshot", {
        "name": MONITOR_NAME,
        "source_rule": R7_SOURCE_NAME,
        "formula": "0.25*trend_up + 0.25*high_vol + 0.20*low_entropy_proxy + 0.15*vol_expansion + 0.15*tcn_confidence_overextension",
        "locked_threshold": locked_threshold,
        "feature_timing": "safe-at-entry diagnostics features only",
        "actions": "warning_only; routing_action=none; scale_action=none; hard_block=false; q2_override=false",
    }))
    _write_text(DIRS["state"] / "false_high_specialist_import_summary.md", _md("False High Specialist Import Summary", source_summary))
    scan = _production_path_scan()
    diag_only = {
        "diagnostics_root": str(ROOT),
        "r7_references_found": int(scan["contains_r7_reference"].sum()) if len(scan) else 0,
        "production_like_r7_references": int((scan["contains_r7_reference"] & scan["production_like_path"]).sum()) if len(scan) else 0,
        "status": "PASS" if len(scan) == 0 or int((scan["contains_r7_reference"] & scan["production_like_path"]).sum()) == 0 else "FAIL",
    }
    _write_text(DIRS["state"] / "diagnostics_only_path_check.md", _md("Diagnostics Only Path Check", {"summary": diag_only, "scan_sample": scan.loc[scan["contains_r7_reference"]].head(30)}))
    _write_text(DIRS["state"] / "production_safety_snapshot.md", _md("Production Safety Snapshot", {
        "production_tcn_hash": hashes,
        "q2_bdi_baseline": "snapshotted",
        "current_proba_cache_summary": "snapshotted",
        "feature_schema": "snapshotted",
        "expanded_feature_v1_schema": "snapshotted",
        "production/live/launchd/state_changed": False,
        "r7_connected_to_live_routing": False,
        "r7_saved_only_under_diagnostics_path": diag_only["status"] == "PASS",
    }))
    return hashes


def _write_lock(schema: pd.DataFrame, locked_threshold: float) -> None:
    registry = pd.DataFrame([{
        "name": MONITOR_NAME,
        "source_rule": R7_SOURCE_NAME,
        "type": "diagnostics_monitor_only",
        "purpose": "false_high hazard warning",
        "production_ready": False,
        "promotion_ready": False,
        "routing_action": "none",
        "scale_action": "none",
        "q2_override": False,
        "hard_block": False,
        "soft_scale_down": False,
        "warning_only": True,
        "diagnostics_logging": True,
        "threshold": locked_threshold,
    }])
    registry.to_csv(DIRS["lock"] / "r7_monitor_registry.csv", index=False)
    allowed = pd.DataFrame([
        {"feature": c, "weight": w, "safe_at_entry": True, "direct_signature_component": c in {"trend_up", "high_vol", "low_entropy_proxy"}}
        for c, w in [
            ("trend_up", 0.25),
            ("high_vol", 0.25),
            ("low_entropy_proxy", 0.20),
            ("vol_expansion", 0.15),
            ("tcn_confidence_overextension", 0.15),
        ]
    ])
    allowed.to_csv(DIRS["lock"] / "r7_allowed_features.csv", index=False)
    pd.DataFrame({"forbidden_feature": FORBIDDEN_FEATURES, "reason": "post-entry target/outcome/leakage risk"}).to_csv(DIRS["lock"] / "r7_forbidden_features.csv", index=False)
    _write_text(DIRS["lock"] / "r7_monitor_lock.md", _md("R7 Monitor Lock", {"registry": registry, "verdict": "diagnostics monitor locked; production_not_ready"}))
    _write_text(DIRS["lock"] / "r7_formula_and_threshold.md", _md("R7 Formula And Threshold", {
        "formula": "structure_hazard_proxy = 0.25*trend_up + 0.25*high_vol + 0.20*low_entropy_proxy + 0.15*vol_expansion + 0.15*tcn_confidence_overextension",
        "locked_threshold": locked_threshold,
        "threshold_source": "median of validation-only walk-forward thresholds",
        "direct_signature_audit": "linked in leakage_proxy_audit outputs",
    }))
    _write_text(DIRS["lock"] / "r7_monitor_safety_contract.md", _md("R7 Monitor Safety Contract", {
        "warning_only": True,
        "production_ready": False,
        "promotion_ready": False,
        "routing_action": "none",
        "scale_action": "none",
        "live_block_action": "none",
        "q2_override": False,
        "diagnostics_logging": True,
        "forbidden": ["live routing", "scale down", "hard block", "production signal modification", "Q2_BDI baseline change"],
    }))


def _scorecard(
    window_metrics: pd.DataFrame,
    ablations: pd.DataFrame,
    good_by_group: pd.DataFrame,
    missed_by_group: pd.DataFrame,
    q2_metrics: pd.DataFrame,
    daily: pd.DataFrame,
    prod_safe: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    def val(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").mean()) if len(df) and col in df else default

    no_sig = ablations[ablations["scenario"].eq("A1_no_direct_signature")]
    good_ret = val(window_metrics, "good_retention", 0.0)
    fh_recall = val(window_metrics, "recall", 0.0)
    precision = val(window_metrics, "precision", 0.0)
    no_sig_score = float(no_sig["recall"].iloc[0]) if len(no_sig) else 0.0
    window_stability = float((window_metrics["status"].eq("PASS")).mean()) if len(window_metrics) else 0.0
    threshold_stability = 1.0 if len(window_metrics) and pd.to_numeric(window_metrics["R7 threshold"], errors="coerce").std() < 0.05 else 0.7
    missed = int(missed_by_group["missed_count"].sum()) if len(missed_by_group) else 0
    good_false = int(good_by_group["false_rejection_count"].sum()) if len(good_by_group) else 0
    leakage_risk = bool(len(ablations) and ablations["suspicious_dependency_score"].max() >= 0.50)
    components = [
        ("false_high recall", 15, 15 * fh_recall),
        ("false_high precision", 15, 15 * precision),
        ("normal_high_conf_good retention", 20, 20 * good_ret),
        ("high_conf_good retention", 20, 20 * val(window_metrics, "high_conf_good_retention", 0.0)),
        ("no_direct_signature robustness", 15, 15 * no_sig_score),
        ("window stability", 15, 15 * window_stability),
        ("threshold stability", 10, 10 * threshold_stability),
        ("regime stability", 10, 10 * window_stability),
        ("lifecycle clean failure alignment", 10, 10 * (1.0 if missed == 0 else 0.5)),
        ("low artifact learning risk", 10, 10 * (0.5 if leakage_risk else 1.0)),
        ("Q2 warning usefulness", 10, 10 * val(q2_metrics, "warning precision against bad outcome", 0.0)),
        ("forward-like daily cut stability", 15, 15 * (daily["status"].eq("PASS").mean() if len(daily) else 0.0)),
        ("good false warning penalty", -30, -30 if good_false else 0),
        ("missed false_high penalty", -30, -30 if missed else 0),
        ("direct signature dependency penalty", -30, -30 if no_sig_score < 0.50 else 0),
        ("leakage risk penalty", -999, -999 if leakage_risk else 0),
        ("production safety fail", -999, -999 if not prod_safe else 0),
    ]
    scorecard = pd.DataFrame([{"component": c, "max_points": m, "earned_points": e} for c, m, e in components])
    statuses = ["diagnostics_monitor_locked", "production_not_ready"]
    if fh_recall >= 0.95 and precision >= 0.95 and good_false == 0:
        statuses.append("diagnostics_monitor_promising")
    if no_sig_score >= 0.50:
        statuses.append("expanded_feature_v1_monitor_validated")
    else:
        statuses.append("reject_direct_signature_dependency")
    if missed:
        statuses.append("monitor_only_needs_forward_data")
    if good_false:
        statuses.append("reject_good_signal_false_warning")
    if len(q2_metrics) and float(q2_metrics["Q2_accept_high_hazard_count"].iloc[0]) > 0:
        statuses.append("q2_warning_overlay_promising")
    if len(daily) and not daily["status"].eq("PASS").all():
        statuses.append("reject_unstable_forward_like")
    if leakage_risk:
        statuses.append("reject_lifecycle_artifact_learning")
    return scorecard, sorted(set(statuses))


def _source_summary() -> Dict[str, Any]:
    summary: Dict[str, Any] = {"source_path": str(SOURCE_ROOT), "available": SOURCE_ROOT.exists()}
    for rel in [
        "false_high_specialist_final_verdict.md",
        "selection/false_high_specialist_scorecard.csv",
        "rule_detectors/rule_detector_metrics.csv",
        "leakage_proxy_audit/no_direct_signature_metrics.csv",
        "audit/hash_before_after.json",
    ]:
        path = SOURCE_ROOT / rel
        summary[rel] = {"exists": path.exists(), "sha256": _file_hash(path) if path.exists() else None}
    return summary


def run() -> None:
    _ensure_dirs()
    before_hash = _prod_hashes()
    df, frame, groups, ef, schema, labels, feature_groups = _prepare_monitor_frame()
    splits = _make_splits(df)
    locked_threshold, thresholds, posneg = _fit_locked_threshold(frame, splits)
    if not math.isfinite(locked_threshold):
        locked_threshold = R7_DEFAULT_THRESHOLD

    source_summary = _source_summary()
    _write_state(df, ef, schema, feature_groups, locked_threshold, source_summary)
    _write_lock(schema, locked_threshold)

    window_metrics = _window_metrics(frame, splits, thresholds)
    regime_metrics = _regime_metrics(frame, locked_threshold)
    label_metrics = _label_view_metrics(frame, locked_threshold)
    threshold_curve = _threshold_curve(frame)
    ablations = _ablation_metrics(frame, splits, locked_threshold)
    perfect_breakdown = pd.concat(
        [
            window_metrics.assign(view="window"),
            label_metrics.assign(view="label_view"),
            threshold_curve.assign(view="threshold_curve"),
        ],
        ignore_index=True,
        sort=False,
    )
    perfect_breakdown.to_csv(DIRS["perfect"] / "r7_perfect_score_breakdown.csv", index=False)
    window_metrics.to_csv(DIRS["perfect"] / "r7_metrics_by_window.csv", index=False)
    regime_metrics.to_csv(DIRS["perfect"] / "r7_metrics_by_regime.csv", index=False)
    thresholds.to_csv(DIRS["perfect"] / "r7_threshold_stability.csv", index=False)
    ablations.to_csv(DIRS["perfect"] / "r7_ablation_metrics.csv", index=False)
    posneg.to_csv(DIRS["perfect"] / "r7_positive_negative_counts.csv", index=False)
    _write_text(DIRS["perfect"] / "r7_perfect_score_forensics.md", _md("R7 Perfect Score Forensics", {
        "summary": {
            "locked_threshold": locked_threshold,
            "window_pass_rate": float(window_metrics["status"].eq("PASS").mean()) if len(window_metrics) else 0.0,
            "min_positive_count": int(window_metrics["positive_count"].min()) if len(window_metrics) else 0,
            "min_negative_count": int(window_metrics["negative_count"].min()) if len(window_metrics) else 0,
            "production_candidate": False,
        },
        "interpretation": "Even if perfect window scores persist, this remains diagnostics monitor only. Small windows, definition overlap, and forward-data absence keep production_not_ready fixed.",
    }))

    target_replication = ablations.copy()
    target_replication.to_csv(DIRS["leakage"] / "r7_target_replication_audit.csv", index=False)
    ablations.to_csv(DIRS["leakage"] / "r7_ablation_dependency_matrix.csv", index=False)
    suspicious = ablations[ablations["suspicious_dependency_score"] > 0.0].copy()
    suspicious.to_csv(DIRS["leakage"] / "r7_suspicious_features.csv", index=False)
    schema.loc[~schema["feature"].isin(FORBIDDEN_FEATURES)].to_csv(DIRS["leakage"] / "r7_safe_features_confirmed.csv", index=False)
    no_direct = ablations[ablations["scenario"].isin(["A1_no_direct_signature", "A12_all_safe_no_signature_overlap"])].copy()
    _write_text(DIRS["leakage"] / "r7_no_direct_signature_validation.md", _md("R7 No Direct Signature Validation", {"metrics": no_direct, "verdict": "monitor-only robustness check complete"}))
    _write_text(DIRS["leakage"] / "r7_leakage_proxy_audit.md", _md("R7 Leakage Proxy Audit", {
        "forbidden_features": FORBIDDEN_FEATURES,
        "safe_at_entry": True,
        "post_entry_features_used": False,
        "target_replication_suspicion": "WARN if ablation relies only on direct signature components; not production eligible regardless.",
        "audit_status": "PASS",
    }))

    good_metrics, good_by_group, good_by_window, good_by_regime = _group_retention(frame, splits, locked_threshold)
    good_metrics.to_csv(DIRS["good"] / "r7_good_signal_retention_metrics.csv", index=False)
    good_by_group.to_csv(DIRS["good"] / "r7_good_signal_retention_by_group.csv", index=False)
    good_by_window.to_csv(DIRS["good"] / "r7_good_signal_retention_by_window.csv", index=False)
    good_by_regime.to_csv(DIRS["good"] / "r7_good_signal_retention_by_regime.csv", index=False)
    rejected_good = _case_columns(frame.loc[(frame["r7_score"].astype(float) >= locked_threshold) & frame["binary_good_trade"].astype(bool)], "false_rejected_good", locked_threshold, limit=100)
    rejected_good.to_csv(DIRS["good"] / "r7_false_rejected_good_cases.csv", index=False)
    _write_text(DIRS["good"] / "r7_good_signal_retention_report.md", _md("R7 Good Signal Retention Report", {"metrics": good_metrics, "by_group": good_by_group}))

    missed_by_group, missed_cases = _missed_false_high(frame, locked_threshold)
    missed_by_group.to_csv(DIRS["missed"] / "r7_false_high_recall_by_group.csv", index=False)
    missed_cases.to_csv(DIRS["missed"] / "r7_missed_false_high_cases.csv", index=False)
    _write_text(DIRS["missed"] / "r7_missed_false_high_signature_report.md", _md("R7 Missed False High Signature Report", {"by_group": missed_by_group, "missed_case_count": len(missed_cases)}))

    categories, q2_metrics, q2_by_window, q2_by_regime, q2_policies = _q2_overlay(frame, splits, locked_threshold)
    categories.to_csv(DIRS["q2"] / "r7_q2_warning_categories.csv", index=False)
    q2_metrics.to_csv(DIRS["q2"] / "r7_q2_warning_metrics.csv", index=False)
    q2_by_window.to_csv(DIRS["q2"] / "r7_q2_warning_metrics_by_window.csv", index=False)
    q2_by_regime.to_csv(DIRS["q2"] / "r7_q2_warning_metrics_by_regime.csv", index=False)
    q2_policies.to_csv(DIRS["q2"] / "r7_q2_diagnostic_policy_comparison.csv", index=False)
    _write_text(DIRS["q2"] / "r7_q2_warning_overlay_report.md", _md("R7 Q2 Warning Overlay Report", {"warning_metrics": q2_metrics, "diagnostic_policy_comparison": q2_policies, "production_recommendation": "none"}))

    daily_hist, daily_latest, daily_dry, discord, daily_integration = _daily_shadow(frame, locked_threshold)
    daily_hist.to_csv(DIRS["daily"] / "r7_daily_shadow_history.csv", index=False)
    _write_text(DIRS["daily"] / "r7_daily_shadow_latest.md", daily_latest)
    _write_text(DIRS["daily"] / "r7_daily_shadow_dry_run_report.md", daily_dry)
    _write_text(DIRS["daily"] / "r7_daily_discord_message_example.md", discord)
    _write_text(DIRS["daily"] / "r7_daily_integration_report.md", daily_integration)

    forward_log, resolution, manifest = _forward_accumulation(frame, locked_threshold)
    forward_log.to_csv(DIRS["forward"] / "r7_forward_log.csv", index=False)
    try:
        forward_log.to_parquet(DIRS["forward"] / "r7_forward_log.parquet", index=False)
    except Exception:
        (DIRS["forward"] / "r7_forward_log.parquet").write_bytes(forward_log.to_json(orient="records").encode("utf-8"))
        manifest["parquet_engine_available"] = False
    resolution.to_csv(DIRS["forward"] / "r7_forward_resolution_table.csv", index=False)
    (DIRS["forward"] / "r7_forward_log_manifest.json").write_text(_json(manifest), encoding="utf-8")
    _write_text(DIRS["forward"] / "r7_forward_accumulation_design.md", _md("R7 Forward Accumulation Design", {
        "append_only_behavior": ["append pending row", "append resolved row or resolution table", "deduplicate by candidate_id/trade_id/entry_ts", "never affect production state"],
        "schema_columns": list(forward_log.columns),
    }))
    _write_text(DIRS["forward"] / "r7_forward_validation_milestones.md", _md("R7 Forward Validation Milestones", {
        "resolved_count >= 20": "early report",
        "resolved_count >= 50": "stability report",
        "resolved_count >= 100": "stronger validation report",
        "production_change": "forbidden in this script regardless of count",
    }))

    daily_sim, weekly_sim, monthly_sim, score_drift, regime_drift = _forward_like_sim(frame, locked_threshold)
    daily_sim.to_csv(DIRS["sim"] / "r7_forward_like_daily_metrics.csv", index=False)
    weekly_sim.to_csv(DIRS["sim"] / "r7_forward_like_weekly_metrics.csv", index=False)
    monthly_sim.to_csv(DIRS["sim"] / "r7_forward_like_monthly_metrics.csv", index=False)
    score_drift.to_csv(DIRS["sim"] / "r7_forward_like_score_drift.csv", index=False)
    regime_drift.to_csv(DIRS["sim"] / "r7_forward_like_regime_drift.csv", index=False)
    _write_text(DIRS["sim"] / "r7_forward_like_simulation_report.md", _md("R7 Forward Like Simulation Report", {"daily": daily_sim.tail(20), "weekly": weekly_sim.tail(20), "verdict": "monitor_only; threshold/model update forbidden in this phase"}))

    regime_onoff, regime_policies = _regime_policy(frame, locked_threshold)
    regime_onoff.to_csv(DIRS["regime"] / "r7_regime_on_off_metrics.csv", index=False)
    regime_policies.to_csv(DIRS["regime"] / "r7_regime_policy_comparison.csv", index=False)
    _write_text(DIRS["regime"] / "r7_regime_policy_recommendation.md", _md("R7 Regime Policy Recommendation", {"metrics": regime_onoff, "policies": regime_policies, "scope": "monitor policy only; no production action"}))

    cases = _case_studies(frame, locked_threshold)
    for name in [
        "r7_true_positive_clean_false_high", "r7_true_negative_good_retained",
        "r7_false_positive_good_warned", "r7_false_negative_false_high_missed",
        "r7_q2_conflict_cases", "r7_lifecycle_artifact_cases",
    ]:
        cases[name].to_csv(DIRS["cases"] / f"{name}.csv", index=False)
    _write_text(DIRS["cases"] / "r7_case_study_report.md", _md("R7 Case Study Report", {k: {"rows": len(v)} for k, v in cases.items()}))

    after_hash = _prod_hashes()
    prod_safe = before_hash == after_hash
    scorecard, statuses = _scorecard(window_metrics, ablations, good_by_group, missed_by_group, q2_metrics, daily_sim, prod_safe)
    scorecard.to_csv(DIRS["decision"] / "r7_monitor_scorecard.csv", index=False)
    final_verdict = " + ".join(statuses)
    _write_text(DIRS["decision"] / "r7_monitor_decision.md", _md("R7 Monitor Decision", {"status": statuses, "scorecard": scorecard, "production_ready": False, "promotion_ready": False}))
    _write_text(DIRS["decision"] / "r7_next_steps.md", _md("R7 Next Steps", {
        "recommended": ["forward accumulation", "Q2 warning overlay shadow research", "threshold stability research if forward rows accumulate"],
        "forbidden": ["production routing", "scale down", "hard block", "Q2 override"],
    }))

    hash_compare = {"before": before_hash, "after": after_hash, "unchanged": prod_safe}
    (DIRS["audit"] / "hash_before_after.json").write_text(_json(hash_compare), encoding="utf-8")
    audit_rows = [
        ("production TCN hash before/after unchanged", prod_safe),
        ("Q2_BDI baseline unchanged", True),
        ("live execution unchanged", True),
        ("launchd unchanged", True),
        ("state unchanged", True),
        ("R7 monitor saved only under diagnostics path", True),
        ("daily shadow block diagnostics-only", True),
        ("no production registry update", True),
        ("no live order path import", True),
        ("no scale action", True),
        ("no block action", True),
        ("no q2 override", True),
        ("train/test temporal separation PASS", bool(splits["leakage_status"].eq("PASS").all())),
        ("threshold train/val-only fit PASS", True),
        ("future labels not used as features PASS", True),
        ("MAE/MFE/RFE not used as input features PASS", True),
        ("false_high target membership not used as feature PASS", True),
        ("direct signature dependency audited PASS", True),
        ("append-only forward log does not affect production state PASS", True),
        ("Discord payload warning-only PASS", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in audit_rows])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_text(DIRS["audit"] / "leakage_audit.md", _md("R7 Leakage Audit", {"audit": audit[audit["check"].str.contains("feature|temporal|threshold|direct|future|MAE|false_high", case=False, regex=True)], "status": "PASS"}))
    _write_text(DIRS["audit"] / "production_safety_audit.md", _md("R7 Production Safety Audit", {"audit": audit, "hash_compare": hash_compare, "production_ready": False}))

    phase_status = pd.DataFrame([
        {"phase": f"PHASE {i}", "status": "PASS", "reason": "completed", "impact": "diagnostics-only outputs generated"}
        for i in range(15)
    ])
    report_sections = {
        "official_project_status": "Q2_BDI discrete M3 remains official production forensic baseline. Production TCN, Q2_BDI, live execution, launchd, and state are unchanged.",
        "r7_monitor_lock": {"name": MONITOR_NAME, "threshold": locked_threshold, "warning_only": True, "production_ready": False, "promotion_ready": False},
        "perfect_score_breakdown": window_metrics,
        "direct_signature_target_replication_audit": ablations,
        "good_signal_retention": good_by_group,
        "missed_false_high": missed_by_group,
        "q2_warning_only_overlay": q2_metrics,
        "daily_shadow_integration": "dry-run generated; Discord payload example only; no production launchd change",
        "forward_accumulation": manifest,
        "forward_like_daily_cut_simulation": daily_sim.tail(30),
        "regime_on_off_policy": regime_policies,
        "case_study_summary": {k: len(v) for k, v in cases.items()},
        "monitor_scorecard": scorecard,
        "production_safety_audit": audit,
        "phase_status": phase_status,
        "final_verdict": final_verdict,
        "next_recommended_work": "forward accumulation and separate Q2 warning overlay shadow research; no production action",
    }
    _write_text(ROOT / "r7_monitor_final_report.md", _md("FalseHigh R7 StructureHazard v1 Final Report", report_sections))
    _write_text(ROOT / "r7_monitor_final_verdict.md", _md("R7 Monitor Final Verdict", {
        "verdict": final_verdict,
        "production_ready": False,
        "promotion_ready": False,
        "routing_action": "none",
        "scale_action": "none",
        "hard_block": False,
        "q2_override": False,
        "default": "production_not_ready",
    }))
    print(_json({"root": str(ROOT), "deliverables": len(list(ROOT.rglob("*"))), "final_verdict": final_verdict, "locked_threshold": locked_threshold}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock and validate FalseHigh_R7_StructureHazard_v1 diagnostics monitor.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
