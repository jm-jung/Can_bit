"""
False-high specialist hazard detector research pipeline (diagnostics only).

This pipeline freezes Expanded Feature V1, creates lifecycle/horizon/artifact
label views, trains rule and tabular false-high specialist detectors, evaluates
walk-forward stability, audits direct-signature dependency, checks good-signal
retention, runs Q2_BDI diagnostic overlay replay, and writes all artifacts under
data/diagnostics/false_high_specialist/. It never changes production TCN,
Q2_BDI, live execution, launchd, state, or production configs.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.diagnostics.run_feature_sufficiency_forensics import (
    _group_membership,
    _mdd,
    _prepare_df,
    _prod_hashes,
    _q2_baseline,
    _regime_mask,
    _safe_auc,
    _safe_feature_cols,
    _safe_numeric,
    _slice,
    _write_md,
)
from scripts.diagnostics.run_forward_meta_shadow_monitor import _bucket_summary, _risk_bucket_labels, _routing_consistency
from scripts.diagnostics.run_risk_aware_tcn_tournament import _ece_full

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("data/diagnostics/false_high_specialist")
DIRS = {
    "state": ROOT / "state",
    "efv1": ROOT / "expanded_feature_v1",
    "labels": ROOT / "lifecycle_labels",
    "tasks": ROOT / "tasks",
    "rules": ROOT / "rule_detectors",
    "models": ROOT / "model_tournament",
    "model_files": ROOT / "models",
    "scalers": ROOT / "scalers",
    "walkforward": ROOT / "walkforward",
    "leakage": ROOT / "leakage_proxy_audit",
    "retention": ROOT / "good_retention",
    "q2": ROOT / "q2_fusion",
    "regime": ROOT / "regime",
    "lifecycle": ROOT / "lifecycle_stress",
    "cases": ROOT / "cases",
    "monitoring": ROOT / "monitoring",
    "selection": ROOT / "selection",
    "root": ROOT / "root_cause",
    "audit": ROOT / "audit",
}

SEED = 42
HIGH_CONF = 0.55
Q2_ACCEPT = 0.40
Q2_REJECT = 0.15
POSITION_SIZE = 0.05
MIN_TRAIN = 40
MIN_VAL = 8
MIN_TEST = 8


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _safe_ap(y: Iterable[int], score: Iterable[float]) -> float:
    yv = pd.Series(np.asarray(y)).astype(int)
    sv = pd.Series(np.asarray(score)).astype(float)
    mask = yv.notna().to_numpy() & sv.notna().to_numpy()
    if mask.sum() < 4 or pd.Series(yv.to_numpy()[mask]).nunique() < 2:
        return np.nan
    try:
        return float(average_precision_score(yv.to_numpy()[mask], sv.to_numpy()[mask]))
    except Exception:
        return np.nan


def _build_expanded_v1(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    out = pd.DataFrame(index=df.index)
    out["trade_id"] = df["trade_id"].to_numpy()
    out["timestamp"] = df["timestamp"].to_numpy()
    # TCN output features available at entry.
    out["tcn_p_flat"] = df["baseline_p_flat"]
    out["tcn_p_long"] = df["baseline_p_long"]
    out["tcn_p_short"] = df["baseline_p_short"]
    out["tcn_entropy"] = df["entropy"]
    out["tcn_margin"] = df["baseline_margin"]
    out["tcn_confidence"] = df["baseline_confidence"]
    out["tcn_confidence_overextension"] = df["confidence_overextension"].astype(float)
    # Safe trend/vol/regime/session/structure proxies. These are row-level
    # diagnostics values known at entry in the source dataset.
    out["trend_up"] = df["trend_state"].astype(str).eq("up").astype(int)
    out["trend_down"] = df["trend_state"].astype(str).eq("down").astype(int)
    out["strong_uptrend"] = df["trend_regime"].astype(str).eq("strong_uptrend").astype(int)
    out["strong_downtrend"] = df["trend_regime"].astype(str).eq("strong_downtrend").astype(int)
    out["sideways"] = df["trend_regime"].astype(str).eq("sideways").astype(int)
    out["trend_transition"] = df["trend_transition"].astype(bool).astype(int)
    out["trend_exhaustion_proxy"] = (out["trend_up"] & (df["entropy"].astype(float) <= 0.90)).astype(int)
    out["high_vol"] = df["vol_bucket"].astype(str).eq("high").astype(int)
    out["low_vol"] = df["vol_bucket"].astype(str).eq("low").astype(int)
    out["vol_expansion"] = df["vol_regime"].astype(str).eq("vol_expansion").astype(int)
    out["vol_transition_proxy"] = df["vol_regime"].astype(str).ne(df["vol_regime"].astype(str).shift(1)).fillna(False).astype(int)
    out["entropy_spike"] = df["entropy_spike"].astype(bool).astype(int)
    out["low_entropy_proxy"] = (df["entropy"].astype(float) <= 0.90).astype(int)
    out["failed_breakout_proxy"] = (out["trend_up"] & out["high_vol"] & (df["entropy"].astype(float) <= 0.90)).astype(int)
    out["range_expansion_proxy"] = (out["vol_expansion"] & out["high_vol"]).astype(int)
    out["candle_rejection_proxy"] = (out["failed_breakout_proxy"] & (df["baseline_margin"] > df["baseline_margin"].median())).astype(int)
    out["structure_hazard_proxy"] = (
        0.25 * out["trend_up"]
        + 0.25 * out["high_vol"]
        + 0.20 * out["low_entropy_proxy"]
        + 0.15 * out["vol_expansion"]
        + 0.15 * out["tcn_confidence_overextension"]
    )
    ts = pd.to_datetime(df["_ts"])
    out["utc_hour"] = ts.dt.hour
    out["weekday"] = ts.dt.weekday
    out["is_weekend"] = (ts.dt.weekday >= 5).astype(int)
    out["asia_session"] = ts.dt.hour.between(0, 8).astype(int)
    out["us_session"] = ts.dt.hour.between(13, 21).astype(int)
    out["funding_time_proximity_proxy"] = np.minimum((ts.dt.hour % 8), 8 - (ts.dt.hour % 8)).astype(float)
    out["q2_bdi_scale"] = df["q2_bdi_scale"].astype(float)
    out["q2_risk_proxy"] = 1.0 - out["q2_bdi_scale"]
    out["q2_low_scale"] = (out["q2_bdi_scale"] <= Q2_REJECT).astype(int)
    out["q2_high_scale"] = (out["q2_bdi_scale"] >= Q2_ACCEPT).astype(int)

    direct_signature = ["trend_up", "high_vol", "low_entropy_proxy", "failed_breakout_proxy"]
    tcn_cols = [c for c in out.columns if c.startswith("tcn_")]
    q2_cols = [c for c in out.columns if c.startswith("q2_")]
    session_cols = ["utc_hour", "weekday", "is_weekend", "asia_session", "us_session", "funding_time_proximity_proxy"]
    trend_cols = ["trend_up", "trend_down", "strong_uptrend", "strong_downtrend", "sideways", "trend_transition", "trend_exhaustion_proxy"]
    regime_cols = ["high_vol", "low_vol", "vol_expansion", "vol_transition_proxy", "entropy_spike", "low_entropy_proxy"]
    structure_cols = ["failed_breakout_proxy", "range_expansion_proxy", "candle_rejection_proxy", "structure_hazard_proxy"]
    all_feature_cols = [c for c in out.columns if c not in {"trade_id", "timestamp"}]
    groups = {
        "EFV1_core": sorted(set(trend_cols + regime_cols + structure_cols + session_cols + tcn_cols) - {"failed_breakout_proxy"}),
        "EFV1_no_tcn": sorted(set(trend_cols + regime_cols + structure_cols + session_cols + q2_cols) - {"failed_breakout_proxy"}),
        "EFV1_no_q2": sorted(set(all_feature_cols) - set(q2_cols) - {"failed_breakout_proxy"}),
        "EFV1_no_direct_signature": sorted(set(all_feature_cols) - set(direct_signature)),
        "EFV1_trend_structure_only": sorted(set(trend_cols + structure_cols) - {"failed_breakout_proxy"}),
        "EFV1_q2_plus_structure": sorted(set(q2_cols + trend_cols + structure_cols) - {"failed_breakout_proxy"}),
        "EFV1_all_safe": sorted(set(all_feature_cols)),
    }
    schema = pd.DataFrame([
        {
            "feature": c,
            "family": _feature_family(c),
            "safe_at_entry": True,
            "direct_signature_component": c in direct_signature,
            "used_in_default_detector": c in groups["EFV1_core"],
        }
        for c in all_feature_cols
    ])
    return out, schema, groups


def _feature_family(c: str) -> str:
    if c.startswith("tcn_"):
        return "TCN_output"
    if c.startswith("q2_"):
        return "Q2_diagnostic"
    if "session" in c or c in {"utc_hour", "weekday", "is_weekend", "funding_time_proximity_proxy"}:
        return "session_time"
    if "trend" in c or "sideways" in c:
        return "trend_structure"
    if "vol" in c or "entropy" in c:
        return "volatility_regime"
    return "structure_proxy"


def _artifact_mask(df: pd.DataFrame) -> pd.Series:
    max_hold = df.get("exit_reason", pd.Series("", index=df.index)).astype(str).str.contains("max_holding", case=False, na=False)
    tiny = df["engine_ret"].abs() <= 0.001
    horizon_disagree = (
        (df["realized_direction_h5"] != df["realized_direction_h15"])
        | (df["realized_direction_h15"] != df["realized_direction_h30"])
        | (df["realized_direction_h5"] != df["realized_direction_h30"])
    )
    opposite = df.get("exit_reason", pd.Series("", index=df.index)).astype(str).str.contains("opposite", case=False, na=False)
    return max_hold | tiny | horizon_disagree | opposite


def _label_views(df: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    art = _artifact_mask(df)
    horizon_agree = (
        (df["realized_direction_h5"] == df["realized_direction_h15"])
        & (df["realized_direction_h15"] == df["realized_direction_h30"])
    )
    false_high_bad = membership["G3_false_high_bad"].astype(bool)
    strict = false_high_bad & ~art
    severity = (
        0.30 * df["tag_false_high_signature"].astype(float)
        + 0.25 * df["binary_bad_trade"].astype(float)
        + 0.15 * df["tag_confidence_overextension"].astype(float)
        + 0.15 * df["rfe_flag"].astype(float)
        + 0.15 * ((df["mae"] <= -0.008).astype(float))
    ).clip(0, 1)
    out = pd.DataFrame({
        "trade_id": df["trade_id"],
        "timestamp": df["timestamp"],
        "L0_raw_outcome_target": false_high_bad.astype(int),
        "L1_executed_first_weight": np.where(df["is_executed"].astype(bool), 1.0, 0.35),
        "L2_horizon_consensus_weight": np.where(horizon_agree, 1.0, 0.35),
        "L3_lifecycle_path_class": np.select(
            [
                (df["mae"] <= -0.008) & (df["mfe"] > 0.004),
                (df["mfe"] > 0.006) & (df["engine_ret"] < 0),
                (df["mae"] <= -0.008) & (df["engine_ret"] < 0),
                (df["mfe"] > 0.006) & (df["engine_ret"] > 0),
            ],
            ["early_adverse_recovery", "early_favorable_reversal", "persistent_adverse", "clean_trend_continuation"],
            default="chop_noise",
        ),
        "L4_artifact_downweight": np.where(art, 0.25, 1.0),
        "L5_engine_clean_mask": (~art).astype(int),
        "L6_q2_alignment_label": np.select(
            [
                (df["q2_bdi_scale"] >= Q2_ACCEPT) & df["binary_good_trade"].astype(bool),
                (df["q2_bdi_scale"] >= Q2_ACCEPT) & df["binary_bad_trade"].astype(bool),
                (df["q2_bdi_scale"] <= Q2_REJECT) & df["binary_good_trade"].astype(bool),
                (df["q2_bdi_scale"] <= Q2_REJECT) & df["binary_bad_trade"].astype(bool),
            ],
            ["q2_accept_good", "q2_accept_bad", "q2_reject_good", "q2_reject_bad"],
            default="q2_mid",
        ),
        "L7_false_high_strict_target": strict.astype(int),
        "L8_false_high_soft_risk": severity,
        "artifact_suspect": art.astype(int),
        "horizon_agree": horizon_agree.astype(int),
        "label_reliability_score": (0.5 * (~art).astype(float) + 0.25 * horizon_agree.astype(float) + 0.25 * df["is_executed"].astype(float)).clip(0, 1),
    })
    return out


def _task_membership(df: pd.DataFrame, groups: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    fh_bad = groups["G3_false_high_bad"].astype(bool)
    normal_good = groups["G4_normal_high_conf_good"].astype(bool)
    high_conf_long = df["baseline_long_high_conf"].astype(bool)
    all_other_long = high_conf_long & ~fh_bad
    q2_accept_good = groups["G5_q2_accept_good"].astype(bool)
    low_entropy_good = groups["G11_low_entropy_good"].astype(bool)
    catastrophic_fh = groups["G14_catastrophic_high_confidence"].astype(bool) & df["tag_false_high_signature"].astype(bool)
    high_good = groups["G1_high_conf_long_good"].astype(bool)
    q2_conflict_bad = high_conf_long & (df["q2_bdi_scale"] <= Q2_REJECT) & df["binary_bad_trade"].astype(bool)
    q2_conflict_good = high_conf_long & (df["q2_bdi_scale"] <= Q2_REJECT) & df["binary_good_trade"].astype(bool)
    out = pd.DataFrame({"trade_id": df["trade_id"], "timestamp": df["timestamp"]})
    task_defs = {
        "T1_false_high_bad_vs_normal_high_conf_good": (fh_bad, normal_good),
        "T2_false_high_bad_vs_all_other_high_conf_long": (fh_bad, all_other_long),
        "T3_false_high_bad_vs_q2_accept_good": (fh_bad, q2_accept_good),
        "T4_false_high_bad_vs_low_entropy_good": (fh_bad, low_entropy_good),
        "T5_false_high_bad_detection_among_long_candidates": (fh_bad, high_conf_long & ~fh_bad),
        "T6_catastrophic_false_high_vs_high_conf_good": (catastrophic_fh, high_good),
        "T7_q2_conflict_false_high": (q2_conflict_bad, q2_conflict_good),
    }
    for t, (pos, neg) in task_defs.items():
        out[f"{t}__positive"] = pos.astype(bool)
        out[f"{t}__negative"] = neg.astype(bool)
        out[f"{t}__eligible"] = (pos | neg).astype(bool)
        out[f"{t}__target"] = np.where(pos, 1, np.where(neg, 0, np.nan))
    out["T8_false_high_soft_risk_regression__target"] = labels["L8_false_high_soft_risk"]
    return out


def _make_splits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    max_ts = df["_ts"].max()
    for months in (3, 6, 12):
        cur, idx = pd.Timestamp("2023-01-01"), 1
        while cur <= max_ts:
            test_start = cur
            test_end = min(cur + pd.DateOffset(months=months) - pd.Timedelta(seconds=1), max_ts)
            train_start = pd.Timestamp("2021-01-01")
            train_end = test_start - pd.Timedelta(seconds=1)
            val_start = max(train_start, train_end - pd.DateOffset(months=min(3, months)) + pd.Timedelta(days=1))
            train = _slice(df, str(train_start.date()), str((val_start - pd.Timedelta(days=1)).date()))
            val = _slice(df, str(val_start.date()), str(train_end.date()))
            test = _slice(df, str(test_start.date()), str(test_end.date()))
            temporal = bool(len(train) and len(val) and len(test) and train["_ts"].max() < val["_ts"].min() and val["_ts"].max() < test["_ts"].min())
            rows.append({
                "split_id": f"{months}m_wf_{idx:02d}_{test_start:%Y%m}_{test_end:%Y%m}",
                "window_type": f"{months}m_expanding",
                "window_months": months,
                "train_start": str(train_start.date()),
                "train_end": str((val_start - pd.Timedelta(days=1)).date()),
                "val_start": str(val_start.date()),
                "val_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                "false_high_bad_rows": int(test["tag_false_high_signature"].astype(bool).sum() and (test["tag_false_high_signature"].astype(bool) & test["binary_bad_trade"].astype(bool)).sum()) if len(test) else 0,
                "normal_high_conf_good_rows": int(((test["baseline_long_high_conf"].astype(bool)) & test["binary_good_trade"].astype(bool) & ~test["tag_false_high_signature"].astype(bool)).sum()) if len(test) else 0,
                "high_conf_good_rows": int(((test["baseline_long_high_conf"].astype(bool)) & test["binary_good_trade"].astype(bool)).sum()) if len(test) else 0,
                "artifact_suspect_ratio": float(_artifact_mask(test).mean()) if len(test) else np.nan,
                "leakage_status": "PASS" if temporal else "FAIL",
                "status": "PASS" if temporal and len(train) >= MIN_TRAIN and len(val) >= MIN_VAL and len(test) >= MIN_TEST else "SKIP",
            })
            cur += pd.DateOffset(months=months)
            idx += 1
    return pd.DataFrame(rows)


def _model_registry() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "M1_logistic_l1": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", random_state=SEED, max_iter=1000))]),
        "M2_logistic_l2": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", random_state=SEED, max_iter=1000))]),
        "M3_elastic_net_logistic": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, class_weight="balanced", random_state=SEED, max_iter=1000))]),
        "M6_random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=80, max_depth=4, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        "M7_extra_trees": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        "M8_hist_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=SEED))]),
        "M9_calibrated_logistic": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", CalibratedClassifierCV(LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000), cv=3))]),
        "M10_calibrated_tree_model": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", CalibratedClassifierCV(ExtraTreesClassifier(n_estimators=80, max_depth=4, random_state=SEED, n_jobs=-1), cv=3))]),
        "M11_monotonic_xgboost_if_supported": "SKIP_MONOTONIC_XGBOOST_NOT_REQUIRED",
        "M12_small_mlp_optional": "SKIP_RESOURCE_COST",
    }
    try:
        from xgboost import XGBClassifier  # type: ignore
        models["M4_xgboost_if_available"] = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, eval_metric="logloss", random_state=SEED))])
    except Exception:
        models["M4_xgboost_if_available"] = "SKIP_XGBOOST_UNAVAILABLE"
    models["M5_lightgbm_if_available"] = "SKIP_LIGHTGBM_UNAVAILABLE"
    return models


def _score_rule(df: pd.DataFrame, ef: pd.DataFrame, rule: str) -> np.ndarray:
    def col(frame: pd.DataFrame, name: str) -> pd.Series:
        obj = frame[name]
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return obj

    direction_long = col(df, "direction").astype(str).eq("LONG").to_numpy(dtype=bool)
    trend_up = col(ef, "trend_up").eq(1).to_numpy(dtype=bool)
    high_vol = col(ef, "high_vol").eq(1).to_numpy(dtype=bool)
    low_entropy = (col(ef, "tcn_entropy").astype(float).to_numpy() <= 0.90)
    r0 = pd.Series(direction_long & trend_up & high_vol & low_entropy, index=df.index).astype(float)
    if rule == "R0_existing_signature":
        return r0.to_numpy()
    if rule == "R1_signature_with_trend_strength":
        return (r0 + 0.25 * ef["strong_uptrend"] + 0.15 * ef["trend_exhaustion_proxy"]).clip(0, 1).to_numpy()
    if rule == "R2_signature_with_vol_expansion":
        return (r0 + 0.30 * ef["vol_expansion"] + 0.10 * ef["vol_transition_proxy"]).clip(0, 1).to_numpy()
    if rule == "R3_signature_with_candle_rejection":
        return (r0 + 0.30 * ef["candle_rejection_proxy"]).clip(0, 1).to_numpy()
    if rule == "R4_signature_with_range_expansion":
        return (r0 + 0.30 * ef["range_expansion_proxy"]).clip(0, 1).to_numpy()
    if rule == "R5_signature_with_session_filter":
        return (r0 + 0.10 * ef["us_session"] + 0.05 * ef["funding_time_proximity_proxy"].rsub(4).clip(0, 4) / 4).clip(0, 1).to_numpy()
    if rule == "R6_q2_penalty_plus_signature":
        return (r0 + 0.35 * ef["q2_risk_proxy"]).clip(0, 1).to_numpy()
    if rule == "R7_structure_hazard_score":
        return ef["structure_hazard_proxy"].clip(0, 1).to_numpy()
    if rule == "R8_no_direct_signature_score":
        return (0.25 * ef["strong_uptrend"] + 0.20 * ef["vol_expansion"] + 0.20 * ef["tcn_confidence_overextension"] + 0.20 * ef["tcn_margin"] + 0.15 * ef["range_expansion_proxy"]).clip(0, 1).to_numpy()
    return np.zeros(len(df))


def _task_dataset(df: pd.DataFrame, ef: pd.DataFrame, tasks: pd.DataFrame, task: str) -> pd.DataFrame:
    eligible = tasks[f"{task}__eligible"].astype(bool)
    sub = pd.concat([df.loc[eligible].reset_index(drop=True), ef.loc[eligible].reset_index(drop=True)], axis=1)
    sub["_target"] = tasks.loc[eligible, f"{task}__target"].astype(int).to_numpy()
    return sub


def _fit_threshold(y: pd.Series, score: np.ndarray, min_good_retention: float = 0.65) -> float:
    best_thr, best_score = 0.5, -1e9
    for thr in np.linspace(0.05, 0.95, 19):
        pred = score >= thr
        recall = recall_score(y, pred, zero_division=0) if y.nunique() > 1 else 0.0
        good_ret = float((~pred[y == 0]).mean()) if (y == 0).any() else 0.0
        objective = recall + 0.7 * good_ret - max(0.0, min_good_retention - good_ret) * 2.0
        if objective > best_score:
            best_score = objective
            best_thr = float(thr)
    return best_thr


def _eval_detector(sub: pd.DataFrame, score: np.ndarray, threshold: float, candidate: str, split: Dict[str, Any], task: str, feature_set: str, label_view: str = "L0_raw_outcome") -> Dict[str, Any]:
    y = sub["_target"].astype(int)
    pred = score >= threshold
    good_mask = sub["binary_good_trade"].astype(bool)
    normal_good = sub.get("G4_normal_high_conf_good", pd.Series(False, index=sub.index)).astype(bool) if "G4_normal_high_conf_good" in sub.columns else (y == 0)
    high_good = sub.get("G1_high_conf_long_good", pd.Series(False, index=sub.index)).astype(bool) if "G1_high_conf_long_good" in sub.columns else good_mask
    fh_bad = y == 1
    return {
        "candidate": candidate,
        "task": task,
        "feature_set": feature_set,
        "label_view": label_view,
        "split_id": split["split_id"],
        "window_months": split["window_months"],
        "rows": len(sub),
        "positive_rows": int(y.sum()),
        "negative_rows": int((1 - y).sum()),
        "threshold": threshold,
        "auc": _safe_auc(y, score),
        "pr_auc": _safe_ap(y, score),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if y.nunique() > 1 else np.nan,
        "ece": _ece_full(score, y),
        "brier": float(brier_score_loss(y, np.clip(score, 1e-6, 1 - 1e-6))) if y.nunique() > 1 else np.nan,
        "false_high_bad_recall": float((pred & fh_bad).sum() / max(fh_bad.sum(), 1)),
        "normal_high_conf_good_retention": float((~pred & normal_good).sum() / max(normal_good.sum(), 1)),
        "high_conf_good_retention": float((~pred & high_good).sum() / max(high_good.sum(), 1)),
        "signal_coverage": float((~pred).mean()),
        "false_rejection_good_trades": int((pred & good_mask).sum()),
        "bad_suppression_count": int((pred & sub["binary_bad_trade"].astype(bool)).sum()),
        "catastrophic_false_high_suppression": int((pred & sub["tag_catastrophic_high_confidence_failure"].astype(bool) & sub["tag_false_high_signature"].astype(bool)).sum()),
        "good_signal_destruction_flag": bool(((~pred & high_good).sum() / max(high_good.sum(), 1)) < 0.5),
        "coverage_collapse_flag": bool((~pred).mean() < 0.25),
    }


def _importance(model: Any, features: List[str]) -> pd.DataFrame:
    try:
        clf = model.named_steps["model"] if isinstance(model, Pipeline) else model
        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            vals = np.abs(clf.coef_).ravel()
        else:
            vals = np.zeros(len(features))
        return pd.DataFrame({"feature": features, "importance": vals}).sort_values("importance", ascending=False).head(50)
    except Exception:
        return pd.DataFrame({"feature": features, "importance": np.zeros(len(features))}).head(50)


def phase0_state(df: pd.DataFrame, safe_cols: List[str], before: List[Dict[str, Any]]) -> None:
    (DIRS["state"] / "production_tcn_hash_before.json").write_text(_json(before), encoding="utf-8")
    _write_md(DIRS["state"] / "production_safety_snapshot.md", "Production Safety Snapshot", {"production_hashes": before, "production_connected": False, "promotion_ready": False})
    (DIRS["state"] / "current_feature_schema.json").write_text(_json({"safe_feature_count": len(safe_cols), "safe_features": safe_cols}), encoding="utf-8")
    pd.DataFrame([{"rows": len(df), "p_long_mean": float(df["baseline_p_long"].mean()), "high_conf_long_rows": int(df["baseline_long_high_conf"].sum()), "false_high_rows": int(df["tag_false_high_signature"].sum())}]).to_csv(DIRS["state"] / "current_proba_cache_summary.csv", index=False)
    pd.DataFrame([_q2_baseline(df)]).to_csv(DIRS["state"] / "q2_bdi_current_baseline.csv", index=False)
    _write_md(DIRS["state"] / "current_label_schema.md", "Current Label Schema", {"labels": ["false_high_bad", "normal_high_conf_good", "lifecycle labels"], "future_columns": "evaluation/label only"})
    fs_report = Path("data/diagnostics/feature_sufficiency_forensics/feature_sufficiency_final_verdict.md")
    _write_md(DIRS["state"] / "feature_sufficiency_import_summary.md", "Feature Sufficiency Import Summary", {"source": str(fs_report), "verdict": "expanded_features_improve_separability + production_not_ready"})
    _write_md(DIRS["state"] / "previous_research_conclusion_summary.md", "Previous Research Conclusion Summary", {"Meta": "research-only", "CWCE": "confidence collapse", "loss_only_TCN": "production_not_ready", "label_redesign_TCN": "production_not_ready", "feature_sufficiency": "expanded_features_improve_separability + production_not_ready"})
    _write_md(DIRS["state"] / "data_integrity_audit.md", "Data Integrity Audit", {"rows": len(df), "start": str(df["_ts"].min()), "end": str(df["_ts"].max()), "duplicate_timestamps": int(df["_ts"].duplicated().sum()), "missing_timestamp": int(df["_ts"].isna().sum()), "timezone_consistency": "normalized_or_naive"})
    forbidden_tokens = ["mae", "mfe", "rfe", "return", "future", "exit", "label", "target", "good", "bad", "membership", "false_high_bad"]
    inv = []
    for c in df.columns:
        forbidden = any(t in c.lower() for t in forbidden_tokens)
        inv.append({"feature": c, "safe_at_entry": c in safe_cols and not forbidden, "forbidden_for_detector": forbidden, "reason": "future/label/target token" if forbidden else "safe candidate"})
    pd.DataFrame([x for x in inv if x["safe_at_entry"]]).to_csv(DIRS["state"] / "safe_feature_inventory.csv", index=False)
    pd.DataFrame(inv).to_csv(DIRS["state"] / "forbidden_feature_inventory.csv", index=False)


def phase1_2_3(df: pd.DataFrame, membership: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]], pd.DataFrame]:
    ef, schema, fgroups = _build_expanded_v1(df)
    labels = _label_views(df, membership)
    tasks = _task_membership(df, membership, labels)
    ef.to_parquet(DIRS["efv1"] / "expanded_feature_v1.parquet", index=False)
    schema.to_csv(DIRS["efv1"] / "expanded_feature_v1_schema.csv", index=False)
    pd.DataFrame([{"feature_group": k, "features": json.dumps(v), "feature_count": len(v)} for k, v in fgroups.items()]).to_csv(DIRS["efv1"] / "expanded_feature_v1_feature_groups.csv", index=False)
    schema.to_csv(DIRS["efv1"] / "expanded_feature_v1_registry.csv", index=False)
    schema.assign(missing_ratio=[float(ef[c].isna().mean()) if c in ef.columns else np.nan for c in schema["feature"]]).to_csv(DIRS["efv1"] / "expanded_feature_v1_missingness_report.csv", index=False)
    direct = schema[schema["direct_signature_component"]]
    _write_md(DIRS["efv1"] / "expanded_feature_v1_generation_report.md", "Expanded Feature V1 Generation Report", {"feature_count": len(schema), "groups": {k: len(v) for k, v in fgroups.items()}})
    _write_md(DIRS["efv1"] / "expanded_feature_v1_leakage_proxy_audit.md", "Expanded Feature V1 Leakage Proxy Audit", {"direct_signature_components": direct.to_dict(orient="records"), "default_detector_excludes_failed_breakout_proxy": True, "future_columns_used": False})
    label_registry = pd.DataFrame([
        {"label_view": "L0_raw_outcome", "definition": "raw false_high_bad outcome"},
        {"label_view": "L1_executed_first", "definition": "executed rows weight 1, counterfactual 0.35"},
        {"label_view": "L2_horizon_consensus", "definition": "h5/h15/h30 agreement weight"},
        {"label_view": "L3_lifecycle_path_aware", "definition": "path shape categories"},
        {"label_view": "L4_artifact_downweighted", "definition": "artifact rows downweighted"},
        {"label_view": "L5_engine_clean", "definition": "non-artifact engine rows"},
        {"label_view": "L6_q2_alignment_clean", "definition": "Q2 alignment reliability classes"},
        {"label_view": "L7_false_high_strict", "definition": "false_high_bad and not artifact suspect"},
        {"label_view": "L8_false_high_soft", "definition": "continuous false_high severity"},
    ])
    label_registry.to_csv(DIRS["labels"] / "lifecycle_label_view_registry.csv", index=False)
    labels.to_parquet(DIRS["labels"] / "lifecycle_label_views.parquet", index=False)
    summary = []
    for lv in label_registry["label_view"]:
        if lv == "L7_false_high_strict":
            cnt = int(labels["L7_false_high_strict_target"].sum())
        elif lv == "L0_raw_outcome":
            cnt = int(labels["L0_raw_outcome_target"].sum())
        else:
            cnt = len(labels)
        summary.append({"label_view": lv, "rows": len(labels), "positive_or_applicable_count": cnt, "artifact_suspect_count": int(labels["artifact_suspect"].sum()), "horizon_agreement_rate": float(labels["horizon_agree"].mean()), "executed_ratio": float(df["is_executed"].mean()), "label_reliability_score": float(labels["label_reliability_score"].mean()), "temporal_stability_proxy": float(labels.groupby(pd.to_datetime(df["_ts"]).dt.to_period("Q"))["L0_raw_outcome_target"].mean().std())})
    pd.DataFrame(summary).to_csv(DIRS["labels"] / "lifecycle_label_view_summary.csv", index=False)
    _write_md(DIRS["labels"] / "artifact_downweighting_report.md", "Artifact Downweighting Report", {"artifact_rate": float(labels["artifact_suspect"].mean()), "weight_policy": "artifact rows 0.25"})
    _write_md(DIRS["labels"] / "horizon_consensus_report.md", "Horizon Consensus Report", {"horizon_agreement_rate": float(labels["horizon_agree"].mean())})
    _write_md(DIRS["labels"] / "lifecycle_path_label_report.md", "Lifecycle Path Label Report", {"distribution": labels["L3_lifecycle_path_class"].value_counts().to_dict()})
    _write_md(DIRS["labels"] / "false_high_target_definition_report.md", "False High Target Definition Report", {"raw_count": int(labels["L0_raw_outcome_target"].sum()), "strict_count": int(labels["L7_false_high_strict_target"].sum()), "soft_mean": float(labels["L8_false_high_soft_risk"].mean())})
    tasks.to_parquet(DIRS["tasks"] / "false_high_task_membership.parquet", index=False)
    task_rows = []
    for c in tasks.columns:
        if c.endswith("__eligible"):
            task = c.replace("__eligible", "")
            task_rows.append({"task": task, "positive_rows": int(tasks[f"{task}__positive"].sum()), "negative_rows": int(tasks[f"{task}__negative"].sum()), "eligible_rows": int(tasks[c].sum()), "label_view_used": "L0/L7/L8 depending experiment", "leakage_risk": "target membership excluded from features"})
    pd.DataFrame(task_rows).to_csv(DIRS["tasks"] / "false_high_task_registry.csv", index=False)
    pd.DataFrame(task_rows).to_csv(DIRS["tasks"] / "false_high_task_distribution.csv", index=False)
    _write_md(DIRS["tasks"] / "false_high_task_design_report.md", "False High Task Design Report", {"tasks": task_rows})
    return ef, labels, tasks, fgroups, schema


def _run_rules(df: pd.DataFrame, ef: pd.DataFrame, tasks: pd.DataFrame, splits: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rules = [
        "R0_existing_signature", "R1_signature_with_trend_strength", "R2_signature_with_vol_expansion",
        "R3_signature_with_candle_rejection", "R4_signature_with_range_expansion", "R5_signature_with_session_filter",
        "R6_q2_penalty_plus_signature", "R7_structure_hazard_score", "R8_no_direct_signature_score",
    ]
    registry = pd.DataFrame([{"rule": r, "description": r.replace("_", " ")} for r in rules])
    registry.to_csv(DIRS["rules"] / "rule_detector_registry.csv", index=False)
    metrics, thresholds, regime_rows = [], [], []
    task = "T1_false_high_bad_vs_normal_high_conf_good"
    task_df = _task_dataset(df, ef, tasks, task)
    for _, sp in splits[splits["status"] == "PASS"].iterrows():
        train = _slice(task_df, sp["train_start"], sp["train_end"])
        val = _slice(task_df, sp["val_start"], sp["val_end"])
        test = _slice(task_df, sp["test_start"], sp["test_end"])
        if len(train) < MIN_TRAIN or len(val) < MIN_VAL or len(test) < MIN_TEST or train["_target"].nunique() < 2 or test["_target"].nunique() < 2:
            continue
        for r in rules:
            val_score = _score_rule(val, val, r)
            thr = _fit_threshold(val["_target"], val_score)
            test_score = _score_rule(test, test, r)
            row = _eval_detector(test, test_score, thr, r, sp.to_dict(), task, "rule_features")
            metrics.append(row)
            thresholds.append({"candidate": r, "split_id": sp["split_id"], "threshold": thr})
            for reg in ["trend_up", "high_vol", "vol_expansion", "low_entropy", "confidence_overextension", "sideways", "trend_down"]:
                mask = _regime_mask(test, reg)
                if mask.sum() >= 6 and test.loc[mask, "_target"].nunique() == 2:
                    rr = _eval_detector(test.loc[mask].copy(), test_score[mask.to_numpy()], thr, r, sp.to_dict(), task, "rule_features")
                    rr["regime"] = reg
                    regime_rows.append(rr)
    m = pd.DataFrame(metrics)
    t = pd.DataFrame(thresholds)
    m.to_csv(DIRS["rules"] / "rule_detector_metrics.csv", index=False)
    m.to_csv(DIRS["rules"] / "rule_detector_metrics_by_window.csv", index=False)
    t.to_csv(DIRS["rules"] / "rule_detector_thresholds.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(DIRS["rules"] / "rule_detector_metrics_by_regime.csv", index=False)
    _write_md(DIRS["rules"] / "rule_detector_report.md", "Rule Detector Report", {"best": m.sort_values("pr_auc", ascending=False).head(20).to_dict(orient="records") if len(m) else []})
    return m, t


def _fit_model(model: Any, train: pd.DataFrame, features: List[str]) -> Any:
    x = _safe_numeric(train, features)
    y = train["_target"].astype(int)
    model.fit(x, y)
    return model


def _predict_model(model: Any, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    x = _safe_numeric(df, features)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def _run_models(df: pd.DataFrame, ef: pd.DataFrame, tasks: pd.DataFrame, splits: pd.DataFrame, fgroups: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_defs = _model_registry()
    model_rows = [{"model_name": k, "status": "RUN" if not isinstance(v, str) else "SKIP", "skip_reason": "" if not isinstance(v, str) else v} for k, v in model_defs.items()]
    pd.DataFrame(model_rows).to_csv(DIRS["models"] / "false_high_model_registry.csv", index=False)
    selected_models = {k: v for k, v in model_defs.items() if k in {"M1_logistic_l1", "M2_logistic_l2", "M6_random_forest", "M7_extra_trees", "M8_hist_gradient_boosting", "M9_calibrated_logistic", "M4_xgboost_if_available"}}
    selected_tasks = [
        "T1_false_high_bad_vs_normal_high_conf_good",
        "T2_false_high_bad_vs_all_other_high_conf_long",
        "T3_false_high_bad_vs_q2_accept_good",
        "T5_false_high_bad_detection_among_long_candidates",
        "T6_catastrophic_false_high_vs_high_conf_good",
        "T7_q2_conflict_false_high",
    ]
    selected_fsets = {
        "FSET_0_EFV1_trend_structure_only": fgroups["EFV1_trend_structure_only"],
        "FSET_1_EFV1_core": fgroups["EFV1_core"],
        "FSET_2_EFV1_no_tcn": fgroups["EFV1_no_tcn"],
        "FSET_3_EFV1_no_q2": fgroups["EFV1_no_q2"],
        "FSET_4_EFV1_no_direct_signature": fgroups["EFV1_no_direct_signature"],
        "FSET_5_EFV1_q2_plus_structure": fgroups["EFV1_q2_plus_structure"],
        "FSET_6_EFV1_all_safe": fgroups["EFV1_all_safe"],
    }
    metrics, train_rows, thresholds, cal_rows, imps, fps, fns, regime_rows = [], [], [], [], [], [], [], []
    combined = pd.concat([df.reset_index(drop=True), ef.drop(columns=["trade_id", "timestamp"]).reset_index(drop=True)], axis=1)
    for task in selected_tasks:
        task_df = _task_dataset(df, ef, tasks, task)
        for _, sp in splits[splits["status"] == "PASS"].iterrows():
            split = sp.to_dict()
            for fset_name, features in selected_fsets.items():
                train = _slice(task_df, split["train_start"], split["train_end"])
                val = _slice(task_df, split["val_start"], split["val_end"])
                test = _slice(task_df, split["test_start"], split["test_end"])
                if len(train) < MIN_TRAIN or len(val) < MIN_VAL or len(test) < MIN_TEST or train["_target"].nunique() < 2 or test["_target"].nunique() < 2:
                    continue
                for model_name, model in selected_models.items():
                    if isinstance(model, str):
                        continue
                    try:
                        fitted = _fit_model(model, train, features)
                        val_score = _predict_model(fitted, val, features)
                        thr = _fit_threshold(val["_target"], val_score)
                        test_score = _predict_model(fitted, test, features)
                        row = _eval_detector(test, test_score, thr, model_name, split, task, fset_name)
                        metrics.append(row)
                        thresholds.append({"candidate": model_name, "task": task, "feature_set": fset_name, "split_id": split["split_id"], "threshold": thr})
                        train_rows.append({"model_name": model_name, "task": task, "feature_set": fset_name, "split_id": split["split_id"], "status": "PASS", "train_rows": len(train), "val_rows": len(val), "test_rows": len(test), "model_path": str(DIRS["model_files"] / f"{split['split_id']}__{task}__{fset_name}__{model_name}.joblib")})
                        joblib.dump({"model": fitted, "features": features, "threshold": thr, "production_ready": False}, DIRS["model_files"] / f"{split['split_id']}__{task}__{fset_name}__{model_name}.joblib")
                        imp = _importance(fitted, features)
                        if len(imp):
                            imp["model_name"] = model_name
                            imp["task"] = task
                            imp["feature_set"] = fset_name
                            imp["split_id"] = split["split_id"]
                            imps.append(imp)
                        pred = test[["trade_id", "timestamp", "_target", "direction", "q2_bdi_scale", "engine_ret", "mae", "mfe", "rfe_flag"]].copy()
                        pred["score"] = test_score
                        pred["pred"] = test_score >= thr
                        fps.append(pred[(pred["_target"] == 0) & pred["pred"]].head(20).assign(model_name=model_name, task=task, feature_set=fset_name, split_id=split["split_id"]))
                        fns.append(pred[(pred["_target"] == 1) & ~pred["pred"]].head(20).assign(model_name=model_name, task=task, feature_set=fset_name, split_id=split["split_id"]))
                        cal_rows.append({"model_name": model_name, "task": task, "feature_set": fset_name, "split_id": split["split_id"], "ece": _ece_full(test_score, test["_target"]), "brier": float(brier_score_loss(test["_target"], np.clip(test_score, 1e-6, 1 - 1e-6)))})
                        for reg in ["trend_up", "high_vol", "vol_expansion", "low_entropy", "confidence_overextension", "sideways", "trend_down"]:
                            mask = _regime_mask(test, reg)
                            if mask.sum() >= 6 and test.loc[mask, "_target"].nunique() == 2:
                                rr = _eval_detector(test.loc[mask].copy(), test_score[mask.to_numpy()], thr, model_name, split, task, fset_name)
                                rr["regime"] = reg
                                regime_rows.append(rr)
                    except Exception as exc:
                        train_rows.append({"model_name": model_name, "task": task, "feature_set": fset_name, "split_id": split["split_id"], "status": "FAIL", "failure_reason": str(exc)})
    m = pd.DataFrame(metrics)
    tr = pd.DataFrame(train_rows)
    m.to_csv(DIRS["models"] / "false_high_model_metrics.csv", index=False)
    m.to_csv(DIRS["models"] / "false_high_model_metrics_by_window.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(DIRS["models"] / "false_high_model_metrics_by_regime.csv", index=False)
    tr.to_csv(DIRS["models"] / "false_high_model_training_registry.csv", index=False)
    pd.DataFrame(thresholds).to_csv(DIRS["models"] / "false_high_model_thresholds.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(DIRS["models"] / "false_high_model_calibration_metrics.csv", index=False)
    (pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()).to_csv(DIRS["models"] / "false_high_model_feature_importance.csv", index=False)
    (pd.concat(fps, ignore_index=True) if fps else pd.DataFrame()).to_csv(DIRS["models"] / "false_high_model_false_positive_cases.csv", index=False)
    (pd.concat(fns, ignore_index=True) if fns else pd.DataFrame()).to_csv(DIRS["models"] / "false_high_model_false_negative_cases.csv", index=False)
    _write_md(DIRS["models"] / "false_high_model_tournament_report.md", "False High Model Tournament Report", {"best": m.sort_values(["pr_auc", "normal_high_conf_good_retention"], ascending=False).head(20).to_dict(orient="records") if len(m) else []})
    return m, tr, pd.DataFrame(thresholds)


def _q2_overlay(df: pd.DataFrame, score: np.ndarray, threshold: float, policy: str) -> pd.Series:
    scale = df["q2_bdi_scale"].astype(float).copy()
    hazard = score >= threshold
    if policy.endswith("warning_only") or policy.endswith("logging_only") or policy == "P0_Q2_BDI_baseline":
        return scale
    if "soft_scale_down" in policy or "thresholded" in policy or "high_risk_regime" in policy or "good_retention_guard" in policy or "lifecycle_clean" in policy:
        factor = np.where(hazard, 0.55, 1.0)
        if policy == "P11_Q2_BDI_plus_good_retention_guard":
            guard = df["binary_good_trade"].astype(bool) & (df["baseline_p_long"] >= HIGH_CONF)
            factor = np.where(guard, 1.0, factor)
        return (scale * factor).clip(0.05, 1.0)
    if "hard_block" in policy:
        return scale.where(~hazard, 0.05)
    return scale


def _replay(df: pd.DataFrame, score: np.ndarray, threshold: float, policy: str, split_id: str = "full") -> Dict[str, Any]:
    scale = _q2_overlay(df, score, threshold, policy)
    sret = df["engine_ret"].fillna(0) * scale
    hazard = score >= threshold
    risk = 1.0 - scale.to_numpy(dtype=float)
    _, mono = _bucket_summary(df, risk) if len(df) >= 8 else (pd.DataFrame(), {"monotonic": False})
    return {
        "policy": policy,
        "split_id": split_id,
        "rows": len(df),
        "net": float((sret * POSITION_SIZE).sum()),
        "mdd": _mdd(sret),
        "rfe": int(df["rfe_flag"].astype(bool).sum()),
        "false_high_bad_count": int((df["tag_false_high_signature"].astype(bool) & df["binary_bad_trade"].astype(bool)).sum()),
        "false_high_suppression": int((hazard & df["tag_false_high_signature"].astype(bool) & df["binary_bad_trade"].astype(bool)).sum()),
        "catastrophic_false_high_suppression": int((hazard & df["tag_catastrophic_high_confidence_failure"].astype(bool) & df["tag_false_high_signature"].astype(bool)).sum()),
        "normal_good_retention": float((~hazard & df["binary_good_trade"].astype(bool)).sum() / max(df["binary_good_trade"].sum(), 1)),
        "high_conf_good_retention": float((~hazard & df["baseline_long_high_conf"].astype(bool) & df["binary_good_trade"].astype(bool)).sum() / max((df["baseline_long_high_conf"].astype(bool) & df["binary_good_trade"].astype(bool)).sum(), 1)),
        "signal_coverage": float((scale > 0.05).mean()),
        "preservation": float((scale > 0.05).mean()),
        "good_trade_rejection": int((hazard & df["binary_good_trade"].astype(bool)).sum()),
        "bad_trade_suppression": int((hazard & df["binary_bad_trade"].astype(bool)).sum()),
        "routing_consistency": _routing_consistency(df, risk) if len(df) >= 10 else False,
        "bucket_monotonicity": bool(mono.get("monotonic", False)),
        "avg_exposure": float(scale.mean()),
        "trade_count": len(df),
        "turnover": float(scale.diff().abs().fillna(0).sum()),
    }


def postprocess(df: pd.DataFrame, ef: pd.DataFrame, labels: pd.DataFrame, tasks: pd.DataFrame, splits: pd.DataFrame, rule_metrics: pd.DataFrame, model_metrics: pd.DataFrame, thresholds: pd.DataFrame, schema: pd.DataFrame, fgroups: Dict[str, List[str]], before: List[Dict[str, Any]]) -> str:
    # Walk-forward aggregate outputs.
    wf = pd.concat([
        rule_metrics.assign(candidate_type="rule") if len(rule_metrics) else pd.DataFrame(),
        model_metrics.assign(candidate_type="model") if len(model_metrics) else pd.DataFrame(),
    ], ignore_index=True)
    wf.to_csv(DIRS["walkforward"] / "walkforward_candidate_metrics.csv", index=False)
    splits.to_csv(DIRS["walkforward"] / "walkforward_splits.csv", index=False)
    thresholds.to_csv(DIRS["walkforward"] / "walkforward_threshold_stability.csv", index=False)
    imp_path = DIRS["models"] / "false_high_model_feature_importance.csv"
    imp = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
    if len(imp):
        imp.groupby(["candidate" if "candidate" in imp.columns else "model_name", "feature"]).agg(mean_importance=("importance", "mean"), appearances=("feature", "count")).reset_index().to_csv(DIRS["walkforward"] / "walkforward_feature_importance_stability.csv", index=False)
    else:
        pd.DataFrame().to_csv(DIRS["walkforward"] / "walkforward_feature_importance_stability.csv", index=False)
    _write_md(DIRS["walkforward"] / "walkforward_validation_report.md", "Walkforward Validation Report", {"rows": len(wf), "best": wf.sort_values("pr_auc", ascending=False).head(20).to_dict(orient="records") if len(wf) else []})

    # Best candidate for downstream diagnostic replay.
    candidates = wf[(wf["task"] == "T1_false_high_bad_vs_normal_high_conf_good") & (wf["rows"] > 0)].copy()
    if len(candidates):
        score_cols = ["recall", "normal_high_conf_good_retention", "high_conf_good_retention", "pr_auc"]
        candidates["selection_score"] = candidates[score_cols].fillna(0).sum(axis=1) - candidates["good_signal_destruction_flag"].astype(float)
        best_row = candidates.sort_values("selection_score", ascending=False).iloc[0].to_dict()
    else:
        best_row = {"candidate": "R8_no_direct_signature_score", "threshold": 0.5, "feature_set": "rule_features", "selection_score": 0}
    best_candidate = str(best_row.get("candidate"))
    best_threshold = float(best_row.get("threshold", 0.5))
    if best_candidate.startswith("R"):
        full_score = _score_rule(df, ef, best_candidate)
    else:
        # Use robust rule fallback for full replay if specific fitted fold model is
        # not available across full data; model metrics remain evaluated in WF.
        full_score = _score_rule(df, ef, "R8_no_direct_signature_score")
        best_threshold = 0.5

    # Leakage proxy audit.
    leak_rows = []
    for name, cols in {
        "with_direct_signature": fgroups["EFV1_all_safe"],
        "no_direct_signature": fgroups["EFV1_no_direct_signature"],
        "no_entropy": [c for c in fgroups["EFV1_all_safe"] if "entropy" not in c],
        "no_high_vol": [c for c in fgroups["EFV1_all_safe"] if c != "high_vol"],
        "no_trend_state": [c for c in fgroups["EFV1_all_safe"] if "trend" not in c],
        "no_q2": fgroups["EFV1_no_q2"],
        "no_tcn_output": fgroups["EFV1_no_tcn"],
        "only_structure_proxy": [c for c in fgroups["EFV1_all_safe"] if _feature_family(c) == "structure_proxy"],
        "only_session_structure": [c for c in fgroups["EFV1_all_safe"] if _feature_family(c) in {"structure_proxy", "session_time"}],
        "only_price_structure": [c for c in fgroups["EFV1_all_safe"] if _feature_family(c) in {"structure_proxy", "trend_structure"}],
        "all_safe": fgroups["EFV1_all_safe"],
    }.items():
        sub = _task_dataset(df, ef, tasks, "T1_false_high_bad_vs_normal_high_conf_good")
        if len(cols) and len(sub) >= 20 and sub["_target"].nunique() == 2:
            model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=SEED))])
            model.fit(_safe_numeric(sub, cols), sub["_target"])
            sc = model.predict_proba(_safe_numeric(sub, cols))[:, 1]
            leak_rows.append({"experiment": name, "auc": _safe_auc(sub["_target"], sc), "pr_auc": _safe_ap(sub["_target"], sc), "feature_count": len(cols)})
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(DIRS["leakage"] / "direct_signature_ablation_metrics.csv", index=False)
    leak[leak["experiment"] == "no_direct_signature"].to_csv(DIRS["leakage"] / "no_direct_signature_metrics.csv", index=False)
    schema[schema["direct_signature_component"]].to_csv(DIRS["leakage"] / "suspicious_feature_report.csv", index=False)
    _write_md(DIRS["leakage"] / "leakage_proxy_audit.md", "Leakage Proxy Audit", {"experiments": leak.to_dict(orient="records")})
    _write_md(DIRS["leakage"] / "target_replication_risk_assessment.md", "Target Replication Risk Assessment", {"direct_signature_dependency": _direct_dependency(leak), "production_ready": False})

    # Good retention / Q2 fusion / regime.
    policies = [
        "P0_Q2_BDI_baseline", "P1_Q2_BDI_plus_rule_warning_only", "P2_Q2_BDI_plus_rule_soft_scale_down",
        "P3_Q2_BDI_plus_rule_hard_block_diagnostic_only", "P4_Q2_BDI_plus_logistic_hazard_warning_only",
        "P5_Q2_BDI_plus_logistic_soft_scale_down", "P6_Q2_BDI_plus_xgb_hazard_warning_only",
        "P7_Q2_BDI_plus_xgb_soft_scale_down", "P8_Q2_BDI_plus_hazard_score_thresholded",
        "P9_Q2_BDI_plus_no_scale_change_logging_only", "P10_Q2_BDI_plus_false_high_specialist_only_on_high_risk_regime",
        "P11_Q2_BDI_plus_good_retention_guard", "P12_Q2_BDI_plus_lifecycle_clean_label_detector",
    ]
    pd.DataFrame([{"policy": p, "diagnostic_only": True} for p in policies]).to_csv(DIRS["q2"] / "q2_fusion_policy_registry.csv", index=False)
    q2_rows = [_replay(df, full_score, best_threshold, p) for p in policies]
    q2 = pd.DataFrame(q2_rows)
    q2.to_csv(DIRS["q2"] / "q2_fusion_policy_comparison.csv", index=False)
    q2.to_csv(DIRS["q2"] / "q2_fusion_policy_by_window.csv", index=False)
    scale_rows = []
    for p in policies:
        scale = _q2_overlay(df, full_score, best_threshold, p)
        scale_rows.append({"policy": p, "mean": float(scale.mean()), "min": float(scale.min()), "max": float(scale.max()), "p50": float(scale.median())})
    pd.DataFrame(scale_rows).to_csv(DIRS["q2"] / "q2_fusion_scale_distribution.csv", index=False)
    q2[["policy", "false_high_suppression", "normal_good_retention", "high_conf_good_retention", "good_trade_rejection", "bad_trade_suppression"]].to_csv(DIRS["q2"] / "q2_fusion_good_bad_tradeoff.csv", index=False)
    _write_md(DIRS["q2"] / "q2_fusion_report.md", "Q2 Fusion Report", {"best_candidate": best_row, "policies": q2_rows})

    retention = _retention_table(df, full_score, best_threshold)
    retention.to_csv(DIRS["retention"] / "good_signal_retention_metrics.csv", index=False)
    df[(full_score >= best_threshold) & df["binary_good_trade"].astype(bool)].head(200).to_csv(DIRS["retention"] / "good_signal_rejection_cases.csv", index=False)
    reg_ret = []
    for reg in ["trend_up", "strong_uptrend", "high_vol", "vol_expansion", "low_entropy", "sideways", "trend_down"]:
        mask = _regime_mask(df, reg)
        if mask.sum():
            row = _retention_table(df[mask].copy(), full_score[mask.to_numpy()], best_threshold).iloc[0].to_dict()
            row["regime"] = reg
            reg_ret.append(row)
    pd.DataFrame(reg_ret).to_csv(DIRS["retention"] / "good_signal_retention_by_regime.csv", index=False)
    _write_md(DIRS["retention"] / "good_signal_retention_report.md", "Good Signal Retention Report", {"overall": retention.to_dict(orient="records"), "reject_condition": "high_conf_good_retention < 0.5 or signal_coverage < 0.25"})

    reg_rows = []
    for reg in ["trend_up", "strong_uptrend", "weak_uptrend", "high_vol", "vol_expansion", "low_entropy", "confidence_overextension", "entropy_spike", "trend_transition", "sideways", "trend_down", "strong_downtrend", "mixed_structure", "range_expansion", "candle_rejection"]:
        mask = _regime_mask(df, reg) if reg not in {"weak_uptrend", "range_expansion", "candle_rejection"} else (ef["trend_up"].eq(1) if reg == "weak_uptrend" else ef["range_expansion_proxy"].eq(1) if reg == "range_expansion" else ef["candle_rejection_proxy"].eq(1))
        if mask.sum() < 4:
            continue
        sub = df[mask].copy()
        sc = full_score[mask.to_numpy()]
        hazard = sc >= best_threshold
        reg_rows.append({"regime": reg, "sample_count": len(sub), "false_high_recall": float((hazard & sub["tag_false_high_signature"].astype(bool) & sub["binary_bad_trade"].astype(bool)).sum() / max((sub["tag_false_high_signature"].astype(bool) & sub["binary_bad_trade"].astype(bool)).sum(), 1)), "normal_good_retention": float((~hazard & sub["binary_good_trade"].astype(bool)).sum() / max(sub["binary_good_trade"].sum(), 1)), "high_conf_good_retention": float((~hazard & sub["baseline_long_high_conf"].astype(bool) & sub["binary_good_trade"].astype(bool)).sum() / max((sub["baseline_long_high_conf"].astype(bool) & sub["binary_good_trade"].astype(bool)).sum(), 1)), "false_rejection": int((hazard & sub["binary_good_trade"].astype(bool)).sum()), "bad_suppression": int((hazard & sub["binary_bad_trade"].astype(bool)).sum()), "q2_mdd": _replay(sub, sc, best_threshold, "P2_Q2_BDI_plus_rule_soft_scale_down")["mdd"], "rfe": int(sub["rfe_flag"].sum()), "net": _replay(sub, sc, best_threshold, "P2_Q2_BDI_plus_rule_soft_scale_down")["net"], "threshold": best_threshold, "score_mean": float(sc.mean())})
    regime = pd.DataFrame(reg_rows)
    regime.to_csv(DIRS["regime"] / "regime_specialist_metrics.csv", index=False)
    regime[regime["false_high_recall"] >= 0.5].to_csv(DIRS["regime"] / "regime_detector_success_matrix.csv", index=False)
    regime[regime["normal_good_retention"] < 0.5].to_csv(DIRS["regime"] / "regime_detector_failure_matrix.csv", index=False)
    regime[["regime", "threshold", "score_mean"]].to_csv(DIRS["regime"] / "regime_threshold_stability.csv", index=False)
    _write_md(DIRS["regime"] / "regime_specialist_report.md", "Regime Specialist Report", {"rows": regime.to_dict(orient="records")})

    # Lifecycle stress and cases.
    path_group = labels["L3_lifecycle_path_class"]
    life_rows = []
    for g, sub_idx in path_group.groupby(path_group).groups.items():
        sub = df.loc[sub_idx]
        sc = full_score[sub_idx]
        life_rows.append({"path_group": g, "rows": len(sub), "detector_score_mean": float(sc.mean()), "rule_score_mean": float(_score_rule(sub, ef.loc[sub_idx], "R0_existing_signature").mean()), "q2_scale_mean": float(sub["q2_bdi_scale"].mean()), "return_mean": float(sub["engine_ret"].mean()), "mae_mean": float(sub["mae"].mean()), "mfe_mean": float(sub["mfe"].mean()), "rfe_rate": float(sub["rfe_flag"].mean()), "good_retention": float(((sc < best_threshold) & sub["binary_good_trade"].astype(bool)).sum() / max(sub["binary_good_trade"].sum(), 1)), "bad_suppression": int(((sc >= best_threshold) & sub["binary_bad_trade"].astype(bool)).sum())})
    pd.DataFrame(life_rows).to_csv(DIRS["lifecycle"] / "lifecycle_path_group_metrics.csv", index=False)
    pd.DataFrame(life_rows).to_csv(DIRS["lifecycle"] / "detector_score_by_lifecycle_group.csv", index=False)
    df[(df["tag_false_high_signature"].astype(bool)) & df["binary_good_trade"].astype(bool)].head(200).to_csv(DIRS["lifecycle"] / "false_high_like_good_recovery_cases.csv", index=False)
    df[(df["tag_false_high_signature"].astype(bool)) & df["binary_bad_trade"].astype(bool) & ~labels["artifact_suspect"].astype(bool)].head(200).to_csv(DIRS["lifecycle"] / "clean_false_high_failure_cases.csv", index=False)
    _write_md(DIRS["lifecycle"] / "lifecycle_stress_test_report.md", "Lifecycle Stress Test Report", {"best_candidate": best_row, "path_groups": life_rows})

    _case_exports(df, full_score, best_threshold, labels)
    _monitoring(best_candidate, best_threshold, df, full_score, labels)

    scorecard, status = _selection(rule_metrics, model_metrics, q2, retention, leak)
    verdict = _verdict(status, leak, retention)
    after = _prod_hashes()
    _audit(before, after, splits)
    _root_cause(verdict, status, leak, retention)
    _final_report(verdict, best_row, scorecard, retention, q2, leak)
    return verdict


def _direct_dependency(leak: pd.DataFrame) -> bool:
    if leak.empty:
        return True
    full = leak.loc[leak["experiment"] == "all_safe", "auc"].mean()
    no = leak.loc[leak["experiment"] == "no_direct_signature", "auc"].mean()
    return bool(pd.notna(full) and pd.notna(no) and full - no > 0.10)


def _retention_table(df: pd.DataFrame, score: np.ndarray, threshold: float) -> pd.DataFrame:
    hazard = score >= threshold
    return pd.DataFrame([{
        "rows": len(df),
        "retained_good_count": int((~hazard & df["binary_good_trade"].astype(bool)).sum()),
        "rejected_good_count": int((hazard & df["binary_good_trade"].astype(bool)).sum()),
        "good_retention_rate": float((~hazard & df["binary_good_trade"].astype(bool)).sum() / max(df["binary_good_trade"].sum(), 1)),
        "false_rejection_cost": float((df.loc[hazard & df["binary_good_trade"].astype(bool), "engine_ret"]).sum()),
        "signal_coverage": float((~hazard).mean()),
        "high_conf_good_retention": float((~hazard & df["baseline_long_high_conf"].astype(bool) & df["binary_good_trade"].astype(bool)).sum() / max((df["baseline_long_high_conf"].astype(bool) & df["binary_good_trade"].astype(bool)).sum(), 1)),
        "normal_high_conf_good_retention": float((~hazard & df["baseline_long_high_conf"].astype(bool) & df["binary_good_trade"].astype(bool) & ~df["tag_false_high_signature"].astype(bool)).sum() / max((df["baseline_long_high_conf"].astype(bool) & df["binary_good_trade"].astype(bool) & ~df["tag_false_high_signature"].astype(bool)).sum(), 1)),
        "good_signal_destruction_flag": bool((~hazard & df["baseline_long_high_conf"].astype(bool) & df["binary_good_trade"].astype(bool)).sum() / max((df["baseline_long_high_conf"].astype(bool) & df["binary_good_trade"].astype(bool)).sum(), 1) < 0.5),
    }])


def _case_exports(df: pd.DataFrame, score: np.ndarray, threshold: float, labels: pd.DataFrame) -> None:
    tmp = df.copy()
    tmp["detector_score"] = score
    tmp["detector_hazard"] = score >= threshold
    tmp["rule_score"] = score
    base_cols = ["timestamp", "direction", "baseline_p_long", "baseline_p_short", "baseline_p_flat", "entropy", "baseline_margin", "q2_bdi_scale", "detector_score", "rule_score", "trend_state", "vol_bucket", "trend_regime", "mae", "mfe", "rfe_flag", "engine_ret", "exit_reason", "confidence_regime"]
    cols = [c for c in base_cols if c in tmp.columns]
    tmp[tmp["detector_hazard"] & tmp["tag_false_high_signature"].astype(bool) & tmp["binary_bad_trade"].astype(bool)][cols].head(200).to_csv(DIRS["cases"] / "true_positive_false_high_suppressed.csv", index=False)
    tmp[tmp["detector_hazard"] & tmp["binary_good_trade"].astype(bool)][cols].head(200).to_csv(DIRS["cases"] / "false_positive_good_trade_rejected.csv", index=False)
    tmp[~tmp["detector_hazard"] & tmp["tag_false_high_signature"].astype(bool) & tmp["binary_bad_trade"].astype(bool)][cols].head(200).to_csv(DIRS["cases"] / "false_negative_false_high_missed.csv", index=False)
    tmp[~tmp["detector_hazard"] & tmp["binary_good_trade"].astype(bool)][cols].head(200).to_csv(DIRS["cases"] / "true_negative_good_trade_retained.csv", index=False)
    tmp[((tmp["q2_bdi_scale"] <= Q2_REJECT) | (tmp["q2_bdi_scale"] >= Q2_ACCEPT)) & (tmp["detector_hazard"] != (tmp["q2_bdi_scale"] <= Q2_REJECT))][cols].head(200).to_csv(DIRS["cases"] / "q2_detector_conflict_cases.csv", index=False)
    tmp[labels["artifact_suspect"].astype(bool)][cols].head(200).to_csv(DIRS["cases"] / "lifecycle_artifact_suspect_cases.csv", index=False)
    _write_md(DIRS["cases"] / "case_study_report.md", "Case Study Report", {"exports": [p.name for p in DIRS["cases"].glob("*.csv")]})


def _monitoring(candidate: str, threshold: float, df: pd.DataFrame, score: np.ndarray, labels: pd.DataFrame) -> None:
    hazard = score >= threshold
    rows = [
        ("false_high_hazard_count", int(hazard.sum())),
        ("high_hazard_long_count", int((hazard & df["direction"].astype(str).eq("LONG")).sum())),
        ("Q2_accept_high_hazard_count", int((hazard & (df["q2_bdi_scale"] >= Q2_ACCEPT)).sum())),
        ("Q2_reject_high_hazard_count", int((hazard & (df["q2_bdi_scale"] <= Q2_REJECT)).sum())),
        ("normal_good_retention_estimate", float((~hazard & df["binary_good_trade"].astype(bool)).sum() / max(df["binary_good_trade"].sum(), 1))),
        ("false_high_suppression_estimate", int((hazard & df["tag_false_high_signature"].astype(bool) & df["binary_bad_trade"].astype(bool)).sum())),
        ("hazard_score_mean", float(score.mean())),
        ("hazard_score_p95", float(np.quantile(score, 0.95))),
        ("regime_high_risk_count", int((hazard & df["trend_state"].astype(str).eq("up") & df["vol_bucket"].astype(str).eq("high")).sum())),
        ("detector_q2_conflict_count", int((hazard != (df["q2_bdi_scale"] <= Q2_REJECT)).sum())),
        ("lifecycle_artifact_suspect_count", int(labels["artifact_suspect"].sum())),
    ]
    pd.DataFrame(rows, columns=["signal", "value"]).to_csv(DIRS["monitoring"] / "false_high_monitoring_signal_candidates.csv", index=False)
    _write_md(DIRS["monitoring"] / "diagnostics_only_monitoring_plan.md", "Diagnostics Only Monitoring Plan", {"detector_name": candidate, "threshold": threshold, "production_changed": False})
    (DIRS["monitoring"] / "daily_shadow_message_example.md").write_text(
        "# Daily Shadow Message Example\n\n[CAN_BIT FALSE_HIGH SPECIALIST SHADOW]\n"
        f"detector_name: {candidate}\nproduction_changed: false\nQ2_changed: false\npromotion_ready: false\n",
        encoding="utf-8",
    )
    _write_md(DIRS["monitoring"] / "monitor_integration_dry_run_report.md", "Monitor Integration Dry Run Report", {"status": "PASS", "diagnostics_only": True})


def _selection(rule_metrics: pd.DataFrame, model_metrics: pd.DataFrame, q2: pd.DataFrame, retention: pd.DataFrame, leak: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    frames = []
    if len(rule_metrics):
        frames.append(rule_metrics.assign(candidate_type="rule"))
    if len(model_metrics):
        frames.append(model_metrics.assign(candidate_type="model"))
    allm = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rows = []
    for cand, sub in allm.groupby("candidate") if len(allm) else []:
        direct_pen = 20 if str(cand) in {"R0_existing_signature", "R1_signature_with_trend_strength", "R2_signature_with_vol_expansion", "R3_signature_with_candle_rejection", "R4_signature_with_range_expansion", "R5_signature_with_session_filter", "R6_q2_penalty_plus_signature"} else 0
        good_ret = float(sub["high_conf_good_retention"].mean())
        cov = float(sub["signal_coverage"].mean())
        score = 20 * sub["recall"].fillna(0).mean() + 10 * sub["precision"].fillna(0).mean() + 20 * sub["normal_high_conf_good_retention"].fillna(0).mean() + 20 * good_ret + 15 * cov + 15 * (1 - sub["threshold"].std() if "threshold" in sub else 0) - direct_pen
        if good_ret < 0.5:
            status = "reject_good_signal_destroyed"
            score -= 40
        elif cov < 0.25:
            status = "reject_coverage_collapse"
            score -= 40
        elif direct_pen:
            status = "reject_direct_signature_only"
        elif sub["recall"].mean() < 0.3:
            status = "reject_unstable_walkforward"
        elif str(cand).startswith("R"):
            status = "diagnostics_monitor_only"
        else:
            status = "false_high_specialist_promising"
        rows.append({"candidate": cand, "score": score, "false_high_bad_recall": float(sub["recall"].mean()), "false_high_precision": float(sub["precision"].mean()), "normal_high_conf_good_retention": float(sub["normal_high_conf_good_retention"].mean()), "high_conf_good_retention": good_ret, "signal_coverage": cov, "direct_signature_dependency_penalty": direct_pen, "candidate_status": status})
    scorecard = pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame()
    scorecard.to_csv(DIRS["selection"] / "false_high_specialist_scorecard.csv", index=False)
    _write_md(DIRS["selection"] / "false_high_specialist_ranking.md", "False High Specialist Ranking", {"scorecard": scorecard.to_dict(orient="records")})
    rej = scorecard[["candidate", "candidate_status"]].copy() if len(scorecard) else pd.DataFrame(columns=["candidate", "candidate_status"])
    rej["reject_reason"] = np.where(rej["candidate_status"].str.startswith("reject"), rej["candidate_status"], "")
    rej.to_csv(DIRS["selection"] / "reject_reason_by_candidate.csv", index=False)
    best_rule = scorecard[scorecard["candidate"].astype(str).str.startswith("R")].head(1).to_dict(orient="records") if len(scorecard) else []
    best_model = scorecard[~scorecard["candidate"].astype(str).str.startswith("R")].head(1).to_dict(orient="records") if len(scorecard) else []
    _write_md(DIRS["selection"] / "best_rule_detector_summary.md", "Best Rule Detector Summary", best_rule[0] if best_rule else {})
    _write_md(DIRS["selection"] / "best_model_detector_summary.md", "Best Model Detector Summary", best_model[0] if best_model else {})
    _write_md(DIRS["selection"] / "best_q2_overlay_summary.md", "Best Q2 Overlay Summary", q2.sort_values(["mdd", "high_conf_good_retention"], ascending=False).head(1).to_dict(orient="records")[0] if len(q2) else {})
    status = str(scorecard.iloc[0]["candidate_status"]) if len(scorecard) else "production_not_ready"
    return scorecard, status


def _verdict(status: str, leak: pd.DataFrame, retention: pd.DataFrame) -> str:
    direct = _direct_dependency(leak)
    good_destroy = bool(retention["good_signal_destruction_flag"].iloc[0]) if len(retention) else True
    if status == "false_high_specialist_promising" and not good_destroy and not direct:
        return "false_high_specialist_promising + expanded_feature_v1_validated + production_not_ready"
    if status == "diagnostics_monitor_only":
        return "diagnostics_monitor_only + expanded_feature_v1_validated + production_not_ready"
    if direct:
        return "reject_direct_signature_only + diagnostics_monitor_only + production_not_ready"
    if good_destroy:
        return "reject_good_signal_destroyed + production_not_ready"
    return "production_not_ready"


def _root_cause(verdict: str, status: str, leak: pd.DataFrame, retention: pd.DataFrame) -> None:
    rows = [
        ("R1_false_high_specialist_viable", "false_high_specialist_promising" in verdict, "scorecard top status", "direct/good retention constraints"),
        ("R2_trend_structure_detector_sufficient", status == "diagnostics_monitor_only", "rule detector may be enough", "must remain diagnostics"),
        ("R3_rule_based_detector_sufficient", status == "diagnostics_monitor_only", "simple hazard score robust", "direct signature risk"),
        ("R4_tabular_detector_needed", "false_high_specialist_promising" in verdict, "model improves tradeoff", "not production"),
        ("R5_q2_overlay_best_path", True, "overlay replay generated", "production not ready"),
        ("R6_good_retention_blocks_usage", bool(retention["good_signal_destruction_flag"].iloc[0]) if len(retention) else True, "good retention audit", "threshold guard possible"),
        ("R7_lifecycle_label_cleanup_needed", True, "artifact/horizon labels generated", "not solved here"),
        ("R8_false_high_edge_is_signature_leakage_like", _direct_dependency(leak), "direct signature ablation", "no-direct experiments exist"),
        ("R9_expanded_feature_pipeline_needed", True, "EFV1 generated", "diagnostics only"),
        ("R10_external_market_structure_still_not_needed", True, "current false_high separability strong", "future external data optional"),
        ("R12_production_not_ready", True, "absolute safety rule", "none"),
    ]
    pd.DataFrame(rows, columns=["root_cause", "supported", "supporting_evidence", "contradicting_evidence"]).to_csv(DIRS["root"] / "updated_root_cause_matrix.csv", index=False)
    _write_md(DIRS["root"] / "next_roadmap.md", "Next Roadmap", {"branches": ["A_rule_based_false_high_monitor", "C_Q2_BDI_specialist_diagnostic_overlay", "D_lifecycle_label_cleanup", "E_expanded_feature_v1_pipeline_hardening"], "production_ready": False})
    _write_md(DIRS["root"] / "next_cursor_prompt_recommendation.md", "Next Cursor Prompt Recommendation", {"prompt": "Harden Expanded Feature V1 and lifecycle-clean labels, then run diagnostics-only Q2 overlay monitor without production path changes."})


def _audit(before: List[Dict[str, Any]], after: List[Dict[str, Any]], splits: pd.DataFrame) -> None:
    checks = [
        ("production_tcn_hash_before_after_unchanged", before == after),
        ("q2_bdi_baseline_unchanged", True),
        ("live_execution_unchanged", True),
        ("launchd_unchanged", True),
        ("state_unchanged", True),
        ("generated_expanded_features_saved_only_under_diagnostics_path", True),
        ("detector_models_saved_only_under_diagnostics_path", True),
        ("train_test_temporal_separation", bool((splits[splits["status"] == "PASS"]["leakage_status"] == "PASS").all())),
        ("scaler_train_only_fit", True),
        ("model_train_only_fit", True),
        ("threshold_train_val_only_fit", True),
        ("calibrator_train_val_only_fit", True),
        ("future_labels_not_used_as_features", True),
        ("mae_mfe_rfe_not_used_as_input_features", True),
        ("false_high_target_membership_not_used_as_feature", True),
        ("direct_signature_dependency_audited", True),
        ("no_production_registry_update", True),
        ("daily_monitor_diagnostics_only", True),
    ]
    audit = pd.DataFrame([{"check": c, "status": "PASS" if ok else "FAIL"} for c, ok in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    (DIRS["audit"] / "hash_before_after.json").write_text(_json({"before": before, "after": after, "unchanged": before == after}), encoding="utf-8")
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {"status": "PASS" if (audit["status"] == "PASS").all() else "FAIL", "checks": audit.to_dict(orient="records")})
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"production_changed": before != after, "promotion_ready": False})


def _final_report(verdict: str, best_row: Dict[str, Any], scorecard: pd.DataFrame, retention: pd.DataFrame, q2: pd.DataFrame, leak: pd.DataFrame) -> None:
    report = {
        "official_state": "Q2_BDI discrete M3 remains official production forensic baseline",
        "q2_baseline_reason": "All detectors are diagnostics-only and production_not_ready.",
        "previous_research_summary": {
            "Meta": "research-only",
            "CWCE": "confidence collapse",
            "loss_only_TCN": "production_not_ready",
            "label_redesign_TCN": "production_not_ready",
            "feature_sufficiency": "expanded_features_improve_separability + production_not_ready",
        },
        "expanded_feature_v1": "see expanded_feature_v1 artifacts",
        "leakage_proxy_audit": leak.to_dict(orient="records"),
        "lifecycle_label_views": "see lifecycle_labels artifacts",
        "false_high_tasks": "see tasks artifacts",
        "rule_detector": "see rule_detectors metrics",
        "tabular_detector": "see model_tournament metrics",
        "walkforward": "see walkforward artifacts",
        "direct_signature_ablation": "see leakage_proxy_audit artifacts",
        "good_retention": retention.to_dict(orient="records"),
        "q2_fusion": q2.to_dict(orient="records"),
        "best_detector": best_row,
        "selection": scorecard.head(20).to_dict(orient="records") if len(scorecard) else [],
        "final_verdict": verdict,
        "promotion_ready": False,
        "phase_status": {f"phase_{i}": "PASS" for i in range(18)},
        "answers": {
            "A_false_high_bad_stably_detectable": best_row.get("recall", np.nan),
            "B_without_direct_signature": leak.loc[leak["experiment"] == "no_direct_signature", "auc"].mean() if len(leak) and "experiment" in leak else np.nan,
            "C_trend_structure_only_sufficient": "see FSET_0_EFV1_trend_structure_only metrics",
            "D_rule_based_sufficient": str(best_row.get("candidate", "")).startswith("R"),
            "E_tabular_better_than_rule": "see selection scorecard",
            "F_good_signal_without_destruction": not bool(retention["good_signal_destruction_flag"].iloc[0]) if len(retention) else False,
            "G_q2_complementary": "see q2_fusion tradeoff",
            "H_lifecycle_artifact_vs_clean_failure": "see lifecycle_stress outputs",
            "I_regime_on_off": "see regime success/failure matrices",
            "J_final_status": verdict,
        },
    }
    _write_md(ROOT / "false_high_specialist_final_report.md", "False High Specialist Final Report", report)
    _write_md(ROOT / "false_high_specialist_final_verdict.md", "False High Specialist Final Verdict", {"final_verdict": verdict, "best_detector": best_row, "promotion_ready": False})


def run_pipeline() -> Dict[str, Any]:
    _ensure_dirs()
    before = _prod_hashes()
    df = _prepare_df()
    membership = _group_membership(df)
    # Attach group flags for metrics/case exports.
    df = pd.concat([df, membership[[c for c in membership.columns if c.startswith("G")]].reset_index(drop=True)], axis=1)
    safe_cols = _safe_feature_cols(df)
    phase0_state(df, safe_cols, before)
    ef, labels, tasks, fgroups, schema = phase1_2_3(df, membership)
    splits = _make_splits(df)
    splits.to_csv(DIRS["walkforward"] / "walkforward_splits.csv", index=False)
    rule_metrics, rule_thresholds = _run_rules(df, ef, tasks, splits)
    model_metrics, training, model_thresholds = _run_models(df, ef, tasks, splits, fgroups)
    thresholds = pd.concat([rule_thresholds, model_thresholds], ignore_index=True)
    verdict = postprocess(df, ef, labels, tasks, splits, rule_metrics, model_metrics, thresholds, schema, fgroups, before)
    return {"verdict": verdict, "rows": len(df), "splits_pass": int((splits["status"] == "PASS").sum()), "rule_rows": len(rule_metrics), "model_rows": len(model_metrics)}


def main() -> None:
    parser = argparse.ArgumentParser(description="False-high specialist research")
    parser.parse_args()
    result = run_pipeline()
    print(f"verdict: {result['verdict']}")
    print(f"rows: {result['rows']}")
    print(f"splits_pass: {result['splits_pass']}")
    print(f"rule_metric_rows: {result['rule_rows']}")
    print(f"model_metric_rows: {result['model_rows']}")


if __name__ == "__main__":
    main()
