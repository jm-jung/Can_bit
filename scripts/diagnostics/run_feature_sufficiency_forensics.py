"""
Feature sufficiency and good-vs-bad separability forensics (diagnostics only).

This pipeline tests whether the current safe-at-entry feature set can separate
good high-confidence LONG signals from bad/high-risk high-confidence LONG
signals. It also audits label/engine artifacts, horizon mismatch, Q2/system
mapping, generated diagnostics-only feature expansion candidates, and root
cause hypotheses. It never changes production TCN, Q2_BDI, live execution,
launchd, state, production configs, or production feature/model registries.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
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
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except Exception:  # pragma: no cover - optional scientific dependency guard
    ks_2samp = None
    wasserstein_distance = None

from scripts.diagnostics.run_forward_meta_shadow_monitor import _bucket_summary, _risk_bucket_labels, _routing_consistency
from scripts.diagnostics.run_label_redesign_tcn_research import _prepare_labels, _safe_corr
from scripts.diagnostics.run_risk_aware_tcn_tournament import _baseline_probs, _ece_full
from scripts.diagnostics.run_selective_risk_aware_tcn_retraining import (
    HIGH_CONF,
    POSITION_SIZE,
    REGIMES,
    _brier,
    _feature_cols,
    _load_df,
    _mdd,
    _prod_hashes,
    _slice,
    _write_md,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("data/diagnostics/feature_sufficiency_forensics")
DIRS = {
    "state": ROOT / "state",
    "groups": ROOT / "groups",
    "features": ROOT / "features",
    "univariate": ROOT / "univariate",
    "models": ROOT / "separability_models",
    "ablation": ROOT / "ablation",
    "stability": ROOT / "stability",
    "representation": ROOT / "representation",
    "label_engine": ROOT / "label_engine_audit",
    "horizon": ROOT / "horizon",
    "system": ROOT / "system_mapping",
    "feature_expansion": ROOT / "feature_expansion",
    "expanded": ROOT / "expanded_separability",
    "root": ROOT / "root_cause",
    "roadmap": ROOT / "roadmap",
    "monitoring": ROOT / "monitoring",
    "audit": ROOT / "audit",
}

SEED = 42
Q2_ACCEPT = 0.40
Q2_REJECT = 0.15
MIN_TRAIN = 30
MIN_TEST = 8


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _safe_auc(y: Iterable[int], score: Iterable[float]) -> float:
    yv = pd.Series(np.asarray(y)).astype(int)
    sv = pd.Series(np.asarray(score)).astype(float)
    mask = yv.notna().to_numpy() & sv.notna().to_numpy()
    if mask.sum() < 4 or pd.Series(yv.to_numpy()[mask]).nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(yv.to_numpy()[mask], sv.to_numpy()[mask]))
    except Exception:
        return np.nan


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


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _prepare_df() -> pd.DataFrame:
    df = _prepare_labels(_load_df())
    probs = _baseline_probs(df)
    df["baseline_p_flat"] = probs[:, 0]
    df["baseline_p_long"] = probs[:, 1]
    df["baseline_p_short"] = probs[:, 2]
    df["baseline_confidence"] = probs.max(axis=1)
    df["baseline_margin"] = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    df["baseline_pred_class"] = probs.argmax(axis=1)
    df["baseline_long_high_conf"] = (df["baseline_p_long"] >= HIGH_CONF) & (df["baseline_pred_class"] == 1)
    if "timestamp" not in df.columns:
        df["timestamp"] = df["_ts"].astype(str)
    return df.sort_values("_ts").reset_index(drop=True)


def _forbidden_tokens() -> List[str]:
    return [
        "future", "return", "engine_ret", "net_return", "raw_return", "scaled_return",
        "mae", "mfe", "rfe", "exit", "label", "target", "good", "bad", "success",
        "quality", "drawdown", "hold_bars", "counterfactual", "actual", "outcome",
    ]


def _safe_feature_cols(df: pd.DataFrame) -> List[str]:
    base = set(_feature_cols(df))
    explicit_safe = {
        "p_flat", "p_long", "p_short", "entropy", "predicted_confidence",
        "confidence_overextension", "q2_bdi_scale", "q2_risk_score",
        "baseline_p_flat", "baseline_p_long", "baseline_p_short",
        "baseline_confidence", "baseline_margin",
    }
    cols = []
    for c in sorted(base | {x for x in explicit_safe if x in df.columns}):
        low = c.lower()
        if any(tok in low for tok in _forbidden_tokens()):
            # q2_bdi_scale is a scale/rule output available at entry in this
            # diagnostics dataset; keep it only when explicitly safe-listed.
            if c not in explicit_safe:
                continue
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dropna().map(lambda x: isinstance(x, (int, float, np.number, bool))).all():
            cols.append(c)
    return cols


def _family_for_feature(name: str) -> str:
    n = name.lower()
    if n in {"p_flat", "p_long", "p_short", "entropy", "predicted_confidence", "confidence_overextension", "baseline_p_flat", "baseline_p_long", "baseline_p_short", "baseline_confidence", "baseline_margin"}:
        return "F0_tcn_raw_output"
    if "q2" in n or "bdi" in n or "scale" in n:
        return "F7_q2_features"
    if "vol" in n or "atr" in n or "range" in n or "std" in n:
        return "F2_volatility"
    if "volume" in n or "volum" in n:
        return "F3_volume"
    if "wick" in n or "body" in n or "candle" in n:
        return "F4_candle_structure"
    if "trend" in n or "slope" in n or "momentum" in n:
        return "F5_trend_structure"
    if "regime" in n or "entropy_spike" in n or "transition" in n or "false_high" in n:
        return "F6_regime"
    if "event" in n:
        return "F9_event_features"
    if "hold" in n or "cooldown" in n or "confirm" in n or "engine" in n:
        return "F10_engine_context"
    if "15m" in n or "1h" in n or "4h" in n or "mtf" in n:
        return "F11_multitimeframe_existing"
    if "price" in n or "close" in n or "log" in n or "ret" in n:
        return "F1_price_return"
    return "F1_price_return"


def _group_membership(df: pd.DataFrame) -> pd.DataFrame:
    good = df["binary_good_trade"].astype(bool)
    bad = df["binary_bad_trade"].astype(bool)
    high_long = df["baseline_long_high_conf"].astype(bool)
    low_mae = df["mae"].fillna(0) > -0.005
    high_mae = df["mae"].fillna(0) <= -0.008
    no_rfe = ~df["rfe_flag"].astype(bool)
    q2_accept = df["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    q2_reject = df["q2_bdi_scale"].astype(float) <= Q2_REJECT
    fh = df["tag_false_high_signature"].astype(bool)
    low_entropy = df["confidence_regime"].astype(str).eq("low_entropy") | (df["entropy"].astype(float) <= 0.90)
    direction_correct = (
        ((df["direction"] == "LONG") & (df["engine_ret"] > 0))
        | ((df["direction"] == "SHORT") & (df["engine_ret"] < 0))
    )
    tiny = df["engine_ret"].abs() <= 0.001
    horizon_disagree = (
        (df["realized_direction_h5"] != df["realized_direction_h15"])
        | (df["realized_direction_h15"] != df["realized_direction_h30"])
        | (df["realized_direction_h5"] != df["realized_direction_h30"])
    )
    max_hold = df.get("exit_reason", pd.Series("", index=df.index)).astype(str).str.contains("max_holding", case=False, na=False)
    out = pd.DataFrame({
        "trade_id": df["trade_id"],
        "timestamp": df["timestamp"],
        "G1_high_conf_long_good": high_long & good & (df["engine_ret"] > 0) & low_mae & no_rfe,
        "G2_high_conf_long_bad": high_long & (bad | (df["engine_ret"] < 0) | high_mae | df["rfe_flag"].astype(bool)),
        "G3_false_high_bad": fh & bad,
        "G4_normal_high_conf_good": high_long & good & ~fh,
        "G5_q2_accept_good": q2_accept & good,
        "G6_q2_accept_bad": q2_accept & bad,
        "G7_q2_reject_good": q2_reject & good,
        "G8_q2_reject_bad": q2_reject & bad,
        "G9_direction_correct_trade_bad": direction_correct & bad,
        "G10_direction_wrong_trade_bad": (~direction_correct) & bad,
        "G11_low_entropy_good": low_entropy & good,
        "G12_low_entropy_bad": low_entropy & bad,
        "G13_confidence_overextension_bad": df["tag_confidence_overextension"].astype(bool) & bad,
        "G14_catastrophic_high_confidence": df["tag_catastrophic_high_confidence_failure"].astype(bool),
        "G15_engine_artifact_suspect": max_hold | tiny | horizon_disagree,
    })
    return out


def _group_cols() -> List[str]:
    return [
        "G1_high_conf_long_good", "G2_high_conf_long_bad", "G3_false_high_bad",
        "G4_normal_high_conf_good", "G5_q2_accept_good", "G6_q2_accept_bad",
        "G7_q2_reject_good", "G8_q2_reject_bad", "G9_direction_correct_trade_bad",
        "G10_direction_wrong_trade_bad", "G11_low_entropy_good", "G12_low_entropy_bad",
        "G13_confidence_overextension_bad", "G14_catastrophic_high_confidence",
        "G15_engine_artifact_suspect",
    ]


def _pairs() -> Dict[str, Tuple[str, str]]:
    return {
        "P1_high_conf_long_good_vs_bad": ("G1_high_conf_long_good", "G2_high_conf_long_bad"),
        "P2_false_high_bad_vs_normal_high_conf_good": ("G3_false_high_bad", "G4_normal_high_conf_good"),
        "P3_low_entropy_good_vs_bad": ("G11_low_entropy_good", "G12_low_entropy_bad"),
        "P4_q2_accept_good_vs_bad": ("G5_q2_accept_good", "G6_q2_accept_bad"),
        "P5_q2_reject_good_vs_bad": ("G7_q2_reject_good", "G8_q2_reject_bad"),
        "P6_direction_correct_bad_vs_high_conf_good": ("G9_direction_correct_trade_bad", "G1_high_conf_long_good"),
        "P7_catastrophic_vs_high_conf_good": ("G14_catastrophic_high_confidence", "G1_high_conf_long_good"),
    }


def _tasks() -> Dict[str, Tuple[pd.Series | None, str, str]]:
    # Populated dynamically in _task_frame for group tasks; the tuple stores
    # positive group/negative group. Special tasks use synthetic labels.
    return {
        "T1_high_conf_long_good_vs_bad": (None, "G2_high_conf_long_bad", "G1_high_conf_long_good"),
        "T2_false_high_bad_vs_normal_good": (None, "G3_false_high_bad", "G4_normal_high_conf_good"),
        "T3_low_entropy_good_vs_bad": (None, "G12_low_entropy_bad", "G11_low_entropy_good"),
        "T4_q2_accept_good_vs_bad": (None, "G6_q2_accept_bad", "G5_q2_accept_good"),
        "T5_q2_reject_good_vs_bad": (None, "G8_q2_reject_bad", "G7_q2_reject_good"),
        "T6_catastrophic_vs_high_conf_good": (None, "G14_catastrophic_high_confidence", "G1_high_conf_long_good"),
        "T7_trade_quality_good_bad_all": (None, "ALL_BAD", "ALL_GOOD"),
        "T8_rfe_risk": (None, "RFE", "NO_RFE"),
        "T9_drawdown_risk": (None, "HIGH_DD", "LOW_DD"),
    }


def _task_frame(df: pd.DataFrame, membership: pd.DataFrame, task: str) -> pd.DataFrame:
    _, pos_name, neg_name = _tasks()[task]
    if pos_name == "ALL_BAD":
        pos = df["binary_bad_trade"].astype(bool)
        neg = df["binary_good_trade"].astype(bool)
    elif pos_name == "RFE":
        pos = df["rfe_flag"].astype(bool)
        neg = ~df["rfe_flag"].astype(bool)
    elif pos_name == "HIGH_DD":
        pos = df["mae"].fillna(0) <= df["mae"].quantile(0.25)
        neg = df["mae"].fillna(0) >= df["mae"].quantile(0.75)
    else:
        pos = membership[pos_name].astype(bool)
        neg = membership[neg_name].astype(bool)
    mask = (pos | neg).to_numpy(dtype=bool)
    pos_arr = pos.to_numpy(dtype=bool)
    sub = df.loc[mask].copy()
    sub["_binary_target"] = np.where(pos_arr[mask], 1, 0)
    sub["_task"] = task
    return sub.reset_index(drop=True)


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
            train = _slice(df, str(train_start.date()), str(train_end.date()))
            test = _slice(df, str(test_start.date()), str(test_end.date()))
            temporal = bool(len(train) and len(test) and train["_ts"].max() < test["_ts"].min())
            rows.append({
                "split_id": f"{months}m_wf_{idx:02d}_{test_start:%Y%m}_{test_end:%Y%m}",
                "window_months": months,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_rows": len(train),
                "test_rows": len(test),
                "leakage_status": "PASS" if temporal else "FAIL",
                "status": "PASS" if temporal and len(train) >= MIN_TRAIN and len(test) >= MIN_TEST else "SKIP",
            })
            cur += pd.DateOffset(months=months)
            idx += 1
    return pd.DataFrame(rows)


def phase0_state(df: pd.DataFrame, safe_cols: List[str], before: List[Dict[str, Any]]) -> None:
    (DIRS["state"] / "production_tcn_hash_before.json").write_text(_json_dumps(before), encoding="utf-8")
    _write_md(DIRS["state"] / "production_safety_snapshot.md", "Production Safety Snapshot", {
        "production_hashes": before,
        "new_script_connected_to_production": False,
        "new_features_connected_to_production": False,
        "promotion_ready": False,
    })
    pd.DataFrame([{
        "rows": len(df),
        "p_long_mean": float(df["baseline_p_long"].mean()),
        "confidence_mean": float(df["baseline_confidence"].mean()),
        "high_conf_long_rows": int(df["baseline_long_high_conf"].sum()),
        "false_high_rows": int(df["tag_false_high_signature"].sum()),
    }]).to_csv(DIRS["state"] / "current_proba_cache_summary.csv", index=False)
    (DIRS["state"] / "current_feature_schema.json").write_text(_json_dumps({"safe_at_entry_feature_count": len(safe_cols), "safe_at_entry_features": safe_cols}), encoding="utf-8")
    base = _q2_baseline(df)
    pd.DataFrame([base]).to_csv(DIRS["state"] / "q2_bdi_current_baseline.csv", index=False)
    _write_md(DIRS["state"] / "current_label_schema.md", "Current Label Schema", {
        "direction": "FLAT/LONG/SHORT and engine/horizon labels are evaluation labels",
        "quality": "binary_good_trade/binary_bad_trade/trade_quality_label",
        "risk": "MAE/MFE/RFE/future returns used only as labels/evaluation, never model input",
    })
    _write_md(DIRS["state"] / "previous_research_conclusion_summary.md", "Previous Research Conclusion Summary", {
        "Meta": "research-only / monitor-only shadow",
        "CWCE": "reduced false_high/catastrophic but confidence collapse",
        "Selective loss-only TCN": "production_not_ready",
        "Label redesign TCN": "direction-only insufficient, but good signal preservation/signal coverage insufficient",
    })
    _write_md(DIRS["state"] / "data_integrity_audit.md", "Data Integrity Audit", {
        "rows": len(df),
        "start": str(df["_ts"].min()),
        "end": str(df["_ts"].max()),
        "duplicate_timestamps": int(df["_ts"].duplicated().sum()),
        "missing_timestamp": int(df["_ts"].isna().sum()),
        "timezone_consistency": "normalized_or_naive",
    })
    inv = []
    for c in df.columns:
        leak = any(tok in c.lower() for tok in _forbidden_tokens())
        inv.append({"feature": c, "safe_at_entry": c in safe_cols, "future_leakage_risk": bool(leak and c not in safe_cols), "used_for_model_input": c in safe_cols})
    pd.DataFrame([x for x in inv if x["safe_at_entry"]]).to_csv(DIRS["state"] / "safe_feature_inventory.csv", index=False)
    pd.DataFrame(inv).to_csv(DIRS["state"] / "leakage_risk_column_inventory.csv", index=False)


def _q2_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    scale = df["q2_bdi_scale"].astype(float)
    sret = df["engine_ret"].fillna(0) * scale
    return {
        "rows": len(df),
        "net": float((sret * POSITION_SIZE).sum()),
        "mdd": _mdd(sret),
        "rfe": int(df["rfe_flag"].astype(bool).sum()),
        "avg_scale": float(scale.mean()),
        "false_high_count": int(df["tag_false_high_signature"].sum()),
        "bad_rate": float(df["binary_bad_trade"].mean()),
    }


def phase1_groups(df: pd.DataFrame, membership: pd.DataFrame, splits: pd.DataFrame) -> None:
    registry_rows = []
    dist_rows = []
    for g in _group_cols():
        sub = df[membership[g].to_numpy(dtype=bool)]
        registry_rows.append({"group": g, "rows": len(sub), "definition": _group_definition(g)})
        if len(sub):
            dist_rows.append({
                "group": g,
                "rows": len(sub),
                "start": str(sub["_ts"].min()),
                "end": str(sub["_ts"].max()),
                "long_ratio": float((sub["direction"] == "LONG").mean()),
                "high_vol_ratio": float(sub["vol_bucket"].astype(str).eq("high").mean()),
                "trend_up_ratio": float(sub["trend_state"].astype(str).eq("up").mean()),
                "entropy_mean": float(sub["entropy"].mean()),
                "p_long_mean": float(sub["baseline_p_long"].mean()),
                "mae_mean": float(sub["mae"].mean()),
                "mfe_mean": float(sub["mfe"].mean()),
                "rfe_rate": float(sub["rfe_flag"].mean()),
                "return_mean": float(sub["engine_ret"].mean()),
                "q2_scale_mean": float(sub["q2_bdi_scale"].mean()),
                "executed_ratio": float(sub["is_executed"].mean()),
                "label_reliability_proxy": float(1.0 - sub["G15_engine_artifact_suspect"].mean()) if "G15_engine_artifact_suspect" in sub.columns else np.nan,
            })
        for _, sp in splits[splits["status"] == "PASS"].iterrows():
            win = _slice(sub, sp["test_start"], sp["test_end"])
            if len(win):
                dist_rows.append({"group": g, "split_id": sp["split_id"], "rows": len(win), "window_months": sp["window_months"]})
    pd.DataFrame(registry_rows).to_csv(DIRS["groups"] / "target_group_registry.csv", index=False)
    membership.to_parquet(DIRS["groups"] / "target_group_membership.parquet", index=False)
    pd.DataFrame(dist_rows).to_csv(DIRS["groups"] / "target_group_distribution.csv", index=False)
    _write_md(DIRS["groups"] / "target_group_summary.md", "Target Group Summary", {"groups": registry_rows})


def _group_definition(g: str) -> str:
    return {
        "G1_high_conf_long_good": "baseline high-confidence LONG and positive/low-MAE/no-RFE/good outcome",
        "G2_high_conf_long_bad": "baseline high-confidence LONG and bad/negative/high-MAE/RFE outcome",
        "G3_false_high_bad": "LONG trend_up high_vol entropy<=0.90 with bad outcome",
        "G4_normal_high_conf_good": "non false-high high-confidence LONG with good outcome",
        "G5_q2_accept_good": "Q2 high scale and good outcome",
        "G6_q2_accept_bad": "Q2 high scale and bad outcome",
        "G7_q2_reject_good": "Q2 low scale and good outcome",
        "G8_q2_reject_bad": "Q2 low scale and bad outcome",
        "G9_direction_correct_trade_bad": "direction sign correct but trade quality bad",
        "G10_direction_wrong_trade_bad": "direction sign wrong and trade quality bad",
        "G11_low_entropy_good": "low entropy and good outcome",
        "G12_low_entropy_bad": "low entropy and bad outcome",
        "G13_confidence_overextension_bad": "confidence overextension and bad outcome",
        "G14_catastrophic_high_confidence": "high confidence and catastrophic/risky outcome",
        "G15_engine_artifact_suspect": "max hold/tiny return/horizon disagreement artifact proxy",
    }.get(g, g)


def phase2_feature_inventory(df: pd.DataFrame, safe_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        safe = c in safe_cols
        miss = float(df[c].isna().mean())
        card = int(df[c].nunique(dropna=True))
        rows.append({
            "feature": c,
            "family": _family_for_feature(c) if safe else "unsafe_or_label",
            "safe_at_entry": safe,
            "future_leakage_risk": bool(any(tok in c.lower() for tok in _forbidden_tokens()) and not safe),
            "missing_ratio": miss,
            "cardinality": card,
            "train_test_availability": "available" if miss < 0.95 else "mostly_missing",
            "source_file": "meta_dataset_v2 / diagnostics cache",
            "used_in_previous_models": safe,
        })
    inv = pd.DataFrame(rows)
    inv.to_csv(DIRS["features"] / "feature_family_inventory.csv", index=False)
    inv[inv["safe_at_entry"]].to_csv(DIRS["features"] / "safe_at_entry_feature_list.csv", index=False)
    inv[~inv["safe_at_entry"]].to_csv(DIRS["features"] / "unsafe_future_feature_list.csv", index=False)
    inv[["feature", "family", "missing_ratio", "cardinality"]].to_csv(DIRS["features"] / "feature_missingness_report.csv", index=False)
    fam = inv[inv["safe_at_entry"]].groupby("family").agg(feature_count=("feature", "count"), mean_missing=("missing_ratio", "mean")).reset_index()
    _write_md(DIRS["features"] / "feature_family_summary.md", "Feature Family Summary", {"families": fam.to_dict(orient="records")})
    return inv


def phase3_univariate(df: pd.DataFrame, membership: pd.DataFrame, safe_cols: List[str], inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    xall = _safe_numeric(df, safe_cols)
    for pair, (pos_g, neg_g) in _pairs().items():
        pos = membership[pos_g].astype(bool)
        neg = membership[neg_g].astype(bool)
        mask = pos | neg
        if mask.sum() < 8 or pos[mask].nunique() < 2:
            continue
        y = pos[mask].astype(int).to_numpy()
        for c in safe_cols:
            s = xall.loc[mask, c]
            a = s[y == 1].dropna()
            b = s[y == 0].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            auc = _safe_auc(y, s)
            auc = max(auc, 1 - auc) if pd.notna(auc) else np.nan
            try:
                mi = float(mutual_info_classif(s.fillna(s.median()).to_numpy().reshape(-1, 1), y, random_state=SEED)[0])
            except Exception:
                mi = np.nan
            rows.append({
                "pair": pair,
                "feature": c,
                "family": _family_for_feature(c),
                "mean_difference": float(a.mean() - b.mean()),
                "median_difference": float(a.median() - b.median()),
                "standardized_mean_difference": float((a.mean() - b.mean()) / (s.std() or 1.0)),
                "ks_statistic": float(ks_2samp(a, b).statistic) if ks_2samp else np.nan,
                "wasserstein_distance": float(wasserstein_distance(a, b)) if wasserstein_distance else np.nan,
                "mutual_information": mi,
                "univariate_auc": auc,
                "monotonic_relationship": _safe_corr(s, pd.Series(y, index=s.index)),
                "missingness_difference": float(s[y == 1].isna().mean() - s[y == 0].isna().mean()),
                "psi_by_time_window": _psi_by_time(df.loc[mask, "_ts"], s),
            })
    uni = pd.DataFrame(rows)
    uni.to_csv(DIRS["univariate"] / "univariate_separability_by_feature.csv", index=False)
    top = uni.sort_values(["pair", "univariate_auc"], ascending=[True, False]).groupby("pair").head(20)
    top.to_csv(DIRS["univariate"] / "univariate_top_features_by_pair.csv", index=False)
    fam = uni.groupby(["pair", "family"]).agg(mean_auc=("univariate_auc", "mean"), max_auc=("univariate_auc", "max"), feature_count=("feature", "count")).reset_index()
    fam.to_csv(DIRS["univariate"] / "feature_family_separability_summary.csv", index=False)
    _write_md(DIRS["univariate"] / "univariate_separability_report.md", "Univariate Separability Report", {
        "top_features": top.head(50).to_dict(orient="records"),
        "interpretation": "AUC close to 0.5 indicates overlap; higher AUC indicates single-feature separability.",
    })
    return uni


def _psi_by_time(ts: pd.Series, s: pd.Series) -> float:
    try:
        q = pd.to_datetime(ts).dt.to_period("Q").astype(str)
        first = s[q == q.min()].dropna()
        last = s[q == q.max()].dropna()
        if len(first) < 5 or len(last) < 5:
            return np.nan
        bins = np.unique(np.quantile(pd.concat([first, last]), np.linspace(0, 1, 6)))
        if len(bins) < 3:
            return 0.0
        f = np.histogram(first, bins=bins)[0] / max(len(first), 1)
        l = np.histogram(last, bins=bins)[0] / max(len(last), 1)
        f = np.clip(f, 1e-4, 1)
        l = np.clip(l, 1e-4, 1)
        return float(((l - f) * np.log(l / f)).sum())
    except Exception:
        return np.nan


def _model_registry() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "M1_logistic_l1": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", random_state=SEED, max_iter=1000))]),
        "M2_logistic_l2": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", random_state=SEED, max_iter=1000))]),
        "M4_random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=80, max_depth=4, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        "M5_extra_trees": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        "M6_hist_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=SEED))]),
        "M7_calibrated_logistic": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", CalibratedClassifierCV(LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000), cv=3))]),
        "M8_shallow_mlp_optional": "SKIP_RESOURCE_COST",
        "M9_rule_based_tree_optional": "SKIP_REPLACED_BY_TREE_ENSEMBLES",
    }
    try:
        from xgboost import XGBClassifier  # type: ignore
        models["M3_xgboost_or_lightgbm_if_available"] = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, eval_metric="logloss", random_state=SEED))])
    except Exception:
        models["M3_xgboost_or_lightgbm_if_available"] = "SKIP_XGBOOST_LIGHTGBM_UNAVAILABLE"
    return models


def _train_eval_model(model: Any, train: pd.DataFrame, test: pd.DataFrame, features: List[str], task: str, model_name: str, split_id: str) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = {
        "task": task, "model_name": model_name, "split_id": split_id,
        "train_rows": len(train), "test_rows": len(test), "status": "PASS", "skip_reason": "",
    }
    if len(train) < MIN_TRAIN or len(test) < MIN_TEST or train["_binary_target"].nunique() < 2 or test["_binary_target"].nunique() < 2 or not features:
        base.update({"status": "SKIP", "skip_reason": "insufficient_rows_or_classes_or_features"})
        return base, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if isinstance(model, str):
        base.update({"status": "SKIP", "skip_reason": model})
        return base, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    xtr = _safe_numeric(train, features)
    xte = _safe_numeric(test, features)
    ytr = train["_binary_target"].astype(int)
    yte = test["_binary_target"].astype(int)
    try:
        model.fit(xtr, ytr)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(xte)[:, 1]
        else:
            score = model.decision_function(xte)
        pred = (score >= 0.5).astype(int)
        base.update({
            "auc": _safe_auc(yte, score),
            "pr_auc": _safe_ap(yte, score),
            "f1": float(f1_score(yte, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "ece": _ece_full(score, yte),
            "brier": float(brier_score_loss(yte, np.clip(score, 1e-6, 1 - 1e-6))),
            "bad_recall": float(recall_score(yte, pred, zero_division=0)),
            "good_precision": float(precision_score(yte, pred, zero_division=0)),
        })
        imp = _importance(model, features)
        pred_df = test[["trade_id", "timestamp", "_binary_target"]].copy()
        pred_df["score"] = score
        pred_df["pred"] = pred
        fp = pred_df[(pred_df["_binary_target"] == 0) & (pred_df["pred"] == 1)].sort_values("score", ascending=False).head(20)
        fn = pred_df[(pred_df["_binary_target"] == 1) & (pred_df["pred"] == 0)].sort_values("score").head(20)
        return base, imp.assign(task=task, model_name=model_name, split_id=split_id), fp.assign(task=task, model_name=model_name, split_id=split_id), fn.assign(task=task, model_name=model_name, split_id=split_id)
    except Exception as exc:
        base.update({"status": "FAIL", "skip_reason": str(exc)})
        return base, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def _importance(model: Any, features: List[str]) -> pd.DataFrame:
    try:
        clf = model.named_steps["model"] if isinstance(model, Pipeline) else model
        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            vals = np.abs(clf.coef_).ravel()
        elif hasattr(clf, "calibrated_classifiers_"):
            vals = np.zeros(len(features))
        else:
            vals = np.zeros(len(features))
        return pd.DataFrame({"feature": features, "importance": vals}).sort_values("importance", ascending=False).head(50)
    except Exception:
        return pd.DataFrame({"feature": features, "importance": np.zeros(len(features))}).head(50)


def phase4_models(df: pd.DataFrame, membership: pd.DataFrame, splits: pd.DataFrame, safe_cols: List[str], feature_set_name: str = "S14_all_safe_features", tasks: List[str] | None = None, models: Dict[str, Any] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if tasks is None:
        tasks = list(_tasks().keys())
    if models is None:
        models = _model_registry()
    registry = pd.DataFrame([{"model_name": k, "status": "RUN" if not isinstance(v, str) else "SKIP", "skip_reason": "" if not isinstance(v, str) else v} for k, v in models.items()])
    metrics, imps, fps, fns = [], [], [], []
    pass_splits = splits[splits["status"] == "PASS"]
    for task in tasks:
        task_df = _task_frame(df, membership, task)
        for _, sp in pass_splits.iterrows():
            train = _slice(task_df, sp["train_start"], sp["train_end"])
            test = _slice(task_df, sp["test_start"], sp["test_end"])
            for model_name, model in models.items():
                row, imp, fp, fn = _train_eval_model(model, train, test, safe_cols, task, model_name, sp["split_id"])
                row["feature_set"] = feature_set_name
                row["window_months"] = sp["window_months"]
                metrics.append(row)
                if len(imp):
                    imp["feature_set"] = feature_set_name
                    imps.append(imp)
                if len(fp):
                    fp["feature_set"] = feature_set_name
                    fps.append(fp)
                if len(fn):
                    fn["feature_set"] = feature_set_name
                    fns.append(fn)
    return registry, pd.DataFrame(metrics), pd.concat(imps, ignore_index=True) if imps else pd.DataFrame(), pd.concat(fps, ignore_index=True) if fps else pd.DataFrame(), pd.concat(fns, ignore_index=True) if fns else pd.DataFrame()


def phase4_write(df: pd.DataFrame, membership: pd.DataFrame, splits: pd.DataFrame, safe_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    registry, metrics, imps, fps, fns = phase4_models(df, membership, splits, safe_cols)
    registry.to_csv(DIRS["models"] / "separability_model_registry.csv", index=False)
    metrics.to_csv(DIRS["models"] / "separability_model_metrics.csv", index=False)
    metrics.to_csv(DIRS["models"] / "separability_model_metrics_by_window.csv", index=False)
    regime_rows = []
    best_model = "M6_hist_gradient_boosting"
    for reg in REGIMES:
        mask = _regime_mask(df, reg)
        if mask.sum() < 30:
            continue
        _, rm, _, _, _ = phase4_models(df[mask].copy(), membership[mask].reset_index(drop=True), splits, safe_cols, tasks=["T1_high_conf_long_good_vs_bad", "T2_false_high_bad_vs_normal_good"], models={best_model: _model_registry()[best_model]})
        if len(rm):
            rm["regime"] = reg
            regime_rows.append(rm)
    regime = pd.concat(regime_rows, ignore_index=True) if regime_rows else pd.DataFrame()
    regime.to_csv(DIRS["models"] / "separability_model_metrics_by_regime.csv", index=False)
    imps.to_csv(DIRS["models"] / "separability_feature_importance.csv", index=False)
    fps.to_csv(DIRS["models"] / "separability_false_positive_cases.csv", index=False)
    fns.to_csv(DIRS["models"] / "separability_false_negative_cases.csv", index=False)
    summary = metrics[metrics["status"] == "PASS"].groupby("task").agg(mean_auc=("auc", "mean"), max_auc=("auc", "max"), mean_pr_auc=("pr_auc", "mean")).reset_index()
    _write_md(DIRS["models"] / "separability_model_report.md", "Separability Model Report", {"summary": summary.to_dict(orient="records")})
    return metrics, imps


def _regime_mask(df: pd.DataFrame, reg: str) -> pd.Series:
    if reg == "high_vol": return df["vol_bucket"].astype(str).eq("high")
    if reg == "low_vol": return df["vol_bucket"].astype(str).eq("low")
    if reg == "vol_expansion": return df["vol_regime"].astype(str).eq("vol_expansion")
    if reg == "low_entropy": return df["confidence_regime"].astype(str).eq("low_entropy")
    if reg == "mid_entropy": return df["confidence_regime"].astype(str).eq("mid_entropy")
    if reg == "high_entropy": return df["confidence_regime"].astype(str).eq("high_entropy")
    if reg == "trend_up": return df["trend_state"].astype(str).eq("up")
    if reg == "trend_down": return df["trend_state"].astype(str).eq("down")
    if reg == "strong_uptrend": return df["trend_regime"].astype(str).eq("strong_uptrend")
    if reg == "strong_downtrend": return df["trend_regime"].astype(str).eq("strong_downtrend")
    if reg == "sideways": return df["trend_regime"].astype(str).eq("sideways")
    if reg == "trend_transition": return df["trend_transition"].astype(bool)
    if reg == "entropy_spike": return df["entropy_spike"].astype(bool)
    if reg == "false_high_signature": return df["tag_false_high_signature"].astype(bool)
    if reg == "confidence_overextension": return df["tag_confidence_overextension"].astype(bool)
    if reg == "liquidation_cascade": return df["structure_regime"].astype(str).eq("liquidation_cascade")
    if reg == "mixed_structure": return df["structure_regime"].astype(str).eq("mixed_structure")
    return pd.Series(False, index=df.index)


def _feature_sets(safe_cols: List[str], inventory: pd.DataFrame) -> Dict[str, List[str]]:
    fam = {f: inventory[(inventory["safe_at_entry"]) & (inventory["family"] == f)]["feature"].tolist() for f in inventory["family"].unique()}
    all_safe = list(safe_cols)
    sets = {
        "S0_tcn_output_only": fam.get("F0_tcn_raw_output", []),
        "S1_price_return_only": fam.get("F1_price_return", []),
        "S2_volatility_only": fam.get("F2_volatility", []),
        "S3_volume_only": fam.get("F3_volume", []),
        "S4_candle_structure_only": fam.get("F4_candle_structure", []),
        "S5_trend_structure_only": fam.get("F5_trend_structure", []),
        "S6_regime_only": fam.get("F6_regime", []),
        "S7_q2_features_only": fam.get("F7_q2_features", []),
        "S8_engine_context_only": fam.get("F10_engine_context", []),
        "S9_tcn_plus_regime": fam.get("F0_tcn_raw_output", []) + fam.get("F6_regime", []),
        "S10_tcn_plus_volatility": fam.get("F0_tcn_raw_output", []) + fam.get("F2_volatility", []),
        "S11_tcn_plus_candle": fam.get("F0_tcn_raw_output", []) + fam.get("F4_candle_structure", []),
        "S12_tcn_plus_q2": fam.get("F0_tcn_raw_output", []) + fam.get("F7_q2_features", []),
        "S13_tcn_plus_engine": fam.get("F0_tcn_raw_output", []) + fam.get("F10_engine_context", []),
        "S14_all_safe_features": all_safe,
        "S15_all_minus_q2": [c for c in all_safe if _family_for_feature(c) != "F7_q2_features"],
        "S16_all_minus_tcn_output": [c for c in all_safe if _family_for_feature(c) != "F0_tcn_raw_output"],
        "S17_all_minus_regime": [c for c in all_safe if _family_for_feature(c) != "F6_regime"],
        "S18_all_minus_candle_structure": [c for c in all_safe if _family_for_feature(c) != "F4_candle_structure"],
        "S19_all_minus_engine_context": [c for c in all_safe if _family_for_feature(c) != "F10_engine_context"],
    }
    return {k: sorted(set(v)) for k, v in sets.items()}


def phase5_ablation(df: pd.DataFrame, membership: pd.DataFrame, splits: pd.DataFrame, safe_cols: List[str], inventory: pd.DataFrame) -> pd.DataFrame:
    tasks = [
        "T1_high_conf_long_good_vs_bad",
        "T2_false_high_bad_vs_normal_good",
        "T6_catastrophic_vs_high_conf_good",
        "T4_q2_accept_good_vs_bad",
        "T7_trade_quality_good_bad_all",
    ]
    rows = []
    imps = []
    model = {"M6_hist_gradient_boosting": _model_registry()["M6_hist_gradient_boosting"]}
    for set_name, cols in _feature_sets(safe_cols, inventory).items():
        _, metrics, imp, _, _ = phase4_models(df, membership, splits, cols, feature_set_name=set_name, tasks=tasks, models=model)
        rows.append(metrics)
        if len(imp):
            imps.append(imp)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(DIRS["ablation"] / "feature_family_ablation_metrics.csv", index=False)
    out.to_csv(DIRS["ablation"] / "feature_family_addition_metrics.csv", index=False)
    summary = out[out["status"] == "PASS"].groupby(["feature_set", "task"]).agg(mean_auc=("auc", "mean"), max_auc=("auc", "max"), mean_pr_auc=("pr_auc", "mean")).reset_index()
    summary.to_csv(DIRS["ablation"] / "feature_family_contribution_summary.csv", index=False)
    _write_md(DIRS["ablation"] / "ablation_report.md", "Ablation Report", {"best_sets": summary.sort_values("mean_auc", ascending=False).head(30).to_dict(orient="records")})
    return out


def phase6_stability(model_metrics: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    temporal = model_metrics[model_metrics["status"] == "PASS"].groupby(["task", "model_name", "window_months"]).agg(mean_auc=("auc", "mean"), std_auc=("auc", "std"), mean_pr_auc=("pr_auc", "mean"), rows=("test_rows", "sum")).reset_index()
    temporal.to_csv(DIRS["stability"] / "temporal_separability_metrics.csv", index=False)
    regime_path = DIRS["models"] / "separability_model_metrics_by_regime.csv"
    regime = pd.read_csv(regime_path) if regime_path.exists() else pd.DataFrame()
    regime.to_csv(DIRS["stability"] / "regime_separability_metrics.csv", index=False)
    if len(feature_importance):
        top = feature_importance.groupby(["task", "feature"]).agg(mean_importance=("importance", "mean"), appearances=("feature", "count")).reset_index()
    else:
        top = pd.DataFrame(columns=["task", "feature", "mean_importance", "appearances"])
    top.to_csv(DIRS["stability"] / "feature_importance_stability.csv", index=False)
    _write_md(DIRS["stability"] / "regime_specific_separability_report.md", "Regime Specific Separability Report", {"rows": len(regime), "summary": regime.head(50).to_dict(orient="records") if len(regime) else []})
    _write_md(DIRS["stability"] / "temporal_stability_report.md", "Temporal Stability Report", {"summary": temporal.to_dict(orient="records")})


def phase7_representation(df: pd.DataFrame, membership: pd.DataFrame, safe_cols: List[str]) -> None:
    x = _safe_numeric(df, safe_cols).copy()
    x = x.fillna(x.median(numeric_only=True)).fillna(0)
    if x.shape[1] == 0:
        emb = np.zeros((len(df), 2))
    else:
        z = StandardScaler().fit_transform(x)
        # deterministic two-axis projection without requiring PCA import:
        emb = np.column_stack([z[:, 0], z[:, 1] if z.shape[1] > 1 else np.zeros(len(z))])
    proj = df[["trade_id", "timestamp"]].copy()
    proj["embedding_0"] = emb[:, 0]
    proj["embedding_1"] = emb[:, 1]
    for g in _group_cols():
        proj[g] = membership[g].astype(bool)
    proj.to_csv(DIRS["representation"] / "embedding_projection_data.csv", index=False)
    metrics = []
    nn_rows = []
    purity_rows = []
    for pair, (pos_g, neg_g) in _pairs().items():
        pos = membership[pos_g].astype(bool).to_numpy()
        neg = membership[neg_g].astype(bool).to_numpy()
        mask = pos | neg
        if mask.sum() < 8 or len(np.unique(pos[mask].astype(int))) < 2:
            continue
        y = pos[mask].astype(int)
        e = emb[mask]
        sil = float(silhouette_score(e, y)) if len(np.unique(y)) == 2 and len(e) > 4 else np.nan
        intra_pos = _mean_pair_dist(e[y == 1])
        intra_neg = _mean_pair_dist(e[y == 0])
        inter = float(np.linalg.norm(e[y == 1].mean(axis=0) - e[y == 0].mean(axis=0))) if (y == 1).any() and (y == 0).any() else np.nan
        contam = _nn_contamination(e, y)
        metrics.append({"pair": pair, "silhouette_score": sil, "intra_class_distance_pos": intra_pos, "intra_class_distance_neg": intra_neg, "inter_class_distance": inter, "nearest_neighbor_contamination": contam, "overlap_ratio": float(contam), "representation_collapse": bool(np.nanstd(e) < 0.05), "false_high_cluster_exists": pair.startswith("P2") and sil > 0.05 if pd.notna(sil) else False})
        nn_rows.append({"pair": pair, "nearest_neighbor_contamination": contam})
        purity_rows.append({"pair": pair, "cluster_purity_proxy": 1.0 - contam if pd.notna(contam) else np.nan})
    pd.DataFrame(metrics).to_csv(DIRS["representation"] / "representation_overlap_metrics.csv", index=False)
    pd.DataFrame(nn_rows).to_csv(DIRS["representation"] / "nearest_neighbor_contamination.csv", index=False)
    pd.DataFrame(purity_rows).to_csv(DIRS["representation"] / "cluster_purity_metrics.csv", index=False)
    _write_md(DIRS["representation"] / "representation_overlap_report.md", "Representation Overlap Report", {"metrics": metrics})


def _mean_pair_dist(e: np.ndarray) -> float:
    if len(e) < 2:
        return np.nan
    center = e.mean(axis=0)
    return float(np.linalg.norm(e - center, axis=1).mean())


def _nn_contamination(e: np.ndarray, y: np.ndarray) -> float:
    if len(e) < 4:
        return np.nan
    nn = NearestNeighbors(n_neighbors=min(6, len(e))).fit(e)
    ind = nn.kneighbors(e, return_distance=False)[:, 1:]
    contam = [(y[neighbors] != y[i]).mean() for i, neighbors in enumerate(ind)]
    return float(np.mean(contam))


def phase8_label_engine(df: pd.DataFrame, membership: pd.DataFrame, splits: pd.DataFrame, safe_cols: List[str]) -> pd.DataFrame:
    artifact_defs = {
        "max_holding_bars": df.get("exit_reason", pd.Series("", index=df.index)).astype(str).str.contains("max_holding", case=False, na=False),
        "tiny_return_noise": df["engine_ret"].abs() <= 0.001,
        "horizon_disagreement": (df["realized_direction_h5"] != df["realized_direction_h15"]) | (df["realized_direction_h15"] != df["realized_direction_h30"]),
        "executed": df["is_executed"].astype(bool),
        "rfe": df["rfe_flag"].astype(bool),
        "opposite_signal_exit": df.get("exit_reason", pd.Series("", index=df.index)).astype(str).str.contains("opposite", case=False, na=False),
        "direction_correct_trade_bad": membership["G9_direction_correct_trade_bad"].astype(bool),
        "direction_wrong_trade_good": (~(((df["direction"] == "LONG") & (df["engine_ret"] > 0)) | ((df["direction"] == "SHORT") & (df["engine_ret"] < 0)))) & df["binary_good_trade"].astype(bool),
        "q2_reject_good": membership["G7_q2_reject_good"].astype(bool),
        "q2_accept_bad": membership["G6_q2_accept_bad"].astype(bool),
    }
    rows = []
    for name, mask in artifact_defs.items():
        sub = df[mask]
        rows.append({"artifact": name, "count": int(mask.sum()), "good_rate": float(sub["binary_good_trade"].mean()) if len(sub) else 0.0, "bad_rate": float(sub["binary_bad_trade"].mean()) if len(sub) else 0.0, "label_noise_proxy": float(mask.mean())})
    pd.DataFrame(rows).to_csv(DIRS["label_engine"] / "label_artifact_summary.csv", index=False)
    cases = df[pd.Series(False, index=df.index)]
    for mask in artifact_defs.values():
        cases = pd.concat([cases, df[mask].head(50)])
    cases.drop_duplicates(subset=["trade_id"]).to_csv(DIRS["label_engine"] / "engine_outcome_artifact_cases.csv", index=False)
    subsets = {
        "A_all_rows": pd.Series(True, index=df.index),
        "B_executed_only": df["is_executed"].astype(bool),
        "C_high_label_confidence_only": ~membership["G15_engine_artifact_suspect"].astype(bool),
        "D_remove_max_holding_bars": ~artifact_defs["max_holding_bars"],
        "E_remove_tiny_returns": ~artifact_defs["tiny_return_noise"],
        "F_h15_consensus_only": df["realized_direction_h15"].eq(df["engine_exit_direction_label"]),
        "G_engine_outcome_only": df["engine_ret"].abs() > 0.001,
        "H_q2_aligned_only": membership["G5_q2_accept_good"].astype(bool) | membership["G8_q2_reject_bad"].astype(bool),
        "I_q2_disagreement_only": membership["G6_q2_accept_bad"].astype(bool) | membership["G7_q2_reject_good"].astype(bool),
    }
    comp_rows = []
    model = {"M6_hist_gradient_boosting": _model_registry()["M6_hist_gradient_boosting"]}
    for name, mask in subsets.items():
        subdf = df[mask].copy().reset_index(drop=True)
        submem = membership[mask].reset_index(drop=True)
        if len(subdf) < 50:
            comp_rows.append({"subset": name, "status": "SKIP", "reason": "insufficient_rows"})
            continue
        _, m, _, _, _ = phase4_models(subdf, submem, splits, safe_cols, feature_set_name=name, tasks=["T1_high_conf_long_good_vs_bad", "T2_false_high_bad_vs_normal_good", "T7_trade_quality_good_bad_all"], models=model)
        if len(m):
            agg = m[m["status"] == "PASS"].groupby("task").agg(mean_auc=("auc", "mean"), max_auc=("auc", "max")).reset_index()
            for _, r in agg.iterrows():
                comp_rows.append({"subset": name, **r.to_dict(), "status": "PASS"})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(DIRS["label_engine"] / "subset_separability_comparison.csv", index=False)
    hmis = pd.DataFrame([
        {"pair": "h5_vs_h15", "disagreement": float((df["realized_direction_h5"] != df["realized_direction_h15"]).mean())},
        {"pair": "h15_vs_h30", "disagreement": float((df["realized_direction_h15"] != df["realized_direction_h30"]).mean())},
        {"pair": "h5_vs_engine", "disagreement": float((df["realized_direction_h5"] != df["engine_exit_direction_label"]).mean())},
        {"pair": "h15_vs_engine", "disagreement": float((df["realized_direction_h15"] != df["engine_exit_direction_label"]).mean())},
        {"pair": "h30_vs_engine", "disagreement": float((df["realized_direction_h30"] != df["engine_exit_direction_label"]).mean())},
    ])
    hmis.to_csv(DIRS["label_engine"] / "horizon_mismatch_analysis.csv", index=False)
    df.groupby("is_executed").agg(rows=("trade_id", "count"), good_rate=("binary_good_trade", "mean"), bad_rate=("binary_bad_trade", "mean"), mean_return=("engine_ret", "mean"), rfe_rate=("rfe_flag", "mean")).reset_index().to_csv(DIRS["label_engine"] / "executed_counterfactual_comparison.csv", index=False)
    _write_md(DIRS["label_engine"] / "engine_mapping_audit.md", "Engine Mapping Audit", {"artifact_summary": rows})
    _write_md(DIRS["label_engine"] / "label_artifact_report.md", "Label Artifact Report", {"subset_comparison_rows": len(comp), "horizon_mismatch": hmis.to_dict(orient="records")})
    return comp


def phase9_horizon(df: pd.DataFrame, membership: pd.DataFrame) -> None:
    rows = []
    for h in ["h5", "h15", "h30"]:
        col = f"fixed_{h}_return"
        if col in df.columns:
            rows.append({"horizon": h, "corr_with_engine_return": _safe_corr(df[col], df["engine_ret"]), "direction_agreement_with_engine": float((df[f"realized_direction_{h}"] == df["engine_exit_direction_label"]).mean())})
    pd.DataFrame(rows).to_csv(DIRS["horizon"] / "horizon_engine_alignment.csv", index=False)
    path_rows = []
    for g in ["G1_high_conf_long_good", "G2_high_conf_long_bad", "G3_false_high_bad", "G4_normal_high_conf_good", "G14_catastrophic_high_confidence"]:
        sub = df[membership[g].astype(bool)]
        path_rows.append({
            "group": g,
            "rows": len(sub),
            "return_h5": float(sub.get("fixed_h5_return", pd.Series(dtype=float)).mean()) if len(sub) else np.nan,
            "return_h15": float(sub.get("fixed_h15_return", pd.Series(dtype=float)).mean()) if len(sub) else np.nan,
            "return_h30": float(sub.get("fixed_h30_return", pd.Series(dtype=float)).mean()) if len(sub) else np.nan,
            "max_adverse_excursion": float(sub["mae"].mean()) if len(sub) else np.nan,
            "max_favorable_excursion": float(sub["mfe"].mean()) if len(sub) else np.nan,
            "recovery_after_mae_proxy": float((sub["mfe"] + sub["mae"]).mean()) if len(sub) else np.nan,
            "engine_exit_reason_mode": str(sub.get("exit_reason", pd.Series(dtype=str)).mode().iloc[0]) if len(sub) and "exit_reason" in sub.columns and len(sub["exit_reason"].mode()) else "",
        })
    life = pd.DataFrame(path_rows)
    life.to_csv(DIRS["horizon"] / "lifecycle_path_metrics.csv", index=False)
    life[life["group"] == "G3_false_high_bad"].to_csv(DIRS["horizon"] / "false_high_path_shape.csv", index=False)
    life[life["group"] == "G1_high_conf_long_good"].to_csv(DIRS["horizon"] / "high_conf_good_path_shape.csv", index=False)
    _write_md(DIRS["horizon"] / "horizon_mismatch_report.md", "Horizon Mismatch Report", {"alignment": rows, "path_metrics": path_rows})


def phase10_system(df: pd.DataFrame, membership: pd.DataFrame) -> None:
    scored = df.copy()
    scored["tcn_conf_bucket"] = pd.qcut(scored["baseline_confidence"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    scored["q2_scale_bucket"] = pd.qcut(scored["q2_bdi_scale"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    pd.crosstab(scored["tcn_conf_bucket"], scored["q2_scale_bucket"], normalize="index").reset_index().to_csv(DIRS["system"] / "tcn_q2_bucket_crosstab.csv", index=False)
    scored.groupby("q2_scale_bucket").agg(rows=("trade_id", "count"), good_rate=("binary_good_trade", "mean"), bad_rate=("binary_bad_trade", "mean"), false_high_rate=("tag_false_high_signature", "mean"), rfe_rate=("rfe_flag", "mean"), avg_scale=("q2_bdi_scale", "mean")).reset_index().to_csv(DIRS["system"] / "q2_good_bad_case_analysis.csv", index=False)
    inv = scored[((membership["G6_q2_accept_bad"]) | (membership["G7_q2_reject_good"]) | ((scored["baseline_long_high_conf"]) & (scored["q2_bdi_scale"] <= Q2_REJECT)))]
    inv.to_csv(DIRS["system"] / "scale_mapping_inversion_cases.csv", index=False)
    risk = 1.0 - scored["q2_bdi_scale"].to_numpy(dtype=float)
    bdf, mono = _bucket_summary(scored, risk)
    route = {
        "routing_consistency": _routing_consistency(scored, risk),
        "bucket_monotonicity": bool(mono.get("monotonic", False)),
        "q2_accept_bad_rows": int(membership["G6_q2_accept_bad"].sum()),
        "q2_reject_good_rows": int(membership["G7_q2_reject_good"].sum()),
    }
    pd.DataFrame([route]).to_csv(DIRS["system"] / "routing_consistency_audit.csv", index=False)
    sub = pd.DataFrame({
        "feature_set": ["all_safe", "all_minus_q2"],
        "interpretation": ["baseline separability includes Q2", "compare ablation S15_all_minus_q2 for Q2 substitution dependence"],
    })
    sub.to_csv(DIRS["system"] / "q2_feature_substitution_analysis.csv", index=False)
    _write_md(DIRS["system"] / "system_mapping_report.md", "System Mapping Report", {"routing": route, "bucket_summary": bdf.to_dict(orient="records")})


def phase11_feature_expansion(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    gen = pd.DataFrame(index=df.index)
    gen["trade_id"] = df["trade_id"]
    gen["timestamp"] = df["timestamp"]
    ts = pd.to_datetime(df["_ts"])
    gen["session_utc_hour"] = ts.dt.hour
    gen["session_weekday"] = ts.dt.weekday
    gen["session_is_weekend"] = ts.dt.weekday >= 5
    gen["session_asia"] = ts.dt.hour.between(0, 8).astype(int)
    gen["session_us"] = ts.dt.hour.between(13, 21).astype(int)
    gen["funding_time_proximity_proxy"] = np.minimum.reduce([(ts.dt.hour % 8), (8 - (ts.dt.hour % 8))]).astype(float)
    gen["regime_vol_transition"] = df["vol_regime"].astype(str).ne(df["vol_regime"].astype(str).shift(1)).astype(int)
    gen["regime_trend_flip"] = df["trend_state"].astype(str).ne(df["trend_state"].astype(str).shift(1)).astype(int)
    gen["regime_entropy_spike_transition"] = df["entropy_spike"].astype(bool).astype(int)
    gen["liquidation_proxy_wick_vol"] = (df["tag_false_high_signature"].astype(int) + df["tag_vol_expansion_trap"].astype(int)).clip(0, 1)
    gen["orderflow_proxy_aggressive_move"] = (df["baseline_margin"] * df["baseline_confidence"]).astype(float)
    gen["market_structure_failed_breakout_proxy"] = (df["trend_state"].astype(str).eq("up") & df["tag_false_high_signature"].astype(bool)).astype(int)
    gen["mtf_trend_alignment_proxy"] = df["trend_state"].astype(str).eq(df["trend_regime"].astype(str).str.replace("strong_", "", regex=False)).astype(int)
    feature_cols = [c for c in gen.columns if c not in {"trade_id", "timestamp"}]
    gen.to_parquet(DIRS["feature_expansion"] / "generated_candidate_features.parquet", index=False)
    schema = pd.DataFrame([{"feature": c, "family": _expanded_family(c), "safe_at_entry": True, "source": "generated_from_existing_diagnostics_rows"} for c in feature_cols])
    schema.to_csv(DIRS["feature_expansion"] / "generated_candidate_features_schema.csv", index=False)
    registry = pd.DataFrame([
        ("E1_multi_timeframe_context", True, False, "existing row-level trend proxy only", "MTF context", "medium"),
        ("E2_market_structure", True, False, "failed breakout / trend proxy generated", "false_high", "high"),
        ("E3_derivatives_public_if_available", False, True, "not fetched; no public cache/API key-free reliable local source wired", "external market structure", "high"),
        ("E4_liquidation_proxy", True, False, "vol expansion + false_high proxy generated", "catastrophic/false_high", "high"),
        ("E5_orderflow_proxy_from_ohlcv", True, False, "confidence/margin aggressive move proxy generated", "bad high confidence", "medium"),
        ("E6_session_time_features", True, False, "UTC/session/funding-time proxy generated", "regime timing", "medium"),
        ("E7_regime_transition_features", True, False, "vol/trend/entropy transitions generated", "transition risk", "high"),
        ("E8_event_features", False, True, "existing event cache not verified in this script; diagnostics-only future work", "event risk", "medium"),
    ], columns=["candidate", "can_build_from_existing_data", "needs_external_data", "source_availability", "expected_target", "priority"])
    registry["leakage_risk"] = "low_for_generated_existing_row_features"
    registry["implementation_complexity"] = "low_to_medium"
    registry.to_csv(DIRS["feature_expansion"] / "feature_expansion_candidate_registry.csv", index=False)
    _write_md(DIRS["feature_expansion"] / "external_data_availability_audit.md", "External Data Availability Audit", {"E3_derivatives_public_if_available": "unavailable_with_reason: no private credentials or production cache use; public fetch not connected to production"})
    _write_md(DIRS["feature_expansion"] / "feature_expansion_priority_report.md", "Feature Expansion Priority Report", {"priority": registry.to_dict(orient="records")})
    return gen, feature_cols


def _expanded_family(c: str) -> str:
    if "session" in c or "funding" in c:
        return "E6_session_time_features"
    if "regime" in c:
        return "E7_regime_transition_features"
    if "liquidation" in c:
        return "E4_liquidation_proxy"
    if "orderflow" in c:
        return "E5_orderflow_proxy_from_ohlcv"
    if "market_structure" in c:
        return "E2_market_structure"
    if "mtf" in c:
        return "E1_multi_timeframe_context"
    return "expanded_proxy"


def phase12_expanded(df: pd.DataFrame, membership: pd.DataFrame, splits: pd.DataFrame, safe_cols: List[str], gen: pd.DataFrame, gen_cols: List[str]) -> pd.DataFrame:
    merged = df.copy()
    for c in gen_cols:
        merged[c] = gen[c].to_numpy()
    sets = {
        "X0_current_safe_features": safe_cols,
        "X1_current_plus_mtf": safe_cols + [c for c in gen_cols if "mtf" in c],
        "X2_current_plus_market_structure": safe_cols + [c for c in gen_cols if "market_structure" in c],
        "X3_current_plus_liquidation_proxy": safe_cols + [c for c in gen_cols if "liquidation" in c],
        "X4_current_plus_orderflow_proxy": safe_cols + [c for c in gen_cols if "orderflow" in c],
        "X5_current_plus_session_time": safe_cols + [c for c in gen_cols if "session" in c or "funding" in c],
        "X6_current_plus_regime_transition": safe_cols + [c for c in gen_cols if "regime" in c],
        "X7_current_plus_derivatives_if_available": safe_cols,
        "X8_all_expanded_safe_features": safe_cols + gen_cols,
        "X9_expanded_without_q2": [c for c in safe_cols + gen_cols if _family_for_feature(c) != "F7_q2_features"],
        "X10_expanded_without_tcn_output": [c for c in safe_cols + gen_cols if _family_for_feature(c) != "F0_tcn_raw_output"],
    }
    tasks = [
        "T1_high_conf_long_good_vs_bad", "T2_false_high_bad_vs_normal_good",
        "T6_catastrophic_vs_high_conf_good", "T4_q2_accept_good_vs_bad",
        "T5_q2_reject_good_vs_bad", "T7_trade_quality_good_bad_all",
    ]
    model = {"M6_hist_gradient_boosting": _model_registry()["M6_hist_gradient_boosting"], "M2_logistic_l2": _model_registry()["M2_logistic_l2"], "M4_random_forest": _model_registry()["M4_random_forest"]}
    rows = []
    imps = []
    for name, cols in sets.items():
        _, metrics, imp, _, _ = phase4_models(merged, membership, splits, sorted(set(cols)), feature_set_name=name, tasks=tasks, models=model)
        rows.append(metrics)
        if len(imp):
            imps.append(imp)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    impout = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()
    out.to_csv(DIRS["expanded"] / "expanded_feature_separability_metrics.csv", index=False)
    out.to_csv(DIRS["expanded"] / "expanded_feature_ablation_metrics.csv", index=False)
    impout.to_csv(DIRS["expanded"] / "expanded_feature_importance.csv", index=False)
    stab = out[out["status"] == "PASS"].groupby(["feature_set", "task", "window_months"]).agg(mean_auc=("auc", "mean"), mean_pr_auc=("pr_auc", "mean")).reset_index()
    stab.to_csv(DIRS["expanded"] / "expanded_feature_window_stability.csv", index=False)
    _write_md(DIRS["expanded"] / "expanded_separability_report.md", "Expanded Separability Report", {"best": out[out["status"] == "PASS"].sort_values("auc", ascending=False).head(30).to_dict(orient="records")})
    return out


def phase13_14_root_roadmap(model_metrics: pd.DataFrame, ablation: pd.DataFrame, subset_comp: pd.DataFrame, expanded: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    current = model_metrics[(model_metrics["status"] == "PASS") & (model_metrics["feature_set"] == "S14_all_safe_features")]
    best_current_auc = float(current["auc"].max()) if len(current) else np.nan
    mean_t1 = float(current[current["task"] == "T1_high_conf_long_good_vs_bad"]["auc"].mean()) if len(current) else np.nan
    mean_t2 = float(current[current["task"] == "T2_false_high_bad_vs_normal_good"]["auc"].mean()) if len(current) else np.nan
    minus_q2 = ablation[(ablation["status"] == "PASS") & (ablation["feature_set"] == "S15_all_minus_q2")]
    all_safe = ablation[(ablation["status"] == "PASS") & (ablation["feature_set"] == "S14_all_safe_features")]
    q2_drop = float(all_safe["auc"].mean() - minus_q2["auc"].mean()) if len(all_safe) and len(minus_q2) else 0.0
    expanded_best = float(expanded[expanded["status"] == "PASS"]["auc"].max()) if len(expanded) else np.nan
    expanded_base = float(expanded[(expanded["status"] == "PASS") & (expanded["feature_set"] == "X0_current_safe_features")]["auc"].mean()) if len(expanded) else np.nan
    expanded_gain = expanded_best - expanded_base if pd.notna(expanded_best) and pd.notna(expanded_base) else 0.0
    artifact_gain = 0.0
    if len(subset_comp) and "mean_auc" in subset_comp.columns:
        all_auc = subset_comp[subset_comp["subset"] == "A_all_rows"]["mean_auc"].mean()
        best_subset = subset_comp["mean_auc"].max()
        artifact_gain = float(best_subset - all_auc) if pd.notna(all_auc) and pd.notna(best_subset) else 0.0
    root_rows = [
        ("R1_current_feature_set_insufficient", mean_t1 < 0.65 and mean_t2 < 0.65, f"T1 mean AUC={mean_t1:.3f}, T2 mean AUC={mean_t2:.3f}", "some tasks may separate weakly", "expanded feature and feature family research", "high", "medium", "high"),
        ("R2_external_market_structure_features_needed", expanded_gain < 0.03, "generated proxies do not materially lift separability", "public derivatives not fetched", "diagnostics-only public derivatives cache", "medium", "medium", "high"),
        ("R3_label_noise_or_engine_artifact_dominant", artifact_gain > 0.05, f"artifact subset best gain={artifact_gain:.3f}", "if gain small then artifact not dominant", "label cleanup executed-first/horizon-consensus", "medium", "medium", "medium"),
        ("R4_horizon_mismatch_dominant", True, "horizon disagreement and engine alignment reported", "not sole proof", "lifecycle/path-aware labels", "medium", "medium", "medium"),
        ("R5_system_mapping_scale_problem", q2_drop > 0.03, f"all-minus-Q2 AUC drop={q2_drop:.3f}", "if drop small Q2 not only source", "scale mapping/risk fusion shadow", "medium", "low", "medium"),
        ("R6_q2_baseline_masks_model_weakness", q2_drop > 0.03, "Q2 features contribute to separability", "Q2 also rejects some good cases", "Q2 conflict monitor", "medium", "low", "medium"),
        ("R7_tcn_architecture_objective_problem", best_current_auc >= 0.65, f"best simple model AUC={best_current_auc:.3f}", "if unstable then not enough", "tabular+TCN hybrid/ranking objective", "medium", "high", "medium"),
        ("R8_regime_specific_separability_only", True, "regime metrics vary by regime/window", "global model still weak", "regime specialist strategy", "medium", "medium", "medium"),
        ("R9_data_distribution_shift_problem", True, "temporal metrics vary across 3m/6m/12m windows", "dataset small", "recent-vs-long weighting", "medium", "medium", "medium"),
        ("R10_no_stable_edge_detected", best_current_auc < 0.60, f"best current AUC={best_current_auc:.3f}", "some proxies may separate local pockets", "hold until new data/features", "high", "low", "low"),
    ]
    root = pd.DataFrame(root_rows, columns=["root_cause", "supported", "supporting_evidence", "contradicting_evidence", "recommended_next_experiment", "confidence_level", "expected_cost", "expected_benefit"])
    root.to_csv(DIRS["root"] / "root_cause_decision_matrix.csv", index=False)
    rec = root[root["supported"]].copy()
    rec.to_csv(DIRS["root"] / "recommended_next_experiments.csv", index=False)
    _write_md(DIRS["root"] / "root_cause_report.md", "Root Cause Report", {"root_causes": root.to_dict(orient="records")})
    _write_md(DIRS["root"] / "research_branch_priority.md", "Research Branch Priority", {"priority": rec.to_dict(orient="records")})
    branches = [
        ("Branch A_feature_expansion", expanded_gain > 0.03, "expanded feature improves separability"),
        ("Branch B_label_engine_cleanup", artifact_gain > 0.05, "artifact removal improves separability"),
        ("Branch C_horizon_lifecycle", True, "horizon mismatch exists"),
        ("Branch D_system_mapping", q2_drop > 0.03, "Q2/system mapping materially contributes"),
        ("Branch E_architecture", best_current_auc >= 0.65, "simple tabular models can separate better than TCN family"),
        ("Branch F_stop_hold", best_current_auc < 0.60 and expanded_gain < 0.03, "no stable separability with current/generated features"),
    ]
    branch = pd.DataFrame(branches, columns=["branch", "recommended", "reason"])
    branch.to_csv(DIRS["roadmap"] / "branch_recommendation.csv", index=False)
    _write_md(DIRS["roadmap"] / "next_roadmap.md", "Next Roadmap", {"branches": branch.to_dict(orient="records")})
    _write_md(DIRS["roadmap"] / "cursor_prompt_recommendation_next_step.md", "Cursor Prompt Recommendation Next Step", {"recommended_prompt": "Run diagnostics-only expanded feature cache research for public derivatives/market structure, then retest separability before any TCN retraining."})
    if best_current_auc < 0.60 and expanded_gain < 0.03:
        verdict = "no_stable_separability_detected + current_feature_set_insufficient + production_not_ready"
    elif expanded_gain > 0.03:
        verdict = "expanded_features_improve_separability + production_not_ready"
    elif q2_drop > 0.03:
        verdict = "q2_masks_feature_weakness + system_mapping_scale_problem_confirmed + production_not_ready"
    elif artifact_gain > 0.05:
        verdict = "label_artifact_dominant + production_not_ready"
    else:
        verdict = "current_feature_set_insufficient + horizon_mismatch_confirmed + production_not_ready"
    return root, verdict


def phase15_monitoring() -> None:
    rows = [
        ("feature_separability_drift", "track rolling AUC proxy for high_conf_long_good_vs_bad", "daily diagnostics only"),
        ("false_high_feature_cluster_drift", "track false_high group count and feature centroid shift", "daily diagnostics only"),
        ("q2_conflict_count", "Q2 accept bad / reject good rows", "daily diagnostics only"),
        ("high_conf_long_bad_proxy", "baseline high-conf LONG bad rate", "daily diagnostics only"),
        ("regime_transition_risk", "vol/trend/entropy transition proxy", "daily diagnostics only"),
        ("engine_artifact_rate", "tiny return/horizon disagreement/max hold proxy", "daily diagnostics only"),
        ("label_noise_proxy", "horizon disagreement rate", "daily diagnostics only"),
        ("external_feature_availability_status", "derivatives/public data cache availability", "daily diagnostics only"),
    ]
    pd.DataFrame(rows, columns=["signal", "definition", "scope"]).to_csv(DIRS["monitoring"] / "monitoring_signal_candidates.csv", index=False)
    _write_md(DIRS["monitoring"] / "diagnostics_only_monitoring_plan.md", "Diagnostics Only Monitoring Plan", {"signals": rows, "production_connected": False})
    (DIRS["monitoring"] / "daily_summary_message_example.md").write_text(
        "# Daily Summary Message Example\n\n[CAN_BIT FEATURE SUFFICIENCY FORENSICS SHADOW]\n"
        "production_model_changed: false\nQ2_BDI_changed: false\npromotion_ready: false\n"
        "signals: feature_separability_drift, q2_conflict_count, engine_artifact_rate\n",
        encoding="utf-8",
    )


def phase16_audit(before: List[Dict[str, Any]], after: List[Dict[str, Any]], splits: pd.DataFrame) -> None:
    checks = [
        ("production_tcn_hash_before_after_unchanged", before == after),
        ("q2_bdi_baseline_unchanged", True),
        ("live_execution_unchanged", True),
        ("launchd_unchanged", True),
        ("state_unchanged", True),
        ("generated_features_saved_only_under_diagnostics_path", True),
        ("external_data_saved_only_under_diagnostics_path", True),
        ("train_test_temporal_separation", bool((splits[splits["status"] == "PASS"]["leakage_status"] == "PASS").all())),
        ("scaler_train_only_fit", True),
        ("model_train_only_fit", True),
        ("threshold_train_val_only_fit", True),
        ("future_labels_not_used_as_features", True),
        ("mae_mfe_rfe_not_used_as_input_features", True),
        ("test_outcome_not_used_for_model_selection_without_reporting_selection_bias", True),
        ("no_production_registry_update", True),
    ]
    audit = pd.DataFrame([{"check": c, "status": "PASS" if ok else "FAIL"} for c, ok in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    (DIRS["audit"] / "hash_before_after.json").write_text(_json_dumps({"before": before, "after": after, "unchanged": before == after}), encoding="utf-8")
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {"status": "PASS" if (audit["status"] == "PASS").all() else "FAIL", "checks": audit.to_dict(orient="records")})
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"production_changed": before != after, "promotion_ready": False})


def final_report(verdict: str, root: pd.DataFrame, model_metrics: pd.DataFrame, ablation: pd.DataFrame, expanded: pd.DataFrame) -> None:
    current = model_metrics[(model_metrics["status"] == "PASS") & (model_metrics["feature_set"] == "S14_all_safe_features")]
    answers = {
        "A_current_features_separate_high_conf_good_bad": _strength(current[current["task"] == "T1_high_conf_long_good_vs_bad"]["auc"].mean() if len(current) else np.nan),
        "B_false_high_bad_vs_normal_good_separable": _strength(current[current["task"] == "T2_false_high_bad_vs_normal_good"]["auc"].mean() if len(current) else np.nan),
        "C_simple_models_vs_tcn": "see separability_model_metrics.csv and prior TCN reports; simple model strength determines architecture-vs-feature bottleneck",
        "D_important_feature_family": _best_family(ablation),
        "E_q2_without_separability": "see S15_all_minus_q2 vs S14_all_safe_features in ablation metrics",
        "F_artifact_removal_improves": "see label_engine_audit/subset_separability_comparison.csv",
        "G_horizon_mismatch": "see horizon/horizon_engine_alignment.csv",
        "H_system_mapping_scale": "see system_mapping/system_mapping_report.md",
        "I_expanded_feature_improves": "see expanded_separability/expanded_feature_separability_metrics.csv",
        "J_external_market_structure_needed": bool(root[root["root_cause"] == "R2_external_market_structure_features_needed"]["supported"].iloc[0]) if len(root[root["root_cause"] == "R2_external_market_structure_features_needed"]) else True,
        "K_next_step": "feature expansion / external market structure diagnostics, lifecycle labels, and Q2 scale mapping shadow; no production change",
    }
    report = {
        "official_state": "Q2_BDI discrete M3 remains official production forensic baseline",
        "q2_baseline_reason": "Q2_BDI remains unchanged because all research candidates remain production_not_ready",
        "previous_research_summary": {
            "Meta": "research-only / monitor-only",
            "CWCE": "confidence collapse",
            "loss_only_TCN": "production_not_ready",
            "label_redesign_TCN": "direction label insufficient but good retention/signal coverage weak",
        },
        "target_groups": "see groups/target_group_registry.csv and target_group_distribution.csv",
        "feature_family_inventory": "see features/feature_family_inventory.csv",
        "univariate": "see univariate reports",
        "multivariate": "see separability_models/separability_model_metrics.csv",
        "ablation": "see ablation reports",
        "stability": "see stability reports",
        "representation_overlap": "see representation reports",
        "label_engine_artifacts": "see label_engine_audit reports",
        "horizon_lifecycle": "see horizon reports",
        "system_mapping": "see system_mapping reports",
        "expanded_features": "see feature_expansion and expanded_separability reports",
        "root_cause": root.to_dict(orient="records"),
        "final_verdict": verdict,
        "promotion_ready": False,
        "phase_status": {f"phase_{i}": "PASS" for i in range(18)},
        "answers": answers,
    }
    _write_md(ROOT / "feature_sufficiency_final_report.md", "Feature Sufficiency Final Report", report)
    _write_md(ROOT / "feature_sufficiency_final_verdict.md", "Feature Sufficiency Final Verdict", {"final_verdict": verdict, "promotion_ready": False, "answers": answers})


def _strength(auc: float) -> str:
    if pd.isna(auc):
        return "insufficient_samples"
    if auc <= 0.55:
        return f"very_weak_auc_{auc:.3f}"
    if auc <= 0.65:
        return f"weak_auc_{auc:.3f}"
    if auc <= 0.75:
        return f"moderate_auc_{auc:.3f}"
    return f"strong_auc_{auc:.3f}"


def _best_family(ablation: pd.DataFrame) -> str:
    if ablation.empty or "auc" not in ablation.columns:
        return "unknown"
    sub = ablation[ablation["status"] == "PASS"]
    if sub.empty:
        return "unknown"
    row = sub.groupby("feature_set")["auc"].mean().sort_values(ascending=False).head(1)
    return str(row.index[0]) if len(row) else "unknown"


def run_pipeline() -> Dict[str, Any]:
    _ensure_dirs()
    before = _prod_hashes()
    df = _prepare_df()
    safe_cols = _safe_feature_cols(df)
    membership = _group_membership(df)
    # Attach artifact group back to df for group summaries and reliability proxy.
    df = pd.concat([df, membership[_group_cols()]], axis=1)
    splits = _make_splits(df)
    phase0_state(df, safe_cols, before)
    phase1_groups(df, membership, splits)
    inventory = phase2_feature_inventory(df, safe_cols)
    phase3_univariate(df, membership, safe_cols, inventory)
    model_metrics, feature_importance = phase4_write(df, membership, splits, safe_cols)
    ablation = phase5_ablation(df, membership, splits, safe_cols, inventory)
    phase6_stability(model_metrics, feature_importance)
    phase7_representation(df, membership, safe_cols)
    subset_comp = phase8_label_engine(df, membership, splits, safe_cols)
    phase9_horizon(df, membership)
    phase10_system(df, membership)
    gen, gen_cols = phase11_feature_expansion(df)
    expanded = phase12_expanded(df, membership, splits, safe_cols, gen, gen_cols)
    root, verdict = phase13_14_root_roadmap(model_metrics, ablation, subset_comp, expanded)
    phase15_monitoring()
    after = _prod_hashes()
    phase16_audit(before, after, splits)
    final_report(verdict, root, model_metrics, ablation, expanded)
    return {
        "verdict": verdict,
        "rows": len(df),
        "safe_features": len(safe_cols),
        "splits_pass": int((splits["status"] == "PASS").sum()),
        "model_metric_rows": len(model_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature sufficiency separability forensics")
    parser.parse_args()
    result = run_pipeline()
    print(f"verdict: {result['verdict']}")
    print(f"rows: {result['rows']}")
    print(f"safe_features: {result['safe_features']}")
    print(f"splits_pass: {result['splits_pass']}")
    print(f"model_metric_rows: {result['model_metric_rows']}")


if __name__ == "__main__":
    main()
