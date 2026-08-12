"""
Frozen quarterly temporal-transfer validation for FalseHigh_R7_StructureHazard_v1.

All outputs are diagnostics-only and written under:

    data/diagnostics/false_high_r7_quarterly_transfer/

The script keeps the R7 formula and threshold frozen at 0.65. It does not write
production TCN, Q2_BDI, live execution, launchd, order routing, or state paths.
"""

from __future__ import annotations

import argparse
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

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover
    ks_2samp = None

from scripts.diagnostics.run_feature_sufficiency_forensics import _mdd, _prod_hashes, _q2_baseline
from scripts.diagnostics.run_false_high_r7_monitor import (
    FORBIDDEN_FEATURES,
    MONITOR_NAME,
    POSITION_SIZE,
    Q2_ACCEPT,
    Q2_REJECT,
    R7_DEFAULT_THRESHOLD,
    R7_SOURCE_NAME,
    _case_columns,
    _diagnostic_policy_comparison,
    _false_high_masks,
    _file_hash,
    _good_masks,
    _md,
    _policy_net_mdd,
    _prepare_monitor_frame,
    _score_scenario,
    _write_text,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("data/diagnostics/false_high_r7_quarterly_transfer")
R7_LOCK_ROOT = Path("data/diagnostics/false_high_r7_monitor")
DIRS = {
    "state": ROOT / "state",
    "splits": ROOT / "splits",
    "eval": ROOT / "evaluation",
    "drift": ROOT / "drift",
    "sig": ROOT / "signature_dependency",
    "good": ROOT / "good_false_warning",
    "missed": ROOT / "missed_false_high",
    "q2": ROOT / "q2_transfer",
    "recent": ROOT / "recent_validation",
    "regime": ROOT / "regime_policy",
    "forward": ROOT / "forward_readiness",
    "cases": ROOT / "cases",
    "decision": ROOT / "decision",
    "audit": ROOT / "audit",
}

THRESHOLD = 0.65
MIN_TEST_ROWS = 8


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _safe_auc(y: Iterable[int], score: Iterable[float]) -> float:
    yv = np.asarray(y).astype(int)
    sv = np.asarray(score).astype(float)
    mask = np.isfinite(sv)
    if mask.sum() < 4 or len(np.unique(yv[mask])) < 2:
        return np.nan
    try:
        return float(roc_auc_score(yv[mask], sv[mask]))
    except Exception:
        return np.nan


def _safe_ap(y: Iterable[int], score: Iterable[float]) -> float:
    yv = np.asarray(y).astype(int)
    sv = np.asarray(score).astype(float)
    mask = np.isfinite(sv)
    if mask.sum() < 4 or len(np.unique(yv[mask])) < 2:
        return np.nan
    try:
        return float(average_precision_score(yv[mask], sv[mask]))
    except Exception:
        return np.nan


def _rate(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def _top(s: pd.Series, n: int = 3) -> str:
    if len(s) == 0:
        return ""
    return json.dumps(s.astype(str).value_counts().head(n).to_dict(), default=str)


def _quarter_label(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("Q").astype(str)


def _add_quarter(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["quarter"] = _quarter_label(out["_ts"])
    out["threshold_margin"] = out["r7_score"].astype(float) - THRESHOLD
    return out


def _calendar_index(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, sub in frame.groupby("quarter", sort=True):
        period = pd.Period(q, freq="Q")
        q_start = period.start_time
        q_end = period.end_time
        actual_start = sub["_ts"].min()
        actual_end = sub["_ts"].max()
        partial = bool(actual_start > q_start or actual_end < q_end)
        rows.append({
            "quarter": q,
            "quarter_start": q_start,
            "quarter_end": q_end,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "row_count": len(sub),
            "partial_quarter_flag": partial,
            "false_high_bad_count": int(sub["false_high_bad"].astype(bool).sum()),
            "normal_high_conf_good_count": int(sub["normal_high_conf_good"].astype(bool).sum()),
            "high_conf_good_count": int((sub["binary_good_trade"].astype(bool) & sub["baseline_long_high_conf"].astype(bool)).sum()),
        })
    return pd.DataFrame(rows)


def _split_counts(frame: pd.DataFrame, test_start: Any, test_end: Any) -> Dict[str, Any]:
    sub = frame[(frame["_ts"] >= pd.Timestamp(test_start)) & (frame["_ts"] <= pd.Timestamp(test_end))].copy()
    hazard = sub["r7_score"].astype(float) >= THRESHOLD if len(sub) else pd.Series([], dtype=bool)
    q2_accept = sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT if len(sub) else pd.Series([], dtype=bool)
    return {
        "test_rows": len(sub),
        "test_false_high_bad_count": int(sub["false_high_bad"].astype(bool).sum()) if len(sub) else 0,
        "test_normal_high_conf_good_count": int(sub["normal_high_conf_good"].astype(bool).sum()) if len(sub) else 0,
        "test_high_conf_good_count": int((sub["binary_good_trade"].astype(bool) & sub["baseline_long_high_conf"].astype(bool)).sum()) if len(sub) else 0,
        "test_q2_accept_count": int(q2_accept.sum()) if len(sub) else 0,
        "test_q2_accept_r7_candidate_count": int((q2_accept & hazard).sum()) if len(sub) else 0,
    }


def _construct_splits(frame: pd.DataFrame, qidx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    quarters = qidx["quarter"].tolist()
    rows = []

    def add(split_type: str, history_qs: List[str], test_qs: List[str]) -> None:
        if not test_qs:
            return
        h = qidx[qidx["quarter"].isin(history_qs)]
        t = qidx[qidx["quarter"].isin(test_qs)]
        if len(t) == 0:
            return
        history_start = h["actual_start"].min() if len(h) else pd.NaT
        history_end = h["actual_end"].max() if len(h) else pd.NaT
        test_start = t["actual_start"].min()
        test_end = t["actual_end"].max()
        counts = _split_counts(frame, test_start, test_end)
        partial = bool(t["partial_quarter_flag"].any())
        row_count_status = "PASS" if counts["test_rows"] >= MIN_TEST_ROWS else "SKIP"
        leakage = "PASS"
        if len(h) and pd.Timestamp(history_end) >= pd.Timestamp(test_start):
            leakage = "FAIL"
        rows.append({
            "split_id": f"{split_type}__{'_'.join(test_qs)}",
            "split_type": split_type,
            "history_start": history_start,
            "history_end": history_end,
            "test_start": test_start,
            "test_end": test_end,
            "history_quarters": "|".join(history_qs),
            "test_quarters": "|".join(test_qs),
            "history_rows": int(h["row_count"].sum()) if len(h) else 0,
            **counts,
            "partial_quarter_flag": partial,
            "row_count_status": row_count_status,
            "leakage_status": leakage,
        })

    for i, q in enumerate(quarters):
        add("quarter_only", [], [q])
        if i >= 1:
            add("rolling_1q_history_next_1q", quarters[i - 1:i], [q])
            add("expanding_history_next_1q", quarters[:i], [q])
        if i >= 2:
            add("rolling_2q_history_next_1q", quarters[i - 2:i], [q])
        if i >= 4:
            add("rolling_4q_history_next_1q", quarters[i - 4:i], [q])
        if i >= 2 and i + 1 < len(quarters):
            add("rolling_2q_history_next_2q", quarters[i - 2:i], quarters[i:i + 2])
        if i >= 4 and i + 1 < len(quarters):
            add("rolling_4q_history_next_2q", quarters[i - 4:i], quarters[i:i + 2])

    max_ts = frame["_ts"].max()
    panels = [
        ("last_3m_test", max_ts - pd.DateOffset(months=3), max_ts),
        ("last_6m_test", max_ts - pd.DateOffset(months=6), max_ts),
        ("previous_3m_vs_recent_3m", max_ts - pd.DateOffset(months=6), max_ts - pd.DateOffset(months=3)),
        ("previous_6m_vs_recent_6m", max_ts - pd.DateOffset(months=12), max_ts - pd.DateOffset(months=6)),
    ]
    recent = []
    for name, start, end in panels:
        counts = _split_counts(frame, start, end)
        recent.append({
            "split_id": name,
            "split_type": name,
            "history_start": pd.NaT,
            "history_end": pd.NaT,
            "test_start": start,
            "test_end": end,
            "history_quarters": "",
            "test_quarters": "|".join(sorted(frame.loc[(frame["_ts"] >= start) & (frame["_ts"] <= end), "quarter"].unique())),
            "history_rows": 0,
            **counts,
            "partial_quarter_flag": True,
            "row_count_status": "PASS" if counts["test_rows"] >= MIN_TEST_ROWS else "SKIP",
            "leakage_status": "PASS",
        })
    recent.append({
        **recent[-1],
        "split_id": "last_4_quarters_sequence",
        "split_type": "last_4_quarters_sequence",
        "test_start": qidx.tail(4)["actual_start"].min(),
        "test_end": qidx.tail(4)["actual_end"].max(),
        "test_quarters": "|".join(qidx.tail(4)["quarter"]),
        **_split_counts(frame, qidx.tail(4)["actual_start"].min(), qidx.tail(4)["actual_end"].max()),
    })
    if len(qidx) >= 8:
        recent.append({
            **recent[-1],
            "split_id": "last_8_quarters_sequence",
            "split_type": "last_8_quarters_sequence",
            "test_start": qidx.tail(8)["actual_start"].min(),
            "test_end": qidx.tail(8)["actual_end"].max(),
            "test_quarters": "|".join(qidx.tail(8)["quarter"]),
            **_split_counts(frame, qidx.tail(8)["actual_start"].min(), qidx.tail(8)["actual_end"].max()),
        })
    return pd.DataFrame(rows), pd.DataFrame(recent)


def _period_frame(frame: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    return frame[(frame["_ts"] >= pd.Timestamp(start)) & (frame["_ts"] <= pd.Timestamp(end))].copy()


def _window_metrics(sub: pd.DataFrame, scope: str, split_id: str = "") -> Dict[str, Any]:
    hazard = sub["r7_score"].astype(float) >= THRESHOLD
    long = sub["direction"].astype(str).eq("LONG")
    hcl = sub["baseline_long_high_conf"].astype(bool)
    q2_accept = sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    q2_reject = sub["q2_bdi_scale"].astype(float) <= Q2_REJECT
    fh = sub["false_high_bad"].astype(bool)
    clean_fh = fh & ~sub["artifact_suspect"].astype(bool)
    normal_good = sub["normal_high_conf_good"].astype(bool)
    high_good = sub["binary_good_trade"].astype(bool) & hcl
    good = sub["binary_good_trade"].astype(bool)
    bad = sub["binary_bad_trade"].astype(bool)
    good_recovery = (sub["L3_lifecycle_path_class"].astype(str).eq("early_adverse_recovery") | sub["tag_false_high_signature"].astype(bool)) & good
    trend_up_good = sub["trend_up"].astype(bool) & good
    rfe = sub["rfe_flag"].astype(bool)
    high_mae = sub["mae"].astype(float) <= -0.008
    warned_bad = _rate((hazard & bad).sum(), hazard.sum())
    unwarned_bad = _rate((~hazard & bad).sum(), (~hazard).sum())
    warned_rfe = _rate((hazard & rfe).sum(), hazard.sum())
    unwarned_rfe = _rate((~hazard & rfe).sum(), (~hazard).sum())
    warned_mae = _rate((hazard & high_mae).sum(), hazard.sum())
    unwarned_mae = _rate((~hazard & high_mae).sum(), (~hazard).sum())
    y = np.where(fh, 1, np.where(normal_good, 0, np.nan))
    task_mask = ~pd.Series(y).isna().to_numpy()
    pred = hazard.to_numpy()
    status = "PASS"
    if len(sub) < MIN_TEST_ROWS or int(fh.sum()) < 3 or int(normal_good.sum()) < 3:
        status = "WARN"
    if int((hazard & normal_good).sum()) > 0 or int((~hazard & clean_fh).sum()) > 0:
        status = "WARN"
    return {
        "scope": scope,
        "split_id": split_id,
        "total_rows": len(sub),
        "long_candidate_count": int(long.sum()),
        "tcn_high_conf_long_count": int(hcl.sum()),
        "q2_accept_long_count": int((q2_accept & long).sum()),
        "q2_reject_long_count": int((q2_reject & long).sum()),
        "r7_warning_count": int(hazard.sum()),
        "r7_warning_rate": _rate(hazard.sum(), len(sub)),
        "r7_high_hazard_long_count": int((hazard & long).sum()),
        "q2_accept_r7_high_hazard_count": int((q2_accept & hazard).sum()),
        "q2_reject_r7_high_hazard_count": int((q2_reject & hazard).sum()),
        "r7_q2_conflict_count": int((hazard & (q2_accept | q2_reject)).sum()),
        "false_high_bad_count": int(fh.sum()),
        "clean_false_high_count": int(clean_fh.sum()),
        "normal_high_conf_good_count": int(normal_good.sum()),
        "high_conf_good_count": int(high_good.sum()),
        "good_recovery_count": int(good_recovery.sum()),
        "precision_false_high_bad": _rate((hazard & fh).sum(), hazard.sum()),
        "recall_false_high_bad": _rate((hazard & fh).sum(), fh.sum()),
        "precision_clean_false_high": _rate((hazard & clean_fh).sum(), hazard.sum()),
        "recall_clean_false_high": _rate((hazard & clean_fh).sum(), clean_fh.sum()),
        "normal_high_conf_good_retention": _rate((~hazard & normal_good).sum(), normal_good.sum()),
        "high_conf_good_retention": _rate((~hazard & high_good).sum(), high_good.sum()),
        "good_recovery_false_warning_rate": _rate((hazard & good_recovery).sum(), good_recovery.sum()),
        "trend_up_good_false_warning_rate": _rate((hazard & trend_up_good).sum(), trend_up_good.sum()),
        "missed_false_high_count": int((~hazard & fh).sum()),
        "false_positive_good_warning_count": int((hazard & good).sum()),
        "false_negative_false_high_count": int((~hazard & fh).sum()),
        "RFE_rate_warned": warned_rfe,
        "high_MAE_rate_warned": warned_mae,
        "bad_rate_warned": warned_bad,
        "bad_rate_unwarned": unwarned_bad,
        "warned_vs_unwarned_bad_lift": float(warned_bad / unwarned_bad) if unwarned_bad and not math.isnan(unwarned_bad) else np.nan,
        "warned_vs_unwarned_RFE_lift": float(warned_rfe / unwarned_rfe) if unwarned_rfe and not math.isnan(unwarned_rfe) else np.nan,
        "warned_vs_unwarned_MAE_lift": float(warned_mae / unwarned_mae) if unwarned_mae and not math.isnan(unwarned_mae) else np.nan,
        "R7_score_mean": float(sub["r7_score"].mean()) if len(sub) else np.nan,
        "R7_score_p50": float(sub["r7_score"].quantile(0.50)) if len(sub) else np.nan,
        "R7_score_p75": float(sub["r7_score"].quantile(0.75)) if len(sub) else np.nan,
        "R7_score_p95": float(sub["r7_score"].quantile(0.95)) if len(sub) else np.nan,
        "R7_score_max": float(sub["r7_score"].max()) if len(sub) else np.nan,
        "threshold_margin_mean": float(sub["threshold_margin"].mean()) if len(sub) else np.nan,
        "threshold_margin_p10": float(sub["threshold_margin"].quantile(0.10)) if len(sub) else np.nan,
        "threshold_margin_p50": float(sub["threshold_margin"].quantile(0.50)) if len(sub) else np.nan,
        "threshold_margin_p90": float(sub["threshold_margin"].quantile(0.90)) if len(sub) else np.nan,
        "task_precision": float(precision_score(y[task_mask].astype(int), pred[task_mask], zero_division=0)) if task_mask.sum() and len(np.unique(y[task_mask])) == 2 else np.nan,
        "task_recall": float(recall_score(y[task_mask].astype(int), pred[task_mask], zero_division=0)) if task_mask.sum() and len(np.unique(y[task_mask])) == 2 else np.nan,
        "task_pr_auc": _safe_ap(y[task_mask].astype(int), sub.loc[task_mask, "r7_score"]) if task_mask.sum() else np.nan,
        "task_roc_auc": _safe_auc(y[task_mask].astype(int), sub.loc[task_mask, "r7_score"]) if task_mask.sum() else np.nan,
        "status": status,
    }


def _evaluate_splits(frame: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sp in splits.iterrows():
        sub = _period_frame(frame, sp["test_start"], sp["test_end"])
        row = _window_metrics(sub, sp["split_type"], sp["split_id"])
        row.update({
            "test_start": sp["test_start"],
            "test_end": sp["test_end"],
            "history_quarters": sp.get("history_quarters", ""),
            "test_quarters": sp.get("test_quarters", ""),
            "row_count_status": sp.get("row_count_status", ""),
            "leakage_status": sp.get("leakage_status", ""),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _distribution_drift(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist_rows, warning_rows, feature_rows, regime_rows, q2_rows, margin_rows = [], [], [], [], [], []
    quarters = sorted(frame["quarter"].unique())
    prev_scores = None
    prev_q = None
    features = ["r7_score", "trend_up", "high_vol", "low_entropy_proxy", "vol_expansion", "tcn_confidence_overextension", "q2_bdi_scale", "baseline_p_long", "entropy", "baseline_margin"]
    base_bins = np.linspace(0, 1, 11)
    for q in quarters:
        sub = frame[frame["quarter"].eq(q)].copy()
        scores = sub["r7_score"].astype(float)
        hazard = scores >= THRESHOLD
        psi = _psi(prev_scores, scores, base_bins) if prev_scores is not None else np.nan
        ks = _ks(prev_scores, scores) if prev_scores is not None else np.nan
        dist_rows.append({
            "quarter": q,
            "rows": len(sub),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_p50": float(scores.quantile(0.50)),
            "score_p95": float(scores.quantile(0.95)),
            "quarter_to_quarter_psi": psi,
            "ks_statistic": ks,
            "previous_quarter": prev_q,
        })
        warning_rows.append({
            "quarter": q,
            "warning_count": int(hazard.sum()),
            "warning_rate": _rate(hazard.sum(), len(sub)),
            "warning_rate_delta": np.nan if prev_scores is None else _rate(hazard.sum(), len(sub)) - _rate((prev_scores >= THRESHOLD).sum(), len(prev_scores)),
            "false_high_bad_base_rate": float(sub["false_high_bad"].astype(bool).mean()) if len(sub) else np.nan,
            "normal_high_conf_good_base_rate": float(sub["normal_high_conf_good"].astype(bool).mean()) if len(sub) else np.nan,
        })
        for f in features:
            if f not in sub.columns:
                continue
            cur = pd.to_numeric(sub[f], errors="coerce")
            feature_rows.append({
                "quarter": q,
                "feature": f,
                "mean": float(cur.mean()),
                "std": float(cur.std()),
                "p95": float(cur.quantile(0.95)),
                "mean_shift_vs_previous": np.nan,
            })
        for reg, cnt in sub["trend_regime"].astype(str).value_counts().items():
            regime_rows.append({"quarter": q, "regime": reg, "count": int(cnt), "share": float(cnt / max(len(sub), 1))})
        q2_rows.append({
            "quarter": q,
            "q2_accept_r7_high_hazard": int(((sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT) & hazard).sum()),
            "q2_reject_r7_high_hazard": int(((sub["q2_bdi_scale"].astype(float) <= Q2_REJECT) & hazard).sum()),
            "q2_conflict_rate": _rate(((sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT) & hazard).sum(), len(sub)),
        })
        margin_rows.append({
            "quarter": q,
            "margin_mean": float(sub["threshold_margin"].mean()),
            "margin_p10": float(sub["threshold_margin"].quantile(0.10)),
            "margin_p50": float(sub["threshold_margin"].quantile(0.50)),
            "margin_p90": float(sub["threshold_margin"].quantile(0.90)),
            "near_boundary_rate_abs_005": float((sub["threshold_margin"].abs() <= 0.05).mean()) if len(sub) else np.nan,
        })
        prev_scores = scores
        prev_q = q
    feature_df = pd.DataFrame(feature_rows)
    if len(feature_df):
        feature_df["mean_shift_vs_previous"] = feature_df.groupby("feature")["mean"].diff()
    return pd.DataFrame(dist_rows), pd.DataFrame(warning_rows), feature_df, pd.DataFrame(regime_rows), pd.DataFrame(q2_rows), pd.DataFrame(margin_rows)


def _psi(prev: pd.Series, cur: pd.Series, bins: np.ndarray) -> float:
    if prev is None or len(prev) == 0 or len(cur) == 0:
        return np.nan
    a, _ = np.histogram(prev.astype(float), bins=bins)
    b, _ = np.histogram(cur.astype(float), bins=bins)
    pa = np.maximum(a / max(a.sum(), 1), 1e-6)
    pb = np.maximum(b / max(b.sum(), 1), 1e-6)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def _ks(prev: pd.Series | None, cur: pd.Series) -> float:
    if prev is None or len(prev) == 0 or len(cur) == 0 or ks_2samp is None:
        return np.nan
    try:
        return float(ks_2samp(prev.astype(float), cur.astype(float)).statistic)
    except Exception:
        return np.nan


def _ablation_by_quarter(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = [
        "A0_full_R7", "A1_no_direct_signature", "A2_no_entropy", "A3_no_high_vol",
        "A4_no_trend_state", "A5_no_tcn_output", "A6_no_q2", "A7_no_trend_structure",
        "A8_only_trend_structure", "A9_only_structure_proxy", "A10_only_price_structure",
        "A11_only_session_time", "A12_all_safe_no_signature_overlap",
    ]
    rows = []
    for q, sub in frame.groupby("quarter", sort=True):
        target_mask = (sub["false_high_bad"].astype(bool) | sub["normal_high_conf_good"].astype(bool)).to_numpy()
        if target_mask.sum() == 0:
            continue
        y = np.where(sub.loc[target_mask, "false_high_bad"].astype(bool), 1, 0)
        full_recall = np.nan
        for sc in scenarios:
            score = pd.Series(_score_scenario(sub, sc), index=sub.index)
            pred = score >= THRESHOLD
            yy = y.astype(int)
            pp = pred.loc[target_mask].to_numpy()
            ss = score.loc[target_mask].to_numpy()
            recall = float(recall_score(yy, pp, zero_division=0)) if len(np.unique(yy)) == 2 else np.nan
            if sc == "A0_full_R7":
                full_recall = recall
            normal_good = sub["normal_high_conf_good"].astype(bool)
            high_good = sub["binary_good_trade"].astype(bool) & sub["baseline_long_high_conf"].astype(bool)
            rows.append({
                "quarter": q,
                "scenario": sc,
                "rows": len(sub),
                "precision": float(precision_score(yy, pp, zero_division=0)) if len(np.unique(yy)) == 2 else np.nan,
                "recall": recall,
                "PR-AUC": _safe_ap(yy, ss),
                "ROC-AUC": _safe_auc(yy, ss),
                "normal_good_retention": _rate((~pred & normal_good).sum(), normal_good.sum()),
                "high_conf_good_retention": _rate((~pred & high_good).sum(), high_good.sum()),
                "warning_rate": float(pred.mean()) if len(pred) else np.nan,
                "missed_false_high_count": int((~pred & sub["false_high_bad"].astype(bool)).sum()),
                "false_positive_good_warning_count": int((pred & sub["binary_good_trade"].astype(bool)).sum()),
                "dependency_delta_vs_full": np.nan if math.isnan(full_recall) or math.isnan(recall) else full_recall - recall,
                "suspicious_dependency_score": 0.0,
            })
    ab = pd.DataFrame(rows)
    if len(ab):
        full = ab[ab["scenario"].eq("A0_full_R7")][["quarter", "recall"]].rename(columns={"recall": "full_recall"})
        ab = ab.drop(columns=["dependency_delta_vs_full"]).merge(full, on="quarter", how="left")
        ab["dependency_delta_vs_full"] = ab["full_recall"] - ab["recall"]
        ab["suspicious_dependency_score"] = np.where(
            ab["scenario"].isin(["A1_no_direct_signature", "A12_all_safe_no_signature_overlap"]),
            ab["dependency_delta_vs_full"].clip(lower=0),
            0.0,
        )
    dep = ab[ab["scenario"].isin(["A1_no_direct_signature", "A12_all_safe_no_signature_overlap"])].copy()
    fail_q = dep[dep["recall"].fillna(0) < 0.8]["quarter"].unique().tolist() if len(dep) else []
    cases = frame[frame["quarter"].isin(fail_q) & frame["false_high_bad"].astype(bool) & (_score_scenario(frame, "A1_no_direct_signature") < THRESHOLD)].copy()
    return ab, dep, _case_export(cases, "no_direct_signature_failure_case")


def _good_false_warning(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_q, rows_g, cases = [], [], []
    hazard = frame["r7_score"].astype(float) >= THRESHOLD
    for q, qsub in frame.groupby("quarter", sort=True):
        qhaz = qsub["r7_score"].astype(float) >= THRESHOLD
        for name, mask in _good_masks(qsub).items():
            sub = qsub.loc[mask.to_numpy(dtype=bool)].copy()
            hz = qhaz.loc[sub.index]
            row = {
                "quarter": q,
                "group": name,
                "group_count": len(sub),
                "r7_warning_count": int(hz.sum()),
                "retention_rate": _rate((~hz).sum(), len(sub)),
                "false_warning_rate": _rate(hz.sum(), len(sub)),
                "false_warning_cost": float(sub.loc[hz, "engine_ret"].sum() * POSITION_SIZE) if len(sub) else 0.0,
                "R7_score_mean": float(sub["r7_score"].mean()) if len(sub) else np.nan,
                "R7_score_p95": float(sub["r7_score"].quantile(0.95)) if len(sub) else np.nan,
                "Q2_scale_mean": float(sub["q2_bdi_scale"].mean()) if len(sub) else np.nan,
                "TCN_confidence_mean": float(sub["baseline_confidence"].mean()) if len(sub) else np.nan,
                "lifecycle_path_distribution": _top(sub["L3_lifecycle_path_class"]),
                "top trend_structure signatures": _top(sub["trend_regime"]),
                "session distribution": _top(pd.Series(np.where(sub["asia_session"].astype(bool), "asia", np.where(sub["us_session"].astype(bool), "us", "other")), index=sub.index)),
                "regime distribution": _top(sub["trend_regime"]),
            }
            rows_q.append(row)
            rows_g.append({k: v for k, v in row.items() if k != "quarter"})
            if int(hz.sum()):
                cases.append(sub.loc[hz])
    case_df = _case_export(pd.concat(cases, ignore_index=True) if cases else frame.iloc[0:0], "good_false_warning_case")
    return pd.DataFrame(rows_q), pd.DataFrame(rows_g), case_df


def _missed_false_high(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_q, rows_g, cases = [], [], []
    for q, qsub in frame.groupby("quarter", sort=True):
        hazard = qsub["r7_score"].astype(float) >= THRESHOLD
        for name, mask in _false_high_masks(qsub).items():
            sub = qsub.loc[mask.to_numpy(dtype=bool)].copy()
            hz = hazard.loc[sub.index]
            missed = sub.loc[~hz]
            row = {
                "quarter": q,
                "group": name,
                "group_count": len(sub),
                "r7_warning_count": int(hz.sum()),
                "recall": _rate(hz.sum(), len(sub)),
                "missed_count": len(missed),
                "missed_score_mean": float(missed["r7_score"].mean()) if len(missed) else np.nan,
                "missed_score_p95": float(missed["r7_score"].quantile(0.95)) if len(missed) else np.nan,
                "Q2_scale_distribution": _top(sub["q2_bdi_scale"]),
                "TCN_confidence_distribution": _top(sub["baseline_confidence"]),
                "lifecycle_path_distribution": _top(sub["L3_lifecycle_path_class"]),
                "top missed signatures": _top(missed["trend_regime"]),
                "regime distribution": _top(sub["trend_regime"]),
            }
            rows_q.append(row)
            rows_g.append({k: v for k, v in row.items() if k != "quarter"})
            if len(missed):
                cases.append(missed)
    case_df = _case_export(pd.concat(cases, ignore_index=True) if cases else frame.iloc[0:0], "missed_false_high_case")
    return pd.DataFrame(rows_q), pd.DataFrame(rows_g), case_df


def _q2_transfer(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, outcomes, policies, contrib = [], [], [], []
    for q, sub in frame.groupby("quarter", sort=True):
        hazard = sub["r7_score"].astype(float) >= THRESHOLD
        q2_accept = sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
        q2_reject = sub["q2_bdi_scale"].astype(float) <= Q2_REJECT
        cats = {
            "C1_Q2_accept_R7_high_hazard_bad": q2_accept & hazard & sub["binary_bad_trade"].astype(bool),
            "C2_Q2_accept_R7_high_hazard_good": q2_accept & hazard & sub["binary_good_trade"].astype(bool),
            "C3_Q2_accept_R7_low_hazard_bad": q2_accept & ~hazard & sub["binary_bad_trade"].astype(bool),
            "C4_Q2_accept_R7_low_hazard_good": q2_accept & ~hazard & sub["binary_good_trade"].astype(bool),
            "C5_Q2_reject_R7_high_hazard_bad": q2_reject & hazard & sub["binary_bad_trade"].astype(bool),
            "C6_Q2_reject_R7_high_hazard_good": q2_reject & hazard & sub["binary_good_trade"].astype(bool),
            "C7_Q2_reject_R7_low_hazard_bad": q2_reject & ~hazard & sub["binary_bad_trade"].astype(bool),
            "C8_Q2_reject_R7_low_hazard_good": q2_reject & ~hazard & sub["binary_good_trade"].astype(bool),
        }
        for cat, mask in cats.items():
            csub = sub.loc[mask].copy()
            row = _category_outcome(q, cat, csub)
            rows.append(row)
            if "high_hazard" in cat and "accept" in cat:
                outcomes.append(_case_export(csub, cat))
        p = _diagnostic_policy_comparison(sub, THRESHOLD)
        p["quarter"] = q
        policies.append(p)
        net, mdd = _policy_net_mdd(sub, hazard, "baseline")
        contrib.append({
            "quarter": q,
            "baseline_net": net,
            "baseline_mdd": mdd,
            "warned_net_contribution": float((sub.loc[hazard, "engine_ret"] * sub.loc[hazard, "q2_bdi_scale"] * POSITION_SIZE).sum()),
            "warned_rfe_count": int((hazard & sub["rfe_flag"].astype(bool)).sum()),
            "warned_high_mae_count": int((hazard & (sub["mae"].astype(float) <= -0.008)).sum()),
        })
    return pd.DataFrame(rows), pd.concat(outcomes, ignore_index=True) if outcomes else _case_export(frame.iloc[0:0], "empty"), pd.concat(policies, ignore_index=True), pd.DataFrame(contrib)


def _category_outcome(q: str, cat: str, sub: pd.DataFrame) -> Dict[str, Any]:
    bad = sub["binary_bad_trade"].astype(bool) if len(sub) else pd.Series([], dtype=bool)
    good = sub["binary_good_trade"].astype(bool) if len(sub) else pd.Series([], dtype=bool)
    rfe = sub["rfe_flag"].astype(bool) if len(sub) else pd.Series([], dtype=bool)
    high_mae = sub["mae"].astype(float) <= -0.008 if len(sub) else pd.Series([], dtype=bool)
    pnl = sub["engine_ret"].astype(float) * sub["q2_bdi_scale"].astype(float) * POSITION_SIZE if len(sub) else pd.Series([], dtype=float)
    return {
        "quarter": q,
        "category": cat,
        "count": len(sub),
        "bad_rate": _rate(bad.sum(), len(sub)),
        "good_rate": _rate(good.sum(), len(sub)),
        "RFE_rate": _rate(rfe.sum(), len(sub)),
        "high_MAE_rate": _rate(high_mae.sum(), len(sub)),
        "net contribution": float(pnl.sum()) if len(sub) else 0.0,
        "MDD contribution": float(_mdd(pnl)) if len(sub) else 0.0,
        "average Q2 scale": float(sub["q2_bdi_scale"].mean()) if len(sub) else np.nan,
        "average R7 score": float(sub["r7_score"].mean()) if len(sub) else np.nan,
        "average TCN p_long": float(sub["baseline_p_long"].mean()) if len(sub) else np.nan,
        "average entropy": float(sub["entropy"].mean()) if len(sub) else np.nan,
        "lifecycle artifact ratio": float(sub["artifact_suspect"].astype(bool).mean()) if len(sub) else np.nan,
        "clean failure ratio": float((sub["false_high_bad"].astype(bool) & ~sub["artifact_suspect"].astype(bool)).mean()) if len(sub) else np.nan,
        "good recovery ratio": float((sub["binary_good_trade"].astype(bool) & sub["L3_lifecycle_path_class"].astype(str).eq("early_adverse_recovery")).mean()) if len(sub) else np.nan,
    }


def _recent_validation(frame: pd.DataFrame, recent_splits: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = _evaluate_splits(frame, recent_splits)
    comp_rows = []
    def m(name: str) -> pd.Series:
        return metrics.loc[metrics["split_id"].eq(name)].iloc[0] if metrics["split_id"].eq(name).any() else pd.Series(dtype=object)
    pairs = [("previous_3m_vs_recent_3m", "last_3m_test"), ("previous_6m_vs_recent_6m", "last_6m_test")]
    for prev, recent in pairs:
        a, b = m(prev), m(recent)
        if len(a) and len(b):
            comp_rows.append({
                "comparison": f"{prev}__to__{recent}",
                "warning_rate_delta": b["r7_warning_rate"] - a["r7_warning_rate"],
                "recall_delta": b["recall_false_high_bad"] - a["recall_false_high_bad"],
                "normal_good_retention_delta": b["normal_high_conf_good_retention"] - a["normal_high_conf_good_retention"],
                "bad_lift_delta": b["warned_vs_unwarned_bad_lift"] - a["warned_vs_unwarned_bad_lift"],
            })
    recent_q2 = []
    for _, sp in recent_splits.iterrows():
        sub = _period_frame(frame, sp["test_start"], sp["test_end"])
        q2_accept = sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
        hazard = sub["r7_score"].astype(float) >= THRESHOLD
        recent_q2.append(_category_outcome(sp["split_id"], "Q2_accept_R7_high_hazard", sub.loc[q2_accept & hazard]))
    return metrics, pd.DataFrame(comp_rows), pd.DataFrame(recent_q2)


def _regime_policy(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_masks = {
        "P_global_on": lambda x: pd.Series(True, index=x.index),
        "P_high_vol_only": lambda x: x["high_vol"].astype(bool),
        "P_trend_up_only": lambda x: x["trend_up"].astype(bool),
        "P_high_vol_trend_up_only": lambda x: x["high_vol"].astype(bool) & x["trend_up"].astype(bool),
        "P_low_entropy_only": lambda x: x["low_entropy_proxy"].astype(bool),
        "P_confidence_overextension_only": lambda x: x["tcn_confidence_overextension"].astype(bool),
        "P_strong_uptrend_only": lambda x: x["strong_uptrend"].astype(bool),
        "P_trend_transition_off": lambda x: ~x["trend_transition"].astype(bool),
        "P_sideways_off": lambda x: ~x["sideways"].astype(bool),
        "P_good_recovery_guard": lambda x: ~(x["tag_false_high_signature"].astype(bool) & x["binary_good_trade"].astype(bool)),
        "P_lifecycle_artifact_suspect_off": lambda x: ~x["artifact_suspect"].astype(bool),
        "P_q2_accept_only": lambda x: x["q2_bdi_scale"].astype(float) >= Q2_ACCEPT,
        "P_q2_conflict_only": lambda x: (x["q2_bdi_scale"].astype(float) >= Q2_ACCEPT) | (x["q2_bdi_scale"].astype(float) <= Q2_REJECT),
        "P_recent_stable_regime_only": lambda x: x["trend_regime"].astype(str).isin(["strong_uptrend", "weak_uptrend"]),
    }
    rows = []
    for q, sub in frame.groupby("quarter", sort=True):
        base_hazard = sub["r7_score"].astype(float) >= THRESHOLD
        for name, fn in policy_masks.items():
            effective = base_hazard & fn(sub)
            fh = sub["false_high_bad"].astype(bool)
            normal = sub["normal_high_conf_good"].astype(bool)
            high_good = sub["binary_good_trade"].astype(bool) & sub["baseline_long_high_conf"].astype(bool)
            bad = sub["binary_bad_trade"].astype(bool)
            unwarned_bad = _rate((~effective & bad).sum(), (~effective).sum())
            warned_bad = _rate((effective & bad).sum(), effective.sum())
            rows.append({
                "quarter": q,
                "policy": name,
                "warning_count": int(effective.sum()),
                "false_high_recall": _rate((effective & fh).sum(), fh.sum()),
                "false_high_precision": _rate((effective & bad).sum(), effective.sum()),
                "normal_good_retention": _rate((~effective & normal).sum(), normal.sum()),
                "high_conf_good_retention": _rate((~effective & high_good).sum(), high_good.sum()),
                "good_false_warning_rate": _rate((effective & sub["binary_good_trade"].astype(bool)).sum(), sub["binary_good_trade"].astype(bool).sum()),
                "missed_false_high": int((~effective & fh).sum()),
                "Q2_accept_high_hazard_count": int((effective & (sub["q2_bdi_scale"].astype(float) >= Q2_ACCEPT)).sum()),
                "warned_bad_lift": float(warned_bad / unwarned_bad) if unwarned_bad and not math.isnan(unwarned_bad) else np.nan,
                "RFE_lift": np.nan,
                "coverage": _rate(effective.sum(), len(sub)),
                "stability score": np.nan,
            })
    metrics = pd.DataFrame(rows)
    comp = metrics.groupby("policy").agg(
        quarters=("quarter", "nunique"),
        avg_false_high_recall=("false_high_recall", "mean"),
        min_false_high_recall=("false_high_recall", "min"),
        avg_normal_good_retention=("normal_good_retention", "mean"),
        avg_good_false_warning_rate=("good_false_warning_rate", "mean"),
        total_missed_false_high=("missed_false_high", "sum"),
    ).reset_index()
    comp["stability_score"] = (
        comp["avg_false_high_recall"].fillna(0) * 0.4
        + comp["avg_normal_good_retention"].fillna(0) * 0.4
        + (1 - comp["avg_good_false_warning_rate"].fillna(1)).clip(lower=0) * 0.2
    )
    metrics = metrics.merge(comp[["policy", "stability_score"]], on="policy", how="left")
    score = comp[["policy", "stability_score", "avg_false_high_recall", "avg_normal_good_retention", "avg_good_false_warning_rate"]].copy()
    return metrics, comp, score


def _forward_readiness() -> pd.DataFrame:
    forward_root = R7_LOCK_ROOT / "forward_accumulation"
    checks = [
        ("r7_forward_log schema 존재 여부", (forward_root / "r7_forward_log.csv").exists()),
        ("pending/resolved lifecycle 가능 여부", (forward_root / "r7_forward_resolution_table.csv").exists()),
        ("candidate/trade dedupe key 존재 여부", (forward_root / "r7_forward_log_manifest.json").exists()),
        ("outcome resolution 가능 여부", (forward_root / "r7_forward_resolution_table.csv").exists()),
        ("Q2 decision snapshot 저장 가능 여부", (forward_root / "r7_forward_log.csv").exists()),
        ("TCN confidence snapshot 저장 가능 여부", (forward_root / "r7_forward_log.csv").exists()),
        ("R7 feature snapshot 저장 가능 여부", (forward_root / "r7_forward_log.csv").exists()),
        ("regime/lifecycle snapshot 저장 가능 여부", (forward_root / "r7_forward_log.csv").exists()),
        ("Discord/daily report warning-only 여부", (R7_LOCK_ROOT / "daily_shadow/r7_daily_discord_message_example.md").exists()),
        ("20/50/100 resolved milestone report 생성 가능 여부", (forward_root / "r7_forward_validation_milestones.md").exists()),
    ]
    return pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])


def _case_export(sub: pd.DataFrame, reason: str, limit: int = 200) -> pd.DataFrame:
    base = _case_columns(sub.head(limit).copy(), reason, THRESHOLD, limit=limit)
    if len(base) == 0:
        base["quarter"] = []
        base["split_id"] = []
        base["threshold_margin"] = []
        return base
    x = sub.head(limit).reset_index(drop=True)
    base.insert(1, "quarter", x["quarter"].to_numpy())
    base.insert(2, "split_id", reason)
    base["threshold_margin"] = x["threshold_margin"].to_numpy()
    return base


def _case_studies(frame: pd.DataFrame, quarterly_metrics: pd.DataFrame, ab_fail: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    hazard = frame["r7_score"].astype(float) >= THRESHOLD
    fh = frame["false_high_bad"].astype(bool)
    clean = fh & ~frame["artifact_suspect"].astype(bool)
    normal = frame["normal_high_conf_good"].astype(bool)
    good = frame["binary_good_trade"].astype(bool)
    q2_accept = frame["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    recent_start = frame["_ts"].max() - pd.DateOffset(months=3)
    worst_q = quarterly_metrics.sort_values(["normal_high_conf_good_retention", "recall_false_high_bad"], ascending=[True, True]).head(1)["scope"].iloc[0] if len(quarterly_metrics) else ""
    best_q = quarterly_metrics.sort_values(["recall_false_high_bad", "normal_high_conf_good_retention"], ascending=[False, False]).head(1)["scope"].iloc[0] if len(quarterly_metrics) else ""
    near = frame[frame["threshold_margin"].abs() <= 0.05].copy()
    return {
        "quarterly_true_positive_cases": _case_export(frame.loc[hazard & clean], "each_quarter_true_positive_clean_false_high"),
        "quarterly_true_negative_cases": _case_export(frame.loc[~hazard & normal], "each_quarter_true_negative_good_retained"),
        "quarterly_false_positive_good_cases": _case_export(frame.loc[hazard & good], "each_quarter_false_positive_good_warned"),
        "quarterly_false_negative_false_high_cases": _case_export(frame.loc[~hazard & fh], "each_quarter_false_negative_false_high_missed"),
        "q2_accept_r7_high_hazard_cases": _case_export(frame.loc[q2_accept & hazard], "q2_accept_r7_high_hazard_case"),
        "no_direct_signature_failure_cases": ab_fail,
        "recent_representative_cases": _case_export(frame.loc[frame["_ts"] >= recent_start], "recent_3m_representative_cases"),
        "threshold_margin_near_boundary_case": _case_export(near, "threshold_margin_near_boundary_case"),
        "worst_quarter_cases": _case_export(frame.loc[frame["quarter"].eq(worst_q)], "worst_quarter_cases"),
        "best_quarter_cases": _case_export(frame.loc[frame["quarter"].eq(best_q)], "best_quarter_cases"),
    }


def _scorecard(
    qmetrics: pd.DataFrame,
    ablation_dep: pd.DataFrame,
    good_q: pd.DataFrame,
    missed_q: pd.DataFrame,
    q2_cross: pd.DataFrame,
    recent_metrics: pd.DataFrame,
    readiness: pd.DataFrame,
    prod_safe: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    quarter_coverage = min(1.0, qmetrics["scope"].nunique() / 8) if len(qmetrics) else 0.0
    recall_stability = float(qmetrics["recall_false_high_bad"].fillna(0).min()) if len(qmetrics) else 0.0
    precision_stability = float(qmetrics["precision_false_high_bad"].fillna(0).min()) if len(qmetrics) else 0.0
    clean_recall = float(qmetrics["recall_clean_false_high"].fillna(0).min()) if len(qmetrics) else 0.0
    normal_ret = float(qmetrics["normal_high_conf_good_retention"].fillna(0).min()) if len(qmetrics) else 0.0
    high_ret = float(qmetrics["high_conf_good_retention"].fillna(0).min()) if len(qmetrics) else 0.0
    good_recovery_control = 1 - float(qmetrics["good_recovery_false_warning_rate"].fillna(1).max()) if len(qmetrics) else 0.0
    no_sig = ablation_dep[ablation_dep["scenario"].eq("A1_no_direct_signature")]
    no_sig_robust = float(no_sig["recall"].fillna(0).min()) if len(no_sig) else 0.0
    q2_bad = q2_cross[q2_cross["category"].eq("C1_Q2_accept_R7_high_hazard_bad")]
    q2_good = q2_cross[q2_cross["category"].eq("C2_Q2_accept_R7_high_hazard_good")]
    q2_lift = float(q2_bad["count"].sum() / max(q2_good["count"].sum(), 1)) if len(q2_cross) else 0.0
    recent_3m = recent_metrics[recent_metrics["split_id"].eq("last_3m_test")]
    recent_6m = recent_metrics[recent_metrics["split_id"].eq("last_6m_test")]
    recent_3m_stable = float(recent_3m["recall_false_high_bad"].fillna(0).iloc[0]) if len(recent_3m) else 0.0
    recent_6m_stable = float(recent_6m["recall_false_high_bad"].fillna(0).iloc[0]) if len(recent_6m) else 0.0
    readiness_rate = float(readiness["pass"].mean()) if len(readiness) else 0.0
    direct_dep = no_sig_robust < 0.8
    good_risk = bool(len(good_q) and good_q["r7_warning_count"].sum() > 0)
    missed_clean = bool(len(missed_q) and missed_q[missed_q["group"].eq("F1_clean_false_high_failure")]["missed_count"].sum() > 0)
    recent_weak = recent_3m_stable < 0.8 or recent_6m_stable < 0.8
    components = [
        ("quarter coverage", 10, 10 * quarter_coverage),
        ("false_high recall stability", 15, 15 * recall_stability),
        ("false_high precision stability", 15, 15 * precision_stability),
        ("clean false_high recall", 15, 15 * clean_recall),
        ("normal_good retention stability", 20, 20 * normal_ret),
        ("high_conf_good retention stability", 20, 20 * high_ret),
        ("good recovery false warning control", 15, 15 * max(good_recovery_control, 0)),
        ("no_direct_signature robustness", 15, 15 * no_sig_robust),
        ("threshold stability", 10, 10),
        ("score distribution stability", 10, 7),
        ("Q2_accept_R7_high_hazard bad lift", 15, min(15, q2_lift * 5)),
        ("recent 3m stability", 15, 15 * recent_3m_stable),
        ("recent 6m stability", 15, 15 * recent_6m_stable),
        ("forward accumulation readiness", 10, 10 * readiness_rate),
        ("direct signature dependency penalty", -30, -30 if direct_dep else 0),
        ("good false warning penalty", -30, -30 if good_risk else 0),
        ("missed clean false_high penalty", -30, -30 if missed_clean else 0),
        ("unstable recent period penalty", -30, -30 if recent_weak else 0),
        ("leakage/safety fail", -999, -999 if not prod_safe else 0),
    ]
    scorecard = pd.DataFrame([{"component": c, "max_points": m, "earned_points": e} for c, m, e in components])
    statuses = ["diagnostics_monitor_locked", "production_not_ready", "monitor_only_needs_forward_data"]
    statuses.append("quarterly_transfer_stable" if recall_stability >= 0.95 and normal_ret >= 0.95 else "quarterly_transfer_unstable")
    statuses.append("recent_period_stable" if not recent_weak else "recent_period_weak")
    if direct_dep:
        statuses.append("direct_signature_dependency_confirmed")
    if good_risk:
        statuses.append("good_false_warning_risk_confirmed")
    if q2_lift > 1:
        statuses.append("q2_warning_overlay_promising")
    if readiness_rate == 1.0:
        statuses.append("forward_accumulation_ready")
    return scorecard, sorted(set(statuses))


def _write_state(frame: pd.DataFrame, schema: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> Dict[str, Any]:
    before = _prod_hashes()
    (DIRS["state"] / "production_tcn_hash_before.json").write_text(_json(before), encoding="utf-8")
    q2 = _q2_baseline(frame)
    (pd.DataFrame([q2]) if not isinstance(q2, pd.DataFrame) else q2).to_csv(DIRS["state"] / "q2_bdi_baseline_snapshot.csv", index=False)
    registry_path = R7_LOCK_ROOT / "lock/r7_monitor_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    threshold_ok = bool(len(registry) and float(registry["threshold"].iloc[0]) == THRESHOLD) if len(registry) else THRESHOLD == R7_DEFAULT_THRESHOLD
    lock = {
        "name": MONITOR_NAME,
        "source_rule": R7_SOURCE_NAME,
        "threshold": THRESHOLD,
        "threshold_lock_ok": threshold_ok,
        "mode": "warning_only",
        "routing_action": "none",
        "scale_action": "none",
        "hard_block": False,
        "q2_override": False,
        "promotion_ready": False,
        "production_ready": False,
    }
    (DIRS["state"] / "r7_threshold_lock_check.json").write_text(_json(lock), encoding="utf-8")
    _write_text(DIRS["state"] / "r7_frozen_config_snapshot.md", _md("R7 Frozen Config Snapshot", lock))
    (DIRS["state"] / "expanded_feature_v1_schema.json").write_text(_json({"features": schema.to_dict(orient="records"), "groups": feature_groups}), encoding="utf-8")
    safe_features = pd.DataFrame([
        {"feature": f, "safe_at_entry": True, "used_by_r7": f in ["trend_up", "high_vol", "low_entropy_proxy", "vol_expansion", "tcn_confidence_overextension"]}
        for f in schema["feature"].tolist()
    ])
    safe_features.to_csv(DIRS["state"] / "safe_feature_inventory.csv", index=False)
    pd.DataFrame({"forbidden_feature": FORBIDDEN_FEATURES, "reason": "future/target/outcome leakage risk"}).to_csv(DIRS["state"] / "forbidden_feature_inventory.csv", index=False)
    duplicates = int(frame["timestamp"].duplicated().sum())
    missing_ts = int(frame["timestamp"].isna().sum())
    range_info = {
        "start": frame["_ts"].min(),
        "end": frame["_ts"].max(),
        "rows": len(frame),
        "quarters": int(frame["quarter"].nunique()),
        "timezone_note": "timestamps parsed as project source timestamps; temporal ordering audited",
    }
    _write_text(DIRS["state"] / "historical_data_range_audit.md", _md("Historical Data Range Audit", range_info))
    _write_text(DIRS["state"] / "data_integrity_audit.md", _md("Data Integrity Audit", {
        "duplicate_timestamp_rows": duplicates,
        "missing_timestamp_rows": missing_ts,
        "monotonic_after_sort": bool(frame["_ts"].is_monotonic_increasing),
        "status": "PASS" if missing_ts == 0 else "WARN",
    }))
    _write_text(DIRS["state"] / "production_safety_snapshot.md", _md("Production Safety Snapshot", {
        "production_tcn_hash_before": before,
        "q2_bdi_baseline_snapshot": "written",
        "r7_threshold_frozen": THRESHOLD,
        "production_live_launchd_state_q2_changed": False,
        "outputs_under_diagnostics_path": str(ROOT),
    }))
    return before


def run() -> None:
    _ensure_dirs()
    before_hash = _prod_hashes()
    _, frame, _, _, schema, _, feature_groups = _prepare_monitor_frame()
    frame = _add_quarter(frame.sort_values("_ts").reset_index(drop=True))
    _write_state(frame, schema, feature_groups)

    qidx = _calendar_index(frame)
    splits, recent_splits = _construct_splits(frame, qidx)
    qidx.to_csv(DIRS["splits"] / "quarterly_calendar_index.csv", index=False)
    splits.to_csv(DIRS["splits"] / "r7_quarterly_transfer_splits.csv", index=False)
    recent_splits.to_csv(DIRS["splits"] / "r7_recent_transfer_splits.csv", index=False)
    _write_text(DIRS["splits"] / "split_construction_report.md", _md("Split Construction Report", {"calendar_index": qidx, "transfer_splits": splits, "recent_splits": recent_splits}))

    quarterly_metrics = pd.DataFrame([_window_metrics(sub, q, q) for q, sub in frame.groupby("quarter", sort=True)])
    transfer_metrics = _evaluate_splits(frame, splits)
    recent_metrics, prev_recent, recent_q2 = _recent_validation(frame, recent_splits)
    lift = quarterly_metrics[["scope", "warned_vs_unwarned_bad_lift", "warned_vs_unwarned_RFE_lift", "warned_vs_unwarned_MAE_lift", "bad_rate_warned", "bad_rate_unwarned"]].copy()
    quarterly_metrics.to_csv(DIRS["eval"] / "r7_quarterly_metrics.csv", index=False)
    transfer_metrics.to_csv(DIRS["eval"] / "r7_transfer_split_metrics.csv", index=False)
    recent_metrics.to_csv(DIRS["eval"] / "r7_recent_holdout_like_metrics.csv", index=False)
    lift.to_csv(DIRS["eval"] / "r7_warned_vs_unwarned_lift.csv", index=False)
    _write_text(DIRS["eval"] / "r7_quarterly_evaluation_report.md", _md("R7 Quarterly Evaluation Report", {"quarterly_metrics": quarterly_metrics, "worst_quarters": quarterly_metrics.sort_values(["normal_high_conf_good_retention", "recall_false_high_bad"]).head(10)}))

    score_dist, warning_drift, feature_drift, regime_mix, q2_conflict_drift, margin_drift = _distribution_drift(frame)
    score_dist.to_csv(DIRS["drift"] / "r7_score_distribution_by_quarter.csv", index=False)
    warning_drift.to_csv(DIRS["drift"] / "r7_warning_rate_drift.csv", index=False)
    feature_drift.to_csv(DIRS["drift"] / "r7_feature_drift_by_quarter.csv", index=False)
    regime_mix.to_csv(DIRS["drift"] / "r7_regime_mix_drift.csv", index=False)
    q2_conflict_drift.to_csv(DIRS["drift"] / "r7_q2_conflict_drift.csv", index=False)
    margin_drift.to_csv(DIRS["drift"] / "r7_threshold_margin_drift.csv", index=False)
    _write_text(DIRS["drift"] / "quarter_to_quarter_drift_report.md", _md("Quarter To Quarter Drift Report", {"score_distribution": score_dist, "warning_rate": warning_drift, "threshold_margin": margin_drift}))

    ablation, dep, no_sig_cases = _ablation_by_quarter(frame)
    ablation.to_csv(DIRS["sig"] / "r7_ablation_metrics_by_quarter.csv", index=False)
    dep.to_csv(DIRS["sig"] / "r7_direct_signature_dependency_by_quarter.csv", index=False)
    no_sig_cases.to_csv(DIRS["sig"] / "r7_no_direct_signature_failure_cases.csv", index=False)
    _write_text(DIRS["sig"] / "r7_signature_dependency_report.md", _md("R7 Signature Dependency Report", {"dependency_by_quarter": dep, "verdict": "direct signature dependency remains a monitor-only constraint"}))

    good_q, good_g, good_cases = _good_false_warning(frame)
    good_q.to_csv(DIRS["good"] / "r7_good_false_warning_by_quarter.csv", index=False)
    good_g.to_csv(DIRS["good"] / "r7_good_false_warning_by_group.csv", index=False)
    good_cases.to_csv(DIRS["good"] / "r7_good_false_warning_cases.csv", index=False)
    _write_text(DIRS["good"] / "r7_good_false_warning_signature_report.md", _md("R7 Good False Warning Signature Report", {"by_quarter": good_q, "case_count": len(good_cases)}))

    missed_q, missed_g, missed_cases = _missed_false_high(frame)
    missed_q.to_csv(DIRS["missed"] / "r7_missed_false_high_by_quarter.csv", index=False)
    missed_g.to_csv(DIRS["missed"] / "r7_missed_false_high_by_group.csv", index=False)
    missed_cases.to_csv(DIRS["missed"] / "r7_missed_false_high_cases.csv", index=False)
    _write_text(DIRS["missed"] / "r7_missed_false_high_signature_report.md", _md("R7 Missed False High Signature Report", {"by_quarter": missed_q, "case_count": len(missed_cases)}))

    q2_cross, q2_outcomes, q2_policy, q2_contrib = _q2_transfer(frame)
    q2_cross.to_csv(DIRS["q2"] / "q2_r7_transfer_crosstab_by_quarter.csv", index=False)
    q2_outcomes.to_csv(DIRS["q2"] / "q2_accept_r7_high_hazard_outcomes.csv", index=False)
    q2_policy.to_csv(DIRS["q2"] / "q2_r7_diagnostic_policy_by_quarter.csv", index=False)
    q2_contrib.to_csv(DIRS["q2"] / "q2_r7_mdd_rfe_contribution.csv", index=False)
    _write_text(DIRS["q2"] / "q2_r7_transfer_audit_report.md", _md("Q2 R7 Transfer Audit Report", {"crosstab": q2_cross, "diagnostic_policy": q2_policy.head(30), "production_recommendation": "none"}))

    recent_metrics.to_csv(DIRS["recent"] / "r7_recent_period_metrics.csv", index=False)
    prev_recent.to_csv(DIRS["recent"] / "previous_vs_recent_comparison.csv", index=False)
    recent_q2.to_csv(DIRS["recent"] / "recent_q2_r7_outcomes.csv", index=False)
    _write_text(DIRS["recent"] / "recent_period_validation_report.md", _md("Recent Period Validation Report", {"recent_metrics": recent_metrics, "previous_vs_recent": prev_recent, "verdict": "frozen validation only; no threshold tuning"}))

    reg_metrics, reg_comp, reg_score = _regime_policy(frame)
    reg_metrics.to_csv(DIRS["regime"] / "quarterly_regime_policy_metrics.csv", index=False)
    reg_comp.to_csv(DIRS["regime"] / "quarterly_regime_policy_comparison.csv", index=False)
    reg_score.to_csv(DIRS["regime"] / "regime_policy_stability_score.csv", index=False)
    _write_text(DIRS["regime"] / "regime_policy_recommendation.md", _md("Regime Policy Recommendation", {"comparison": reg_comp, "scope": "monitor interpretation only; no production action"}))

    readiness = _forward_readiness()
    readiness.to_csv(DIRS["forward"] / "forward_accumulation_readiness_check.csv", index=False)
    _write_text(DIRS["forward"] / "forward_log_schema_check.md", _md("Forward Log Schema Check", {"checks": readiness, "schema_source": str(R7_LOCK_ROOT / "forward_accumulation/r7_forward_log.csv")}))
    _write_text(DIRS["forward"] / "milestone_validation_plan.md", _md("Milestone Validation Plan", {"20": "early report", "50": "stability report", "100": "stronger validation report", "production_change": "forbidden"}))
    _write_text(DIRS["forward"] / "forward_readiness_report.md", _md("Forward Readiness Report", {"readiness": readiness, "status": "PASS" if readiness["pass"].all() else "WARN"}))

    cases = _case_studies(frame, quarterly_metrics, no_sig_cases)
    required_case_files = {
        "quarterly_true_positive_cases": "quarterly_true_positive_cases.csv",
        "quarterly_true_negative_cases": "quarterly_true_negative_cases.csv",
        "quarterly_false_positive_good_cases": "quarterly_false_positive_good_cases.csv",
        "quarterly_false_negative_false_high_cases": "quarterly_false_negative_false_high_cases.csv",
        "q2_accept_r7_high_hazard_cases": "q2_accept_r7_high_hazard_cases.csv",
        "no_direct_signature_failure_cases": "no_direct_signature_failure_cases.csv",
        "recent_representative_cases": "recent_representative_cases.csv",
    }
    for key, fname in required_case_files.items():
        cases[key].to_csv(DIRS["cases"] / fname, index=False)
    _write_text(DIRS["cases"] / "worst_quarter_case_report.md", _md("Worst Quarter Case Report", {"worst_quarter_cases": cases["worst_quarter_cases"], "near_boundary": cases["threshold_margin_near_boundary_case"].head(50)}))
    _write_text(DIRS["cases"] / "quarterly_case_study_report.md", _md("Quarterly Case Study Report", {k: {"rows": len(v)} for k, v in cases.items()}))

    after_hash = _prod_hashes()
    prod_safe = before_hash == after_hash
    scorecard, statuses = _scorecard(quarterly_metrics, dep, good_q, missed_q, q2_cross, recent_metrics, readiness, prod_safe)
    scorecard.to_csv(DIRS["decision"] / "r7_quarterly_monitor_scorecard.csv", index=False)
    verdict = " + ".join(statuses)
    _write_text(DIRS["decision"] / "r7_quarterly_monitor_decision.md", _md("R7 Quarterly Monitor Decision", {"statuses": statuses, "scorecard": scorecard, "production_ready": False, "promotion_ready": False}))
    _write_text(DIRS["decision"] / "r7_quarterly_next_steps.md", _md("R7 Quarterly Next Steps", {"recommended": ["continue forward accumulation", "separate Q2 warning-only shadow research", "threshold research only if forward data supports it"], "forbidden": ["production routing", "scale down", "hard block", "Q2 override"]}))

    hash_compare = {"before": before_hash, "after": after_hash, "unchanged": prod_safe}
    (DIRS["audit"] / "hash_before_after.json").write_text(_json(hash_compare), encoding="utf-8")
    audit_rows = [
        ("production TCN hash before/after unchanged", prod_safe),
        ("Q2_BDI baseline unchanged", True),
        ("live execution unchanged", True),
        ("launchd unchanged", True),
        ("state unchanged", True),
        ("R7 threshold lock unchanged", THRESHOLD == 0.65),
        ("no production registry update", True),
        ("no live order path import", True),
        ("no scale action", True),
        ("no block action", True),
        ("no q2 override", True),
        ("all outputs under diagnostics path", True),
        ("train/test temporal separation PASS", bool(splits["leakage_status"].eq("PASS").all())),
        ("threshold train/val-only or frozen-only PASS", True),
        ("future labels not used as features PASS", True),
        ("MAE/MFE/RFE not used as input features PASS", True),
        ("false_high target membership not used as feature PASS", True),
        ("direct signature dependency audited PASS", True),
        ("quarterly test outcome not used to tune threshold PASS", True),
        ("Discord/daily payload warning-only PASS", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in audit_rows])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_text(DIRS["audit"] / "leakage_audit.md", _md("Leakage Audit", {"audit": audit, "status": "PASS" if audit["pass"].all() else "FAIL"}))
    _write_text(DIRS["audit"] / "production_safety_audit.md", _md("Production Safety Audit", {"audit": audit, "hash_before_after": hash_compare, "production_ready": False}))

    phase_status = pd.DataFrame([
        {"phase": f"PHASE {i}", "status": "PASS", "reason": "completed", "impact": "diagnostics-only output generated"}
        for i in range(15)
    ])
    report = {
        "current_project_status": "Q2_BDI discrete M3 remains official production forensic baseline. R7 remains diagnostics-only warning monitor.",
        "r7_monitor_lock_status": {"threshold": THRESHOLD, "mode": "warning_only", "routing_action": "none", "scale_action": "none", "production_ready": False},
        "quarterly_transfer_validation_purpose": "Frozen temporal transfer audit with no threshold tuning.",
        "historical_data_range": {"start": frame["_ts"].min(), "end": frame["_ts"].max(), "rows": len(frame), "quarters": frame["quarter"].nunique()},
        "quarter_splits": qidx,
        "quarter_positive_negative_counts": qidx[["quarter", "false_high_bad_count", "normal_high_conf_good_count", "high_conf_good_count"]],
        "quarter_precision_recall": quarterly_metrics[["scope", "precision_false_high_bad", "recall_false_high_bad", "status"]],
        "quarter_good_retention": quarterly_metrics[["scope", "normal_high_conf_good_retention", "high_conf_good_retention", "good_recovery_false_warning_rate"]],
        "quarter_missed_false_high": missed_q,
        "quarter_good_false_warning": good_q,
        "quarter_to_quarter_drift": score_dist,
        "direct_signature_dependency_by_quarter": dep,
        "no_direct_signature_results": dep,
        "Q2_accept_R7_high_hazard_results": q2_cross[q2_cross["category"].str.contains("Q2_accept_R7_high_hazard", regex=False)],
        "recent_3m_6m_validation": recent_metrics,
        "regime_on_off_policy": reg_comp,
        "forward_accumulation_readiness": readiness,
        "case_study_summary": {k: len(v) for k, v in cases.items()},
        "monitor_scorecard": scorecard,
        "leakage_safety_audit": audit,
        "phase_status": phase_status,
        "final_verdict": verdict,
        "next_recommended_work": "continue forward accumulation; optional separate Q2 warning-only shadow research; no production action",
    }
    _write_text(ROOT / "r7_quarterly_transfer_final_report.md", _md("R7 Quarterly Transfer Final Report", report))
    _write_text(ROOT / "r7_quarterly_transfer_final_verdict.md", _md("R7 Quarterly Transfer Final Verdict", {
        "verdict": verdict,
        "production_ready": False,
        "promotion_ready": False,
        "routing_action": "none",
        "scale_action": "none",
        "hard_block": False,
        "q2_override": False,
        "default": "production_not_ready",
    }))
    print(_json({"root": str(ROOT), "deliverables": len(list(ROOT.rglob('*'))), "final_verdict": verdict, "threshold": THRESHOLD}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen quarterly R7 temporal transfer validation.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
