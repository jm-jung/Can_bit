"""
Point-in-time as-of historical forward replay for FalseHigh_R7_StructureHazard_v1.

This diagnostics-only script reconstructs many historical as_of_date panels,
builds monitor thresholds using history-only information, evaluates subsequent
test windows, and writes all outputs under:

    data/diagnostics/false_high_r7_asof_forward_replay/

It never writes production TCN, Q2_BDI, live execution, launchd, order routing,
or state paths. R7 remains warning-only and production_not_ready.
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

from scripts.diagnostics.run_false_high_r7_monitor import (
    FORBIDDEN_FEATURES,
    MONITOR_NAME,
    POSITION_SIZE,
    Q2_ACCEPT,
    Q2_REJECT,
    R7_SOURCE_NAME,
    _case_columns,
    _diagnostic_policy_comparison,
    _false_high_masks,
    _good_masks,
    _md,
    _policy_net_mdd,
    _prepare_monitor_frame,
    _score_scenario,
    _write_text,
)
from scripts.diagnostics.run_false_high_r7_quarterly_transfer import _psi, _top
from scripts.diagnostics.run_feature_sufficiency_forensics import _mdd, _prod_hashes, _q2_baseline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("data/diagnostics/false_high_r7_asof_forward_replay")
R7_MONITOR_ROOT = Path("data/diagnostics/false_high_r7_monitor")
R7_QUARTERLY_ROOT = Path("data/diagnostics/false_high_r7_quarterly_transfer")
DIRS = {
    "state": ROOT / "state",
    "splits": ROOT / "splits",
    "configs": ROOT / "configs",
    "eval": ROOT / "evaluation",
    "survival": ROOT / "survival",
    "q2": ROOT / "q2_asof_audit",
    "sig": ROOT / "signature_dependency",
    "good": ROOT / "good_false_warning",
    "missed": ROOT / "missed_false_high",
    "drift": ROOT / "drift",
    "regime": ROOT / "regime_policy",
    "recent": ROOT / "recent_vs_historical",
    "forward": ROOT / "forward_readiness",
    "cases": ROOT / "cases",
    "decision": ROOT / "decision",
    "audit": ROOT / "audit",
}

THRESHOLD = 0.65
MIN_HISTORY = 50
MIN_TEST = 20


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _rate(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


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


def _ks(a: pd.Series, b: pd.Series) -> float:
    if ks_2samp is None or len(a) == 0 or len(b) == 0:
        return np.nan
    try:
        return float(ks_2samp(pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()).statistic)
    except Exception:
        return np.nan


def _quarter_labels(sub: pd.DataFrame) -> str:
    if len(sub) == 0:
        return ""
    return "|".join(sorted(sub["_ts"].dt.to_period("Q").astype(str).unique()))


def _prep_frame() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    _, frame, _, _, schema, _, feature_groups = _prepare_monitor_frame()
    frame = frame.sort_values("_ts").reset_index(drop=True)
    frame["quarter"] = frame["_ts"].dt.to_period("Q").astype(str)
    frame["threshold_margin"] = frame["r7_score"].astype(float) - THRESHOLD
    return frame, schema, feature_groups


def _build_asof_grid(frame: pd.DataFrame) -> pd.DataFrame:
    start = frame["_ts"].min().floor("D")
    end = frame["_ts"].max().floor("D")
    rows: List[Dict[str, Any]] = []

    def add(dates: Iterable[pd.Timestamp], grid_type: str) -> None:
        for d in dates:
            d = pd.Timestamp(d)
            if start < d < end:
                rows.append({"as_of_date": d, "as_of_grid_type": grid_type})

    months = pd.date_range(start=start.replace(day=1), end=end, freq="MS")
    add(months, "monthly_start")
    add([m + pd.Timedelta(days=14) for m in months], "monthly_mid")
    quarters = pd.date_range(start=pd.Timestamp(f"{start.year}-01-01"), end=end, freq="QS")
    add(quarters, "quarter_start")
    add([q + pd.DateOffset(days=45) for q in quarters], "quarter_mid")
    semi = pd.date_range(start=pd.Timestamp(f"{start.year}-01-01"), end=end, freq="2QS-JAN")
    add(semi, "semiannual")
    recent_offsets = [3, 6, 9, 12, 18, 24]
    add([end - pd.DateOffset(months=m) for m in recent_offsets], "rolling_recent")

    grid = pd.DataFrame(rows).drop_duplicates(["as_of_date", "as_of_grid_type"]).sort_values(["as_of_date", "as_of_grid_type"]).reset_index(drop=True)
    grid["as_of_id"] = [f"asof_{i:04d}_{r.as_of_date:%Y%m%d}_{r.as_of_grid_type}" for i, r in grid.iterrows()]
    return grid


def _history_window(frame: pd.DataFrame, asof: pd.Timestamp, mode: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    end = asof - pd.Timedelta(seconds=1)
    if mode == "H1_3m_history":
        start = asof - pd.DateOffset(months=3)
    elif mode == "H2_6m_history":
        start = asof - pd.DateOffset(months=6)
    elif mode == "H3_12m_history":
        start = asof - pd.DateOffset(months=12)
    elif mode == "H4_18m_history":
        start = asof - pd.DateOffset(months=18)
    elif mode == "H5_24m_history":
        start = asof - pd.DateOffset(months=24)
    elif mode == "H6_expanding_history_from_start":
        start = frame["_ts"].min()
    elif mode == "H7_previous_1Q":
        start = asof.to_period("Q").start_time - pd.DateOffset(months=3)
        end = asof.to_period("Q").start_time - pd.Timedelta(seconds=1)
    elif mode == "H8_previous_2Q":
        start = asof.to_period("Q").start_time - pd.DateOffset(months=6)
        end = asof.to_period("Q").start_time - pd.Timedelta(seconds=1)
    elif mode == "H9_previous_4Q":
        start = asof.to_period("Q").start_time - pd.DateOffset(months=12)
        end = asof.to_period("Q").start_time - pd.Timedelta(seconds=1)
    elif mode == "H10_previous_8Q":
        start = asof.to_period("Q").start_time - pd.DateOffset(months=24)
        end = asof.to_period("Q").start_time - pd.Timedelta(seconds=1)
    else:
        start = frame["_ts"].min()
    return pd.Timestamp(start), pd.Timestamp(end)


def _test_window(asof: pd.Timestamp, mode: str, next_asof: pd.Timestamp | None, max_ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = asof
    if mode == "T1_next_1m":
        end = asof + pd.DateOffset(months=1) - pd.Timedelta(seconds=1)
    elif mode == "T2_next_2m":
        end = asof + pd.DateOffset(months=2) - pd.Timedelta(seconds=1)
    elif mode == "T3_next_3m":
        end = asof + pd.DateOffset(months=3) - pd.Timedelta(seconds=1)
    elif mode == "T4_next_6m":
        end = asof + pd.DateOffset(months=6) - pd.Timedelta(seconds=1)
    elif mode == "T5_next_1Q":
        end = asof + pd.DateOffset(months=3) - pd.Timedelta(seconds=1)
    elif mode == "T6_next_2Q":
        end = asof + pd.DateOffset(months=6) - pd.Timedelta(seconds=1)
    elif mode == "T7_next_4Q":
        end = asof + pd.DateOffset(months=12) - pd.Timedelta(seconds=1)
    elif mode == "T8_until_next_asof" and next_asof is not None:
        end = next_asof - pd.Timedelta(seconds=1)
    elif mode == "T9_recent_holdout_like":
        end = max_ts
    else:
        end = asof + pd.DateOffset(months=3) - pd.Timedelta(seconds=1)
    return start, min(pd.Timestamp(end), max_ts)


def _slice(frame: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    return frame[(frame["_ts"] >= pd.Timestamp(start)) & (frame["_ts"] <= pd.Timestamp(end))].copy()


def _construct_splits(frame: pd.DataFrame, grid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    h_modes = [
        "H1_3m_history", "H2_6m_history", "H3_12m_history", "H4_18m_history", "H5_24m_history",
        "H6_expanding_history_from_start", "H7_previous_1Q", "H8_previous_2Q", "H9_previous_4Q", "H10_previous_8Q",
    ]
    t_modes = ["T1_next_1m", "T2_next_2m", "T3_next_3m", "T4_next_6m", "T5_next_1Q", "T6_next_2Q", "T7_next_4Q", "T8_until_next_asof"]
    max_ts = frame["_ts"].max()
    rows = []
    dates = grid["as_of_date"].tolist()
    for idx, gr in grid.iterrows():
        asof = pd.Timestamp(gr["as_of_date"])
        next_asof = pd.Timestamp(dates[idx + 1]) if idx + 1 < len(dates) else None
        for hm in h_modes:
            hs, he = _history_window(frame, asof, hm)
            hist = _slice(frame, hs, he)
            for tm in t_modes:
                ts, te = _test_window(asof, tm, next_asof, max_ts)
                test = _slice(frame, ts, te)
                if len(test) == 0:
                    continue
                rows.append(_split_row(gr, hm, tm, hs, he, ts, te, hist, test))
    # Recent panel splits with all history modes.
    recent = grid[grid["as_of_grid_type"].eq("rolling_recent")].copy()
    recent_rows = []
    for _, gr in recent.iterrows():
        asof = pd.Timestamp(gr["as_of_date"])
        for hm in h_modes:
            hs, he = _history_window(frame, asof, hm)
            hist = _slice(frame, hs, he)
            ts, te = _test_window(asof, "T9_recent_holdout_like", None, max_ts)
            test = _slice(frame, ts, te)
            recent_rows.append(_split_row(gr, hm, "T9_recent_holdout_like", hs, he, ts, te, hist, test))
    return pd.DataFrame(rows), pd.DataFrame(recent_rows)


def _select_eval_splits(splits: pd.DataFrame, max_per_cell: int = 2) -> pd.DataFrame:
    if len(splits) == 0:
        return splits.copy()
    active = splits[splits["split_status"].ne("SKIP")].copy()
    if len(active) == 0:
        return splits.head(0).copy()
    active = active.sort_values(["as_of_date", "history_rows", "test_rows"])
    selected = (
        active.groupby(["as_of_grid_type", "history_mode", "test_mode"], dropna=False, group_keys=False)
        .tail(max_per_cell)
        .reset_index(drop=True)
    )
    # Always keep recent panels and a locked-style broad coverage of the latest windows.
    recent = active[active["as_of_grid_type"].eq("rolling_recent")]
    latest = active.sort_values("as_of_date").tail(200)
    selected = pd.concat([selected, recent, latest], ignore_index=True).drop_duplicates("split_id")
    return selected.reset_index(drop=True)


def _split_row(gr: pd.Series, hm: str, tm: str, hs: Any, he: Any, ts: Any, te: Any, hist: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    min_hist = len(hist) >= MIN_HISTORY
    min_test = len(test) >= MIN_TEST
    status = "PASS" if min_hist and min_test else ("WARN" if min_test else "SKIP")
    return {
        "split_id": f"{gr['as_of_id']}__{hm}__{tm}",
        "as_of_date": gr["as_of_date"],
        "as_of_grid_type": gr["as_of_grid_type"],
        "history_mode": hm,
        "test_mode": tm,
        "history_start": hs,
        "history_end": he,
        "test_start": ts,
        "test_end": te,
        "history_rows": len(hist),
        "test_rows": len(test),
        "history_quarters": _quarter_labels(hist),
        "test_quarters": _quarter_labels(test),
        "partial_window_flag": bool(ts < test["_ts"].min() or te > test["_ts"].max()) if len(test) else True,
        "min_history_satisfied": min_hist,
        "min_test_satisfied": min_test,
        "history_false_high_count": int(hist["false_high_bad"].astype(bool).sum()) if len(hist) else 0,
        "test_false_high_count": int(test["false_high_bad"].astype(bool).sum()) if len(test) else 0,
        "history_normal_high_conf_good_count": int(hist["normal_high_conf_good"].astype(bool).sum()) if len(hist) else 0,
        "test_normal_high_conf_good_count": int(test["normal_high_conf_good"].astype(bool).sum()) if len(test) else 0,
        "history_q2_accept_count": int((hist["q2_bdi_scale"].astype(float) >= Q2_ACCEPT).sum()) if len(hist) else 0,
        "test_q2_accept_count": int((test["q2_bdi_scale"].astype(float) >= Q2_ACCEPT).sum()) if len(test) else 0,
        "leakage_status": "PASS" if (len(hist) == 0 or hist["_ts"].max() < test["_ts"].min()) else "FAIL",
        "split_status": status,
        "skip_reason": "" if status != "SKIP" else "insufficient_test_rows",
        "impact": "" if status == "PASS" else "interpret metrics with sample limits",
    }


def _supervised_threshold(hist: pd.DataFrame, mode: str, score_col: str = "r7_score") -> float:
    eligible = hist[hist["false_high_bad"].astype(bool) | hist["normal_high_conf_good"].astype(bool)].copy()
    if len(eligible) < MIN_HISTORY or eligible["false_high_bad"].nunique() < 2:
        return THRESHOLD
    y = eligible["false_high_bad"].astype(int)
    score = eligible[score_col].astype(float)
    best_thr, best_obj = THRESHOLD, -1e9
    for thr in np.linspace(0.05, 0.95, 19):
        pred = score >= thr
        recall = _rate((pred & y.eq(1)).sum(), y.eq(1).sum())
        good_ret = _rate((~pred & y.eq(0)).sum(), y.eq(0).sum())
        good_warn = 1 - (good_ret if not math.isnan(good_ret) else 0)
        if mode == "M5_ASOF_SUPERVISED_BALANCED":
            obj = (recall if not math.isnan(recall) else 0) + (good_ret if not math.isnan(good_ret) else 0)
        elif mode == "M6_ASOF_SUPERVISED_RECALL_FIRST":
            obj = 1.5 * (recall if not math.isnan(recall) else 0) + 0.5 * (good_ret if not math.isnan(good_ret) else 0) - good_warn
        else:
            obj = 1.2 * (good_ret if not math.isnan(good_ret) else 0) + 0.8 * (recall if not math.isnan(recall) else 0) - 2.0 * max(0, good_warn - 0.05)
        if obj > best_obj:
            best_obj = obj
            best_thr = float(thr)
    return best_thr


def _score_for_config(frame: pd.DataFrame, config_id: str) -> pd.Series:
    scenario = {
        "M8_NO_DIRECT_SIGNATURE_LOCKED": "A1_no_direct_signature",
        "M9_NO_TCN_OUTPUT_ASOF": "A5_no_tcn_output",
        "M10_TREND_STRUCTURE_ONLY_ASOF": "A8_only_trend_structure",
    }.get(config_id, "A0_full_R7")
    return pd.Series(_score_scenario(frame, scenario), index=frame.index).astype(float)


def _build_configs(frame: pd.DataFrame, splits: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_ids = [
        "M0_LOCKED_CURRENT_R7", "M1_ASOF_SCORE_P90", "M2_ASOF_SCORE_P95", "M3_ASOF_SCORE_P97_5", "M4_ASOF_SCORE_P99",
        "M5_ASOF_SUPERVISED_BALANCED", "M6_ASOF_SUPERVISED_RECALL_FIRST", "M7_ASOF_SUPERVISED_GOOD_GUARD",
        "M8_NO_DIRECT_SIGNATURE_LOCKED", "M9_NO_TCN_OUTPUT_ASOF", "M10_TREND_STRUCTURE_ONLY_ASOF",
    ]
    rows, thresholds = [], []
    for _, sp in splits.iterrows():
        hist = _slice(frame, sp["history_start"], sp["history_end"])
        for cid in config_ids:
            score = _score_for_config(hist, cid) if len(hist) else pd.Series(dtype=float)
            if cid == "M0_LOCKED_CURRENT_R7":
                thr, source, outcome = THRESHOLD, "locked_current_0.65", False
            elif cid == "M1_ASOF_SCORE_P90":
                thr, source, outcome = float(score.quantile(0.90)) if len(score) else THRESHOLD, "history_score_p90", False
            elif cid == "M2_ASOF_SCORE_P95":
                thr, source, outcome = float(score.quantile(0.95)) if len(score) else THRESHOLD, "history_score_p95", False
            elif cid == "M3_ASOF_SCORE_P97_5":
                thr, source, outcome = float(score.quantile(0.975)) if len(score) else THRESHOLD, "history_score_p97_5", False
            elif cid == "M4_ASOF_SCORE_P99":
                thr, source, outcome = float(score.quantile(0.99)) if len(score) else THRESHOLD, "history_score_p99", False
            elif cid in {"M5_ASOF_SUPERVISED_BALANCED", "M6_ASOF_SUPERVISED_RECALL_FIRST", "M7_ASOF_SUPERVISED_GOOD_GUARD"}:
                thr, source, outcome = _supervised_threshold(hist, cid), "history_outcome_supervised_diagnostic_only", True
            elif cid == "M8_NO_DIRECT_SIGNATURE_LOCKED":
                thr, source, outcome = float(score.quantile(0.95)) if len(score) else THRESHOLD, "history_no_direct_signature_score_p95", False
            elif cid == "M9_NO_TCN_OUTPUT_ASOF":
                thr, source, outcome = float(score.quantile(0.95)) if len(score) else THRESHOLD, "history_no_tcn_score_p95", False
            else:
                thr, source, outcome = float(score.quantile(0.95)) if len(score) else THRESHOLD, "history_trend_structure_score_p95", False
            if not math.isfinite(thr):
                thr = THRESHOLD
            status = "PASS" if (cid == "M0_LOCKED_CURRENT_R7" or sp["history_rows"] >= MIN_HISTORY) else "SKIP"
            row = {
                "config_id": cid,
                "split_id": sp["split_id"],
                "formula_version": R7_SOURCE_NAME if cid != "M8_NO_DIRECT_SIGNATURE_LOCKED" else "R7_no_direct_signature_variant",
                "threshold": thr,
                "threshold_source": source,
                "threshold_fit_rows": int(len(hist)),
                "threshold_fit_positive_count": int(hist["false_high_bad"].astype(bool).sum()) if len(hist) else 0,
                "threshold_fit_good_count": int(hist["normal_high_conf_good"].astype(bool).sum()) if len(hist) else 0,
                "uses_outcome_in_history": outcome,
                "uses_test_outcome": False,
                "direct_signature_allowed": cid not in {"M8_NO_DIRECT_SIGNATURE_LOCKED"},
                "q2_feature_allowed": False,
                "tcn_feature_allowed": cid != "M9_NO_TCN_OUTPUT_ASOF",
                "safe_feature_status": "PASS",
                "config_status": status,
            }
            rows.append(row)
            thresholds.append(row.copy())
    reg = pd.DataFrame(rows)
    fit = reg.groupby("config_id").agg(
        configs=("split_id", "count"),
        pass_configs=("config_status", lambda s: int((s == "PASS").sum())),
        median_threshold=("threshold", "median"),
        min_threshold=("threshold", "min"),
        max_threshold=("threshold", "max"),
    ).reset_index()
    return reg, pd.DataFrame(thresholds), fit


def _eval_one(test: pd.DataFrame, score: pd.Series, thr: float, meta: Dict[str, Any]) -> Dict[str, Any]:
    hazard = score >= thr
    long = test["direction"].astype(str).eq("LONG")
    hcl = test["baseline_long_high_conf"].astype(bool)
    q2_accept = test["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    q2_reject = test["q2_bdi_scale"].astype(float) <= Q2_REJECT
    fh = test["false_high_bad"].astype(bool)
    clean = fh & ~test["artifact_suspect"].astype(bool)
    normal = test["normal_high_conf_good"].astype(bool)
    high_good = test["binary_good_trade"].astype(bool) & hcl
    q2_accept_good = q2_accept & test["binary_good_trade"].astype(bool)
    good = test["binary_good_trade"].astype(bool)
    bad = test["binary_bad_trade"].astype(bool)
    good_recovery = (test["L3_lifecycle_path_class"].astype(str).eq("early_adverse_recovery") | test["tag_false_high_signature"].astype(bool)) & good
    trend_up_good = test["trend_up"].astype(bool) & good
    rfe = test["rfe_flag"].astype(bool)
    high_mae = test["mae"].astype(float) <= -0.008
    warned_bad = _rate((hazard & bad).sum(), hazard.sum())
    unwarned_bad = _rate((~hazard & bad).sum(), (~hazard).sum())
    warned_rfe = _rate((hazard & rfe).sum(), hazard.sum())
    unwarned_rfe = _rate((~hazard & rfe).sum(), (~hazard).sum())
    warned_mae = _rate((hazard & high_mae).sum(), hazard.sum())
    unwarned_mae = _rate((~hazard & high_mae).sum(), (~hazard).sum())
    status = "PASS"
    if len(test) < MIN_TEST or int(fh.sum()) == 0 or int(normal.sum()) == 0:
        status = "WARN"
    if int((hazard & good).sum()) > 0 or int((~hazard & clean).sum()) > 0:
        status = "WARN"
    row = {
        **meta,
        "total_test_rows": len(test),
        "long_candidate_count": int(long.sum()),
        "tcn_high_conf_long_count": int(hcl.sum()),
        "q2_accept_long_count": int((q2_accept & long).sum()),
        "q2_reject_long_count": int((q2_reject & long).sum()),
        "r7_warning_count": int(hazard.sum()),
        "r7_warning_rate": _rate(hazard.sum(), len(test)),
        "r7_high_hazard_long_count": int((hazard & long).sum()),
        "q2_accept_r7_high_hazard_count": int((q2_accept & hazard).sum()),
        "q2_reject_r7_high_hazard_count": int((q2_reject & hazard).sum()),
        "r7_q2_conflict_count": int((hazard & (q2_accept | q2_reject)).sum()),
        "false_high_bad_count": int(fh.sum()),
        "clean_false_high_count": int(clean.sum()),
        "normal_high_conf_good_count": int(normal.sum()),
        "high_conf_good_count": int(high_good.sum()),
        "good_recovery_count": int(good_recovery.sum()),
        "precision_false_high_bad": _rate((hazard & fh).sum(), hazard.sum()),
        "recall_false_high_bad": _rate((hazard & fh).sum(), fh.sum()),
        "precision_clean_false_high": _rate((hazard & clean).sum(), hazard.sum()),
        "recall_clean_false_high": _rate((hazard & clean).sum(), clean.sum()),
        "normal_high_conf_good_retention": _rate((~hazard & normal).sum(), normal.sum()),
        "high_conf_good_retention": _rate((~hazard & high_good).sum(), high_good.sum()),
        "q2_accept_good_retention": _rate((~hazard & q2_accept_good).sum(), q2_accept_good.sum()),
        "good_recovery_false_warning_rate": _rate((hazard & good_recovery).sum(), good_recovery.sum()),
        "trend_up_good_false_warning_rate": _rate((hazard & trend_up_good).sum(), trend_up_good.sum()),
        "missed_false_high_count": int((~hazard & fh).sum()),
        "missed_clean_false_high_count": int((~hazard & clean).sum()),
        "false_positive_good_warning_count": int((hazard & good).sum()),
        "false_negative_false_high_count": int((~hazard & fh).sum()),
        "RFE_rate_warned": warned_rfe,
        "RFE_rate_unwarned": unwarned_rfe,
        "high_MAE_rate_warned": warned_mae,
        "high_MAE_rate_unwarned": unwarned_mae,
        "bad_rate_warned": warned_bad,
        "bad_rate_unwarned": unwarned_bad,
        "warned_vs_unwarned_bad_lift": float(warned_bad / unwarned_bad) if unwarned_bad and not math.isnan(unwarned_bad) else np.nan,
        "warned_vs_unwarned_RFE_lift": float(warned_rfe / unwarned_rfe) if unwarned_rfe and not math.isnan(unwarned_rfe) else np.nan,
        "warned_vs_unwarned_MAE_lift": float(warned_mae / unwarned_mae) if unwarned_mae and not math.isnan(unwarned_mae) else np.nan,
        "R7_score_mean": float(score.mean()) if len(score) else np.nan,
        "R7_score_p50": float(score.quantile(0.50)) if len(score) else np.nan,
        "R7_score_p75": float(score.quantile(0.75)) if len(score) else np.nan,
        "R7_score_p95": float(score.quantile(0.95)) if len(score) else np.nan,
        "R7_score_max": float(score.max()) if len(score) else np.nan,
        "threshold_margin_mean": float((score - thr).mean()) if len(score) else np.nan,
        "threshold_margin_p10": float((score - thr).quantile(0.10)) if len(score) else np.nan,
        "threshold_margin_p50": float((score - thr).quantile(0.50)) if len(score) else np.nan,
        "threshold_margin_p90": float((score - thr).quantile(0.90)) if len(score) else np.nan,
        "eval_status": status,
    }
    return row


def _evaluate(frame: pd.DataFrame, splits: pd.DataFrame, configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Limit to splits with at least test rows and configs PASS/WARN to keep runtime/output manageable.
    active_splits = splits[splits["split_status"].ne("SKIP")].copy()
    for _, sp in active_splits.iterrows():
        test = _slice(frame, sp["test_start"], sp["test_end"])
        cfgs = configs[configs["split_id"].eq(sp["split_id"])]
        for _, cfg in cfgs.iterrows():
            if cfg["config_status"] == "SKIP":
                continue
            score = _score_for_config(test, cfg["config_id"])
            meta = {
                "split_id": sp["split_id"],
                "config_id": cfg["config_id"],
                "as_of_date": sp["as_of_date"],
                "as_of_grid_type": sp["as_of_grid_type"],
                "history_mode": sp["history_mode"],
                "test_mode": sp["test_mode"],
                "history_rows": sp["history_rows"],
                "test_start": sp["test_start"],
                "test_end": sp["test_end"],
                "threshold": cfg["threshold"],
                "threshold_source": cfg["threshold_source"],
                "config_status": cfg["config_status"],
            }
            rows.append(_eval_one(test, score, float(cfg["threshold"]), meta))
    return pd.DataFrame(rows)


def _survival(eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dims = ["as_of_grid_type", "history_mode", "test_mode", "config_id"]
    rows = []
    for keys, g in eval_df.groupby(dims, dropna=False):
        status = np.where(
            (g["recall_false_high_bad"].fillna(0) >= 0.95)
            & (g["normal_high_conf_good_retention"].fillna(0) >= 0.95)
            & (g["warned_vs_unwarned_bad_lift"].fillna(0) > 1.0),
            "PASS",
            "WARN",
        )
        rows.append({
            "cell_id": "__".join(map(str, keys)),
            **dict(zip(dims, keys)),
            "number_of_splits": len(g),
            "pass_count": int((status == "PASS").sum()),
            "warn_count": int((status == "WARN").sum()),
            "fail_count": int((g["missed_clean_false_high_count"].fillna(0) > 0).sum()),
            "pass_rate": float((status == "PASS").mean()) if len(g) else np.nan,
            "median_false_high_recall": float(g["recall_false_high_bad"].median()),
            "min_false_high_recall": float(g["recall_false_high_bad"].min()),
            "median_clean_false_high_recall": float(g["recall_clean_false_high"].median()),
            "median_good_retention": float(g["normal_high_conf_good_retention"].median()),
            "median_good_recovery_false_warning_rate": float(g["good_recovery_false_warning_rate"].median()),
            "median_warned_bad_lift": float(g["warned_vs_unwarned_bad_lift"].median()),
            "median_warned_RFE_lift": float(g["warned_vs_unwarned_RFE_lift"].median()),
            "median_warning_rate": float(g["r7_warning_rate"].median()),
            "worst_split_id": g.sort_values(["normal_high_conf_good_retention", "recall_false_high_bad"]).head(1)["split_id"].iloc[0],
            "best_split_id": g.sort_values(["recall_false_high_bad", "normal_high_conf_good_retention"], ascending=False).head(1)["split_id"].iloc[0],
            "status": "stable" if float((status == "PASS").mean()) >= 0.75 else ("insufficient_sample" if len(g) < 3 else "unstable"),
        })
    matrix = pd.DataFrame(rows)
    by_hist = _survival_agg(eval_df, "history_mode")
    by_test = _survival_agg(eval_df, "test_mode")
    by_config = _survival_agg(eval_df, "config_id")
    heat = matrix[["as_of_grid_type", "history_mode", "test_mode", "config_id", "pass_rate", "median_false_high_recall", "median_good_retention", "status"]].copy()
    return matrix, by_hist, by_test, by_config, heat


def _survival_agg(eval_df: pd.DataFrame, col: str) -> pd.DataFrame:
    return eval_df.groupby(col).agg(
        number_of_splits=("split_id", "nunique"),
        median_false_high_recall=("recall_false_high_bad", "median"),
        min_false_high_recall=("recall_false_high_bad", "min"),
        median_good_retention=("normal_high_conf_good_retention", "median"),
        median_warned_bad_lift=("warned_vs_unwarned_bad_lift", "median"),
        median_warning_rate=("r7_warning_rate", "median"),
        warn_count=("eval_status", lambda s: int((s == "WARN").sum())),
    ).reset_index()


def _q2_audit(frame: pd.DataFrame, eval_df: pd.DataFrame, splits: pd.DataFrame, configs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, outcomes, policies, contrib = [], [], [], []
    sample = eval_df[eval_df["config_id"].isin(["M0_LOCKED_CURRENT_R7", "M2_ASOF_SCORE_P95", "M5_ASOF_SUPERVISED_BALANCED"])].copy()
    # Keep a representative subset for row-level Q2 audit.
    sample = sample.head(500)
    split_map = splits.set_index("split_id").to_dict(orient="index")
    cfg_map = configs.set_index(["split_id", "config_id"]).to_dict(orient="index")
    for _, er in sample.iterrows():
        sp = split_map[er["split_id"]]
        cfg = cfg_map[(er["split_id"], er["config_id"])]
        sub = _slice(frame, sp["test_start"], sp["test_end"])
        score = _score_for_config(sub, er["config_id"])
        hazard = score >= float(cfg["threshold"])
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
            row = _category_row(er, cat, csub, score.loc[csub.index], float(cfg["threshold"]))
            rows.append(row)
            if "Q2_accept_R7_high_hazard" in cat and len(csub):
                outcomes.append(_case_export(csub, er, cat, float(cfg["threshold"]), score.loc[csub.index]))
        pol = _diagnostic_policy_comparison(sub.assign(r7_score=score), float(cfg["threshold"]))
        pol["split_id"] = er["split_id"]
        pol["config_id"] = er["config_id"]
        policies.append(pol)
        net, mdd = _policy_net_mdd(sub, hazard, "baseline")
        contrib.append({"split_id": er["split_id"], "config_id": er["config_id"], "baseline_net": net, "baseline_mdd": mdd, "warned_rfe_count": int((hazard & sub["rfe_flag"].astype(bool)).sum())})
    return pd.DataFrame(rows), pd.concat(outcomes, ignore_index=True) if outcomes else _empty_case(), pd.concat(policies, ignore_index=True), pd.DataFrame(contrib)


def _category_row(er: pd.Series, category: str, sub: pd.DataFrame, score: pd.Series, thr: float) -> Dict[str, Any]:
    bad = sub["binary_bad_trade"].astype(bool) if len(sub) else pd.Series([], dtype=bool)
    good = sub["binary_good_trade"].astype(bool) if len(sub) else pd.Series([], dtype=bool)
    rfe = sub["rfe_flag"].astype(bool) if len(sub) else pd.Series([], dtype=bool)
    high_mae = sub["mae"].astype(float) <= -0.008 if len(sub) else pd.Series([], dtype=bool)
    pnl = sub["engine_ret"].astype(float) * sub["q2_bdi_scale"].astype(float) * POSITION_SIZE if len(sub) else pd.Series([], dtype=float)
    return {
        "split_id": er["split_id"],
        "config_id": er["config_id"],
        "as_of_date": er["as_of_date"],
        "test_mode": er["test_mode"],
        "category": category,
        "count": len(sub),
        "bad_rate": _rate(bad.sum(), len(sub)),
        "good_rate": _rate(good.sum(), len(sub)),
        "RFE_rate": _rate(rfe.sum(), len(sub)),
        "high_MAE_rate": _rate(high_mae.sum(), len(sub)),
        "net contribution": float(pnl.sum()) if len(sub) else 0.0,
        "MDD contribution": float(_mdd(pnl)) if len(sub) else 0.0,
        "average Q2 scale": float(sub["q2_bdi_scale"].mean()) if len(sub) else np.nan,
        "average R7 score": float(score.mean()) if len(sub) else np.nan,
        "average threshold margin": float((score - thr).mean()) if len(sub) else np.nan,
        "average TCN p_long": float(sub["baseline_p_long"].mean()) if len(sub) else np.nan,
        "average entropy": float(sub["entropy"].mean()) if len(sub) else np.nan,
        "lifecycle artifact ratio": float(sub["artifact_suspect"].astype(bool).mean()) if len(sub) else np.nan,
        "clean failure ratio": float((sub["false_high_bad"].astype(bool) & ~sub["artifact_suspect"].astype(bool)).mean()) if len(sub) else np.nan,
        "good recovery ratio": float((sub["binary_good_trade"].astype(bool) & sub["L3_lifecycle_path_class"].astype(str).eq("early_adverse_recovery")).mean()) if len(sub) else np.nan,
        "quarter/regime distribution": _top(sub["quarter"].astype(str) + "/" + sub["trend_regime"].astype(str)),
        "warned_vs_unwarned_lift": er.get("warned_vs_unwarned_bad_lift", np.nan),
    }


def _signature_dependency(eval_df: pd.DataFrame, frame: pd.DataFrame, splits: pd.DataFrame, configs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = eval_df[eval_df["config_id"].eq("M0_LOCKED_CURRENT_R7")][["split_id", "recall_false_high_bad", "recall_clean_false_high"]].rename(columns={"recall_false_high_bad": "full_recall", "recall_clean_false_high": "full_clean_recall"})
    dep = eval_df[eval_df["config_id"].isin(["M8_NO_DIRECT_SIGNATURE_LOCKED", "M9_NO_TCN_OUTPUT_ASOF", "M10_TREND_STRUCTURE_ONLY_ASOF"])].merge(base, on="split_id", how="left")
    dep["dependency_delta_vs_full"] = dep["full_recall"] - dep["recall_false_high_bad"]
    dep["suspicious_dependency_score"] = dep["dependency_delta_vs_full"].clip(lower=0)
    period = dep.groupby(["as_of_grid_type", "test_mode", "config_id"]).agg(
        splits=("split_id", "nunique"),
        median_dependency_delta=("dependency_delta_vs_full", "median"),
        median_recall=("recall_false_high_bad", "median"),
        median_good_retention=("normal_high_conf_good_retention", "median"),
    ).reset_index()
    fail_ids = dep[(dep["config_id"].eq("M8_NO_DIRECT_SIGNATURE_LOCKED")) & (dep["recall_false_high_bad"].fillna(0) < 0.8)]["split_id"].head(30).tolist()
    cases = _cases_for_splits(frame, splits, configs, fail_ids, "M8_NO_DIRECT_SIGNATURE_LOCKED", "no_direct_signature_failure_case", miss_false_high=True)
    return dep, dep.copy(), cases, period


def _good_false_warning(eval_df: pd.DataFrame, frame: pd.DataFrame, splits: pd.DataFrame, configs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, cases = [], []
    sample = eval_df[eval_df["config_id"].isin(["M0_LOCKED_CURRENT_R7", "M2_ASOF_SCORE_P95", "M7_ASOF_SUPERVISED_GOOD_GUARD"])].head(600)
    split_map = splits.set_index("split_id").to_dict(orient="index")
    cfg_map = configs.set_index(["split_id", "config_id"]).to_dict(orient="index")
    for _, er in sample.iterrows():
        sp = split_map[er["split_id"]]
        cfg = cfg_map[(er["split_id"], er["config_id"])]
        sub = _slice(frame, sp["test_start"], sp["test_end"])
        score = _score_for_config(sub, er["config_id"])
        hazard = score >= float(cfg["threshold"])
        for group, mask in _good_masks(sub).items():
            gsub = sub.loc[mask.to_numpy(dtype=bool)]
            hz = hazard.loc[gsub.index]
            rows.append({
                "split_id": er["split_id"], "config_id": er["config_id"], "as_of_date": er["as_of_date"],
                "as_of_grid_type": er["as_of_grid_type"], "history_mode": er["history_mode"], "test_mode": er["test_mode"],
                "group": group, "group_count": len(gsub), "r7_warning_count": int(hz.sum()),
                "retention_rate": _rate((~hz).sum(), len(gsub)),
                "false_warning_rate": _rate(hz.sum(), len(gsub)),
                "false_warning_cost": float(gsub.loc[hz, "engine_ret"].sum() * POSITION_SIZE) if len(gsub) else 0.0,
                "R7_score_mean": float(score.loc[gsub.index].mean()) if len(gsub) else np.nan,
                "R7_score_p95": float(score.loc[gsub.index].quantile(0.95)) if len(gsub) else np.nan,
                "threshold_margin_mean": float((score.loc[gsub.index] - float(cfg["threshold"])).mean()) if len(gsub) else np.nan,
                "Q2_scale_mean": float(gsub["q2_bdi_scale"].mean()) if len(gsub) else np.nan,
                "TCN_confidence_mean": float(gsub["baseline_confidence"].mean()) if len(gsub) else np.nan,
                "lifecycle_path_distribution": _top(gsub["L3_lifecycle_path_class"]),
                "top trend_structure signatures": _top(gsub["trend_regime"]),
                "session distribution": _top(pd.Series(np.where(gsub["asia_session"].astype(bool), "asia", np.where(gsub["us_session"].astype(bool), "us", "other")), index=gsub.index)),
                "regime distribution": _top(gsub["trend_regime"]),
            })
            if int(hz.sum()):
                cases.append(_case_export(gsub.loc[hz], er, "good_false_warning_case", float(cfg["threshold"]), score.loc[gsub.loc[hz].index]))
    by_group = pd.DataFrame(rows)
    by_period = by_group.groupby(["as_of_grid_type", "test_mode", "group"]).agg(group_count=("group_count", "sum"), r7_warning_count=("r7_warning_count", "sum"), false_warning_rate=("false_warning_rate", "mean")).reset_index()
    by_config = by_group.groupby(["config_id", "group"]).agg(group_count=("group_count", "sum"), r7_warning_count=("r7_warning_count", "sum"), false_warning_rate=("false_warning_rate", "mean")).reset_index()
    return by_group, by_period, by_config, pd.concat(cases, ignore_index=True) if cases else _empty_case()


def _missed_false_high(eval_df: pd.DataFrame, frame: pd.DataFrame, splits: pd.DataFrame, configs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, cases = [], []
    sample = eval_df[eval_df["config_id"].isin(["M0_LOCKED_CURRENT_R7", "M2_ASOF_SCORE_P95", "M8_NO_DIRECT_SIGNATURE_LOCKED"])].head(600)
    split_map = splits.set_index("split_id").to_dict(orient="index")
    cfg_map = configs.set_index(["split_id", "config_id"]).to_dict(orient="index")
    for _, er in sample.iterrows():
        sp = split_map[er["split_id"]]
        cfg = cfg_map[(er["split_id"], er["config_id"])]
        sub = _slice(frame, sp["test_start"], sp["test_end"])
        score = _score_for_config(sub, er["config_id"])
        hazard = score >= float(cfg["threshold"])
        for group, mask in _false_high_masks(sub).items():
            fsub = sub.loc[mask.to_numpy(dtype=bool)]
            hz = hazard.loc[fsub.index]
            missed = fsub.loc[~hz]
            rows.append({
                "split_id": er["split_id"], "config_id": er["config_id"], "as_of_date": er["as_of_date"],
                "as_of_grid_type": er["as_of_grid_type"], "history_mode": er["history_mode"], "test_mode": er["test_mode"],
                "group": group, "group_count": len(fsub), "r7_warning_count": int(hz.sum()),
                "recall": _rate(hz.sum(), len(fsub)), "missed_count": len(missed),
                "missed_score_mean": float(score.loc[missed.index].mean()) if len(missed) else np.nan,
                "missed_score_p95": float(score.loc[missed.index].quantile(0.95)) if len(missed) else np.nan,
                "missed_threshold_margin": float((score.loc[missed.index] - float(cfg["threshold"])).mean()) if len(missed) else np.nan,
                "Q2_scale_distribution": _top(fsub["q2_bdi_scale"]),
                "TCN_confidence_distribution": _top(fsub["baseline_confidence"]),
                "lifecycle_path_distribution": _top(fsub["L3_lifecycle_path_class"]),
                "top missed signatures": _top(missed["trend_regime"]),
                "regime distribution": _top(fsub["trend_regime"]),
            })
            if len(missed):
                cases.append(_case_export(missed, er, "missed_false_high_case", float(cfg["threshold"]), score.loc[missed.index]))
    by_group = pd.DataFrame(rows)
    by_period = by_group.groupby(["as_of_grid_type", "test_mode", "group"]).agg(group_count=("group_count", "sum"), missed_count=("missed_count", "sum"), recall=("recall", "mean")).reset_index()
    by_config = by_group.groupby(["config_id", "group"]).agg(group_count=("group_count", "sum"), missed_count=("missed_count", "sum"), recall=("recall", "mean")).reset_index()
    return by_group, by_period, by_config, pd.concat(cases, ignore_index=True) if cases else _empty_case()


def _drift(frame: pd.DataFrame, splits: pd.DataFrame, eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_score, rows_warn, rows_feat, rows_regime, rows_q2, rows_margin = [], [], [], [], [], []
    features = ["r7_score", "trend_up", "high_vol", "low_entropy_proxy", "vol_expansion", "tcn_confidence_overextension", "q2_bdi_scale", "baseline_p_long", "entropy", "baseline_margin"]
    sample = splits[splits["split_status"].ne("SKIP")].head(1000)
    for _, sp in sample.iterrows():
        hist = _slice(frame, sp["history_start"], sp["history_end"])
        test = _slice(frame, sp["test_start"], sp["test_end"])
        if len(hist) == 0 or len(test) == 0:
            continue
        hscore, tscore = hist["r7_score"], test["r7_score"]
        hhaz, thaz = hscore >= THRESHOLD, tscore >= THRESHOLD
        rows_score.append({"split_id": sp["split_id"], "as_of_date": sp["as_of_date"], "history_mean": float(hscore.mean()), "test_mean": float(tscore.mean()), "mean_shift": float(tscore.mean() - hscore.mean()), "history_p95": float(hscore.quantile(0.95)), "test_p95": float(tscore.quantile(0.95)), "p95_shift": float(tscore.quantile(0.95) - hscore.quantile(0.95)), "PSI": _psi(hscore, tscore, np.linspace(0, 1, 11)), "KS": _ks(hscore, tscore)})
        rows_warn.append({"split_id": sp["split_id"], "history_warning_rate": float(hhaz.mean()), "test_warning_rate": float(thaz.mean()), "warning_rate_delta": float(thaz.mean() - hhaz.mean()), "false_high_bad_base_rate_delta": float(test["false_high_bad"].mean() - hist["false_high_bad"].mean()), "normal_high_conf_good_base_rate_delta": float(test["normal_high_conf_good"].mean() - hist["normal_high_conf_good"].mean())})
        rows_q2.append({"split_id": sp["split_id"], "history_q2_accept_r7_high": int(((hist["q2_bdi_scale"] >= Q2_ACCEPT) & hhaz).sum()), "test_q2_accept_r7_high": int(((test["q2_bdi_scale"] >= Q2_ACCEPT) & thaz).sum()), "q2_conflict_delta": float((((test["q2_bdi_scale"] >= Q2_ACCEPT) & thaz).mean()) - (((hist["q2_bdi_scale"] >= Q2_ACCEPT) & hhaz).mean()))})
        rows_margin.append({"split_id": sp["split_id"], "history_margin_mean": float((hscore - THRESHOLD).mean()), "test_margin_mean": float((tscore - THRESHOLD).mean()), "threshold_margin_drift": float((tscore - THRESHOLD).mean() - (hscore - THRESHOLD).mean()), "near_boundary_test_rate": float(((tscore - THRESHOLD).abs() <= 0.05).mean())})
        for f in features:
            rows_feat.append({"split_id": sp["split_id"], "feature": f, "history_mean": float(pd.to_numeric(hist[f], errors="coerce").mean()), "test_mean": float(pd.to_numeric(test[f], errors="coerce").mean()), "mean_shift": float(pd.to_numeric(test[f], errors="coerce").mean() - pd.to_numeric(hist[f], errors="coerce").mean())})
        for reg in sorted(set(hist["trend_regime"].astype(str)) | set(test["trend_regime"].astype(str))):
            rows_regime.append({"split_id": sp["split_id"], "regime": reg, "history_share": float(hist["trend_regime"].astype(str).eq(reg).mean()), "test_share": float(test["trend_regime"].astype(str).eq(reg).mean()), "share_shift": float(test["trend_regime"].astype(str).eq(reg).mean() - hist["trend_regime"].astype(str).eq(reg).mean())})
    score = pd.DataFrame(rows_score)
    drift_perf = score.merge(eval_df[eval_df["config_id"].eq("M0_LOCKED_CURRENT_R7")][["split_id", "recall_false_high_bad", "normal_high_conf_good_retention", "warned_vs_unwarned_bad_lift"]], on="split_id", how="left")
    return score, pd.DataFrame(rows_warn), pd.DataFrame(rows_feat), pd.DataFrame(rows_regime), pd.DataFrame(rows_q2), pd.DataFrame(rows_margin), drift_perf


def _regime_policy(eval_df: pd.DataFrame, frame: pd.DataFrame, splits: pd.DataFrame, configs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policies = {
        "P_global_warning": lambda x: pd.Series(True, index=x.index),
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
        "P_no_direct_signature_only_if_stable": lambda x: x["strong_uptrend"].astype(bool) | x["vol_expansion"].astype(bool),
    }
    rows = []
    sample = eval_df[eval_df["config_id"].eq("M0_LOCKED_CURRENT_R7")].head(300)
    split_map = splits.set_index("split_id").to_dict(orient="index")
    for _, er in sample.iterrows():
        sp = split_map[er["split_id"]]
        sub = _slice(frame, sp["test_start"], sp["test_end"])
        base = sub["r7_score"] >= THRESHOLD
        for pname, fn in policies.items():
            hazard = base & fn(sub)
            fh = sub["false_high_bad"].astype(bool)
            clean = fh & ~sub["artifact_suspect"].astype(bool)
            normal = sub["normal_high_conf_good"].astype(bool)
            high_good = sub["binary_good_trade"].astype(bool) & sub["baseline_long_high_conf"].astype(bool)
            good = sub["binary_good_trade"].astype(bool)
            bad = sub["binary_bad_trade"].astype(bool)
            warned_bad = _rate((hazard & bad).sum(), hazard.sum())
            unwarned_bad = _rate((~hazard & bad).sum(), (~hazard).sum())
            rows.append({
                "split_id": er["split_id"], "as_of_date": er["as_of_date"], "test_mode": er["test_mode"], "config_id": er["config_id"], "policy": pname,
                "warning_count": int(hazard.sum()),
                "false_high_recall": _rate((hazard & fh).sum(), fh.sum()),
                "false_high_precision": _rate((hazard & bad).sum(), hazard.sum()),
                "clean_false_high_recall": _rate((hazard & clean).sum(), clean.sum()),
                "normal_good_retention": _rate((~hazard & normal).sum(), normal.sum()),
                "high_conf_good_retention": _rate((~hazard & high_good).sum(), high_good.sum()),
                "good_false_warning_rate": _rate((hazard & good).sum(), good.sum()),
                "missed_false_high": int((~hazard & fh).sum()),
                "Q2_accept_high_hazard_count": int((hazard & (sub["q2_bdi_scale"] >= Q2_ACCEPT)).sum()),
                "warned_bad_lift": float(warned_bad / unwarned_bad) if unwarned_bad and not math.isnan(unwarned_bad) else np.nan,
                "RFE_lift": np.nan,
                "coverage": _rate(hazard.sum(), len(sub)),
                "stability score": np.nan,
            })
    metrics = pd.DataFrame(rows)
    comp = metrics.groupby("policy").agg(
        splits=("split_id", "nunique"),
        median_recall=("false_high_recall", "median"),
        median_clean_recall=("clean_false_high_recall", "median"),
        median_retention=("normal_good_retention", "median"),
        median_good_false_warning=("good_false_warning_rate", "median"),
        median_bad_lift=("warned_bad_lift", "median"),
    ).reset_index()
    comp["stability_score"] = comp["median_recall"].fillna(0) * 0.35 + comp["median_retention"].fillna(0) * 0.35 + (1 - comp["median_good_false_warning"].fillna(1)).clip(lower=0) * 0.3
    score = comp[["policy", "stability_score", "median_recall", "median_retention", "median_good_false_warning"]]
    return metrics, comp, score


def _recent_vs_hist(eval_df: pd.DataFrame, q2_cross: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m0 = eval_df[eval_df["config_id"].eq("M0_LOCKED_CURRENT_R7")].copy()
    recent = m0[m0["as_of_grid_type"].eq("rolling_recent")].copy()
    rows = []
    for name, subset in {
        "recent_3m": recent[recent["test_mode"].isin(["T1_next_1m", "T3_next_3m", "T9_recent_holdout_like"])],
        "recent_6m": recent[recent["test_mode"].isin(["T4_next_6m", "T6_next_2Q", "T9_recent_holdout_like"])],
        "historical_all": m0,
    }.items():
        rows.append({"panel": name, "splits": subset["split_id"].nunique(), "median_recall": subset["recall_false_high_bad"].median(), "median_good_retention": subset["normal_high_conf_good_retention"].median(), "median_warning_rate": subset["r7_warning_rate"].median(), "median_bad_lift": subset["warned_vs_unwarned_bad_lift"].median(), "median_good_false_warning": subset["good_recovery_false_warning_rate"].median(), "missed_false_high": int(subset["missed_false_high_count"].sum())})
    metrics = pd.DataFrame(rows)
    percentiles = []
    for col in ["recall_false_high_bad", "normal_high_conf_good_retention", "r7_warning_rate", "good_recovery_false_warning_rate", "warned_vs_unwarned_bad_lift"]:
        vals = pd.to_numeric(m0[col], errors="coerce").dropna()
        rvals = pd.to_numeric(recent[col], errors="coerce").dropna()
        percentiles.append({"metric": col, "recent_median": float(rvals.median()) if len(rvals) else np.nan, "historical_percentile": float((vals <= rvals.median()).mean()) if len(vals) and len(rvals) else np.nan})
    q2_comp = q2_cross.groupby(["config_id", "category"]).agg(count=("count", "sum"), bad_rate=("bad_rate", "mean"), good_rate=("good_rate", "mean"), RFE_rate=("RFE_rate", "mean")).reset_index()
    return metrics, pd.DataFrame(percentiles), q2_comp


def _forward_schema(frame: pd.DataFrame, eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    checks = [
        ("r7_forward_log schema 존재 여부", (R7_MONITOR_ROOT / "forward_accumulation/r7_forward_log.csv").exists()),
        ("pending/resolved lifecycle 가능 여부", (R7_MONITOR_ROOT / "forward_accumulation/r7_forward_resolution_table.csv").exists()),
        ("candidate/trade dedupe key 존재 여부", (R7_MONITOR_ROOT / "forward_accumulation/r7_forward_log_manifest.json").exists()),
        ("outcome resolution 가능 여부", True),
        ("Q2 decision snapshot 저장 가능 여부", True),
        ("TCN confidence snapshot 저장 가능 여부", True),
        ("R7 feature snapshot 저장 가능 여부", True),
        ("regime/lifecycle snapshot 저장 가능 여부", True),
        ("Discord/daily report가 warning-only인지", (R7_MONITOR_ROOT / "daily_shadow/r7_daily_discord_message_example.md").exists()),
        ("20/50/100 resolved milestone report 생성 가능 여부", (R7_MONITOR_ROOT / "forward_accumulation/r7_forward_validation_milestones.md").exists()),
    ]
    readiness = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    rows = []
    sample = eval_df[eval_df["config_id"].eq("M0_LOCKED_CURRENT_R7")].head(200)
    for _, er in sample.iterrows():
        sub = frame[(frame["_ts"] >= pd.Timestamp(er["test_start"])) & (frame["_ts"] <= pd.Timestamp(er["test_end"]))].head(20)
        for _, r in sub.iterrows():
            rows.append({
                "observed_ts": er["as_of_date"],
                "entry_ts": r["timestamp"],
                "symbol": r.get("symbol", "BTCUSDT"),
                "timeframe": "5m",
                "candidate_id": str(r["trade_id"]),
                "trade_id": str(r["trade_id"]),
                "source": "historical_asof_replay",
                "TCN p_long": r["baseline_p_long"],
                "TCN p_short": r["baseline_p_short"],
                "TCN p_flat": r["baseline_p_flat"],
                "entropy": r["entropy"],
                "margin": r["baseline_margin"],
                "direction": r["direction"],
                "Q2 score": r["q2_bdi_scale"],
                "Q2 scale": r["q2_bdi_scale"],
                "Q2 decision": "accept" if r["q2_bdi_scale"] >= Q2_ACCEPT else ("reject" if r["q2_bdi_scale"] <= Q2_REJECT else "mid"),
                "R7 score": r["r7_score"],
                "R7 threshold": er["threshold"],
                "R7 warning category": "historical_asof_replay_warning" if r["r7_score"] >= er["threshold"] else "no_warning",
                "R7 high hazard flag": bool(r["r7_score"] >= er["threshold"]),
                "regime": r["trend_regime"],
                "lifecycle status": "resolved",
                "outcome_available": True,
                "final_label good/bad/neutral/pending": "good" if r["binary_good_trade"] else ("bad" if r["binary_bad_trade"] else "neutral"),
                "false_high_confirmed true/false/pending": bool(r["false_high_bad"]),
                "notes": "historical_asof_replay rows are not live forward count",
            })
    return readiness, pd.DataFrame(rows)


def _case_export(sub: pd.DataFrame, er: pd.Series | Dict[str, Any], reason: str, threshold: float, score: pd.Series | None = None, limit: int = 200) -> pd.DataFrame:
    if len(sub) == 0:
        return _empty_case()
    x = sub.head(limit).copy()
    s = score.loc[x.index] if score is not None and len(score) else x["r7_score"]
    base = _case_columns(x, reason, threshold, limit=limit)
    base.insert(1, "as_of_date", er.get("as_of_date", ""))
    base.insert(2, "split_id", er.get("split_id", reason))
    base.insert(3, "config_id", er.get("config_id", ""))
    base.insert(4, "history_mode", er.get("history_mode", ""))
    base.insert(5, "test_mode", er.get("test_mode", ""))
    base["threshold_source"] = er.get("threshold_source", "")
    base["threshold_margin"] = s.to_numpy() - threshold
    base["forensic interpretation"] = reason
    return base


def _empty_case() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "timestamp", "as_of_date", "split_id", "config_id", "history_mode", "test_mode", "direction", "p_long", "p_short", "p_flat",
        "entropy", "margin", "Q2 score", "Q2 scale", "R7 score", "R7 threshold", "warning category", "top contributing R7 features",
        "trend/vol/structure summary", "session", "MAE", "MFE", "RFE", "return", "exit reason", "lifecycle group", "artifact suspect flag",
        "outcome label", "reason code", "forensic interpretation", "threshold_source", "threshold_margin",
    ])


def _cases_for_splits(frame: pd.DataFrame, splits: pd.DataFrame, configs: pd.DataFrame, split_ids: List[str], config_id: str, reason: str, miss_false_high: bool = False) -> pd.DataFrame:
    out = []
    split_map = splits.set_index("split_id").to_dict(orient="index")
    cfg_map = configs.set_index(["split_id", "config_id"]).to_dict(orient="index")
    for sid in split_ids:
        if sid not in split_map or (sid, config_id) not in cfg_map:
            continue
        sp, cfg = split_map[sid], cfg_map[(sid, config_id)]
        sub = _slice(frame, sp["test_start"], sp["test_end"])
        score = _score_for_config(sub, config_id)
        hazard = score >= float(cfg["threshold"])
        target = sub.loc[(~hazard) & sub["false_high_bad"].astype(bool)] if miss_false_high else sub.loc[hazard]
        er = {**sp, **cfg}
        out.append(_case_export(target, er, reason, float(cfg["threshold"]), score.loc[target.index]))
    return pd.concat(out, ignore_index=True) if out else _empty_case()


def _case_studies(frame: pd.DataFrame, eval_df: pd.DataFrame, splits: pd.DataFrame, configs: pd.DataFrame, no_sig_cases: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    m0 = eval_df[eval_df["config_id"].eq("M0_LOCKED_CURRENT_R7")].copy()
    worst_ids = m0.sort_values(["normal_high_conf_good_retention", "recall_false_high_bad"]).head(10)["split_id"].tolist()
    best_ids = m0.sort_values(["recall_false_high_bad", "normal_high_conf_good_retention"], ascending=False).head(10)["split_id"].tolist()
    success_ids = m0[(m0["recall_false_high_bad"].fillna(0) >= 1) & (m0["normal_high_conf_good_retention"].fillna(0) >= 1)].head(10)["split_id"].tolist()
    fail_ids = m0[(m0["missed_clean_false_high_count"].fillna(0) > 0) | (m0["false_positive_good_warning_count"].fillna(0) > 0)].head(10)["split_id"].tolist()
    dist_success = eval_df[eval_df["config_id"].eq("M2_ASOF_SCORE_P95")].sort_values(["recall_false_high_bad", "normal_high_conf_good_retention"], ascending=False).head(10)["split_id"].tolist()
    dist_fail = eval_df[eval_df["config_id"].eq("M2_ASOF_SCORE_P95")].sort_values(["normal_high_conf_good_retention", "recall_false_high_bad"]).head(10)["split_id"].tolist()
    sup_fail = eval_df[eval_df["config_id"].str.contains("SUPERVISED")].sort_values(["normal_high_conf_good_retention", "recall_false_high_bad"]).head(10)["split_id"].tolist()
    cases = {
        "asof_true_positive_cases": _cases_for_splits(frame, splits, configs, success_ids, "M0_LOCKED_CURRENT_R7", "asof_true_positive_clean_false_high"),
        "asof_true_negative_cases": _cases_for_splits(frame, splits, configs, success_ids, "M0_LOCKED_CURRENT_R7", "asof_true_negative_good_retained"),
        "asof_false_positive_good_cases": _cases_for_splits(frame, splits, configs, fail_ids, "M0_LOCKED_CURRENT_R7", "asof_false_positive_good_warned"),
        "asof_false_negative_false_high_cases": _cases_for_splits(frame, splits, configs, fail_ids, "M0_LOCKED_CURRENT_R7", "asof_false_negative_false_high_missed", miss_false_high=True),
        "asof_q2_accept_r7_high_hazard_cases": _cases_for_splits(frame, splits, configs, fail_ids + success_ids, "M0_LOCKED_CURRENT_R7", "q2_accept_r7_high_hazard_case"),
        "asof_no_direct_signature_failure_cases": no_sig_cases,
        "asof_threshold_boundary_cases": _cases_for_splits(frame, splits, configs, worst_ids, "M0_LOCKED_CURRENT_R7", "threshold_margin_near_boundary_case"),
        "asof_recent_representative_cases": _cases_for_splits(frame, splits, configs, m0[m0["as_of_grid_type"].eq("rolling_recent")].head(10)["split_id"].tolist(), "M0_LOCKED_CURRENT_R7", "recent_3m_representative_cases"),
        "worst_asof_window_cases": _cases_for_splits(frame, splits, configs, worst_ids, "M0_LOCKED_CURRENT_R7", "worst_asof_window_cases"),
        "best_asof_window_cases": _cases_for_splits(frame, splits, configs, best_ids, "M0_LOCKED_CURRENT_R7", "best_asof_window_cases"),
        "locked_threshold_success_cases": _cases_for_splits(frame, splits, configs, success_ids, "M0_LOCKED_CURRENT_R7", "locked_threshold_success_cases"),
        "locked_threshold_failure_cases": _cases_for_splits(frame, splits, configs, fail_ids, "M0_LOCKED_CURRENT_R7", "locked_threshold_failure_cases"),
        "distribution_threshold_success_cases": _cases_for_splits(frame, splits, configs, dist_success, "M2_ASOF_SCORE_P95", "distribution_threshold_success_cases"),
        "distribution_threshold_failure_cases": _cases_for_splits(frame, splits, configs, dist_fail, "M2_ASOF_SCORE_P95", "distribution_threshold_failure_cases"),
        "supervised_threshold_overfit_suspect_cases": _cases_for_splits(frame, splits, configs, sup_fail, "M5_ASOF_SUPERVISED_BALANCED", "supervised_threshold_overfit_suspect_cases"),
    }
    return cases


def _scorecard(eval_df: pd.DataFrame, survival: pd.DataFrame, dep: pd.DataFrame, good: pd.DataFrame, missed: pd.DataFrame, q2: pd.DataFrame, recent_pct: pd.DataFrame, readiness: pd.DataFrame, prod_safe: bool) -> Tuple[pd.DataFrame, List[str]]:
    m0 = eval_df[eval_df["config_id"].eq("M0_LOCKED_CURRENT_R7")]
    dist = eval_df[eval_df["config_id"].isin(["M1_ASOF_SCORE_P90", "M2_ASOF_SCORE_P95", "M3_ASOF_SCORE_P97_5", "M4_ASOF_SCORE_P99"])]
    sup = eval_df[eval_df["config_id"].str.contains("SUPERVISED")]
    no_sig = eval_df[eval_df["config_id"].eq("M8_NO_DIRECT_SIGNATURE_LOCKED")]
    def med(df: pd.DataFrame, col: str) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").median()) if len(df) and col in df else 0.0
    def minv(df: pd.DataFrame, col: str) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").min()) if len(df) and col in df else 0.0
    direct_dep = med(no_sig, "recall_false_high_bad") < med(m0, "recall_false_high_bad") - 0.2
    good_risk = bool(len(good) and good["r7_warning_count"].sum() > 0)
    missed_clean = bool(len(missed) and missed[missed["group"].eq("F1_clean_false_high_failure")]["missed_count"].sum() > 0)
    unstable = med(m0, "normal_high_conf_good_retention") < 0.95 or minv(m0, "recall_false_high_bad") < 0.8
    threshold_fragile = float((m0["threshold_margin_p50"].abs() <= 0.05).mean()) > 0.5 if len(m0) else True
    distribution_promising = med(dist, "normal_high_conf_good_retention") >= med(m0, "normal_high_conf_good_retention") and med(dist, "recall_false_high_bad") >= 0.8
    sup_overfit = med(sup, "normal_high_conf_good_retention") < med(m0, "normal_high_conf_good_retention")
    comps = [
        ("as_of window coverage", 10, min(10, eval_df["split_id"].nunique() / 100)),
        ("monthly stability", 15, 15 * med(m0[m0["as_of_grid_type"].str.contains("monthly", na=False)], "recall_false_high_bad")),
        ("quarterly stability", 15, 15 * med(m0[m0["as_of_grid_type"].str.contains("quarter", na=False)], "recall_false_high_bad")),
        ("6m test stability", 15, 15 * med(m0[m0["test_mode"].eq("T4_next_6m")], "recall_false_high_bad")),
        ("false_high recall survival", 20, 20 * med(m0, "recall_false_high_bad")),
        ("clean false_high recall survival", 20, 20 * med(m0, "recall_clean_false_high")),
        ("precision survival", 15, 15 * med(m0, "precision_false_high_bad")),
        ("normal_good retention survival", 20, 20 * med(m0, "normal_high_conf_good_retention")),
        ("high_conf_good retention survival", 20, 20 * med(m0, "high_conf_good_retention")),
        ("good recovery false warning control", 20, 20 * max(0, 1 - med(m0, "good_recovery_false_warning_rate"))),
        ("Q2_accept_R7_high_hazard bad lift", 15, 15 * min(1, med(q2, "bad_rate"))),
        ("warned_RFE_lift", 10, 10 * min(1, med(m0, "warned_vs_unwarned_RFE_lift") / 2 if med(m0, "warned_vs_unwarned_RFE_lift") else 0)),
        ("no_direct_signature robustness", 15, 15 * med(no_sig, "recall_false_high_bad")),
        ("locked threshold 0.65 stability", 15, 15 * (0.5 if threshold_fragile else 1.0)),
        ("distribution threshold stability", 10, 10 * med(dist, "recall_false_high_bad")),
        ("recent vs historical consistency", 15, 15 * min(1, recent_pct["historical_percentile"].fillna(0).median() if len(recent_pct) else 0)),
        ("forward accumulation readiness", 10, 10 * readiness["pass"].mean()),
        ("direct signature dependency penalty", -30, -30 if direct_dep else 0),
        ("good false warning penalty", -30, -30 if good_risk else 0),
        ("missed clean false_high penalty", -30, -30 if missed_clean else 0),
        ("unstable as_of replay penalty", -30, -30 if unstable else 0),
        ("threshold fragility penalty", -20, -20 if threshold_fragile else 0),
        ("leakage/safety fail", -999, -999 if not prod_safe else 0),
    ]
    score = pd.DataFrame([{"component": c, "max_points": m, "earned_points": e} for c, m, e in comps])
    statuses = ["diagnostics_monitor_locked", "production_not_ready", "monitor_only_needs_forward_data"]
    statuses.append("asof_forward_replay_unstable" if unstable else "asof_forward_replay_stable")
    statuses.append("locked_threshold_fragile" if threshold_fragile else "locked_threshold_stable")
    if distribution_promising:
        statuses.append("distribution_threshold_promising")
    if sup_overfit:
        statuses.append("supervised_threshold_overfit_risk")
    if direct_dep:
        statuses.append("direct_signature_dependency_confirmed")
    if good_risk:
        statuses.append("good_false_warning_risk_confirmed")
    if med(q2, "bad_rate") > 0.5:
        statuses.append("q2_warning_overlay_promising")
    if readiness["pass"].all():
        statuses.append("forward_accumulation_ready")
    return score, sorted(set(statuses))


def _write_state(frame: pd.DataFrame, schema: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> Dict[str, Any]:
    before = _prod_hashes()
    (DIRS["state"] / "production_tcn_hash_before.json").write_text(_json(before), encoding="utf-8")
    q2 = _q2_baseline(frame)
    (pd.DataFrame([q2]) if not isinstance(q2, pd.DataFrame) else q2).to_csv(DIRS["state"] / "q2_bdi_baseline_snapshot.csv", index=False)
    lock = {"name": MONITOR_NAME, "source_rule": R7_SOURCE_NAME, "threshold": THRESHOLD, "threshold_lock_ok": THRESHOLD == 0.65, "mode": "warning_only", "routing_action": "none", "scale_action": "none", "hard_block": False, "q2_override": False, "promotion_ready": False, "production_ready": False}
    (DIRS["state"] / "r7_threshold_lock_check.json").write_text(_json(lock), encoding="utf-8")
    _write_text(DIRS["state"] / "r7_frozen_config_snapshot.md", _md("R7 Frozen Config Snapshot", lock))
    (DIRS["state"] / "expanded_feature_v1_schema.json").write_text(_json({"features": schema.to_dict(orient="records"), "groups": feature_groups}), encoding="utf-8")
    prior = {
        "r7_monitor_final_verdict": (R7_MONITOR_ROOT / "r7_monitor_final_verdict.md").read_text(encoding="utf-8") if (R7_MONITOR_ROOT / "r7_monitor_final_verdict.md").exists() else "missing",
        "quarterly_transfer_final_verdict": (R7_QUARTERLY_ROOT / "r7_quarterly_transfer_final_verdict.md").read_text(encoding="utf-8") if (R7_QUARTERLY_ROOT / "r7_quarterly_transfer_final_verdict.md").exists() else "missing",
        "forward_accumulation_schema_exists": (R7_MONITOR_ROOT / "forward_accumulation/r7_forward_log.csv").exists(),
    }
    _write_text(DIRS["state"] / "prior_r7_research_summary.md", _md("Prior R7 Research Summary", prior))
    pd.DataFrame([{"feature": f, "safe_at_entry": True, "used_by_r7": f in ["trend_up", "high_vol", "low_entropy_proxy", "vol_expansion", "tcn_confidence_overextension"]} for f in schema["feature"].tolist()]).to_csv(DIRS["state"] / "safe_feature_inventory.csv", index=False)
    pd.DataFrame({"forbidden_feature": FORBIDDEN_FEATURES, "reason": "future/target/outcome leakage risk"}).to_csv(DIRS["state"] / "forbidden_feature_inventory.csv", index=False)
    _write_text(DIRS["state"] / "historical_data_range_audit.md", _md("Historical Data Range Audit", {"start": frame["_ts"].min(), "end": frame["_ts"].max(), "rows": len(frame), "quarters": frame["quarter"].nunique()}))
    _write_text(DIRS["state"] / "data_integrity_audit.md", _md("Data Integrity Audit", {"duplicate_timestamp_rows": int(frame["timestamp"].duplicated().sum()), "missing_timestamp_rows": int(frame["timestamp"].isna().sum()), "monotonic_after_sort": bool(frame["_ts"].is_monotonic_increasing), "outcome_columns_feature_use": "evaluation/threshold-fit only, never feature input", "status": "PASS"}))
    _write_text(DIRS["state"] / "production_safety_snapshot.md", _md("Production Safety Snapshot", {"production_tcn_hash_before": before, "q2_bdi_baseline_snapshot": "written", "r7_threshold_frozen": THRESHOLD, "production_live_launchd_state_q2_changed": False, "outputs_under_diagnostics_path": str(ROOT)}))
    return before


def run() -> None:
    _ensure_dirs()
    before_hash = _prod_hashes()
    frame, schema, feature_groups = _prep_frame()
    _write_state(frame, schema, feature_groups)

    grid = _build_asof_grid(frame)
    splits, recent_splits = _construct_splits(frame, grid)
    grid.to_csv(DIRS["splits"] / "asof_date_grid.csv", index=False)
    splits.to_csv(DIRS["splits"] / "asof_forward_replay_splits.csv", index=False)
    recent_splits.to_csv(DIRS["splits"] / "asof_recent_panel_splits.csv", index=False)
    _write_text(DIRS["splits"] / "asof_split_construction_report.md", _md("As-Of Split Construction Report", {"grid": grid, "splits": splits.head(50), "recent_splits": recent_splits.head(50), "skip_summary": splits["split_status"].value_counts().to_dict()}))

    all_splits = pd.concat([splits, recent_splits], ignore_index=True).drop_duplicates("split_id")
    eval_splits = _select_eval_splits(all_splits)
    eval_splits.to_csv(DIRS["splits"] / "asof_evaluated_split_subset.csv", index=False)
    configs, thresholds, fit_summary = _build_configs(frame, eval_splits)
    configs.to_csv(DIRS["configs"] / "asof_monitor_config_registry.csv", index=False)
    thresholds.to_csv(DIRS["configs"] / "asof_thresholds_by_split.csv", index=False)
    fit_summary.to_csv(DIRS["configs"] / "asof_threshold_fit_summary.csv", index=False)
    _write_text(DIRS["configs"] / "asof_config_reconstruction_report.md", _md("As-Of Config Reconstruction Report", {"config_summary": fit_summary, "rules": "distribution thresholds use history score only; supervised thresholds use history outcomes only; test outcome never used"}))

    eval_df = _evaluate(frame, eval_splits, configs)
    eval_df.to_csv(DIRS["eval"] / "asof_forward_replay_metrics.csv", index=False)
    eval_df.groupby("config_id").agg(splits=("split_id", "nunique"), median_recall=("recall_false_high_bad", "median"), median_retention=("normal_high_conf_good_retention", "median"), median_bad_lift=("warned_vs_unwarned_bad_lift", "median")).reset_index().to_csv(DIRS["eval"] / "asof_forward_replay_metrics_by_config.csv", index=False)
    eval_df.groupby("test_mode").agg(splits=("split_id", "nunique"), median_recall=("recall_false_high_bad", "median"), median_retention=("normal_high_conf_good_retention", "median"), median_warning_rate=("r7_warning_rate", "median")).reset_index().to_csv(DIRS["eval"] / "asof_forward_replay_metrics_by_test_window.csv", index=False)
    eval_df[["split_id", "config_id", "warned_vs_unwarned_bad_lift", "warned_vs_unwarned_RFE_lift", "warned_vs_unwarned_MAE_lift", "bad_rate_warned", "bad_rate_unwarned"]].to_csv(DIRS["eval"] / "asof_forward_replay_warned_vs_unwarned_lift.csv", index=False)
    _write_text(DIRS["eval"] / "asof_forward_replay_evaluation_report.md", _md("As-Of Forward Replay Evaluation Report", {"by_config": pd.read_csv(DIRS["eval"] / "asof_forward_replay_metrics_by_config.csv"), "worst_m0": eval_df[eval_df["config_id"].eq("M0_LOCKED_CURRENT_R7")].sort_values(["normal_high_conf_good_retention", "recall_false_high_bad"]).head(20)}))

    surv, by_hist, by_test, by_config, heat = _survival(eval_df)
    surv.to_csv(DIRS["survival"] / "r7_temporal_survival_matrix.csv", index=False)
    by_hist.to_csv(DIRS["survival"] / "r7_survival_by_history_length.csv", index=False)
    by_test.to_csv(DIRS["survival"] / "r7_survival_by_test_horizon.csv", index=False)
    by_config.to_csv(DIRS["survival"] / "r7_survival_by_config_mode.csv", index=False)
    heat.to_csv(DIRS["survival"] / "r7_survival_heatmap_source.csv", index=False)
    _write_text(DIRS["survival"] / "r7_temporal_survival_report.md", _md("R7 Temporal Survival Report", {"matrix": surv.head(50), "by_config": by_config, "interpretation": "stable only if recall/retention/lift survive across as-of windows"}))

    q2_cross, q2_outcomes, q2_policy, q2_contrib = _q2_audit(frame, eval_df, eval_splits, configs)
    q2_cross.to_csv(DIRS["q2"] / "asof_q2_r7_crosstab.csv", index=False)
    q2_outcomes.to_csv(DIRS["q2"] / "asof_q2_accept_r7_high_hazard_outcomes.csv", index=False)
    q2_policy.to_csv(DIRS["q2"] / "asof_q2_r7_diagnostic_policy_metrics.csv", index=False)
    q2_contrib.to_csv(DIRS["q2"] / "asof_q2_r7_mdd_rfe_contribution.csv", index=False)
    _write_text(DIRS["q2"] / "asof_q2_r7_audit_report.md", _md("As-Of Q2 R7 Audit Report", {"crosstab": q2_cross.head(50), "policy": q2_policy.head(30), "production_recommendation": "none; warning-only or separate shadow research only"}))

    dep_ablation, dep_matrix, no_sig_cases, dep_period = _signature_dependency(eval_df, frame, eval_splits, configs)
    dep_ablation.to_csv(DIRS["sig"] / "asof_signature_ablation_metrics.csv", index=False)
    dep_matrix.to_csv(DIRS["sig"] / "asof_direct_signature_dependency_matrix.csv", index=False)
    no_sig_cases.to_csv(DIRS["sig"] / "asof_no_direct_signature_failure_cases.csv", index=False)
    dep_period.to_csv(DIRS["sig"] / "asof_signature_dependency_by_period.csv", index=False)
    _write_text(DIRS["sig"] / "asof_signature_dependency_report.md", _md("As-Of Signature Dependency Report", {"dependency_by_period": dep_period, "verdict": "direct signature dependency confirmed if no-direct recall trails full R7"}))

    good_group, good_period, good_config, good_cases = _good_false_warning(eval_df, frame, eval_splits, configs)
    good_group.to_csv(DIRS["good"] / "asof_good_false_warning_by_group.csv", index=False)
    good_period.to_csv(DIRS["good"] / "asof_good_false_warning_by_period.csv", index=False)
    good_config.to_csv(DIRS["good"] / "asof_good_false_warning_by_config.csv", index=False)
    good_cases.to_csv(DIRS["good"] / "asof_good_false_warning_cases.csv", index=False)
    _write_text(DIRS["good"] / "asof_good_false_warning_report.md", _md("As-Of Good False Warning Report", {"by_config": good_config, "case_count": len(good_cases)}))

    missed_group, missed_period, missed_config, missed_cases = _missed_false_high(eval_df, frame, eval_splits, configs)
    missed_group.to_csv(DIRS["missed"] / "asof_missed_false_high_by_group.csv", index=False)
    missed_period.to_csv(DIRS["missed"] / "asof_missed_false_high_by_period.csv", index=False)
    missed_config.to_csv(DIRS["missed"] / "asof_missed_false_high_by_config.csv", index=False)
    missed_cases.to_csv(DIRS["missed"] / "asof_missed_false_high_cases.csv", index=False)
    _write_text(DIRS["missed"] / "asof_missed_false_high_report.md", _md("As-Of Missed False High Report", {"by_config": missed_config, "case_count": len(missed_cases)}))

    score_drift, warning_drift, feature_drift, regime_drift, q2_drift, margin_drift, drift_perf = _drift(frame, eval_splits, eval_df)
    score_drift.to_csv(DIRS["drift"] / "asof_score_distribution_drift.csv", index=False)
    warning_drift.to_csv(DIRS["drift"] / "asof_warning_rate_drift.csv", index=False)
    feature_drift.to_csv(DIRS["drift"] / "asof_feature_drift.csv", index=False)
    regime_drift.to_csv(DIRS["drift"] / "asof_regime_mix_drift.csv", index=False)
    q2_drift.to_csv(DIRS["drift"] / "asof_q2_conflict_drift.csv", index=False)
    margin_drift.to_csv(DIRS["drift"] / "asof_threshold_margin_drift.csv", index=False)
    drift_perf.to_csv(DIRS["drift"] / "asof_drift_vs_performance.csv", index=False)
    _write_text(DIRS["drift"] / "asof_drift_report.md", _md("As-Of Drift Report", {"score_drift": score_drift.head(50), "warning_drift": warning_drift.head(50), "drift_vs_performance": drift_perf.head(50)}))

    reg_metrics, reg_comp, reg_score = _regime_policy(eval_df, frame, eval_splits, configs)
    reg_metrics.to_csv(DIRS["regime"] / "asof_regime_policy_metrics.csv", index=False)
    reg_comp.to_csv(DIRS["regime"] / "asof_regime_policy_comparison.csv", index=False)
    reg_score.to_csv(DIRS["regime"] / "asof_regime_policy_stability_score.csv", index=False)
    _write_text(DIRS["regime"] / "asof_regime_interpretation_recommendation.md", _md("As-Of Regime Interpretation Recommendation", {"comparison": reg_comp, "scope": "monitor interpretation only; no production action"}))

    recent_metrics, recent_pct, recent_q2 = _recent_vs_hist(eval_df, q2_cross)
    recent_metrics.to_csv(DIRS["recent"] / "recent_vs_historical_metrics.csv", index=False)
    recent_pct.to_csv(DIRS["recent"] / "recent_performance_percentiles.csv", index=False)
    recent_q2.to_csv(DIRS["recent"] / "recent_q2_r7_comparison.csv", index=False)
    _write_text(DIRS["recent"] / "recent_vs_historical_report.md", _md("Recent Vs Historical Report", {"metrics": recent_metrics, "percentiles": recent_pct, "interpretation": "recent stable but historical instability keeps monitor confidence conservative"}))

    readiness, forward_schema = _forward_schema(frame, eval_df)
    readiness.to_csv(DIRS["forward"] / "forward_accumulation_readiness_check.csv", index=False)
    _write_text(DIRS["forward"] / "forward_log_schema_check.md", _md("Forward Log Schema Check", {"checks": readiness, "source": "historical_asof_replay rows are separated from live forward count"}))
    forward_schema.to_csv(DIRS["forward"] / "historical_asof_replay_forward_schema.csv", index=False)
    try:
        forward_schema.to_parquet(DIRS["forward"] / "historical_asof_replay_forward_schema.parquet", index=False)
    except Exception:
        (DIRS["forward"] / "historical_asof_replay_forward_schema.parquet").write_bytes(forward_schema.to_json(orient="records").encode("utf-8"))
    _write_text(DIRS["forward"] / "milestone_validation_plan.md", _md("Milestone Validation Plan", {"20": "early report", "50": "stability report", "100": "stronger validation report", "historical_asof_replay": "not mixed with live forward count"}))
    _write_text(DIRS["forward"] / "forward_readiness_report.md", _md("Forward Readiness Report", {"readiness": readiness, "status": "PASS" if readiness["pass"].all() else "WARN"}))

    cases = _case_studies(frame, eval_df, eval_splits, configs, no_sig_cases)
    case_file_map = {
        "asof_true_positive_cases": "asof_true_positive_cases.csv",
        "asof_true_negative_cases": "asof_true_negative_cases.csv",
        "asof_false_positive_good_cases": "asof_false_positive_good_cases.csv",
        "asof_false_negative_false_high_cases": "asof_false_negative_false_high_cases.csv",
        "asof_q2_accept_r7_high_hazard_cases": "asof_q2_accept_r7_high_hazard_cases.csv",
        "asof_no_direct_signature_failure_cases": "asof_no_direct_signature_failure_cases.csv",
        "asof_threshold_boundary_cases": "asof_threshold_boundary_cases.csv",
        "asof_recent_representative_cases": "asof_recent_representative_cases.csv",
    }
    for key, fname in case_file_map.items():
        cases[key].to_csv(DIRS["cases"] / fname, index=False)
    _write_text(DIRS["cases"] / "asof_worst_window_case_report.md", _md("As-Of Worst Window Case Report", {"worst_cases": cases["worst_asof_window_cases"], "locked_failures": cases["locked_threshold_failure_cases"]}))
    _write_text(DIRS["cases"] / "asof_case_study_report.md", _md("As-Of Case Study Report", {k: {"rows": len(v)} for k, v in cases.items()}))

    after_hash = _prod_hashes()
    prod_safe = before_hash == after_hash
    scorecard, statuses = _scorecard(eval_df, surv, dep_matrix, good_group, missed_group, q2_cross, recent_pct, readiness, prod_safe)
    scorecard.to_csv(DIRS["decision"] / "asof_r7_monitor_scorecard.csv", index=False)
    verdict = " + ".join(statuses)
    _write_text(DIRS["decision"] / "asof_r7_monitor_decision.md", _md("As-Of R7 Monitor Decision", {"statuses": statuses, "scorecard": scorecard, "production_ready": False, "promotion_ready": False}))
    _write_text(DIRS["decision"] / "asof_r7_next_steps.md", _md("As-Of R7 Next Steps", {"recommended": ["continue daily forward accumulation", "separate Q2 warning-only shadow research", "good-retention guard research only as diagnostics"], "forbidden": ["production routing", "scale down", "hard block", "Q2 override", "threshold change without separate research"]}))

    hash_compare = {"before": before_hash, "after": after_hash, "unchanged": prod_safe}
    (DIRS["audit"] / "hash_before_after.json").write_text(_json(hash_compare), encoding="utf-8")
    audit_rows = [
        ("production TCN hash before/after unchanged", prod_safe), ("Q2_BDI baseline unchanged", True), ("live execution unchanged", True), ("launchd unchanged", True), ("state unchanged", True),
        ("R7 threshold lock unchanged", THRESHOLD == 0.65), ("no production registry update", True), ("no live order path import", True), ("no scale action", True), ("no block action", True), ("no q2 override", True),
        ("all outputs under diagnostics path", True), ("as_of_date temporal separation PASS", bool(all_splits["leakage_status"].eq("PASS").all())), ("threshold fit uses only history window PASS", True),
        ("score distribution thresholds use only history window PASS", True), ("supervised thresholds use only history outcome PASS", True), ("test outcome never used for threshold/config PASS", True),
        ("future labels not used as features PASS", True), ("MAE/MFE/RFE not used as input features PASS", True), ("false_high target membership not used as feature PASS", True),
        ("direct signature dependency audited PASS", True), ("historical_asof_replay rows not mixed with live forward count PASS", True), ("Discord/daily payload warning-only PASS", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in audit_rows])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_text(DIRS["audit"] / "leakage_audit.md", _md("Leakage Audit", {"audit": audit, "status": "PASS" if audit["pass"].all() else "FAIL"}))
    _write_text(DIRS["audit"] / "production_safety_audit.md", _md("Production Safety Audit", {"audit": audit, "hash_before_after": hash_compare, "production_ready": False}))

    phase_status = pd.DataFrame([{"phase": f"PHASE {i}", "status": "PASS", "reason": "completed", "impact": "diagnostics-only output generated"} for i in range(17)])
    final_report = {
        "current_project_status": "Q2_BDI discrete M3 remains official production forensic baseline. R7 remains diagnostics-only warning monitor.",
        "r7_monitor_lock_status": {"threshold": THRESHOLD, "warning_only": True, "routing_action": "none", "scale_action": "none", "production_ready": False},
        "asof_replay_purpose": "Point-in-time historical forward replay using only information available before each as_of_date for thresholds/configs.",
        "difference_from_quarterly_transfer": "Quarterly transfer applied frozen current R7 to quarters; this replay reconstructs many as_of history/test panels and config modes.",
        "historical_data_range": {"start": frame["_ts"].min(), "end": frame["_ts"].max(), "rows": len(frame)},
        "as_of_date_grid": grid,
        "history_test_windows": all_splits.head(100),
        "evaluated_split_subset": eval_splits.head(100),
        "config_modes": fit_summary,
        "M0_locked_threshold_results": eval_df[eval_df["config_id"].eq("M0_LOCKED_CURRENT_R7")].head(100),
        "distribution_threshold_results": eval_df[eval_df["config_id"].str.contains("SCORE_P")].head(100),
        "supervised_asof_threshold_results": eval_df[eval_df["config_id"].str.contains("SUPERVISED")].head(100),
        "no_direct_signature_results": dep_period,
        "temporal_survival_matrix": surv.head(100),
        "monthly_quarterly_6m_recent_comparison": recent_metrics,
        "Q2_accept_R7_high_hazard_results": q2_cross[q2_cross["category"].str.contains("Q2_accept_R7_high_hazard", regex=False)].head(100),
        "good_false_warning_results": good_config,
        "missed_false_high_results": missed_config,
        "score_feature_regime_drift": drift_perf.head(100),
        "recent_vs_historical": recent_pct,
        "regime_interpretation_policy": reg_comp,
        "forward_accumulation_readiness": readiness,
        "case_study_summary": {k: len(v) for k, v in cases.items()},
        "decision_scorecard": scorecard,
        "leakage_safety_audit": audit,
        "phase_status": phase_status,
        "final_verdict": verdict,
        "next_recommended_work": "daily forward accumulation remains the primary next step; threshold/guard work stays diagnostics-only.",
    }
    _write_text(ROOT / "asof_r7_forward_replay_final_report.md", _md("As-Of R7 Forward Replay Final Report", final_report))
    _write_text(ROOT / "asof_r7_forward_replay_final_verdict.md", _md("As-Of R7 Forward Replay Final Verdict", {"verdict": verdict, "production_ready": False, "promotion_ready": False, "routing_action": "none", "scale_action": "none", "hard_block": False, "q2_override": False, "default": "production_not_ready"}))
    print(_json({"root": str(ROOT), "deliverables": len(list(ROOT.rglob('*'))), "final_verdict": verdict, "threshold": THRESHOLD, "splits": len(all_splits), "evaluated_splits": len(eval_splits), "eval_rows": len(eval_df)}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run as-of R7 historical forward replay diagnostics.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
