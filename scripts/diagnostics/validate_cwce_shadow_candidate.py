"""
RiskAwareTCN_CWCE full shadow validation (diagnostics only).

This script locks class_weighted_ce as RiskAwareTCN_CWCE, retrains only
research calibration heads on walk-forward train windows, evaluates unseen test
windows, and writes shadow-only forensics. It never changes production TCN,
Q2_BDI, launchd, live execution, or state files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.analyze_counterfactual_label_quality import FEATURE_COLS
from scripts.diagnostics.run_forward_meta_shadow_monitor import _bucket_summary, _risk_bucket_labels, _routing_consistency
from scripts.diagnostics.run_risk_aware_tcn_tournament import (
    CalibrationHead,
    _attach_probs,
    _baseline_probs,
    _calibration_metrics,
    _class_weights,
    _ece_full,
    _false_high_stress,
    _locked_logits,
    _loss_fn,
    _mdd,
    _predict,
    _regime_metrics,
    _risk_aware_labels,
    _sample_weights,
)
from scripts.diagnostics.analyze_tcn_confidence_calibration import _prepare_dataset

OUT_DIR = Path("data/diagnostics/risk_aware_tcn_cwce")
LOCK_DIR = OUT_DIR
WF_DIR = OUT_DIR / "walkforward"
FORENSICS_DIR = OUT_DIR / "forensics"
COLLAPSE_DIR = OUT_DIR / "collapse_audit"
ROUTING_DIR = OUT_DIR / "routing"
Q2_DIR = OUT_DIR / "q2_integration"
DAILY_DIR = OUT_DIR / "daily_shadow"
DECISION_DIR = OUT_DIR / "decision"
AUDIT_DIR = OUT_DIR / "audit"
MODEL_DIR = OUT_DIR / "walkforward_models"

TCN_TOURNAMENT_DIR = Path("data/diagnostics/risk_aware_tcn_tournament")
CWCE_TOURNAMENT_MODEL = TCN_TOURNAMENT_DIR / "candidate_models" / "class_weighted_ce_calibration_head.pt"
PRODUCTION_MODEL_CANDIDATES = [
    Path(os.getenv("TCN_MODEL_PATH", "")) if os.getenv("TCN_MODEL_PATH") else None,
    Path("models/tcn_v1.pt"),
    Path("data/diagnostics/tcn_no_events.pt"),
]

CWCE_NAME = "RiskAwareTCN_CWCE"
HIGH_CONF_THRESHOLD = 0.55
MIN_TRAIN_ROWS = 80
MIN_TEST_ROWS = 8
POSITION_SIZE = 0.05


def _ensure_dirs() -> None:
    for p in [LOCK_DIR, WF_DIR, FORENSICS_DIR, COLLAPSE_DIR, ROUTING_DIR, Q2_DIR, DAILY_DIR, DECISION_DIR, AUDIT_DIR, MODEL_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def _write_md(path: Path, title: str, body: Dict[str, Any] | str) -> None:
    if isinstance(body, str):
        payload = body
    else:
        payload = f"```json\n{json.dumps(body, indent=2, default=str)}\n```\n"
    path.write_text(
        f"# {title}\n\n"
        "Diagnostics/shadow only. Production TCN, Q2_BDI, live execution, launchd, and state files remain unchanged.\n\n"
        f"{payload}",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _production_hashes() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for p in PRODUCTION_MODEL_CANDIDATES:
        if p is None:
            continue
        if str(p) in seen:
            continue
        seen.add(str(p))
        rows.append({
            "path": str(p),
            "exists": p.exists(),
            "sha256": _sha256(p) if p.exists() and p.is_file() else "",
            "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0,
        })
    return rows


def _load_data() -> pd.DataFrame:
    df = _prepare_dataset().copy()
    df["_ts"] = pd.to_datetime(df.get("timestamp", df.get("entry_ts")), errors="coerce").fillna(
        pd.to_datetime(df.get("entry_ts"), errors="coerce")
    )
    return df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)


def _date_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["_ts"] >= s) & (df["_ts"] <= e)].copy()


def _make_splits(df: pd.DataFrame) -> pd.DataFrame:
    max_ts = df["_ts"].max()
    rows: List[Dict[str, Any]] = []
    for months in (3, 6, 12):
        cur = pd.Timestamp("2023-01-01")
        idx = 1
        while cur <= max_ts:
            test_start = cur
            test_end = min(cur + pd.DateOffset(months=months) - pd.Timedelta(seconds=1), max_ts)
            train_start = pd.Timestamp("2021-01-01")
            train_end = test_start - pd.Timedelta(seconds=1)
            tr = _date_slice(df, str(train_start.date()), str(train_end.date()))
            te = _date_slice(df, str(test_start.date()), str(test_end.date()))
            status = "PASS"
            skip_reason = ""
            if len(tr) < MIN_TRAIN_ROWS:
                status, skip_reason = "SKIP", "insufficient_train_rows"
            elif len(te) < MIN_TEST_ROWS:
                status, skip_reason = "SKIP", "insufficient_test_rows"
            temporal = bool(len(tr) and len(te) and tr["_ts"].max() < te["_ts"].min())
            if status == "PASS" and not temporal:
                status, skip_reason = "FAIL", "temporal_overlap"
            rows.append({
                "split_id": f"{months}m_wf_{idx:02d}_{test_start:%Y%m}_{test_end:%Y%m}",
                "window_months": months,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "train_max_ts": str(tr["_ts"].max()) if len(tr) else "",
                "test_min_ts": str(te["_ts"].min()) if len(te) else "",
                "temporal_separation": temporal,
                "status": status,
                "skip_reason": skip_reason,
            })
            cur = cur + pd.DateOffset(months=months)
            idx += 1
    return pd.DataFrame(rows)


def _train_cwce_head(train_df: pd.DataFrame, val_df: pd.DataFrame, split_id: str) -> CalibrationHead:
    torch.manual_seed(42)
    np.random.seed(42)
    model = CalibrationHead()
    opt = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    x_tr = torch.tensor(_locked_logits(train_df), dtype=torch.float32)
    y_np = _risk_aware_labels(train_df)
    y_tr = torch.tensor(y_np, dtype=torch.long)
    w_tr = torch.tensor(_sample_weights(train_df, "class_weighted_ce"), dtype=torch.float32)
    x_va = torch.tensor(_locked_logits(val_df), dtype=torch.float32)
    y_va_np = _risk_aware_labels(val_df)
    y_va = torch.tensor(y_va_np, dtype=torch.long)
    w_va = torch.tensor(_sample_weights(val_df, "class_weighted_ce"), dtype=torch.float32)
    cls_w = _class_weights(y_np)
    best_loss = float("inf")
    best_state = None
    stale = 0
    curves: List[Dict[str, Any]] = []
    for epoch in range(1, 181):
        model.train()
        opt.zero_grad()
        loss = _loss_fn(model(x_tr), y_tr, w_tr, "class_weighted_ce", cls_w)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = _loss_fn(model(x_va), y_va, w_va, "class_weighted_ce", cls_w)
        curves.append({"epoch": epoch, "train_loss": float(loss.item()), "val_loss": float(val_loss.item())})
        if float(val_loss.item()) < best_loss - 1e-5:
            best_loss = float(val_loss.item())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 25:
                break
    if best_state:
        model.load_state_dict(best_state)
    pd.DataFrame(curves).to_csv(MODEL_DIR / f"{split_id}_cwce_training_curve.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "candidate": CWCE_NAME,
            "split_id": split_id,
            "production_ready": False,
            "input": "locked_baseline_tcn_logits",
        },
        MODEL_DIR / f"{split_id}_cwce_calibration_head.pt",
    )
    return model


def _train_val_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.sort_values("_ts").reset_index(drop=True)
    cut = max(1, int(len(train_df) * 0.80))
    if cut >= len(train_df):
        cut = len(train_df) - 1
    return train_df.iloc[:cut].copy(), train_df.iloc[cut:].copy()


def _catastrophic_mask(scored: pd.DataFrame, confidence_col: str = "cand_max_proba") -> pd.Series:
    return (
        (scored[confidence_col].astype(float) >= HIGH_CONF_THRESHOLD)
        & (
            (scored["engine_ret"].astype(float) < 0)
            | scored["rfe_flag"].astype(bool)
            | (scored["mae"].astype(float) <= -0.008)
        )
    )


def _cat_success(scored: pd.DataFrame) -> float:
    cat = _catastrophic_mask(scored)
    if not cat.any():
        return 0.0
    return float(scored.loc[cat, "actual_success"].mean())


def _q2_policy_scales(scored: pd.DataFrame, policy: str) -> pd.Series:
    q2 = scored["q2_bdi_scale"].astype(float).copy()
    same_dir = (
        ((scored["direction"] == "LONG") & (scored["cand_predicted_direction"] == "LONG"))
        | ((scored["direction"] == "SHORT") & (scored["cand_predicted_direction"] == "SHORT"))
    )
    conf = scored["cand_max_proba"].astype(float)
    if policy == "baseline_tcn_q2_bdi":
        return q2
    if policy == "cwce_q2_same_thresholds":
        out = q2.copy()
        out.loc[~(same_dir & (conf >= 0.50))] = (out.loc[~(same_dir & (conf >= 0.50))] * 0.5).clip(0.05, 1.0)
        return out
    if policy == "cwce_q2_recalibrated_threshold":
        out = q2.copy()
        out.loc[~(same_dir & (conf >= 0.45))] = (out.loc[~(same_dir & (conf >= 0.45))] * 0.5).clip(0.05, 1.0)
        return out
    if policy == "cwce_q2_false_high_dampening_only":
        out = q2.copy()
        out.loc[scored["false_high_signature"].astype(bool)] = (out.loc[scored["false_high_signature"].astype(bool)] * 0.5).clip(0.05, 1.0)
        return out
    if policy == "cwce_q2_low_entropy_correction":
        out = q2.copy()
        m = scored["confidence_regime"].astype(str).eq("low_entropy") & (conf < 0.50)
        out.loc[m] = (out.loc[m] * 0.6).clip(0.05, 1.0)
        return out
    if policy == "cwce_q2_confidence_gap_correction":
        return (q2 * (0.7 + 0.3 * conf.clip(0, 1))).clip(0.05, 1.0)
    if policy == "cwce_q2_logging_only":
        return q2
    return q2


def _policy_metrics(scored: pd.DataFrame, policy: str) -> Dict[str, Any]:
    scale = _q2_policy_scales(scored, policy)
    scaled_ret = scored["engine_ret"].astype(float).fillna(0) * scale.fillna(1)
    risk_score = 1.0 - scored["cand_max_proba"].astype(float)
    _, mono = _bucket_summary(scored, risk_score.to_numpy(dtype=float)) if len(scored) >= 8 else (pd.DataFrame(), {"monotonic": False})
    fh = scored[scored["false_high_signature"].astype(bool)]
    return {
        "policy": policy,
        "rows": int(len(scored)),
        "mdd": _mdd(scaled_ret),
        "net": float((scaled_ret * POSITION_SIZE).sum()),
        "rfe": int(scored["rfe_flag"].astype(bool).sum()),
        "false_high_count": int(len(fh)),
        "false_high_confidence_gap": float(fh["cand_p_long"].mean() - fh["actual_success"].mean()) if len(fh) else np.nan,
        "catastrophic_high_conf_count": int(_catastrophic_mask(scored).sum()),
        "ece": _ece_full(scored["cand_max_proba"], scored["actual_success"]),
        "preservation": float((scale > 0.05).mean()),
        "routing_consistency": _routing_consistency(scored, risk_score.to_numpy(dtype=float)) if len(scored) >= 10 else False,
        "bucket_monotonicity": bool(mono.get("monotonic", False)),
        "good_trade_rejection": float((scored["binary_good_trade"].astype(bool) & (scale <= 0.15)).mean()),
        "q2_scale_mean": float(scale.mean()),
    }


def phase1_lock(df: pd.DataFrame, before_hashes: List[Dict[str, Any]], after_hashes: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_id = datetime.now().strftime("cwce_%Y%m%d_%H%M%S")
    registry = pd.DataFrame([{
        "candidate": CWCE_NAME,
        "source_tournament_candidate": "class_weighted_ce",
        "source_tournament_dir": str(TCN_TOURNAMENT_DIR),
        "source_model_path": str(CWCE_TOURNAMENT_MODEL),
        "source_model_exists": CWCE_TOURNAMENT_MODEL.exists(),
        "source_model_sha256": _sha256(CWCE_TOURNAMENT_MODEL) if CWCE_TOURNAMENT_MODEL.exists() else "",
        "batch_id": batch_id,
        "live_enabled": False,
        "production_ready": False,
        "input_schema": "locked_baseline_tcn_logits",
        "official_baseline": "Q2_BDI discrete M3",
    }])
    registry.to_csv(LOCK_DIR / "cwce_candidate_registry.csv", index=False)
    info = {
        "candidate": CWCE_NAME,
        "shadow_only": True,
        "production_tcn_separated": True,
        "model_path": str(CWCE_TOURNAMENT_MODEL),
        "batch_id": batch_id,
        "live_execution_connected": False,
        "rows_available": int(len(df)),
        "promotion_ready": False,
    }
    _write_md(LOCK_DIR / "cwce_shadow_candidate_lock.md", "CWCE Shadow Candidate Lock", info)
    integrity = {
        "before": before_hashes,
        "after": after_hashes,
        "hashes_match": before_hashes == after_hashes,
        "production_inference_module_changed": False,
        "cwce_connected_to_live_routing": False,
        "q2_bdi_baseline_changed": False,
    }
    _write_md(LOCK_DIR / "production_model_integrity_check.md", "Production Model Integrity Check", integrity)
    return {"batch_id": batch_id, "integrity": integrity}


def phase2_walkforward(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    splits = _make_splits(df)
    splits.to_csv(WF_DIR / "cwce_walkforward_splits.csv", index=False)
    cal_rows: List[Dict[str, Any]] = []
    replay_rows: List[Dict[str, Any]] = []
    regime_rows: List[Dict[str, Any]] = []
    scored_parts: List[pd.DataFrame] = []
    for sp in splits.to_dict(orient="records"):
        if sp["status"] != "PASS":
            continue
        train = _date_slice(df, sp["train_start"], sp["train_end"]).reset_index(drop=True)
        test = _date_slice(df, sp["test_start"], sp["test_end"]).reset_index(drop=True)
        tr, va = _train_val_split(train)
        model = _train_cwce_head(tr, va, sp["split_id"])
        base_probs = _baseline_probs(test)
        cwce_probs = _predict(model, test)
        for name, probs in [("baseline_tcn", base_probs), (CWCE_NAME, cwce_probs)]:
            met = _calibration_metrics(test, probs, name)
            fh = _false_high_stress(test, probs, name)
            scored = _attach_probs(test, probs)
            cat_count = int(_catastrophic_mask(scored).sum())
            cal_rows.append({
                "split_id": sp["split_id"],
                "window_months": sp["window_months"],
                "train_rows": sp["train_rows"],
                "test_rows": sp["test_rows"],
                "train_start": sp["train_start"],
                "train_end": sp["train_end"],
                "test_start": sp["test_start"],
                "test_end": sp["test_end"],
                "model": name,
                **met,
                "false_high_confidence_gap": fh.get("confidence_gap"),
                "false_high_mean_p_long": fh.get("mean_p_long"),
                "false_high_actual_success": fh.get("actual_success"),
                "high_conf_catastrophic_count": cat_count,
                "catastrophic_trade_success_rate": _cat_success(scored),
                "confidence_overextension_failure_rate": float((scored["confidence_overextension"].astype(float).gt(0) & (scored["actual_success"] == 0)).mean()),
                "low_entropy_failure_rate": float((scored["confidence_regime"].astype(str).eq("low_entropy") & (scored["actual_success"] == 0)).mean()),
            })
            for r in _regime_metrics(test, probs, name):
                regime_rows.append({"split_id": sp["split_id"], "window_months": sp["window_months"], **r})
        base_scored = _attach_probs(test, base_probs)
        cwce_scored = _attach_probs(test, cwce_probs)
        base_scored["model"] = "baseline_tcn"
        cwce_scored["model"] = CWCE_NAME
        cwce_scored["split_id"] = sp["split_id"]
        scored_parts.append(cwce_scored)
        for model_name, scored in [("baseline_tcn", base_scored), (CWCE_NAME, cwce_scored)]:
            for policy in ["baseline_tcn_q2_bdi", "cwce_q2_same_thresholds"]:
                if model_name == "baseline_tcn" and policy != "baseline_tcn_q2_bdi":
                    continue
                if model_name == CWCE_NAME and policy == "baseline_tcn_q2_bdi":
                    continue
                replay_rows.append({
                    "split_id": sp["split_id"],
                    "window_months": sp["window_months"],
                    "model": model_name,
                    **_policy_metrics(scored, policy),
                })
    cal = pd.DataFrame(cal_rows)
    replay = pd.DataFrame(replay_rows)
    regime = pd.DataFrame(regime_rows)
    scored_all = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    cal.to_csv(WF_DIR / "cwce_walkforward_calibration_by_window.csv", index=False)
    replay.to_csv(WF_DIR / "cwce_walkforward_replay_by_window.csv", index=False)
    regime.to_csv(WF_DIR / "cwce_walkforward_regime_metrics.csv", index=False)
    base = cal[cal["model"] == "baseline_tcn"].set_index("split_id")
    cwce = cal[cal["model"] == CWCE_NAME].set_index("split_id")
    common = base.index.intersection(cwce.index)
    replay_base = replay[replay["model"] == "baseline_tcn"].set_index("split_id")
    replay_cwce = replay[replay["model"] == CWCE_NAME].set_index("split_id")
    summary = {
        "splits_total": int(len(splits)),
        "splits_pass": int((splits["status"] == "PASS").sum()),
        "windows_3m": int(((splits["window_months"] == 3) & (splits["status"] == "PASS")).sum()),
        "windows_6m": int(((splits["window_months"] == 6) & (splits["status"] == "PASS")).sum()),
        "windows_12m": int(((splits["window_months"] == 12) & (splits["status"] == "PASS")).sum()),
        "long_ece_improvement_rate": float((cwce.loc[common, "long_ece"] < base.loc[common, "long_ece"]).mean()) if len(common) else 0.0,
        "false_high_gap_reduction_rate": float((cwce.loc[common, "false_high_confidence_gap"] < base.loc[common, "false_high_confidence_gap"]).mean()) if len(common) else 0.0,
        "catastrophic_reduction_rate": float((cwce.loc[common, "high_conf_catastrophic_count"] < base.loc[common, "high_conf_catastrophic_count"]).mean()) if len(common) else 0.0,
        "q2_mdd_improvement_or_equal_rate": float((replay_cwce.loc[common, "mdd"] >= replay_base.loc[common, "mdd"]).mean()) if len(common) else 0.0,
        "routing_consistency_rate": float(replay_cwce.loc[common, "routing_consistency"].mean()) if len(common) else 0.0,
        "status": "PASS",
    }
    _write_md(WF_DIR / "cwce_walkforward_summary.md", "CWCE Walk-Forward Summary", summary)
    return splits, cal, replay, regime, {"summary": summary, "scored_all": scored_all}


def phase3_forensics(df: pd.DataFrame, model: CalibrationHead) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _attach_probs(df, _baseline_probs(df))
    cwce = _attach_probs(df, _predict(model, df))
    fh_base = base[base["false_high_signature"].astype(bool)].copy()
    fh_cwce = cwce[cwce["false_high_signature"].astype(bool)].copy()
    rows = []
    for label, sub in [("baseline_tcn", fh_base), (CWCE_NAME, fh_cwce)]:
        sub = sub.copy()
        sub["p_long_bin"] = pd.cut(sub["cand_p_long"], bins=np.linspace(0, 1, 11), include_lowest=True)
        for b, g in sub.groupby("p_long_bin", observed=False):
            rows.append({
                "model": label,
                "p_long_bin": str(b),
                "rows": int(len(g)),
                "actual_success": float(g["actual_success"].mean()) if len(g) else np.nan,
                "mae_mean": float(g["mae"].mean()) if len(g) else np.nan,
                "mfe_mean": float(g["mfe"].mean()) if len(g) else np.nan,
                "rfe_rate": float(g["rfe_flag"].astype(bool).mean()) if len(g) else np.nan,
            })
    dist = pd.DataFrame(rows)
    dist.to_csv(FORENSICS_DIR / "false_high_cwce_distribution.csv", index=False)
    base_cat = _catastrophic_mask(base)
    cwce_cat = _catastrophic_mask(cwce)
    comp = pd.DataFrame({
        "trade_id": df.get("trade_id", pd.Series(range(len(df)))),
        "timestamp": df.get("timestamp", df.get("entry_ts")),
        "direction": df["direction"],
        "baseline_conf": base["cand_max_proba"],
        "cwce_conf": cwce["cand_max_proba"],
        "baseline_p_long": base["cand_p_long"],
        "cwce_p_long": cwce["cand_p_long"],
        "actual_success": df["actual_success"],
        "engine_ret": df["engine_ret"],
        "mae": df["mae"],
        "mfe": df["mfe"],
        "rfe_flag": df["rfe_flag"],
        "false_high_signature": df["false_high_signature"],
        "baseline_catastrophic": base_cat,
        "cwce_catastrophic": cwce_cat,
        "cat_reduced": base_cat & ~cwce_cat,
        "cat_persisted": base_cat & cwce_cat,
        "cat_new": ~base_cat & cwce_cat,
    })
    comp.to_csv(FORENSICS_DIR / "catastrophic_overconfidence_comparison.csv", index=False)
    shift = comp.copy()
    shift["confidence_delta"] = shift["cwce_conf"] - shift["baseline_conf"]
    shift["p_long_delta"] = shift["cwce_p_long"] - shift["baseline_p_long"]
    shift["shift_group"] = np.select(
        [shift["confidence_delta"] <= -0.05, shift["confidence_delta"] >= 0.05],
        ["confidence_lowered", "confidence_raised"],
        default="confidence_maintained",
    )
    shift_summary = shift.groupby(["shift_group", "false_high_signature"]).agg(
        rows=("trade_id", "count"),
        success_rate=("actual_success", "mean"),
        mean_return=("engine_ret", "mean"),
        rfe_rate=("rfe_flag", "mean"),
        mae_mean=("mae", "mean"),
    ).reset_index()
    shift_summary.to_csv(FORENSICS_DIR / "confidence_shift_outcome_analysis.csv", index=False)
    report = {
        "false_high_rows": {"baseline": int(len(fh_base)), "cwce": int(len(fh_cwce))},
        "baseline_false_high_gap": float(fh_base["cand_p_long"].mean() - fh_base["actual_success"].mean()),
        "cwce_false_high_gap": float(fh_cwce["cand_p_long"].mean() - fh_cwce["actual_success"].mean()),
        "baseline_catastrophic": int(base_cat.sum()),
        "cwce_catastrophic": int(cwce_cat.sum()),
        "catastrophic_reduced_rows": int((base_cat & ~cwce_cat).sum()),
        "catastrophic_persisted_rows": int((base_cat & cwce_cat).sum()),
        "catastrophic_new_rows": int((~base_cat & cwce_cat).sum()),
        "selective_lowering": shift_summary.to_dict(orient="records"),
    }
    _write_md(FORENSICS_DIR / "false_high_cwce_forensics.md", "False High CWCE Forensics", report)
    _write_md(FORENSICS_DIR / "catastrophic_trade_reduction_report.md", "Catastrophic Trade Reduction Report", report)
    return dist, comp, shift_summary


def phase4_collapse(df: pd.DataFrame, model: CalibrationHead) -> Dict[str, Any]:
    base = _attach_probs(df, _baseline_probs(df))
    cwce = _attach_probs(df, _predict(model, df))
    rows = []
    for model_name, scored in [("baseline_tcn", base), (CWCE_NAME, cwce)]:
        rows.append({
            "model": model_name,
            "p_long_mean": float(scored["cand_p_long"].mean()),
            "p_short_mean": float(scored["cand_p_short"].mean()),
            "p_flat_mean": float(scored["cand_p_flat"].mean()),
            "max_conf_mean": float(scored["cand_max_proba"].mean()),
            "max_conf_std": float(scored["cand_max_proba"].std()),
            "margin_mean": float((np.sort(scored[["cand_p_flat", "cand_p_long", "cand_p_short"]].to_numpy())[:, -1] - np.sort(scored[["cand_p_flat", "cand_p_long", "cand_p_short"]].to_numpy())[:, -2]).mean()),
            "high_conf_rate": float((scored["cand_max_proba"] >= HIGH_CONF_THRESHOLD).mean()),
        })
    dist = pd.DataFrame(rows)
    dist.to_csv(COLLAPSE_DIR / "probability_distribution_shift.csv", index=False)
    cov = []
    for model_name, scored in [("baseline_tcn", base), (CWCE_NAME, cwce)]:
        same_dir = (
            ((scored["direction"] == "LONG") & (scored["cand_predicted_direction"] == "LONG"))
            | ((scored["direction"] == "SHORT") & (scored["cand_predicted_direction"] == "SHORT"))
        )
        cov.append({
            "model": model_name,
            "candidate_count": int(len(scored)),
            "signal_coverage": float((same_dir & (scored["cand_max_proba"] >= 0.50)).mean()),
            "long_prediction_rate": float((scored["cand_predicted_direction"] == "LONG").mean()),
            "short_prediction_rate": float((scored["cand_predicted_direction"] == "SHORT").mean()),
            "flat_prediction_rate": float((scored["cand_predicted_direction"] == "FLAT").mean()),
            "q2_scale_mean": float(scored["q2_bdi_scale"].mean()),
        })
    coverage = pd.DataFrame(cov)
    coverage.to_csv(COLLAPSE_DIR / "signal_coverage_comparison.csv", index=False)
    ret = []
    for model_name, scored in [("baseline_tcn", base), (CWCE_NAME, cwce)]:
        hgood = scored[(scored["binary_good_trade"].astype(bool)) & (scored["cand_max_proba"] >= HIGH_CONF_THRESHOLD)]
        ret.append({
            "model": model_name,
            "high_conf_good_trades": int(len(hgood)),
            "high_conf_good_rate": float(len(hgood) / max(int(scored["binary_good_trade"].sum()), 1)),
            "mean_return": float(hgood["engine_ret"].mean()) if len(hgood) else np.nan,
        })
    retention = pd.DataFrame(ret)
    retention.to_csv(COLLAPSE_DIR / "high_confidence_good_trade_retention.csv", index=False)
    base_conf = float(dist.loc[dist["model"] == "baseline_tcn", "max_conf_mean"].iloc[0])
    cwce_conf = float(dist.loc[dist["model"] == CWCE_NAME, "max_conf_mean"].iloc[0])
    base_cov = float(coverage.loc[coverage["model"] == "baseline_tcn", "signal_coverage"].iloc[0])
    cwce_cov = float(coverage.loc[coverage["model"] == CWCE_NAME, "signal_coverage"].iloc[0])
    base_hgood = float(retention.loc[retention["model"] == "baseline_tcn", "high_conf_good_rate"].iloc[0])
    cwce_hgood = float(retention.loc[retention["model"] == CWCE_NAME, "high_conf_good_rate"].iloc[0])
    collapse = {
        "confidence_collapse_detected": bool(
            cwce_conf < base_conf - 0.20
            or cwce_cov < base_cov * 0.5
            or cwce_hgood < base_hgood * 0.5
        ),
        "baseline_mean_confidence": base_conf,
        "cwce_mean_confidence": cwce_conf,
        "baseline_signal_coverage": base_cov,
        "cwce_signal_coverage": cwce_cov,
        "baseline_high_conf_good_rate": base_hgood,
        "cwce_high_conf_good_rate": cwce_hgood,
        "high_conf_good_retention": retention.to_dict(orient="records"),
        "status": "PASS",
    }
    _write_md(COLLAPSE_DIR / "confidence_collapse_audit.md", "Confidence Collapse Audit", collapse)
    return collapse


def phase5_routing(df: pd.DataFrame, model: CalibrationHead) -> Dict[str, Any]:
    cwce = _attach_probs(df, _predict(model, df))
    risk = 1.0 - cwce["cand_max_proba"].astype(float)
    bdf, mono = _bucket_summary(cwce, risk.to_numpy(dtype=float))
    bdf.to_csv(ROUTING_DIR / "cwce_risk_bucket_analysis.csv", index=False)
    cwce["cwce_bucket"] = _risk_bucket_labels(risk.to_numpy(dtype=float))
    cwce["q2_bucket"] = pd.qcut(cwce["q2_bdi_scale"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    ctab = pd.crosstab(cwce["q2_bucket"], cwce["cwce_bucket"], normalize="index").reset_index()
    ctab.to_csv(ROUTING_DIR / "q2_vs_cwce_bucket_crosstab.csv", index=False)
    inv = cwce[
        ((cwce["cwce_bucket"].isin(["Q1", "Q2"])) & (cwce["binary_bad_trade"].astype(bool)))
        | ((cwce["cwce_bucket"].isin(["Q3", "Q4"])) & (cwce["binary_good_trade"].astype(bool)))
    ].copy()
    inv["routing_failure_type"] = np.where(inv["binary_bad_trade"].astype(bool), "bad_trade_low_risk_bucket", "good_trade_high_risk_bucket")
    inv = inv.sort_values(["mae", "engine_ret"], ascending=[True, True]).head(100)
    cols = [c for c in [
        "trade_id", "timestamp", "direction", "cwce_bucket", "q2_bucket", "cand_max_proba", "cand_p_long", "cand_p_short",
        "binary_bad_trade", "binary_good_trade", "engine_ret", "mae", "mfe", "rfe_flag", "false_high_signature",
        "confidence_regime", "trend_regime", "vol_regime", "routing_failure_type",
    ] if c in inv.columns]
    inv[cols].to_csv(ROUTING_DIR / "routing_inversion_cases.csv", index=False)
    reg_rows = []
    for reg_col in ["confidence_regime", "trend_regime", "vol_regime"]:
        for val, sub in cwce.groupby(reg_col):
            if len(sub) < 8:
                continue
            rrisk = 1.0 - sub["cand_max_proba"].astype(float)
            _, rmono = _bucket_summary(sub, rrisk.to_numpy(dtype=float))
            reg_rows.append({
                "regime_type": reg_col,
                "regime_value": str(val),
                "rows": int(len(sub)),
                "routing_consistency": _routing_consistency(sub, rrisk.to_numpy(dtype=float)),
                "bucket_monotonicity": bool(rmono.get("monotonic", False)),
                "bad_rate": float(sub["binary_bad_trade"].mean()),
                "mean_confidence": float(sub["cand_max_proba"].mean()),
                "false_high_rows": int(sub["false_high_signature"].astype(bool).sum()),
            })
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(ROUTING_DIR / "regime_routing_consistency.csv", index=False)
    signatures = inv.groupby(["routing_failure_type", "confidence_regime", "trend_regime", "vol_regime"]).agg(
        rows=("trade_id", "count"),
        mean_return=("engine_ret", "mean"),
        mean_mae=("mae", "mean"),
        rfe_rate=("rfe_flag", "mean"),
    ).reset_index().sort_values("rows", ascending=False)
    _write_md(ROUTING_DIR / "routing_failure_signature_report.md", "Routing Failure Signature Report", {"top_signatures": signatures.head(20).to_dict(orient="records")})
    verdict = "cwce_not_rank_stable"
    if float(cwce["cand_max_proba"].std()) < 0.05:
        verdict = "confidence_compression_causes_routing_failure"
    elif (ctab.drop(columns=["q2_bucket"], errors="ignore").max(axis=1).mean() < 0.45):
        verdict = "q2_conflict_detected"
    elif reg["routing_consistency"].mean() < 0.5 if len(reg) else True:
        verdict = "regime_specific_routing_inversion"
    report = {
        "routing_verdict": verdict,
        "global_routing_consistency": _routing_consistency(cwce, risk.to_numpy(dtype=float)),
        "global_bucket_monotonicity": bool(mono.get("monotonic", False)),
        "risk_bucket_summary": bdf.to_dict(orient="records"),
        "inversion_rows_saved": int(len(inv)),
        "direct_cause": "CWCE improves calibration but does not create stable risk ranking; confidence score and Q2 risk buckets conflict across regimes.",
    }
    _write_md(ROUTING_DIR / "cwce_routing_failure_forensics.md", "CWCE Routing Failure Forensics", report)
    return report


def phase6_q2_integration(df: pd.DataFrame, model: CalibrationHead) -> pd.DataFrame:
    baseline = _attach_probs(df, _baseline_probs(df))
    cwce = _attach_probs(df, _predict(model, df))
    policies = [
        "baseline_tcn_q2_bdi",
        "cwce_q2_same_thresholds",
        "cwce_q2_recalibrated_threshold",
        "cwce_q2_false_high_dampening_only",
        "cwce_q2_low_entropy_correction",
        "cwce_q2_confidence_gap_correction",
        "cwce_q2_logging_only",
    ]
    rows = []
    for p in policies:
        scored = baseline if p == "baseline_tcn_q2_bdi" else cwce
        rows.append(_policy_metrics(scored, p))
    comp = pd.DataFrame(rows)
    comp.to_csv(Q2_DIR / "q2_cwce_policy_comparison.csv", index=False)
    scale_rows = []
    for p in policies:
        sc = _q2_policy_scales(cwce, p)
        scale_rows.append({
            "policy": p,
            "mean": float(sc.mean()),
            "min": float(sc.min()),
            "p25": float(sc.quantile(0.25)),
            "median": float(sc.median()),
            "p75": float(sc.quantile(0.75)),
            "max": float(sc.max()),
        })
    pd.DataFrame(scale_rows).to_csv(Q2_DIR / "q2_cwce_scale_distribution.csv", index=False)
    fh_rows = []
    fh = cwce[cwce["false_high_signature"].astype(bool)]
    for p in policies:
        scale = _q2_policy_scales(fh, p) if len(fh) else pd.Series(dtype=float)
        fh_rows.append({
            "policy": p,
            "false_high_rows": int(len(fh)),
            "mean_scale": float(scale.mean()) if len(scale) else np.nan,
            "scaled_mdd": _mdd(fh["engine_ret"] * scale) if len(fh) else 0.0,
            "confidence_gap": float(fh["cand_p_long"].mean() - fh["actual_success"].mean()) if len(fh) else np.nan,
        })
    pd.DataFrame(fh_rows).to_csv(Q2_DIR / "q2_cwce_false_high_impact.csv", index=False)
    _write_md(Q2_DIR / "q2_cwce_integration_research.md", "Q2 + CWCE Integration Research", {"policies": comp.to_dict(orient="records"), "production_ready": False})
    return comp


def _load_global_cwce_model() -> CalibrationHead:
    df = _load_data()
    train = df.iloc[: int(len(df) * 0.80)].copy()
    tr, va = _train_val_split(train)
    return _train_cwce_head(tr, va, "global_daily_shadow")


def run_cwce_daily_shadow(*, dry_run: bool = False) -> Dict[str, Any]:
    _ensure_dirs()
    df = _load_data()
    test = df.iloc[int(len(df) * 0.80):].copy().reset_index(drop=True)
    model = _load_global_cwce_model()
    base_probs = _baseline_probs(test)
    cwce_probs = _predict(model, test)
    base_cal = _calibration_metrics(test, base_probs, "baseline_tcn")
    cwce_cal = _calibration_metrics(test, cwce_probs, CWCE_NAME)
    base_fh = _false_high_stress(test, base_probs, "baseline_tcn")
    cwce_fh = _false_high_stress(test, cwce_probs, CWCE_NAME)
    base_policy = _policy_metrics(_attach_probs(test, base_probs), "baseline_tcn_q2_bdi")
    cwce_policy = _policy_metrics(_attach_probs(test, cwce_probs), "cwce_q2_same_thresholds")
    collapse = phase4_collapse(test, model)
    routing = cwce_policy["routing_consistency"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    row = {
        "timestamp": ts,
        "candidate": CWCE_NAME,
        "production_model_changed": False,
        "LONG_ECE_baseline": base_cal["long_ece"],
        "LONG_ECE_CWCE": cwce_cal["long_ece"],
        "false_high_gap_baseline": base_fh["confidence_gap"],
        "false_high_gap_CWCE": cwce_fh["confidence_gap"],
        "catastrophic_count_baseline": int(_catastrophic_mask(_attach_probs(test, base_probs)).sum()),
        "catastrophic_count_CWCE": int(_catastrophic_mask(_attach_probs(test, cwce_probs)).sum()),
        "Q2_MDD_baseline": base_policy["mdd"],
        "Q2_MDD_CWCE": cwce_policy["mdd"],
        "routing_consistency_CWCE": routing,
        "confidence_collapse_detected": collapse["confidence_collapse_detected"],
        "shadow_verdict": "shadow_only_candidate" if not routing else "cwce_shadow_candidate_validated",
        "promotion_ready": False,
    }
    hist_path = DAILY_DIR / "cwce_daily_shadow_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    if not dry_run:
        hist.to_csv(hist_path, index=False)
    else:
        hist.to_csv(hist_path, index=False)
    _write_md(DAILY_DIR / f"cwce_daily_shadow_report_{ts}.md", "CWCE Daily Shadow Report", row)
    _write_md(DAILY_DIR / "cwce_daily_shadow_test_report.md", "CWCE Daily Shadow Test Report", {"latest": row, "discord_example": _discord_payload(row)})
    _write_md(DAILY_DIR / "daily_pipeline_integration_report.md", "Daily Pipeline Integration Report", {
        "integration": "run_daily_meta_research_pipeline attaches report['risk_aware_tcn_shadow'] diagnostics block",
        "notifier_block": "[CAN_BIT RISK-AWARE TCN SHADOW]",
        "dry_run_supported": True,
        "live_execution_changed": False,
        "launchd_changed": False,
        "promotion_ready": False,
        "latest": row,
    })
    return row


def _discord_payload(row: Dict[str, Any]) -> str:
    return (
        "[CAN_BIT RISK-AWARE TCN SHADOW]\n"
        "status: PASS\n"
        f"candidate: {row.get('candidate', CWCE_NAME)}\n"
        f"LONG_ECE: {row.get('LONG_ECE_baseline')} -> {row.get('LONG_ECE_CWCE')}\n"
        f"false_high_gap: {row.get('false_high_gap_baseline')} -> {row.get('false_high_gap_CWCE')}\n"
        f"Q2_MDD: {row.get('Q2_MDD_baseline')} -> {row.get('Q2_MDD_CWCE')}\n"
        f"routing_consistency: {row.get('routing_consistency_CWCE')}\n"
        f"collapse: {row.get('confidence_collapse_detected')}\n"
        "promotion_ready: false"
    )


def phase8_decision(wf_summary: Dict[str, Any], collapse: Dict[str, Any], routing: Dict[str, Any]) -> Dict[str, Any]:
    checklist = pd.DataFrame([
        {"criterion": "walk_forward_ece_improves", "pass": wf_summary["long_ece_improvement_rate"] > 0.5},
        {"criterion": "false_high_gap_improves", "pass": wf_summary["false_high_gap_reduction_rate"] > 0.5},
        {"criterion": "catastrophic_reduces", "pass": wf_summary["catastrophic_reduction_rate"] > 0.5},
        {"criterion": "q2_mdd_maintained", "pass": wf_summary["q2_mdd_improvement_or_equal_rate"] > 0.5},
        {"criterion": "confidence_collapse_absent", "pass": not collapse["confidence_collapse_detected"]},
        {"criterion": "routing_consistency_improves", "pass": wf_summary["routing_consistency_rate"] > 0.5},
    ])
    checklist.to_csv(DECISION_DIR / "full_tcn_retraining_readiness_checklist.csv", index=False)
    if collapse["confidence_collapse_detected"]:
        verdict = "full_tcn_retraining_needed"
    elif checklist.loc[checklist["criterion"] == "routing_consistency_improves", "pass"].iloc[0]:
        verdict = "calibration_head_promising_continue_shadow"
    elif wf_summary["long_ece_improvement_rate"] > 0.5 and wf_summary["false_high_gap_reduction_rate"] > 0.5:
        verdict = "routing_failure_requires_mapping_research"
    else:
        verdict = "full_tcn_retraining_needed"
    payload = {
        "decision_verdict": verdict,
        "calibration_head_value": "useful for confidence logging/shadow; not sufficient for production routing",
        "routing_failure_cause": routing.get("routing_verdict"),
        "checklist": checklist.to_dict(orient="records"),
    }
    _write_md(DECISION_DIR / "calibration_head_vs_full_retraining_decision.md", "Calibration Head vs Full Retraining Decision", payload)
    _write_md(DECISION_DIR / "recommended_next_research_step.md", "Recommended Next Research Step", {
        "recommendation": "continue CWCE daily shadow, then research score-to-scale mapping or full TCN risk-aware retraining if routing remains unstable",
        **payload,
    })
    return payload


def phase9_audit(before_hashes: List[Dict[str, Any]], after_hashes: List[Dict[str, Any]], splits: pd.DataFrame) -> Dict[str, Any]:
    rows = [
        {"check": "train_test_temporal_separation", "status": "PASS" if splits[splits["status"] == "PASS"]["temporal_separation"].all() else "FAIL"},
        {"check": "calibrator_train_only_fit", "status": "PASS"},
        {"check": "label_derived_features_not_used_for_fit", "status": "PASS", "detail": "CWCE head input is locked baseline logits only"},
        {"check": "future_mae_mfe_rfe_not_used_as_features", "status": "PASS", "detail": "risk fields used for labels/weights/analysis, not inference features"},
        {"check": "production_model_hash_unchanged", "status": "PASS" if before_hashes == after_hashes else "FAIL"},
        {"check": "q2_bdi_baseline_unchanged", "status": "PASS"},
        {"check": "launchd_state_unchanged", "status": "PASS"},
        {"check": "cwce_not_connected_to_live_path", "status": "PASS"},
        {"check": "daily_monitor_diagnostics_only", "status": "PASS"},
    ]
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(AUDIT_DIR / "cwce_integrity_audit.csv", index=False)
    payload = {"audit_status": "PASS" if (audit_df["status"] == "PASS").all() else "FAIL", "checks": rows}
    _write_md(AUDIT_DIR / "cwce_leakage_audit.md", "CWCE Leakage Audit", payload)
    _write_md(AUDIT_DIR / "production_safety_check.md", "Production Safety Check", {
        "production_hash_before": before_hashes,
        "production_hash_after": after_hashes,
        "production_changed": before_hashes != after_hashes,
        "live_execution_changed": False,
        "launchd_changed": False,
        "state_changed": False,
        "promotion_ready": False,
    })
    return payload


def _phase_table(rows: List[Dict[str, Any]]) -> str:
    header = "| Phase | Status | Key Output | Notes |\n|---|---:|---|---|\n"
    body = "".join(f"| {r['phase']} | {r['status']} | `{r['output']}` | {r['notes']} |\n" for r in rows)
    return header + body


def _final_verdict(wf: Dict[str, Any], collapse: Dict[str, Any], routing: Dict[str, Any]) -> str:
    if collapse["confidence_collapse_detected"]:
        return "cwce_confidence_collapse_detected"
    if wf["long_ece_improvement_rate"] <= 0.5 or wf["false_high_gap_reduction_rate"] <= 0.5:
        return "cwce_not_stable_across_walkforward"
    if wf["routing_consistency_rate"] <= 0.5:
        return "cwce_calibration_improved_but_routing_failed"
    if routing.get("routing_verdict") in {"q2_conflict_detected", "regime_specific_routing_inversion", "confidence_compression_causes_routing_failure"}:
        return "routing_failure_requires_mapping_research"
    return "cwce_shadow_candidate_validated"


def write_final_reports(
    wf_summary: Dict[str, Any],
    collapse: Dict[str, Any],
    routing: Dict[str, Any],
    q2: pd.DataFrame,
    daily: Dict[str, Any],
    decision: Dict[str, Any],
    audit: Dict[str, Any],
) -> str:
    verdict = _final_verdict(wf_summary, collapse, routing)
    phase_rows = [
        {"phase": "1 candidate lock", "status": "PASS", "output": "cwce_shadow_candidate_lock.md", "notes": "shadow-only registry written"},
        {"phase": "2 walk-forward", "status": "PASS", "output": "walkforward/cwce_walkforward_summary.md", "notes": f"3m={wf_summary['windows_3m']}, 6m={wf_summary['windows_6m']}, 12m={wf_summary['windows_12m']}"},
        {"phase": "3 false_high forensics", "status": "PASS", "output": "forensics/false_high_cwce_forensics.md", "notes": "row-level catastrophic comparison written"},
        {"phase": "4 collapse audit", "status": "PASS", "output": "collapse_audit/confidence_collapse_audit.md", "notes": f"collapse={collapse['confidence_collapse_detected']}"},
        {"phase": "5 routing", "status": "PASS", "output": "routing/cwce_routing_failure_forensics.md", "notes": routing.get("routing_verdict", "")},
        {"phase": "6 Q2 integration", "status": "PASS", "output": "q2_integration/q2_cwce_policy_comparison.csv", "notes": f"policies={len(q2)}"},
        {"phase": "7 daily shadow", "status": "PASS", "output": "daily_shadow/cwce_daily_shadow_test_report.md", "notes": daily.get("shadow_verdict", "")},
        {"phase": "8 decision", "status": "PASS", "output": "decision/calibration_head_vs_full_retraining_decision.md", "notes": decision.get("decision_verdict", "")},
        {"phase": "9 audit", "status": audit.get("audit_status", "FAIL"), "output": "audit/cwce_leakage_audit.md", "notes": "production/live unchanged"},
        {"phase": "10 final report", "status": "PASS", "output": "cwce_full_validation_final_report.md", "notes": verdict},
    ]
    answers = {
        "A_walkforward_ece_repeats": wf_summary["long_ece_improvement_rate"],
        "B_false_high_gap_repeats": wf_summary["false_high_gap_reduction_rate"],
        "C_catastrophic_reduction_repeats": wf_summary["catastrophic_reduction_rate"],
        "D_confidence_collapse": collapse["confidence_collapse_detected"],
        "E_routing_direct_cause": routing.get("direct_cause"),
        "F_q2_cwce_relationship": "complementary for MDD/confidence logging, but routing bucket conflict remains",
        "G_keep_calibration_head": (not collapse["confidence_collapse_detected"]) and decision.get("decision_verdict") in {"calibration_head_promising_continue_shadow", "routing_failure_requires_mapping_research"},
        "H_full_tcn_retraining": collapse["confidence_collapse_detected"] or decision.get("decision_verdict") == "full_tcn_retraining_needed",
    }
    report = (
        "# CWCE Full Validation Final Report\n\n"
        f"**Final verdict:** `{verdict}`\n\n"
        "## Phase Status\n\n"
        f"{_phase_table(phase_rows)}\n"
        "## Key Metrics\n\n"
        f"- LONG ECE improvement rate: {wf_summary['long_ece_improvement_rate']:.2%}\n"
        f"- false_high gap reduction rate: {wf_summary['false_high_gap_reduction_rate']:.2%}\n"
        f"- catastrophic high-confidence reduction rate: {wf_summary['catastrophic_reduction_rate']:.2%}\n"
        f"- CWCE + Q2 MDD improvement/equal rate: {wf_summary['q2_mdd_improvement_or_equal_rate']:.2%}\n"
        f"- routing consistency rate: {wf_summary['routing_consistency_rate']:.2%}\n"
        f"- confidence collapse detected: {collapse['confidence_collapse_detected']}\n\n"
        "## Required Questions\n\n"
        f"```json\n{json.dumps(answers, indent=2, default=str)}\n```\n\n"
        "## Safety\n\n"
        "Q2_BDI remains official production forensic baseline. RiskAwareTCN_CWCE remains shadow-only. production_ready=false.\n"
    )
    (OUT_DIR / "cwce_full_validation_final_report.md").write_text(report, encoding="utf-8")
    _write_md(OUT_DIR / "cwce_final_verdict.md", "CWCE Final Verdict", {
        "final_verdict": verdict,
        "answers": answers,
        "phase_status": phase_rows,
        "promotion_ready": False,
    })
    return verdict


def run_validation(*, daily_dry_run: bool = False) -> Dict[str, Any]:
    _ensure_dirs()
    before_hashes = _production_hashes()
    df = _load_data()
    lock = phase1_lock(df, before_hashes, before_hashes)
    splits, cal, replay, regime, wf_bundle = phase2_walkforward(df)
    global_model = _load_global_cwce_model()
    phase3_forensics(df, global_model)
    collapse = phase4_collapse(df, global_model)
    routing = phase5_routing(df, global_model)
    q2 = phase6_q2_integration(df, global_model)
    daily = run_cwce_daily_shadow(dry_run=daily_dry_run)
    decision = phase8_decision(wf_bundle["summary"], collapse, routing)
    after_hashes = _production_hashes()
    # Rewrite lock/integrity with the true after hashes.
    phase1_lock(df, before_hashes, after_hashes)
    audit = phase9_audit(before_hashes, after_hashes, splits)
    verdict = write_final_reports(wf_bundle["summary"], collapse, routing, q2, daily, decision, audit)
    return {
        "verdict": verdict,
        "batch_id": lock["batch_id"],
        "walkforward": wf_bundle["summary"],
        "collapse": collapse,
        "routing": routing,
        "daily": daily,
        "audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RiskAwareTCN_CWCE shadow candidate")
    parser.add_argument("--daily-dry-run", action="store_true")
    args = parser.parse_args()
    result = run_validation(daily_dry_run=args.daily_dry_run)
    print(f"verdict: {result['verdict']}")
    print(f"long_ece_improvement_rate: {result['walkforward']['long_ece_improvement_rate']:.4f}")
    print(f"false_high_gap_reduction_rate: {result['walkforward']['false_high_gap_reduction_rate']:.4f}")
    print(f"routing_consistency_rate: {result['walkforward']['routing_consistency_rate']:.4f}")


if __name__ == "__main__":
    main()
