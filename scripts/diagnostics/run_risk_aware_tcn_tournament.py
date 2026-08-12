"""
Risk-aware TCN retraining tournament (diagnostics only).

Production TCN/Q2/live routing are not changed. This tournament locks current
TCN inference outputs as the baseline and trains research-only calibration
heads over those outputs with different risk-aware losses/weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.analyze_tcn_confidence_calibration import (
    _brier,
    _mdd,
    _prepare_dataset,
)
from scripts.diagnostics.run_forward_meta_shadow_monitor import _bucket_summary, _routing_consistency

OUT_DIR = Path("data/diagnostics/risk_aware_tcn_tournament")
MODEL_DIR = OUT_DIR / "candidate_models"
CURVE_DIR = OUT_DIR / "training_curves"
POSITION_SIZE = 0.05
SEED = 42
EPOCHS = 180
LR = 0.03
PATIENCE = 25
CLASS_NAMES = ["FLAT", "LONG", "SHORT"]
CANDIDATES = [
    "baseline_cross_entropy",
    "focal_loss",
    "class_weighted_ce",
    "drawdown_weighted_loss",
    "RFE_weighted_loss",
    "false_high_penalty_loss",
    "entropy_aware_loss",
    "confidence_calibration_loss",
    "composite_risk_aware_loss",
]


@dataclass
class CandidateResult:
    name: str
    model_path: str
    curve_path: str
    best_val_loss: float
    epochs_trained: int
    train_loss_final: float
    val_loss_final: float


class CalibrationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _set_seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["df_idx", "entry_ts"], na_position="last").reset_index(drop=True)
    n = len(df)
    i1, i2 = int(n * 0.60), int(n * 0.80)
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def _locked_logits(df: pd.DataFrame) -> np.ndarray:
    p = df[["p_flat", "p_long", "p_short"]].astype(float).to_numpy()
    p = np.clip(p, 1e-5, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)


def _baseline_probs(df: pd.DataFrame) -> np.ndarray:
    p = df[["p_flat", "p_long", "p_short"]].astype(float).to_numpy()
    p = np.clip(p, 1e-6, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _ece_full(conf: pd.Series, success: pd.Series, n_bins: int = 10) -> float:
    tmp = pd.DataFrame({
        "confidence": np.asarray(conf, dtype=float),
        "success": np.asarray(success, dtype=float),
    }).dropna()
    if tmp.empty:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    labels = range(n_bins)
    tmp["bin"] = pd.cut(tmp["confidence"].clip(0, 1), bins=bins, labels=labels, include_lowest=True)
    err = 0.0
    for _, g in tmp.groupby("bin", observed=False):
        if g.empty:
            continue
        err += (len(g) / len(tmp)) * abs(float(g["confidence"].mean()) - float(g["success"].mean()))
    return float(err)


def _risk_aware_labels(df: pd.DataFrame) -> np.ndarray:
    y = np.zeros(len(df), dtype=np.int64)
    good = df["binary_good_trade"].astype(bool).to_numpy()
    direction = df["direction"].astype(str).to_numpy()
    y[good & (direction == "LONG")] = 1
    y[good & (direction == "SHORT")] = 2
    return y


def _sample_weights(df: pd.DataFrame, candidate: str) -> np.ndarray:
    w = np.ones(len(df), dtype=np.float32)
    bad = df["binary_bad_trade"].astype(bool).to_numpy()
    high_conf = df["predicted_confidence"].astype(float).to_numpy() >= 0.55
    false_high = df["false_high_signature"].astype(bool).to_numpy()
    mae = np.abs(pd.to_numeric(df["mae"], errors="coerce").fillna(0).to_numpy())
    rfe = df["rfe_flag"].astype(bool).to_numpy()
    low_entropy = (df["entropy"].astype(float).to_numpy() <= 0.90)
    overext = pd.to_numeric(df["confidence_overextension"], errors="coerce").fillna(0).to_numpy() > 0
    vol_exp = df["vol_regime"].astype(str).eq("vol_expansion").to_numpy()

    if candidate == "drawdown_weighted_loss":
        w *= np.clip(1.0 + mae * 180.0, 1.0, 5.0)
    elif candidate == "RFE_weighted_loss":
        w *= np.where(rfe, 3.5, 1.0)
    elif candidate == "false_high_penalty_loss":
        w *= np.where(false_high & bad, 4.0, 1.0)
    elif candidate == "entropy_aware_loss":
        w *= np.where(low_entropy & bad, 3.2, 1.0)
    elif candidate == "confidence_calibration_loss":
        w *= np.where(high_conf & bad, 3.0, 1.0)
    elif candidate == "composite_risk_aware_loss":
        w *= np.where(false_high & bad, 3.0, 1.0)
        w *= np.where(rfe, 2.0, 1.0)
        w *= np.where((low_entropy | overext | vol_exp) & bad, 1.8, 1.0)
        w *= np.clip(1.0 + mae * 120.0, 1.0, 4.0)
    return np.clip(w, 0.25, 8.0).astype(np.float32)


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(float)
    total = counts.sum()
    weights = total / (3.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def _loss_fn(logits: torch.Tensor, y: torch.Tensor, sample_w: torch.Tensor, candidate: str, class_w: torch.Tensor) -> torch.Tensor:
    ce_weight = class_w if candidate == "class_weighted_ce" else None
    ce = F.cross_entropy(logits, y, reduction="none", weight=ce_weight)
    if candidate == "focal_loss":
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** 2.0) * ce
    else:
        loss = ce

    if candidate in {
        "drawdown_weighted_loss",
        "RFE_weighted_loss",
        "false_high_penalty_loss",
        "entropy_aware_loss",
        "confidence_calibration_loss",
        "composite_risk_aware_loss",
    }:
        loss = loss * sample_w

    if candidate in {"confidence_calibration_loss", "composite_risk_aware_loss"}:
        prob = F.softmax(logits, dim=1)
        one_hot = F.one_hot(y, num_classes=3).float()
        brier = ((prob - one_hot) ** 2).sum(dim=1)
        loss = loss + 0.35 * brier * sample_w
    return loss.mean()


def _train_candidate(candidate: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[CandidateResult, CalibrationHead]:
    _set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)

    model = CalibrationHead()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    x_tr = torch.tensor(_locked_logits(train_df), dtype=torch.float32)
    y_tr_np = _risk_aware_labels(train_df)
    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    w_tr = torch.tensor(_sample_weights(train_df, candidate), dtype=torch.float32)

    x_va = torch.tensor(_locked_logits(val_df), dtype=torch.float32)
    y_va_np = _risk_aware_labels(val_df)
    y_va = torch.tensor(y_va_np, dtype=torch.long)
    w_va = torch.tensor(_sample_weights(val_df, candidate), dtype=torch.float32)
    cls_w = _class_weights(y_tr_np)

    best_loss = math.inf
    best_state: Dict[str, Any] | None = None
    stale = 0
    curves: List[Dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        tr_logits = model(x_tr)
        tr_loss = _loss_fn(tr_logits, y_tr, w_tr, candidate, cls_w)
        tr_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(x_va)
            va_loss = _loss_fn(va_logits, y_va, w_va, candidate, cls_w)
        curves.append({"epoch": epoch, "train_loss": float(tr_loss.item()), "val_loss": float(va_loss.item())})
        if float(va_loss.item()) < best_loss - 1e-5:
            best_loss = float(va_loss.item())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    curve_path = CURVE_DIR / f"{candidate}_training_curve.csv"
    pd.DataFrame(curves).to_csv(curve_path, index=False)
    model_path = MODEL_DIR / f"{candidate}_calibration_head.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "candidate": candidate,
        "class_names": CLASS_NAMES,
        "input": "locked_baseline_tcn_logits",
        "production_ready": False,
    }, model_path)
    joblib.dump({"candidate": candidate, "curve": curves[-5:]}, MODEL_DIR / f"{candidate}_metadata.joblib")
    last = curves[-1]
    return CandidateResult(
        name=candidate,
        model_path=str(model_path),
        curve_path=str(curve_path),
        best_val_loss=best_loss,
        epochs_trained=len(curves),
        train_loss_final=float(last["train_loss"]),
        val_loss_final=float(last["val_loss"]),
    ), model


def _predict(model: CalibrationHead | None, df: pd.DataFrame) -> np.ndarray:
    if model is None:
        return _baseline_probs(df)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(_locked_logits(df), dtype=torch.float32))
        return F.softmax(logits, dim=1).cpu().numpy()


def _attach_probs(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["cand_p_flat"] = probs[:, 0]
    out["cand_p_long"] = probs[:, 1]
    out["cand_p_short"] = probs[:, 2]
    out["cand_max_proba"] = probs.max(axis=1)
    out["cand_predicted_direction"] = pd.Series(np.argmax(probs, axis=1)).map({0: "FLAT", 1: "LONG", 2: "SHORT"}).to_numpy()
    return out


def _calibration_metrics(df: pd.DataFrame, probs: np.ndarray, candidate: str) -> Dict[str, Any]:
    p_long = probs[:, 1]
    long_mask = df["direction"].astype(str).eq("LONG").to_numpy()
    success = df["actual_success"].astype(int).to_numpy()
    long_conf = pd.Series(p_long[long_mask])
    long_success = pd.Series(success[long_mask])
    max_conf = pd.Series(probs.max(axis=1))
    return {
        "candidate": candidate,
        "rows": int(len(df)),
        "long_ece": _ece_full(long_conf, long_success),
        "global_ece": _ece_full(max_conf, pd.Series(success)),
        "brier_score": _brier(max_conf, pd.Series(success)),
        "calibration_error": float(abs(max_conf.mean() - success.mean())),
        "overconfidence_score": float(np.maximum(max_conf.to_numpy() - success, 0).mean()),
        "underconfidence_score": float(np.maximum(success - max_conf.to_numpy(), 0).mean()),
        "log_loss_proxy": float(log_loss(_risk_aware_labels(df), np.clip(probs, 1e-6, 1.0), labels=[0, 1, 2])),
        "mean_confidence": float(max_conf.mean()),
        "success_rate": float(success.mean()),
    }


def _regime_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "high_vol": df["vol_bucket"].astype(str).eq("high"),
        "low_entropy": df["confidence_regime"].astype(str).eq("low_entropy"),
        "false_high_signature": df["false_high_signature"].astype(bool),
        "confidence_overextension": df["confidence_overextension"].astype(float) > 0,
        "vol_expansion": df["vol_regime"].astype(str).eq("vol_expansion"),
        "trend_transition": df["trend_transition"].astype(bool),
    }


def _regime_metrics(df: pd.DataFrame, probs: np.ndarray, candidate: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scored = _attach_probs(df, probs)
    for regime, mask in _regime_masks(scored).items():
        sub = scored[mask].copy()
        if sub.empty:
            rows.append({"candidate": candidate, "regime": regime, "rows": 0})
            continue
        long_mask = sub["direction"].astype(str).eq("LONG").to_numpy()
        rows.append({
            "candidate": candidate,
            "regime": regime,
            "rows": int(len(sub)),
            "long_ece": _ece_full(sub.loc[long_mask, "cand_p_long"], sub.loc[long_mask, "actual_success"]) if long_mask.any() else np.nan,
            "brier_score": _brier(sub["cand_max_proba"], sub["actual_success"]),
            "calibration_error": float(abs(sub["cand_max_proba"].mean() - sub["actual_success"].mean())),
            "overconfidence_score": float(np.maximum(sub["cand_max_proba"].to_numpy() - sub["actual_success"].to_numpy(), 0).mean()),
            "high_conf_failure_rate": float(((sub["cand_max_proba"] >= 0.55) & (sub["actual_success"] == 0)).mean()),
            "success_rate": float(sub["actual_success"].mean()),
            "mean_confidence": float(sub["cand_max_proba"].mean()),
        })
    return rows


def _false_high_stress(df: pd.DataFrame, probs: np.ndarray, candidate: str) -> Dict[str, Any]:
    scored = _attach_probs(df, probs)
    fh = scored[scored["false_high_signature"].astype(bool)].copy()
    if fh.empty:
        return {"candidate": candidate, "rows": 0}
    return {
        "candidate": candidate,
        "rows": int(len(fh)),
        "mean_p_long": float(fh["cand_p_long"].mean()),
        "actual_success": float(fh["actual_success"].mean()),
        "confidence_gap": float(fh["cand_p_long"].mean() - fh["actual_success"].mean()),
        "catastrophic_trade_frequency": float(((fh["mae"] <= -0.008) | fh["rfe_flag"].astype(bool)).mean()),
        "rfe_rate": float(fh["rfe_flag"].astype(bool).mean()),
        "mae_mean": float(fh["mae"].mean()),
        "drawdown_contribution": _mdd(fh["engine_ret"]),
        "confidence_saturation": float((fh["cand_p_long"] >= 0.60).mean()),
    }


def _risk_bucket_monotonicity(df: pd.DataFrame, scores: np.ndarray) -> Tuple[bool, bool]:
    if len(df) < 12:
        return False, False
    bdf, mono = _bucket_summary(df, scores)
    return bool(mono.get("monotonic", False)), _routing_consistency(df, scores)


def _replay_metrics(df: pd.DataFrame, probs: np.ndarray, candidate: str) -> List[Dict[str, Any]]:
    scored = _attach_probs(df, probs)
    rows: List[Dict[str, Any]] = []
    q2_scale = scored["q2_bdi_scale"].astype(float).copy()
    cand_conf = scored["cand_max_proba"].astype(float)
    same_dir = (
        ((scored["direction"] == "LONG") & (scored["cand_predicted_direction"] == "LONG"))
        | ((scored["direction"] == "SHORT") & (scored["cand_predicted_direction"] == "SHORT"))
    )
    candidate_only_scale = pd.Series(np.where(same_dir & (cand_conf >= 0.50), 1.0, 0.15), index=scored.index)
    candidate_q2_scale = q2_scale.copy()
    low_conf_or_mismatch = ~(same_dir & (cand_conf >= 0.50))
    candidate_q2_scale.loc[low_conf_or_mismatch] = (candidate_q2_scale.loc[low_conf_or_mismatch] * 0.5).clip(0.05, 1.0)
    q2_risk_score = 1.0 - q2_scale
    cand_risk_score = 1.0 - cand_conf

    configs = [
        ("baseline_tcn_q2_bdi", q2_scale, q2_risk_score),
        ("candidate_tcn_q2_bdi", candidate_q2_scale, cand_risk_score),
        ("candidate_tcn_only", candidate_only_scale, cand_risk_score),
    ]
    for replay, scale, risk_score in configs:
        mono, routing = _risk_bucket_monotonicity(scored, risk_score.to_numpy(dtype=float))
        scaled_ret = scored["engine_ret"].fillna(0) * scale.fillna(1)
        rows.append({
            "candidate": candidate,
            "replay": replay,
            "mdd": _mdd(scaled_ret),
            "net": float((scaled_ret * POSITION_SIZE).sum()),
            "rfe": int(scored["rfe_flag"].astype(bool).sum()),
            "false_high_count": int(scored["false_high_signature"].astype(bool).sum()),
            "preservation": float((scale > 0.05).mean()),
            "routing_consistency": routing,
            "bucket_monotonicity": mono,
            "mean_scale": float(scale.mean()),
        })
    return rows


def _shadow_forward(df: pd.DataFrame, probs: np.ndarray, candidate: str, baseline_probs: np.ndarray) -> Dict[str, Any]:
    scored = _attach_probs(df, probs)
    baseline_max = baseline_probs.max(axis=1)
    cand_max = probs.max(axis=1)
    fh = scored["false_high_signature"].astype(bool)
    return {
        "candidate": candidate,
        "rows": int(len(df)),
        "proba_drift_mean": float(np.mean(cand_max - baseline_max)),
        "proba_drift_abs_mean": float(np.mean(np.abs(cand_max - baseline_max))),
        "calibration_drift": float(_ece_full(pd.Series(cand_max), scored["actual_success"]) - _ece_full(pd.Series(baseline_max), scored["actual_success"])),
        "false_high_mean_p_long": float(scored.loc[fh, "cand_p_long"].mean()) if fh.any() else np.nan,
        "false_high_confidence_gap": float(scored.loc[fh, "cand_p_long"].mean() - scored.loc[fh, "actual_success"].mean()) if fh.any() else np.nan,
        "regime_robustness_proxy": float(np.mean([m["calibration_error"] < 0.12 for m in _regime_metrics(df, probs, candidate) if m.get("rows", 0) >= 20])),
        "confidence_mean": float(cand_max.mean()),
        "confidence_std": float(cand_max.std()),
        "promotion_ready": False,
    }


def _baseline_snapshot(df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    probs = _baseline_probs(test_df)
    metrics = _calibration_metrics(test_df, probs, "production_tcn_locked_baseline")
    snapshot = {
        "production_baseline": "Q2_BDI discrete M3",
        "tcn_baseline": "locked current production TCN inference outputs from meta_dataset_v2",
        "rows_total": int(len(df)),
        "test_rows": int(len(test_df)),
        "feature_schema": ["p_flat", "p_long", "p_short"],
        "risk_feature_schema_reference": sorted([c for c in df.columns if c in {
            "mae", "mfe", "rfe_flag", "false_high_signature", "confidence_overextension",
            "entropy", "vol_regime", "trend_transition", "q2_bdi_scale",
        }]),
        "inference_outputs": {
            "p_long_mean": float(test_df["p_long"].mean()),
            "p_short_mean": float(test_df["p_short"].mean()),
            "p_flat_mean": float(test_df["p_flat"].mean()),
        },
        "calibration_metrics": metrics,
        "production_model_overwritten": False,
        "live_routing_changed": False,
    }
    return snapshot


def _loss_design() -> List[Dict[str, Any]]:
    return [
        {"candidate": "baseline_cross_entropy", "target": "reference CE calibration head", "loss_change": "none"},
        {"candidate": "focal_loss", "target": "hard confidence failures", "loss_change": "focal gamma=2"},
        {"candidate": "class_weighted_ce", "target": "FLAT/LONG/SHORT imbalance", "loss_change": "inverse class weights"},
        {"candidate": "drawdown_weighted_loss", "target": "large MAE/drawdown failures", "loss_change": "sample weight from abs(MAE)"},
        {"candidate": "RFE_weighted_loss", "target": "risk_force_exit failures", "loss_change": "RFE sample weight boost"},
        {"candidate": "false_high_penalty_loss", "target": "LONG/up/high_vol/low_entropy false_high failures", "loss_change": "false_high bad-trade weight boost"},
        {"candidate": "entropy_aware_loss", "target": "low entropy overconfidence", "loss_change": "low_entropy bad-trade weight boost"},
        {"candidate": "confidence_calibration_loss", "target": "high-confidence wrong calls", "loss_change": "CE + Brier calibration penalty"},
        {"candidate": "composite_risk_aware_loss", "target": "combined false_high/RFE/drawdown/regime traps", "loss_change": "combined risk weights + Brier"},
    ]


def _failure_mapping() -> pd.DataFrame:
    src = Path("data/diagnostics/tcn_confidence_calibration/failure_cluster_loss_mapping.csv")
    if src.exists():
        return pd.read_csv(src)
    return pd.DataFrame(_loss_design())


def _final_verdict(
    cal: pd.DataFrame,
    fh: pd.DataFrame,
    replay: pd.DataFrame,
    shadow: pd.DataFrame,
    baseline: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    base_long_ece = float(baseline["calibration_metrics"]["long_ece"])
    base_fh_gap = float(fh[fh["candidate"] == "production_tcn_locked_baseline"]["confidence_gap"].iloc[0])
    base_cat = float(fh[fh["candidate"] == "production_tcn_locked_baseline"]["catastrophic_trade_frequency"].iloc[0])
    q2_base_mdd = float(replay[(replay["candidate"] == "production_tcn_locked_baseline") & (replay["replay"] == "baseline_tcn_q2_bdi")]["mdd"].iloc[0])

    rows: List[Dict[str, Any]] = []
    for cand in cal["candidate"].unique():
        if cand == "production_tcn_locked_baseline":
            continue
        c = cal[cal["candidate"] == cand].iloc[0]
        f = fh[fh["candidate"] == cand].iloc[0]
        q = replay[(replay["candidate"] == cand) & (replay["replay"] == "candidate_tcn_q2_bdi")].iloc[0]
        s = shadow[shadow["candidate"] == cand].iloc[0]
        success = {
            "long_ece_improved": float(c["long_ece"]) < base_long_ece,
            "false_high_gap_reduced": float(f["confidence_gap"]) < base_fh_gap,
            "catastrophic_reduced": float(f["catastrophic_trade_frequency"]) <= base_cat,
            "q2_mdd_maintained_or_improved": float(q["mdd"]) >= q2_base_mdd,
            "routing_consistency": bool(q["routing_consistency"]),
            "regime_robustness": float(s["regime_robustness_proxy"]) >= 0.5,
            "confidence_not_collapsed": 0.35 <= float(c["mean_confidence"]) <= 0.75,
        }
        score = sum(success.values())
        rows.append({
            "candidate": cand,
            "success_count": score,
            **success,
            "long_ece": float(c["long_ece"]),
            "false_high_gap": float(f["confidence_gap"]),
            "q2_mdd": float(q["mdd"]),
            "mean_confidence": float(c["mean_confidence"]),
        })
    decision = pd.DataFrame(rows).sort_values(["success_count", "long_ece"], ascending=[False, True])
    best = decision.iloc[0].to_dict() if len(decision) else {}
    if not best:
        return "baseline_tcn_still_best", {"best_candidate": {}}
    if best["candidate"] == "false_high_penalty_loss" and best["false_high_gap_reduced"]:
        verdict = "false_high_penalty_effective"
    elif best["success_count"] >= 6:
        verdict = "risk_aware_training_promising"
    elif best["long_ece_improved"] and best["false_high_gap_reduced"]:
        verdict = "calibration_improved"
    elif best["catastrophic_reduced"] and best["false_high_gap_reduced"]:
        verdict = "catastrophic_overconfidence_reduced"
    else:
        verdict = "production_not_ready"
    if best["success_count"] >= 5 and verdict != "production_not_ready":
        verdict = "shadow_only_candidate"
    return verdict, {"best_candidate": best, "candidate_decision_table": decision.to_dict(orient="records")}


def _write_md(path: Path, title: str, payload: Dict[str, Any]) -> None:
    path.write_text(
        f"# {title}\n\n"
        "Diagnostics/research only. Production TCN, Q2_BDI, live execution, and launchd remain unchanged.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )


def run_tournament() -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = _prepare_dataset()
    train_df, val_df, test_df = _time_split(df)
    baseline = _baseline_snapshot(df, test_df)
    _write_md(OUT_DIR / "baseline_tcn_snapshot.md", "Baseline TCN Snapshot", baseline)
    pd.DataFrame([baseline["calibration_metrics"]]).to_csv(OUT_DIR / "baseline_calibration_metrics.csv", index=False)

    design = _loss_design()
    _write_md(OUT_DIR / "loss_design_specification.md", "Loss Design Specification", {"candidates": design})
    failure_mapping = _failure_mapping()
    failure_mapping.to_csv(OUT_DIR / "failure_cluster_loss_mapping.csv", index=False)

    training_rows: List[Dict[str, Any]] = []
    registry_rows: List[Dict[str, Any]] = []
    all_cal: List[Dict[str, Any]] = []
    all_regime: List[Dict[str, Any]] = []
    all_fh: List[Dict[str, Any]] = []
    all_replay: List[Dict[str, Any]] = []
    all_shadow: List[Dict[str, Any]] = []

    baseline_probs = _baseline_probs(test_df)
    baseline_name = "production_tcn_locked_baseline"
    all_cal.append(_calibration_metrics(test_df, baseline_probs, baseline_name))
    all_regime.extend(_regime_metrics(test_df, baseline_probs, baseline_name))
    all_fh.append(_false_high_stress(test_df, baseline_probs, baseline_name))
    all_replay.extend(_replay_metrics(test_df, baseline_probs, baseline_name))
    all_shadow.append(_shadow_forward(test_df, baseline_probs, baseline_name, baseline_probs))

    for candidate in CANDIDATES:
        result, model = _train_candidate(candidate, train_df, val_df)
        probs = _predict(model, test_df)
        training_rows.append(result.__dict__)
        registry_rows.append({
            "candidate": candidate,
            "artifact_path": result.model_path,
            "curve_path": result.curve_path,
            "base_tcn_overwritten": False,
            "live_enabled": False,
            "feature_schema": "locked_baseline_tcn_logits",
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        })
        all_cal.append(_calibration_metrics(test_df, probs, candidate))
        all_regime.extend(_regime_metrics(test_df, probs, candidate))
        all_fh.append(_false_high_stress(test_df, probs, candidate))
        all_replay.extend(_replay_metrics(test_df, probs, candidate))
        all_shadow.append(_shadow_forward(test_df, probs, candidate, baseline_probs))

    training_df = pd.DataFrame(training_rows)
    registry_df = pd.DataFrame(registry_rows)
    cal_df = pd.DataFrame(all_cal)
    regime_df = pd.DataFrame(all_regime)
    fh_df = pd.DataFrame(all_fh)
    replay_df = pd.DataFrame(all_replay)
    shadow_df = pd.DataFrame(all_shadow)

    training_df.to_csv(OUT_DIR / "training_summary.csv", index=False)
    registry_df.to_csv(OUT_DIR / "candidate_model_registry.csv", index=False)
    cal_df.to_csv(OUT_DIR / "candidate_calibration_comparison.csv", index=False)
    regime_df.to_csv(OUT_DIR / "regime_calibration_comparison.csv", index=False)
    fh_df.to_csv(OUT_DIR / "false_high_stress_test.csv", index=False)
    replay_df.to_csv(OUT_DIR / "q2_replay_candidate_comparison.csv", index=False)
    shadow_df.to_csv(OUT_DIR / "shadow_forward_candidate_monitor.csv", index=False)

    _write_md(
        OUT_DIR / "false_high_candidate_comparison.md",
        "False High Candidate Comparison",
        {
            "baseline_gap": float(fh_df[fh_df["candidate"] == baseline_name]["confidence_gap"].iloc[0]),
            "best_by_gap": fh_df.sort_values("confidence_gap").head(5).to_dict(orient="records"),
        },
    )
    _write_md(
        OUT_DIR / "replay_risk_comparison.md",
        "Replay Risk Comparison",
        {
            "baseline_q2": replay_df[(replay_df["candidate"] == baseline_name) & (replay_df["replay"] == "baseline_tcn_q2_bdi")].to_dict(orient="records"),
            "best_candidate_q2_mdd": replay_df[replay_df["replay"] == "candidate_tcn_q2_bdi"].sort_values("mdd", ascending=False).head(8).to_dict(orient="records"),
        },
    )
    _write_md(
        OUT_DIR / "candidate_forward_shadow_report.md",
        "Candidate Forward Shadow Report",
        {
            "best_regime_robustness": shadow_df.sort_values("regime_robustness_proxy", ascending=False).head(8).to_dict(orient="records"),
            "production_ready": False,
        },
    )
    verdict, verdict_payload = _final_verdict(cal_df, fh_df, replay_df, shadow_df, baseline)
    _write_md(
        OUT_DIR / "risk_aware_tcn_final_verdict.md",
        "Risk-Aware TCN Final Verdict",
        {
            "final_verdict": verdict,
            **verdict_payload,
            "baseline_locked": True,
            "production_model_replaced": False,
            "allowed_state": "shadow_only_candidate" if verdict == "shadow_only_candidate" else "research_only",
        },
    )
    return {
        "verdict": verdict,
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "best_candidate": verdict_payload.get("best_candidate", {}).get("candidate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk-aware TCN tournament diagnostics")
    parser.parse_args()
    result = run_tournament()
    print(f"verdict: {result['verdict']}")
    print(f"rows: {result['rows']}")
    print(f"test_rows: {result['test_rows']}")
    print(f"best_candidate: {result.get('best_candidate')}")


if __name__ == "__main__":
    main()
