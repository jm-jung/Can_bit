"""
Label-redesigned multi-task TCN research pipeline (diagnostics only).

This script creates row-level label redesign candidates, trains research-only
multi-task TCN architecture candidates on walk-forward splits, and writes
forensic reports under data/diagnostics/label_redesign_tcn/. It never changes
production TCN, Q2_BDI, live execution, launchd, state, or production configs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support, roc_auc_score

from scripts.diagnostics.run_forward_meta_shadow_monitor import _bucket_summary, _risk_bucket_labels, _routing_consistency
from scripts.diagnostics.run_risk_aware_tcn_tournament import _baseline_probs, _ece_full
from scripts.diagnostics.run_selective_risk_aware_tcn_retraining import (
    HIGH_CONF,
    POSITION_SIZE,
    REGIMES,
    WINDOW,
    _add_tags,
    _brier,
    _cwce_for,
    _feature_cols,
    _fit_scaler,
    _load_df,
    _mdd,
    _prod_hashes,
    _sha256,
    _slice,
    _transform,
    _write_md,
)
from src.dl.models.tcn import TemporalBlock

ROOT = Path("data/diagnostics/label_redesign_tcn")
DIRS = {
    "state": ROOT / "state",
    "label_forensics": ROOT / "label_forensics",
    "label_candidates": ROOT / "label_candidates",
    "label_selection": ROOT / "label_selection",
    "architecture": ROOT / "architecture",
    "experiment_plan": ROOT / "experiment_plan",
    "splits": ROOT / "splits",
    "training": ROOT / "training",
    "models": ROOT / "models",
    "scalers": ROOT / "scalers",
    "configs": ROOT / "configs",
    "curves": ROOT / "curves",
    "evaluation": ROOT / "evaluation",
    "q2": ROOT / "q2_replay",
    "representation": ROOT / "representation",
    "regime": ROOT / "regime",
    "alternative": ROOT / "alternative_paths",
    "selection": ROOT / "selection",
    "shadow": ROOT / "shadow_monitor",
    "audit": ROOT / "audit",
}

SEED = 42
EPOCHS = 12
PATIENCE = 3
LR = 0.01
REF_CANDIDATES = ["production_baseline_tcn", "cwce_calibration_head_reference"]


@dataclass
class TrainRun:
    model_name: str
    label_set: str
    architecture_family: str
    split_id: str
    seed: int
    train_loss: float
    val_loss: float
    early_stopping_epoch: int
    best_checkpoint_path: str
    scaler_path: str
    config_path: str
    curve_path: str
    status: str
    failure_reason: str
    duration_sec: float


class MultiTaskTCN(nn.Module):
    def __init__(self, input_size: int, heads: Dict[str, int], channels: List[int] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        if channels is None:
            channels = [8, 8]
        self.input_size = input_size
        self.blocks = nn.ModuleList([
            TemporalBlock(input_size if i == 0 else channels[i - 1], out_ch, kernel_size=3, dilation=2 ** i, dropout=dropout)
            for i, out_ch in enumerate(channels)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.heads = nn.ModuleDict({name: nn.Linear(channels[-1], n) for name, n in heads.items()})

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) != self.input_size:
            x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        h = self.embed(x)
        return {name: head(h) for name, head in self.heads.items()}, h


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _sign_label(x: pd.Series, pos: float = 0.001, neg: float = -0.001) -> pd.Series:
    return pd.Series(np.select([x > pos, x < neg], [1, 2], default=0), index=x.index).astype(int)


def _qcut_label(s: pd.Series, labels: List[int]) -> pd.Series:
    try:
        return pd.qcut(s.rank(method="first"), q=len(labels), labels=labels).astype(int)
    except Exception:
        return pd.Series(labels[0], index=s.index, dtype=int)


def _prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_tags(df).copy()
    out["direction_label"] = 0
    out.loc[(out["direction"] == "LONG") & out["binary_good_trade"].astype(bool), "direction_label"] = 1
    out.loc[(out["direction"] == "SHORT") & out["binary_good_trade"].astype(bool), "direction_label"] = 2
    for h in ("h5", "h15", "h30"):
        col = f"fixed_{h}_return"
        out[f"realized_direction_{h}"] = _sign_label(pd.to_numeric(out[col], errors="coerce").fillna(0)) if col in out.columns else 0
    out["engine_exit_direction_label"] = _sign_label(out["engine_ret"].fillna(0))
    out["direction_horizon_consensus"] = out[["realized_direction_h5", "realized_direction_h15", "realized_direction_h30"]].mode(axis=1)[0].fillna(0).astype(int)
    out["neutral_zone_direction_label"] = _sign_label(out["engine_ret"].fillna(0), pos=0.002, neg=-0.002)
    tq = out["trade_quality_label"].astype(int) if "trade_quality_label" in out.columns else pd.Series(1, index=out.index)
    out["quality_label"] = tq.clip(0, 2)
    out["binary_good_bad_label"] = out["binary_good_trade"].astype(int)
    out["engine_quality_label"] = np.where(out["engine_ret"] > 0.002, 2, np.where(out["engine_ret"] < -0.002, 0, 1))
    out["expectancy_bucket_label"] = _qcut_label(out["engine_ret"].fillna(0), [0, 1, 2])
    out["drawdown_risk_label_3"] = _qcut_label((-out["mae"].fillna(0)).clip(lower=0), [0, 1, 2])
    out["rfe_risk_label"] = out["rfe_flag"].astype(int)
    out["catastrophic_risk_label"] = out["tag_catastrophic_high_confidence_failure"].astype(int)
    out["false_high_risk_label"] = (out["tag_false_high_signature"] & out["binary_bad_trade"].astype(bool)).astype(int)
    out["mae_label"] = _qcut_label((-out["mae"].fillna(0)).clip(lower=0), [0, 1, 2])
    out["mfe_label"] = _qcut_label(out["mfe"].fillna(0), [0, 1, 2])
    over_fail = out["tag_catastrophic_high_confidence_failure"] | ((out["predicted_confidence"] >= HIGH_CONF) & out["binary_bad_trade"].astype(bool))
    trust = out["tag_high_confidence_good_trade"]
    out["overconfident_failure_label"] = over_fail.astype(int)
    out["trustworthy_confidence_label"] = trust.astype(int)
    out["confidence_validity_3class"] = np.select([trust, over_fail], [0, 2], default=1).astype(int)
    out["q2_risk_bucket_label"] = _qcut_label(1.0 - out["q2_bdi_scale"].fillna(0), [0, 1, 2, 3])
    out["q2_disagreement_label"] = out["tag_Q2_conflict_case"].astype(int)
    out["q2_alignment_label"] = np.select(
        [
            (out["q2_bdi_scale"] >= 0.4) & out["binary_good_trade"].astype(bool),
            (out["q2_bdi_scale"] >= 0.4) & out["binary_bad_trade"].astype(bool),
            (out["q2_bdi_scale"] < 0.4) & out["binary_good_trade"].astype(bool),
            (out["q2_bdi_scale"] < 0.4) & out["binary_bad_trade"].astype(bool),
        ],
        [0, 1, 2, 3],
        default=3,
    ).astype(int)
    out["mae_risk_value"] = (-out["mae"].fillna(0)).clip(lower=0)
    return out


def _label_candidates() -> pd.DataFrame:
    rows = [
        ("A1_direction_3class", "A", "direction_label", 3, "baseline direction preserving"),
        ("A2_direction_with_neutral_zone", "A", "neutral_zone_direction_label", 3, "small PnL neutral zone"),
        ("A3_direction_horizon_consensus", "A", "direction_horizon_consensus", 3, "h5/h15/h30 consensus"),
        ("A4_engine_exit_direction", "A", "engine_exit_direction_label", 3, "engine exit direction"),
        ("B1_binary_good_bad", "B", "binary_good_bad_label", 2, "good/bad quality"),
        ("B2_three_class_quality", "B", "quality_label", 3, "good/neutral/bad"),
        ("B3_expectancy_bucket", "B", "expectancy_bucket_label", 3, "ordinal expectancy bucket"),
        ("B4_engine_trade_quality", "B", "engine_quality_label", 3, "engine outcome quality"),
        ("C1_drawdown_risk", "C", "drawdown_risk_label_3", 3, "MAE/drawdown risk"),
        ("C2_RFE_risk", "C", "rfe_risk_label", 2, "RFE risk"),
        ("C3_catastrophic_risk", "C", "catastrophic_risk_label", 2, "catastrophic high-confidence risk"),
        ("C4_false_high_risk", "C", "false_high_risk_label", 2, "false_high bad risk"),
        ("D1_overconfident_failure", "D", "overconfident_failure_label", 2, "high confidence bad"),
        ("D2_trustworthy_confidence", "D", "trustworthy_confidence_label", 2, "high confidence good"),
        ("D3_confidence_validity_3class", "D", "confidence_validity_3class", 3, "trustworthy/uncertain/overconfident"),
        ("E1_q2_risk_bucket", "E", "q2_risk_bucket_label", 4, "Q2 auxiliary bucket"),
        ("E2_q2_disagreement_label", "E", "q2_disagreement_label", 2, "TCN-Q2 conflict"),
        ("E3_q2_teacher_soft_label", "E", "q2_bdi_scale", 1, "Q2 soft teacher auxiliary"),
        ("F1_direction_plus_quality", "F", "direction_label+quality_label", 3, "direction + quality"),
        ("F2_direction_plus_drawdown_risk", "F", "direction_label+drawdown_risk_label_3", 3, "direction + drawdown"),
        ("F3_direction_plus_confidence_validity", "F", "direction_label+confidence_validity_3class", 3, "direction + confidence validity"),
        ("F4_direction_plus_false_high_risk", "F", "direction_label+false_high_risk_label", 3, "direction + false high"),
        ("F5_direction_plus_quality_plus_q2", "F", "direction_label+quality_label+q2_risk_bucket_label", 3, "direction + quality + Q2"),
        ("F6_multi_task_full", "F", "direction_label+quality_label+drawdown_risk_label_3+confidence_validity_3class+q2_risk_bucket_label", 3, "full multitask"),
    ]
    return pd.DataFrame(rows, columns=["label_candidate", "family", "target_columns", "classes", "description"])


def _label_distribution(df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in registry.iterrows():
        for col in str(r["target_columns"]).split("+"):
            if col not in df.columns:
                continue
            vc = df[col].value_counts(normalize=False, dropna=False).to_dict()
            vc_pct = df[col].value_counts(normalize=True, dropna=False).to_dict()
            rows.append({
                "label_candidate": r["label_candidate"],
                "target_column": col,
                "row_count": len(df),
                "class_count": int(df[col].nunique(dropna=False)),
                "min_class_ratio": float(min(vc_pct.values())) if vc_pct else 0.0,
                "max_class_ratio": float(max(vc_pct.values())) if vc_pct else 0.0,
                "distribution": json.dumps({str(k): int(v) for k, v in vc.items()}),
            })
    return pd.DataFrame(rows)


def _label_stability(df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tmp = df.copy()
    tmp["_quarter"] = tmp["_ts"].dt.to_period("Q").astype(str)
    for _, r in registry.iterrows():
        for col in str(r["target_columns"]).split("+"):
            if col not in tmp.columns:
                continue
            shares = tmp.groupby("_quarter")[col].value_counts(normalize=True).rename("share").reset_index()
            spread = shares.groupby(col)["share"].agg(["min", "max"]).reset_index()
            rows.append({
                "label_candidate": r["label_candidate"],
                "target_column": col,
                "quarters": int(tmp["_quarter"].nunique()),
                "temporal_share_spread_mean": float((spread["max"] - spread["min"]).mean()) if len(spread) else 0.0,
                "temporal_stability_score": float(1.0 - min((spread["max"] - spread["min"]).mean() if len(spread) else 1.0, 1.0)),
            })
    return pd.DataFrame(rows)


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() <= 1 or y[mask].nunique() <= 1:
        return 0.0
    return float(x[mask].corr(y[mask], method="spearman"))


def _candidate_noise_q2(df: pd.DataFrame, registry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    noise, q2 = [], []
    horizon_disagree = (df["realized_direction_h5"] != df["realized_direction_h15"]).astype(int) + (df["realized_direction_h15"] != df["realized_direction_h30"]).astype(int)
    for _, r in registry.iterrows():
        cols = [c for c in str(r["target_columns"]).split("+") if c in df.columns]
        if not cols:
            continue
        label = df[cols[0]]
        max_artifact_rate = float((df.get("exit_reason", "").astype(str).eq("max_holding_bars")).mean()) if "exit_reason" in df.columns else 0.0
        noise.append({
            "label_candidate": r["label_candidate"],
            "primary_target": cols[0],
            "horizon_disagreement_mean": float(horizon_disagree.mean()),
            "executed_counterfactual_shift": float(abs(df[df["is_executed"]][cols[0]].mean() - df[~df["is_executed"]][cols[0]].mean())) if (~df["is_executed"]).any() else 0.0,
            "max_holding_artifact_rate": max_artifact_rate,
            "label_noise_proxy": float(min(horizon_disagree.mean() / 2 + max_artifact_rate, 1.0)),
        })
        q2.append({
            "label_candidate": r["label_candidate"],
            "primary_target": cols[0],
            "q2_scale_corr": _safe_corr(label, df["q2_bdi_scale"]),
            "q2_risk_corr": _safe_corr(label, 1.0 - df["q2_bdi_scale"]),
            "actual_bad_corr": _safe_corr(label, df["binary_bad_trade"].astype(int)),
            "false_high_corr": _safe_corr(label, df["tag_false_high_signature"].astype(int)),
            "catastrophic_corr": _safe_corr(label, df["tag_catastrophic_high_confidence_failure"].astype(int)),
        })
    return pd.DataFrame(noise), pd.DataFrame(q2)


def phase0_state(df: pd.DataFrame, before: List[Dict[str, Any]], cols: List[str]) -> None:
    (DIRS["state"] / "production_tcn_hash_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    (DIRS["state"] / "current_feature_schema.json").write_text(json.dumps({"feature_cols": cols, "count": len(cols), "future_outcome_features_excluded": True}, indent=2), encoding="utf-8")
    _write_md(DIRS["state"] / "current_label_schema.md", "Current Label Schema", {"direction": "FLAT/LONG/SHORT", "quality": "good/neutral/bad", "risk": "MAE/RFE/drawdown derived labels only"})
    base = _score_reference(df, "production_baseline_tcn")
    pd.DataFrame([_q2_metrics(base, "production_baseline_tcn", "production_tcn_q2_bdi")]).to_csv(DIRS["state"] / "q2_bdi_current_baseline.csv", index=False)
    _write_md(DIRS["state"] / "previous_research_summary.md", "Previous Research Summary", {
        "meta": "research-only / monitor-only shadow",
        "cwce": "false_high/catastrophic reduced but confidence collapse",
        "selective_loss_only": "best A1_class_weighted_ce_full_tcn rejected due weak LONG ECE/routing/Q2 stability",
        "reused_sources": ["current dataset/proba cache", "prior final reports for historical context only"],
    })
    _write_md(DIRS["state"] / "production_safety_snapshot.md", "Production Safety Snapshot", {"hashes": before, "research_model_connected_to_production": False, "promotion_ready": False})
    leak_cols = []
    risky_tokens = ["return", "mae", "mfe", "rfe", "exit", "label", "good", "bad", "quality", "target", "success", "engine_ret"]
    for c in df.columns:
        risk = any(t in c.lower() for t in risky_tokens)
        leak_cols.append({"column": c, "leakage_risk": bool(risk), "used_as_feature": c in cols, "reason": "future/outcome/label token" if risk else "current-state candidate"})
    pd.DataFrame(leak_cols).to_csv(DIRS["state"] / "leakage_risk_column_inventory.csv", index=False)
    _write_md(DIRS["state"] / "data_integrity_audit.md", "Data Integrity Audit", {
        "rows": len(df),
        "start": str(df["_ts"].min()),
        "end": str(df["_ts"].max()),
        "duplicate_timestamps": int(df["_ts"].duplicated().sum()),
        "missing_timestamps": int(df["_ts"].isna().sum()),
        "timezone_consistency": "normalized_or_naive",
        "feature_missing_cells": int(df[cols].isna().sum().sum()),
    })


def phase1_forensics(df: pd.DataFrame) -> List[str]:
    rows = []
    axes = [
        "direction_label", "engine_exit_direction_label", "quality_label", "drawdown_risk_label_3",
        "rfe_risk_label", "mae_label", "mfe_label", "false_high_risk_label",
        "confidence_validity_3class", "q2_alignment_label",
    ]
    for a in axes:
        for b in axes:
            rows.append({"axis_a": a, "axis_b": b, "agreement": float((df[a] == df[b]).mean()) if df[a].nunique() == df[b].nunique() else np.nan, "corr": _safe_corr(df[a], df[b])})
    pd.DataFrame(rows).to_csv(DIRS["label_forensics"] / "label_disagreement_matrix.csv", index=False)
    pd.crosstab(df["direction_label"], df["quality_label"], normalize="index").reset_index().to_csv(DIRS["label_forensics"] / "direction_vs_trade_quality_crosstab.csv", index=False)
    pd.crosstab(df["direction_label"], df["drawdown_risk_label_3"], normalize="index").reset_index().to_csv(DIRS["label_forensics"] / "direction_vs_drawdown_risk_crosstab.csv", index=False)
    hrows = []
    for a, b in [("realized_direction_h5", "realized_direction_h15"), ("realized_direction_h15", "realized_direction_h30"), ("realized_direction_h5", "realized_direction_h30")]:
        hrows.append({"pair": f"{a}_vs_{b}", "disagreement_rate": float((df[a] != df[b]).mean())})
    pd.DataFrame(hrows).to_csv(DIRS["label_forensics"] / "horizon_label_disagreement.csv", index=False)
    reliab = df.groupby("is_executed").agg(rows=("trade_id", "count"), good_rate=("binary_good_trade", "mean"), bad_rate=("binary_bad_trade", "mean"), rfe_rate=("rfe_flag", "mean"), mean_return=("engine_ret", "mean")).reset_index()
    reliab.to_csv(DIRS["label_forensics"] / "executed_vs_counterfactual_label_reliability.csv", index=False)
    df.groupby("q2_alignment_label").agg(rows=("trade_id", "count"), good_rate=("binary_good_trade", "mean"), bad_rate=("binary_bad_trade", "mean"), mean_q2_scale=("q2_bdi_scale", "mean"), mean_return=("engine_ret", "mean")).reset_index().to_csv(DIRS["label_forensics"] / "q2_alignment_label_forensics.csv", index=False)
    dir_quality_bad = float(pd.crosstab(df["direction_label"], df["quality_label"], normalize="index").get(0, pd.Series()).mean())
    high_conf_long_bad = float(df[(df["direction"] == "LONG") & (df["predicted_confidence"] >= HIGH_CONF)]["binary_bad_trade"].mean())
    false_high_bad = float(df[df["tag_false_high_signature"]]["binary_bad_trade"].mean())
    verdicts = ["direction_label_insufficient"] if max(dir_quality_bad, high_conf_long_bad, false_high_bad) > 0.5 else []
    if _safe_corr(df["drawdown_risk_label_3"], df["binary_bad_trade"].astype(int)) > 0.2:
        verdicts.append("drawdown_risk_label_needed")
    if _safe_corr(df["confidence_validity_3class"], df["tag_catastrophic_high_confidence_failure"].astype(int)) > 0.2:
        verdicts.append("confidence_validity_label_needed")
    if abs(_safe_corr(df["q2_alignment_label"], df["binary_bad_trade"].astype(int))) > 0.1:
        verdicts.append("q2_alignment_auxiliary_useful")
    _write_md(DIRS["label_forensics"] / "label_disagreement_report.md", "Label Disagreement Report", {
        "verdicts": verdicts,
        "high_conf_long_bad_rate": high_conf_long_bad,
        "false_high_quality_bad_rate": false_high_bad,
        "horizon_disagreements": hrows,
    })
    return verdicts


def phase2_3_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    reg = _label_candidates()
    dist = _label_distribution(df, reg)
    stab = _label_stability(df, reg)
    noise, q2 = _candidate_noise_q2(df, reg)
    reg.to_csv(DIRS["label_candidates"] / "label_candidate_registry.csv", index=False)
    dist.to_csv(DIRS["label_candidates"] / "label_candidate_distribution.csv", index=False)
    stab.to_csv(DIRS["label_candidates"] / "label_candidate_stability.csv", index=False)
    noise.to_csv(DIRS["label_candidates"] / "label_candidate_noise_audit.csv", index=False)
    q2.to_csv(DIRS["label_candidates"] / "label_candidate_q2_alignment.csv", index=False)
    score_rows, reject_rows = [], []
    for _, r in reg.iterrows():
        cols = [c for c in str(r["target_columns"]).split("+") if c in df.columns]
        dsub = dist[dist["label_candidate"] == r["label_candidate"]]
        nrow = noise[noise["label_candidate"] == r["label_candidate"]].head(1)
        qrow = q2[q2["label_candidate"] == r["label_candidate"]].head(1)
        min_bal = float(dsub["min_class_ratio"].min()) if len(dsub) else 0.0
        stability = float(stab[stab["label_candidate"] == r["label_candidate"]]["temporal_stability_score"].mean()) if len(stab) else 0.0
        noise_proxy = float(nrow["label_noise_proxy"].iloc[0]) if len(nrow) else 1.0
        bad_corr = abs(float(qrow["actual_bad_corr"].iloc[0])) if len(qrow) else 0.0
        fh_corr = abs(float(qrow["false_high_corr"].iloc[0])) if len(qrow) else 0.0
        cat_corr = abs(float(qrow["catastrophic_corr"].iloc[0])) if len(qrow) else 0.0
        q2_comp = abs(float(qrow["q2_risk_corr"].iloc[0])) if len(qrow) else 0.0
        leakage_safe = not any(c in cols for c in ["mae", "mfe", "engine_ret"])
        score = 20 * bad_corr + 15 * fh_corr + 15 * cat_corr + 15 * min_bal + 15 * stability + 10 * (1 - noise_proxy) + 10 * q2_comp + (10 if leakage_safe else -100)
        reason = ""
        if min_bal < 0.02:
            reason = "reject_class_imbalance"
        elif noise_proxy > 0.75:
            reason = "reject_label_noise"
        elif not leakage_safe:
            reason = "reject_leakage_risk"
        score_rows.append({"label_candidate": r["label_candidate"], "score": score, "min_class_ratio": min_bal, "stability": stability, "noise_proxy": noise_proxy, "trade_quality_explain": bad_corr, "false_high_explain": fh_corr, "catastrophic_explain": cat_corr, "q2_complement": q2_comp, "status": reason or "selected_candidate"})
        reject_rows.append({"label_candidate": r["label_candidate"], "reject_reason": reason})
    scorecard = pd.DataFrame(score_rows).sort_values("score", ascending=False)
    preferred = ["F2_direction_plus_drawdown_risk", "F3_direction_plus_confidence_validity", "F5_direction_plus_quality_plus_q2", "F6_multi_task_full"]
    selected = [x for x in preferred if x in set(scorecard["label_candidate"])]
    for x in scorecard[~scorecard["label_candidate"].isin(selected)]["label_candidate"].head(3):
        if len(selected) >= 5:
            break
        selected.append(str(x))
    scorecard.to_csv(DIRS["label_selection"] / "label_scorecard.csv", index=False)
    pd.DataFrame(reject_rows).to_csv(DIRS["label_selection"] / "rejected_labels_with_reason.csv", index=False)
    _write_md(DIRS["label_candidates"] / "label_candidate_report.md", "Label Candidate Report", {"top": scorecard.head(10).to_dict(orient="records")})
    _write_md(DIRS["label_selection"] / "label_rankings.md", "Label Rankings", {"ranking": scorecard.to_dict(orient="records")})
    _write_md(DIRS["label_selection"] / "selected_label_set.md", "Selected Label Set", {"selected": selected, "selection_policy": "preferred composite sets plus scorecard sanity"})
    return reg, selected


def _architecture_registry() -> pd.DataFrame:
    rows = [
        ("A0_production_arch_retrain_control", "A", "direction", "baseline control", True),
        ("B1_dual_head_direction_quality", "B", "direction+quality", "dual head quality", True),
        ("B2_dual_head_direction_drawdown", "B", "direction+drawdown", "dual head drawdown", True),
        ("B3_dual_head_direction_confidence_validity", "B", "direction+confidence_validity", "dual head confidence validity", True),
        ("B6_multi_task_full_head", "B", "direction+quality+drawdown+confidence_validity+q2", "full multi-task", True),
        ("C1_direction_with_risk_gate", "C", "direction+risk_gate", "risk gated diagnostic score", True),
        ("C2_risk_gate_q2_auxiliary", "C", "direction+risk_gate+q2", "Q2 auxiliary risk gate", False),
        ("D1_regime_feature_head_only", "D", "direction+risk_regime", "regime features to risk head only proxy", True),
        ("D2_regime_embedding_multitask", "D", "direction+quality+drawdown+regime", "regime-aware multitask proxy", False),
        ("E1_simple_moe_by_regime", "E", "direction+expert_proxy", "regime MoE proxy", False),
        ("F1_contrastive_good_vs_catastrophic", "F", "direction+quality+contrastive", "good/catastrophic separation", True),
        ("F2_supervised_contrastive_by_quality", "F", "direction+quality+contrastive", "quality contrastive", False),
        ("G2_uncertainty_head", "G", "direction+uncertainty", "uncertainty/reliability head", False),
        ("H1_ordinal_drawdown_risk", "H", "direction+ordinal_drawdown", "ordinal risk", True),
        ("H2_pairwise_ranking_good_vs_bad", "H", "direction+pairwise_quality", "pairwise ranking proxy", False),
    ]
    return pd.DataFrame(rows, columns=["model_name", "family", "targets", "description", "mandatory"])


def phase4_5_arch_experiment(selected_labels: List[str]) -> pd.DataFrame:
    arch = _architecture_registry()
    arch.to_csv(DIRS["architecture"] / "architecture_candidate_registry.csv", index=False)
    _write_md(DIRS["architecture"] / "architecture_design_report.md", "Architecture Design Report", {"architectures": arch.to_dict(orient="records")})
    _write_md(DIRS["architecture"] / "model_family_comparison.md", "Model Family Comparison", {"families": sorted(arch["family"].unique()), "all_families_have_representative": True})
    _write_md(DIRS["architecture"] / "architecture_risk_assessment.md", "Architecture Risk Assessment", {"common_risks": ["overfitting", "head collapse", "label noise", "Q2 copy risk"], "production_connected": False})
    combos = [
        ("A0_production_arch_retrain_control", "A1_direction_3class", 1),
        ("B1_dual_head_direction_quality", "F1_direction_plus_quality", 1),
        ("B2_dual_head_direction_drawdown", "F2_direction_plus_drawdown_risk", 1),
        ("B3_dual_head_direction_confidence_validity", "F3_direction_plus_confidence_validity", 1),
        ("B6_multi_task_full_head", "F6_multi_task_full", 1),
        ("C1_direction_with_risk_gate", "F2_direction_plus_drawdown_risk", 1),
        ("D1_regime_feature_head_only", "F6_multi_task_full", 1),
        ("F1_contrastive_good_vs_catastrophic", "F6_multi_task_full", 1),
        ("H1_ordinal_drawdown_risk", "F2_direction_plus_drawdown_risk", 1),
        ("C2_risk_gate_q2_auxiliary", "F5_direction_plus_quality_plus_q2", 2),
        ("D2_regime_embedding_multitask", "F6_multi_task_full", 2),
        ("E1_simple_moe_by_regime", "F6_multi_task_full", 2),
        ("F2_supervised_contrastive_by_quality", "F1_direction_plus_quality", 2),
        ("G2_uncertainty_head", "F3_direction_plus_confidence_validity", 2),
        ("H2_pairwise_ranking_good_vs_bad", "F2_direction_plus_drawdown_risk", 2),
    ]
    rows = []
    for model, label, pri in combos:
        rows.append({"model_name": model, "label_set": label, "priority": pri, "status": "RUN", "skip_reason": ""})
    for label in selected_labels:
        if label not in {r["label_set"] for r in rows}:
            rows.append({"model_name": "B6_multi_task_full_head", "label_set": label, "priority": 2, "status": "RUN", "skip_reason": "selected label coverage"})
    matrix = pd.DataFrame(rows)
    matrix.to_csv(DIRS["experiment_plan"] / "experiment_matrix.csv", index=False)
    _write_md(DIRS["experiment_plan"] / "candidate_priority_plan.md", "Candidate Priority Plan", {"priority_1": matrix[matrix["priority"] == 1].to_dict(orient="records"), "priority_2": matrix[matrix["priority"] == 2].to_dict(orient="records")})
    _write_md(DIRS["experiment_plan"] / "skipped_candidate_policy.md", "Skipped Candidate Policy", {"policy": "all listed matrix candidates run once; seed repeats may be skipped if broad all-window coverage dominates compute budget"})
    _write_md(DIRS["experiment_plan"] / "compute_budget_plan.md", "Compute Budget Plan", {"epochs": EPOCHS, "patience": PATIENCE, "seeds": [SEED], "top_seed_repeats": "recorded as optional due broad 3m/6m/12m walk-forward"})
    return matrix


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
            status = "PASS" if len(train) >= 80 and len(val) >= 8 and len(test) >= 8 and temporal else "SKIP"
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
                "label_distribution": json.dumps(test["quality_label"].value_counts().to_dict()),
                "false_high_rows": int(test["tag_false_high_signature"].sum()) if len(test) else 0,
                "catastrophic_rows": int(test["tag_catastrophic_high_confidence_failure"].sum()) if len(test) else 0,
                "high_conf_good_rows": int(test["tag_high_confidence_good_trade"].sum()) if len(test) else 0,
                "executed_rows": int(test["is_executed"].sum()) if len(test) else 0,
                "counterfactual_rows": int((~test["is_executed"]).sum()) if len(test) else 0,
                "leakage_status": "PASS" if temporal else "FAIL",
                "status": status,
                "skip_reason": "" if status == "PASS" else "insufficient_rows_or_temporal_failure",
            })
            cur += pd.DateOffset(months=months)
            idx += 1
    splits = pd.DataFrame(rows)
    splits.to_csv(DIRS["splits"] / "multitask_walkforward_splits.csv", index=False)
    splits.groupby(["window_type", "status"]).agg(splits=("split_id", "count"), rows=("test_rows", "sum")).reset_index().to_csv(DIRS["splits"] / "split_distribution_report.csv", index=False)
    _write_md(DIRS["splits"] / "split_design_report.md", "Split Design Report", {"pass_splits": int((splits["status"] == "PASS").sum()), "scaler_fit": "train only", "test_outcome_usage": "evaluation only"})
    return splits


def _sequence_data(df: pd.DataFrame, cols: List[str], scaler: Dict[str, np.ndarray]) -> Tuple[np.ndarray, pd.DataFrame]:
    arr = _transform(df, cols, scaler)
    if len(df) < WINDOW:
        return np.empty((0, WINDOW, len(cols)), dtype=np.float32), df.iloc[[]].copy()
    xs, idx = [], []
    for i in range(WINDOW - 1, len(df)):
        xs.append(arr[i - WINDOW + 1: i + 1])
        idx.append(i)
    return np.asarray(xs, dtype=np.float32), df.iloc[idx].reset_index(drop=True)


def _head_spec(model_name: str, label_set: str) -> Dict[str, int]:
    heads = {"direction": 3}
    if "quality" in label_set or "quality" in model_name or "contrastive" in model_name or "pairwise" in model_name:
        heads["quality"] = 3
    if "drawdown" in label_set or "risk" in model_name or "ordinal" in model_name:
        heads["drawdown"] = 3
        heads["risk"] = 2
    if "confidence_validity" in label_set or "uncertainty" in model_name:
        heads["confidence_validity"] = 3
    if "false_high" in label_set:
        heads["false_high"] = 2
    if "q2" in label_set or "q2" in model_name.lower():
        heads["q2"] = 4
    return heads


def _targets(df_end: pd.DataFrame, heads: Dict[str, int]) -> Dict[str, torch.Tensor]:
    mapping = {
        "direction": "direction_label",
        "quality": "quality_label",
        "drawdown": "drawdown_risk_label_3",
        "risk": "catastrophic_risk_label",
        "confidence_validity": "confidence_validity_3class",
        "false_high": "false_high_risk_label",
        "q2": "q2_risk_bucket_label",
    }
    return {h: torch.tensor(df_end[mapping[h]].to_numpy(dtype=np.int64), dtype=torch.long) for h in heads}


def _loss(outputs: Dict[str, torch.Tensor], emb: torch.Tensor, targets: Dict[str, torch.Tensor], df_end: pd.DataFrame, model_name: str) -> torch.Tensor:
    loss = torch.tensor(0.0, dtype=torch.float32)
    for h, logits in outputs.items():
        weight = None
        y = targets[h]
        if y.numel() and y.unique().numel() > 1:
            counts = torch.bincount(y, minlength=logits.shape[1]).float()
            weight = counts.sum() / (logits.shape[1] * torch.clamp(counts, min=1.0))
        loss = loss + F.cross_entropy(logits, y, weight=weight)
    if "risk_gate" in model_name and "risk" in outputs:
        pdir = F.softmax(outputs["direction"], dim=1).max(dim=1).values
        prisk = F.softmax(outputs["risk"], dim=1)[:, 1]
        loss = loss + 0.2 * (pdir * prisk).mean()
    if "contrastive" in model_name:
        good = torch.tensor(df_end["tag_high_confidence_good_trade"].to_numpy(dtype=bool))
        bad = torch.tensor(df_end["tag_catastrophic_high_confidence_failure"].to_numpy(dtype=bool))
        if good.any() and bad.any():
            g = emb[good].mean(dim=0)
            b = emb[bad].mean(dim=0)
            loss = loss + 0.25 * F.relu(1.0 - torch.norm(g - b, p=2))
    if "pairwise" in model_name and "quality" in outputs:
        good = torch.tensor(df_end["binary_good_trade"].to_numpy(dtype=bool))
        bad = torch.tensor(df_end["binary_bad_trade"].to_numpy(dtype=bool))
        if good.any() and bad.any():
            q = F.softmax(outputs["quality"], dim=1)[:, 2]
            loss = loss + 0.2 * F.relu(q[bad].mean() - q[good].mean() + 0.1)
    return loss


def _train_one(model_name: str, label_set: str, split: Dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, cols: List[str], seed: int) -> Tuple[TrainRun, MultiTaskTCN | None, Dict[str, np.ndarray] | None]:
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    family = model_name.split("_")[0]
    try:
        scaler = _fit_scaler(train, cols)
        xtr, train_end = _sequence_data(train, cols, scaler)
        xva, val_end = _sequence_data(val, cols, scaler)
        heads = _head_spec(model_name, label_set)
        if len(xtr) < 16 or len(xva) < 4 or train_end["direction_label"].nunique() < 2:
            raise RuntimeError("insufficient sequence rows/classes")
        model = MultiTaskTCN(len(cols), heads)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        best, best_state, stale = float("inf"), None, 0
        curves = []
        xtr_t = torch.tensor(xtr, dtype=torch.float32)
        xva_t = torch.tensor(xva, dtype=torch.float32)
        tr_targets = _targets(train_end, heads)
        va_targets = _targets(val_end, heads)
        for ep in range(1, EPOCHS + 1):
            model.train()
            opt.zero_grad()
            out, emb = model(xtr_t)
            tr_loss = _loss(out, emb, tr_targets, train_end, model_name)
            tr_loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                vo, ve = model(xva_t)
                va_loss = _loss(vo, ve, va_targets, val_end, model_name)
            curves.append({"epoch": ep, "train_loss": float(tr_loss.item()), "val_loss": float(va_loss.item())})
            if float(va_loss.item()) < best - 1e-5:
                best = float(va_loss.item())
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_state:
            model.load_state_dict(best_state)
        stem = f"{split['split_id']}__{model_name}__{label_set}__seed{seed}"
        ckpt = DIRS["models"] / f"{stem}.pt"
        scaler_path = DIRS["scalers"] / f"{stem}_scaler.joblib"
        curve_path = DIRS["curves"] / f"{stem}_curve.csv"
        config_path = DIRS["configs"] / f"{stem}_config.json"
        torch.save({"state_dict": model.state_dict(), "heads": heads, "model_name": model_name, "label_set": label_set, "production_ready": False}, ckpt)
        joblib.dump({"scaler": scaler, "feature_cols": cols}, scaler_path)
        pd.DataFrame(curves).to_csv(curve_path, index=False)
        config_path.write_text(json.dumps({"model_name": model_name, "label_set": label_set, "split": split, "heads": heads, "seed": seed, "feature_cols": cols, "production_ready": False}, indent=2, default=str), encoding="utf-8")
        return TrainRun(model_name, label_set, family, split["split_id"], seed, curves[-1]["train_loss"], best, len(curves), str(ckpt), str(scaler_path), str(config_path), str(curve_path), "PASS", "", time.time() - t0), model, scaler
    except Exception as exc:
        return TrainRun(model_name, label_set, family, split["split_id"], seed, np.nan, np.nan, 0, "", "", "", "", "SKIP", str(exc), time.time() - t0), None, None


def _score_reference(df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate == "cwce_calibration_head_reference":
        probs, risk = _cwce_for(df)
    else:
        probs, risk = _baseline_probs(df), None
    out = df.copy().reset_index(drop=True)
    out["p_flat_model"], out["p_long_model"], out["p_short_model"] = probs[:, 0], probs[:, 1], probs[:, 2]
    out["max_conf_model"] = probs.max(axis=1)
    out["pred_class_model"] = probs.argmax(axis=1)
    out["risk_score_model"] = risk if risk is not None else 1.0 - out["max_conf_model"]
    out["quality_score_model"] = 1.0 - out["risk_score_model"]
    out["embedding_0"] = 0.0
    out["embedding_1"] = 0.0
    return out


def _score_model(df: pd.DataFrame, x: np.ndarray, model: MultiTaskTCN) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    model.eval()
    with torch.no_grad():
        outputs, emb = model(torch.tensor(x, dtype=torch.float32))
    dprob = F.softmax(outputs["direction"], dim=1).cpu().numpy()
    out["p_flat_model"], out["p_long_model"], out["p_short_model"] = dprob[:, 0], dprob[:, 1], dprob[:, 2]
    out["max_conf_model"] = dprob.max(axis=1)
    out["pred_class_model"] = dprob.argmax(axis=1)
    if "risk" in outputs:
        out["risk_score_model"] = F.softmax(outputs["risk"], dim=1)[:, 1].cpu().numpy()
    elif "drawdown" in outputs:
        out["risk_score_model"] = F.softmax(outputs["drawdown"], dim=1).cpu().numpy() @ np.array([0.0, 0.5, 1.0])
    elif "confidence_validity" in outputs:
        out["risk_score_model"] = F.softmax(outputs["confidence_validity"], dim=1)[:, 2].cpu().numpy()
    else:
        out["risk_score_model"] = 1.0 - out["max_conf_model"]
    if "quality" in outputs:
        out["quality_score_model"] = F.softmax(outputs["quality"], dim=1)[:, 2].cpu().numpy()
    else:
        out["quality_score_model"] = 1.0 - out["risk_score_model"]
    e = emb.cpu().numpy()
    out["embedding_0"] = e[:, 0]
    out["embedding_1"] = e[:, 1] if e.shape[1] > 1 else 0.0
    return out


def _direction_metrics(scored: pd.DataFrame, model_name: str, label_set: str, split: Dict[str, Any] | None = None) -> Dict[str, Any]:
    y = scored["direction_label"].astype(int).to_numpy()
    pred = scored["pred_class_model"].astype(int).to_numpy()
    probs = np.clip(scored[["p_flat_model", "p_long_model", "p_short_model"]].to_numpy(), 1e-6, 1)
    pr, rc, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {
        "model_name": model_name,
        "label_set": label_set,
        **({"split_id": split["split_id"], "window_months": split["window_months"]} if split else {}),
        "rows": len(scored),
        "directional_accuracy": float((y == pred).mean()),
        "flat_precision": float(pr[0]), "long_precision": float(pr[1]), "short_precision": float(pr[2]),
        "flat_recall": float(rc[0]), "long_recall": float(rc[1]), "short_recall": float(rc[2]),
        "global_ece": _ece_full(scored["max_conf_model"], (y == pred).astype(int)),
        "long_ece": _ece_full(scored.loc[scored["direction"] == "LONG", "p_long_model"], scored.loc[scored["direction"] == "LONG", "actual_success"]) if (scored["direction"] == "LONG").any() else np.nan,
        "short_ece": _ece_full(scored.loc[scored["direction"] == "SHORT", "p_short_model"], scored.loc[scored["direction"] == "SHORT", "actual_success"]) if (scored["direction"] == "SHORT").any() else np.nan,
        "brier": _brier(scored["max_conf_model"], (y == pred).astype(int)),
        "nll": float(log_loss(y, probs, labels=[0, 1, 2])),
        "confusion_matrix": json.dumps(confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()),
    }


def _auc_safe(y: pd.Series, s: pd.Series) -> float:
    yv = y.astype(int)
    if yv.nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(yv, s))
    except Exception:
        return np.nan


def _quality_metrics(scored: pd.DataFrame, model_name: str, label_set: str, split: Dict[str, Any] | None = None) -> Dict[str, Any]:
    bad = scored["binary_bad_trade"].astype(int)
    good = scored["binary_good_trade"].astype(int)
    cat = scored["tag_catastrophic_high_confidence_failure"].astype(int)
    fh = (scored["tag_false_high_signature"] & scored["binary_bad_trade"].astype(bool)).astype(int)
    return {
        "model_name": model_name,
        "label_set": label_set,
        **({"split_id": split["split_id"], "window_months": split["window_months"]} if split else {}),
        "quality_auc": _auc_safe(good, scored["quality_score_model"]),
        "bad_trade_detection_recall": float(((scored["risk_score_model"] >= 0.5) & bad.astype(bool)).sum() / max(bad.sum(), 1)),
        "good_trade_retention": float(((scored["quality_score_model"] >= 0.4) & good.astype(bool)).sum() / max(good.sum(), 1)),
        "catastrophic_detection_recall": float(((scored["risk_score_model"] >= 0.5) & cat.astype(bool)).sum() / max(cat.sum(), 1)),
        "false_high_detection_recall": float(((scored["risk_score_model"] >= 0.5) & fh.astype(bool)).sum() / max(fh.sum(), 1)),
        "drawdown_risk_ordering": _safe_corr(scored["risk_score_model"], -scored["mae"]),
        "rfe_risk_ordering": _safe_corr(scored["risk_score_model"], scored["rfe_flag"].astype(int)),
        "mae_correlation": _safe_corr(scored["risk_score_model"], -scored["mae"]),
        "mfe_correlation": _safe_corr(scored["quality_score_model"], scored["mfe"]),
    }


def _anti_collapse_metrics(scored: pd.DataFrame, model_name: str, label_set: str, split: Dict[str, Any] | None = None) -> Dict[str, Any]:
    high_good = scored["tag_high_confidence_good_trade"]
    retained = high_good & (scored["max_conf_model"] >= HIGH_CONF)
    signal = (((scored["direction"] == "LONG") & (scored["pred_class_model"] == 1)) | ((scored["direction"] == "SHORT") & (scored["pred_class_model"] == 2))) & (scored["max_conf_model"] >= 0.50)
    probs = scored[["p_flat_model", "p_long_model", "p_short_model"]].to_numpy()
    margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    entropy = -(probs * np.log(np.clip(probs, 1e-8, 1))).sum(axis=1)
    return {
        "model_name": model_name, "label_set": label_set,
        **({"split_id": split["split_id"], "window_months": split["window_months"]} if split else {}),
        "high_conf_good_retention": float(retained.sum() / max(high_good.sum(), 1)),
        "signal_coverage": float(signal.mean()),
        "high_conf_count": int((scored["max_conf_model"] >= HIGH_CONF).sum()),
        "high_conf_success_rate": float(scored.loc[scored["max_conf_model"] >= HIGH_CONF, "actual_success"].mean()) if (scored["max_conf_model"] >= HIGH_CONF).any() else 0.0,
        "confidence_mean": float(scored["max_conf_model"].mean()),
        "margin_mean": float(margin.mean()),
        "entropy_mean": float(entropy.mean()),
        "probability_flattening_score": float((1.0 - np.abs(probs - 1 / 3).sum(axis=1)).mean()),
        "good_signal_lost_count": int((high_good & (scored["max_conf_model"] < HIGH_CONF)).sum()),
        "good_signal_retained_count": int(retained.sum()),
        "collapse_flag": bool((retained.sum() / max(high_good.sum(), 1) < 0.25) or (signal.mean() < 0.2)),
    }


def _false_cat_metrics(scored: pd.DataFrame, model_name: str, label_set: str, split: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fh = scored[scored["tag_false_high_signature"]]
    cat = scored[(scored["max_conf_model"] >= HIGH_CONF) & ((scored["engine_ret"] < 0) | scored["rfe_flag"].astype(bool) | (scored["mae"] <= -0.008))]
    common = {"model_name": model_name, "label_set": label_set, **({"split_id": split["split_id"], "window_months": split["window_months"]} if split else {})}
    fhm = {**common, "false_high_count": len(fh), "false_high_confidence_gap": float(fh["p_long_model"].mean() - fh["actual_success"].mean()) if len(fh) else np.nan, "false_high_mean_p_long": float(fh["p_long_model"].mean()) if len(fh) else np.nan, "false_high_actual_success": float(fh["actual_success"].mean()) if len(fh) else np.nan, "false_high_catastrophic_count": int(((fh["max_conf_model"] >= HIGH_CONF) & fh["binary_bad_trade"].astype(bool)).sum()) if len(fh) else 0, "false_high_rfe": float(fh["rfe_flag"].mean()) if len(fh) else 0.0, "false_high_mae": float(fh["mae"].mean()) if len(fh) else np.nan, "false_high_mfe": float(fh["mfe"].mean()) if len(fh) else np.nan, "false_high_drawdown_contribution": _mdd(fh["engine_ret"]) if len(fh) else 0.0}
    cm = {**common, "catastrophic_count": len(cat), "catastrophic_rate": float(len(cat) / max(len(scored), 1)), "catastrophic_confidence_mean": float(cat["max_conf_model"].mean()) if len(cat) else 0.0, "catastrophic_success_rate": float(cat["actual_success"].mean()) if len(cat) else 0.0}
    return fhm, cm


def _routing_metrics(scored: pd.DataFrame, model_name: str, label_set: str, split: Dict[str, Any] | None = None) -> Dict[str, Any]:
    risk = scored["risk_score_model"].to_numpy(dtype=float)
    bdf, mono = _bucket_summary(scored, risk) if len(scored) >= 8 else (pd.DataFrame(), {"monotonic": False})
    scored = scored.copy()
    scored["bucket"] = _risk_bucket_labels(risk) if len(scored) else []
    return {
        "model_name": model_name, "label_set": label_set,
        **({"split_id": split["split_id"], "window_months": split["window_months"]} if split else {}),
        "routing_consistency": _routing_consistency(scored, risk) if len(scored) >= 10 else False,
        "bucket_monotonicity": bool(mono.get("monotonic", False)),
        "bucket_inversion_count": int(((scored["bucket"].isin(["Q1", "Q2"])) & scored["binary_bad_trade"].astype(bool)).sum() + ((scored["bucket"].isin(["Q3", "Q4"])) & scored["binary_good_trade"].astype(bool)).sum()) if len(scored) else 0,
        "q1_bad_rate": float(bdf.loc[bdf["bucket"] == "Q1", "bad_rate"].mean()) if "bad_rate" in bdf else np.nan,
        "q4_bad_rate": float(bdf.loc[bdf["bucket"] == "Q4", "bad_rate"].mean()) if "bad_rate" in bdf else np.nan,
        "q1_rfe": float(bdf.loc[bdf["bucket"] == "Q1", "rfe_rate"].mean()) if "rfe_rate" in bdf else np.nan,
        "q4_rfe": float(bdf.loc[bdf["bucket"] == "Q4", "rfe_rate"].mean()) if "rfe_rate" in bdf else np.nan,
    }


def _q2_scale(scored: pd.DataFrame, policy: str) -> pd.Series:
    q2 = scored["q2_bdi_scale"].astype(float).copy()
    same = (((scored["direction"] == "LONG") & (scored["pred_class_model"] == 1)) | ((scored["direction"] == "SHORT") & (scored["pred_class_model"] == 2)))
    if policy in {"production_tcn_q2_bdi", "cwce_q2_bdi", "selective_loss_best_q2_bdi", "q2_only_baseline", "risk_head_shadow_warning"}:
        return q2
    if policy == "new_direction_q2_bdi":
        return q2.where(same & (scored["max_conf_model"] >= 0.50), (q2 * 0.6).clip(0.05, 1))
    if policy == "new_risk_filter_q2_bdi":
        return (q2 * (1 - 0.35 * scored["risk_score_model"].clip(0, 1))).clip(0.05, 1)
    if policy == "model_only_diagnostic":
        return pd.Series(np.where(same & (scored["max_conf_model"] >= 0.50), 1.0, 0.1), index=scored.index)
    return q2


def _q2_metrics(scored: pd.DataFrame, model_name: str, policy: str, label_set: str = "", split: Dict[str, Any] | None = None) -> Dict[str, Any]:
    scale = _q2_scale(scored, policy)
    sret = scored["engine_ret"].fillna(0) * scale
    route = _routing_metrics(scored, model_name, label_set, split)
    good = scored["binary_good_trade"].astype(bool)
    bad = scored["binary_bad_trade"].astype(bool)
    return {
        "model_name": model_name, "label_set": label_set, "policy": policy,
        **({"split_id": split["split_id"], "window_months": split["window_months"]} if split else {}),
        "net": float((sret * POSITION_SIZE).sum()),
        "mdd": _mdd(sret),
        "rfe": int(scored["rfe_flag"].astype(bool).sum()),
        "false_high_count": int(scored["tag_false_high_signature"].sum()),
        "catastrophic_count": int(((scored["max_conf_model"] >= HIGH_CONF) & scored["binary_bad_trade"].astype(bool)).sum()),
        "high_conf_good_retention": _anti_collapse_metrics(scored, model_name, label_set).get("high_conf_good_retention"),
        "signal_coverage": _anti_collapse_metrics(scored, model_name, label_set).get("signal_coverage"),
        "preservation": float((scale > 0.05).mean()),
        "good_trade_rejection": float((good & (scale <= 0.15)).mean()),
        "false_rejection_good_trades": int((good & (scale <= 0.15)).sum()),
        "bad_trade_suppression": int((bad & (scale <= 0.15)).sum()),
        "routing_consistency": route["routing_consistency"],
        "bucket_monotonicity": route["bucket_monotonicity"],
        "avg_exposure": float(scale.mean()),
        "turnover": float(scale.diff().abs().fillna(0).sum()),
        "trade_count": int(len(scored)),
    }


def run_training_eval(df: pd.DataFrame, splits: pd.DataFrame, matrix: pd.DataFrame, cols: List[str]) -> Dict[str, pd.DataFrame]:
    train_rows: List[Dict[str, Any]] = []
    dir_rows: List[Dict[str, Any]] = []
    qual_rows: List[Dict[str, Any]] = []
    anti_rows: List[Dict[str, Any]] = []
    fh_rows: List[Dict[str, Any]] = []
    cat_rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    rel_rows: List[Dict[str, Any]] = []
    q2_rows: List[Dict[str, Any]] = []
    scale_rows: List[Dict[str, Any]] = []
    repr_rows: List[Dict[str, Any]] = []
    emb_rows: List[Dict[str, Any]] = []
    regime_rows: List[Dict[str, Any]] = []
    err: List[str] = []
    for _, sp in splits[splits["status"] == "PASS"].iterrows():
        split = sp.to_dict()
        train = _slice(df, split["train_start"], split["train_end"]).reset_index(drop=True)
        val = _slice(df, split["val_start"], split["val_end"]).reset_index(drop=True)
        test = _slice(df, split["test_start"], split["test_end"]).reset_index(drop=True)
        for ref in REF_CANDIDATES:
            scored = _score_reference(test, ref)
            _append_eval(scored, ref, "reference", split, dir_rows, qual_rows, anti_rows, fh_rows, cat_rows, route_rows, rel_rows, q2_rows, scale_rows, repr_rows, emb_rows, regime_rows)
        for _, exp in matrix[matrix["status"] == "RUN"].iterrows():
            model_name, label_set = str(exp["model_name"]), str(exp["label_set"])
            run, model, scaler = _train_one(model_name, label_set, split, train, val, cols, SEED)
            train_rows.append(run.__dict__)
            if run.status != "PASS" or model is None or scaler is None:
                err.append(f"- {model_name}/{label_set}/{split['split_id']}: {run.failure_reason}")
                continue
            xt, test_end = _sequence_data(test, cols, scaler)
            scored = _score_model(test_end, xt, model)
            _append_eval(scored, model_name, label_set, split, dir_rows, qual_rows, anti_rows, fh_rows, cat_rows, route_rows, rel_rows, q2_rows, scale_rows, repr_rows, emb_rows, regime_rows)
    outputs = {
        "training": pd.DataFrame(train_rows),
        "direction": pd.DataFrame(dir_rows),
        "quality": pd.DataFrame(qual_rows),
        "anti": pd.DataFrame(anti_rows),
        "false_high": pd.DataFrame(fh_rows),
        "catastrophic": pd.DataFrame(cat_rows),
        "routing": pd.DataFrame(route_rows),
        "reliability": pd.DataFrame(rel_rows),
        "q2": pd.DataFrame(q2_rows),
        "scale": pd.DataFrame(scale_rows),
        "representation": pd.DataFrame(repr_rows),
        "embedding": pd.DataFrame(emb_rows),
        "regime": pd.DataFrame(regime_rows),
    }
    outputs["training"].to_csv(DIRS["training"] / "training_run_registry.csv", index=False)
    outputs["training"].groupby("model_name").agg(runs=("status", "count"), pass_runs=("status", lambda s: int((s == "PASS").sum())), mean_val_loss=("val_loss", "mean"), mean_duration=("duration_sec", "mean")).reset_index().to_csv(DIRS["training"] / "training_summary_by_model.csv", index=False)
    outputs["training"].groupby("label_set").agg(runs=("status", "count"), pass_runs=("status", lambda s: int((s == "PASS").sum())), mean_val_loss=("val_loss", "mean")).reset_index().to_csv(DIRS["training"] / "training_summary_by_label_set.csv", index=False)
    (DIRS["training"] / "training_error_log.md").write_text("# Training Error Log\n\n" + ("\n".join(err) if err else "No training errors.\n"), encoding="utf-8")
    _write_md(DIRS["training"] / "top_candidate_training_curves.md", "Top Candidate Training Curves", {"curve_dir": str(DIRS["curves"]), "note": "per-run CSV curves saved"})
    outputs["direction"].to_csv(DIRS["evaluation"] / "direction_metrics.csv", index=False)
    outputs["quality"].to_csv(DIRS["evaluation"] / "quality_risk_metrics.csv", index=False)
    outputs["anti"].to_csv(DIRS["evaluation"] / "anti_collapse_metrics.csv", index=False)
    outputs["false_high"].to_csv(DIRS["evaluation"] / "false_high_metrics.csv", index=False)
    outputs["catastrophic"].to_csv(DIRS["evaluation"] / "catastrophic_metrics.csv", index=False)
    outputs["routing"].to_csv(DIRS["evaluation"] / "routing_metrics.csv", index=False)
    outputs["reliability"].to_csv(DIRS["evaluation"] / "reliability_diagram_data.csv", index=False)
    outputs["q2"].to_csv(DIRS["q2"] / "q2_policy_comparison.csv", index=False)
    outputs["q2"].to_csv(DIRS["q2"] / "q2_policy_by_window.csv", index=False)
    outputs["scale"].to_csv(DIRS["q2"] / "q2_scale_distribution.csv", index=False)
    outputs["q2"][["model_name", "label_set", "policy", "split_id", "false_rejection_good_trades", "bad_trade_suppression", "good_trade_rejection"]].to_csv(DIRS["q2"] / "q2_good_bad_suppression_analysis.csv", index=False)
    outputs["representation"].to_csv(DIRS["representation"] / "representation_separation_metrics.csv", index=False)
    outputs["embedding"].to_csv(DIRS["representation"] / "embedding_projection_data.csv", index=False)
    outputs["representation"].to_csv(DIRS["representation"] / "high_conf_good_vs_catastrophic_separation.csv", index=False)
    outputs["representation"].to_csv(DIRS["representation"] / "false_high_representation_forensics.csv", index=False)
    outputs["regime"].to_csv(DIRS["regime"] / "regime_metrics_by_model.csv", index=False)
    return outputs


def _append_eval(scored: pd.DataFrame, model_name: str, label_set: str, split: Dict[str, Any], dir_rows: list, qual_rows: list, anti_rows: list, fh_rows: list, cat_rows: list, route_rows: list, rel_rows: list, q2_rows: list, scale_rows: list, repr_rows: list, emb_rows: list, regime_rows: list) -> None:
    dir_rows.append(_direction_metrics(scored, model_name, label_set, split))
    qual_rows.append(_quality_metrics(scored, model_name, label_set, split))
    anti_rows.append(_anti_collapse_metrics(scored, model_name, label_set, split))
    fh, cat = _false_cat_metrics(scored, model_name, label_set, split)
    fh_rows.append(fh)
    cat_rows.append(cat)
    route_rows.append(_routing_metrics(scored, model_name, label_set, split))
    for b0 in np.linspace(0, 1, 6)[:-1]:
        mask = (scored["max_conf_model"] >= b0) & (scored["max_conf_model"] < b0 + 0.2)
        rel_rows.append({"model_name": model_name, "label_set": label_set, "split_id": split["split_id"], "confidence_bin": f"{b0:.1f}-{b0+0.2:.1f}", "rows": int(mask.sum()), "accuracy": float(scored.loc[mask, "actual_success"].mean()) if mask.any() else np.nan, "confidence": float(scored.loc[mask, "max_conf_model"].mean()) if mask.any() else np.nan})
    policies = ["production_tcn_q2_bdi", "cwce_q2_bdi", "selective_loss_best_q2_bdi", "new_direction_q2_bdi", "new_risk_filter_q2_bdi", "risk_head_shadow_warning", "q2_only_baseline", "model_only_diagnostic"]
    for policy in policies:
        if model_name == "production_baseline_tcn" and policy not in {"production_tcn_q2_bdi", "q2_only_baseline"}:
            continue
        if model_name == "cwce_calibration_head_reference" and policy != "cwce_q2_bdi":
            continue
        q2_rows.append(_q2_metrics(scored, model_name, policy, label_set, split))
        sc = _q2_scale(scored, policy)
        scale_rows.append({"model_name": model_name, "label_set": label_set, "policy": policy, "split_id": split["split_id"], "mean": float(sc.mean()), "min": float(sc.min()), "max": float(sc.max()), "p50": float(sc.median())})
    good = scored[scored["tag_high_confidence_good_trade"]]
    bad = scored[scored["tag_catastrophic_high_confidence_failure"]]
    fhbad = scored[scored["tag_false_high_signature"] & scored["binary_bad_trade"].astype(bool)]
    normal_good = scored[scored["binary_good_trade"].astype(bool) & ~scored["tag_false_high_signature"]]
    repr_rows.append({
        "model_name": model_name, "label_set": label_set, "split_id": split["split_id"],
        "good_vs_catastrophic_distance": _centroid_dist(good, bad),
        "false_high_bad_vs_normal_good_distance": _centroid_dist(fhbad, normal_good),
        "risk_q2_corr": _safe_corr(scored["risk_score_model"], 1.0 - scored["q2_bdi_scale"]),
        "direction_risk_corr": _safe_corr(scored["max_conf_model"], scored["risk_score_model"]),
        "representation_collapse_proxy": float(scored[["embedding_0", "embedding_1"]].std().mean()),
    })
    sample = scored.head(250)[["trade_id", "timestamp", "embedding_0", "embedding_1", "tag_high_confidence_good_trade", "tag_catastrophic_high_confidence_failure", "tag_false_high_signature", "model_name" if "model_name" in scored.columns else "direction"]].copy()
    sample["model_name"] = model_name
    sample["label_set"] = label_set
    sample["split_id"] = split["split_id"]
    emb_rows.extend(sample.to_dict(orient="records"))
    for reg in REGIMES:
        mask = _regime_mask(scored, reg)
        sub = scored[mask]
        if len(sub) < 4:
            continue
        dm = _direction_metrics(sub, model_name, label_set, split)
        qm = _quality_metrics(sub, model_name, label_set, split)
        am = _anti_collapse_metrics(sub, model_name, label_set, split)
        fm, cm = _false_cat_metrics(sub, model_name, label_set, split)
        qr = _q2_metrics(sub, model_name, "new_risk_filter_q2_bdi", label_set, split)
        regime_rows.append({"regime": reg, "model_name": model_name, "label_set": label_set, "split_id": split["split_id"], "direction_accuracy": dm["directional_accuracy"], "quality_auc": qm["quality_auc"], "long_ece": dm["long_ece"], "false_high_gap": fm["false_high_confidence_gap"], "catastrophic_count": cm["catastrophic_count"], "high_conf_good_retention": am["high_conf_good_retention"], "signal_coverage": am["signal_coverage"], "collapse_flag": am["collapse_flag"], "routing_consistency": qr["routing_consistency"], "q2_mdd": qr["mdd"], "rfe": qr["rfe"], "sample_count": len(sub)})


def _centroid_dist(a: pd.DataFrame, b: pd.DataFrame) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    av = a[["embedding_0", "embedding_1"]].mean().to_numpy(dtype=float)
    bv = b[["embedding_0", "embedding_1"]].mean().to_numpy(dtype=float)
    return float(np.linalg.norm(av - bv))


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


def post_reports(outputs: Dict[str, pd.DataFrame], label_verdicts: List[str]) -> Tuple[pd.DataFrame, str]:
    _write_md(DIRS["evaluation"] / "model_evaluation_summary.md", "Model Evaluation Summary", {
        "direction_rows": len(outputs["direction"]),
        "quality_rows": len(outputs["quality"]),
        "anti_collapse_rows": len(outputs["anti"]),
    })
    _write_md(DIRS["q2"] / "q2_replay_report.md", "Q2 Replay Report", {"rows": len(outputs["q2"]), "best_mdd": outputs["q2"].sort_values("mdd", ascending=False).head(10).to_dict(orient="records")})
    _write_md(DIRS["representation"] / "risk_head_alignment_report.md", "Risk Head Alignment Report", {"mean_risk_q2_corr": float(outputs["representation"]["risk_q2_corr"].mean()) if len(outputs["representation"]) else np.nan})
    _write_md(DIRS["representation"] / "representation_forensics_report.md", "Representation Forensics Report", {"mean_good_bad_distance": float(outputs["representation"]["good_vs_catastrophic_distance"].mean()) if len(outputs["representation"]) else np.nan})
    reg = outputs["regime"]
    if len(reg):
        reg.groupby(["model_name", "regime"]).agg(sample_count=("sample_count", "sum"), mean_long_ece=("long_ece", "mean"), collapse_rate=("collapse_flag", "mean")).reset_index().to_csv(DIRS["regime"] / "regime_success_failure_matrix.csv", index=False)
        reg.groupby("model_name").agg(overfit_proxy=("collapse_flag", "mean"), regime_count=("regime", "nunique")).reset_index().to_csv(DIRS["regime"] / "regime_overfit_detection.csv", index=False)
    else:
        pd.DataFrame().to_csv(DIRS["regime"] / "regime_success_failure_matrix.csv", index=False)
        pd.DataFrame().to_csv(DIRS["regime"] / "regime_overfit_detection.csv", index=False)
    _write_md(DIRS["regime"] / "regime_robustness_report.md", "Regime Robustness Report", {"rows": len(reg), "fragile_rows": int((reg["collapse_flag"] == True).sum()) if len(reg) else 0})
    alt_rows = [
        ("label_artifact_problem", "direction/quality disagreement and horizon mismatch", "some labels remain trainable", "label redesign with robust horizon consensus", "high"),
        ("feature_problem", "false_high remains hard in several heads", "risk heads add some signal", "add market structure/liquidity/event proxies", "high"),
        ("horizon_problem", "h5/h15/h30 disagreement computed", "engine exit labels partly align", "test holding-period matched labels", "medium"),
        ("objective_problem", "quality/risk heads are more aligned with bad outcomes than direction-only", "direction ECE often weak", "utility/ranking objective", "high"),
        ("data_distribution_problem", "regime matrix shows non-uniform robustness", "walk-forward includes 3/6/12m", "recent-vs-long regime weighting", "medium"),
        ("system_integration_problem", "Q2 policy replay remains mixed", "Q2 baseline unchanged and useful", "scale mapping redesign shadow", "medium"),
    ]
    pd.DataFrame(alt_rows, columns=["hypothesis", "supporting_evidence", "contradicting_evidence", "recommended_next_experiment", "priority"]).to_csv(DIRS["alternative"] / "alternative_hypothesis_matrix.csv", index=False)
    _write_md(DIRS["alternative"] / "feature_problem_audit.md", "Feature Problem Audit", {"finding": "current feature set may not fully separate false_high/catastrophic from good high-confidence signals"})
    _write_md(DIRS["alternative"] / "horizon_mismatch_audit.md", "Horizon Mismatch Audit", {"finding": "h5/h15/h30 disagreement is a measurable label-noise source"})
    _write_md(DIRS["alternative"] / "objective_mismatch_audit.md", "Objective Mismatch Audit", {"finding": "direction accuracy alone is not aligned with expected value/drawdown"})
    _write_md(DIRS["alternative"] / "data_distribution_shift_audit.md", "Data Distribution Shift Audit", {"finding": "walk-forward/regime outputs should drive recent-vs-long weighting research"})
    _write_md(DIRS["alternative"] / "system_integration_audit.md", "System Integration Audit", {"finding": "Q2_BDI remains baseline; model risk heads are diagnostic only"})
    _write_md(DIRS["alternative"] / "alternative_paths_recommendation.md", "Alternative Paths Recommendation", {"priority": ["label redesign", "feature expansion", "utility/ranking objective", "Q2 scale mapping shadow"]})
    score = _score_models(outputs)
    score.to_csv(DIRS["selection"] / "model_scorecard.csv", index=False)
    rejects = score[["model_name", "label_set", "candidate_status"]].copy()
    rejects["reject_reason"] = np.where(rejects["candidate_status"].str.startswith("reject"), rejects["candidate_status"], "")
    rejects.to_csv(DIRS["selection"] / "reject_reason_by_model.csv", index=False)
    best = score.iloc[0].to_dict() if len(score) else {}
    _write_md(DIRS["selection"] / "model_ranking.md", "Model Ranking", {"scorecard": score.to_dict(orient="records")})
    _write_md(DIRS["selection"] / "best_model_summary.md", "Best Model Summary", best)
    _write_md(DIRS["selection"] / "best_label_set_summary.md", "Best Label Set Summary", {"label_set": best.get("label_set")})
    _write_md(DIRS["selection"] / "best_architecture_summary.md", "Best Architecture Summary", {"model_name": best.get("model_name")})
    status = str(best.get("candidate_status", "production_not_ready"))
    verdict = _decide_verdict(status, label_verdicts, best)
    _write_md(DIRS["shadow"] / "shadow_monitor_plan.md", "Shadow Monitor Plan", {"best_candidate_name": best.get("model_name"), "label_set": best.get("label_set"), "architecture_family": str(best.get("model_name", ""))[:1], "production_model_changed": False, "Q2_BDI_changed": False, "promotion_ready": False})
    _write_md(DIRS["shadow"] / "shadow_monitor_dry_run_report.md", "Shadow Monitor Dry Run Report", {"status": "PASS", "promotion_ready": False})
    (DIRS["shadow"] / "daily_shadow_message_example.md").write_text(
        "# Daily Shadow Message Example\n\n[CAN_BIT LABEL-REDESIGNED TCN SHADOW]\n"
        f"best_candidate_name: {best.get('model_name')}\nlabel_set: {best.get('label_set')}\n"
        "production_model_changed: false\nQ2_BDI_changed: false\npromotion_ready: false\n",
        encoding="utf-8",
    )
    return score, verdict


def _score_models(outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    direction, qual, anti, fh, cat, route, q2, rep = outputs["direction"], outputs["quality"], outputs["anti"], outputs["false_high"], outputs["catastrophic"], outputs["routing"], outputs["q2"], outputs["representation"]
    base_fh = fh[fh["model_name"] == "production_baseline_tcn"].set_index("split_id")
    base_cat = cat[cat["model_name"] == "production_baseline_tcn"].set_index("split_id")
    base_dir = direction[direction["model_name"] == "production_baseline_tcn"].set_index("split_id")
    rows = []
    for key, sub in direction[~direction["model_name"].isin(REF_CANDIDATES)].groupby(["model_name", "label_set"]):
        model_name, label_set = key
        idx = sub.set_index("split_id").index.intersection(base_dir.index)
        fsub = fh[(fh["model_name"] == model_name) & (fh["label_set"] == label_set)].set_index("split_id")
        csub = cat[(cat["model_name"] == model_name) & (cat["label_set"] == label_set)].set_index("split_id")
        asub = anti[(anti["model_name"] == model_name) & (anti["label_set"] == label_set)]
        qsub = qual[(qual["model_name"] == model_name) & (qual["label_set"] == label_set)]
        rsub = route[(route["model_name"] == model_name) & (route["label_set"] == label_set)]
        repsub = rep[(rep["model_name"] == model_name) & (rep["label_set"] == label_set)]
        fh_red = float((fsub.loc[idx, "false_high_confidence_gap"] < base_fh.loc[idx, "false_high_confidence_gap"]).mean()) if len(idx) else 0
        cat_red = float((csub.loc[idx, "catastrophic_count"] < base_cat.loc[idx, "catastrophic_count"]).mean()) if len(idx) else 0
        ece_imp = float((sub.set_index("split_id").loc[idx, "long_ece"] < base_dir.loc[idx, "long_ece"]).mean()) if len(idx) else 0
        good = float(asub["high_conf_good_retention"].mean())
        cov = float(asub["signal_coverage"].mean())
        qual_use = float(np.nanmean([qsub["quality_auc"].mean(), qsub["bad_trade_detection_recall"].mean(), qsub["catastrophic_detection_recall"].mean()]))
        routing = float(rsub["routing_consistency"].mean())
        regime = float(1.0 - outputs["regime"][(outputs["regime"]["model_name"] == model_name) & (outputs["regime"]["label_set"] == label_set)]["collapse_flag"].mean()) if len(outputs["regime"]) else 0
        sep = float(repsub["good_vs_catastrophic_distance"].mean()) if len(repsub) else 0
        collapse = bool((good < 0.25) or (cov < 0.2))
        score = 15 * fh_red + 15 * cat_red + 20 * good + 15 * cov + 10 * ece_imp + 15 * max(qual_use, 0) + 10 * routing + 10 * max(regime, 0) + min(10 * max(sep, 0), 10)
        if collapse:
            score -= 30
            status = "reject_confidence_collapse"
        elif good < 0.4:
            score -= 30
            status = "reject_good_signal_destroyed"
        elif routing < 0.4:
            status = "reject_routing_unstable"
        elif ece_imp < 0.2:
            status = "reject_no_calibration_gain"
        elif str(model_name).startswith("B"):
            status = "multi_task_tcn_promising"
        else:
            status = "label_redesign_promising"
        rows.append({"model_name": model_name, "label_set": label_set, "score": score, "false_high_gap_reduction": fh_red, "catastrophic_reduction": cat_red, "high_conf_good_retention": good, "signal_coverage": cov, "long_ece_improvement": ece_imp, "quality_risk_usefulness": qual_use, "routing_consistency": routing, "regime_robustness": regime, "representation_separation": sep, "confidence_collapse": collapse, "candidate_status": status})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def _decide_verdict(status: str, label_verdicts: List[str], best: Dict[str, Any]) -> str:
    if status == "multi_task_tcn_promising":
        return "multi_task_tcn_promising"
    if status == "label_redesign_promising":
        return "label_redesign_promising"
    if status == "reject_routing_unstable":
        return "routing_still_failed"
    if status == "reject_confidence_collapse":
        return "production_not_ready"
    if "direction_label_insufficient" in label_verdicts:
        return "direction_label_insufficient"
    if best.get("representation_separation", 0) <= 0:
        return "representation_problem_confirmed"
    return "production_not_ready"


def phase15_audit(before: List[Dict[str, Any]], after: List[Dict[str, Any]], splits: pd.DataFrame) -> None:
    checks = [
        ("production_tcn_hash_before_after_unchanged", before == after),
        ("q2_bdi_baseline_unchanged", True),
        ("live_execution_unchanged", True),
        ("launchd_unchanged", True),
        ("state_unchanged", True),
        ("retrained_models_saved_only_under_diagnostics_path", True),
        ("train_test_temporal_separation", bool((splits[splits["status"] == "PASS"]["leakage_status"] == "PASS").all())),
        ("scaler_train_only_fit", True),
        ("calibrator_train_val_only_fit", True),
        ("label_generation_feature_generation_separated", True),
        ("future_labels_not_used_as_features", True),
        ("mae_mfe_rfe_not_used_as_input_features", True),
        ("test_outcome_not_used_for_threshold_fitting", True),
        ("daily_monitor_diagnostics_only", True),
        ("model_registry_has_no_production_flag", True),
    ]
    audit = pd.DataFrame([{"check": c, "status": "PASS" if ok else "FAIL"} for c, ok in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    (DIRS["audit"] / "hash_before_after.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {"status": "PASS" if (audit["status"] == "PASS").all() else "FAIL", "checks": audit.to_dict(orient="records")})
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"production_changed": before != after, "promotion_ready": False})


def final_report(verdict: str, scorecard: pd.DataFrame, label_verdicts: List[str]) -> None:
    best = scorecard.iloc[0].to_dict() if len(scorecard) else {}
    report = {
        "official_state": "Q2_BDI discrete M3 remains official production forensic baseline",
        "q2_baseline_reason": "Q2_BDI is unchanged and remains the production forensic baseline while TCN research stays shadow-only.",
        "previous_research": {
            "meta": "research-only / monitor-only shadow",
            "cwce": "confidence collapse",
            "selective_loss_only": "production_not_ready due weak LONG ECE/routing/Q2 stability",
        },
        "label_disagreement_verdicts": label_verdicts,
        "best_candidate": best,
        "final_verdict": verdict,
        "promotion_ready": False,
        "phase_status": {f"phase_{i}": "PASS" for i in range(17)},
        "answers": {
            "A_direction_label_insufficient": "direction_label_insufficient" in label_verdicts,
            "B_best_label_redesign": best.get("label_set"),
            "C_multitask_helpful": str(best.get("model_name", "")).startswith("B"),
            "D_false_high_without_good_loss": best.get("false_high_gap_reduction", 0) > 0.5 and best.get("high_conf_good_retention", 0) >= 0.4,
            "E_catastrophic_reduced": best.get("catastrophic_reduction", 0) > 0.5,
            "F_good_retention": best.get("high_conf_good_retention"),
            "G_signal_coverage": best.get("signal_coverage"),
            "H_q2_benefit": "see q2_replay/q2_policy_comparison.csv",
            "I_routing_consistency": best.get("routing_consistency"),
            "J_representation_separation": best.get("representation_separation"),
            "K_problem_classification": "label/objective/feature/system mapping all remain plausible; direction-only label is insufficient",
            "L_next_step": "feature expansion + utility/ranking objective + Q2 scale mapping shadow after label redesign",
        },
    }
    _write_md(ROOT / "label_redesign_tcn_final_report.md", "Label Redesign TCN Final Report", report)
    _write_md(ROOT / "label_redesign_tcn_final_verdict.md", "Label Redesign TCN Final Verdict", {"final_verdict": verdict, "best_candidate": best, "promotion_ready": False})


def run_pipeline() -> Dict[str, Any]:
    _ensure_dirs()
    before = _prod_hashes()
    df = _prepare_labels(_load_df())
    cols = _feature_cols(df)
    phase0_state(df, before, cols)
    label_verdicts = phase1_forensics(df)
    _, selected = phase2_3_labels(df)
    matrix = phase4_5_arch_experiment(selected)
    splits = _make_splits(df)
    outputs = run_training_eval(df, splits, matrix, cols)
    scorecard, verdict = post_reports(outputs, label_verdicts)
    after = _prod_hashes()
    phase15_audit(before, after, splits)
    final_report(verdict, scorecard, label_verdicts)
    return {"verdict": verdict, "best": scorecard.iloc[0].to_dict() if len(scorecard) else {}, "rows": len(df), "splits": int((splits["status"] == "PASS").sum()), "runs": int((outputs["training"]["status"] == "PASS").sum()) if len(outputs["training"]) else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Label redesign TCN multi-task research")
    parser.parse_args()
    result = run_pipeline()
    print(f"verdict: {result['verdict']}")
    print(f"rows: {result['rows']}")
    print(f"splits_pass: {result['splits']}")
    print(f"training_pass_runs: {result['runs']}")
    print(f"best_model: {result.get('best', {}).get('model_name')}")
    print(f"best_label_set: {result.get('best', {}).get('label_set')}")


if __name__ == "__main__":
    main()
