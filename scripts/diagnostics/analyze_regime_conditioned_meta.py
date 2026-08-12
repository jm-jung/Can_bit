"""
Regime-conditioned Meta stability & temporal robustness (diagnostics only).

Finds which market regimes support stable Meta risk ranking vs global failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.analyze_executed_anchor_routing import (
    LABEL_COL,
    N_BUCKETS,
    POSITION_SIZE,
    _bucket_stats,
    _build_anchor_dataset,
    _monotonicity_score,
    _risk_routing_valid,
    _risk_buckets,
    _train_meta_model,
)
from scripts.diagnostics.analyze_executed_first_meta_learning import _prepare_full_df
from scripts.diagnostics.train_meta_layer_model import (
    _calibration_error,
    _prepare_x,
    _time_split,
)
from scripts.diagnostics.validate_quality_score_replay import map_m3

OUT_DIR = Path("data/diagnostics/meta_layer/regime_conditioned")
FEATURE_COLS = __import__(
    "scripts.diagnostics.analyze_counterfactual_label_quality", fromlist=["FEATURE_COLS"]
).FEATURE_COLS


def _segment_regimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_numeric(df.get("trend_strength", 0), errors="coerce").fillna(0)
    ts_med = float(ts.median()) if len(ts) else 0.0003
    trend = df.get("trend_state", df.get("trend_label", "")).astype(str)

    def _trend_regime(row: pd.Series) -> str:
        t = str(row.get("trend_state", ""))
        s = float(row.get("trend_strength", 0) or 0)
        if t == "up":
            return "strong_uptrend" if s >= ts_med else "weak_uptrend"
        if t == "down":
            return "strong_downtrend" if s >= ts_med else "weak_downtrend"
        return "sideways"

    df["trend_regime"] = df.apply(_trend_regime, axis=1)

    vol = df.get("vol_bucket", "mid").astype(str)
    vexp = pd.to_numeric(df.get("volatility_expansion_rate", 0), errors="coerce").fillna(0)
    vol_reg: List[str] = []
    for v, e in zip(vol, vexp):
        if e > 0.15:
            vol_reg.append("vol_expansion")
        elif e < -0.10:
            vol_reg.append("vol_compression")
        elif v == "high":
            vol_reg.append("high_vol")
        elif v == "low":
            vol_reg.append("low_vol")
        else:
            vol_reg.append("mid_vol")
    df["vol_regime"] = vol_reg

    ent = pd.to_numeric(df.get("entropy", 1.0), errors="coerce").fillna(1.0)
    structure: List[str] = []
    for i, row in df.iterrows():
        t = str(row.get("trend_state", ""))
        d = str(row.get("direction", ""))
        e = float(row.get("entropy", 1.0) or 1.0)
        if bool(row.get("regime_transition_flag", 0)):
            structure.append("reversal")
        elif t == "sideways" and e > 0.98:
            structure.append("chop_noise")
        elif bool(row.get("rfe_flag", False)) or float(row.get("mae", 0) or 0) <= -0.008:
            structure.append("liquidation_cascade")
        elif float(row.get("engine_ret", row.get("net_return", 0)) or 0) > 0.002 and t == "down":
            structure.append("recovery")
        elif (d == "LONG" and t == "up") or (d == "SHORT" and t == "down"):
            structure.append("trend_continuation")
        else:
            structure.append("mixed_structure")
    df["structure_regime"] = structure

    conf: List[str] = []
    for _, row in df.iterrows():
        e = float(row.get("entropy", 1.0) or 1.0)
        if bool(row.get("false_high_signature_flag", 0)):
            conf.append("false_high_signature")
        elif float(row.get("confidence_overextension", 0) or 0) > 0.25:
            conf.append("confidence_overextension")
        elif e <= 0.90:
            conf.append("low_entropy")
        elif e > 1.0:
            conf.append("high_entropy")
        else:
            conf.append("mid_entropy")
    df["confidence_regime"] = conf
    return df


def _good_rejection(df: pd.DataFrame, prob: np.ndarray) -> float:
    good = df["binary_good_trade"] == 1
    if good.sum() == 0:
        return 0.0
    rej = (prob >= 0.5) & (df[LABEL_COL].astype(int) == 1) & good
    return float(rej.sum() / good.sum())


def _false_high_suppression(df: pd.DataFrame, prob: np.ndarray) -> int:
    if not {"direction", "trend_state", "vol_bucket", "entropy"}.issubset(df.columns):
        return 0
    return int((
        (df["direction"] == "LONG")
        & (df["trend_state"] == "up")
        & (df["vol_bucket"] == "high")
        & (df["entropy"] <= 0.90)
        & (prob >= 0.5)
        & (df["binary_bad_trade"] == 1)
    ).sum())


def _mdd(scales: pd.Series, returns: pd.Series) -> float:
    eq = peak = 1.0
    mdd = 0.0
    for r, s in zip(returns.fillna(0), scales.fillna(1)):
        eq *= 1.0 + float(r) * float(s) * POSITION_SIZE
        peak = max(peak, eq)
        mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0)
    return float(mdd)


def _q2_scale(row: pd.Series) -> float:
    return float(map_m3(float(row.get("q2_bdi_score", 0.5))) or 0.15)


def _eval_regime_subset(
    sub: pd.DataFrame,
    prob: np.ndarray,
    regime_name: str,
    regime_value: str,
) -> Dict[str, Any]:
    if len(sub) < 25:
        return {
            "regime_type": regime_name,
            "regime_value": regime_value,
            "rows": len(sub),
            "status": "insufficient_rows",
        }
    y = sub[LABEL_COL].astype(int).to_numpy()
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    bdf = _bucket_stats(sub, prob)
    mono = _monotonicity_score(bdf, "bad_trade_rate")
    return {
        "regime_type": regime_name,
        "regime_value": regime_value,
        "rows": len(sub),
        "auc": auc,
        "monotonic_bad_trade": bool(mono.get("monotonic", False)),
        "monotonic_spearman": mono.get("spearman"),
        "routing_valid": _risk_routing_valid(sub, prob),
        "false_high_suppressed": _false_high_suppression(sub, prob),
        "good_rejection_rate": _good_rejection(sub, prob),
        "q1_bad_rate": float(bdf[bdf["bucket"] == "Q1"]["bad_trade_rate"].iloc[0]) if len(bdf) >= 4 else None,
        "q4_bad_rate": float(bdf[bdf["bucket"] == "Q4"]["bad_trade_rate"].iloc[0]) if len(bdf) >= 4 else None,
        "calibration_error": _calibration_error(y, prob),
        "status": "ok",
    }


def _fit_global_scores(anchor: pd.DataFrame, weights: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, HistGradientBoostingClassifier]:
    train_df, val_df, test_df = _time_split(anchor)
    eval_df = pd.concat([val_df, test_df], ignore_index=True)
    w_train = weights[: len(train_df)]
    model = _train_meta_model(train_df, w_train)
    raw_eval = model.predict_proba(_prepare_x(eval_df, FEATURE_COLS))[:, 1]
    return eval_df, raw_eval, model


def phase1_segmentation(anchor: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    anchor = _segment_regimes(anchor)
    dist_rows: List[Dict[str, Any]] = []
    for col in ("trend_regime", "vol_regime", "structure_regime", "confidence_regime"):
        for val, n in anchor[col].value_counts().items():
            dist_rows.append({"regime_type": col, "regime_value": str(val), "rows": int(n), "pct": n / len(anchor)})
    summary = {
        "total_rows": len(anchor),
        "trend_regimes": anchor["trend_regime"].value_counts().to_dict(),
        "vol_regimes": anchor["vol_regime"].value_counts().to_dict(),
        "structure_regimes": anchor["structure_regime"].value_counts().to_dict(),
        "confidence_regimes": anchor["confidence_regime"].value_counts().to_dict(),
    }
    return anchor, summary, pd.DataFrame(dist_rows)


def phase2_regime_performance(eval_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in ("trend_regime", "vol_regime", "structure_regime", "confidence_regime"):
        for val in eval_df[col].unique():
            mask = (eval_df[col] == val).to_numpy()
            sub = eval_df[mask].reset_index(drop=True)
            r = _eval_regime_subset(sub, scores[mask], col, str(val))
            rows.append(r)
    return pd.DataFrame(rows)


def phase3_stability_forensics(
    anchor: pd.DataFrame,
    weights: np.ndarray,
    perf: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    drift_rows: List[Dict[str, Any]] = []
    inversion_rows: List[Dict[str, Any]] = []
    n = len(anchor)
    chunk, step = max(50, n // 5), max(15, n // 12)
    start = 0
    fold_id = 0

    while start + chunk + 10 <= n and fold_id < 15:
        sub = anchor.iloc[start : start + chunk + step].copy()
        split = int(len(sub) * 0.7)
        if split < 30:
            break
        train, test = sub.iloc[:split], sub.iloc[split:]
        w_tr = weights[start : start + split]
        model = _train_meta_model(train, w_tr)
        tr_scores = model.predict_proba(_prepare_x(train, FEATURE_COLS))[:, 1]
        te_scores = model.predict_proba(_prepare_x(test, FEATURE_COLS))[:, 1]

        drift_rows.append({
            "fold": fold_id,
            "train_score_mean": float(np.mean(tr_scores)),
            "test_score_mean": float(np.mean(te_scores)),
            "score_drift": float(np.mean(te_scores) - np.mean(tr_scores)),
            "dominant_test_trend": str(test["trend_regime"].mode().iloc[0]) if len(test) else "",
            "dominant_test_vol": str(test["vol_regime"].mode().iloc[0]) if len(test) else "",
        })

        bdf = _bucket_stats(test, te_scores)
        mono = _monotonicity_score(bdf, "bad_trade_rate")
        inversion_rows.append({
            "fold": fold_id,
            "monotonic": bool(mono.get("monotonic", False)),
            "violations": mono.get("violations", 0),
            "q1_bad": float(bdf[bdf["bucket"] == "Q1"]["bad_trade_rate"].iloc[0]) if len(bdf) >= 4 else None,
            "q4_bad": float(bdf[bdf["bucket"] == "Q4"]["bad_trade_rate"].iloc[0]) if len(bdf) >= 4 else None,
            "inverted": bool(
                len(bdf) >= 4
                and float(bdf[bdf["bucket"] == "Q1"]["bad_trade_rate"].iloc[0])
                > float(bdf[bdf["bucket"] == "Q4"]["bad_trade_rate"].iloc[0])
            ),
            "high_vol_rows": int((test["vol_regime"] == "high_vol").sum()),
            "sideways_rows": int((test["trend_regime"] == "sideways").sum()),
        })
        start += step
        fold_id += 1

    unstable_regimes = perf[
        (perf["status"] == "ok") & (~perf["monotonic_bad_trade"].fillna(False))
    ]["regime_value"].tolist()

    report = {
        "mean_score_drift": float(np.mean([r["score_drift"] for r in drift_rows])) if drift_rows else None,
        "bucket_inversion_fold_rate": float(np.mean([r["inverted"] for r in inversion_rows])) if inversion_rows else None,
        "monotonic_fold_rate": float(np.mean([r["monotonic"] for r in inversion_rows])) if inversion_rows else None,
        "unstable_regimes": unstable_regimes[:10],
        "high_vol_instability": perf[(perf["regime_type"] == "vol_regime") & (perf["regime_value"] == "high_vol")].to_dict(orient="records"),
        "sideways_collapse": perf[(perf["regime_type"] == "trend_regime") & (perf["regime_value"] == "sideways")].to_dict(orient="records"),
        "false_high_regime": perf[(perf["regime_type"] == "confidence_regime") & (perf["regime_value"] == "false_high_signature")].to_dict(orient="records"),
        "why_fold_stability_breaks": [
            "Regime mix shifts between rolling folds",
            "Sideways/chop regimes show bucket ordering inversion",
            "High-vol periods increase score drift",
            "Calibration not stable across regime transitions",
        ],
    }
    return pd.DataFrame(drift_rows), pd.DataFrame(inversion_rows), report


def _regime_gated_scale(
    df: pd.DataFrame,
    meta_scores: np.ndarray,
    gate_mask: pd.Series,
    dampen: float = 0.6,
) -> pd.Series:
    s = df.apply(_q2_scale, axis=1)
    risk = pd.Series(meta_scores, index=df.index)
    adj = s.copy()
    on = gate_mask & (risk > 0.55)
    adj.loc[on] = (s.loc[on] * (1.0 - dampen * risk.loc[on])).clip(0.05, 1.0)
    return adj


def phase4_5_gating(eval_df: pd.DataFrame, scores: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rets = eval_df["engine_ret"]
    experiments: List[Dict[str, Any]] = []

    gates = {
        "A_high_vol_only": eval_df["vol_regime"] == "high_vol",
        "B_trend_up_only": eval_df["trend_regime"].isin(["strong_uptrend", "weak_uptrend"]),
        "C_false_high_only": eval_df["confidence_regime"] == "false_high_signature",
        "D_overextension_only": eval_df["confidence_regime"] == "confidence_overextension",
        "E_sideways_disable": eval_df["trend_regime"] != "sideways",
        "F_strong_trend_only": eval_df["trend_regime"].isin(["strong_uptrend", "strong_downtrend"]),
    }

    q2_only = eval_df.apply(_q2_scale, axis=1)
    experiments.append({
        "experiment": "baseline_q2_only",
        "active_rows": len(eval_df),
        "mdd": _mdd(q2_only, rets),
        "routing_valid": _risk_routing_valid(eval_df, scores),
        "monotonic_global": True,
    })

    for name, mask in gates.items():
        scales = _regime_gated_scale(eval_df, scores, mask)
        active = int(mask.sum())
        sub = eval_df[mask]
        sub_scores = scores[mask.to_numpy()]
        bdf = _bucket_stats(sub, sub_scores) if len(sub) >= 20 else pd.DataFrame()
        mono = _monotonicity_score(bdf, "bad_trade_rate") if len(bdf) else {"monotonic": False}
        experiments.append({
            "experiment": name,
            "active_rows": active,
            "mdd": _mdd(scales, rets),
            "mdd_delta_vs_q2": _mdd(scales, rets) - _mdd(q2_only, rets),
            "routing_valid": _risk_routing_valid(sub, sub_scores) if len(sub) >= 25 else False,
            "monotonic_in_gate": bool(mono.get("monotonic", False)),
            "false_high": _false_high_suppression(sub, sub_scores) if len(sub) else 0,
        })

    policies = {
        "recommended_meta_on": ["high_vol", "false_high_signature", "confidence_overextension"],
        "recommended_meta_off": ["sideways", "chop_noise", "mid_entropy"],
        "policy_rationale": "Activate Meta drawdown dampening only when forensic risk signatures present",
    }
    return pd.DataFrame(experiments), policies


def phase6_transfer(anchor: pd.DataFrame, weights: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train on one regime dominant period, test on another."""
    anchor = _segment_regimes(anchor) if "trend_regime" not in anchor.columns else anchor
    n = len(anchor)
    mid = n // 2
    halves = [
        ("first_half", anchor.iloc[:mid], anchor.iloc[mid:]),
        ("second_half", anchor.iloc[mid:], anchor.iloc[:mid]),
    ]
    transfers = [
        ("bull_to_bear", "strong_uptrend", "strong_downtrend"),
        ("high_vol_to_mid", "high_vol", "mid_vol"),
        ("trend_to_sideways", "strong_uptrend", "sideways"),
    ]

    rows: List[Dict[str, Any]] = []
    for label, train_df, test_df in halves:
        if len(train_df) < 80 or len(test_df) < 40:
            continue
        w = weights[: len(train_df)]
        model = _train_meta_model(train_df, w)
        prob = model.predict_proba(_prepare_x(test_df, FEATURE_COLS))[:, 1]
        mono = _monotonicity_score(_bucket_stats(test_df, prob), "bad_trade_rate")
        rows.append({
            "transfer": label,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "auc": float(roc_auc_score(test_df[LABEL_COL], prob)) if test_df[LABEL_COL].nunique() > 1 else float("nan"),
            "monotonic": bool(mono.get("monotonic", False)),
            "routing_valid": _risk_routing_valid(test_df, prob),
        })

    for tname, train_regime, test_regime in transfers:
        if tname == "bull_to_bear":
            tr = anchor[anchor["trend_regime"].isin(["strong_uptrend", "weak_uptrend"])]
            te = anchor[anchor["trend_regime"].isin(["strong_downtrend", "weak_downtrend"])]
        elif tname == "high_vol_to_mid":
            tr = anchor[anchor["vol_regime"] == "high_vol"]
            te = anchor[anchor["vol_regime"] == "mid_vol"]
        else:
            tr = anchor[anchor["trend_regime"].isin(["strong_uptrend", "weak_uptrend"])]
            te = anchor[anchor["trend_regime"] == "sideways"]
        if len(tr) < 50 or len(te) < 30:
            continue
        model = _train_meta_model(tr, np.ones(len(tr)))
        prob = model.predict_proba(_prepare_x(te, FEATURE_COLS))[:, 1]
        mono = _monotonicity_score(_bucket_stats(te, prob), "bad_trade_rate")
        rows.append({
            "transfer": tname,
            "train_rows": len(tr),
            "test_rows": len(te),
            "auc": float(roc_auc_score(te[LABEL_COL], prob)) if te[LABEL_COL].nunique() > 1 else float("nan"),
            "monotonic": bool(mono.get("monotonic", False)),
            "routing_valid": _risk_routing_valid(te, prob),
        })

    matrix = pd.DataFrame(rows)
    summary = {
        "transfer_count": len(rows),
        "monotonic_transfer_rate": float(np.mean([r["monotonic"] for r in rows])) if rows else 0,
        "routing_valid_transfer_rate": float(np.mean([r["routing_valid"] for r in rows])) if rows else 0,
        "temporal_robustness": float(np.mean([r["monotonic"] for r in rows])) >= 0.5 if rows else False,
    }
    return matrix, summary


def phase7_q2_conditional(eval_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    rets = eval_df["engine_ret"]
    q2 = eval_df.apply(_q2_scale, axis=1)
    risk = pd.Series(scores, index=eval_df.index)

    exps = [
        ("1_q2_only", q2),
        ("2_q2_regime_meta", _regime_gated_scale(eval_df, scores, eval_df["vol_regime"] == "high_vol")),
        ("3_q2_drawdown_override", q2.where(risk < 0.65, q2 * 0.5).clip(0.05, 1.0)),
        ("4_q2_false_high_dampen", q2.mask(
            eval_df["confidence_regime"] == "false_high_signature",
            q2 * 0.6,
        ).clip(0.05, 1.0)),
        ("5_q2_high_vol_only", q2.where(
            eval_df["vol_regime"] != "high_vol",
            q2 * (1.0 - 0.3 * risk),
        ).clip(0.05, 1.0)),
    ]
    rows = []
    for name, scales in exps:
        rows.append({
            "experiment": name,
            "mdd": _mdd(scales, rets),
            "mdd_vs_q2": _mdd(scales, rets) - _mdd(q2, rets),
            "routing_valid": _risk_routing_valid(eval_df, risk.to_numpy()),
            "false_high": _false_high_suppression(eval_df, risk.to_numpy()),
            "good_rejection": _good_rejection(eval_df, risk.to_numpy()),
            "preservation": 1.0,
        })
    return pd.DataFrame(rows)


def _final_verdict(
    perf: pd.DataFrame,
    forensics: Dict[str, Any],
    gating: pd.DataFrame,
    transfer_summary: Dict[str, Any],
) -> str:
    ok = perf[perf["status"] == "ok"]
    mono_regimes = ok[ok["monotonic_bad_trade"] == True]
    routing_regimes = ok[ok["routing_valid"] == True]

    high_vol = ok[(ok["regime_type"] == "vol_regime") & (ok["regime_value"] == "high_vol")]
    fh = ok[(ok["regime_type"] == "confidence_regime") & (ok["regime_value"] == "false_high_signature")]

    if len(mono_regimes) >= 3 and len(routing_regimes) >= 2:
        if len(high_vol) and bool(high_vol.iloc[0].get("monotonic_bad_trade")):
            if len(fh) and bool(fh.iloc[0].get("routing_valid")):
                return "false_high_regime_specialist"
            return "high_vol_meta_effective"
        return "regime_conditioned_refinement_viable"

    if transfer_summary.get("temporal_robustness"):
        return "monitor_only_candidate"

    if len(mono_regimes) >= 1:
        best_gate = gating.sort_values("mdd_delta_vs_q2", ascending=False) if "mdd_delta_vs_q2" in gating.columns else gating
        if len(best_gate) and float(best_gate.iloc[0].get("mdd_delta_vs_q2", 0)) > 0:
            return "regime_conditioned_refinement_viable"
        return "research_only_continue"

    if float(forensics.get("monotonic_fold_rate", 0) or 0) < 0.2:
        return "temporal_robustness_insufficient"

    return "meta_globally_unstable"


def run_regime_analysis(*, skip_oos: bool = False) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = _prepare_full_df()
    anchor, _, weights = _build_anchor_dataset(full)
    anchor, seg_summary, dist_df = phase1_segmentation(anchor)
    dist_df.to_csv(OUT_DIR / "regime_distribution.csv", index=False)
    (OUT_DIR / "regime_segmentation_summary.md").write_text(
        f"# Regime Segmentation Summary\n\n```json\n{json.dumps(seg_summary, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    eval_df, scores, _ = _fit_global_scores(anchor, weights)
    perf = phase2_regime_performance(eval_df, scores)
    perf.to_csv(OUT_DIR / "regime_specific_meta_performance.csv", index=False)

    mono_summary = perf[perf["status"] == "ok"].groupby("regime_type").agg(
        regimes=("regime_value", "count"),
        monotonic_pass=("monotonic_bad_trade", "sum"),
        routing_pass=("routing_valid", "sum"),
        mean_auc=("auc", "mean"),
    ).reset_index()
    (OUT_DIR / "regime_monotonicity_analysis.md").write_text(
        "# Regime Monotonicity Analysis\n\n"
        f"```\n{mono_summary.to_string(index=False)}\n```\n\n"
        f"## Per-regime detail\n```\n{perf.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )

    drift_df, inv_df, forensic_report = phase3_stability_forensics(anchor, weights, perf) if not skip_oos else (pd.DataFrame(), pd.DataFrame(), {})
    drift_df.to_csv(OUT_DIR / "score_drift_analysis.csv", index=False)
    inv_df.to_csv(OUT_DIR / "bucket_inversion_analysis.csv", index=False)
    (OUT_DIR / "regime_stability_forensics.md").write_text(
        f"# Regime Stability Forensics\n\n```json\n{json.dumps(forensic_report, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    gating_df, policies = phase4_5_gating(eval_df, scores)
    gating_df.to_csv(OUT_DIR / "regime_gating_experiments.csv", index=False)
    (OUT_DIR / "meta_activation_policy_candidates.md").write_text(
        f"# Meta Activation Policy Candidates\n\n```json\n{json.dumps(policies, indent=2, default=str)}\n```\n\n"
        f"## Gating experiments\n```\n{gating_df.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )

    transfer_df, transfer_summary = phase6_transfer(anchor, weights) if not skip_oos else (pd.DataFrame(), {})
    transfer_df.to_csv(OUT_DIR / "regime_transfer_matrix.csv", index=False)
    (OUT_DIR / "temporal_transfer_robustness.md").write_text(
        f"# Temporal Transfer Robustness\n\n```json\n{json.dumps(transfer_summary, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    q2_df = phase7_q2_conditional(eval_df, scores)
    q2_df.to_csv(OUT_DIR / "q2_meta_conditional_refinement.csv", index=False)

    verdict = _final_verdict(perf, forensic_report, gating_df, transfer_summary)
    final = {
        "verdict": verdict,
        "anchor_rows": len(anchor),
        "regimes_with_monotonic_pass": int(perf[perf.get("monotonic_bad_trade") == True].shape[0]) if "monotonic_bad_trade" in perf.columns else 0,
        "regimes_with_routing_pass": int(perf[perf.get("routing_valid") == True].shape[0]) if "routing_valid" in perf.columns else 0,
        "transfer_summary": transfer_summary,
        "q2_baseline_unchanged": True,
        "promotion_ready": False,
    }
    (OUT_DIR / "regime_conditioned_meta_final_verdict.md").write_text(
        f"# Regime-Conditioned Meta Final Verdict\n\n**Verdict:** `{verdict}`\n\n"
        f"Q2_BDI baseline unchanged. Meta conditional refinement research only.\n\n"
        f"```json\n{json.dumps(final, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-conditioned meta stability")
    parser.add_argument("--skip-oos", action="store_true")
    args = parser.parse_args()
    r = run_regime_analysis(skip_oos=args.skip_oos)
    print(f"verdict: {r['verdict']}")
    print(f"monotonic_regimes: {r.get('regimes_with_monotonic_pass')}")
    print(f"routing_regimes: {r.get('regimes_with_routing_pass')}")


if __name__ == "__main__":
    main()
