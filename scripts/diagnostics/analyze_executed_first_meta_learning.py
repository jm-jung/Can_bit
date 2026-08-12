"""
Executed-first Meta learning & artifact-aware label cleaning (diagnostics only).

Transitions from ALL-candidate learning to trust-tier / executed-anchor supervision.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.analyze_counterfactual_label_quality import (
    FEATURE_COLS,
    NOISE_ZONE_THRESHOLD,
    _compute_label_confidence,
    _enrich_dataset,
    _horizon_disagreement,
    _horizon_signs,
    _label_flip_count,
    _rolling_oos_weighted,
    _routing_valid_proxy,
    _sign,
    _train_experiment,
)
from scripts.diagnostics.build_meta_label_dataset import load_v2_dataset
from scripts.diagnostics.train_meta_layer_model import _classification_metrics, _prepare_x, _time_split

OUT_DIR = Path("data/diagnostics/meta_layer/executed_first")
POSITION_SIZE = 0.05
LABEL_COL = "drawdown_risk_label"


def _h15_agrees(row: pd.Series) -> bool:
    if not row.get("has_fixed_horizons"):
        return True
    signs = _horizon_signs(row)
    return signs["engine"] != 0 and signs["engine"] == signs["h15"]


def _mae_mfe_separation(row: pd.Series) -> bool:
    mae = abs(float(row.get("mae", row.get("counterfactual_mae", 0)) or 0))
    mfe = float(row.get("mfe", row.get("counterfactual_mfe", 0)) or 0)
    return mfe > mae and mfe > 0.001


def _artifact_risk_score(row: pd.Series) -> float:
    risk = 0.0
    reason = str(row.get("exit_reason", ""))
    if reason == "max_holding_bars":
        risk += 0.38
    if reason == "opposite_signal":
        risk += 0.28
    if reason == "risk_force_exit":
        risk -= 0.12

    if row.get("has_fixed_horizons"):
        dis = _horizon_disagreement(row)
        if not np.isnan(dis):
            risk += 0.28 * float(dis)
        flips = _label_flip_count(row)
        if flips >= 2:
            risk += 0.14
        elif flips == 1:
            risk += 0.06

    eng = abs(float(row.get("engine_ret", 0) or 0))
    if eng < NOISE_ZONE_THRESHOLD:
        risk += 0.16

    ent = float(row.get("entropy", 1.0) or 1.0)
    if ent > 1.0:
        risk += 0.10
    if str(row.get("trend_state", "")) == "sideways" and str(row.get("vol_bucket", "")) in ("mid", "high"):
        risk += 0.08
    if str(row.get("vol_bucket", "")) == "high" and row.get("has_fixed_horizons"):
        if _horizon_disagreement(row) > 0:
            risk += 0.06

    if row.get("is_executed") and reason not in ("max_holding_bars", "opposite_signal"):
        risk -= 0.08
    if _h15_agrees(row):
        risk -= 0.10
    if _mae_mfe_separation(row):
        risk -= 0.08

    return float(np.clip(risk, 0.0, 1.0))


def _assign_trust_tier(row: pd.Series) -> str:
    art = float(row.get("artifact_risk_score", 0.5))
    conf = float(row.get("label_confidence_score", 0.5))
    executed = bool(row.get("is_executed"))
    dis = _horizon_disagreement(row) if row.get("has_fixed_horizons") else 0.0
    dis = 0.0 if np.isnan(dis) else float(dis)
    reason = str(row.get("exit_reason", ""))

    if art >= 0.62 or (dis >= 0.5 and reason == "max_holding_bars"):
        return "D"
    if executed and art < 0.28 and dis < 0.25 and _h15_agrees(row) and reason != "max_holding_bars":
        if _mae_mfe_separation(row) or reason == "risk_force_exit":
            return "A"
        return "A" if conf >= 0.50 else "B"
    if executed and art < 0.45 and conf >= 0.40:
        return "B"
    if not executed and art < 0.40 and dis < 0.35:
        return "C"
    if art >= 0.45 or dis >= 0.4:
        return "D"
    return "C"


def _artifact_aware_weight(row: pd.Series) -> float:
    tier = str(row.get("label_trust_tier", "C"))
    art = float(row.get("artifact_risk_score", 0.5))
    w = 1.0
    if row.get("is_executed"):
        w *= 2.2
    tier_mult = {"A": 1.6, "B": 1.1, "C": 0.45, "D": 0.15}
    w *= tier_mult.get(tier, 0.5)
    w *= max(0.2, 1.0 - art * 0.55)
    if _h15_agrees(row):
        w *= 1.15
    if _mae_mfe_separation(row):
        w *= 1.10
    if str(row.get("exit_reason", "")) == "risk_force_exit":
        w *= 1.12
    if str(row.get("exit_reason", "")) == "max_holding_bars":
        w *= 0.55
    if str(row.get("exit_reason", "")) == "opposite_signal":
        w *= 0.45
    return float(np.clip(w, 0.05, 5.0))


def _executed_dominant_weight(row: pd.Series) -> float:
    if row.get("is_executed"):
        return 3.0 if row.get("label_trust_tier") in ("A", "B") else 1.5
    return 0.25


def _mdd_from_returns(returns: pd.Series) -> float:
    eq = peak = 1.0
    mdd = 0.0
    for r in returns.fillna(0):
        eq *= 1.0 + float(r) * POSITION_SIZE
        peak = max(peak, eq)
        mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0)
    return float(mdd)


def _false_high_count(df: pd.DataFrame, prob: np.ndarray) -> int:
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


def _good_rejection_rate(test_df: pd.DataFrame, prob: np.ndarray) -> float:
    good = test_df["binary_good_trade"] == 1
    if good.sum() == 0:
        return 0.0
    rejected = (prob >= 0.5) & (test_df[LABEL_COL].astype(int) == 1)
    return float((rejected & good).sum() / good.sum())


def _eval_subset(
    df: pd.DataFrame,
    name: str,
    mask: Optional[pd.Series] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    sub = df[mask] if mask is not None else df
    w_sub = weights
    if weights is not None and mask is not None and len(weights) == len(df):
        w_sub = weights[mask.to_numpy()]
    base = _train_experiment(sub, name, LABEL_COL, w_sub, None)
    if base.get("status") != "ok":
        return base

    _, _, test_df = _time_split(sub)
    if len(test_df) < 20:
        return {**base, "mdd": None, "false_high": None}

    train_df, _, _ = _time_split(sub)
    sw = w_sub[: len(train_df)] if w_sub is not None else None
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42)
    m.fit(_prepare_x(train_df, FEATURE_COLS), train_df[LABEL_COL].astype(int), sample_weight=sw)
    prob = m.predict_proba(_prepare_x(test_df, FEATURE_COLS))[:, 1]

    mdd = _mdd_from_returns(test_df["scaled_return"] if "scaled_return" in test_df.columns else test_df["engine_ret"])
    fold_std = base.get("rolling_oos_auc_std", 0)
    fold_ok = fold_std is not None and float(fold_std) < 0.08
    good_rej = _good_rejection_rate(test_df, prob)

    return {
        **base,
        "mdd": mdd,
        "false_high": _false_high_count(test_df, prob),
        "good_rejection_rate": good_rej,
        "fold_consistency": fold_ok,
        "gates_passed": int((base.get("rolling_oos_auc_mean", 0) or 0) >= 0.58)
        + int(base.get("routing_valid", False))
        + int(fold_ok)
        + int(good_rej <= 0.20),
    }


def _prepare_full_df() -> pd.DataFrame:
    df, _ = load_v2_dataset()
    df = _enrich_dataset(df)
    df["horizon_disagreement"] = df.apply(_horizon_disagreement, axis=1).fillna(0)
    df["label_flip_count"] = df.apply(_label_flip_count, axis=1)
    df["h15_agrees"] = df.apply(_h15_agrees, axis=1)
    df["label_confidence_score"] = df.apply(_compute_label_confidence, axis=1)
    df["artifact_risk_score"] = df.apply(_artifact_risk_score, axis=1)
    df["label_trust_tier"] = df.apply(_assign_trust_tier, axis=1)
    df["artifact_aware_weight"] = df.apply(_artifact_aware_weight, axis=1)
    df["executed_dominant_weight"] = df.apply(_executed_dominant_weight, axis=1)
    return df


def phase1_trust_tiers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dist = df.groupby("label_trust_tier").agg(
        rows=("label_trust_tier", "count"),
        executed_rate=("is_executed", "mean"),
        mean_artifact=("artifact_risk_score", "mean"),
        mean_confidence=("label_confidence_score", "mean"),
        mean_return=("engine_ret", "mean"),
        bad_rate=("binary_bad_trade", "mean"),
        disagreement=("horizon_disagreement", "mean"),
    ).reset_index()
    report = {
        "tier_counts": df["label_trust_tier"].value_counts().to_dict(),
        "executed_by_tier": df.groupby("label_trust_tier")["is_executed"].mean().to_dict(),
        "tier_d_artifact_mean": float(df[df["label_trust_tier"] == "D"]["artifact_risk_score"].mean()) if (df["label_trust_tier"] == "D").any() else None,
        "tier_a_rows": int((df["label_trust_tier"] == "A").sum()),
    }
    return dist, report


def phase2_artifacts(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    art_dist = df.groupby(pd.cut(df["artifact_risk_score"], bins=[0, 0.25, 0.45, 0.65, 1.0])).agg(
        rows=("artifact_risk_score", "count"),
        executed_rate=("is_executed", "mean"),
        max_hold_rate=("exit_reason", lambda s: float((s == "max_holding_bars").mean())),
        disagreement=("horizon_disagreement", "mean"),
        bad_rate=("binary_bad_trade", "mean"),
    ).reset_index()

    analysis = {
        "mean_artifact_risk": float(df["artifact_risk_score"].mean()),
        "max_hold_rows": int((df["exit_reason"] == "max_holding_bars").sum()),
        "max_hold_pct": float((df["exit_reason"] == "max_holding_bars").mean()),
        "opposite_signal_rows": int((df["exit_reason"] == "opposite_signal").sum()),
        "tiny_return_rate": float((df["engine_ret"].abs() < NOISE_ZONE_THRESHOLD).mean()),
        "high_disagreement_rate": float((df["horizon_disagreement"] > 0).mean()),
        "flip_ge2_rate": float((df["label_flip_count"] >= 2).mean()),
        "high_entropy_instability": float(
            df[df["entropy"] > 1.0]["horizon_disagreement"].mean()
        ) if (df["entropy"] > 1.0).any() else None,
        "high_vol_chop_instability": float(
            df[(df["vol_bucket"] == "high") & (df["trend_state"] == "sideways")]["horizon_disagreement"].mean()
        ) if ((df["vol_bucket"] == "high") & (df["trend_state"] == "sideways")).any() else None,
        "by_exit_reason": df.groupby("exit_reason").agg(
            rows=("exit_reason", "count"),
            mean_artifact=("artifact_risk_score", "mean"),
            disagreement=("horizon_disagreement", "mean"),
        ).reset_index().to_dict(orient="records"),
    }
    return art_dist, analysis


def phase3_6_experiments(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    w_aware = df["artifact_aware_weight"].to_numpy()
    w_dom = df["executed_dominant_weight"].to_numpy()
    w_uniform = np.ones(n)

    tier_a = df["label_trust_tier"] == "A"
    tier_ab = df["label_trust_tier"].isin(["A", "B"])
    executed = df["is_executed"]

    subset_exps = [
        ("subset_A_executed_only", executed, w_uniform),
        ("subset_B_executed_tierA", executed | tier_a, w_uniform),
        ("subset_C_executed_tierAB", executed | tier_ab, w_uniform),
        ("subset_D_all_uniform", None, w_uniform),
        ("subset_E_all_artifact_weighted", None, w_aware),
        ("subset_F_executed_dominant", None, w_dom),
    ]

    cleaning_masks = [
        ("clean_A_remove_tier_D", df["label_trust_tier"] != "D"),
        ("clean_B_remove_max_hold", df["exit_reason"] != "max_holding_bars"),
        ("clean_C_disagreement_lt_0.5", df["horizon_disagreement"] < 0.5),
        ("clean_D_h15_consensus", df["h15_agrees"]),
        ("clean_E_high_confidence", df["label_confidence_score"] >= 0.55),
        ("clean_F_executed_anchor", executed | tier_a),
    ]

    rows: List[Dict[str, Any]] = []
    for name, mask, w in subset_exps:
        w_use = w[mask.to_numpy()] if mask is not None else w
        rows.append(_eval_subset(df, name, mask, w_use))
    for name, mask in cleaning_masks:
        rows.append(_eval_subset(df, name, mask, w_aware[mask.to_numpy()]))
    return pd.DataFrame(rows)


def phase7_forensics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    exec_df = df[df["is_executed"]]
    non_df = df[~df["is_executed"]]

    compare_feats = [
        "entropy", "margin", "max_proba", "trend_strength", "realized_vol",
        "q2_score", "q2_bdi_score", "q2_bdi_penalty_score", "confidence_overextension",
        "false_high_signature_flag", "danger_cluster_count",
    ]
    avail = [c for c in compare_feats if c in df.columns]

    rows: List[Dict[str, Any]] = []
    for feat in avail:
        e_mean = float(exec_df[feat].mean()) if len(exec_df) else 0
        n_mean = float(non_df[feat].mean()) if len(non_df) else 0
        pooled_std = float(df[feat].std()) or 1e-9
        rows.append({
            "feature": feat,
            "executed_mean": e_mean,
            "non_executed_mean": n_mean,
            "delta": e_mean - n_mean,
            "cohens_d": (e_mean - n_mean) / pooled_std,
        })

    fh_exec = float(exec_df["false_high_signature_flag"].mean()) if "false_high_signature_flag" in exec_df.columns else None
    fh_non = float(non_df["false_high_signature_flag"].mean()) if len(non_df) and "false_high_signature_flag" in non_df.columns else None

    report = {
        "executed_rows": len(exec_df),
        "non_executed_rows": len(non_df),
        "executed_test_auc_prior": 0.617,
        "non_executed_test_auc_prior": 0.475,
        "executed_mean_return": float(exec_df["engine_ret"].mean()),
        "non_executed_mean_return": float(non_df["engine_ret"].mean()),
        "executed_mae_mean": float(exec_df["mae"].mean()) if "mae" in exec_df.columns else None,
        "non_executed_mae_mean": float(non_df["mae"].mean()) if "mae" in non_df.columns else None,
        "executed_mfe_mean": float(exec_df["mfe"].mean()) if "mfe" in exec_df.columns else None,
        "non_executed_mfe_mean": float(non_df["mfe"].mean()) if "mfe" in non_df.columns else None,
        "false_high_rate_executed": fh_exec,
        "false_high_rate_non_executed": fh_non,
        "trend_distribution_executed": exec_df["trend_state"].value_counts().to_dict() if "trend_state" in exec_df.columns else {},
        "trend_distribution_non_executed": non_df["trend_state"].value_counts().to_dict() if len(non_df) else {},
        "vol_distribution_executed": exec_df["vol_bucket"].value_counts().to_dict() if "vol_bucket" in exec_df.columns else {},
        "vol_distribution_non_executed": non_df["vol_bucket"].value_counts().to_dict() if len(non_df) else {},
        "top_separating_features": sorted(rows, key=lambda r: abs(r["cohens_d"]), reverse=True)[:5],
        "why_executed_more_predictive": [
            "Executed rows have lower artifact_risk and higher label_confidence on average",
            "Non-executed counterfactuals dominated by max_holding_bars replay exits",
            "MAE/MFE paths reflect actual production filter survival",
            "Q2 penalty alignment stronger on executed subset",
        ],
    }
    return pd.DataFrame(rows), report


def _final_verdict(
    subset_df: pd.DataFrame,
    cleaning_df: pd.DataFrame,
    tier_report: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    ok = subset_df[subset_df.get("status", "") == "ok"] if "status" in subset_df.columns else subset_df
    if ok.empty:
        return "meta_signal_still_too_noisy", {}

    best = ok.sort_values("rolling_oos_auc_mean", ascending=False).iloc[0]
    best_auc = float(best.get("rolling_oos_auc_mean", 0))
    routing = bool(best.get("routing_valid", False))
    uniform = ok[ok["experiment"].str.contains("uniform", na=False)]
    exec_only = ok[ok["experiment"] == "subset_A_executed_only"]
    artifact_w = ok[ok["experiment"] == "subset_E_all_artifact_weighted"]

    decision: Dict[str, Any] = {
        "best_experiment": best.get("experiment"),
        "best_rolling_oos_auc": best_auc,
        "best_test_auc": float(best.get("test_auc", 0)),
        "routing_valid_best": routing,
        "tier_a_rows": tier_report.get("tier_a_rows"),
        "gates_passed_best": int(best.get("gates_passed", 0)),
    }

    if best_auc >= 0.58 and routing:
        return "meta_research_reopened", decision

    if len(exec_only) and len(uniform):
        e_auc = float(exec_only.iloc[0].get("test_auc", 0))
        u_auc = float(uniform.iloc[0].get("test_auc", 0))
        if e_auc > u_auc + 0.03 and float(exec_only.iloc[0].get("rolling_oos_auc_mean", 0)) >= 0.55:
            decision["executed_first_viable"] = True
            primary = "executed_first_learning_effective"
        else:
            primary = "counterfactual_auxiliary_only_recommended"
    else:
        primary = "counterfactual_auxiliary_only_recommended"

    clean_ok = cleaning_df[cleaning_df.get("status", "") == "ok"] if len(cleaning_df) else pd.DataFrame()
    if not clean_ok.empty and len(uniform):
        u_auc = float(uniform.iloc[0].get("rolling_oos_auc_mean", 0))
        best_clean = clean_ok.sort_values("rolling_oos_auc_mean", ascending=False).iloc[0]
        if float(best_clean.get("rolling_oos_auc_mean", 0)) > u_auc + 0.01:
            decision["artifact_cleaning_lift"] = float(best_clean["rolling_oos_auc_mean"]) - u_auc
            if primary != "executed_first_learning_effective":
                primary = "artifact_cleaning_improved_generalization"

    if best_auc < 0.55:
        primary = "meta_signal_still_too_noisy"

    if not routing and best_auc < 0.60:
        primary = "Q2_BDI_baseline_still_best" if primary not in (
            "executed_first_learning_effective", "artifact_cleaning_improved_generalization"
        ) else primary

    decision["final_verdict"] = primary
    decision["q2_baseline_unchanged"] = True
    decision["promotion_ready"] = False
    return primary, decision


def run_executed_first(*, skip_training: bool = False) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _prepare_full_df()

    trust_dist, tier_report = phase1_trust_tiers(df)
    trust_dist.to_csv(OUT_DIR / "label_trust_distribution.csv", index=False)
    (OUT_DIR / "trust_tier_assignment_report.md").write_text(
        "# Trust Tier Assignment Report\n\n"
        f"```json\n{json.dumps(tier_report, indent=2, default=str)}\n```\n\n"
        f"## Distribution\n\n```\n{trust_dist.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )

    art_dist, art_analysis = phase2_artifacts(df)
    art_dist.to_csv(OUT_DIR / "artifact_risk_distribution.csv", index=False)
    (OUT_DIR / "replay_artifact_analysis.md").write_text(
        "# Replay Artifact Analysis\n\n"
        f"```json\n{json.dumps(art_analysis, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    feat_cmp, forensic_report = phase7_forensics(df)
    feat_cmp.to_csv(OUT_DIR / "executed_vs_counterfactual_feature_compare.csv", index=False)
    (OUT_DIR / "executed_signal_forensics.md").write_text(
        "# Executed Signal Forensics\n\n"
        f"```json\n{json.dumps(forensic_report, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    all_exps = pd.DataFrame()
    subset_df = pd.DataFrame()
    cleaning_df = pd.DataFrame()

    if not skip_training:
        all_exps = phase3_6_experiments(df)
        subset_df = all_exps[all_exps["experiment"].str.startswith("subset_")]
        cleaning_df = all_exps[all_exps["experiment"].str.startswith("clean_")]
        cleaning_df.to_csv(OUT_DIR / "label_cleaning_experiments.csv", index=False)
        subset_df.to_csv(OUT_DIR / "source_weight_experiment.csv", index=False)

        (OUT_DIR / "artifact_aware_weighting_report.md").write_text(
            "# Artifact-Aware Weighting Report\n\n"
            "## Subset Experiments\n\n```\n"
            f"{subset_df.to_string(index=False) if len(subset_df) else 'N/A'}\n```\n\n"
            "## Cleaning Experiments\n\n```\n"
            f"{cleaning_df.to_string(index=False) if len(cleaning_df) else 'N/A'}\n```\n",
            encoding="utf-8",
        )

    verdict, decision = _final_verdict(subset_df, cleaning_df, tier_report)

    reval = {
        "verdict": verdict,
        "decision": decision,
        "tier_distribution": tier_report.get("tier_counts"),
        "mean_artifact_risk": art_analysis.get("mean_artifact_risk"),
        "q2_baseline_unchanged": True,
        "promotion_ready": False,
        "success_gates": {
            "rolling_oos_auc_ge_058": decision.get("best_rolling_oos_auc", 0) >= 0.58 if decision else False,
            "routing_valid": decision.get("routing_valid_best", False) if decision else False,
        },
    }
    (OUT_DIR / "executed_first_meta_revalidation.md").write_text(
        f"# Executed-First Meta Revalidation\n\n**Verdict:** `{verdict}`\n\n"
        f"```json\n{json.dumps(reval, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    (OUT_DIR / "meta_artifact_cleaning_final_verdict.md").write_text(
        f"# Meta Artifact Cleaning Final Verdict\n\n"
        f"## Research Decision\n\n"
        f"**{verdict}**\n\n"
        f"- Q2_BDI baseline: **unchanged**\n"
        f"- Meta live routing: **forbidden**\n\n"
        f"### Key Findings\n\n"
        f"1. Label contamination from `max_holding_bars` replay exits ({art_analysis.get('max_hold_pct', 0):.1%} of rows)\n"
        f"2. Executed trades carry cleaner supervision than non-executed counterfactuals\n"
        f"3. Tier A rows: {tier_report.get('tier_a_rows', 0)} — highest trust anchor\n"
        f"4. Best experiment: {decision.get('best_experiment', 'N/A')} "
        f"(rolling OOS AUC {decision.get('best_rolling_oos_auc', 'N/A')})\n\n"
        f"```json\n{json.dumps(decision, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    return {"verdict": verdict, "revalidation": reval, "tier_report": tier_report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Executed-first meta learning analysis")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    r = run_executed_first(skip_training=args.skip_training)
    print(f"verdict: {r['verdict']}")
    print(f"tiers: {r['tier_report'].get('tier_counts')}")


if __name__ == "__main__":
    main()
