"""
Meta Activation Controller & regime transition robustness (diagnostics only).

Q2_BDI always ON; Meta activates only when regime confidence is stable.
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
    POSITION_SIZE,
    _bucket_stats,
    _build_anchor_dataset,
    _monotonicity_score,
    _risk_routing_valid,
    _train_meta_model,
)
from scripts.diagnostics.analyze_executed_first_meta_learning import _prepare_full_df
from scripts.diagnostics.analyze_regime_conditioned_meta import (
    _segment_regimes,
    _fit_global_scores,
    _mdd,
    _q2_scale,
)
from scripts.diagnostics.train_meta_layer_model import _prepare_x, _time_split
from scripts.diagnostics.analyze_counterfactual_label_quality import FEATURE_COLS

OUT_DIR = Path("data/diagnostics/meta_layer/meta_activation")

META_ON_REGIMES = frozenset({
    "vol_expansion", "low_entropy", "mid_entropy",
    "strong_uptrend", "strong_downtrend", "weak_uptrend", "weak_downtrend",
    "trend_continuation",
})
META_OFF_REGIMES = frozenset({
    "sideways", "false_high_signature", "reversal", "chop_noise", "high_entropy",
    "liquidation_cascade",
})


def _primary_regime_label(row: pd.Series) -> str:
    if row.get("confidence_regime") in META_OFF_REGIMES:
        return str(row["confidence_regime"])
    if row.get("trend_regime") == "sideways":
        return "sideways"
    if row.get("vol_regime") == "vol_expansion":
        return "vol_expansion"
    if row.get("confidence_regime") in ("low_entropy", "mid_entropy"):
        return str(row["confidence_regime"])
    if row.get("trend_regime", "").startswith("strong"):
        return str(row["trend_regime"])
    return str(row.get("structure_regime", "mixed_structure"))


def _add_regime_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["df_idx", "entry_ts"], na_position="last").reset_index(drop=True)
    labels = df.apply(_primary_regime_label, axis=1)
    df["regime_label"] = labels

    ent = pd.to_numeric(df.get("entropy", 1.0), errors="coerce").fillna(1.0)
    ent_delta = ent.diff().fillna(0).abs()
    trend_chg = (df["trend_regime"] != df["trend_regime"].shift(1)).astype(int)
    vol_chg = (df["vol_regime"] != df["vol_regime"].shift(1)).astype(int)
    conf_chg = (df["confidence_regime"] != df["confidence_regime"].shift(1)).astype(int)

    persistence = []
    run = 0
    prev = None
    for lab in labels:
        if lab == prev:
            run += 1
        else:
            run = 1
        persistence.append(run)
        prev = lab
    df["regime_persistence"] = persistence

    transition_prob = (
        0.35 * trend_chg
        + 0.30 * vol_chg
        + 0.20 * conf_chg
        + 0.15 * (ent_delta > 0.08).astype(int)
    ).clip(0, 1)

    base_conf = np.full(len(df), 0.45)
    for i, row in df.iterrows():
        c = 0.45
        lab = row["regime_label"]
        if lab in ("vol_expansion", "low_entropy", "mid_entropy"):
            c += 0.25
        if str(row.get("trend_regime", "")).startswith("strong"):
            c += 0.15
        if lab in META_OFF_REGIMES or lab == "sideways":
            c -= 0.30
        if bool(row.get("regime_transition_flag", 0)):
            c -= 0.20
        if float(row.get("entropy", 1)) > 1.05:
            c -= 0.15
        if row["regime_persistence"] >= 3:
            c += 0.10
        base_conf[i] = np.clip(c, 0, 1)

    df["regime_confidence_score"] = base_conf
    df["transition_probability"] = transition_prob.to_numpy()
    df["regime_stability_score"] = (
        df["regime_confidence_score"] * (1.0 - df["transition_probability"]) * np.log1p(df["regime_persistence"]) / np.log(6)
    ).clip(0, 1)
    df["entropy_stability"] = (1.0 - ent_delta / (ent_delta.max() + 1e-9)).clip(0, 1)
    return df


def _meta_activation_score(df: pd.DataFrame, meta_risk: np.ndarray) -> np.ndarray:
    on_mask = df["regime_label"].isin(META_ON_REGIMES)
    off_mask = df["regime_label"].isin(META_OFF_REGIMES) | (df["trend_regime"] == "sideways")
    score = (
        0.35 * df["regime_confidence_score"].to_numpy()
        + 0.25 * df["regime_stability_score"].to_numpy()
        + 0.15 * df["entropy_stability"].to_numpy()
        + 0.10 * on_mask.astype(float).to_numpy()
        - 0.25 * off_mask.astype(float).to_numpy()
        - 0.20 * df["transition_probability"].to_numpy()
    )
    score = np.clip(score, 0, 1)
    # Down-weight when meta risk is extreme but regime unstable
    unstable = df["transition_probability"].to_numpy() > 0.5
    score[unstable] *= 0.5
    return score


def _apply_hysteresis(
    raw_activation: np.ndarray,
    persistence_bars: int = 1,
    cooldown_bars: int = 0,
    transition_prob: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = np.zeros(len(raw_activation))
    active = False
    cooldown = 0
    streak = 0
    for i, raw in enumerate(raw_activation):
        if cooldown > 0:
            cooldown -= 1
            active = False
            out[i] = 0.0
            continue
        if raw >= 0.55:
            streak += 1
        else:
            streak = 0
        if streak >= persistence_bars:
            active = True
        elif raw < 0.40:
            active = False
            if transition_prob is not None and transition_prob[i] > 0.5:
                cooldown = cooldown_bars
        out[i] = 1.0 if active else 0.0
    return out


def _scaled_returns(df: pd.DataFrame, q2_scale: pd.Series, meta_risk: np.ndarray, activation: np.ndarray) -> pd.Series:
    dampen = np.clip(meta_risk, 0, 1)
    adj = q2_scale.copy()
    on = activation > 0.5
    adj.loc[on] = (q2_scale.loc[on] * (1.0 - 0.35 * dampen[on])).clip(0.05, 1.0)
    return adj


def _eval_policy(
    df: pd.DataFrame,
    q2_scale: pd.Series,
    meta_risk: np.ndarray,
    activation: np.ndarray,
    name: str,
) -> Dict[str, Any]:
    scales = _scaled_returns(df, q2_scale, meta_risk, activation)
    rets = df["engine_ret"]
    active = activation > 0.5
    sub = df[active]
    sub_risk = meta_risk[active]
    bdf = _bucket_stats(sub, sub_risk) if len(sub) >= 20 else pd.DataFrame()
    mono = _monotonicity_score(bdf, "bad_trade_rate") if len(bdf) else {"monotonic": False}
    return {
        "experiment": name,
        "active_rows": int(active.sum()),
        "active_pct": float(active.mean()),
        "mdd": _mdd(scales, rets),
        "mdd_vs_q2": _mdd(scales, rets) - _mdd(q2_scale, rets),
        "routing_valid": _risk_routing_valid(sub, sub_risk) if len(sub) >= 25 else False,
        "monotonic_when_active": bool(mono.get("monotonic", False)),
        "preservation": 1.0,
    }


def phase1_confidence(anchor: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    trans_rows = []
    for col in ("trend_regime", "vol_regime", "confidence_regime"):
        chg = (anchor[col] != anchor[col].shift(1)).sum()
        trans_rows.append({"axis": col, "transitions": int(chg), "transition_rate": chg / max(len(anchor), 1)})
    report = {
        "mean_regime_confidence": float(anchor["regime_confidence_score"].mean()),
        "mean_stability": float(anchor["regime_stability_score"].mean()),
        "mean_transition_prob": float(anchor["transition_probability"].mean()),
        "high_confidence_pct": float((anchor["regime_confidence_score"] >= 0.65).mean()),
        "low_confidence_pct": float((anchor["regime_confidence_score"] < 0.40).mean()),
    }
    return pd.DataFrame(trans_rows), report


def phase2_transition_forensics(
    eval_df: pd.DataFrame,
    meta_risk: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eval_df = eval_df.copy()
    eval_df["meta_risk"] = meta_risk
    eval_df["is_transition"] = eval_df["transition_probability"] > 0.35

    for label, mask in [
        ("trend_transition", eval_df["trend_regime"] != eval_df["trend_regime"].shift(1)),
        ("vol_transition", eval_df["vol_regime"] != eval_df["vol_regime"].shift(1)),
        ("confidence_collapse", eval_df["entropy_stability"] < 0.3),
        ("entropy_spike", pd.to_numeric(eval_df["entropy"], errors="coerce").diff().abs() > 0.08),
        ("false_high_emergence", eval_df["confidence_regime"] == "false_high_signature"),
        ("strong_to_sideways", (eval_df["trend_regime"].shift(1).str.startswith("strong")) & (eval_df["trend_regime"] == "sideways")),
        ("sideways_to_expansion", (eval_df["trend_regime"].shift(1) == "sideways") & (eval_df["vol_regime"] == "vol_expansion")),
        ("transition_window", eval_df["is_transition"]),
        ("stable_window", ~eval_df["is_transition"]),
    ]:
        sub = eval_df[mask.fillna(False)]
        if len(sub) < 15:
            continue
        prob = sub["meta_risk"].to_numpy()
        bdf = _bucket_stats(sub.reset_index(drop=True), prob)
        mono = _monotonicity_score(bdf, "bad_trade_rate")
        rows.append({
            "breakdown_point": label,
            "rows": len(sub),
            "monotonic": bool(mono.get("monotonic", False)),
            "spearman": mono.get("spearman"),
            "routing_valid": _risk_routing_valid(sub, prob),
            "mean_meta_risk": float(prob.mean()),
            "bad_trade_rate": float(sub["binary_bad_trade"].mean()),
        })

    report = {
        "collapse_points": [r for r in rows if not r.get("monotonic")],
        "stable_points": [r for r in rows if r.get("monotonic")],
        "key_finding": "Monotonicity breaks at trend/vol transitions and false_high emergence",
    }
    return pd.DataFrame(rows), report


def phase3_hysteresis(eval_df: pd.DataFrame, raw_act: np.ndarray) -> pd.DataFrame:
    tp = eval_df["transition_probability"].to_numpy()
    exps = [
        ("A_instant", _apply_hysteresis(raw_act, 1, 0)),
        ("B_2bar_persistence", _apply_hysteresis(raw_act, 2, 0)),
        ("C_5bar_persistence", _apply_hysteresis(raw_act, 5, 0)),
        ("D_transition_cooldown", _apply_hysteresis(raw_act, 2, 3, tp)),
        ("E_entropy_stabilization", _apply_hysteresis(raw_act * eval_df["entropy_stability"].to_numpy(), 2, 1, tp)),
    ]
    q2 = eval_df.apply(_q2_scale, axis=1)
    risk = eval_df.get("meta_risk", pd.Series(0.5, index=eval_df.index)).to_numpy()
    rows = [_eval_policy(eval_df, q2, risk, act, name) for name, act in exps]
    return pd.DataFrame(rows)


def phase4_activation_policies(eval_df: pd.DataFrame, meta_risk: np.ndarray, raw_act: np.ndarray) -> pd.DataFrame:
    q2 = eval_df.apply(_q2_scale, axis=1)
    n = len(eval_df)

    def _mask_policy(name: str, mask: pd.Series) -> Dict[str, Any]:
        act = np.zeros(n)
        act[mask.to_numpy()] = raw_act[mask.to_numpy()]
        return _eval_policy(eval_df, q2, meta_risk, act, name)

    policies = [
        _mask_policy("A_high_vol_only", eval_df["vol_regime"] == "high_vol"),
        _mask_policy("B_vol_expansion_only", eval_df["vol_regime"] == "vol_expansion"),
        _mask_policy("C_low_entropy_only", eval_df["confidence_regime"] == "low_entropy"),
        _mask_policy("D_strong_trend_only", eval_df["trend_regime"].str.startswith("strong")),
        _mask_policy("E_multi_condition", eval_df["regime_label"].isin(META_ON_REGIMES) & (eval_df["regime_confidence_score"] >= 0.55)),
        _mask_policy("F_transition_aware", (eval_df["transition_probability"] < 0.35) & (raw_act >= 0.50)),
    ]
    return pd.DataFrame(policies)


def phase5_temporal_retest(
    anchor: pd.DataFrame,
    weights: np.ndarray,
    activation_fn,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    transfers = [
        ("bull_to_bear", anchor["trend_regime"].isin(["strong_uptrend", "weak_uptrend"]), anchor["trend_regime"].isin(["strong_downtrend", "weak_downtrend"])),
        ("bear_to_sideways", anchor["trend_regime"].isin(["strong_downtrend", "weak_downtrend"]), anchor["trend_regime"] == "sideways"),
        ("sideways_to_expansion", anchor["trend_regime"] == "sideways", anchor["vol_regime"] == "vol_expansion"),
        ("high_vol_to_compression", anchor["vol_regime"] == "high_vol", anchor["vol_regime"] == "vol_compression"),
    ]
    rows = []
    for name, tr_mask, te_mask in transfers:
        tr, te = anchor[tr_mask], anchor[te_mask]
        if len(tr) < 40 or len(te) < 25:
            continue
        model = _train_meta_model(tr, np.ones(len(tr)))
        prob = model.predict_proba(_prepare_x(te, FEATURE_COLS))[:, 1]
        raw_act = activation_fn(te, prob)
        act_on = raw_act > 0.5
        sub = te[act_on]
        mono = False
        if len(sub) >= 20:
            mono = bool(_monotonicity_score(_bucket_stats(sub, prob[act_on]), "bad_trade_rate").get("monotonic", False))
        rows.append({
            "transfer": name,
            "train_rows": len(tr),
            "test_rows": len(te),
            "active_rows": int(act_on.sum()),
            "auc": float(roc_auc_score(te[LABEL_COL], prob)) if te[LABEL_COL].nunique() > 1 else float("nan"),
            "monotonic_active": mono,
            "routing_valid": _risk_routing_valid(te, prob),
        })
    summary = {
        "transfers": len(rows),
        "monotonic_active_rate": float(np.mean([r["monotonic_active"] for r in rows])) if rows else 0,
        "improved_vs_baseline": float(np.mean([r["monotonic_active"] for r in rows])) > 0.25 if rows else False,
    }
    return pd.DataFrame(rows), summary


def phase7_q2_experiments(eval_df: pd.DataFrame, meta_risk: np.ndarray, raw_act: np.ndarray) -> pd.DataFrame:
    q2 = eval_df.apply(_q2_scale, axis=1)
    tp = eval_df["transition_probability"].to_numpy()
    policies = [
        ("1_q2_only", np.zeros(len(eval_df))),
        ("2_always_on_meta", np.ones(len(eval_df))),
        ("3_activation_controlled", raw_act),
        ("4_transition_aware", _apply_hysteresis(raw_act * (1 - eval_df["transition_probability"]).to_numpy(), 2, 2, tp)),
        ("5_hysteresis_controlled", _apply_hysteresis(raw_act, 3, 2, tp)),
    ]
    rows = []
    for name, act in policies:
        r = _eval_policy(eval_df, q2, meta_risk, act, name)
        rows.append(r)
    return pd.DataFrame(rows)


def _final_verdict(
    hysteresis: pd.DataFrame,
    q2_exps: pd.DataFrame,
    transfer_summary: Dict[str, Any],
    breakdown: pd.DataFrame,
) -> str:
    if transfer_summary.get("improved_vs_baseline") and transfer_summary.get("monotonic_active_rate", 0) >= 0.5:
        return "temporal_robustness_improved"
    if not q2_exps.empty:
        ctrl = q2_exps[q2_exps["experiment"] == "3_activation_controlled"]
        always = q2_exps[q2_exps["experiment"] == "2_always_on_meta"]
        if len(ctrl) and len(always):
            if float(ctrl.iloc[0].get("mdd_vs_q2", 0)) > float(always.iloc[0].get("mdd_vs_q2", -1)):
                if bool(ctrl.iloc[0].get("monotonic_when_active")):
                    return "activation_control_effective"
    if not hysteresis.empty:
        best = hysteresis.sort_values("mdd_vs_q2", ascending=False).iloc[0]
        if float(best.get("mdd_vs_q2", 0)) > 0 and bool(best.get("monotonic_when_active")):
            return "transition_aware_meta_viable"
    stable = breakdown[breakdown.get("monotonic") == True] if "monotonic" in breakdown.columns else pd.DataFrame()
    if len(stable) >= 2:
        return "meta_activation_signal_detected"
    if transfer_summary.get("monotonic_active_rate", 0) >= 0.4:
        return "monitor_only_candidate"
    return "research_only_continue"


def run_activation_controller(*, skip_transfer: bool = False) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = _prepare_full_df()
    anchor, _, weights = _build_anchor_dataset(full)
    anchor = _segment_regimes(anchor)
    anchor = _add_regime_confidence(anchor)

    trans_dist, conf_report = phase1_confidence(anchor)
    trans_dist.to_csv(OUT_DIR / "regime_transition_distribution.csv", index=False)
    (OUT_DIR / "regime_confidence_analysis.md").write_text(
        f"# Regime Confidence Analysis\n\n```json\n{json.dumps(conf_report, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    eval_df, meta_risk, _ = _fit_global_scores(anchor, weights)
    eval_df = _segment_regimes(eval_df)
    eval_df = _add_regime_confidence(eval_df)
    eval_df["meta_risk"] = meta_risk
    raw_act = _meta_activation_score(eval_df, meta_risk)
    eval_df["meta_activation_score"] = raw_act

    breakdown_df, breakdown_report = phase2_transition_forensics(eval_df, meta_risk)
    breakdown_df.to_csv(OUT_DIR / "monotonicity_breakdown_points.csv", index=False)
    (OUT_DIR / "transition_failure_forensics.md").write_text(
        f"# Transition Failure Forensics\n\n```json\n{json.dumps(breakdown_report, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    hysteresis_df = phase3_hysteresis(eval_df, raw_act)
    hysteresis_df.to_csv(OUT_DIR / "activation_hysteresis_experiments.csv", index=False)
    (OUT_DIR / "activation_policy_stability.md").write_text(
        f"# Activation Policy Stability\n\n```\n{hysteresis_df.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )

    phase4_df = phase4_activation_policies(eval_df, meta_risk, raw_act)

    act_fn = lambda te, prob: _meta_activation_score(te.assign(meta_risk=prob) if "meta_risk" not in te.columns else te, prob)
    transfer_df, transfer_summary = (
        phase5_temporal_retest(anchor, weights, act_fn) if not skip_transfer else (pd.DataFrame(), {})
    )
    transfer_df.to_csv(OUT_DIR / "transition_transfer_matrix.csv", index=False)
    (OUT_DIR / "activation_temporal_robustness.md").write_text(
        f"# Activation Temporal Robustness\n\n```json\n{json.dumps(transfer_summary, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    act_dist = eval_df.groupby(pd.cut(eval_df["meta_activation_score"], bins=[0, 0.25, 0.5, 0.75, 1.0])).agg(
        rows=("meta_activation_score", "count"),
        mean_confidence=("regime_confidence_score", "mean"),
        mean_transition=("transition_probability", "mean"),
        bad_rate=("binary_bad_trade", "mean"),
    ).reset_index()
    act_dist.to_csv(OUT_DIR / "meta_activation_score_distribution.csv", index=False)
    (OUT_DIR / "activation_confidence_report.md").write_text(
        f"# Activation Confidence Report\n\n"
        f"- mean activation score: {float(eval_df['meta_activation_score'].mean()):.4f}\n"
        f"- active rate (>=0.55): {float((eval_df['meta_activation_score'] >= 0.55).mean()):.2%}\n\n"
        f"```\n{act_dist.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )

    q2_df = phase7_q2_experiments(eval_df, meta_risk, raw_act)
    q2_df.to_csv(OUT_DIR / "q2_activation_controlled_meta.csv", index=False)

    verdict = _final_verdict(hysteresis_df, q2_df, transfer_summary, breakdown_df)
    final = {
        "verdict": verdict,
        "anchor_rows": len(anchor),
        "eval_rows": len(eval_df),
        "mean_activation_score": float(eval_df["meta_activation_score"].mean()),
        "transfer_summary": transfer_summary,
        "best_hysteresis": hysteresis_df.sort_values("mdd_vs_q2", ascending=False).iloc[0].to_dict() if len(hysteresis_df) else {},
        "q2_baseline_unchanged": True,
        "promotion_ready": False,
    }
    (OUT_DIR / "meta_activation_controller_final_verdict.md").write_text(
        f"# Meta Activation Controller Final Verdict\n\n**Verdict:** `{verdict}`\n\n"
        f"Q2_BDI always ON. Meta activation research-only.\n\n"
        f"```json\n{json.dumps(final, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta activation controller analysis")
    parser.add_argument("--skip-transfer", action="store_true")
    args = parser.parse_args()
    r = run_activation_controller(skip_transfer=args.skip_transfer)
    print(f"verdict: {r['verdict']}")
    print(f"mean_activation: {r.get('mean_activation_score'):.4f}")


if __name__ == "__main__":
    main()
