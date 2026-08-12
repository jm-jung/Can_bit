"""
Executed-anchor Meta routing calibration & monotonic risk bucket validation.

Diagnostics only — validates risk ranking / routing refinement, not direction prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.analyze_counterfactual_label_quality import (
    FEATURE_COLS,
    NOISE_ZONE_THRESHOLD,
    _routing_valid_proxy,
)
from scripts.diagnostics.analyze_executed_first_meta_learning import _prepare_full_df
from scripts.diagnostics.train_meta_layer_model import (
    _calibration_error,
    _classification_metrics,
    _prepare_x,
    _time_split,
)
from scripts.diagnostics.validate_quality_score_replay import map_m3

OUT_DIR = Path("data/diagnostics/meta_layer/executed_anchor")
LABEL_COL = "drawdown_risk_label"
POSITION_SIZE = 0.05
N_BUCKETS = 4


def _build_anchor_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Primary executed Tier A/B; auxiliary counterfactual at low weight."""
    base_exclude = (
        (df["label_trust_tier"] == "D")
        | (df["horizon_disagreement"] >= 0.5)
        | (df["engine_ret"].abs() < NOISE_ZONE_THRESHOLD)
    )
    cf_artifact = (df["exit_reason"] == "max_holding_bars") & (~df["is_executed"])

    primary = df["is_executed"] & df["label_trust_tier"].isin(["A", "B"]) & ~base_exclude
    auxiliary = (
        (~df["is_executed"])
        & df["label_trust_tier"].isin(["C"])
        & (df["artifact_risk_score"] < 0.40)
        & (df["horizon_disagreement"] < 0.35)
        & ~base_exclude
        & ~cf_artifact
    )
    mask = primary | auxiliary
    anchor = df[mask].copy().reset_index(drop=True)

    weights = np.ones(len(anchor))
    weights[~anchor["is_executed"].to_numpy()] = 0.25
    tier_mult = anchor["label_trust_tier"].map({"A": 1.3, "B": 1.0, "C": 0.25}).fillna(0.25)
    weights = weights * tier_mult.to_numpy()
    return anchor, mask, weights


def _anchor_summary(anchor: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": len(anchor),
        "executed_rows": int(anchor["is_executed"].sum()),
        "counterfactual_rows": int((~anchor["is_executed"]).sum()),
        "tier_distribution": anchor["label_trust_tier"].value_counts().to_dict(),
        "label_distribution": {
            "drawdown_risk_label": anchor[LABEL_COL].value_counts().to_dict(),
            "binary_bad_trade": anchor["binary_bad_trade"].value_counts().to_dict() if "binary_bad_trade" in anchor.columns else {},
        },
        "long_ratio": float((anchor["direction"] == "LONG").mean()),
        "vol_distribution": anchor["vol_bucket"].value_counts().to_dict() if "vol_bucket" in anchor.columns else {},
        "trend_distribution": anchor["trend_state"].value_counts().to_dict() if "trend_state" in anchor.columns else {},
        "mean_artifact_risk": float(anchor["artifact_risk_score"].mean()),
        "max_hold_excluded": True,
    }


def _distribution_csv(anchor: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "rows", "value": len(anchor)},
        {"metric": "executed_ratio", "value": float(anchor["is_executed"].mean())},
        {"metric": "long_ratio", "value": float((anchor["direction"] == "LONG").mean())},
        {"metric": "bad_trade_rate", "value": float(anchor["binary_bad_trade"].mean())},
        {"metric": "drawdown_risk_rate", "value": float(anchor[LABEL_COL].mean())},
        {"metric": "mean_artifact_risk", "value": float(anchor["artifact_risk_score"].mean())},
    ]
    for tier, n in anchor["label_trust_tier"].value_counts().items():
        rows.append({"metric": f"tier_{tier}", "value": int(n)})
    for vol, n in anchor.get("vol_bucket", pd.Series()).value_counts().items():
        rows.append({"metric": f"vol_{vol}", "value": int(n)})
    return pd.DataFrame(rows)


def _train_meta_model(
    train_df: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
) -> HistGradientBoostingClassifier:
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=42)
    m.fit(
        _prepare_x(train_df, FEATURE_COLS),
        train_df[LABEL_COL].astype(int),
        sample_weight=weights,
    )
    return m


def _calibrate_scores(
    raw_val: np.ndarray,
    y_val: np.ndarray,
    raw_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {"raw": raw_test}

    platt = LogisticRegression(max_iter=500)
    platt.fit(raw_val.reshape(-1, 1), y_val)
    out["platt"] = platt.predict_proba(raw_test.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    out["isotonic"] = iso.predict(raw_test)

    best_t, best_nll = 1.0, 1e9
    eps = 1e-6
    raw_v = np.clip(raw_val, eps, 1 - eps)
    for t in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        logits = np.log(raw_v / (1 - raw_v)) / t
        prob = 1 / (1 + np.exp(-logits))
        nll = -np.mean(y_val * np.log(np.clip(prob, eps, 1)) + (1 - y_val) * np.log(np.clip(1 - prob, eps, 1)))
        if nll < best_nll:
            best_t, best_nll = t, nll
    raw_te = np.clip(raw_test, eps, 1 - eps)
    logits_te = np.log(raw_te / (1 - raw_te)) / best_t
    out["temperature"] = 1 / (1 + np.exp(-logits_te))

    out["quantile"] = pd.Series(raw_test).rank(pct=True).to_numpy()
    return out


def _calibration_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, prob)),
        "calibration_error": _calibration_error(y, prob),
    }


def _risk_buckets(scores: np.ndarray, n: int = N_BUCKETS) -> np.ndarray:
    ranks = pd.Series(scores).rank(method="first")
    return pd.qcut(ranks, q=n, labels=[f"Q{i+1}" for i in range(n)]).astype(str).to_numpy()


def _bucket_stats(df: pd.DataFrame, scores: np.ndarray, score_col: str = "meta_score") -> pd.DataFrame:
    tmp = df.copy()
    tmp[score_col] = scores
    tmp["risk_bucket"] = _risk_buckets(scores, N_BUCKETS)
    rows: List[Dict[str, Any]] = []
    for bucket in [f"Q{i+1}" for i in range(N_BUCKETS)]:
        g = tmp[tmp["risk_bucket"] == bucket]
        if g.empty:
            continue
        fh = 0
        if {"direction", "trend_state", "vol_bucket", "entropy"}.issubset(g.columns):
            fh = int((
                (g["direction"] == "LONG")
                & (g["trend_state"] == "up")
                & (g["vol_bucket"] == "high")
                & (g["entropy"] <= 0.90)
                & (g["binary_bad_trade"] == 1)
            ).sum())
        rows.append({
            "bucket": bucket,
            "trades": len(g),
            "score_mean": float(g[score_col].mean()),
            "score_min": float(g[score_col].min()),
            "score_max": float(g[score_col].max()),
            "bad_trade_rate": float(g["binary_bad_trade"].mean()) if "binary_bad_trade" in g.columns else None,
            "drawdown_risk_rate": float(g[LABEL_COL].mean()),
            "rfe_rate": float(g["rfe_flag"].astype(bool).mean()) if "rfe_flag" in g.columns else None,
            "false_high_count": fh,
            "false_high_rate": fh / max(len(g), 1),
            "mae_mean": float(g["mae"].mean()) if "mae" in g.columns else None,
            "mfe_mean": float(g["mfe"].mean()) if "mfe" in g.columns else None,
            "expectancy": float(g["engine_ret"].mean()),
            "routing_consistency": float(g["binary_good_trade"].mean()) if "binary_good_trade" in g.columns else None,
        })
    return pd.DataFrame(rows)


def _monotonicity_score(bucket_df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if bucket_df.empty or len(bucket_df) < 2:
        return {"monotonic": False, "spearman": None, "violations": 0}
    vals = bucket_df.sort_values("bucket")[col].to_numpy()
    violations = int(sum(vals[i] > vals[i + 1] for i in range(len(vals) - 1)))
    spearman = float(pd.Series(range(len(vals))).corr(pd.Series(vals), method="spearman"))
    return {
        "metric": col,
        "monotonic": violations == 0,
        "violations": violations,
        "spearman": spearman,
        "values_by_bucket": dict(zip(bucket_df.sort_values("bucket")["bucket"], vals)),
    }


def _risk_routing_valid(df: pd.DataFrame, scores: np.ndarray) -> bool:
    """Higher meta risk score should map to worse outcomes."""
    tmp = df.copy()
    tmp["meta_score"] = scores
    good = tmp[tmp["binary_good_trade"] == 1]["meta_score"]
    bad = tmp[tmp["binary_bad_trade"] == 1]["meta_score"]
    rfe = tmp[tmp["rfe_flag"].astype(bool)]["meta_score"] if "rfe_flag" in tmp.columns else pd.Series(dtype=float)
    if len(good) < 5 or len(bad) < 5:
        return False
    if len(rfe) >= 3:
        return float(rfe.mean()) > float(bad.mean()) > float(good.mean())
    return float(bad.mean()) > float(good.mean())


def _routing_overlap_analysis(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    tmp = df.copy()
    tmp["meta_score"] = scores
    tmp["risk_bucket"] = _risk_buckets(scores)
    tmp["q2_bucket"] = pd.qcut(tmp["q2_bdi_score"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    ct = pd.crosstab(tmp["risk_bucket"], tmp["q2_bucket"], normalize="index")
    rows = []
    for rb in ct.index:
        for qb in ct.columns:
            rows.append({"meta_bucket": str(rb), "q2_bucket": str(qb), "overlap_pct": float(ct.loc[rb, qb])})
    return pd.DataFrame(rows)


def _routing_failure_forensics(
    df: pd.DataFrame,
    scores: np.ndarray,
    bucket_df: pd.DataFrame,
) -> Dict[str, Any]:
    tmp = df.copy()
    tmp["meta_score"] = scores
    winners = tmp[tmp["binary_good_trade"] == 1]["meta_score"]
    losers = tmp[tmp["binary_bad_trade"] == 1]["meta_score"]
    rfe = tmp[tmp["rfe_flag"].astype(bool)]["meta_score"] if "rfe_flag" in tmp.columns else pd.Series(dtype=float)

    routing_valid = _risk_routing_valid(tmp, scores)

    score_std = float(np.std(scores))
    score_range = float(np.max(scores) - np.min(scores))

    mono_bad = _monotonicity_score(bucket_df, "bad_trade_rate")
    mono_ddr = _monotonicity_score(bucket_df, "drawdown_risk_rate")
    mono_rfe = _monotonicity_score(bucket_df, "rfe_rate") if "rfe_rate" in bucket_df.columns else {"monotonic": False}

    high_vol = tmp[tmp["vol_bucket"] == "high"]
    low_vol = tmp[tmp["vol_bucket"] != "high"]
    long_only = tmp[tmp["direction"] == "LONG"]

    fh_cluster = tmp[tmp.get("false_high_signature_flag", 0).astype(bool)]

    return {
        "routing_valid_proxy": routing_valid,
        "winner_mean_score": float(winners.mean()) if len(winners) else None,
        "loser_mean_score": float(losers.mean()) if len(losers) else None,
        "rfe_mean_score": float(rfe.mean()) if len(rfe) else None,
        "score_collapse": score_std < 0.05 or score_range < 0.15,
        "score_std": score_std,
        "score_range": score_range,
        "calibration_saturation": float((scores > 0.95).mean() + (scores < 0.05).mean()),
        "high_vol_score_std": float(high_vol["meta_score"].std()) if len(high_vol) else None,
        "long_bias_mean_score": float(long_only["meta_score"].mean()) if len(long_only) else None,
        "false_high_cluster_mean_score": float(fh_cluster["meta_score"].mean()) if len(fh_cluster) else None,
        "monotonic_bad_trade": mono_bad,
        "monotonic_drawdown_risk": mono_ddr,
        "monotonic_rfe": mono_rfe,
        "bucket_overlap_issue": not mono_bad.get("monotonic", False),
        "why_routing_fails": [
            "Score separation between winners/losers may be inverted or collapsed",
            "LONG bias concentrates scores in narrow band",
            "Q2 conflict regions overlap meta risk buckets",
            "High-vol chop increases bucket instability",
        ],
    }


def _q2_scale(row: pd.Series) -> float:
    return float(map_m3(float(row.get("q2_bdi_score", 0.5))) or 0.15)


def _q2_aligned_experiments(df: pd.DataFrame, meta_scores: np.ndarray) -> pd.DataFrame:
    tmp = df.copy()
    tmp["meta_score"] = meta_scores
    tmp["q2_scale"] = tmp.apply(_q2_scale, axis=1)

    def _mdd(scales: pd.Series) -> float:
        rets = tmp["engine_ret"] * scales
        eq = peak = 1.0
        mdd = 0.0
        for r in rets:
            eq *= 1.0 + float(r) * POSITION_SIZE
            peak = max(peak, eq)
            mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0)
        return float(mdd)

    def _routing(scales: pd.Series) -> bool:
        t = tmp.copy()
        t["scale"] = scales
        w = t[t["binary_good_trade"] == 1]["scale"]
        l = t[t["binary_bad_trade"] == 1]["scale"]
        if len(w) < 5 or len(l) < 5:
            return False
        return float(w.mean()) > float(l.mean())

    experiments: List[Dict[str, Any]] = []

    # A: Q2 only
    s_a = tmp["q2_scale"]
    experiments.append({
        "experiment": "A_q2_only",
        "mean_scale": float(s_a.mean()),
        "mdd": _mdd(s_a),
        "routing_valid": _routing(s_a),
        "preservation": 1.0,
    })

    # B: Q2 + Meta confidence rerank (reduce scale when meta risk high)
    s_b = (tmp["q2_scale"] * (1.0 - 0.25 * tmp["meta_score"])).clip(0.05, 1.0)
    experiments.append({
        "experiment": "B_q2_meta_confidence_rerank",
        "mean_scale": float(s_b.mean()),
        "mdd": _mdd(s_b),
        "routing_valid": _routing(s_b),
        "preservation": 1.0,
    })

    # C: Q2 + Meta bucket refinement
    bucket = _risk_buckets(meta_scores)
    mult = {"Q1": 1.05, "Q2": 1.0, "Q3": 0.85, "Q4": 0.65}
    s_c = (tmp["q2_scale"] * tmp.assign(rb=bucket)["rb"].map(mult).fillna(0.9)).clip(0.05, 1.0)
    experiments.append({
        "experiment": "C_q2_meta_bucket_refinement",
        "mean_scale": float(s_c.mean()),
        "mdd": _mdd(s_c),
        "routing_valid": _routing(s_c),
        "preservation": 1.0,
    })

    # D: Q2 + Meta drawdown override
    s_d = tmp["q2_scale"].where(tmp["meta_score"] < 0.65, tmp["q2_scale"] * 0.5).clip(0.05, 1.0)
    experiments.append({
        "experiment": "D_q2_meta_drawdown_override",
        "mean_scale": float(s_d.mean()),
        "mdd": _mdd(s_d),
        "routing_valid": _routing(s_d),
        "preservation": float((s_d > 0).mean()),
    })

    # E: Q2 + Meta false_high dampening
    fh = tmp.get("false_high_signature_flag", 0).astype(bool)
    s_e = tmp["q2_scale"].copy()
    s_e.loc[fh & (tmp["meta_score"] > 0.5)] *= 0.6
    s_e = s_e.clip(0.05, 1.0)
    experiments.append({
        "experiment": "E_q2_meta_false_high_dampening",
        "mean_scale": float(s_e.mean()),
        "mdd": _mdd(s_e),
        "routing_valid": _routing(s_e),
        "preservation": 1.0,
    })

    return pd.DataFrame(experiments)


def _rolling_oos_routing(
    anchor: pd.DataFrame,
    weights: np.ndarray,
) -> List[Dict[str, Any]]:
    folds: List[Dict[str, Any]] = []
    n = len(anchor)
    chunk, step = max(50, n // 5), max(15, n // 12)
    start = 0
    while start + chunk + 10 <= n and len(folds) < 20:
        sub = anchor.iloc[start : start + chunk + step].copy()
        split = int(len(sub) * 0.7)
        if split < 30 or len(sub) - split < 15:
            break
        train, test = sub.iloc[:split], sub.iloc[split:]
        w_tr = weights[start : start + split]
        m = _train_meta_model(train, w_tr)
        raw = m.predict_proba(_prepare_x(test, FEATURE_COLS))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(m.predict_proba(_prepare_x(train, FEATURE_COLS))[:, 1], train[LABEL_COL].astype(int))
        cal = iso.predict(raw)

        bdf = _bucket_stats(test, cal)
        mono = _monotonicity_score(bdf, "bad_trade_rate")
        routing = _risk_routing_valid(test, cal)

        folds.append({
            "fold_start": start,
            "test_rows": len(test),
            "auc": float(roc_auc_score(test[LABEL_COL], cal)) if test[LABEL_COL].nunique() > 1 else float("nan"),
            "monotonic_bad_trade": mono.get("monotonic", False),
            "routing_valid": routing,
            "calibration_error": _calibration_error(test[LABEL_COL].to_numpy(), cal),
        })
        start += step
    return folds


def _final_verdict(
    mono_report: Dict[str, Any],
    routing_forensics: Dict[str, Any],
    oos_folds: List[Dict[str, Any]],
    q2_exps: pd.DataFrame,
) -> str:
    mono_pass = mono_report.get("overall_monotonic", False)
    routing_proxy = routing_forensics.get("routing_valid_proxy", False)

    if oos_folds:
        mono_fold_rate = float(np.mean([f.get("monotonic_bad_trade", False) for f in oos_folds]))
        routing_fold_rate = float(np.mean([f.get("routing_valid", False) for f in oos_folds]))
    else:
        mono_fold_rate = routing_fold_rate = 0.0

    q2_viable = False
    if not q2_exps.empty:
        q2_only = q2_exps[q2_exps["experiment"] == "A_q2_only"]
        hybrids = q2_exps[q2_exps["experiment"].str.startswith(("B_", "C_", "D_", "E_"))]
        if len(q2_only) and len(hybrids):
            q2_mdd = float(q2_only.iloc[0]["mdd"])
            best_h = hybrids.sort_values("mdd", ascending=False).iloc[0]
            if float(best_h["mdd"]) >= q2_mdd and bool(best_h["routing_valid"]):
                q2_viable = True

    if mono_pass and routing_proxy and mono_fold_rate >= 0.6:
        return "monotonic_risk_structure_detected"
    if routing_proxy and mono_fold_rate >= 0.5:
        return "executed_anchor_calibration_effective"
    if q2_viable:
        return "q2_refinement_layer_viable"
    if mono_pass:
        return "research_only_continue"
    if routing_fold_rate >= 0.5 and mono_fold_rate >= 0.4:
        return "monitor_only_candidate"
    return "meta_routing_still_unstable"


def run_routing_validation(*, skip_oos: bool = False) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = _prepare_full_df()
    anchor, _, weights = _build_anchor_dataset(full)

    summary = _anchor_summary(anchor)
    dist_df = _distribution_csv(anchor)
    dist_df.to_csv(OUT_DIR / "executed_anchor_distribution.csv", index=False)
    (OUT_DIR / "executed_anchor_dataset_summary.md").write_text(
        f"# Executed-Anchor Dataset Summary\n\n```json\n{json.dumps(summary, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    train_df, val_df, test_df = _time_split(anchor)
    w_train = weights[: len(train_df)]
    model = _train_meta_model(train_df, w_train)

    raw_val = model.predict_proba(_prepare_x(val_df, FEATURE_COLS))[:, 1]
    raw_test = model.predict_proba(_prepare_x(test_df, FEATURE_COLS))[:, 1]
    y_val = val_df[LABEL_COL].astype(int).to_numpy()
    y_test = test_df[LABEL_COL].astype(int).to_numpy()

    calibrated = _calibrate_scores(raw_val, y_val, raw_test)

    # Bucket analysis on val+test (held-out 40%) for stable quartiles
    eval_df = pd.concat([val_df, test_df], ignore_index=True)
    raw_eval = np.concatenate([
        model.predict_proba(_prepare_x(val_df, FEATURE_COLS))[:, 1],
        raw_test,
    ])
    y_eval = np.concatenate([y_val, y_test])
    cal_eval = _calibrate_scores(raw_val, y_val, raw_eval)

    cal_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    for method, prob in calibrated.items():
        met = _calibration_metrics(y_test, prob)
        cal_rows.append({"method": method, **met})
        for i in range(5):
            lo, hi = i / 5, (i + 1) / 5
            mask = (prob >= lo) & (prob < hi)
            if mask.sum() == 0:
                continue
            curve_rows.append({
                "method": method,
                "bin_lo": lo,
                "bin_hi": hi,
                "pred_mean": float(prob[mask].mean()),
                "obs_rate": float(y_test[mask].mean()),
                "count": int(mask.sum()),
            })

    cal_df = pd.DataFrame(cal_rows)
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(OUT_DIR / "calibration_curve.csv", index=False)
    best_method = cal_df.sort_values("calibration_error").iloc[0]["method"] if len(cal_df) else "isotonic"
    best_scores = cal_eval.get(str(best_method), cal_eval.get("isotonic", raw_eval))

    (OUT_DIR / "meta_score_calibration_report.md").write_text(
        "# Meta Score Calibration Report\n\n"
        f"Best method by calibration error: **{best_method}**\n\n"
        f"```\n{cal_df.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )
    (OUT_DIR / "calibration_error_analysis.md").write_text(
        "# Calibration Error Analysis\n\n"
        f"```json\n{json.dumps(cal_df.to_dict(orient='records'), indent=2)}\n```\n",
        encoding="utf-8",
    )

    bucket_df = _bucket_stats(eval_df, best_scores)
    bucket_df.to_csv(OUT_DIR / "risk_bucket_distribution.csv", index=False)

    mono_metrics = ["bad_trade_rate", "drawdown_risk_rate", "rfe_rate", "false_high_rate", "expectancy"]
    mono_results = [_monotonicity_score(bucket_df, c) for c in mono_metrics if c in bucket_df.columns]
    overall_mono = sum(1 for m in mono_results if m.get("monotonic")) >= 2
    mono_report = {
        "overall_monotonic": overall_mono,
        "details": mono_results,
        "q1_bad_rate": float(bucket_df[bucket_df["bucket"] == "Q1"]["bad_trade_rate"].iloc[0]) if len(bucket_df) else None,
        "q4_bad_rate": float(bucket_df[bucket_df["bucket"] == "Q4"]["bad_trade_rate"].iloc[0]) if len(bucket_df) else None,
    }

    (OUT_DIR / "monotonic_risk_bucket_analysis.md").write_text(
        "# Monotonic Risk Bucket Analysis\n\n"
        f"```\n{bucket_df.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )
    (OUT_DIR / "monotonicity_score_report.md").write_text(
        "# Monotonicity Score Report\n\n"
        f"**Overall monotonic:** {overall_mono}\n\n"
        f"```json\n{json.dumps(mono_report, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    routing_forensics = _routing_failure_forensics(eval_df, best_scores, bucket_df)
    overlap_df = _routing_overlap_analysis(eval_df, best_scores)
    overlap_df.to_csv(OUT_DIR / "routing_overlap_analysis.csv", index=False)
    (OUT_DIR / "routing_failure_forensics.md").write_text(
        "# Routing Failure Forensics\n\n"
        f"```json\n{json.dumps(routing_forensics, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    q2_exps = _q2_aligned_experiments(eval_df, best_scores)
    q2_exps.to_csv(OUT_DIR / "q2_aligned_routing_experiments.csv", index=False)

    oos_folds = [] if skip_oos else _rolling_oos_routing(anchor, weights)
    oos_summary = {
        "folds": len(oos_folds),
        "mean_auc": float(np.nanmean([f["auc"] for f in oos_folds])) if oos_folds else None,
        "monotonic_fold_rate": float(np.mean([f["monotonic_bad_trade"] for f in oos_folds])) if oos_folds else None,
        "routing_valid_fold_rate": float(np.mean([f["routing_valid"] for f in oos_folds])) if oos_folds else None,
    }
    (OUT_DIR / "executed_anchor_routing_oos_validation.md").write_text(
        "# Executed-Anchor Routing OOS Validation\n\n"
        f"```json\n{json.dumps(oos_summary, indent=2, default=str)}\n```\n\n"
        f"## Folds\n```json\n{json.dumps(oos_folds[:10], indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    verdict = _final_verdict(mono_report, routing_forensics, oos_folds, q2_exps)
    final = {
        "verdict": verdict,
        "anchor_rows": len(anchor),
        "best_calibration": best_method,
        "overall_monotonic": overall_mono,
        "routing_valid_proxy": routing_forensics.get("routing_valid_proxy"),
        "oos_summary": oos_summary,
        "q2_baseline_unchanged": True,
        "promotion_ready": False,
    }
    (OUT_DIR / "executed_anchor_final_verdict.md").write_text(
        f"# Executed-Anchor Final Verdict\n\n**Verdict:** `{verdict}`\n\n"
        f"Q2_BDI baseline unchanged. Meta live routing forbidden.\n\n"
        f"```json\n{json.dumps(final, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Executed-anchor routing validation")
    parser.add_argument("--skip-oos", action="store_true")
    args = parser.parse_args()
    r = run_routing_validation(skip_oos=args.skip_oos)
    print(f"verdict: {r['verdict']}")
    print(f"anchor_rows: {r['anchor_rows']}")
    print(f"monotonic: {r['overall_monotonic']}")
    print(f"routing_valid: {r['routing_valid_proxy']}")


if __name__ == "__main__":
    main()
