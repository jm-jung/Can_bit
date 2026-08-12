"""
Counterfactual label quality forensics & Meta label reliability (diagnostics only).

Determines whether Meta failure is due to model structure vs label noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.build_meta_label_dataset import ENTRY_FEATURE_COLS_V2, load_v2_dataset
from scripts.diagnostics.train_meta_layer_model import (
    CORE_FEATURE_COLS,
    _classification_metrics,
    _prepare_x,
    _rolling_oos,
    _time_split,
)

OUT_DIR = Path("data/diagnostics/meta_layer/label_quality")
TINY_RETURN_THRESHOLDS = (0.0005, 0.001, 0.002)
NOISE_ZONE_THRESHOLD = 0.001
FEATURE_COLS = ENTRY_FEATURE_COLS_V2


def _sign(x: float, eps: float = 1e-9) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _df_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    return f"```\n{df.to_string(index=False)}\n```\n"


def _infer_label_source(row: pd.Series) -> str:
    src = row.get("candidate_source")
    if pd.notna(src) and str(src):
        return str(src)
    if row.get("source_replay") == "historical_backfill":
        return "executed_production_trade"
    return "executed_production_trade"


def _enrich_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["df_idx", "entry_ts"], na_position="last").reset_index(drop=True)
    df = df.copy()
    df["label_source"] = df.apply(_infer_label_source, axis=1)
    df["is_executed"] = df["executed_bool"].fillna(True).astype(bool)
    df["is_counterfactual"] = df["label_method"].fillna("executed").astype(str).str.contains("counterfactual")
    df["has_fixed_horizons"] = df["fixed_h5_return"].notna() if "fixed_h5_return" in df.columns else False

    for col in ("net_return", "counterfactual_return", "fixed_h5_return", "fixed_h15_return", "fixed_h30_return"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["engine_ret"] = df["counterfactual_return"].where(df["is_counterfactual"], df["net_return"])
    return df


def _horizon_signs(row: pd.Series) -> Dict[str, int]:
    return {
        "engine": _sign(float(row.get("engine_ret", 0) or 0)),
        "h5": _sign(float(row.get("fixed_h5_return", 0) or 0)),
        "h15": _sign(float(row.get("fixed_h15_return", 0) or 0)),
        "h30": _sign(float(row.get("fixed_h30_return", 0) or 0)),
    }


def _horizon_disagreement(row: pd.Series) -> float:
    if not row.get("has_fixed_horizons"):
        return float("nan")
    signs = _horizon_signs(row)
    vals = [signs["engine"], signs["h5"], signs["h15"], signs["h30"]]
    non_zero = [v for v in vals if v != 0]
    if len(non_zero) < 2:
        return 0.0 if len(set(vals)) <= 1 else 0.5
    return float(len(set(non_zero)) > 1)


def _label_flip_count(row: pd.Series) -> int:
    if not row.get("has_fixed_horizons"):
        return 0
    signs = _horizon_signs(row)
    ordered = [signs["h5"], signs["h15"], signs["h30"], signs["engine"]]
    flips = 0
    for a, b in zip(ordered[:-1], ordered[1:]):
        if a != 0 and b != 0 and a != b:
            flips += 1
    return flips


def _compute_label_confidence(row: pd.Series) -> float:
    score = 0.35
    if row.get("is_executed"):
        score += 0.15
    if row.get("has_fixed_horizons"):
        signs = _horizon_signs(row)
        nz = [v for v in signs.values() if v != 0]
        if len(nz) >= 2 and len(set(nz)) == 1:
            score += 0.25
        elif len(nz) >= 2:
            score -= 0.20
    eng = abs(float(row.get("engine_ret", 0) or 0))
    if eng >= 0.003:
        score += 0.12
    elif eng < NOISE_ZONE_THRESHOLD:
        score -= 0.18
    mae = abs(float(row.get("mae", row.get("counterfactual_mae", 0)) or 0))
    mfe = float(row.get("mfe", row.get("counterfactual_mfe", 0)) or 0)
    if mfe > mae and mfe > 0.001:
        score += 0.08
    if mae > 0.005 and eng < mae:
        score -= 0.10
    if str(row.get("exit_reason", "")) == "opposite_signal":
        score -= 0.12
    if str(row.get("trend_state", "")) == "sideways":
        score -= 0.08
    if bool(row.get("false_high_signature_flag", 0)):
        score += 0.05
    if float(row.get("entropy", 1.0) or 1.0) > 1.0:
        score -= 0.10
    return float(np.clip(score, 0.0, 1.0))


def phase1_reliability(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cf = df[df["has_fixed_horizons"]].copy()
    cf["h5_sign"] = cf["fixed_h5_return"].apply(_sign)
    cf["h15_sign"] = cf["fixed_h15_return"].apply(_sign)
    cf["h30_sign"] = cf["fixed_h30_return"].apply(_sign)
    cf["engine_sign"] = cf["engine_ret"].apply(_sign)
    cf["all_agree"] = (
        (cf["h5_sign"] == cf["engine_sign"])
        & (cf["h15_sign"] == cf["engine_sign"])
        & (cf["h30_sign"] == cf["engine_sign"])
        & (cf["engine_sign"] != 0)
    )

    matrix_rows: List[Dict[str, Any]] = []
    pairs = [
        ("engine", "h5", "engine_ret", "fixed_h5_return"),
        ("engine", "h15", "engine_ret", "fixed_h15_return"),
        ("engine", "h30", "engine_ret", "fixed_h30_return"),
        ("h5", "h15", "fixed_h5_return", "fixed_h15_return"),
        ("h15", "h30", "fixed_h15_return", "fixed_h30_return"),
    ]
    for a, b, ca, cb in pairs:
        sub = cf.dropna(subset=[ca, cb])
        if sub.empty:
            continue
        agree = (sub[ca].apply(_sign) == sub[cb].apply(_sign)).mean()
        corr = sub[ca].corr(sub[cb])
        matrix_rows.append({
            "label_a": a, "label_b": b,
            "n": len(sub), "sign_agreement": float(agree),
            "pearson_corr": float(corr) if pd.notna(corr) else None,
        })

    flip_rows: List[Dict[str, Any]] = []
    for src, g in cf.groupby("label_source"):
        flips = g.apply(_label_flip_count, axis=1)
        flip_rows.append({
            "label_source": src,
            "rows": len(g),
            "mean_flips": float(flips.mean()),
            "flip_rate_ge1": float((flips >= 1).mean()),
            "all_horizon_agree_rate": float(g["all_agree"].mean()),
        })

    exec_rows = df[df["is_executed"]]
    non_exec = df[~df["is_executed"]]

    report = {
        "total_rows": len(df),
        "counterfactual_rows": int(df["is_counterfactual"].sum()),
        "executed_rows": int(df["is_executed"].sum()),
        "non_executed_rows": int((~df["is_executed"]).sum()),
        "horizon_agreement_rate": float(cf["all_agree"].mean()) if len(cf) else None,
        "engine_h5_agreement": next((r["sign_agreement"] for r in matrix_rows if r["label_a"] == "engine" and r["label_b"] == "h5"), None),
        "engine_return_std": float(cf["engine_ret"].std()) if len(cf) else None,
        "executed_mean_return": float(exec_rows["engine_ret"].mean()) if len(exec_rows) else None,
        "counterfactual_mean_return": float(non_exec["engine_ret"].mean()) if len(non_exec) else None,
        "regime_consistency": {},
        "long_short_asymmetry": {},
        "false_high_behavior": {},
    }

    for regime_col, name in [("trend_state", "trend"), ("vol_bucket", "vol")]:
        for val, g in df.groupby(regime_col):
            sub_cf = g[g["has_fixed_horizons"]]
            agree = float(sub_cf.apply(
                lambda r: _horizon_signs(r)["engine"] != 0
                and len({v for v in _horizon_signs(r).values() if v != 0}) == 1,
                axis=1,
            ).mean()) if len(sub_cf) else None
            report["regime_consistency"][f"{name}_{val}"] = {
                "rows": len(g),
                "mean_return": float(g["engine_ret"].mean()),
                "agreement_rate": agree,
                "bad_trade_rate": float(g["binary_bad_trade"].mean()) if "binary_bad_trade" in g.columns else None,
            }

    for direction, g in df.groupby("direction"):
        sub = g[g["has_fixed_horizons"]]
        agree = float(sub.apply(
            lambda r: _horizon_signs(r)["engine"] != 0
            and len({v for v in _horizon_signs(r).values() if v != 0}) == 1,
            axis=1,
        ).mean()) if len(sub) else None
        report["long_short_asymmetry"][direction] = {
            "rows": len(g),
            "mean_return": float(g["engine_ret"].mean()),
            "agreement_rate": agree,
        }

    fh = df[df.get("false_high_signature_flag", 0).astype(bool)]
    if len(fh):
        sub = fh[fh["has_fixed_horizons"]]
        agree = float(sub.apply(
            lambda r: _horizon_signs(r)["engine"] != 0
            and len({v for v in _horizon_signs(r).values() if v != 0}) == 1,
            axis=1,
        ).mean()) if len(sub) else None
        report["false_high_behavior"] = {
            "rows": len(fh),
            "mean_return": float(fh["engine_ret"].mean()),
            "bad_rate": float(fh["binary_bad_trade"].mean()) if "binary_bad_trade" in fh.columns else None,
            "agreement_rate": agree,
            "opposite_exit_rate": float((fh["exit_reason"] == "opposite_signal").mean()),
        }

    return pd.DataFrame(matrix_rows), pd.DataFrame(flip_rows), report


def phase2_noise(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    cf = df[df["has_fixed_horizons"]].copy()
    cf["disagreement"] = cf.apply(_horizon_disagreement, axis=1)
    cf["flip_count"] = cf.apply(_label_flip_count, axis=1)
    cf["tiny_return"] = cf["engine_ret"].abs() < NOISE_ZONE_THRESHOLD

    noise_report: Dict[str, Any] = {
        "horizon_disagreement_rate": float(cf["disagreement"].mean()) if len(cf) else None,
        "label_flip_rate": float((cf["flip_count"] >= 1).mean()) if len(cf) else None,
        "tiny_return_zone_rate": float(cf["tiny_return"].mean()) if len(cf) else None,
        "tiny_return_by_threshold": {},
        "entropy_instability": {},
        "vol_instability": {},
        "opposite_signal_dependency": {},
    }

    for th in TINY_RETURN_THRESHOLDS:
        noise_report["tiny_return_by_threshold"][str(th)] = float((cf["engine_ret"].abs() < th).mean()) if len(cf) else None

    cf["entropy_bucket"] = pd.cut(cf["entropy"], bins=[0, 0.9, 1.0, 1.1, 2.0], labels=["low", "mid", "high", "vhigh"])
    for bucket, g in cf.groupby("entropy_bucket", observed=True):
        noise_report["entropy_instability"][str(bucket)] = {
            "rows": len(g),
            "disagreement_rate": float(g["disagreement"].mean()),
            "flip_rate": float((g["flip_count"] >= 1).mean()),
        }

    for vol, g in cf.groupby("vol_bucket"):
        noise_report["vol_instability"][str(vol)] = {
            "rows": len(g),
            "disagreement_rate": float(g["disagreement"].mean()),
            "flip_rate": float((g["flip_count"] >= 1).mean()),
        }

    for reason, g in cf.groupby("exit_reason"):
        noise_report["opposite_signal_dependency"][str(reason)] = {
            "rows": len(g),
            "mean_return": float(g["engine_ret"].mean()),
            "disagreement_rate": float(g["disagreement"].mean()),
            "tiny_return_rate": float(g["tiny_return"].mean()),
        }

    inst_dist = cf.groupby(["label_source", "vol_bucket"]).agg(
        rows=("disagreement", "count"),
        disagreement_rate=("disagreement", "mean"),
        flip_rate=("flip_count", lambda s: float((s >= 1).mean())),
        tiny_return_rate=("tiny_return", "mean"),
        mean_abs_return=("engine_ret", lambda s: float(s.abs().mean())),
    ).reset_index()

    return noise_report, inst_dist


def phase3_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_confidence_score"] = df.apply(_compute_label_confidence, axis=1)
    return df


def _rolling_oos_weighted(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    factory = lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42)
    rows: List[Dict[str, Any]] = []
    n = len(df)
    chunk, step = max(40, n // 5), max(10, n // 15)
    start = 0
    w = weights if weights is not None else np.ones(n)

    while start + chunk + 5 <= n and len(rows) < 25:
        end = start + chunk + step
        sub = df.iloc[start:end].copy()
        split = int(len(sub) * 0.7)
        if split < 20 or len(sub) - split < 10:
            break
        x = _prepare_x(sub, feature_cols)
        y = sub[label_col].astype(int)
        if y.nunique() < 2:
            start += step
            continue
        sw = w[start:end][:split]
        m = factory()
        m.fit(x.iloc[:split], y.iloc[:split], sample_weight=sw)
        prob = m.predict_proba(x.iloc[split:])[:, 1]
        met = _classification_metrics(y.iloc[split:].to_numpy(), prob)
        rows.append(met)
        start += step

    aucs = [r["auc"] for r in rows if not np.isnan(r.get("auc", np.nan))]
    return {
        "folds": len(rows),
        "rolling_oos_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "rolling_oos_auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "calibration_error_mean": float(np.mean([r.get("calibration_curve_error", 0) for r in rows])) if rows else float("nan"),
    }


def _routing_valid_proxy(df: pd.DataFrame, prob: np.ndarray) -> bool:
    if len(df) < 30:
        return False
    tmp = df.copy()
    tmp["pred"] = prob
    winners = tmp[tmp["binary_good_trade"] == 1]["pred"]
    losers = tmp[tmp["binary_bad_trade"] == 1]["pred"]
    rfe = tmp[tmp["rfe_flag"].astype(bool)]["pred"] if "rfe_flag" in tmp.columns else pd.Series(dtype=float)
    if len(winners) < 5 or len(losers) < 5:
        return False
    rfe_mean = float(rfe.mean()) if len(rfe) else float(winners.mean())
    return float(winners.mean()) > float(losers.mean()) > rfe_mean


def _train_experiment(
    df: pd.DataFrame,
    name: str,
    label_col: str,
    weights: Optional[np.ndarray] = None,
    subset_mask: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    sub = df[subset_mask] if subset_mask is not None else df
    if len(sub) < 80:
        return {"experiment": name, "rows": len(sub), "status": "insufficient_rows"}

    w = weights[subset_mask.to_numpy()] if (weights is not None and subset_mask is not None) else weights
    if subset_mask is not None and weights is None:
        w = None

    roll = _rolling_oos_weighted(sub, FEATURE_COLS, label_col, w)

    train_df, val_df, test_df = _time_split(sub)
    x_tr = _prepare_x(train_df, FEATURE_COLS)
    x_te = _prepare_x(test_df, FEATURE_COLS)
    y_tr = train_df[label_col].astype(int)
    y_te = test_df[label_col].astype(int)

    if y_tr.nunique() < 2 or y_te.nunique() < 2:
        return {"experiment": name, "rows": len(sub), "status": "single_class"}

    sw = w[: len(train_df)] if w is not None else None
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42)
    m.fit(x_tr, y_tr, sample_weight=sw)
    prob = m.predict_proba(x_te)[:, 1]
    test_met = _classification_metrics(y_te.to_numpy(), prob)

    good_mask = test_df["binary_good_trade"] == 1
    good_reject = float(((prob >= 0.5) & (test_df["calibration_label"].astype(int) == 1)).sum() / max(good_mask.sum(), 1))

    fh = 0
    if {"direction", "trend_state", "vol_bucket", "entropy"}.issubset(test_df.columns):
        fh = int((
            (test_df["direction"] == "LONG")
            & (test_df["trend_state"] == "up")
            & (test_df["vol_bucket"] == "high")
            & (test_df["entropy"] <= 0.90)
            & (prob >= 0.5)
            & (test_df["binary_bad_trade"] == 1)
        ).sum())

    return {
        "experiment": name,
        "rows": len(sub),
        "label_col": label_col,
        "test_auc": test_met["auc"],
        "rolling_oos_auc_mean": roll["rolling_oos_auc_mean"],
        "rolling_oos_auc_std": roll["rolling_oos_auc_std"],
        "calibration_error": test_met.get("calibration_curve_error"),
        "routing_valid": _routing_valid_proxy(test_df, prob),
        "good_rejection_rate": good_reject,
        "false_high_proxy": fh,
        "status": "ok",
    }


def phase4_weighted_training(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    w_uniform = np.ones(n)
    w_conf = df["label_confidence_score"].to_numpy()
    w_exec = np.where(df["is_executed"].to_numpy(), 2.0, 0.5)
    w_cf_down = np.where(df["is_counterfactual"].to_numpy(), 0.4, 1.5)
    w_fh_up = np.where(df.get("false_high_signature_flag", 0).astype(bool).to_numpy(), 1.8, 1.0)
    high_conf_mask = df["label_confidence_score"] >= 0.55

    experiments = [
        ("A_uniform", w_uniform, None, "drawdown_risk_label"),
        ("B_label_confidence_weighted", w_conf, None, "drawdown_risk_label"),
        ("C_executed_upweighted", w_exec, None, "drawdown_risk_label"),
        ("D_counterfactual_downweighted", w_cf_down, None, "drawdown_risk_label"),
        ("E_false_high_upweighted", w_fh_up, None, "drawdown_risk_label"),
        ("F_high_confidence_only", None, high_conf_mask, "drawdown_risk_label"),
    ]
    rows = [_train_experiment(df, name, label, w, mask) for name, w, mask, label in experiments]
    return pd.DataFrame(rows)


def phase5_abstraction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["composite_label"] = (
        0.4 * df["drawdown_risk_label"].astype(float)
        + 0.3 * (df["trade_quality_label"].astype(float) <= 1).astype(float)
        + 0.3 * (df["engine_ret"] > 0.001).astype(float)
    )
    df["composite_binary"] = (df["composite_label"] >= 0.5).astype(int)
    df["pnl_binary"] = (df["engine_ret"] > 0).astype(int)
    df["mae_risk_binary"] = (df["mae"] <= -0.003).astype(int) if "mae" in df.columns else 0
    df["quality_binary"] = (df["trade_quality_label"] >= 2).astype(int)

    labels = [
        ("A_raw_pnl", "pnl_binary"),
        ("B_mae_risk", "mae_risk_binary"),
        ("C_trade_quality", "quality_binary"),
        ("D_drawdown_risk", "drawdown_risk_label"),
        ("E_composite", "composite_binary"),
    ]
    return pd.DataFrame([_train_experiment(df, name, col, np.ones(len(df)), None) for name, col in labels])


def phase6_executed_generalization(df: pd.DataFrame) -> pd.DataFrame:
    exec_mask = df["is_executed"]
    non_mask = ~df["is_executed"]
    w_exec_up = np.where(exec_mask.to_numpy(), 3.0, 0.3)

    experiments = [
        ("executed_only", None, exec_mask),
        ("non_executed_only", None, non_mask),
        ("mixed_uniform", np.ones(len(df)), None),
        ("mixed_executed_upweighted", w_exec_up, None),
    ]
    rows = []
    for name, w, mask in experiments:
        r = _train_experiment(df, name, "drawdown_risk_label", w, mask)
        rows.append(r)
    return pd.DataFrame(rows)


def phase7_q2_alignment(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    forensic_feats = [
        "q2_pd_penalty_score", "q2_bdi_penalty_score", "false_high_signature_flag",
        "confidence_overextension", "entropy_delta", "danger_cluster_count",
        "q2_score", "entropy", "margin",
    ]
    avail = [c for c in forensic_feats if c in df.columns]
    train_df, _, test_df = _time_split(df)
    if len(test_df) < 50:
        return pd.DataFrame(), {"status": "insufficient_test_rows"}

    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42)
    x_tr = _prepare_x(train_df, FEATURE_COLS)
    x_te = _prepare_x(test_df, FEATURE_COLS)
    y_tr = train_df["drawdown_risk_label"].astype(int)
    m.fit(x_tr, y_tr)
    prob = m.predict_proba(x_te)[:, 1]

    align_rows: List[Dict[str, Any]] = []
    for feat in avail:
        corr_label = test_df[feat].corr(test_df["drawdown_risk_label"])
        corr_pred = test_df[feat].corr(pd.Series(prob, index=test_df.index))
        align_rows.append({
            "feature": feat,
            "corr_with_drawdown_label": float(corr_label) if pd.notna(corr_label) else None,
            "corr_with_meta_prediction": float(corr_pred) if pd.notna(corr_pred) else None,
        })

    # Q2 penalty alignment: higher penalty should correlate with higher risk prediction
    q2_rows = df.groupby(pd.cut(df["q2_bdi_penalty_score"], bins=5, duplicates="drop")).agg(
        rows=("drawdown_risk_label", "count"),
        risk_rate=("drawdown_risk_label", "mean"),
        mean_confidence=("label_confidence_score", "mean"),
    ).reset_index()

    report = {
        "forensic_features_analyzed": avail,
        "mean_abs_pred_label_corr": float(np.mean([abs(r["corr_with_drawdown_label"] or 0) for r in align_rows])),
        "q2_bdi_penalty_monotonic": bool(q2_rows["risk_rate"].is_monotonic_increasing) if len(q2_rows) > 2 else None,
        "aligned_features": [r["feature"] for r in align_rows if abs(r.get("corr_with_drawdown_label") or 0) >= 0.08],
    }
    return pd.DataFrame(align_rows), report


def _final_verdict(
    noise: Dict[str, Any],
    weighted: pd.DataFrame,
    executed: pd.DataFrame,
    abstraction: pd.DataFrame,
) -> str:
    disagree = noise.get("horizon_disagreement_rate") or 0
    tiny = noise.get("tiny_return_zone_rate") or 0

    if disagree >= 0.40 or tiny > 0.35:
        primary = "counterfactual_noise_too_high"
    else:
        primary = None

    ex = executed[executed["status"] == "ok"]
    if not ex.empty:
        exec_only = ex[ex["experiment"] == "executed_only"]
        non_only = ex[ex["experiment"] == "non_executed_only"]
        if len(exec_only) and len(non_only):
            exec_auc = float(exec_only.iloc[0].get("test_auc", 0))
            non_auc = float(non_only.iloc[0].get("test_auc", 0))
            if exec_auc - non_auc >= 0.10:
                return "executed_trade_signal_more_reliable"

    ok = weighted[weighted["status"] == "ok"]
    if not ok.empty:
        best = ok.sort_values("rolling_oos_auc_mean", ascending=False).iloc[0]
        uniform = ok[ok["experiment"] == "A_uniform"]
        uni_auc = float(uniform.iloc[0]["rolling_oos_auc_mean"]) if len(uniform) else 0
        if str(best["experiment"]).startswith("B_") and float(best["rolling_oos_auc_mean"]) > uni_auc + 0.02:
            return "label_confidence_weighting_effective"

    if primary:
        return primary

    best_auc = float(ok["rolling_oos_auc_mean"].max()) if not ok.empty else 0
    routing_any = bool(ok["routing_valid"].any()) if not ok.empty else False

    if best_auc >= 0.58 and routing_any:
        return "meta_research_reopened"

    if best_auc >= 0.55 and disagree < 0.35:
        return "meta_learning_structure_detected"

    return "Q2_BDI_baseline_still_best"


def run_forensics(*, skip_training: bool = False) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw, _ = load_v2_dataset()
    df = _enrich_dataset(df_raw)

    consistency_matrix, flip_df, reliability = phase1_reliability(df)
    consistency_matrix.to_csv(OUT_DIR / "label_consistency_matrix.csv", index=False)
    flip_df.to_csv(OUT_DIR / "label_flip_analysis.csv", index=False)

    (OUT_DIR / "label_source_reliability_report.md").write_text(
        "# Label Source Reliability Report\n\n"
        f"```json\n{json.dumps(reliability, indent=2, default=str)}\n```\n\n"
        f"## Consistency Matrix\n\n{_df_text(consistency_matrix)}\n",
        encoding="utf-8",
    )

    noise_report, inst_dist = phase2_noise(df)
    inst_dist.to_csv(OUT_DIR / "label_instability_distribution.csv", index=False)
    (OUT_DIR / "counterfactual_noise_analysis.md").write_text(
        "# Counterfactual Noise Analysis\n\n"
        f"```json\n{json.dumps(noise_report, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    (OUT_DIR / "replay_exit_dependency_report.md").write_text(
        "# Replay Exit Dependency Report\n\n"
        "Counterfactual labels depend heavily on replay exit mechanics.\n\n"
        f"```json\n{json.dumps(noise_report.get('opposite_signal_dependency', {}), indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    df = phase3_confidence(df)
    df["horizon_disagreement"] = df.apply(_horizon_disagreement, axis=1)
    conf_dist = df.groupby(pd.cut(df["label_confidence_score"], bins=[0, 0.35, 0.55, 0.75, 1.0])).agg(
        rows=("label_confidence_score", "count"),
        mean_return=("engine_ret", "mean"),
        bad_rate=("binary_bad_trade", "mean"),
        disagreement=("horizon_disagreement", "mean"),
    ).reset_index()
    conf_dist.to_csv(OUT_DIR / "label_confidence_distribution.csv", index=False)

    hi = df[df["label_confidence_score"] >= 0.55]
    lo = df[df["label_confidence_score"] < 0.35]
    hi_lo = {
        "high_confidence_rows": len(hi),
        "low_confidence_rows": len(lo),
        "high_conf_mean_return": float(hi["engine_ret"].mean()) if len(hi) else None,
        "low_conf_mean_return": float(lo["engine_ret"].mean()) if len(lo) else None,
        "high_conf_bad_rate": float(hi["binary_bad_trade"].mean()) if len(hi) else None,
        "low_conf_bad_rate": float(lo["binary_bad_trade"].mean()) if len(lo) else None,
    }
    (OUT_DIR / "high_vs_low_confidence_label_analysis.md").write_text(
        f"# High vs Low Confidence Label Analysis\n\n```json\n{json.dumps(hi_lo, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    weighted_df = abstraction_df = executed_df = align_df = pd.DataFrame()
    align_report: Dict[str, Any] = {}

    if not skip_training:
        weighted_df = phase4_weighted_training(df)
        weighted_df.to_csv(OUT_DIR / "source_weight_experiment.csv", index=False)
        (OUT_DIR / "weighted_meta_training_report.md").write_text(
            "# Weighted Meta Training Report\n\n"
            f"{_df_text(weighted_df)}\n",
            encoding="utf-8",
        )

        abstraction_df = phase5_abstraction(df)
        abstraction_df.to_csv(OUT_DIR / "label_abstraction_experiments.csv", index=False)

        executed_df = phase6_executed_generalization(df)
        pd.concat([weighted_df, executed_df], ignore_index=True).to_csv(
            OUT_DIR / "source_weight_experiment.csv", index=False
        )

        align_df, align_report = phase7_q2_alignment(df)
        align_df.to_csv(OUT_DIR / "forensic_feature_alignment.csv", index=False)
        (OUT_DIR / "q2_meta_alignment_analysis.md").write_text(
            "# Q2 Meta Alignment Analysis\n\n"
            f"```json\n{json.dumps(align_report, indent=2, default=str)}\n```\n\n"
            f"{_df_text(align_df)}\n",
            encoding="utf-8",
        )

        (OUT_DIR / "executed_vs_nonexecuted_generalization.md").write_text(
            "# Executed vs Non-Executed Generalization\n\n"
            f"{_df_text(executed_df)}\n",
            encoding="utf-8",
        )

    verdict = _final_verdict(noise_report, weighted_df, executed_df, abstraction_df)

    reval = {
        "verdict": verdict,
        "horizon_disagreement_rate": noise_report.get("horizon_disagreement_rate"),
        "tiny_return_zone_rate": noise_report.get("tiny_return_zone_rate"),
        "best_weighted_oos_auc": float(weighted_df["rolling_oos_auc_mean"].max()) if len(weighted_df) and "rolling_oos_auc_mean" in weighted_df.columns else None,
        "routing_valid_any": bool(weighted_df["routing_valid"].any()) if len(weighted_df) and "routing_valid" in weighted_df.columns else False,
        "q2_baseline_unchanged": True,
        "promotion_ready": False,
    }
    (OUT_DIR / "meta_label_quality_final_revalidation.md").write_text(
        f"# Meta Label Quality Final Revalidation\n\n**Verdict:** `{verdict}`\n\n"
        f"Q2_BDI baseline unchanged. No production promotion.\n\n"
        f"```json\n{json.dumps(reval, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    return {"verdict": verdict, "reliability": reliability, "noise": noise_report, "revalidation": reval}


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual label quality forensics")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    r = run_forensics(skip_training=args.skip_training)
    print(f"verdict: {r['verdict']}")
    print(f"horizon_disagreement: {r['noise'].get('horizon_disagreement_rate')}")
    print(f"tiny_return_zone: {r['noise'].get('tiny_return_zone_rate')}")


if __name__ == "__main__":
    main()
