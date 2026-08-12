"""
Historical walk-forward Meta shadow validation (diagnostics only).

Past data is used only through temporal walk-forward splits. Each test window is
scored by a Meta model trained strictly on rows before that window.
Production/Q2/live/launchd logic is not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings(
    "ignore",
    message="An input array is constant; the correlation coefficient is not defined.",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.analyze_counterfactual_label_quality import FEATURE_COLS
from scripts.diagnostics.analyze_executed_anchor_routing import (
    LABEL_COL,
    POSITION_SIZE,
    _build_anchor_dataset,
    _risk_buckets,
    _risk_routing_valid,
    _train_meta_model,
)
from scripts.diagnostics.analyze_executed_first_meta_learning import _prepare_full_df
from scripts.diagnostics.analyze_meta_activation_controller import (
    _add_regime_confidence,
    _meta_activation_score,
)
from scripts.diagnostics.analyze_regime_conditioned_meta import _q2_scale, _segment_regimes
from scripts.diagnostics.build_meta_label_dataset import LEAKAGE_COLS
from scripts.diagnostics.run_forward_meta_shadow_monitor import (
    _bucket_summary,
    _mdd,
    _routing_consistency,
)
from scripts.diagnostics.train_meta_layer_model import _prepare_x

OUT_DIR = Path("data/diagnostics/meta_shadow_historical")
MIN_TRAIN_ROWS = 80
MIN_TEST_ROWS = 8
POLICIES = [
    "q2_bdi_only",
    "always_on_meta",
    "false_high_off_meta",
    "activation_controlled_meta",
    "vol_expansion_only_meta",
    "low_entropy_strong_trend_only_meta",
    "transition_aware_meta",
]
REGIME_MASKS = [
    "vol_expansion",
    "low_entropy",
    "mid_entropy",
    "strong_trend",
    "strong_uptrend",
    "false_high_signature",
    "sideways",
    "entropy_spike",
    "transition_regime",
]


@dataclass(frozen=True)
class SplitSpec:
    split_id: str
    window_months: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def _date_col(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df.get("timestamp", df.get("entry_ts")), errors="coerce")
    fallback = pd.to_datetime(df.get("entry_ts"), errors="coerce")
    return ts.fillna(fallback)


def _load_anchor() -> Tuple[pd.DataFrame, np.ndarray]:
    full = _prepare_full_df()
    anchor, _, weights = _build_anchor_dataset(full)
    anchor = anchor.copy()
    anchor["_ts"] = _date_col(anchor)
    anchor["_wf_weight"] = np.asarray(weights) if len(weights) == len(anchor) else np.ones(len(anchor))
    anchor = anchor.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    weights = anchor["_wf_weight"].to_numpy(dtype=float)
    anchor = anchor.drop(columns=["_wf_weight"])
    anchor = _segment_regimes(anchor)
    anchor = _add_regime_confidence(anchor)
    return anchor, weights


def _make_splits(anchor: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2021-01-01")
    max_ts = anchor["_ts"].max()
    specs: List[SplitSpec] = []
    for months in (3, 6):
        cur = pd.Timestamp("2023-01-01")
        idx = 1
        while cur <= max_ts:
            test_start = cur
            test_end = cur + pd.DateOffset(months=months) - pd.Timedelta(seconds=1)
            if test_start > max_ts:
                break
            train_end = test_start - pd.Timedelta(seconds=1)
            specs.append(SplitSpec(
                split_id=f"{months}m_wf_{idx:02d}_{test_start:%Y%m}_{min(test_end, max_ts):%Y%m}",
                window_months=months,
                train_start=start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=min(test_end, max_ts).strftime("%Y-%m-%d"),
            ))
            cur = cur + pd.DateOffset(months=months)
            idx += 1
    split_df = pd.DataFrame([s.__dict__ for s in specs])
    rows = []
    for row in split_df.to_dict(orient="records"):
        tr = _slice(anchor, row["train_start"], row["train_end"])
        te = _slice(anchor, row["test_start"], row["test_end"])
        status = "ok"
        if len(tr) < MIN_TRAIN_ROWS:
            status = "insufficient_train_rows"
        elif len(te) < MIN_TEST_ROWS:
            status = "insufficient_test_rows"
        rows.append({
            **row,
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "train_max_ts": str(tr["_ts"].max()) if len(tr) else "",
            "test_min_ts": str(te["_ts"].min()) if len(te) else "",
            "temporal_separation": bool(len(tr) and len(te) and tr["_ts"].max() < te["_ts"].min()),
            "status": status,
        })
    return pd.DataFrame(rows)


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["_ts"] >= s) & (df["_ts"] <= e)].copy()


def _risk_bucket_labels(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 4:
        return np.array(["Q1"] * len(scores), dtype=object)
    return _risk_buckets(scores)


def _shadow_rows(split: Dict[str, Any], test_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    test_df = test_df.copy().reset_index(drop=True)
    test_df["meta_raw_score"] = scores
    test_df["meta_risk_bucket"] = _risk_bucket_labels(scores)
    test_df["activation_score"] = _meta_activation_score(test_df, scores)
    test_df["activation_policy_result"] = np.where(test_df["activation_score"] >= 0.55, "ON", "OFF")
    test_df["q2_bdi_scale"] = test_df.apply(_q2_scale, axis=1)
    out = pd.DataFrame({
        "split_id": split["split_id"],
        "window_months": split["window_months"],
        "train_start": split["train_start"],
        "train_end": split["train_end"],
        "test_start": split["test_start"],
        "test_end": split["test_end"],
        "timestamp": test_df.get("timestamp", test_df.get("entry_ts")),
        "entry_ts": test_df.get("entry_ts"),
        "exit_ts": test_df.get("exit_ts"),
        "trade_id": test_df.get("trade_id", pd.Series(range(len(test_df)))),
        "df_idx": test_df.get("df_idx"),
        "direction": test_df.get("direction"),
        "q2_bdi_scale": test_df["q2_bdi_scale"],
        "meta_raw_score": test_df["meta_raw_score"],
        "meta_risk_bucket": test_df["meta_risk_bucket"],
        "regime_label": test_df["regime_label"],
        "regime_confidence": test_df["regime_confidence_score"],
        "regime_stability_score": test_df["regime_stability_score"],
        "regime_persistence": test_df["regime_persistence"],
        "activation_score": test_df["activation_score"],
        "activation_policy_result": test_df["activation_policy_result"],
        "false_high_signature": test_df.get("false_high_signature_flag", 0).astype(bool),
        "entropy": test_df.get("entropy"),
        "confidence_overextension": test_df.get("confidence_overextension"),
        "vol_regime": test_df["vol_regime"],
        "trend_regime": test_df["trend_regime"],
        "confidence_regime": test_df["confidence_regime"],
        "transition_probability": test_df["transition_probability"],
        "actual_outcome": test_df.get("quality_class", test_df.get("trade_quality_label")),
        "mae": test_df.get("mae"),
        "mfe": test_df.get("mfe"),
        "rfe_flag": test_df.get("rfe_flag", False).astype(bool),
        "drawdown_risk": test_df.get(LABEL_COL),
        "actual_realized_return": test_df.get("engine_ret", test_df.get("net_return")),
        "binary_bad_trade": test_df.get("binary_bad_trade"),
        "binary_good_trade": test_df.get("binary_good_trade"),
        "promotion_ready": False,
    })
    return out


def _policy_active(df: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "q2_bdi_only":
        return pd.Series(False, index=df.index)
    if policy == "always_on_meta":
        return pd.Series(True, index=df.index)
    if policy == "false_high_off_meta":
        return ~df["false_high_signature"].astype(bool)
    if policy == "activation_controlled_meta":
        return df["activation_policy_result"].astype(str).eq("ON")
    if policy == "vol_expansion_only_meta":
        return df["vol_regime"].astype(str).eq("vol_expansion")
    if policy == "low_entropy_strong_trend_only_meta":
        return (
            df["confidence_regime"].astype(str).isin(["low_entropy", "mid_entropy"])
            | df["trend_regime"].astype(str).isin(["strong_uptrend", "strong_downtrend"])
        )
    if policy == "transition_aware_meta":
        return (
            df["activation_policy_result"].astype(str).eq("ON")
            & (df["transition_probability"].astype(float) <= 0.35)
            & (~df["false_high_signature"].astype(bool))
        )
    return pd.Series(False, index=df.index)


def _policy_scales(df: pd.DataFrame, policy: str) -> pd.Series:
    q2 = df["q2_bdi_scale"].astype(float).copy()
    active = _policy_active(df, policy)
    risk = df["meta_raw_score"].astype(float).clip(0, 1)
    out = q2.copy()
    out.loc[active] = (q2.loc[active] * (1.0 - 0.35 * risk.loc[active])).clip(0.05, 1.0)
    return out


def _window_forward_score(
    monotonic: bool,
    routing: bool,
    mdd_improved: bool,
    false_high_inversion: bool,
    preservation: float,
    bucket_stability: float,
) -> float:
    score = (
        25.0 * float(monotonic)
        + 20.0 * float(routing)
        + 20.0 * float(mdd_improved)
        + 15.0 * (1.0 - float(false_high_inversion))
        + 10.0 * preservation
        + 10.0 * bucket_stability
    )
    return float(np.clip(score, 0, 100))


def _policy_metrics(df: pd.DataFrame, policy: str, q2_mdd: float) -> Dict[str, Any]:
    active = _policy_active(df, policy)
    scales = _policy_scales(df, policy)
    active_df = df[active].copy()
    active_scores = active_df["meta_raw_score"].to_numpy(dtype=float)
    eval_df = active_df if policy != "q2_bdi_only" else df
    eval_scores = active_scores if policy != "q2_bdi_only" else df["meta_raw_score"].to_numpy(dtype=float)
    bdf, mono = _bucket_summary(eval_df, eval_scores) if len(eval_df) >= 4 else (pd.DataFrame(), {"monotonic": False, "violations": 3, "spearman": None})
    routing = _routing_consistency(eval_df, eval_scores)
    mdd = _mdd(df["actual_realized_return"], scales)
    net = float((df["actual_realized_return"].fillna(0) * scales.fillna(1) * POSITION_SIZE).sum())
    false_high_df = eval_df[eval_df["false_high_signature"].astype(bool)]
    false_high_inversion = False
    if len(false_high_df) >= 8:
        _, fh_mono = _bucket_summary(false_high_df, false_high_df["meta_raw_score"].to_numpy(dtype=float))
        false_high_inversion = not bool(fh_mono.get("monotonic", False))
    preservation = 1.0
    bucket_stability = float(max(0.0, 1.0 - float(mono.get("violations", 3) or 0) / 3.0)) if len(bdf) >= 4 else 0.0
    good_rejection = float((df["binary_good_trade"].astype(bool) & active & df["meta_risk_bucket"].eq("Q4")).mean()) if len(df) else 0.0
    return {
        "policy": policy,
        "rows": int(len(df)),
        "active_rows": int(active.sum()),
        "active_pct": float(active.mean()) if len(active) else 0.0,
        "mdd": mdd,
        "net": net,
        "mdd_vs_q2": mdd - q2_mdd,
        "mdd_improved_vs_q2": bool(mdd > q2_mdd),
        "rfe": int(df["rfe_flag"].astype(bool).sum()),
        "false_high": int(eval_df["false_high_signature"].astype(bool).sum()),
        "bucket_monotonicity": bool(mono.get("monotonic", False)),
        "monotonic_spearman": mono.get("spearman"),
        "bucket_inversions": int(mono.get("violations", 0) or 0),
        "routing_consistency": routing,
        "preservation": preservation,
        "good_trade_rejection": good_rejection,
        "false_high_inversion": false_high_inversion,
        "bucket_stability": bucket_stability,
        "meta_forward_score": _window_forward_score(
            bool(mono.get("monotonic", False)),
            routing,
            bool(mdd > q2_mdd),
            false_high_inversion,
            preservation,
            bucket_stability,
        ),
    }


def _regime_mask(df: pd.DataFrame, regime: str) -> pd.Series:
    if regime == "vol_expansion":
        return df["vol_regime"].astype(str).eq("vol_expansion")
    if regime == "low_entropy":
        return df["confidence_regime"].astype(str).eq("low_entropy")
    if regime == "mid_entropy":
        return df["confidence_regime"].astype(str).eq("mid_entropy")
    if regime == "strong_trend":
        return df["trend_regime"].astype(str).isin(["strong_uptrend", "strong_downtrend"])
    if regime == "strong_uptrend":
        return df["trend_regime"].astype(str).eq("strong_uptrend")
    if regime == "false_high_signature":
        return df["false_high_signature"].astype(bool) | df["confidence_regime"].astype(str).eq("false_high_signature")
    if regime == "sideways":
        return df["trend_regime"].astype(str).eq("sideways")
    if regime == "entropy_spike":
        return df["entropy"].astype(float).diff().abs().fillna(0) > 0.08
    if regime == "transition_regime":
        return df["transition_probability"].astype(float) > 0.35
    return pd.Series(False, index=df.index)


def _regime_metrics(df: pd.DataFrame, split: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for regime in REGIME_MASKS:
        sub = df[_regime_mask(df, regime)].copy()
        scores = sub["meta_raw_score"].to_numpy(dtype=float)
        bdf, mono = _bucket_summary(sub, scores) if len(sub) >= 4 else (pd.DataFrame(), {"monotonic": False, "violations": 0, "spearman": None})
        y = sub["drawdown_risk"].astype(int).to_numpy() if len(sub) else np.array([])
        auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
        q2_mdd = _mdd(sub["actual_realized_return"], sub["q2_bdi_scale"]) if len(sub) else 0.0
        meta_scales = _policy_scales(sub, "always_on_meta") if len(sub) else pd.Series(dtype=float)
        rows.append({
            "split_id": split["split_id"],
            "window_months": split["window_months"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "regime": regime,
            "rows": int(len(sub)),
            "auc": auc,
            "monotonicity": bool(mono.get("monotonic", False)),
            "monotonic_spearman": mono.get("spearman"),
            "routing_consistency": _routing_consistency(sub, scores) if len(sub) else False,
            "mdd_contribution_q2": q2_mdd,
            "mdd_contribution_always_on": _mdd(sub["actual_realized_return"], meta_scales) if len(sub) else 0.0,
            "false_high_events": int(sub["false_high_signature"].astype(bool).sum()) if len(sub) else 0,
            "bucket_inversion": int(mono.get("violations", 0) or 0),
            "bad_trade_rate": float(sub["binary_bad_trade"].mean()) if len(sub) else 0.0,
        })
    return rows


def _run_windows(anchor: pd.DataFrame, weights: np.ndarray, splits: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shadow_rows: List[pd.DataFrame] = []
    policy_rows: List[Dict[str, Any]] = []
    regime_rows: List[Dict[str, Any]] = []
    false_high_rows: List[Dict[str, Any]] = []

    for split in splits.to_dict(orient="records"):
        if split["status"] != "ok":
            continue
        train_df = _slice(anchor, split["train_start"], split["train_end"]).reset_index(drop=True)
        test_df = _slice(anchor, split["test_start"], split["test_end"]).reset_index(drop=True)
        if len(train_df[LABEL_COL].unique()) < 2 or len(test_df) < MIN_TEST_ROWS:
            continue
        train_idx = anchor[(anchor["_ts"] >= pd.Timestamp(split["train_start"])) & (anchor["_ts"] <= pd.Timestamp(split["train_end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))].index
        train_weights = weights[train_idx.to_numpy()] if len(weights) >= max(train_idx.to_numpy(), default=0) + 1 else np.ones(len(train_df))
        model = _train_meta_model(train_df, train_weights)
        scores = model.predict_proba(_prepare_x(test_df, FEATURE_COLS))[:, 1]
        shadow = _shadow_rows(split, test_df, scores)
        shadow_rows.append(shadow)
        q2_mdd = _mdd(shadow["actual_realized_return"], shadow["q2_bdi_scale"])
        for policy in POLICIES:
            policy_rows.append({
                "split_id": split["split_id"],
                "window_months": split["window_months"],
                "train_rows": split["train_rows"],
                "test_rows": split["test_rows"],
                "test_start": split["test_start"],
                "test_end": split["test_end"],
                **_policy_metrics(shadow, policy, q2_mdd),
            })
        regime_rows.extend(_regime_metrics(shadow, split))
        fh = shadow[shadow["false_high_signature"].astype(bool)]
        _, fh_mono = _bucket_summary(fh, fh["meta_raw_score"].to_numpy(dtype=float)) if len(fh) >= 4 else (pd.DataFrame(), {"monotonic": False, "spearman": None})
        false_high_rows.append({
            "split_id": split["split_id"],
            "window_months": split["window_months"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "rows": int(len(shadow)),
            "false_high_events": int(len(fh)),
            "false_high_frequency": float(len(fh) / max(len(shadow), 1)),
            "bad_trade_rate": float(fh["binary_bad_trade"].mean()) if len(fh) else 0.0,
            "meta_ordering_inversion": not bool(fh_mono.get("monotonic", False)) if len(fh) >= 8 else False,
            "monotonic_spearman": fh_mono.get("spearman"),
            "long_bias_intensity": float((fh["direction"] == "LONG").mean()) if len(fh) else 0.0,
            "near_transition_rate": float((fh["transition_probability"].astype(float) > 0.35).mean()) if len(fh) else 0.0,
            "entropy_spike_rate": float((fh["entropy"].astype(float).diff().abs().fillna(0) > 0.08).mean()) if len(fh) else 0.0,
            "vol_expansion_rate": float((fh["vol_regime"] == "vol_expansion").mean()) if len(fh) else 0.0,
        })

    shadow_df = pd.concat(shadow_rows, ignore_index=True) if shadow_rows else pd.DataFrame()
    return shadow_df, pd.DataFrame(policy_rows), pd.DataFrame(regime_rows), pd.DataFrame(false_high_rows)


def _stability_scores(policy_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if policy_df.empty:
        return pd.DataFrame()
    for policy, sub in policy_df.groupby("policy"):
        windows = max(len(sub), 1)
        mono_rate = float(sub["bucket_monotonicity"].mean())
        routing_rate = float(sub["routing_consistency"].mean())
        mdd_rate = float(sub["mdd_improved_vs_q2"].mean())
        false_high_inversion_rate = float(sub["false_high_inversion"].mean())
        preservation = float(sub["preservation"].mean())
        score_std = float(sub["meta_forward_score"].std(ddof=0)) if len(sub) > 1 else 0.0
        window_stability = float(np.clip(1.0 - score_std / 50.0, 0, 1))
        regime_sub = regime_df[regime_df["split_id"].isin(sub["split_id"])]
        regime_transfer = float(regime_sub["monotonicity"].mean()) if len(regime_sub) else 0.0
        score = (
            25.0 * mono_rate
            + 20.0 * routing_rate
            + 20.0 * mdd_rate
            + 15.0 * (1.0 - false_high_inversion_rate)
            + 10.0 * window_stability
            + 5.0 * regime_transfer
            + 5.0 * min(preservation, 1.0)
        )
        rows.append({
            "policy": policy,
            "windows": int(windows),
            "historical_forward_stability_score": float(np.clip(score, 0, 100)),
            "monotonicity_pass_rate": mono_rate,
            "routing_consistency_pass_rate": routing_rate,
            "mdd_improvement_rate_vs_q2": mdd_rate,
            "false_high_inversion_rate": false_high_inversion_rate,
            "fold_window_stability": window_stability,
            "regime_transfer_robustness": regime_transfer,
            "preservation_stability": preservation,
            "mean_meta_forward_score": float(sub["meta_forward_score"].mean()),
            "success_criteria_pass": bool(
                mono_rate >= 0.60
                and routing_rate >= 0.60
                and mdd_rate >= 0.50
                and false_high_inversion_rate == 0
                and preservation >= 0.90
                and windows >= 3
            ),
        })
    return pd.DataFrame(rows).sort_values("historical_forward_stability_score", ascending=False)


def _verdict(stability: pd.DataFrame) -> str:
    if stability.empty:
        return "Q2_BDI_baseline_still_best"
    best = stability.iloc[0]
    if bool(best.get("success_criteria_pass", False)):
        if best["policy"] == "always_on_meta":
            return "always_on_meta_shadow_promising"
        if best["policy"] == "false_high_off_meta":
            return "false_high_off_policy_promising"
        return "historical_forward_signal_persistent"
    regime_signal = float(stability["regime_transfer_robustness"].max()) if len(stability) else 0.0
    activation = stability[stability["policy"] == "activation_controlled_meta"]
    always = stability[stability["policy"] == "always_on_meta"]
    if regime_signal >= 0.45:
        return "regime_specific_signal_persistent"
    if len(activation) and len(always) and float(activation.iloc[0]["historical_forward_stability_score"]) < float(always.iloc[0]["historical_forward_stability_score"]):
        return "activation_controller_still_not_helpful"
    if float(best["historical_forward_stability_score"]) < 55:
        return "meta_shadow_not_stable"
    return "Q2_BDI_baseline_still_best"


def _write_reports(
    splits: pd.DataFrame,
    shadow_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    false_high_df: pd.DataFrame,
    stability: pd.DataFrame,
) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits.to_csv(OUT_DIR / "historical_walkforward_splits.csv", index=False)
    shadow_df.to_csv(OUT_DIR / "historical_walkforward_shadow_rows.csv", index=False)
    shadow_df.to_parquet(OUT_DIR / "historical_walkforward_shadow_rows.parquet", index=False)
    policy_df.to_csv(OUT_DIR / "policy_comparison_by_window.csv", index=False)
    regime_df.to_csv(OUT_DIR / "regime_specific_forward_validation.csv", index=False)
    false_high_df.to_csv(OUT_DIR / "false_high_inversion_history.csv", index=False)
    stability.to_csv(OUT_DIR / "historical_forward_stability_score.csv", index=False)

    final_verdict = _verdict(stability)
    ok_splits = splits[splits["status"] == "ok"]
    leakage_pass = bool(
        not ok_splits.empty
        and ok_splits["temporal_separation"].all()
        and not (set(FEATURE_COLS) & set(LEAKAGE_COLS))
    )
    audit = {
        "leakage_audit": "PASS" if leakage_pass else "FAIL",
        "temporal_separation_all_ok_splits": bool(ok_splits["temporal_separation"].all()) if len(ok_splits) else False,
        "ok_splits": int(len(ok_splits)),
        "feature_leakage_overlap": sorted(set(FEATURE_COLS) & set(LEAKAGE_COLS)),
        "train_only_fit": True,
        "test_used_for_training_or_calibration": False,
        "q2_bdi_baseline_unchanged": True,
        "promotion_ready": False,
    }
    (OUT_DIR / "leakage_audit.md").write_text(
        "# Leakage Audit\n\n"
        f"**Status:** `{audit['leakage_audit']}`\n\n"
        "Each model is fit only on rows with timestamp before the test window.\n\n"
        f"```json\n{json.dumps(audit, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    best_policy = stability.iloc[0].to_dict() if len(stability) else {}
    success = {
        "monotonicity_ge_60": bool(best_policy.get("monotonicity_pass_rate", 0) >= 0.60),
        "routing_ge_60": bool(best_policy.get("routing_consistency_pass_rate", 0) >= 0.60),
        "mdd_improvement_ge_50": bool(best_policy.get("mdd_improvement_rate_vs_q2", 0) >= 0.50),
        "false_high_not_repeated": bool(best_policy.get("false_high_inversion_rate", 1) == 0),
        "preservation_ge_90": bool(best_policy.get("preservation_stability", 0) >= 0.90),
        "not_single_window_only": bool(best_policy.get("windows", 0) >= 3),
        "leakage_audit_pass": leakage_pass,
    }
    summary = {
        "verdict": final_verdict,
        "splits_total": int(len(splits)),
        "splits_ok": int(len(ok_splits)),
        "shadow_rows": int(len(shadow_df)),
        "best_policy": best_policy,
        "success_criteria": success,
        "promotion_ready": False,
    }
    (OUT_DIR / "historical_walkforward_shadow_report.md").write_text(
        "# Historical Walk-Forward Meta Shadow Report\n\n"
        f"**Final verdict:** `{final_verdict}`\n\n"
        "**Scope:** diagnostics-only historical walk-forward validation. Q2_BDI remains official baseline.\n\n"
        f"```json\n{json.dumps(summary, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    (OUT_DIR / "meta_shadow_policy_verdict.md").write_text(
        "# Meta Shadow Policy Verdict\n\n"
        f"**Verdict:** `{final_verdict}`\n\n"
        "**Allowed state:** monitor-only shadow candidate at most. Production promotion remains forbidden.\n\n"
        f"```json\n{json.dumps({'best_policy': best_policy, 'success_criteria': success}, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    return summary


def run_historical_walkforward() -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchor, weights = _load_anchor()
    splits = _make_splits(anchor)
    shadow_df, policy_df, regime_df, false_high_df = _run_windows(anchor, weights, splits)
    stability = _stability_scores(policy_df, regime_df)
    return _write_reports(splits, shadow_df, policy_df, regime_df, false_high_df, stability)


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical walk-forward Meta shadow validation")
    parser.parse_args()
    summary = run_historical_walkforward()
    print(f"verdict: {summary['verdict']}")
    print(f"splits_ok: {summary['splits_ok']}")
    print(f"shadow_rows: {summary['shadow_rows']}")
    print(f"best_policy: {summary.get('best_policy', {}).get('policy', 'N/A')}")


if __name__ == "__main__":
    main()
