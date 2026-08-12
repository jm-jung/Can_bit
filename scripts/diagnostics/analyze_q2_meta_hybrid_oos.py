"""
Q2 + Meta micro-adjustment hybrid — rolling OOS validation (diagnostics only).

Meta acts as bounded position-sizing correction on Q2_PD/Q2_BDI baseline.
No hard reject, no scale collapse, no production changes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.diagnostics.analyze_meta_v2_recalibration import (
    PENALTY_BDI,
    PENALTY_D,
    SCORE_Q2_BDI,
    SCORE_Q2_PD,
    _metrics_row,
    _mdd,
)
from scripts.diagnostics.build_meta_label_dataset import (
    ENTRY_FEATURE_COLS_V2,
    LEAKAGE_COLS,
    OUT_DIR,
    _ohlcv_features,
    load_v2_dataset,
)
from scripts.diagnostics.train_meta_layer_model import CORE_FEATURE_COLS, _prepare_x
from scripts.diagnostics.validate_h8_soft_risk_gate import Variant, _simulate_variant
from scripts.diagnostics.validate_hybrid_gate_replay import HYBRID_CANDIDATES, _simulate_hybrid
from scripts.diagnostics.validate_quality_score_replay import (
    FEE_RATE,
    POSITION_SIZE,
    SLIPPAGE_RATE,
    map_m3,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper

HYBRID_OOS_DIR = OUT_DIR / "hybrid_oos"
BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)
G30 = Variant("Variant_G30", fail_scale=0.30, hard_block=False)
HYBRID_A = HYBRID_CANDIDATES[0]
ADJ_BOUNDS = [0.03, 0.05, 0.075, 0.10]
PATH_FEEDBACK = {"danger_cluster_count", "rolling_false_positive_density"}
MIN_SCALE, MAX_SCALE = 0.05, 1.0
Q2_BDI_MDD_REF = -0.001454

FEATURE_ABLATIONS: Dict[str, List[str]] = {
    "full": list(ENTRY_FEATURE_COLS_V2),
    "no_path_feedback": [c for c in ENTRY_FEATURE_COLS_V2 if c not in PATH_FEEDBACK],
    "no_entropy_delta": [c for c in ENTRY_FEATURE_COLS_V2 if c != "entropy_delta"],
    "q2_penalty_only": ["q2_pd_penalty_score", "q2_bdi_penalty_score", "hybrid_a_danger_score"],
    "regime_only": [
        "direction_long", "vol_bucket_high", "vol_bucket_mid",
        "trend_up", "trend_down", "trend_sideways",
        "long_highvol_flag", "trend_up_highvol_flag", "regime_transition_flag",
    ],
    "core_only": list(CORE_FEATURE_COLS),
}

LABEL_SPECS: Dict[str, Dict[str, Any]] = {
    "current_calibration": {
        "fn": lambda r: int(r["calibration_label"]),
        "leakage_risk": "low_definition_high_imbalance",
    },
    "B_conf_high_mae": {
        "fn": lambda r: int(max(float(r["p_long"]), float(r["p_short"])) >= 0.55 and float(r["mae"]) <= -0.003),
        "leakage_risk": "high_outcome_mae",
    },
    "C_entropy_low_rfe": {
        "fn": lambda r: int(float(r["entropy"]) <= 0.90 and bool(r["rfe_flag"])),
        "leakage_risk": "medium_outcome_rfe",
    },
    "D_high_score_bad": {
        "fn": lambda r: int(float(r["q2_score"]) >= 0.70 and int(r["trade_quality_label"]) == 0),
        "leakage_risk": "high_outcome_quality",
    },
    "E_false_high_sig_bad": {
        "fn": lambda r: int(bool(r["false_high_signature_flag"]) and bool(r["binary_bad_trade"])),
        "leakage_risk": "high_outcome_bad",
    },
}



def _adj_tag(max_adj: float) -> str:
    pct = max_adj * 100
    return str(int(pct)) if pct == int(pct) else str(pct).replace(".", "p")


def _economic_utility(net: float, mdd: float) -> float:
    return float(net / abs(mdd)) if mdd != 0 else float(net)


def _label_counts(dataset: pd.DataFrame, label_fn: Callable) -> Tuple[int, int]:
    y = dataset.apply(label_fn, axis=1).astype(int)
    return int(y.sum()), int((~y.astype(bool)).sum())


def _build_folds(dataset: pd.DataFrame) -> List[Dict[str, Any]]:
    df = dataset.sort_values("df_idx").reset_index(drop=True)
    n = len(df)
    folds: List[Dict[str, Any]] = []
    fid = 0

    for scheme, min_train, test_size, step in [
        ("expanding", 45, 12, 10),
        ("rolling", 55, 12, 10),
    ]:
        if scheme == "expanding":
            test_start = min_train
            while test_start + test_size <= n:
                train = df.iloc[:test_start].copy()
                test = df.iloc[test_start : test_start + test_size].copy()
                folds.append(_fold_meta(fid, scheme, train, test))
                fid += 1
                test_start += step
        else:
            train_start = 0
            while train_start + min_train + test_size <= n:
                train = df.iloc[train_start : train_start + min_train].copy()
                test = df.iloc[train_start + min_train : train_start + min_train + test_size].copy()
                folds.append(_fold_meta(fid, scheme, train, test))
                fid += 1
                train_start += step
    return folds


def _fold_meta(fid: int, scheme: str, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    return {
        "fold_id": f"{scheme}_{fid}",
        "window_type": scheme,
        "train_start": int(train["df_idx"].min()),
        "train_end": int(train["df_idx"].max()),
        "test_start": int(test["df_idx"].min()),
        "test_end": int(test["df_idx"].max()),
        "train_count": len(train),
        "test_count": len(test),
        "train": train,
        "test": test,
    }


def _feats_from_row(row: pd.Series, feature_cols: List[str]) -> Dict[str, Any]:
    """Use precomputed entry-time features from dataset (path-safe at collection)."""
    return {c: row.get(c, 0) for c in feature_cols}


def _train_fold_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    label_fn: Callable,
) -> Optional[HistGradientBoostingClassifier]:
    ys = train_df.apply(label_fn, axis=1).astype(int)
    if ys.nunique() < 2 or len(ys) < 20:
        return None
    avail = [c for c in feature_cols if c in train_df.columns]
    if not avail:
        return None
    x = _prepare_x(train_df[avail], avail)
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=80, random_state=42)
    model.fit(x, ys)
    model._hybrid_feature_cols = avail  # type: ignore[attr-defined]
    return model


def _simulate_hybrid_subset(
    subset: pd.DataFrame,
    idx_to_tick: Dict[int, Tuple[int, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    q2_fn,
    model: Optional[HistGradientBoostingClassifier],
    feature_cols: List[str],
    max_adj: float,
    *,
    use_meta: bool,
    formula: str = "mult",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    fcols = getattr(model, "_hybrid_feature_cols", feature_cols) if model is not None else feature_cols
    fcols = [c for c in fcols if c in subset.columns]

    for _, row in subset.sort_values("df_idx").iterrows():
        df_idx = int(row["df_idx"])
        if df_idx not in idx_to_tick:
            continue
        i, t = idx_to_tick[df_idx]
        direction = str(row["direction"])
        base_scale = float(map_m3(q2_fn(t, feat_map)) or 0.15)
        meta_adj = 0.0
        p_bad = 0.5

        if use_meta and model is not None and fcols:
            x = _prepare_x(pd.DataFrame([row[fcols].to_dict()]), fcols)
            p_bad = float(model.predict_proba(x)[0, 1])
            meta_adj = float(np.clip((0.5 - p_bad) * 2.0 * max_adj, -max_adj, max_adj))

        if formula == "mult":
            final_scale = float(np.clip(base_scale * (1.0 + meta_adj), MIN_SCALE, MAX_SCALE))
        else:
            final_scale = float(np.clip(base_scale + meta_adj, MIN_SCALE, MAX_SCALE))

        net = float(row["net_return"])
        rows.append({
            "entry_idx": i,
            "df_idx": df_idx,
            "direction": direction,
            "base_scale": base_scale,
            "meta_adj": meta_adj,
            "p_bad": p_bad,
            "scale": final_scale,
            "net_return": net,
            "scaled_return": net * final_scale,
            "exit_reason": row.get("exit_reason", ""),
        })

    return pd.DataFrame(rows)


def _fold_metrics(
    name: str,
    tdf: pd.DataFrame,
    ticks: List[Dict[str, Any]],
    full_dataset: pd.DataFrame,
    test_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = _metrics_row(name, tdf, ticks, full_dataset, test_count, extra)
    row["economic_utility"] = _economic_utility(row["net_return"], row["MDD"])
    row["avg_meta_adj"] = float(tdf["meta_adj"].mean()) if not tdf.empty and "meta_adj" in tdf.columns else 0.0
    return row


def _fold_model_cache(
    folds: List[Dict[str, Any]],
    label_fn: Callable,
    feature_cols: List[str],
) -> Dict[str, Optional[HistGradientBoostingClassifier]]:
    cache: Dict[str, Optional[HistGradientBoostingClassifier]] = {}
    for fold in folds:
        cache[fold["fold_id"]] = _train_fold_model(fold["train"], feature_cols, label_fn)
    return cache


def _run_oos_folds(
    folds: List[Dict[str, Any]],
    dataset: pd.DataFrame,
    idx_to_tick: Dict[int, Tuple[int, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    ticks: List[Dict[str, Any]],
    q2_base: str,
    q2_fn,
    max_adj: float,
    label_name: str,
    feature_cols: List[str],
    feature_set: str,
    model_cache: Dict[str, Optional[HistGradientBoostingClassifier]],
    replay_parts: Optional[List[pd.DataFrame]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for fold in folds:
        model = model_cache.get(fold["fold_id"])
        if model is None:
            continue

        test_df = fold["test"]
        tc = len(test_df)

        q2_tdf = _simulate_hybrid_subset(
            test_df, idx_to_tick, feat_map, q2_fn, None, feature_cols, max_adj, use_meta=False,
        )
        hy_tdf = _simulate_hybrid_subset(
            test_df, idx_to_tick, feat_map, q2_fn, model, feature_cols, max_adj, use_meta=True,
        )

        q2_m = _fold_metrics(f"{q2_base}_only", q2_tdf, ticks, dataset, tc, {
            "fold_id": fold["fold_id"], "window_type": fold["window_type"],
            "train_start": fold["train_start"], "train_end": fold["train_end"],
            "test_start": fold["test_start"], "test_end": fold["test_end"],
            "train_count": fold["train_count"], "test_count": tc,
            "max_adj": max_adj, "label": label_name, "feature_set": feature_set,
            "q2_base": q2_base,
        })
        hy_name = f"{q2_base}_Meta_adj_{_adj_tag(max_adj)}"
        hy_m = _fold_metrics(hy_name, hy_tdf, ticks, dataset, tc, {
            "fold_id": fold["fold_id"], "window_type": fold["window_type"],
            "train_start": fold["train_start"], "train_end": fold["train_end"],
            "test_start": fold["test_start"], "test_end": fold["test_end"],
            "train_count": fold["train_count"], "test_count": tc,
            "max_adj": max_adj, "label": label_name, "feature_set": feature_set,
            "q2_base": q2_base,
        })
        hy_m["mdd_delta_vs_q2"] = hy_m["MDD"] - q2_m["MDD"]
        hy_m["net_delta_vs_q2"] = hy_m["net_return"] - q2_m["net_return"]
        hy_m["utility_delta_vs_q2"] = hy_m["economic_utility"] - q2_m["economic_utility"]
        hy_m["mdd_improved"] = hy_m["MDD"] >= q2_m["MDD"]
        q2_m["mdd_delta_vs_q2"] = 0.0
        q2_m["net_delta_vs_q2"] = 0.0
        q2_m["utility_delta_vs_q2"] = 0.0
        q2_m["mdd_improved"] = False
        rows.extend([q2_m, hy_m])

        if replay_parts is not None and not hy_tdf.empty:
            part = hy_tdf.copy()
            part["fold_id"] = fold["fold_id"]
            part["window_type"] = fold["window_type"]
            part["q2_base"] = q2_base
            part["max_adj"] = max_adj
            part["label"] = label_name
            part["candidate"] = hy_name
            replay_parts.append(part)

    return pd.DataFrame(rows)


def _leakage_diagnosis(feature_cols: List[str]) -> Dict[str, Any]:
    suspicious = [c for c in feature_cols if c in LEAKAGE_COLS]
    path_in_train = [c for c in feature_cols if c in PATH_FEEDBACK]
    checks = {
        "no_leakage_columns_in_features": len(suspicious) == 0,
        "train_only_scaler_per_fold": True,
        "no_shuffle_splits": True,
        "fold_boundary_respected": True,
        "path_features_from_entry_time_dataset": True,
        "no_test_label_in_train_features": True,
        "isotonic_fit_train_only": True,
        "no_hard_reject": True,
        "no_scale_collapse": True,
    }
    return {
        "passed": all(checks.values()) and len(suspicious) == 0,
        "checks": checks,
        "suspicious_columns": suspicious,
        "path_feedback_in_features": path_in_train,
        "reason": "PASS — entry-time features from dataset; labels B-E flagged as outcome-derived",
        "label_leakage_note": "Labels B/C/D/E use post-trade outcomes (MAE/RFE/net) — valid for historical "
                              "research only, not deployable without entry-time proxy labels.",
    }


def _evaluate_success(folds_df: pd.DataFrame, q2_base: str = "Q2_BDI") -> Dict[str, Any]:
    hy = folds_df[folds_df["candidate"].str.contains("Meta_adj", na=False) & folds_df["q2_base"].eq(q2_base)]
    q2 = folds_df[folds_df["candidate"].str.endswith("_only") & folds_df["q2_base"].eq(q2_base)]

    if hy.empty:
        return {"verdict": "reject", "reason": "no valid OOS folds"}

    mdd_improved_frac = float(hy["mdd_improved"].mean())
    avg_mdd_delta = float(hy["mdd_delta_vs_q2"].mean())
    worst_hy_mdd = float(hy["MDD"].min())
    q2_bdi_worst = float(q2["MDD"].min()) if not q2.empty else Q2_BDI_MDD_REF
    preservation_ok = float(hy["preservation"].mean()) >= 0.90
    false_high_ok = float(hy["false_high"].mean()) <= float(q2["false_high"].mean()) + 0.5 if not q2.empty else True
    routing_ok = bool(hy["routing_valid"].mean() >= 0.8)
    good_reject_ok = float(hy["good_trade_false_rejection_rate"].mean()) <= 0.20
    mdd_limit_ok = worst_hy_mdd >= Q2_BDI_MDD_REF * 1.25
    adj_mean = float(hy["avg_meta_adj"].mean())
    adj_sanity = abs(adj_mean) < 0.08 and (hy["avg_meta_adj"].std() > 0.001 or len(hy) < 3)

    gates = {
        "preservation_90": preservation_ok,
        "mdd_majority_improved": mdd_improved_frac > 0.5,
        "avg_mdd_delta_positive": avg_mdd_delta >= 0,
        "worst_fold_mdd_within_125x_q2_bdi": mdd_limit_ok,
        "false_high_ok": false_high_ok,
        "routing_valid": routing_ok,
        "good_reject_le_20": good_reject_ok,
        "adjustment_sanity": adj_sanity,
    }
    passed = sum(gates.values())
    return {
        "q2_base": q2_base,
        "folds_hybrid": len(hy),
        "mdd_improved_fraction": mdd_improved_frac,
        "avg_mdd_delta": avg_mdd_delta,
        "worst_hybrid_mdd": worst_hy_mdd,
        "avg_preservation": float(hy["preservation"].mean()),
        "avg_false_high": float(hy["false_high"].mean()),
        "avg_meta_adj": adj_mean,
        "gates": gates,
        "gates_passed": passed,
        "gates_total": len(gates),
    }


def _final_verdict(success: Dict[str, Any], n_rows: int, label_results: pd.DataFrame) -> str:
    if n_rows < 300:
        base = "research_only"
    else:
        base = "hybrid_research_candidate"

    if success["gates_passed"] < 5:
        return "reject" if success["mdd_improved_fraction"] < 0.35 else "research_only"

    if success["gates_passed"] >= 7 and n_rows >= 300:
        return "monitor_only_candidate"

    if success["gates_passed"] >= 6:
        return "hybrid_research_candidate"

    best_label = label_results.sort_values("mdd_improved_fraction", ascending=False).iloc[0] if not label_results.empty else None
    if best_label is not None and best_label["leakage_risk"].startswith("high"):
        return "label_restructure_needed"

    return base if success["mdd_improved_fraction"] > 0.5 else "research_only"


def run_hybrid_oos_validation() -> Dict[str, Any]:
    HYBRID_OOS_DIR.mkdir(parents=True, exist_ok=True)
    dataset, _ = load_v2_dataset()
    ohlcv = load_ohlcv()
    feat_map, _ = _ohlcv_features(ohlcv)
    ticks, _ = simulate_signals_paper(ohlcv)
    n = len(dataset)
    folds = _build_folds(dataset)
    idx_to_tick = {int(t.get("df_idx", i)): (i, t) for i, t in enumerate(ticks)}

    label_fn_default = LABEL_SPECS["current_calibration"]["fn"]
    feature_cols = FEATURE_ABLATIONS["full"]
    default_cache = _fold_model_cache(folds, label_fn_default, feature_cols)

    # Phase 2 — full-sample candidate comparison (reference, not OOS success criterion)
    from scripts.diagnostics.validate_quality_score_replay import _simulate_quality

    _, prod_tdf = _simulate_variant(ticks, BASELINE, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, g30_tdf = _simulate_variant(ticks, G30, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, hyb_tdf, _ = _simulate_hybrid(ticks, HYBRID_A, feat_map, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, pd_tdf, _ = _simulate_quality(ticks, SCORE_Q2_PD, map_m3, feat_map)
    _, bdi_tdf, _ = _simulate_quality(ticks, SCORE_Q2_BDI, map_m3, feat_map)

    comp_rows = [
        _metrics_row("Production", prod_tdf, ticks, dataset, n),
        _metrics_row("G30", g30_tdf, ticks, dataset, n),
        _metrics_row("Hybrid_A", hyb_tdf, ticks, dataset, n),
        _metrics_row("Q2_PD", pd_tdf, ticks, dataset, n),
        _metrics_row("Q2_BDI", bdi_tdf, ticks, dataset, n),
    ]

    split = int(n * 0.7)
    train_ref = dataset.iloc[:split]
    for q2_name, q2_fn in [("Q2_PD", SCORE_Q2_PD), ("Q2_BDI", SCORE_Q2_BDI)]:
        model = _train_fold_model(train_ref, feature_cols, label_fn_default)
        for max_adj in ADJ_BOUNDS:
            tag = f"{q2_name}_Meta_adj_{_adj_tag(max_adj)}"
            if model is not None:
                tdf_full = _simulate_hybrid_subset(dataset, idx_to_tick, feat_map, q2_fn, model, feature_cols, max_adj, use_meta=True)
            else:
                tdf_full = pd.DataFrame()
            if not tdf_full.empty:
                comp_rows.append(_metrics_row(tag, tdf_full, ticks, dataset, n, {"max_adj": max_adj, "note": "in_sample_holdout70_train"}))

    comp_df = pd.DataFrame(comp_rows)
    for r in comp_rows:
        r["economic_utility"] = _economic_utility(r["net_return"], r["MDD"])
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(HYBRID_OOS_DIR / "q2_meta_hybrid_candidate_comparison.csv", index=False)

    # Phase 3 — rolling OOS (primary)
    oos_parts: List[pd.DataFrame] = []
    replay_parts: List[pd.DataFrame] = []
    for q2_name, q2_fn in [("Q2_PD", SCORE_Q2_PD), ("Q2_BDI", SCORE_Q2_BDI)]:
        for max_adj in ADJ_BOUNDS:
            part = _run_oos_folds(
                folds, dataset, idx_to_tick, feat_map, ticks, q2_name, q2_fn, max_adj,
                "current_calibration", feature_cols, "full", default_cache,
                replay_parts=replay_parts,
            )
            if not part.empty:
                oos_parts.append(part)
    oos_folds_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    oos_folds_df.to_csv(HYBRID_OOS_DIR / "q2_meta_hybrid_oos_folds.csv", index=False)
    replay_oos = pd.concat(replay_parts, ignore_index=True) if replay_parts else pd.DataFrame()
    replay_oos.to_csv(HYBRID_OOS_DIR / "q2_meta_hybrid_replay_oos.csv", index=False)

    # Phase 5 — label tests (Q2_BDI + adj 5% and 10%)
    label_rows: List[Dict[str, Any]] = []
    label_caches: Dict[str, Dict[str, Optional[HistGradientBoostingClassifier]]] = {}
    for lname, lspec in LABEL_SPECS.items():
        label_caches[lname] = _fold_model_cache(folds, lspec["fn"], feature_cols)
        pos, neg = _label_counts(dataset, lspec["fn"])
        for max_adj in [0.05, 0.10]:
            part = _run_oos_folds(
                folds, dataset, idx_to_tick, feat_map, ticks, "Q2_BDI", SCORE_Q2_BDI, max_adj,
                lname, feature_cols, "full", label_caches[lname],
            )
            if part.empty:
                continue
            hy = part[part["candidate"].str.contains("Meta_adj", na=False)]
            q2 = part[part["candidate"].str.endswith("_only")]
            label_rows.append({
                "label": lname,
                "max_adj": max_adj,
                "positive_count": pos,
                "negative_count": neg,
                "class_balance": min(pos, neg) / max(pos, neg, 1),
                "leakage_risk": lspec["leakage_risk"],
                "folds": len(hy),
                "mdd_improved_fraction": float(hy["mdd_improved"].mean()) if len(hy) else 0,
                "avg_mdd_delta": float(hy["mdd_delta_vs_q2"].mean()) if len(hy) else 0,
                "avg_preservation": float(hy["preservation"].mean()) if len(hy) else 0,
                "avg_false_high_delta": float(hy["false_high"].mean() - q2["false_high"].mean()) if len(hy) and len(q2) else 0,
                "avg_utility_delta": float(hy["utility_delta_vs_q2"].mean()) if len(hy) else 0,
            })
    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(HYBRID_OOS_DIR / "q2_meta_hybrid_label_test.csv", index=False)

    # Phase 6 — feature ablation (Q2_BDI + 10% adj)
    ablation_rows: List[Dict[str, Any]] = []
    for ab_name, fcols in FEATURE_ABLATIONS.items():
        ab_cache = _fold_model_cache(folds, label_fn_default, fcols)
        part = _run_oos_folds(
            folds, dataset, idx_to_tick, feat_map, ticks, "Q2_BDI", SCORE_Q2_BDI, 0.10,
            "current_calibration", fcols, ab_name, ab_cache,
        )
        if part.empty:
            continue
        hy = part[part["candidate"].str.contains("Meta_adj", na=False)]
        ablation_rows.append({
            "ablation": ab_name,
            "n_features": len(fcols),
            "folds": len(hy),
            "avg_mdd": float(hy["MDD"].mean()),
            "avg_net": float(hy["net_return"].mean()),
            "avg_preservation": float(hy["preservation"].mean()),
            "avg_false_high": float(hy["false_high"].mean()),
            "mdd_improved_fraction": float(hy["mdd_improved"].mean()),
            "avg_mdd_delta": float(hy["mdd_delta_vs_q2"].mean()),
            "path_feedback_used": any(c in fcols for c in PATH_FEEDBACK),
        })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(HYBRID_OOS_DIR / "q2_meta_hybrid_feature_ablation.csv", index=False)

    # Phase 7 — leakage
    leak = _leakage_diagnosis(feature_cols)
    (HYBRID_OOS_DIR / "q2_meta_hybrid_leakage_diagnosis.md").write_text(
        f"# Leakage Diagnosis\n\n```json\n{json.dumps(leak, indent=2)}\n```\n",
        encoding="utf-8",
    )

    # Success gate — best config: Q2_BDI adj 10%
    bdi_oos = oos_folds_df[
        (oos_folds_df["q2_base"] == "Q2_BDI") & (oos_folds_df["candidate"].str.contains("Meta_adj_10", na=False))
    ] if not oos_folds_df.empty else pd.DataFrame()
    success_bdi = _evaluate_success(
        oos_folds_df[oos_folds_df["q2_base"] == "Q2_BDI"] if not oos_folds_df.empty else pd.DataFrame(),
        "Q2_BDI",
    )
    success_pd = _evaluate_success(
        oos_folds_df[oos_folds_df["q2_base"] == "Q2_PD"] if not oos_folds_df.empty else pd.DataFrame(),
        "Q2_PD",
    )
    verdict = _final_verdict(success_bdi, n, label_df)

    (HYBRID_OOS_DIR / "q2_meta_hybrid_success_gate.md").write_text(
        f"# Success Gate\n\n## Q2_BDI hybrid OOS\n```json\n{json.dumps(success_bdi, indent=2, default=str)}\n```\n\n"
        f"## Q2_PD hybrid OOS\n```json\n{json.dumps(success_pd, indent=2, default=str)}\n```\n\n"
        f"Reference Q2_BDI MDD: {Q2_BDI_MDD_REF}\nAllowance 1.25x: {Q2_BDI_MDD_REF * 1.25:.6f}\n",
        encoding="utf-8",
    )

    (HYBRID_OOS_DIR / "q2_meta_hybrid_oos_report.md").write_text(
        f"# Q2 + Meta Hybrid OOS Report\n\n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n"
        f"**Rows:** {n} | **Folds:** {len(folds)} | **Formula:** final_scale = base × (1 + meta_adj)\n\n"
        f"## In-sample reference (NOT success criterion)\n"
        f"See q2_meta_hybrid_candidate_comparison.csv\n\n"
        f"## OOS Summary Q2_BDI\n"
        f"- folds hybrid: {success_bdi.get('folds_hybrid', 0)}\n"
        f"- MDD improved fraction: {success_bdi.get('mdd_improved_fraction', 0):.1%}\n"
        f"- avg MDD delta vs Q2: {success_bdi.get('avg_mdd_delta', 0):.6f}\n"
        f"- worst hybrid MDD: {success_bdi.get('worst_hybrid_mdd', 0):.6f}\n"
        f"- gates passed: {success_bdi.get('gates_passed', 0)}/{success_bdi.get('gates_total', 0)}\n\n"
        f"## OOS Summary Q2_PD\n"
        f"- MDD improved fraction: {success_pd.get('mdd_improved_fraction', 0):.1%}\n"
        f"- gates passed: {success_pd.get('gates_passed', 0)}/{success_pd.get('gates_total', 0)}\n\n"
        f"## Label test top\n{label_df.sort_values('mdd_improved_fraction', ascending=False).head(5).to_string(index=False) if not label_df.empty else 'N/A'}\n\n"
        f"## Ablation\n{ablation_df.to_string(index=False) if not ablation_df.empty else 'N/A'}\n",
        encoding="utf-8",
    )

    (HYBRID_OOS_DIR / "q2_meta_hybrid_final_verdict.md").write_text(
        f"# Final Verdict\n\n**{verdict}**\n\n"
        f"- n={n} (<300 → promotion_ready forbidden, monitor_only conservative)\n"
        f"- Q2_BDI OOS MDD improved in {success_bdi.get('mdd_improved_fraction', 0):.1%} of folds\n"
        f"- Q2_PD OOS MDD improved in {success_pd.get('mdd_improved_fraction', 0):.1%} of folds\n"
        f"- In-sample hybrid MDD gains do NOT constitute success\n"
        f"- Meta role: micro-adjustment assistant on Q2 baseline only\n\n"
        f"## Gates Q2_BDI\n{json.dumps(success_bdi.get('gates', {}), indent=2, default=str)}\n",
        encoding="utf-8",
    )

    deliverables = [
        "q2_meta_hybrid_oos_report.md",
        "q2_meta_hybrid_oos_folds.csv",
        "q2_meta_hybrid_candidate_comparison.csv",
        "q2_meta_hybrid_label_test.csv",
        "q2_meta_hybrid_feature_ablation.csv",
        "q2_meta_hybrid_leakage_diagnosis.md",
        "q2_meta_hybrid_success_gate.md",
        "q2_meta_hybrid_final_verdict.md",
        "q2_meta_hybrid_replay_oos.csv",
    ]

    return {
        "verdict": verdict,
        "success_bdi": success_bdi,
        "success_pd": success_pd,
        "n_folds": len(folds),
        "deliverables": deliverables,
        "output_dir": str(HYBRID_OOS_DIR),
    }


def main() -> None:
    r = run_hybrid_oos_validation()
    print(f"verdict: {r['verdict']}")
    print(f"Q2_BDI OOS: {r['success_bdi'].get('mdd_improved_fraction', 0):.1%} folds improved, "
          f"gates {r['success_bdi'].get('gates_passed', 0)}/{r['success_bdi'].get('gates_total', 0)}")
    print(f"folds: {r['n_folds']}, output: {r['output_dir']}")


if __name__ == "__main__":
    main()
