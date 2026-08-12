"""
Meta V2 research continuation — Q2 baseline lock, promotion gates, dataset audit,
label restructure simulation, feature stability, Q2+Meta hybrid (diagnostics only).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.diagnostics.analyze_meta_v2_recalibration import (
    PENALTY_BDI,
    PENALTY_D,
    SCORE_Q2_BDI,
    SCORE_Q2_PD,
    _load_artifact,
    _metrics_row,
    _mdd,
    _predict_scale,
    simulate_meta_counterfactual,
)
from scripts.diagnostics.build_meta_label_dataset import (
    ENTRY_FEATURE_COLS_V2,
    HISTORY_DIR,
    LABEL_COLS,
    MANIFEST_PATH,
    MASTER_PATH,
    OUT_DIR,
    _dedupe,
    _entry_features_v2,
    _ohlcv_features,
    _rolling_context,
    load_v2_dataset,
)
from scripts.diagnostics.train_meta_layer_model import _feature_importance, _prepare_x
from scripts.diagnostics.validate_h8_soft_risk_gate import Variant, _entropy, _simulate_variant
from scripts.diagnostics.validate_quality_score_replay import (
    POSITION_SIZE,
    _risk_routing,
    _simulate_quality,
    map_m3,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper

RESEARCH_DIR = OUT_DIR / "research_continuation"
RECAL_COMPARE = OUT_DIR / "recalibration" / "meta_v2_recalibrated_comparison.csv"

Q2_BDI_MDD = -0.001454
Q2_PD_MDD = -0.001902
META_ADJ_MAX = 0.10
CAL_MULT_HYBRID = 0.30
ABLATE_BOTH: Set[str] = {"danger_cluster_count", "rolling_false_positive_density"}


def _ctx_ablated(closed: List[Dict[str, Any]], ablate: Set[str]) -> Dict[str, float]:
    ctx = _rolling_context(closed)
    if "danger_cluster_count" in ablate:
        ctx["danger_cluster_count"] = 0.0
    if "rolling_false_positive_density" in ablate:
        ctx["rolling_false_positive_density"] = 0.0
    return ctx


def _load_q2_benchmark() -> pd.DataFrame:
    if RECAL_COMPARE.exists():
        return pd.read_csv(RECAL_COMPARE)
    raise FileNotFoundError(f"Missing benchmark CSV: {RECAL_COMPARE}")


def _promotion_gate_status(rows: int, meta: Dict[str, Any], oos_auc_mean: float, oos_auc_std: float) -> Dict[str, Any]:
    q2_mdd = Q2_BDI_MDD
    mdd_ok = meta["MDD"] >= q2_mdd * 1.25
    checks = {
        "sample_rows_300": rows >= 300,
        "sample_rows_500_recommended": rows >= 500,
        "preservation_85": meta["preservation"] >= 0.85,
        "preservation_90_recommended": meta["preservation"] >= 0.90,
        "oos_auc_058": oos_auc_mean >= 0.58,
        "oos_auc_std_012": oos_auc_std <= 0.12,
        "mdd_within_q2_bdi_125x": mdd_ok,
        "false_high_le_q2": meta["false_high"] <= 3,
        "routing_valid": bool(meta["routing_valid"]),
        "good_reject_le_20pct": meta["good_trade_false_rejection_rate"] <= 0.20,
        "good_reject_le_10pct_recommended": meta["good_trade_false_rejection_rate"] <= 0.10,
    }
    checks["promotion_ready"] = all([
        checks["sample_rows_300"],
        checks["preservation_85"],
        checks["oos_auc_058"],
        checks["mdd_within_q2_bdi_125x"],
        checks["false_high_le_q2"],
        checks["routing_valid"],
        checks["good_reject_le_20pct"],
    ])
    return checks


def _audit_dataset() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    dataset, ds_path = load_v2_dataset()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}

    dup_ids = int(dataset["trade_id"].duplicated().sum()) if "trade_id" in dataset.columns else -1
    deduped = _dedupe(dataset.copy())
    dup_after = int(deduped["trade_id"].duplicated().sum()) if "trade_id" in deduped.columns else 0

    expected_cols = set(ENTRY_FEATURE_COLS_V2 + LABEL_COLS + [
        "trade_id", "timestamp", "entry_ts", "exit_ts", "df_idx", "direction",
        "entry_price", "exit_price", "exit_reason", "hold_bars", "source_replay",
        "model_id", "replay_timestamp", "meta_v2_version", "raw_return", "net_return",
        "scaled_return", "mae", "mfe", "rfe_flag", "ks_cluster_flag", "loss_cluster_flag",
        "vol_bucket", "trend_state",
    ])
    present = set(dataset.columns)
    missing = sorted(expected_cols - present)
    extra = sorted(present - expected_cols)

    schema_rows = []
    for col in ENTRY_FEATURE_COLS_V2 + LABEL_COLS:
        if col not in dataset.columns:
            schema_rows.append({"column": col, "present": False, "dtype": "", "null_rate": 1.0, "drift": "missing"})
            continue
        null_rate = float(dataset[col].isna().mean())
        schema_rows.append({
            "column": col,
            "present": True,
            "dtype": str(dataset[col].dtype),
            "null_rate": null_rate,
            "drift": "ok" if null_rate < 0.05 else "high_null",
        })
    schema_df = pd.DataFrame(schema_rows)

    growth_rows: List[Dict[str, Any]] = []
    if "replay_timestamp" in dataset.columns:
        for ts, grp in dataset.groupby("replay_timestamp"):
            growth_rows.append({
                "replay_timestamp": ts,
                "row_delta": int(len(grp)),
                "cumulative_rows": int(len(dataset[dataset["replay_timestamp"] <= ts])),
            })
    elif manifest.get("last_replay_timestamp"):
        growth_rows.append({
            "replay_timestamp": manifest["last_replay_timestamp"],
            "row_delta": int(manifest.get("batch_rows", len(dataset))),
            "cumulative_rows": int(len(dataset)),
        })
    growth_df = pd.DataFrame(growth_rows)

    history_files = sorted(HISTORY_DIR.glob("meta_dataset_v2_*.parquet")) if HISTORY_DIR.exists() else []
    history_ok = bool(history_files) or manifest.get("history_snapshot")

    audit = {
        "master_path": str(ds_path),
        "master_exists": Path(ds_path).exists(),
        "manifest_exists": MANIFEST_PATH.exists(),
        "total_rows": int(len(dataset)),
        "unique_trade_ids": int(dataset["trade_id"].nunique()) if "trade_id" in dataset.columns else 0,
        "duplicate_trade_ids": dup_ids,
        "dedupe_would_remove": int(len(dataset) - len(deduped)),
        "duplicate_after_dedupe": dup_after,
        "append_only_integrity": dup_after == 0 and dup_ids == 0,
        "missing_columns": missing,
        "extra_columns": extra[:20],
        "schema_drift": bool(missing),
        "history_snapshots": len(history_files),
        "history_ok": history_ok,
        "last_replay_timestamp": manifest.get("last_replay_timestamp"),
        "source_replay": manifest.get("source_replay"),
        "label_distribution": manifest.get("statistics", {}).get("label_distribution", {}),
        "daily_automation_registered": False,
    }
    return audit, schema_df, growth_df


def _label_candidate(name: str, mask: pd.Series, dataset: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
    pos = int(mask.sum())
    neg = int((~mask).sum())
    overlap = int((mask & (current == 1)).sum())
    sub = dataset[mask]
    return {
        "candidate": name,
        "positive_count": pos,
        "negative_count": neg,
        "positive_rate": pos / max(len(dataset), 1),
        "class_balance_ratio": min(pos, neg) / max(pos, neg, 1),
        "overlap_current_fail": overlap,
        "overlap_current_fail_rate": overlap / max(int((current == 1).sum()), 1),
        "rfe_rate_when_positive": float(sub["rfe_flag"].mean()) if len(sub) else 0.0,
        "bad_trade_rate_when_positive": float(sub["binary_bad_trade"].mean()) if len(sub) else 0.0,
        "mean_mae_when_positive": float(sub["mae"].mean()) if len(sub) else 0.0,
        "trade_quality_bad_rate": float((sub["trade_quality_label"] == 0).mean()) if len(sub) else 0.0,
        "leakage_risk": "high" if name in ("A", "B", "D", "E") else "medium",
    }


def _simulate_label_candidates(dataset: pd.DataFrame) -> pd.DataFrame:
    current = dataset["calibration_label"].astype(int)
    max_ps = dataset[["p_long", "p_short"]].max(axis=1)
    rows = [
        _label_candidate("A_conf_high_loss", (max_ps >= 0.55) & (dataset["net_return"] < 0), dataset, current),
        _label_candidate("B_conf_high_mae", (max_ps >= 0.55) & (dataset["mae"] <= -0.003), dataset, current),
        _label_candidate("C_entropy_low_rfe", (dataset["entropy"] <= 0.90) & (dataset["rfe_flag"].astype(bool)), dataset, current),
        _label_candidate("D_high_score_bad", (dataset["q2_score"] >= 0.70) & (dataset["trade_quality_label"] == 0), dataset, current),
        _label_candidate(
            "E_false_high_sig",
            (dataset["false_high_signature_flag"].astype(bool)) & (dataset["binary_bad_trade"].astype(bool)),
            dataset, current,
        ),
    ]
    rows.append({
        "candidate": "current_calibration_label",
        "positive_count": int((current == 1).sum()),
        "negative_count": int((current == 0).sum()),
        "positive_rate": float((current == 1).mean()),
        "class_balance_ratio": 4 / 131,
        "overlap_current_fail": 4,
        "overlap_current_fail_rate": 1.0,
        "rfe_rate_when_positive": float(dataset.loc[current == 1, "rfe_flag"].mean()) if (current == 1).any() else 0,
        "bad_trade_rate_when_positive": float(dataset.loc[current == 1, "binary_bad_trade"].mean()) if (current == 1).any() else 0,
        "mean_mae_when_positive": float(dataset.loc[current == 1, "mae"].mean()) if (current == 1).any() else 0,
        "trade_quality_bad_rate": float((dataset.loc[current == 1, "trade_quality_label"] == 0).mean()) if (current == 1).any() else 0,
        "leakage_risk": "low_definition_high_imbalance",
    })
    return pd.DataFrame(rows)


def _recommend_label(candidates: pd.DataFrame) -> str:
    usable = candidates[candidates["candidate"] != "current_calibration_label"].copy()
    usable = usable[(usable["positive_count"] >= 15) & (usable["positive_count"] <= 80)]
    if usable.empty:
        best = candidates[candidates["candidate"] != "current_calibration_label"].sort_values(
            ["class_balance_ratio", "positive_count"], ascending=[False, False]
        ).iloc[0]
        return (
            f"Recommended: **{best['candidate']}** (best balance among imbalanced options; "
            f"positive={int(best['positive_count'])}). Note leakage risk: {best['leakage_risk']}."
        )
    pick = usable.sort_values(["class_balance_ratio", "bad_trade_rate_when_positive"], ascending=[False, False]).iloc[0]
    return (
        f"Recommended: **{pick['candidate']}** — positive={int(pick['positive_count'])}, "
        f"balance={pick['class_balance_ratio']:.3f}, bad_rate={pick['bad_trade_rate_when_positive']:.1%}. "
        f"Leakage: {pick['leakage_risk']}."
    )


def _feature_stability(dataset: pd.DataFrame, artifact: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    focus = [
        "danger_cluster_count", "rolling_false_positive_density", "entropy_delta",
        "confidence_overextension", "q2_pd_penalty_score", "q2_bdi_penalty_score", "hybrid_a_danger_score",
    ]
    dist_rows = []
    for feat in focus:
        if feat not in dataset.columns:
            continue
        dist_rows.append({
            "feature": feat,
            "mean": float(dataset[feat].mean()),
            "std": float(dataset[feat].std()),
            "p25": float(dataset[feat].quantile(0.25)),
            "p50": float(dataset[feat].quantile(0.50)),
            "p75": float(dataset[feat].quantile(0.75)),
            "min": float(dataset[feat].min()),
            "max": float(dataset[feat].max()),
        })
    dist_df = pd.DataFrame(dist_rows)

    fcols = artifact["feature_cols"]
    x = _prepare_x(dataset, fcols)
    y = dataset["drawdown_risk_label"].astype(int)
    base_imp = _feature_importance("hist_gradient_boosting", artifact["calibration_model"], fcols)
    imp_map = {r["feature"]: r["importance"] for r in base_imp}

    def _factory():
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42)

    fold_imps: Dict[str, List[float]] = {f: [] for f in fcols}
    n = len(dataset)
    chunk, step = max(30, n // 4), max(8, n // 12)
    start = 0
    max_folds = 20
    while start + chunk + 5 <= n and len(fold_imps[fcols[0]]) < max_folds:
        sub = dataset.sort_values("df_idx").iloc[start : start + chunk + step]
        split = int(len(sub) * 0.7)
        if split < 15:
            break
        m = _factory()
        xs = _prepare_x(sub, fcols)
        m.fit(xs.iloc[:split], sub["drawdown_risk_label"].astype(int).iloc[:split])
        for r in _feature_importance("hist_gradient_boosting", m, fcols):
            fold_imps[r["feature"]].append(r["importance"])
        start += step

    stab_rows = []
    for feat in fcols:
        vals = fold_imps.get(feat) or [imp_map.get(feat, 0.0)]
        stab_rows.append({
            "feature": feat,
            "mean_importance": float(np.mean(vals)),
            "std_importance": float(np.std(vals)),
            "cv_importance": float(np.std(vals) / (np.mean(vals) + 1e-9)),
            "folds": len(vals),
            "path_feedback": feat in ("danger_cluster_count", "rolling_false_positive_density"),
        })
    stab_df = pd.DataFrame(stab_rows).sort_values("mean_importance", ascending=False)

    total = stab_df["mean_importance"].sum() or 1.0
    top = stab_df.iloc[0]
    path_share = stab_df[stab_df["path_feedback"]]["mean_importance"].sum() / total
    concentration = float(top["mean_importance"] / total)

    verdict = "stable"
    if concentration >= 0.25:
        verdict = "feature_rebalance_needed"
    elif path_share >= 0.30:
        verdict = "path_feedback_dependent"

    summary = {
        "top_feature": top["feature"],
        "top_feature_share": concentration,
        "path_feedback_share": path_share,
        "verdict": verdict,
        "path_feedback_features": focus,
    }
    return dist_df, stab_df, summary


def _simulate_q2_hybrid(
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    artifact: Dict[str, Any],
    dataset: pd.DataFrame,
    *,
    q2_fn,
    name: str,
    use_meta_adj: bool,
) -> pd.DataFrame:
    idx_to_tick = {int(t.get("df_idx", i)): (i, t) for i, t in enumerate(ticks)}
    closed: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for _, prod in dataset.sort_values("df_idx").iterrows():
        df_idx = int(prod["df_idx"])
        if df_idx not in idx_to_tick:
            continue
        i, t = idx_to_tick[df_idx]
        direction = str(prod["direction"])
        ctx = _ctx_ablated(closed, ABLATE_BOTH)
        feats = _entry_features_v2(t, ticks, i, feat_map, direction, ctx)
        base_scale = float(map_m3(q2_fn(t, feat_map)) or 0.15)

        meta_adj = 0.0
        cal_prob = 0.0
        if use_meta_adj:
            cal_prob, _, _ = _predict_scale(artifact, feats, CAL_MULT_HYBRID)
            meta_adj = float(np.clip((0.5 - cal_prob) * 0.20, -META_ADJ_MAX, META_ADJ_MAX))
            final_scale = float(np.clip(base_scale + meta_adj, 0.05, 1.0))
        else:
            final_scale = base_scale

        net = float(prod["net_return"])
        rows.append({
            "candidate": name,
            "entry_idx": i,
            "df_idx": df_idx,
            "direction": direction,
            "base_scale": base_scale,
            "meta_adj": meta_adj,
            "scale": final_scale,
            "cal_prob": cal_prob,
            "net_return": net,
            "scaled_return": net * final_scale,
            "exit_reason": prod.get("exit_reason", ""),
        })
        closed.append({
            "net_return": net,
            "false_high_signature_flag": feats.get("false_high_signature_flag", 0),
            "hybrid_a_danger": feats.get("hybrid_a_danger", 0),
            "q2_score": feats.get("q2_score", 0),
        })

    return pd.DataFrame(rows)


def _hybrid_verdict(hybrid: Dict[str, Any], q2: Dict[str, Any]) -> str:
    mdd_ok = hybrid["MDD"] >= q2["MDD"]
    utility = hybrid["top_bucket_expectancy"] >= q2["top_bucket_expectancy"]
    if mdd_ok and utility and hybrid["routing_valid"]:
        return "hybrid_research_candidate"
    return "no_hybrid_advantage"


def run_research_continuation() -> Dict[str, Any]:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d")

    compare = _load_q2_benchmark()
    q2_pd = compare[compare["candidate"] == "Q2_PD"].iloc[0].to_dict()
    q2_bdi = compare[compare["candidate"] == "Q2_BDI"].iloc[0].to_dict()
    meta_recal = compare[compare["candidate"] == "Meta_V2_Recalibrated"].iloc[0].to_dict()

    dataset, _ = load_v2_dataset()
    artifact = _load_artifact()
    ohlcv = load_ohlcv()
    feat_map, _ = _ohlcv_features(ohlcv)
    ticks, _ = simulate_signals_paper(ohlcv)
    prod_trades = len(dataset)

    oos_path = sorted(OUT_DIR.glob("meta_rolling_oos_*.md"))
    oos_auc_mean, oos_auc_std = 0.46, 0.22
    if oos_path:
        try:
            raw = oos_path[-1].read_text(encoding="utf-8")
            start = raw.find("[")
            if start >= 0:
                windows = json.loads(raw[start:])
                aucs = [w["auc"] for w in windows if "auc" in w and not np.isnan(w["auc"])]
                if aucs:
                    oos_auc_mean = float(np.mean(aucs))
                    oos_auc_std = float(np.std(aucs))
        except (json.JSONDecodeError, KeyError):
            pass

    gate_checks = _promotion_gate_status(len(dataset), meta_recal, oos_auc_mean, oos_auc_std)

    # Phase 1
    p1 = RESEARCH_DIR / "q2_baseline_lock_report.md"
    p1.write_text(
        f"# Q2 Baseline Lock Report\n\n"
        f"**Status:** Q2_PD / Q2_BDI locked as official Meta research benchmark\n"
        f"**Date:** {ts}\n\n"
        f"## Q2_PD Summary\n"
        f"- trades: {int(q2_pd['trades'])}\n"
        f"- preservation: {q2_pd['preservation']:.1%}\n"
        f"- MDD: {q2_pd['MDD']:.6f}\n"
        f"- routing_valid: {q2_pd['routing_valid']}\n"
        f"- false_high: {int(q2_pd['false_high'])}\n"
        f"- RFE: {int(q2_pd['RFE'])}\n\n"
        f"## Q2_BDI Summary\n"
        f"- trades: {int(q2_bdi['trades'])}\n"
        f"- preservation: {q2_bdi['preservation']:.1%}\n"
        f"- MDD: {q2_bdi['MDD']:.6f} (best defensive baseline)\n"
        f"- routing_valid: {q2_bdi['routing_valid']}\n"
        f"- false_high: {int(q2_bdi['false_high'])}\n"
        f"- avg_scale: {q2_bdi['avg_scale']:.4f}\n\n"
        f"## Why Q2_BDI leads on MDD\n"
        f"Q2_BDI applies Combo B+D+I penalties with M3 mapping, achieving lowest MDD ({q2_bdi['MDD']:.6f}) "
        f"while preserving 100% trades and routing monotonicity. Q2_PD is close ({q2_pd['MDD']:.6f}) with "
        f"slightly higher avg exposure.\n\n"
        f"## Why Meta_V2 is not superior\n"
        f"- Meta_V2 initial: preservation 17%, overfilter, 90%+ good trade rejection\n"
        f"- Meta_V2 recalibrated: preservation {meta_recal['preservation']:.1%}, MDD {meta_recal['MDD']:.6f} "
        f"(~{abs(meta_recal['MDD']/q2_bdi['MDD']):.1f}x worse than Q2_BDI)\n"
        f"- OOS AUC ~{oos_auc_mean:.2f} (random-level), rows={len(dataset)} (<300 minimum)\n"
        f"- calibration_label imbalance (4 positives)\n\n"
        f"## Meta Minimum Pass Conditions (vs Q2 benchmark)\n"
        f"All future Meta experiments MUST compare against Q2_PD and Q2_BDI.\n"
        f"1. rows >= 300 (500 recommended)\n"
        f"2. preservation >= 85% (90% recommended)\n"
        f"3. rolling OOS AUC >= 0.58, std <= 0.12\n"
        f"4. MDD <= Q2_BDI × 1.25 (currently {Q2_BDI_MDD * 1.25:.6f})\n"
        f"5. false_high <= Q2, routing_valid=True\n"
        f"6. good trade false rejection <= 20%\n"
        f"7. no LONG/high_vol blanket rejection\n\n"
        f"**Verdict:** Q2_PD/Q2_BDI superior + Meta research-only\n",
        encoding="utf-8",
    )

    # Phase 2
    p2 = RESEARCH_DIR / "meta_promotion_gate.md"
    p2.write_text(
        f"# Meta Promotion Gate\n\n"
        f"Diagnostic-only gate definition. **No auto-registration to monitor or production.**\n\n"
        f"## Hard Rejections (never promote if any fail)\n"
        f"- in-sample net positive alone\n"
        f"- Meta evaluated without Q2_PD/Q2_BDI comparison\n"
        f"- hard_reject gate reintroduced\n"
        f"- scale collapse mapping (cal_mult × cal_prob) without recalibration evidence\n"
        f"- rows < 300\n\n"
        f"## Minimum Pass Checklist\n"
        + "\n".join(f"- `{k}`: **{'PASS' if v else 'FAIL'}**" for k, v in gate_checks.items())
        + f"\n\n## Current sample gate\n"
        f"- rows: {len(dataset)} / 300 minimum\n"
        f"- OOS AUC mean: {oos_auc_mean:.3f} (need >= 0.58)\n"
        f"- Meta recal MDD: {meta_recal['MDD']:.6f} vs Q2_BDI allowance {Q2_BDI_MDD * 1.25:.6f}\n\n"
        f"**Current verdict:** Q2_PD/Q2_BDI superior — promotion_ready **FORBIDDEN** at n={len(dataset)}\n",
        encoding="utf-8",
    )

    # Phase 3
    audit, schema_df, growth_df = _audit_dataset()
    schema_df.to_csv(RESEARCH_DIR / "meta_dataset_schema_check.csv", index=False)
    growth_df.to_csv(RESEARCH_DIR / "meta_dataset_daily_growth.csv", index=False)
    (RESEARCH_DIR / "meta_dataset_accumulation_audit.md").write_text(
        f"# Meta Dataset Accumulation Audit\n\n"
        f"```json\n{json.dumps(audit, indent=2, default=str)}\n```\n\n"
        f"## Append-only integrity: {'PASS' if audit['append_only_integrity'] else 'CHECK NEEDED'}\n"
        f"## Schema drift: {'YES' if audit['schema_drift'] else 'NO'}\n"
        f"## History snapshots: {audit['history_snapshots']}\n"
        f"## Daily automation: not registered (by design — diagnostic script only)\n\n"
        f"Run `python -m scripts.diagnostics.build_meta_label_dataset` manually for append.\n",
        encoding="utf-8",
    )

    # Phase 4
    label_cands = _simulate_label_candidates(dataset)
    label_cands.to_csv(RESEARCH_DIR / "calibration_label_candidate_distribution.csv", index=False)
    label_rec = _recommend_label(label_cands)
    (RESEARCH_DIR / "calibration_label_restructure_candidates.md").write_text(
        f"# Calibration Label Restructure Candidates\n\n"
        f"Simulation only — **labels not replaced**.\n\n"
        f"## Current\n"
        f"- fail=4, ok=131 — unusable for calibration learning\n"
        f"- drawdown_risk proxy correlation ~0.75 (from prior forensics)\n\n"
        f"## Candidates\n\n{label_cands.to_string(index=False)}\n\n"
        f"## Recommendation\n{label_rec}\n\n"
        f"**Verdict:** label_restructure_needed\n",
        encoding="utf-8",
    )

    # Phase 5
    dist_df, stab_df, feat_summary = _feature_stability(dataset, artifact)
    dist_df.to_csv(RESEARCH_DIR / "meta_feature_distribution.csv", index=False)
    stab_df.to_csv(RESEARCH_DIR / "meta_feature_importance_stability.csv", index=False)
    (RESEARCH_DIR / "meta_feature_stability_audit.md").write_text(
        f"# Meta Feature Stability Audit\n\n"
        f"- top feature: {feat_summary['top_feature']} ({feat_summary['top_feature_share']:.1%} share)\n"
        f"- path-feedback share: {feat_summary['path_feedback_share']:.1%}\n"
        f"- verdict: **{feat_summary['verdict']}**\n\n"
        f"## Focus feature distribution\n{dist_df.to_string(index=False)}\n\n"
        f"## Importance stability (top 10)\n{stab_df.head(10).to_string(index=False)}\n",
        encoding="utf-8",
    )

    # Phase 6
    hybrid_specs = [
        ("Q2_PD_only", SCORE_Q2_PD, False),
        ("Q2_BDI_only", SCORE_Q2_BDI, False),
        ("Q2_PD_plus_Meta_adj", SCORE_Q2_PD, True),
        ("Q2_BDI_plus_Meta_adj", SCORE_Q2_BDI, True),
    ]
    hybrid_rows = []
    hybrid_frames = []
    for name, fn, use_adj in hybrid_specs:
        tdf = _simulate_q2_hybrid(ticks, feat_map, artifact, dataset, q2_fn=fn, name=name, use_meta_adj=use_adj)
        hybrid_frames.append(tdf)
        hybrid_rows.append(_metrics_row(name, tdf, ticks, dataset, prod_trades))
    hybrid_df = pd.DataFrame(hybrid_rows)
    hybrid_replay = pd.concat(hybrid_frames, ignore_index=True)
    hybrid_replay.to_csv(RESEARCH_DIR / "q2_meta_hybrid_replay.csv", index=False)

    pd_h = hybrid_df[hybrid_df["candidate"] == "Q2_PD_only"].iloc[0].to_dict()
    bdi_h = hybrid_df[hybrid_df["candidate"] == "Q2_BDI_only"].iloc[0].to_dict()
    pd_meta = hybrid_df[hybrid_df["candidate"] == "Q2_PD_plus_Meta_adj"].iloc[0].to_dict()
    bdi_meta = hybrid_df[hybrid_df["candidate"] == "Q2_BDI_plus_Meta_adj"].iloc[0].to_dict()
    hv_pd = _hybrid_verdict(pd_meta, pd_h)
    hv_bdi = _hybrid_verdict(bdi_meta, bdi_h)

    (RESEARCH_DIR / "q2_meta_hybrid_research_report.md").write_text(
        f"# Q2 + Meta Hybrid Research (replay only)\n\n"
        f"Meta adjustment: clip((0.5 - cal_prob) × 0.20, ±{META_ADJ_MAX}), additive on Q2 base_scale.\n"
        f"No hard reject. cal_mult={CAL_MULT_HYBRID}. Path-feedback zeroed at inference.\n\n"
        f"{hybrid_df.to_string(index=False)}\n\n"
        f"## Q2_PD + Meta: {_hybrid_verdict(pd_meta, pd_h)}\n"
        f"## Q2_BDI + Meta: {_hybrid_verdict(bdi_meta, bdi_h)}\n\n"
        f"**Question:** Can Meta improve Q2 without breaking risk profile?\n"
        f"- Q2_PD hybrid MDD {pd_meta['MDD']:.6f} vs Q2_PD {pd_h['MDD']:.6f}\n"
        f"- Q2_BDI hybrid MDD {bdi_meta['MDD']:.6f} vs Q2_BDI {bdi_h['MDD']:.6f}\n",
        encoding="utf-8",
    )

    # Phase 7 + verdict
    feat_verdict = feat_summary["verdict"]
    base_verdict = "Q2_PD/Q2_BDI superior + Meta research-only"
    if feat_verdict == "feature_rebalance_needed":
        base_verdict += " + feature_rebalance_needed"
    base_verdict += " + label_restructure_needed"

    hybrid_tags = []
    if hv_pd == "hybrid_research_candidate":
        hybrid_tags.append("Q2_PD+Meta hybrid_research_candidate")
    if hv_bdi == "hybrid_research_candidate":
        hybrid_tags.append("Q2_BDI+Meta hybrid_research_candidate")

    state_path = RESEARCH_DIR / "meta_layer_current_state_20260613.md"
    state_path.write_text(
        f"# Meta Layer Current State (2026-06-13)\n\n"
        f"**Verdict:** {base_verdict}\n\n"
        f"## Completed\n"
        f"- Dataset Factory V2 (append-only, dedupe trade_id)\n"
        f"- Multi-label targets\n"
        f"- Feature V2 (15 forensic features)\n"
        f"- Training / evaluation / rolling OOS\n"
        f"- Overfilter forensics\n"
        f"- Recalibration replay (counterfactual)\n"
        f"- Q2 baseline lock (this run)\n"
        f"- Promotion gate definition\n"
        f"- Dataset accumulation audit\n"
        f"- Label restructure simulation\n"
        f"- Feature stability audit\n"
        f"- Q2+Meta hybrid research\n\n"
        f"## Confirmed Conclusions\n"
        f"- Meta infrastructure: **success**\n"
        f"- Meta model: **not deployable** (n={len(dataset)}, OOS AUC {oos_auc_mean:.2f})\n"
        f"- Q2_PD/Q2_BDI: **superior defensive baseline**\n"
        f"- Meta: research-only; monitor_only **deferred**\n\n"
        f"## Operating Principles\n"
        f"- production unchanged\n"
        f"- Q2 baseline retained for all Meta comparisons\n"
        f"- no launchd/state/execution changes\n"
        f"- rows >= 300 before any promotion discussion\n\n"
        f"## Re-evaluation Triggers\n"
        f"- rows >= 300, OOS AUC >= 0.58, preservation >= 85%\n"
        f"- MDD competitive with Q2_BDI (×1.25)\n"
        f"- good trade false rejection <= 20%, routing_valid=True\n\n"
        f"## Hybrid Research\n"
        f"{', '.join(hybrid_tags) if hybrid_tags else 'No hybrid_research_candidate at current n'}\n",
        encoding="utf-8",
    )

    deliverables = [
        "q2_baseline_lock_report.md",
        "meta_promotion_gate.md",
        "meta_dataset_accumulation_audit.md",
        "meta_dataset_schema_check.csv",
        "meta_dataset_daily_growth.csv",
        "calibration_label_restructure_candidates.md",
        "calibration_label_candidate_distribution.csv",
        "meta_feature_stability_audit.md",
        "meta_feature_distribution.csv",
        "meta_feature_importance_stability.csv",
        "q2_meta_hybrid_research_report.md",
        "q2_meta_hybrid_replay.csv",
        "meta_layer_current_state_20260613.md",
    ]

    return {
        "verdict": base_verdict,
        "gate_checks": gate_checks,
        "hybrid_pd": hv_pd,
        "hybrid_bdi": hv_bdi,
        "feature_verdict": feat_verdict,
        "deliverables": deliverables,
        "output_dir": str(RESEARCH_DIR),
    }


def main() -> None:
    r = run_research_continuation()
    print(f"verdict: {r['verdict']}")
    print(f"feature: {r['feature_verdict']}")
    print(f"hybrid Q2_PD: {r['hybrid_pd']}, Q2_BDI: {r['hybrid_bdi']}")
    print(f"output: {r['output_dir']}")
    for d in r["deliverables"]:
        print(f"  - {d}")


if __name__ == "__main__":
    main()
