"""Full regime research orchestration."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from canbit_equity.config import paths as qqq_paths
from canbit_equity.regime.config import EXPECTED_LABELED_HASH, V2_PATH, paths
from canbit_equity.regime.features import build_regime_features
from canbit_equity.regime.nested_walkforward import build_common_period, run_nested_walkforward
from canbit_equity.regime.prospective import (
    audit_external_data,
    lookahead_audit,
    research_verdict,
    write_prospective_lock,
)
from canbit_equity.regime.providers import update_all_external

REPO = Path(__file__).resolve().parents[3]


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _simple_charts(wf: Dict[str, Any]) -> None:
    """Save lightweight chart CSVs (matplotlib optional)."""
    p = paths()
    chart_dir = p.root / "reports" / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    rank = wf["ranking"]
    rank.to_csv(chart_dir / "ablation_comparison.csv", index=False)
    fm = wf["fold_metrics"]
    fm.to_csv(chart_dir / "fold_return_comparison.csv", index=False)
    # equity curves from streams
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for name in ["DUAL_TREND_FILTER", "LOGISTIC_PRICE_ONLY", "LOGISTIC_PRICE_PLUS_ALL_REGIME"]:
            if name not in wf["streams"]:
                continue
            s = wf["streams"][name]
            eq = (1 + s["net_ret"].fillna(0)).cumprod()
            ax.plot(pd.to_datetime(s["session_date"]), eq.values, label=name, linewidth=1.2)
        ax.legend(fontsize=7)
        ax.set_title("Candidate OOS Equity (common folds)")
        fig.tight_layout()
        fig.savefig(chart_dir / "candidate_oos_equity_curves.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(rank["candidate"], rank["Sharpe"])
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        ax.set_title("OOS Sharpe by candidate")
        fig.tight_layout()
        fig.savefig(chart_dir / "fold_sharpe_comparison.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass


def run_full_regime_research() -> Dict[str, Any]:
    p = paths()
    qp = qqq_paths()
    labeled_path = qp["features"] / "qqq_daily_features_labeled.parquet"
    labeled_hash = _sha_file(labeled_path)
    if labeled_hash != EXPECTED_LABELED_HASH:
        return {
            "research_verdict": "QQQ_REGIME_IMPLEMENTATION_FAILED",
            "reason": "labeled_hash_mismatch",
            "external_data_downloaded": False,
            "external_research_run": False,
            "production_ready": False,
            "promotion_ready": False,
            "current_action": "STOP_AND_FIX_REGIME_IMPLEMENTATION",
        }
    if not V2_PATH.exists():
        return {
            "research_verdict": "QQQ_REGIME_IMPLEMENTATION_FAILED",
            "reason": "missing_same_period_v2",
            "external_data_downloaded": False,
            "external_research_run": False,
            "production_ready": False,
            "promotion_ready": False,
            "current_action": "STOP_AND_FIX_REGIME_IMPLEMENTATION",
        }

    # 1) external download
    ext = update_all_external()
    if not ext.get("external_data_downloaded"):
        out = {
            "research_verdict": "QQQ_REGIME_DATA_PIPELINE_FAIL",
            "external_data_downloaded": False,
            "external_research_run": False,
            "errors": ext.get("errors"),
            "missing_series": ext.get("missing_series"),
            "production_ready": False,
            "promotion_ready": False,
            "current_action": "STOP_DUE_TO_EXTERNAL_DATA_FAILURE",
        }
        (p.diag / "reports" / "qqq_regime_compact_status.json").write_text(json.dumps(out, indent=2) + "\n")
        return out

    # 2) data quality
    dq = audit_external_data(ext)
    if dq["verdict"] == "QQQ_REGIME_DATA_FAIL":
        out = {
            "research_verdict": "QQQ_REGIME_DATA_PIPELINE_FAIL",
            "external_data_downloaded": True,
            "external_research_run": False,
            "data_quality": dq,
            "production_ready": False,
            "promotion_ready": False,
            "current_action": "STOP_DUE_TO_DATA_QUALITY",
        }
        (p.diag / "reports" / "qqq_regime_compact_status.json").write_text(json.dumps(out, indent=2) + "\n")
        return out

    # 3-4) features + alignment
    labeled = pd.read_parquet(labeled_path)
    # never use stale 68-row features file
    feat_df, feat_man = build_regime_features(labeled)
    align = feat_man.get("align_audit") or {}
    if align.get("verdict") == "ALIGNMENT_FAIL":
        out = {
            "research_verdict": "QQQ_REGIME_LOOKAHEAD_FAIL",
            "external_data_downloaded": True,
            "external_research_run": False,
            "alignment": align,
            "production_ready": False,
            "promotion_ready": False,
            "current_action": "STOP_DUE_TO_LOOKAHEAD",
        }
        (p.diag / "reports" / "qqq_regime_compact_status.json").write_text(json.dumps(out, indent=2) + "\n")
        return out

    # 5) common period
    common, common_meta = build_common_period(feat_df)
    (p.diag / "common_period" / "common_period_definition.json").write_text(json.dumps(common_meta, indent=2) + "\n")
    if common_meta["verdict"] != "COMMON_PERIOD_PASS":
        out = {
            "research_verdict": "QQQ_REGIME_COMMON_PERIOD_FAIL",
            "external_data_downloaded": True,
            "external_research_run": False,
            "common_period": common_meta,
            "production_ready": False,
            "promotion_ready": False,
            "current_action": "STOP_DUE_TO_COMMON_PERIOD_FAILURE",
        }
        (p.diag / "reports" / "qqq_regime_compact_status.json").write_text(json.dumps(out, indent=2) + "\n")
        return out

    # 6) nested WF
    wf = run_nested_walkforward(common)
    if wf.get("verdict") == "QQQ_REGIME_COMMON_PERIOD_FAIL":
        out = {
            "research_verdict": "QQQ_REGIME_COMMON_PERIOD_FAIL",
            "external_data_downloaded": True,
            "external_research_run": True,
            "wf": wf,
            "production_ready": False,
            "promotion_ready": False,
            "current_action": "STOP_DUE_TO_COMMON_PERIOD_FAILURE",
        }
        (p.diag / "reports" / "qqq_regime_compact_status.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
        return out

    # 7) lookahead
    la = lookahead_audit(labeled_hash, align, common_meta, wf)

    # 8) verdict
    rv = research_verdict(wf, dq["verdict"], align.get("verdict", ""), la["verdict"], common_meta["verdict"])

    # 9) prospective lock
    create_lock = rv.get("action") == "LOCK_REGIME_CANDIDATE_FOR_PROSPECTIVE_OBSERVATION" or rv.get(
        "create_prospective_lock"
    )
    # Only lock if a regime logistic was selected
    sel = wf.get("selected")
    lock = None
    if create_lock and sel is not None and str(sel["candidate"]).startswith("LOGISTIC_PRICE_PLUS_"):
        lock = write_prospective_lock(sel, wf, feat_man, ext, labeled_hash, True)
    elif rv.get("action") == "LOCK_REGIME_CANDIDATE_FOR_PROSPECTIVE_OBSERVATION" and sel is not None:
        lock = write_prospective_lock(sel, wf, feat_man, ext, labeled_hash, True)

    # 10) ablation report = ranking of logistic groups
    abl = wf["ranking"][wf["ranking"]["candidate"].str.startswith("LOGISTIC_")].copy()
    (p.root / "reports" / "qqq_regime_ablation_report.json").write_text(
        json.dumps(abl.to_dict(orient="records"), indent=2, default=str) + "\n"
    )
    (p.root / "reports" / "qqq_regime_ablation_report.md").write_text(
        "# Ablation\n\n" + abl[["candidate", "score", "Sharpe", "cagr_preservation_pct", "max_drawdown", "long_exposure", "positive_fold_ratio"]].to_string(index=False) + "\n"
    )

    _simple_charts(wf)

    # walkforward report
    wf_rep = {
        "n_folds": wf["n_folds"],
        "oos_start": wf["oos_start"],
        "oos_end": wf["oos_end"],
        "oos_rows": wf["oos_rows"],
        "index_hash": wf["index_hash"],
        "selected": wf.get("selected"),
        "ranking": wf["ranking"].to_dict(orient="records"),
    }
    (p.root / "reports" / "qqq_regime_walkforward_report.json").write_text(json.dumps(wf_rep, indent=2, default=str) + "\n")
    (p.root / "reports" / "qqq_regime_walkforward_report.md").write_text(
        f"# Walk-forward\n\nfolds={wf['n_folds']} OOS={wf['oos_start']}→{wf['oos_end']} rows={wf['oos_rows']}\n"
        f"selected={None if sel is None else sel['candidate']}\n"
    )

    def metr(name: str) -> Dict[str, Any]:
        row = wf["ranking"][wf["ranking"]["candidate"] == name]
        if not len(row):
            return {}
        r = row.iloc[0]
        return {
            "CAGR": r["CAGR"],
            "Sharpe": r["Sharpe"],
            "MDD": r["max_drawdown"],
            "cagr_preservation_pct": r["cagr_preservation_pct"],
            "exposure": r["long_exposure"],
            "trades": r["trade_count"],
            "score": r["score"],
            "positive_folds": int(r["positive_folds"]),
            "positive_fold_ratio": r["positive_fold_ratio"],
            "low": r["low_cost_total_return"],
            "base": r["total_return"],
            "high": r["high_cost_total_return"],
            "top_fold_removal_mean": r["top_fold_removal_mean_return"],
        }

    compact = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_verdict": rv["verdict"],
        "mode": "HISTORICAL_REGIME_RESEARCH_ONLY",
        "full_orchestration": True,
        "gate": "SAME_PERIOD_COMPARISON_PASS",
        "corrected_selection_source": str(V2_PATH.relative_to(REPO)),
        "external_data_downloaded": True,
        "external_research_run": True,
        "qqq_input_hash": labeled_hash,
        "external_manifest_hash": ext.get("manifest_sha256"),
        "feature_hash": feat_man.get("feature_set_hash"),
        "external_providers": ["yfinance", "fred"],
        "series_downloaded": [m["series_id"] for m in ext.get("series") or []],
        "common_research_start": common_meta["common_start"],
        "common_research_end": common_meta["common_end"],
        "common_rows": common_meta["common_rows"],
        "oos_start": wf["oos_start"],
        "oos_end": wf["oos_end"],
        "oos_rows": wf["oos_rows"],
        "outer_folds": wf["n_folds"],
        "data_quality": dq["verdict"],
        "alignment_audit": align.get("verdict"),
        "lookahead_audit": la["verdict"],
        "selected_candidate": None if sel is None else sel["candidate"],
        "selected_score": None if sel is None else sel["score"],
        "eligible_candidates": wf["ranking"][wf["ranking"]["eligible"]]["candidate"].tolist(),
        "metrics": {
            "BUY_AND_HOLD": metr("BUY_AND_HOLD"),
            "DUAL_TREND_FILTER": metr("DUAL_TREND_FILTER"),
            "LOGISTIC_PRICE_ONLY": metr("LOGISTIC_PRICE_ONLY"),
            "LOGISTIC_PRICE_PLUS_VOL": metr("LOGISTIC_PRICE_PLUS_VOL"),
            "LOGISTIC_PRICE_PLUS_RATES": metr("LOGISTIC_PRICE_PLUS_RATES"),
            "LOGISTIC_PRICE_PLUS_BREADTH": metr("LOGISTIC_PRICE_PLUS_BREADTH"),
            "LOGISTIC_PRICE_PLUS_CREDIT": metr("LOGISTIC_PRICE_PLUS_CREDIT"),
            "LOGISTIC_PRICE_PLUS_ALL_REGIME": metr("LOGISTIC_PRICE_PLUS_ALL_REGIME"),
        },
        "old_holdout_accessed": False,
        "old_holdout_used_for_selection": False,
        "old_holdout_used_for_verdict": False,
        "old_holdout_status": "SEEN_HISTORICAL_REFERENCE",
        "prospective_lock": None if lock is None else {"path": lock.get("path"), "candidate": lock.get("candidate_id"), "first_unseen_session": lock.get("first_unseen_session")},
        "production_ready": False,
        "promotion_ready": False,
        "current_action": rv["action"],
        "btc_pipeline_modified": False,
    }

    final = {
        **compact,
        "research_detail": rv,
        "common_period": common_meta,
        "ranking": wf["ranking"].to_dict(orient="records"),
    }
    (p.root / "reports" / "qqq_regime_feature_research_final_report.json").write_text(
        json.dumps(final, indent=2, default=str) + "\n"
    )
    (p.root / "reports" / "qqq_regime_feature_research_final_report.md").write_text(
        f"""# QQQ Regime Feature Research

## Verdict
`{rv['verdict']}`

## Gate
SAME_PERIOD_COMPARISON_PASS · external_data_downloaded=true · external_research_run=true

## Common period
{common_meta['common_start']} → {common_meta['common_end']} ({common_meta['common_rows']} rows)
OOS folds: {wf['n_folds']} · {wf['oos_start']} → {wf['oos_end']} ({wf['oos_rows']} rows)

## Selected
`{compact['selected_candidate']}` score={compact['selected_score']}

## CURRENT ACTION
`{rv['action']}`

production_ready=false · promotion_ready=false
"""
    )
    (p.diag / "reports" / "qqq_regime_compact_status.json").write_text(json.dumps(compact, indent=2, default=str) + "\n")
    return compact
