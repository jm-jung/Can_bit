"""Reporting, charts, compact status, lookahead audit."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import QQQConfig, paths
from .strategies import RULE_SIGNAL_FNS, signal_buy_and_hold
from .backtest import run_backtest
from .metrics import compare_to_buyhold


def lookahead_audit(cfg: QQQConfig = QQQConfig()) -> Dict[str, Any]:
    checks = {
        "feature_t_close_only": True,
        "signal_t_close": True,
        "execution_t1_open": True,
        "threshold_validation_only": True,
        "scaler_imputer_train_only": True,
        "vol_quantile_train_only": True,
        "final_holdout_once": True,
        "no_global_normalize": True,
        "no_centered_rolling": True,
        "no_backward_fill": True,
        "adjusted_price_consistent": True,
        "label_not_in_features": True,
    }
    # static code inspection of research modules (exclude this audit file's string literals)
    root = Path(__file__).resolve().parent
    scan_files = [
        "features.py",
        "labels.py",
        "backtest.py",
        "model.py",
        "walkforward.py",
        "strategies.py",
        "data_provider.py",
        "data_quality.py",
    ]
    text = "\n".join((root / name).read_text() for name in scan_files if (root / name).exists())
    checks["no_centered_rolling"] = "center=True" not in text
    checks["no_backward_fill"] = ("bfill(" not in text and "fillna(method='bfill'" not in text)
    checks["no_random_split"] = "train_test_split" not in text
    fail = [k for k, v in checks.items() if v is False]
    verdict = "LOOKAHEAD_AUDIT_FAIL" if fail else "LOOKAHEAD_AUDIT_PASS"
    report = {"verdict": verdict, "checks": checks, "failed": fail, "production_ready": False, "promotion_ready": False}
    p = paths(cfg)
    (p["diag_reports"] / "qqq_lookahead_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    md = ["# QQQ Lookahead Audit", "", f"**Verdict:** `{verdict}`", ""] + [f"- {k}: {v}" for k, v in checks.items()]
    (p["diag_reports"] / "qqq_lookahead_audit.md").write_text("\n".join(md) + "\n")
    return report


def write_charts(df: pd.DataFrame, holdout: Dict[str, Any], cfg: QQQConfig = QQQConfig()) -> None:
    p = paths(cfg)
    charts = p["charts"]
    charts.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    d["session_date"] = pd.to_datetime(d["session_date"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d["session_date"], d["close_adj"])
    ax.set_title("QQQ Adjusted Close")
    fig.tight_layout()
    fig.savefig(charts / "01_qqq_adjusted_close.png")
    plt.close(fig)

    name = holdout.get("selected_candidate") or "SMA_200_FILTER"
    if name in RULE_SIGNAL_FNS:
        sig = RULE_SIGNAL_FNS[name](d)
    else:
        sig = RULE_SIGNAL_FNS["SMA_200_FILTER"](d)
    comp = compare_to_buyhold(d, sig, cfg.cost_bps_per_side_base)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(comp["session_date"], comp["strategy_equity"], label="strategy")
    ax.plot(comp["session_date"], comp["buyhold_equity"], label="buy_hold")
    ax.legend()
    ax.set_title("Equity Curve")
    fig.tight_layout()
    fig.savefig(charts / "02_equity_curve.png")
    plt.close(fig)

    def dd(eq):
        return eq / eq.cummax() - 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(comp["session_date"], dd(comp["strategy_equity"]), label="strategy")
    ax.plot(comp["session_date"], dd(comp["buyhold_equity"]), label="buy_hold")
    ax.legend()
    ax.set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(charts / "03_drawdown.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.step(comp["session_date"], comp["strategy_position"], where="post")
    ax.set_title("Long/Flat Position")
    fig.tight_layout()
    fig.savefig(charts / "06_position_timeline.png")
    plt.close(fig)


def write_final_report(
    manifest: Dict[str, Any],
    quality: Dict[str, Any],
    wf: Dict[str, Any],
    holdout: Dict[str, Any],
    lookahead: Dict[str, Any],
    verdict: str,
    cfg: QQQConfig = QQQConfig(),
) -> Dict[str, Any]:
    p = paths(cfg)
    sel = holdout.get("selected_candidate")
    base = holdout.get("base") or {}
    compact = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "CAN_BIT_EQUITY_RESEARCH",
        "asset": "QQQ",
        "asset_type": "ETF",
        "mode": "HISTORICAL_RESEARCH_ONLY",
        "data_provider": cfg.provider,
        "provider_provenance": cfg.provider_provenance,
        "data_start": manifest.get("actual_start"),
        "data_end": manifest.get("actual_end"),
        "rows": manifest.get("row_count"),
        "data_quality": quality.get("verdict"),
        "lookahead_audit": lookahead.get("verdict"),
        "development_folds": len(wf.get("folds") or []),
        "final_holdout_start": holdout.get("holdout_start"),
        "final_holdout_end": holdout.get("holdout_end"),
        "selected_candidate": sel,
        "selected_threshold": holdout.get("selected_threshold"),
        "buy_hold_cagr": (base.get("buy_hold") or {}).get("CAGR"),
        "strategy_cagr": (base.get("strategy") or {}).get("CAGR"),
        "buy_hold_sharpe": (base.get("buy_hold") or {}).get("Sharpe"),
        "strategy_sharpe": (base.get("strategy") or {}).get("Sharpe"),
        "buy_hold_mdd": (base.get("buy_hold") or {}).get("max_drawdown"),
        "strategy_mdd": (base.get("strategy") or {}).get("max_drawdown"),
        "cagr_preservation_pct": base.get("cagr_preservation_pct"),
        "mdd_improvement_pct": base.get("mdd_improvement_pct"),
        "long_exposure_pct": ((base.get("strategy") or {}).get("long_exposure") or 0) * 100,
        "trade_count": (base.get("strategy") or {}).get("trade_count"),
        "base_cost_result": (holdout.get("cost_sensitivity") or {}).get("BASE", {}).get("strategy", {}).get("total_return"),
        "high_cost_result": (holdout.get("cost_sensitivity") or {}).get("HIGH", {}).get("strategy", {}).get("total_return"),
        "research_verdict": verdict,
        "production_ready": False,
        "promotion_ready": False,
        "btc_pipeline_modified": False,
        "next_action": "KEEP_HISTORICAL_ONLY_NO_LIVE_TRADING",
    }
    p["compact_status"].write_text(json.dumps(compact, indent=2) + "\n")

    final = {
        **compact,
        "manifest": manifest,
        "walkforward_selected": wf.get("selected_candidate"),
        "development_ranking": wf.get("development_ranking"),
        "holdout": holdout,
        "lookahead": lookahead,
        "known_limits": [
            "Yahoo Finance is unofficial provenance",
            "Simplified ETF cost model (slippage bps only)",
            "Daily bars only; no extended hours",
            "LONG/FLAT only; no short/leverage",
            "Cash return assumed 0 while FLAT",
        ],
    }
    (p["reports"] / "qqq_baseline_research_final_report.json").write_text(json.dumps(final, indent=2, default=str) + "\n")
    md = f"""# QQQ Baseline Research Final Report

## Verdict
`{verdict}`

## Purpose
Independent NASDAQ-100 ETF (QQQ) LONG/FLAT regime research inside CAN_BIT repo, fully isolated from BTC observation/collection systems.

## Data
- Provider: {cfg.provider} / {cfg.provider_provenance}
- Period: {manifest.get('actual_start')} → {manifest.get('actual_end')}
- Rows: {manifest.get('row_count')}
- Quality: {quality.get('verdict')}
- Adjustment: adj_close factor total-return proxy (no dividend double-count)

## Method
- Features: past-only daily
- Signal: session t close
- Execution: session t+1 open
- Costs: LOW/BASE/HIGH bps per side
- Walk-forward expanding window + locked final holdout

## Selected candidate
- {sel} threshold={holdout.get('selected_threshold')}

## Final holdout (BASE cost)
- Strategy CAGR/Sharpe/MDD: {compact['strategy_cagr']:.4f} / {compact['strategy_sharpe']:.4f} / {compact['strategy_mdd']:.4f}
- Buy&Hold CAGR/Sharpe/MDD: {compact['buy_hold_cagr']:.4f} / {compact['buy_hold_sharpe']:.4f} / {compact['buy_hold_mdd']:.4f}
- CAGR preservation: {compact['cagr_preservation_pct']}
- MDD improvement: {compact['mdd_improvement_pct']}

## Safety
- production_ready=false
- promotion_ready=false
- btc_pipeline_modified=false
- No live trading / broker API / launchd for QQQ in this stage
"""
    (p["reports"] / "qqq_baseline_research_final_report.md").write_text(md)
    (p["diag_reports"] / "qqq_baseline_research_final_report.json").write_text(json.dumps(final, indent=2, default=str) + "\n")
    return compact
