#!/usr/bin/env python3
"""QQQ baseline research CLI (rules / walk-forward / final holdout)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.config import QQQConfig, ensure_dirs, paths
from canbit_equity.reporting import lookahead_audit, write_charts, write_final_report
from canbit_equity.walkforward import evaluate_rules, research_verdict, run_final_holdout, run_walkforward


def _load_labeled(cfg: QQQConfig) -> pd.DataFrame:
    p = paths(cfg)
    path = p["features"] / "qqq_daily_features_labeled.parquet"
    if not path.exists():
        raise FileNotFoundError("labeled features missing; run data pipeline --update first")
    return pd.read_parquet(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rules-only", action="store_true")
    g.add_argument("--walk-forward", action="store_true")
    g.add_argument("--final-holdout", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = QQQConfig()
    ensure_dirs(cfg)
    df = _load_labeled(cfg)

    if args.rules_only:
        out = {
            "LOW": evaluate_rules(df, cfg.cost_bps_per_side_low, cfg),
            "BASE": evaluate_rules(df, cfg.cost_bps_per_side_base, cfg),
            "HIGH": evaluate_rules(df, cfg.cost_bps_per_side_high, cfg),
            "production_ready": False,
            "promotion_ready": False,
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    if args.walk_forward:
        wf = run_walkforward(df, cfg, cost_bps=cfg.cost_bps_per_side_base)
        print(json.dumps({"verdict": "WALKFORWARD_OK", "selected_candidate": wf.get("selected_candidate"), "folds": len(wf.get("folds") or []), "production_ready": False, "promotion_ready": False}, indent=2, default=str))
        (paths(cfg)["reports"] / "qqq_walkforward_report.md").write_text(
            f"# Walk-forward\n\nfolds={len(wf.get('folds') or [])}\nselected={wf.get('selected_candidate')}\n"
        )
        return 0

    # final holdout requires prior wf artifact
    p = paths(cfg)
    wf_path = p["reports"] / "qqq_walkforward_report.json"
    if not wf_path.exists():
        wf = run_walkforward(df, cfg, cost_bps=cfg.cost_bps_per_side_base)
    else:
        wf = json.loads(wf_path.read_text())
    holdout = run_final_holdout(df, wf, cfg, cost_bps=cfg.cost_bps_per_side_base)
    print(json.dumps({"verdict": "HOLDOUT_OK", **{k: holdout[k] for k in ["selected_candidate", "selected_threshold", "holdout_start", "holdout_end", "base"]}, "production_ready": False, "promotion_ready": False}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
