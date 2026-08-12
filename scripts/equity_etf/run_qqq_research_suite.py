#!/usr/bin/env python3
"""QQQ research suite orchestrator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.config import QQQConfig, ensure_dirs, paths
from canbit_equity.data_provider import download_qqq_daily
from canbit_equity.data_quality import audit_normalized
from canbit_equity.features import build_features
from canbit_equity.labels import add_labels
from canbit_equity.reporting import lookahead_audit, write_charts, write_final_report
from canbit_equity.walkforward import research_verdict, run_final_holdout, run_walkforward


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--audit-only", action="store_true")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = QQQConfig()
    ensure_dirs(cfg)
    p = paths(cfg)

    if args.status:
        compact = {}
        if p["compact_status"].exists():
            compact = json.loads(p["compact_status"].read_text())
        print(json.dumps(compact or {"status": "NOT_RUN", "production_ready": False, "promotion_ready": False}, indent=2))
        return 0

    if args.audit_only:
        out = {"config": cfg.to_dict(), "lookahead": lookahead_audit(cfg), "paths": {k: str(v) for k, v in p.items() if not str(k).endswith("repo")}}
        print(json.dumps(out, indent=2, default=str))
        return 0

    # full suite
    try:
        df, manifest = download_qqq_daily(cfg)
    except Exception as exc:
        print(json.dumps({"research_verdict": "QQQ_DATA_PROVIDER_BLOCKED", "error": type(exc).__name__, "detail": str(exc)[:300], "production_ready": False, "promotion_ready": False}, indent=2))
        return 2

    quality = audit_normalized(df, cfg)
    if quality["verdict"] == "QQQ_DATA_PIPELINE_FAIL":
        print(json.dumps({"research_verdict": "QQQ_DATA_PIPELINE_FAILED", "quality": quality, "production_ready": False, "promotion_ready": False}, indent=2, default=str))
        return 3

    feats, _ = build_features(df, cfg)
    labeled = add_labels(feats, cfg, cfg.cost_bps_per_side_base)
    labeled_path = p["features"] / "qqq_daily_features_labeled.parquet"
    labeled.to_parquet(labeled_path, index=False)

    la = lookahead_audit(cfg)
    wf = run_walkforward(labeled, cfg, cfg.cost_bps_per_side_base)
    holdout = run_final_holdout(labeled, wf, cfg, cfg.cost_bps_per_side_base)
    verdict = research_verdict(wf, holdout, quality["verdict"], la["verdict"])
    try:
        write_charts(labeled, holdout, cfg)
    except Exception:
        pass
    compact = write_final_report(manifest, quality, wf, holdout, la, verdict, cfg)
    print(json.dumps(compact, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
