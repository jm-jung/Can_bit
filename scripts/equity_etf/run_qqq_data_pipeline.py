#!/usr/bin/env python3
"""QQQ historical data pipeline CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.config import QQQConfig, ensure_dirs, paths
from canbit_equity.data_provider import download_qqq_daily
from canbit_equity.data_quality import audit_normalized
from canbit_equity.features import build_features
from canbit_equity.labels import add_labels
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--update", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = QQQConfig()
    ensure_dirs(cfg)
    p = paths(cfg)

    if args.status:
        man = {}
        if p["manifest_latest"].exists():
            man = json.loads(p["manifest_latest"].read_text())
        out = {
            "symbol": "QQQ",
            "manifest_present": p["manifest_latest"].exists(),
            "normalized_present": p["normalized_file"].exists(),
            "features_present": p["features_file"].exists(),
            "actual_start": man.get("actual_start"),
            "actual_end": man.get("actual_end"),
            "row_count": man.get("row_count"),
            "production_ready": False,
            "promotion_ready": False,
        }
        print(json.dumps(out, indent=2))
        return 0

    try:
        df, manifest = download_qqq_daily(cfg)
    except Exception as exc:
        out = {"verdict": "QQQ_DATA_PROVIDER_BLOCKED", "error": type(exc).__name__, "detail": str(exc)[:300], "production_ready": False, "promotion_ready": False}
        print(json.dumps(out, indent=2))
        return 2

    quality = audit_normalized(df, cfg)
    if quality["verdict"] == "QQQ_DATA_PIPELINE_FAIL":
        print(json.dumps({"verdict": quality["verdict"], "manifest": manifest, "quality": quality, "production_ready": False, "promotion_ready": False}, indent=2, default=str))
        return 3

    feats, fman = build_features(df, cfg)
    labeled = add_labels(feats, cfg, cost_bps_per_side=cfg.cost_bps_per_side_base)
    labeled_path = p["features"] / "qqq_daily_features_labeled.parquet"
    labeled.to_parquet(labeled_path, index=False)
    out = {
        "verdict": quality["verdict"],
        "manifest": manifest,
        "quality": quality,
        "feature_manifest": fman,
        "labeled_rows": int(len(labeled)),
        "labeled_path": str(labeled_path),
        "production_ready": False,
        "promotion_ready": False,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
