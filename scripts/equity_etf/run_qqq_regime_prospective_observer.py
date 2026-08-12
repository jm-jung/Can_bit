#!/usr/bin/env python3
"""QQQ regime prospective observer CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.regime.prospective.daily import run_daily_update
from canbit_equity.regime.prospective.data_update import load_live_features, update_sources_and_features
from canbit_equity.regime.prospective.lock import audit_lock
from canbit_equity.regime.prospective.model import build_or_load_frozen_model
from canbit_equity.regime.prospective.outcomes import mature_outcomes
from canbit_equity.regime.prospective.shadow import update_shadow_from_predictions
import pandas as pd
from canbit_equity.regime.prospective.config import PROSP_ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-only", action="store_true")
    g.add_argument("--predict-latest", action="store_true")
    g.add_argument("--mature-outcomes", action="store_true")
    g.add_argument("--update-shadow", action="store_true")
    g.add_argument("--data-quality", action="store_true")
    g.add_argument("--rebuild-derived-state", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.audit_only:
        la = audit_lock()
        ma = build_or_load_frozen_model()
        print(json.dumps({"lock": la, "model": ma}, indent=2, default=str))
        return 0 if la["verdict"] == "QQQ_PROSPECTIVE_LOCK_PASS" and ma["verdict"] in (
            "QQQ_PROSPECTIVE_MODEL_PASS",
            "QQQ_PROSPECTIVE_MODEL_REBUILT_FROM_LOCK",
        ) else 2

    if args.data_quality:
        out = update_sources_and_features()
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("data_quality_verdict") != "QQQ_REGIME_DATA_FAIL" else 2

    if args.mature_outcomes or args.update_shadow or args.rebuild_derived_state:
        live = load_live_features()
        strict = pd.read_parquet(PROSP_ROOT / "predictions/strict_predictions.parquet")
        late = pd.read_parquet(PROSP_ROOT / "predictions/late_predictions.parquet")
        if args.mature_outcomes or args.rebuild_derived_state:
            matured = mature_outcomes(live, strict, late)
        else:
            matured = {"status": "SKIPPED"}
        if args.update_shadow or args.rebuild_derived_state:
            shadow = update_shadow_from_predictions(live, strict, late)
        else:
            shadow = {"status": "SKIPPED"}
        # refresh derived state metrics via daily if rebuild
        if args.rebuild_derived_state:
            out = run_daily_update()
            out["rebuild_derived_state"] = "LEDGERS_PRESERVED_METRICS_REFRESHED"
            out["mature_outcomes"] = matured
            out["shadow_update"] = shadow
            print(json.dumps(out, indent=2, default=str))
            return 0
        print(json.dumps({"mature_outcomes": matured, "shadow_update": shadow}, indent=2, default=str))
        return 0

    if args.predict_latest:
        out = run_daily_update()
        print(json.dumps(out, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
