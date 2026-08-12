#!/usr/bin/env python3
"""QQQ regime feature research CLI (walk-forward / ablation / seen-reference)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.regime.config import paths
from canbit_equity.regime.orchestration import run_full_regime_research


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--walk-forward", action="store_true", dest="walk_forward")
    g.add_argument("--ablation", action="store_true")
    g.add_argument("--seen-reference-only", action="store_true", dest="seen_reference_only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    p = paths()

    if args.seen_reference_only:
        out = {
            "holdout_status": "SEEN_REFERENCE_ONLY",
            "confirmatory_value": "NONE",
            "used_for_selection": False,
            "used_for_score": False,
            "used_for_verdict": False,
            "note": "Default --full does not run seen reference. Explicit call records policy only; no candidate change.",
        }
        (p.diag / "reports" / "seen_reference_policy.json").write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return 0

    # walk-forward / ablation: run full research (idempotent enough) then print subset
    compact = run_full_regime_research()
    if args.ablation:
        abl_path = p.root / "reports" / "qqq_regime_ablation_report.json"
        print(json.dumps(json.loads(abl_path.read_text()) if abl_path.exists() else compact, indent=2, default=str))
    else:
        print(json.dumps(compact, indent=2, default=str))
    return 0 if compact.get("external_research_run") else 2


if __name__ == "__main__":
    raise SystemExit(main())
