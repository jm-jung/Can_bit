#!/usr/bin/env python3
"""Corrected same-period QQQ candidate selection (v2). No network. No legacy overwrite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.config import QQQConfig
from canbit_equity.selection.same_period import DIAG, run_corrected_selection, verify_fixed_input


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--audit-only", action="store_true", dest="audit_only")
    g.add_argument("--status", action="store_true")
    g.add_argument("--compare-candidates", action="store_true", dest="compare_candidates")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = QQQConfig()

    if args.status:
        path = DIAG / "reports/qqq_same_period_selection_compact.json"
        print(json.dumps(json.loads(path.read_text()) if path.exists() else {"status": "NOT_RUN"}, indent=2))
        return 0

    if args.audit_only:
        out = verify_fixed_input(cfg)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("ok") else 2

    if args.full or args.compare_candidates:
        result = run_corrected_selection(cfg)
        if args.compare_candidates and "ranking" in result:
            print(json.dumps(result["ranking"].to_dict(orient="records"), indent=2, default=str))
        else:
            payload = result.get("compact") or result
            print(json.dumps(payload, indent=2, default=str))
        if result.get("verdict") == "SAME_PERIOD_SELECTION_CORRECTION_FAILED":
            return 2
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
