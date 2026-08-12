#!/usr/bin/env python3
"""QQQ regime external data pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.regime.config import paths
from canbit_equity.regime.prospective import audit_external_data
from canbit_equity.regime.providers import update_all_external


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--update", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    p = paths()
    man_path = p.root / "manifests" / "qqq_regime_external_data_manifest.json"
    if args.status:
        if not man_path.exists():
            print(json.dumps({"status": "NO_MANIFEST", "external_data_downloaded": False}, indent=2))
            return 1
        man = json.loads(man_path.read_text())
        print(json.dumps({"status": "OK", "n_series": man.get("n_ok"), "manifest_sha256": man.get("manifest_sha256"), "errors": man.get("errors")}, indent=2))
        return 0
    man = update_all_external()
    dq = audit_external_data(man)
    print(json.dumps({"download": {"ok": man.get("external_data_downloaded"), "errors": man.get("errors"), "manifest_sha256": man.get("manifest_sha256")}, "data_quality": dq}, indent=2, default=str))
    return 0 if man.get("external_data_downloaded") and dq["verdict"] != "QQQ_REGIME_DATA_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
