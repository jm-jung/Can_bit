#!/usr/bin/env python3
"""QQQ regime prospective daily update CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.regime.prospective.config import DIAG_ROOT, PROSP_ROOT
from canbit_equity.regime.prospective.daily import run_daily_update
from canbit_equity.regime.prospective.discord_notify import maybe_notify_from_state
from canbit_equity.regime.prospective.status import build_status_payload


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--update", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.status:
        payload = build_status_payload(compact=True)
        print(json.dumps(payload, indent=2, default=str))
        # nonzero only for integrity-critical status (never for mere staleness)
        if payload.get("status") == "CORRUPT":
            return 2
        return 0

    try:
        out = run_daily_update()
    except Exception as exc:
        err = {
            "status": "UPDATE_FAILED",
            "error": type(exc).__name__,
            "error_message": str(exc)[:500],
            "production_ready": False,
            "promotion_ready": False,
            "execution_enabled": False,
            "private_calls": 0,
            "order_calls": 0,
        }
        print(json.dumps(err, indent=2, default=str))
        return 1

    state = json.loads((PROSP_ROOT / "state/prospective_state.json").read_text())
    disc = maybe_notify_from_state(state, dry_run=False)
    out["discord_result"] = {k: disc.get(k) for k in disc if k != "content"}
    print(json.dumps(out, indent=2, default=str))
    if out.get("operational_status") == "CRITICAL":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
