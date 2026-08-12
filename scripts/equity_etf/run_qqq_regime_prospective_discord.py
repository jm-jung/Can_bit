#!/usr/bin/env python3
"""QQQ regime prospective Discord CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.regime.prospective.config import PROSP_ROOT
from canbit_equity.regime.prospective.discord_notify import flush_outbox, maybe_notify_from_state, send_test
from canbit_equity.regime.prospective.status import build_status_payload


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--send-test", action="store_true")
    g.add_argument("--flush-outbox", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.send_test:
        print(json.dumps(send_test(), indent=2))
        return 0
    if args.flush_outbox:
        print(json.dumps(flush_outbox(), indent=2))
        return 0

    # Dry-run uses ledger-aware status overlay (no forced fake provenance).
    state = build_status_payload(compact=True)
    # Simulate the 12:20 transient pending case against existing prediction
    if state.get("prediction_available") and state.get("locked_signal") in ("LONG", "FLAT"):
        state = {
            **state,
            "new_prediction": False,
            # Force the observed 12:20 refresh shape; do not keep STATUS_READ_ONLY_OVERLAY.
            "current_refresh_status": "PENDING_OR_MISSING",
            "operational_status": "WARNING",
        }
    res = maybe_notify_from_state(state, dry_run=True)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
