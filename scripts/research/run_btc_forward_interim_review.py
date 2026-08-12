#!/usr/bin/env python3
"""Read-only BTC strict-live forward interim review CLI."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

os.environ["CANBIT_INTERIM_REVIEW_READ_ONLY"] = "1"

from canbit_research.btc_forward_interim import (  # noqa: E402
    InterimReviewError,
    build_status,
    run_full_review,
    run_review,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--since-effective-t0", action="store_true", help="alias of --full")
    g.add_argument("--window-days", type=int, default=None)
    ap.add_argument("--as-of-utc", type=str, default=None, help="Deterministic analysis as-of (UTC ISO)")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # Mutual exclusion: --full/--since-effective-t0 already exclusive with --window-days via group.
    # Extra guard if someone wires flags differently later.
    if (args.full or args.since_effective_t0) and args.window_days is not None:
        err = {"error": "ERROR_MUTUALLY_EXCLUSIVE_SCOPE", "message": "--full and --window-days cannot both be set"}
        print(json.dumps(err, indent=2))
        return 2

    if args.status:
        out = build_status()
        print(json.dumps(out, indent=2, default=str))
        return 0

    try:
        if args.window_days is not None:
            out = run_review(
                analysis_scope="ROLLING_WINDOW",
                window_days=args.window_days,
                as_of_utc=args.as_of_utc,
            )
        else:
            out = run_full_review(as_of_utc=args.as_of_utc)
    except InterimReviewError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}, indent=2))
        return 2

    if args.compact:
        from canbit_research.btc_forward_interim import OUT

        out = json.loads((OUT / "reports/btc_forward_interim_compact.json").read_text())
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
