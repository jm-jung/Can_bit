"""CLI for provenance-aware microstructure gap detection and recovery."""

from __future__ import annotations

import argparse
import json
from typing import Any

try:
    import microstructure_gap_backfill as gap
except ModuleNotFoundError:  # package import in tests
    from scripts.diagnostics import microstructure_gap_backfill as gap


def dump(value: Any) -> str:
    return json.dumps(gap.clean(value), ensure_ascii=False, indent=2, default=str)


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true")
    action.add_argument("--detect-only", action="store_true")
    action.add_argument("--backfill", action="store_true")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream", action="append", choices=gap.STREAM_ORDER)
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.status or not any([args.detect_only, args.backfill, args.audit]):
        result = gap.status_payload()
    elif args.audit:
        result = gap.run_audit(persist=not args.dry_run)
    else:
        gaps = gap.detect_gaps(
            streams=args.stream,
            start_override=args.start_utc,
            end_override=args.end_utc,
            detection_reason="MANUAL_CLI",
        )
        if args.detect_only or args.dry_run:
            result = {
                "verdict": "GAP_DETECT_DRY_RUN" if args.dry_run else "GAP_DETECTED",
                "dry_run": bool(args.dry_run),
                "mutated": False,
                "detected_gap_count": len(gaps),
                "gaps": gaps,
                "private_endpoint_calls": 0,
                "order_endpoint_calls": 0,
                "production_ready": False,
                "promotion_ready": False,
            }
        else:
            result = gap.run_backfill(gaps)
    print(dump(result) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
