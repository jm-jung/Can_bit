#!/usr/bin/env python3
"""CLI: Daily Observation Quality Rollup (diagnostics-only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

import microstructure_observation_quality as oq  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--latest", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--audit", action="store_true")
    g.add_argument("--backfill-history", action="store_true")
    p.add_argument("--date")
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--from-t0", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.status:
        result = oq.daily_status()
    elif args.audit:
        result = oq.daily_audit()
    elif args.backfill_history or args.from_t0:
        result = oq.run_daily_rollup(from_t0=True)
    elif args.date:
        result = oq.run_daily_rollup(start_date=args.date, end_date=args.date)
    elif args.start_date or args.end_date:
        result = oq.run_daily_rollup(start_date=args.start_date, end_date=args.end_date or args.start_date)
    else:
        result = oq.run_daily_rollup(latest_only=True)

    if args.json:
        print(json.dumps(oq.clean(result), ensure_ascii=False, indent=2, default=str))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
