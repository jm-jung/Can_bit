#!/usr/bin/env python3
"""CLI: Observation Quality Suite (taker forensics + daily rollup + readiness)."""
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
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--full-audit", action="store_true")
    g.add_argument("--daily-update", action="store_true")
    g.add_argument("--readiness", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--refresh-self-heal-metrics", action="store_true")
    p.add_argument("--refresh-taker-forensics", action="store_true")
    p.add_argument("--json", action="store_true", help="Print JSON to stdout")
    p.add_argument(
        "--compact",
        action="store_true",
        help="With --status: emit compact summary JSON and write compact report files",
    )
    args = p.parse_args()

    if args.full_audit:
        result = oq.run_full_audit()
    elif args.daily_update:
        result = oq.run_daily_update(refresh_taker=args.refresh_taker_forensics)
    elif args.readiness:
        result = oq.run_readiness(persist=True)
    elif args.refresh_self_heal_metrics:
        result = oq.refresh_self_heal_metrics_history(persist=True)
    elif args.status and args.compact:
        result = oq.compact_status(persist=True)
    else:
        result = oq.suite_status()

    if args.json or (args.status and args.compact):
        print(json.dumps(oq.clean(result), ensure_ascii=False, indent=2, default=str))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
