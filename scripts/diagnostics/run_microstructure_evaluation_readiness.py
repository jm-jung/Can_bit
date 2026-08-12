#!/usr/bin/env python3
"""CLI: Evaluation Readiness Gate (diagnostics-only, read-only)."""
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
    g.add_argument("--status", action="store_true")
    g.add_argument("--audit", action="store_true")
    g.add_argument("--explain", action="store_true")
    g.add_argument("--write-report", action="store_true")
    p.add_argument("--as-of-utc")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    as_of = oq.parse_ts(args.as_of_utc) if args.as_of_utc else None
    result = oq.run_readiness(as_of=as_of, persist=bool(args.write_report or args.status or args.explain or args.audit or True))

    if args.explain:
        out = {
            "verdict": result.get("verdict"),
            "blockers": result.get("blockers"),
            "warnings": result.get("warnings"),
            "conditions": result.get("conditions"),
            "estimate": result.get("estimate"),
            "composition": result.get("marker_composition_pct"),
            "week_distribution": result.get("week_distribution"),
        }
    elif args.audit:
        out = {
            "verdict": result.get("verdict"),
            "integrity": [c for c in result.get("conditions", []) if c.get("kind") == "integrity"],
            "ops": [c for c in result.get("conditions", []) if c.get("kind") == "ops"],
            "taker_forensics_verdict": result.get("taker_forensics_verdict"),
            "production_ready": False,
            "promotion_ready": False,
        }
    else:
        out = result

    if args.json:
        print(json.dumps(oq.clean(out), ensure_ascii=False, indent=2, default=str))
    else:
        print(out.get("verdict"), out.get("blockers"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
