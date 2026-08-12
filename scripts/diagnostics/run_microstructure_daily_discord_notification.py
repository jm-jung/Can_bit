#!/usr/bin/env python3
"""CLI for microstructure daily observation-quality Discord notifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/diagnostics"))

import microstructure_daily_discord_notification as n


def main() -> int:
    p = argparse.ArgumentParser(description="CAN_BIT observation quality daily Discord notifier")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--send-test", action="store_true")
    g.add_argument("--retry-outbox", action="store_true")
    p.add_argument("--force", action="store_true", help="Allow resend for same target day")
    p.add_argument("--skip-daily-update", action="store_true", help="Use existing compact only (diagnostics)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--secret-path", default=str(n.DEFAULT_SECRET))
    args = p.parse_args()

    n.ensure_dirs()
    secret_path = Path(args.secret_path)

    if args.status:
        result = n.status_payload()
    elif args.retry_outbox:
        url, meta = n.load_webhook_url(secret_path)
        result = {"verdict": "OUTBOX_RETRY", "secret_present": meta.get("present"), **n.retry_outbox(url)}
    elif args.send_test:
        result = n.run_notification(send_test=True, skip_daily_update=True, secret_path=secret_path)
    elif args.dry_run:
        result = n.run_notification(dry_run=True, force=args.force, skip_daily_update=args.skip_daily_update, secret_path=secret_path)
    else:
        result = n.run_notification(dry_run=False, force=args.force, skip_daily_update=args.skip_daily_update, secret_path=secret_path)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        for k in (
            "verdict",
            "notifier_status",
            "target_day_utc",
            "operational_severity",
            "readiness",
            "discord_success",
            "http_status",
            "duplicate_send_prevented",
            "pending_outbox_count",
        ):
            if k in result:
                print(f"{k}={result[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
