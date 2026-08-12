#!/usr/bin/env python3
"""
B_with_meta 일일 리포트 생성 후 Discord 웹훅으로 전송.

사용:
  python scripts/send_b_with_meta_daily_discord.py
  python scripts/send_b_with_meta_daily_discord.py --dry-run
  python scripts/send_b_with_meta_daily_discord.py --no-send   # 생성만

cron 예: 매일 09:00 KST → 0 0 * * * (UTC 00:00) 또는 0 9 * * * (서버가 KST면)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.meta_daily_report import generate_daily_report, FR2_DIR
from src.monitoring.notify_discord import send_b_with_meta_daily_report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="B_with_meta 일일 리포트 생성 후 Discord 전송")
    parser.add_argument("--dry-run", action="store_true", help="전송하지 않고 payload만 출력")
    parser.add_argument("--no-send", action="store_true", help="리포트만 생성, Discord 전송 안 함")
    args = parser.parse_args()

    md_path, json_path = generate_daily_report(output_dir=FR2_DIR, write_json=True)
    if md_path is None:
        print("[ERROR] Failed to generate report.", file=sys.stderr)
        return 1
    print(f"Report: {md_path}")

    if args.no_send:
        return 0

    send_b_with_meta_daily_report(
        daily_report_md_path=str(md_path),
        summary_json_path=str(json_path) if json_path else None,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print("Discord send requested.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
