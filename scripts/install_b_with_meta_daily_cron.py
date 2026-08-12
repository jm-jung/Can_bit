#!/usr/bin/env python3
"""
Install cron entry for B_with_meta daily report at 12:00 KST (local time).

Usage:
  python scripts/install_b_with_meta_daily_cron.py --dry-run
  python scripts/install_b_with_meta_daily_cron.py --install

Notes:
- Cron runs with a minimal environment; we call a wrapper shell script that sources .env.
- This script does NOT require the webhook URL as an argument; it uses existing .env.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _read_crontab() -> str:
    try:
        p = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
        # When no crontab exists, some systems return exit code 1 and message.
        return p.stdout if p.stdout else ""
    except Exception:
        return ""


def _write_crontab(content: str) -> None:
    subprocess.run(["crontab", "-"], input=content, text=True, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install cron for B_with_meta daily Discord report (12:00 KST).")
    parser.add_argument("--install", action="store_true", help="Actually install into crontab")
    parser.add_argument("--dry-run", action="store_true", help="Print the cron line and resulting crontab only")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    runner = project_root / "scripts" / "run_b_with_meta_daily_report_noon_kst.sh"
    log_path = project_root / "logs" / "daily_b_with_meta_report_cron.log"
    runner_cmd = f'{runner} >> "{log_path}" 2>&1'

    marker = "# can_bit: b_with_meta daily report (12:00 KST)"
    cron_line = f"0 12 * * * {runner_cmd}  {marker}"

    existing = _read_crontab().splitlines()
    # Remove previous lines we installed (idempotent)
    cleaned = [ln for ln in existing if marker not in ln]
    if cleaned and cleaned[-1].strip() != "":
        cleaned.append("")
    cleaned.append(cron_line)
    cleaned.append("")

    new_content = "\n".join(cleaned)

    if args.dry_run and not args.install:
        print("### Cron line to be installed")
        print(cron_line)
        print("\n### New crontab content")
        print(new_content)
        return 0

    if not args.install:
        print("Refusing to modify crontab without --install. Use --dry-run to preview.")
        return 2

    # Ensure logs dir exists
    (project_root / "logs").mkdir(parents=True, exist_ok=True)
    # Make runner executable if possible
    try:
        os.chmod(runner, 0o755)
    except Exception:
        pass

    _write_crontab(new_content)
    print("Installed cron entry:")
    print(cron_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

