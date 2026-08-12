#!/usr/bin/env python3
"""
FR2 rolling metrics refresh cron 설치기.

기본:
- 매시간(1시간 간격) 갱신
- 서버 로컬 타임존 기준 cron schedule 적용
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _read_crontab() -> str:
    p = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    return p.stdout or ""


def _write_crontab(content: str) -> None:
    subprocess.run(["crontab", "-"], input=content, text=True, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install cron for FR2 meta rolling metrics refresh.")
    parser.add_argument("--install", action="store_true", help="Actually install into crontab")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    parser.add_argument("--minute", type=int, default=5, help="Run minute (0-59). Default 5.")
    parser.add_argument("--interval-hours", type=int, default=1, help="Interval hours. Default 1.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    wrapper = project_root / "scripts" / "run_fr2_meta_metrics_refresh.sh"
    log_path = project_root / "logs" / "fr2_meta_metrics_refresh_cron.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    marker = "# can_bit: fr2 meta metrics refresh cron"
    cron_line = f"{args.minute} */{args.interval_hours} * * * {wrapper} >> \"{log_path}\" 2>&1  {marker}"

    existing = _read_crontab().splitlines()
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

    # Ensure wrapper executable
    try:
        os.chmod(wrapper, 0o755)
    except Exception:
        pass

    _write_crontab(new_content)
    print("Installed cron entry:")
    print(cron_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

