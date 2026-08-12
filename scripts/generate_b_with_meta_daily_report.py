#!/usr/bin/env python3
"""
B_with_meta 일일 운영 리포트 생성 CLI.

사용:
  python scripts/generate_b_with_meta_daily_report.py
  python -m scripts.generate_b_with_meta_daily_report
  python scripts/generate_b_with_meta_daily_report.py --date 2026-03-18
  python scripts/generate_b_with_meta_daily_report.py --no-json

출력:
  data/diagnostics/fr2/B_WITH_META_DAILY_REPORT_YYYYMMDD.md
  data/diagnostics/fr2/b_with_meta_daily_summary_YYYYMMDD.json (기본)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# 프로젝트 루트를 path에 넣어서 src 임포트 가능하게
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.meta_daily_report import generate_daily_report, FR2_DIR


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B_with_meta 일일 운영 리포트 생성 (state_log, snapshot, trade/risk 로그 기반)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="리포트 기준일 YYYY-MM-DD (기본: 오늘 UTC)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"출력 디렉터리 (기본: {FR2_DIR})",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="JSON 요약 파일 생성 안 함",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=24.0,
        help="집계 구간(시간) (기본: 24)",
    )
    args = parser.parse_args()

    report_date = None
    if args.date:
        try:
            report_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"[ERROR] Invalid --date: {args.date}. Use YYYY-MM-DD.", file=sys.stderr)
            return 1

    output_dir = Path(args.output_dir) if args.output_dir else None
    md_path, json_path = generate_daily_report(
        report_date=report_date,
        output_dir=output_dir,
        write_json=not args.no_json,
        lookback_hours=args.lookback_hours,
    )

    if md_path is None:
        print("[ERROR] Failed to write report.", file=sys.stderr)
        return 1

    print(f"Report written: {md_path}")
    if json_path:
        print(f"Summary JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
