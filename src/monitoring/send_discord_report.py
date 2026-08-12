"""
Discord 리포트 전송 CLI (주간 리포트 기반 Decision 포함).
사용: python -m src.monitoring.send_discord_report --days 7 --dry-run
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.monitoring.generate_weekly_report import generate_weekly_report
from src.monitoring.notify_discord import send_daily_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Discord 일일 리포트 전송 (주간 Decision 포함)")
    parser.add_argument("--days", type=int, default=7, help="주간 리포트 기간 (기본 7일)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="전송하지 않고 payload만 콘솔에 출력",
    )
    args = parser.parse_args()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    report_path = generate_weekly_report(
        start_date=start_date,
        end_date=end_date,
        decision_thresholds=None,
    )
    if report_path is None:
        weekly_dir = PROJECT_ROOT / "data" / "monitoring_reports"
        if not weekly_dir.exists():
            print("[send_discord_report] 주간 리포트를 생성할 수 없습니다.", file=sys.stderr)
            sys.exit(1)
        json_files = sorted(weekly_dir.glob("weekly_report_*.json"), reverse=True)
        if not json_files:
            print("[send_discord_report] 주간 리포트 JSON이 없습니다.", file=sys.stderr)
            sys.exit(1)
        weekly_report_path = str(json_files[0])
    else:
        weekly_report_path = str(Path(report_path).with_suffix(".json"))

    summary_path = None
    monitoring_dir = PROJECT_ROOT / "data" / "monitoring"
    if monitoring_dir.exists():
        summaries = sorted(monitoring_dir.glob("monitor_guard_stage2_summary_*.json"), reverse=True)
        if summaries:
            summary_path = str(summaries[0])

    ohlcv_info = {"symbol": "BTCUSDT", "timeframe": "5m", "latest_ts": "N/A", "new_candles": 0}
    try:
        ohlcv_path = PROJECT_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv"
        if ohlcv_path.exists():
            with open(ohlcv_path, "r", encoding="utf-8") as f:
                line = f.readline()
                if line:
                    headers = line.strip().split(",")
                    last_line = None
                    for last_line in f:
                        pass
                    if "timestamp" in headers and last_line:
                        ohlcv_info["latest_ts"] = last_line.strip().split(",")[0]
    except Exception:
        pass

    cache_path = str(PROJECT_ROOT / "data" / "cache" / "ml_predictions" / "ml_tcn_BTCUSDT_5m_proba.parquet")
    if not Path(cache_path).exists():
        cache_path = None

    send_daily_report(
        summary_json_path=summary_path,
        ohlcv_info=ohlcv_info,
        weekly_report_path=weekly_report_path,
        pipeline_status="정상 완료",
        cache_path=cache_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
