#!/usr/bin/env python3
"""
Stage-2 CAP 값별 성적 분해 리포트 생성 스크립트

단기/장기 구간에서 CAP 값별(1.0/0.8/0.6) 성적을 분해하여 리포트를 생성합니다.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 공통 파라미터
COMMON_ARGS = [
    "--strategy", "ml_tcn",
    "--symbol", "BTCUSDT",
    "--timeframe", "5m",
    "--direction", "long",
    "--use-optimized-threshold",
    "--signal-confirmation-bars", "1",
    "--min-hold-bars", "12",
    "--cooldown-bars", "12",
    "--use-strategy-guard",
    "--strategy-guard-v2",
    "--strategy-guard-v2-mode", "soft",
    "--strategy-guard-v2-scale-floor", "0.02",
    "--use-stage2",
    "--no-save",
]


def run_backtest(period_name: str, start_date: str, end_date: str) -> dict:
    """백테스트 실행 및 결과 추출"""
    cmd = [
        sys.executable, "-m", "src.backtest.run_ml_xgb_backtest"
    ] + COMMON_ARGS + [
        "--start-date", start_date,
        "--end-date", end_date,
    ]
    
    logger.info(f"[{period_name}] 백테스트 실행 중... ({start_date} ~ {end_date})")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    
    if result.returncode != 0:
        logger.error(f"[{period_name}] 백테스트 실패: {result.stderr[:500]}")
        return {
            "period": period_name,
            "error": result.stderr[:500],
            "returncode": result.returncode,
        }
    
    # 로그 파싱
    log_output = result.stdout + result.stderr
    metrics = parse_backtest_log(log_output, period_name, start_date, end_date)
    
    return metrics


def parse_backtest_log(log_output: str, period_name: str, start_date: str, end_date: str) -> dict:
    """백테스트 로그에서 메트릭을 추출합니다."""
    metrics = {
        "period": period_name,
        "start_date": start_date,
        "end_date": end_date,
        "total_return": None,
        "max_drawdown": None,
        "total_trades": None,
        "win_rate": None,
        "cap_breakdown": {},
    }
    
    # Backtest Summary 파싱
    summary_match = re.search(
        r"Total Return: ([\d.-]+)%\s+"
        r"Win Rate: ([\d.-]+)%\s+"
        r"Max Drawdown: ([\d.-]+)%\s+"
        r"Total Trades: (\d+)",
        log_output,
    )
    if summary_match:
        metrics["total_return"] = float(summary_match.group(1))
        metrics["win_rate"] = float(summary_match.group(2))
        metrics["max_drawdown"] = float(summary_match.group(3))
        metrics["total_trades"] = int(summary_match.group(4))
    
    # CAP breakdown 파싱
    breakdown_match = re.search(
        r"\[요청 4\] CAP 값별 성적 분해.*?"
        r"cap=1\.0: trades=(\d+), win_rate=([\d.]+)%, mean_profit=([\d.-]+), "
        r"median_profit=([\d.-]+), mean_holding=([\d.]+), total_contribution=([\d.-]+).*?"
        r"cap=0\.8: trades=(\d+), win_rate=([\d.]+)%, mean_profit=([\d.-]+), "
        r"median_profit=([\d.-]+), mean_holding=([\d.]+), total_contribution=([\d.-]+).*?"
        r"cap=0\.6: trades=(\d+), win_rate=([\d.]+)%, mean_profit=([\d.-]+), "
        r"median_profit=([\d.-]+), mean_holding=([\d.]+), total_contribution=([\d.-]+)",
        log_output,
        re.DOTALL,
    )
    
    if breakdown_match:
        # cap=1.0
        metrics["cap_breakdown"][1.0] = {
            "trade_count": int(breakdown_match.group(1)),
            "win_rate": float(breakdown_match.group(2)) / 100,
            "mean_profit": float(breakdown_match.group(3)),
            "median_profit": float(breakdown_match.group(4)),
            "mean_holding_bars": float(breakdown_match.group(5)),
            "total_contribution": float(breakdown_match.group(6)),
        }
        # cap=0.8
        metrics["cap_breakdown"][0.8] = {
            "trade_count": int(breakdown_match.group(7)),
            "win_rate": float(breakdown_match.group(8)) / 100,
            "mean_profit": float(breakdown_match.group(9)),
            "median_profit": float(breakdown_match.group(10)),
            "mean_holding_bars": float(breakdown_match.group(11)),
            "total_contribution": float(breakdown_match.group(12)),
        }
        # cap=0.6
        metrics["cap_breakdown"][0.6] = {
            "trade_count": int(breakdown_match.group(13)),
            "win_rate": float(breakdown_match.group(14)) / 100,
            "mean_profit": float(breakdown_match.group(15)),
            "median_profit": float(breakdown_match.group(16)),
            "mean_holding_bars": float(breakdown_match.group(17)),
            "total_contribution": float(breakdown_match.group(18)),
        }
    else:
        # 개별 파싱 시도
        for cap_bucket in [1.0, 0.8, 0.6]:
            bucket_match = re.search(
                rf"cap={cap_bucket}: trades=(\d+), win_rate=([\d.]+)%, "
                rf"mean_profit=([\d.-]+), median_profit=([\d.-]+), "
                rf"mean_holding=([\d.]+), total_contribution=([\d.-]+)",
                log_output,
            )
            if bucket_match:
                metrics["cap_breakdown"][cap_bucket] = {
                    "trade_count": int(bucket_match.group(1)),
                    "win_rate": float(bucket_match.group(2)) / 100,
                    "mean_profit": float(bucket_match.group(3)),
                    "median_profit": float(bucket_match.group(4)),
                    "mean_holding_bars": float(bucket_match.group(5)),
                    "total_contribution": float(bucket_match.group(6)),
                }
            else:
                # "해당 구간에 발생하지 않음" 체크
                no_occurrence_match = re.search(
                    rf"cap={cap_bucket}: 해당 구간에 발생하지 않음",
                    log_output,
                )
                if no_occurrence_match:
                    metrics["cap_breakdown"][cap_bucket] = None
    
    return metrics


def generate_report(short_term: dict, long_term: dict, report_path: Path):
    """분해 리포트 생성"""
    with open(report_path, "w") as f:
        f.write("# Stage-2 CAP 값별 성적 분해 리포트\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 단기 구간
        f.write("## 단기 구간 (2025-01-01 ~ 2025-03-01)\n\n")
        f.write("| CAP 값 | Trade Count | Win Rate | Mean Profit | Median Profit | Mean Holding | Total Contribution |\n")
        f.write("|--------|-------------|----------|-------------|---------------|--------------|---------------------|\n")
        
        for cap_bucket in [1.0, 0.8, 0.6]:
            stats = short_term.get("cap_breakdown", {}).get(cap_bucket)
            if stats:
                f.write(
                    f"| {cap_bucket} | {stats['trade_count']} | {stats['win_rate']:.2%} | "
                    f"{stats['mean_profit']:.6f} | {stats['median_profit']:.6f} | "
                    f"{stats['mean_holding_bars']:.1f} | {stats['total_contribution']:.6f} |\n"
                )
            else:
                f.write(f"| {cap_bucket} | 0 | N/A | N/A | N/A | N/A | N/A |\n")
        
        f.write("\n## 장기 구간 (2023-01-01 ~ 2024-12-31)\n\n")
        f.write("| CAP 값 | Trade Count | Win Rate | Mean Profit | Median Profit | Mean Holding | Total Contribution |\n")
        f.write("|--------|-------------|----------|-------------|---------------|--------------|---------------------|\n")
        
        for cap_bucket in [1.0, 0.8, 0.6]:
            stats = long_term.get("cap_breakdown", {}).get(cap_bucket)
            if stats:
                f.write(
                    f"| {cap_bucket} | {stats['trade_count']} | {stats['win_rate']:.2%} | "
                    f"{stats['mean_profit']:.6f} | {stats['median_profit']:.6f} | "
                    f"{stats['mean_holding_bars']:.1f} | {stats['total_contribution']:.6f} |\n"
                )
            else:
                f.write(f"| {cap_bucket} | 0 | N/A | N/A | N/A | N/A | N/A |\n")
        
        f.write("\n## 결론\n\n")
        f.write("CAP 값별 성적 분해 결과를 통해 각 CAP 그룹의 기여도를 확인할 수 있습니다.\n")


def main():
    """메인 실행"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "data" / "experiments" / f"stage2_cap_bucket_breakdown_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 단기 구간 실행
    logger.info("단기 구간 실행 중...")
    short_term = run_backtest("short_term", "2025-01-01", "2025-03-01")
    
    # 장기 구간 실행
    logger.info("장기 구간 실행 중...")
    long_term = run_backtest("long_term", "2023-01-01", "2024-12-31")
    
    # JSON 저장
    breakdown_data = {
        "short_term": short_term,
        "long_term": long_term,
    }
    breakdown_file = results_dir / "breakdown.json"
    with open(breakdown_file, "w") as f:
        json.dump(breakdown_data, f, indent=2)
    
    # 리포트 생성
    report_path = PROJECT_ROOT / "docs" / "stage2_cap_bucket_breakdown.md"
    generate_report(short_term, long_term, report_path)
    
    logger.info(f"분해 리포트 생성 완료. 결과: {results_dir}")
    logger.info(f"리포트: {report_path}")


if __name__ == "__main__":
    main()
