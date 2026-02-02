#!/usr/bin/env python3
"""
Guard v2 실전 품질 검증/분해/관측 실행 스크립트
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_guard_v2_reports import (
    extract_guard_metrics_from_log,
    generate_scale_distribution_report,
    generate_scale_trade_link_report,
    generate_overtrading_check_report,
    generate_final_observation_report,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.ml_backtest_types import BacktestResult


def run_backtest(period_name: str, start_date: str, end_date: str) -> dict:
    """백테스트 실행 및 결과 추출"""
    cmd = [
        sys.executable, "-m", "src.backtest.run_ml_xgb_backtest",
        "--strategy", "ml_tcn",
        "--symbol", "BTCUSDT",
        "--timeframe", "5m",
        "--direction", "long",
        "--start-date", start_date,
        "--end-date", end_date,
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
    
    print(f"[{period_name}] 백테스트 실행 중... ({start_date} ~ {end_date})")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    # 결과를 JSON으로 파싱 (stdout에서 추출)
    # 실제로는 BacktestResult 객체를 받아야 하지만, 여기서는 로그에서 추출
    log_file = Path(f"/tmp/guard_v2_observation_{period_name}.log")
    with open(log_file, "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)
    
    # 로그에서 메트릭 추출
    metrics = extract_guard_metrics_from_log(log_file)
    
    return metrics, log_file


def main():
    """메인 실행"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/backtest_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 단기 구간 로그 파일 사용 (이미 실행됨)
    print("단기 구간 로그에서 메트릭 추출 중...")
    short_log = Path("/tmp/guard_v2_observation_short.log")
    short_term_metrics = extract_guard_metrics_from_log(short_log) if short_log.exists() else {}
    
    # 장기 구간 로그 파일 사용 (이미 실행됨)
    print("장기 구간 로그에서 메트릭 추출 중...")
    long_log = Path("/tmp/guard_v2_observation_long.log")
    long_term_metrics = extract_guard_metrics_from_log(long_log) if long_log.exists() else {}
    
    # 리포트 생성
    print("리포트 생성 중...")
    generate_scale_distribution_report(short_term_metrics, output_dir, "short_term")
    generate_scale_distribution_report(long_term_metrics, output_dir, "long_term")
    
    generate_scale_trade_link_report(short_term_metrics, output_dir, "short_term")
    generate_scale_trade_link_report(long_term_metrics, output_dir, "long_term")
    
    generate_overtrading_check_report(short_term_metrics, short_log, output_dir, "short_term")
    generate_overtrading_check_report(long_term_metrics, long_log, output_dir, "long_term")
    
    generate_final_observation_report(short_term_metrics, long_term_metrics, output_dir)
    
    print(f"모든 리포트 생성 완료: {output_dir}")


if __name__ == "__main__":
    main()
