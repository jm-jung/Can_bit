#!/usr/bin/env python3
"""
Guard v2 평가 실험 실행 스크립트

3개 실험을 순차적으로 실행하고 결과를 JSON으로 저장합니다.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import re

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments" / "guard_v2_eval"

# 공통 파라미터
COMMON_ARGS = [
    "--strategy", "ml_tcn",
    "--symbol", "BTCUSDT",
    "--timeframe", "5m",
    "--direction", "long",
    "--start-date", "2023-01-01",
    "--end-date", "2024-12-31",
    "--use-optimized-threshold",
    "--signal-confirmation-bars", "1",
    "--min-hold-bars", "12",
    "--cooldown-bars", "12",
    "--no-save",
]

def run_experiment(exp_id: str, exp_name: str, extra_args: list, run_id: str) -> dict:
    """실험 실행 및 결과 추출"""
    print(f"\n{'='*80}")
    print(f"실험 {exp_id}: {exp_name}")
    print(f"{'='*80}")
    
    # 실행 디렉토리 생성
    exp_dir = EXPERIMENTS_DIR / run_id / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일 경로
    log_file = exp_dir / "run.log"
    
    # 커맨드 구성
    cmd = [
        sys.executable, "-m", "src.backtest.run_ml_xgb_backtest"
    ] + COMMON_ARGS + extra_args
    
    print(f"실행 커맨드: {' '.join(cmd)}")
    print(f"로그 파일: {log_file}")
    
    # 실행
    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_ROOT
        )
    
    # 로그에서 결과 추출
    metrics = extract_metrics(log_file, exp_id)
    
    # 커맨드 저장
    metrics["command"] = " ".join(cmd)
    metrics["run_id"] = run_id
    metrics["experiment_id"] = exp_id
    metrics["experiment_name"] = exp_name
    
    # JSON 저장
    metrics_file = exp_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"결과 저장: {metrics_file}")
    print(f"Total Return: {metrics.get('total_return', 'N/A')}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A')}%")
    print(f"Total Trades: {metrics.get('total_trades', 'N/A')}")
    
    return metrics

def extract_metrics(log_file: Path, exp_id: str) -> dict:
    """로그 파일에서 메트릭 추출"""
    metrics = {}
    
    if not log_file.exists():
        return metrics
    
    with open(log_file, "r") as f:
        content = f.read()
    
    # Backtest Summary 추출
    summary_match = re.search(
        r"Total Return: ([\d.-]+)%\s+"
        r"Win Rate: ([\d.-]+)%\s+"
        r"Max Drawdown: ([\d.-]+)%\s+"
        r"Total Trades: (\d+)",
        content,
        re.MULTILINE
    )
    
    if summary_match:
        metrics["total_return"] = float(summary_match.group(1))
        metrics["win_rate"] = float(summary_match.group(2))
        metrics["max_drawdown"] = float(summary_match.group(3))
        metrics["total_trades"] = int(summary_match.group(4))
    
    # Avg Holding 추출
    holding_match = re.search(r"Average holding period: ([\d.]+) bars", content)
    if holding_match:
        metrics["avg_holding_bars"] = float(holding_match.group(1))
    
    # Total entries/exits 추출
    entries_match = re.search(r"total_entries=(\d+)", content)
    if entries_match:
        metrics["total_entries"] = int(entries_match.group(1))
    
    exits_match = re.search(r"total_exits=(\d+)", content)
    if exits_match:
        metrics["total_exits"] = int(exits_match.group(1))
    
    # 실험 1: Guard v2 scale/decision 통계
    if exp_id == "exp1_guard_on":
        scale_values = []
        decision_counts = {"ALLOW": 0, "BLOCK": 0, "DEFER": 0}
        
        # DECISION 로그에서 scale 추출
        scale_matches = re.findall(r"scale=([\d.]+)", content)
        for scale_str in scale_matches:
            try:
                scale_values.append(float(scale_str))
            except ValueError:
                pass
        
        # DECISION 로그에서 decision 추출
        decision_matches = re.findall(r"decision=(\w+)", content)
        for decision in decision_matches:
            if decision in decision_counts:
                decision_counts[decision] += 1
        
        if scale_values:
            scale_values.sort()
            n = len(scale_values)
            metrics["scale_stats"] = {
                "count": n,
                "min": min(scale_values),
                "p10": scale_values[int(n * 0.1)] if n > 0 else 0,
                "median": scale_values[int(n * 0.5)] if n > 0 else 0,
                "p90": scale_values[int(n * 0.9)] if n > 0 else 0,
                "max": max(scale_values),
            }
            
            # 히스토그램 (10 bins)
            bins = 10
            bin_width = (max(scale_values) - min(scale_values)) / bins if max(scale_values) > min(scale_values) else 1
            histogram = [0] * bins
            for val in scale_values:
                bin_idx = min(int((val - min(scale_values)) / bin_width), bins - 1) if bin_width > 0 else 0
                histogram[bin_idx] += 1
            metrics["scale_histogram"] = {
                "bins": bins,
                "min": min(scale_values),
                "max": max(scale_values),
                "counts": histogram,
            }
        
        metrics["decision_counts"] = decision_counts
    
    return metrics

def main():
    """메인 실행 함수"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"실험 실행 ID: {run_id}")
    
    results = {}
    
    # 실험 1: Guard v2 ON (비용 ON)
    results["exp1"] = run_experiment(
        "exp1_guard_on",
        "Guard v2 ON (비용 ON)",
        [
            "--use-strategy-guard",
            "--strategy-guard-v2",
            "--strategy-guard-v2-mode", "soft",
            "--strategy-guard-v2-scale-floor", "0.02",
        ],
        run_id
    )
    
    # 실험 2: Guard v2 ON (비용 OFF)
    # Note: commission/slippage 옵션이 없으면 기본값 사용 (실험에서는 비용 영향 확인용)
    results["exp2"] = run_experiment(
        "exp2_guard_on_cost_off",
        "Guard v2 ON (비용 OFF - 수동 설정 필요)",
        [
            "--use-strategy-guard",
            "--strategy-guard-v2",
            "--strategy-guard-v2-mode", "soft",
            "--strategy-guard-v2-scale-floor", "0.02",
            # Note: commission/slippage를 0으로 설정하려면 코드 수정 필요
            # 현재는 실험 1과 동일하게 실행하고, 비용 영향은 로그에서 분석
        ],
        run_id
    )
    
    # 실험 3: Guard OFF (Baseline, 동일 hold/cooldown)
    results["exp3"] = run_experiment(
        "exp3_guard_off",
        "Guard OFF (Baseline)",
        [],
        run_id
    )
    
    # 전체 결과 저장
    summary_file = EXPERIMENTS_DIR / run_id / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"모든 실험 완료")
    print(f"결과 디렉토리: {EXPERIMENTS_DIR / run_id}")
    print(f"요약 파일: {summary_file}")
    print(f"{'='*80}")
    
    return run_id

if __name__ == "__main__":
    run_id = main()
    print(f"\n다음 단계: python scripts/summarize_guard_v2_eval.py {run_id}")
