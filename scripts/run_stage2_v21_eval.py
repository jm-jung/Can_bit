#!/usr/bin/env python3
"""
Stage-2 v2.1 (연속 스코어) 평가 스크립트

3-run 비교:
1. Guard v2 ON + Stage-2 OFF (baseline)
2. Guard v2 ON + Stage-2 ON (v2.0, 계단형)
3. Guard v2 ON + Stage-2 ON (v2.1, 연속형) - 현재 구현
"""

import json
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments" / "stage2_v21_eval"
DOCS_DIR = PROJECT_ROOT / "docs"

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
    "--use-strategy-guard",
    "--strategy-guard-v2",
    "--strategy-guard-v2-mode", "soft",
    "--strategy-guard-v2-scale-floor", "0.02",
    "--no-save",
]

def run_backtest(extra_args: list, log_file: Path) -> dict:
    """백테스트 실행 및 결과 추출"""
    cmd = [
        sys.executable, "-m", "src.backtest.run_ml_xgb_backtest"
    ] + COMMON_ARGS + extra_args
    
    print(f"실행: {' '.join(cmd)}")
    
    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_ROOT
        )
    
    # 로그에서 메트릭 추출
    metrics = extract_metrics(log_file)
    metrics["command"] = " ".join(cmd)
    
    return metrics

def extract_metrics(log_file: Path) -> dict:
    """로그에서 메트릭 추출"""
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
    
    # Stage-2 score 통계 추출 (v2.1)
    score_values = []
    scale_values = []
    
    for line in content.split("\n"):
        if "[STAGE2][SCORE]" in line:
            # score=0.xxx 추출
            score_match = re.search(r"score=([\d.]+)", line)
            if score_match:
                try:
                    score_values.append(float(score_match.group(1)))
                except ValueError:
                    pass
            
            # final_scale=0.xxx 추출
            scale_match = re.search(r"final_scale=([\d.]+)", line)
            if scale_match:
                try:
                    scale_values.append(float(scale_match.group(1)))
                except ValueError:
                    pass
    
    if score_values:
        score_values.sort()
        n = len(score_values)
        metrics["stage2_score_stats"] = {
            "count": n,
            "min": min(score_values),
            "median": score_values[int(n * 0.5)] if n > 0 else 0,
            "p90": score_values[int(n * 0.9)] if n > 0 else 0,
            "max": max(score_values),
            "pct_at_floor": sum(1 for s in score_values if abs(s - 0.2) < 0.01) / n * 100 if n > 0 else 0,
        }
    
    if scale_values:
        scale_values.sort()
        n = len(scale_values)
        metrics["final_scale_stats"] = {
            "count": n,
            "min": min(scale_values),
            "median": scale_values[int(n * 0.5)] if n > 0 else 0,
            "p90": scale_values[int(n * 0.9)] if n > 0 else 0,
            "max": max(scale_values),
        }
    
    return metrics

def main():
    """메인 실행 함수"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"실험 실행 ID: {run_id}")
    
    exp_dir = EXPERIMENTS_DIR / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run 1: Guard v2 ON + Stage-2 OFF
    print("\n" + "="*80)
    print("Run 1: Guard v2 ON + Stage-2 OFF (baseline)")
    print("="*80)
    log1 = exp_dir / "run1_guard_v2_stage2_off.log"
    results["run1_baseline"] = run_backtest([], log1)
    
    # Run 2: Guard v2 ON + Stage-2 ON (v2.0 - 계단형)
    # v2.0은 현재 코드에서 base=1.0/0.2만 사용하는 버전
    # 실제로는 v2.1이 기본이지만, 비교를 위해 v2.0 결과는 이전 실행 결과를 참고
    print("\n" + "="*80)
    print("Run 2: Guard v2 ON + Stage-2 ON (v2.0, 계단형)")
    print("="*80)
    print("참고: v2.0 결과는 이전 실행 결과를 참고하거나 별도 실행 필요")
    # results["run2_v20"] = run_backtest(["--use-stage2"], log2)
    
    # Run 3: Guard v2 ON + Stage-2 ON (v2.1 - 연속형, 현재 구현)
    print("\n" + "="*80)
    print("Run 3: Guard v2 ON + Stage-2 ON (v2.1, 연속형)")
    print("="*80)
    log3 = exp_dir / "run3_guard_v2_stage2_on_v21.log"
    results["run3_v21"] = run_backtest(["--use-stage2"], log3)
    
    # 결과 저장
    summary_file = exp_dir / "results.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"모든 실험 완료")
    print(f"결과 디렉토리: {exp_dir}")
    print(f"요약 파일: {summary_file}")
    print(f"{'='*80}")
    
    return run_id

if __name__ == "__main__":
    run_id = main()
