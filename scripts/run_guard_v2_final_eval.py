#!/usr/bin/env python3
"""
Guard v2 실전 가능 여부 평가 통합 검증 루프

STEP 1: 거래 비용 OFF 실험
STEP 2: margin/entropy 미니 스윕 (12-run)
STEP 3: Stage-2/threshold 재검증 (조건부)
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import re

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments" / "guard_v2_final_eval"
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
    "--no-save",
]

# Guard v2 기본 파라미터
GUARD_V2_ARGS = [
    "--use-strategy-guard",
    "--strategy-guard-v2",
    "--strategy-guard-v2-mode", "soft",
    "--strategy-guard-v2-scale-floor", "0.02",
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
    
    # Scale 통계 (Guard v2가 있는 경우)
    scale_values = []
    scale_matches = re.findall(r"scale=([\d.]+)", content)
    for scale_str in scale_matches:
        try:
            scale_values.append(float(scale_str))
        except ValueError:
            pass
    
    if scale_values:
        scale_values.sort()
        n = len(scale_values)
        metrics["scale_stats"] = {
            "count": n,
            "min": min(scale_values),
            "median": scale_values[int(n * 0.5)] if n > 0 else 0,
            "p90": scale_values[int(n * 0.9)] if n > 0 else 0,
            "max": max(scale_values),
        }
    
    return metrics

def step1_cost_off(run_id: str) -> dict:
    """STEP 1: 거래 비용 OFF 실험"""
    print("\n" + "="*80)
    print("STEP 1: 거래 비용 OFF 실험")
    print("="*80)
    
    exp_dir = EXPERIMENTS_DIR / run_id / "step1"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Guard v2 ON, 비용 ON (기준)
    print("\n[1-1] Guard v2 ON, 비용 ON (기준)")
    log1 = exp_dir / "guard_on_cost_on.log"
    metrics1 = run_backtest(GUARD_V2_ARGS, log1)
    
    # Guard v2 ON, 비용 OFF
    print("\n[1-2] Guard v2 ON, 비용 OFF")
    log2 = exp_dir / "guard_on_cost_off.log"
    metrics2 = run_backtest(
        GUARD_V2_ARGS + ["--commission-rate", "0", "--slippage-rate", "0"],
        log2
    )
    
    results = {
        "cost_on": metrics1,
        "cost_off": metrics2,
    }
    
    # JSON 저장
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def step2_sweep(run_id: str) -> dict:
    """STEP 2: margin/entropy 미니 스윕 (12-run)"""
    print("\n" + "="*80)
    print("STEP 2: margin/entropy 미니 스윕 (12-run)")
    print("="*80)
    
    exp_dir = EXPERIMENTS_DIR / run_id / "step2"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    min_margins = [0.03, 0.05, 0.08]
    max_entropies = [0.60, 0.65, 0.70, 0.75]
    
    results = {}
    
    run_idx = 0
    for min_margin in min_margins:
        for max_entropy in max_entropies:
            run_idx += 1
            run_name = f"m{min_margin}_e{max_entropy}"
            
            print(f"\n[{run_idx}/12] min_margin={min_margin}, max_entropy={max_entropy}")
            
            extra_args = GUARD_V2_ARGS + [
                "--strategy-guard-v2-min-margin", str(min_margin),
                "--strategy-guard-v2-max-entropy", str(max_entropy),
            ]
            
            log_file = exp_dir / f"{run_name}.log"
            metrics = run_backtest(extra_args, log_file)
            metrics["min_margin"] = min_margin
            metrics["max_entropy"] = max_entropy
            
            results[run_name] = metrics
    
    # JSON 저장
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def step3_stage2(run_id: str, should_run: bool) -> dict:
    """STEP 3: Stage-2/threshold 재검증 (조건부)"""
    if not should_run:
        print("\n" + "="*80)
        print("STEP 3: Stage-2/threshold 재검증 (건너뜀)")
        print("="*80)
        print("조건 미충족: STEP 1 또는 STEP 2 결과가 양수이거나 개선됨")
        return {}
    
    print("\n" + "="*80)
    print("STEP 3: Stage-2/threshold 재검증")
    print("="*80)
    
    exp_dir = EXPERIMENTS_DIR / run_id / "step3"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Stage-2 OFF
    print("\n[3-1] Guard v2 ON, Stage-2 OFF")
    log1 = exp_dir / "stage2_off.log"
    metrics1 = run_backtest(GUARD_V2_ARGS, log1)
    results["stage2_off"] = metrics1
    
    # Stage-2 ON
    print("\n[3-2] Guard v2 ON, Stage-2 ON")
    log2 = exp_dir / "stage2_on.log"
    metrics2 = run_backtest(
        GUARD_V2_ARGS + ["--use-stage2"],
        log2
    )
    results["stage2_on"] = metrics2
    
    # Threshold 비교 (optimized vs default)
    print("\n[3-3] Guard v2 ON, threshold=0.5 (default)")
    log3 = exp_dir / "threshold_default.log"
    metrics3 = run_backtest(
        [arg for arg in GUARD_V2_ARGS if arg != "--use-optimized-threshold"],
        log3
    )
    results["threshold_default"] = metrics3
    
    # JSON 저장
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """메인 실행 함수"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"실험 실행 ID: {run_id}")
    
    all_results = {}
    
    # STEP 1: 거래 비용 OFF 실험
    step1_results = step1_cost_off(run_id)
    all_results["step1"] = step1_results
    
    # STEP 2: margin/entropy 스윕 (병렬 가능하지만 순차 실행)
    step2_results = step2_sweep(run_id)
    all_results["step2"] = step2_results
    
    # STEP 3: 조건부 실행
    cost_off_return = step1_results.get("cost_off", {}).get("total_return")
    step2_all_negative = all(
        r.get("total_return", 0) < 0
        for r in step2_results.values()
    )
    
    should_run_step3 = (
        (cost_off_return is not None and cost_off_return < 0) or
        step2_all_negative
    )
    
    step3_results = step3_stage2(run_id, should_run_step3)
    all_results["step3"] = step3_results
    
    # 전체 결과 저장
    summary_file = EXPERIMENTS_DIR / run_id / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"모든 실험 완료")
    print(f"결과 디렉토리: {EXPERIMENTS_DIR / run_id}")
    print(f"요약 파일: {summary_file}")
    print(f"{'='*80}")
    
    print(f"\n다음 단계: python scripts/summarize_guard_v2_final_eval.py {run_id}")
    
    return run_id

if __name__ == "__main__":
    run_id = main()
