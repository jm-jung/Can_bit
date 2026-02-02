#!/usr/bin/env python3
"""
정밀 분석 자료 추출 스크립트

v2.0 (Stage-2 OFF)와 v2.1 (Stage-2 ON) 실행 결과를 비교하여
요청 1-3 자료를 추출합니다.
"""

import json
import subprocess
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

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

def run_and_extract(extra_args: list, run_name: str) -> dict:
    """백테스트 실행 및 정밀 분석 자료 추출"""
    cmd = [
        sys.executable, "-m", "src.backtest.run_ml_xgb_backtest"
    ] + COMMON_ARGS + extra_args
    
    print(f"\n{'='*80}")
    print(f"실행: {run_name}")
    print(f"{'='*80}")
    print(f"커맨드: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    output = result.stdout
    
    # 정밀 분석 통계 추출
    analysis = {}
    
    # [요청 1] Final Scale 분포
    final_scale_match = re.search(
        r"\[요청 1\] Final Scale 분포.*?final_scale_min: ([\d.]+).*?"
        r"final_scale_median: ([\d.]+).*?"
        r"final_scale_p90: ([\d.]+).*?"
        r"final_scale_max: ([\d.]+).*?"
        r"final_scale_mean: ([\d.]+)",
        output,
        re.DOTALL
    )
    
    if final_scale_match:
        analysis["final_scale_stats"] = {
            "min": float(final_scale_match.group(1)),
            "median": float(final_scale_match.group(2)),
            "p90": float(final_scale_match.group(3)),
            "max": float(final_scale_match.group(4)),
            "mean": float(final_scale_match.group(5)),
        }
    
    # [요청 2] Stage-2 차단/완화 통계
    block_match = re.search(
        r"\[요청 2\] Stage-2 차단/완화 통계:.*?"
        r"stage2_hard_block_count: (\d+).*?"
        r"stage2_soft_gated_count: (\d+).*?"
        r"stage2_total_entries: (\d+).*?"
        r"entries_attempted: (\d+)",
        output,
        re.DOTALL
    )
    
    if block_match:
        analysis["stage2_stats"] = {
            "hard_block_count": int(block_match.group(1)),
            "soft_gated_count": int(block_match.group(2)),
            "total_entries": int(block_match.group(3)),
            "entries_attempted": int(block_match.group(4)),
        }
    
    # [요청 3] 샘플 추출
    samples = []
    sample_blocks = re.findall(
        r"샘플 \d+:.*?(?=샘플 \d+:|================================================================================)",
        output,
        re.DOTALL
    )
    
    for block in sample_blocks[:10]:
        sample = {}
        idx_match = re.search(r"idx=(\d+)", block)
        if idx_match:
            sample["idx"] = int(idx_match.group(1))
        
        ts_match = re.search(r"ts=([^,]+)", block)
        if ts_match:
            sample["ts"] = ts_match.group(1).strip()
        
        signal_match = re.search(r"signal=(\w+)", block)
        if signal_match:
            sample["signal"] = signal_match.group(1)
        
        trade_match = re.search(r"stage2_trade=(True|False|None)", block)
        if trade_match:
            val = trade_match.group(1)
            sample["stage2_trade"] = True if val == "True" else False if val == "False" else None
        
        reason_match = re.search(r"stage2_reason=([^\n]+)", block)
        if reason_match:
            sample["stage2_reason"] = reason_match.group(1).strip()
        
        p_diff_match = re.search(r"p_diff=([\d.]+|None)", block)
        if p_diff_match:
            val = p_diff_match.group(1)
            sample["p_diff"] = float(val) if val != "None" else None
        
        entropy_match = re.search(r"entropy=([\d.]+|None)", block)
        if entropy_match:
            val = entropy_match.group(1)
            sample["entropy"] = float(val) if val != "None" else None
        
        edge_match = re.search(r"edge=(True|False)", block)
        if edge_match:
            sample["edge"] = edge_match.group(1) == "True"
        
        base_match = re.search(r"base=([\d.]+)", block)
        if base_match:
            sample["base"] = float(base_match.group(1))
        
        regime_match = re.search(r"regime_factor=([\d.]+)", block)
        if regime_match:
            sample["regime_factor"] = float(regime_match.group(1))
        
        risk_match = re.search(r"risk_factor=([\d.]+)", block)
        if risk_match:
            sample["risk_factor"] = float(risk_match.group(1))
        
        ev_match = re.search(r"ev_factor=([\d.]+)", block)
        if ev_match:
            sample["ev_factor"] = float(ev_match.group(1))
        
        score_match = re.search(r"stage2_score=([\d.]+)", block)
        if score_match:
            sample["stage2_score"] = float(score_match.group(1))
        
        guard_match = re.search(r"guard_scale=([\d.]+)", block)
        if guard_match:
            sample["guard_scale"] = float(guard_match.group(1))
        
        final_match = re.search(r"final_scale=([\d.]+)", block)
        if final_match:
            sample["final_scale"] = float(final_match.group(1))
        
        if sample:
            samples.append(sample)
    
    analysis["samples"] = samples
    
    # Backtest Summary 추출
    summary_match = re.search(
        r"Total Return: ([\d.-]+)%\s+"
        r"Win Rate: ([\d.-]+)%\s+"
        r"Max Drawdown: ([\d.-]+)%\s+"
        r"Total Trades: (\d+)",
        output,
        re.MULTILINE
    )
    
    if summary_match:
        analysis["backtest_summary"] = {
            "total_return": float(summary_match.group(1)),
            "win_rate": float(summary_match.group(2)),
            "max_drawdown": float(summary_match.group(3)),
            "total_trades": int(summary_match.group(4)),
        }
    
    return analysis

def main():
    """메인 실행 함수"""
    results = {}
    
    # Run 1: Guard v2 ON + Stage-2 OFF (v2.0)
    results["v20_stage2_off"] = run_and_extract([], "v2.0 (Guard v2 ON + Stage-2 OFF)")
    
    # Run 2: Guard v2 ON + Stage-2 ON (v2.1)
    results["v21_stage2_on"] = run_and_extract(["--use-stage2"], "v2.1 (Guard v2 ON + Stage-2 ON)")
    
    # 결과 저장
    output_file = PROJECT_ROOT / "data" / "experiments" / "precision_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # 비교 표 출력
    print("\n" + "="*80)
    print("정밀 분석 결과 비교")
    print("="*80)
    
    print("\n[요청 1] Final Scale 분포 비교 (ENTRY 시점 기준):")
    print("-" * 80)
    v20 = results["v20_stage2_off"].get("final_scale_stats", {})
    v21 = results["v21_stage2_on"].get("final_scale_stats", {})
    
    print(f"{'지표':<20} {'v2.0 (Stage-2 OFF)':<25} {'v2.1 (Stage-2 ON)':<25}")
    print("-" * 80)
    if v20 and v21:
        print(f"{'min':<20} {v20.get('min', 0):<25.6f} {v21.get('min', 0):<25.6f}")
        print(f"{'median':<20} {v20.get('median', 0):<25.6f} {v21.get('median', 0):<25.6f}")
        print(f"{'p90':<20} {v20.get('p90', 0):<25.6f} {v21.get('p90', 0):<25.6f}")
        print(f"{'max':<20} {v20.get('max', 0):<25.6f} {v21.get('max', 0):<25.6f}")
        print(f"{'mean':<20} {v20.get('mean', 0):<25.6f} {v21.get('mean', 0):<25.6f}")
    
    print("\n[요청 2] Stage-2 차단/완화 통계:")
    print("-" * 80)
    v20_stats = results["v20_stage2_off"].get("stage2_stats", {})
    v21_stats = results["v21_stage2_on"].get("stage2_stats", {})
    
    print(f"{'지표':<30} {'v2.0 (Stage-2 OFF)':<25} {'v2.1 (Stage-2 ON)':<25}")
    print("-" * 80)
    print(f"{'hard_block_count':<30} {v20_stats.get('hard_block_count', 0):<25} {v21_stats.get('hard_block_count', 0):<25}")
    print(f"{'soft_gated_count':<30} {v20_stats.get('soft_gated_count', 0):<25} {v21_stats.get('soft_gated_count', 0):<25}")
    print(f"{'total_entries':<30} {v20_stats.get('total_entries', 0):<25} {v21_stats.get('total_entries', 0):<25}")
    print(f"{'entries_attempted':<30} {v20_stats.get('entries_attempted', 0):<25} {v21_stats.get('entries_attempted', 0):<25}")
    
    print("\n[요청 3] Stage-2 v2.1 스코어 계산 근거 샘플 (10개):")
    print("-" * 80)
    samples = results["v21_stage2_on"].get("samples", [])
    for idx, sample in enumerate(samples[:10], 1):
        print(f"\n샘플 {idx}:")
        print(f"  idx={sample.get('idx')}, ts={sample.get('ts')}, signal={sample.get('signal')}")
        print(f"  stage2_trade={sample.get('stage2_trade')}, stage2_reason={sample.get('stage2_reason')}")
        print(f"  파싱된 입력값: p_diff={sample.get('p_diff')}, entropy={sample.get('entropy')}, edge={sample.get('edge')}")
        print(f"  base={sample.get('base')}, regime_factor={sample.get('regime_factor')}, "
              f"risk_factor={sample.get('risk_factor')}, ev_factor={sample.get('ev_factor')}")
        print(f"  stage2_score={sample.get('stage2_score')}, guard_scale={sample.get('guard_scale')}, "
              f"final_scale={sample.get('final_scale')}")
    
    print(f"\n{'='*80}")
    print(f"결과 저장: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
