#!/usr/bin/env python3
"""
StrategyGuard 2차 파라미터 스윕 결과 요약 리포트 생성.

Usage:
    python scripts/summarize_guard_sweep_v2.py \
        --results-dir data/experiments/guard_sweep_v2_20250120_123456 \
        --out docs/stage2_guard_param_sweep_v2.md \
        --top-n 10 \
        --baseline-return -5.35 \
        --baseline-maxdd 5.21
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import glob


def escape_markdown(text: str) -> str:
    """Markdown 테이블에서 특수문자 escape"""
    if text == "" or text is None:
        return "N/A"
    return str(text).replace("|", "\\|").replace("\n", " ")


def load_results(results_dir: Path) -> list[dict]:
    """results.json 또는 개별 run 결과 파일들 로드"""
    results = []
    
    # 전체 results.json이 있으면 사용
    results_json = results_dir / "results.json"
    if results_json.exists():
        with open(results_json, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        # 개별 run 디렉토리에서 result.json 로드
        for result_file in results_dir.glob("*/result.json"):
            with open(result_file, "r", encoding="utf-8") as f:
                results.append(json.load(f))
    
    return results


def summarize_sweep_v2_results(
    results_dir: Path,
    out_path: Path,
    top_n: int = 10,
    baseline_return: float = -5.35,
    baseline_maxdd: float = 5.21,
) -> None:
    """2차 스윕 결과를 요약하여 리포트 생성"""
    # 결과 로드
    results = load_results(results_dir)
    
    if not results:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# StrategyGuard 2차 파라미터 스윕 결과 요약\n\n")
            f.write("⚠️ **결과 파일을 찾을 수 없습니다.**\n\n")
        return
    
    # 필터: block_rate > 0 AND total_trades >= recent_trades_window
    filtered_results = []
    for r in results:
        block_rate = r.get("metrics", {}).get("block_rate", 0)
        total_trades = r.get("metrics", {}).get("total_trades", 0)
        window = r.get("guard_params", {}).get("recent_trades_window", 0)
        
        if block_rate > 0 and total_trades >= window:
            filtered_results.append(r)
    
    if not filtered_results:
        # 필터 조건을 만족하는 run이 없으면 리포트에 기록
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# StrategyGuard 2차 파라미터 스윕 결과 요약\n\n")
            f.write("**생성일:** " + str(results_dir.name) + "\n\n")
            f.write("## 요약\n\n")
            f.write("⚠️ **필터 조건을 만족하는 run이 없습니다.**\n\n")
            f.write("필터 조건: `block_rate > 0` AND `total_trades >= recent_trades_window`\n\n")
            f.write(f"- 총 실행: {len(results)} runs\n")
            f.write(f"- 필터 통과: 0 runs\n\n")
        return
    
    # 우선순위 정렬: final_return desc, max_drawdown asc, sharpe desc
    filtered_results_sorted = sorted(
        filtered_results,
        key=lambda x: (
            x.get("metrics", {}).get("final_return", 0),
            -x.get("metrics", {}).get("max_drawdown", 0),
            x.get("metrics", {}).get("sharpe", 0),
        ),
        reverse=True,
    )
    
    # 상위 N개 선택
    top_runs = filtered_results_sorted[:top_n]
    
    # 1차 대비 개선 여부 판별
    improved_runs = []
    for r in filtered_results:
        return_val = r.get("metrics", {}).get("final_return", 0)
        maxdd_val = r.get("metrics", {}).get("max_drawdown", 0)
        
        # Return 개선 또는 MaxDD 개선
        if return_val > baseline_return or maxdd_val < baseline_maxdd:
            improved_runs.append(r)
    
    # 리포트 생성
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# StrategyGuard 2차 파라미터 스윕 결과 요약\n\n")
        f.write("**생성일:** " + str(results_dir.name) + "\n\n")
        f.write("## 요약\n\n")
        f.write(f"- 총 실행: {len(results)} runs\n")
        f.write(f"- 필터 통과 (block_rate > 0 AND trades >= window): {len(filtered_results)} runs\n")
        f.write(f"- 상위 {top_n}개 요약 (final_return desc, max_drawdown asc, sharpe desc)\n\n")
        
        # 1차 대비 개선 여부
        f.write("## 1차 대비 개선 여부\n\n")
        f.write(f"**1차 Top 결과 기준:** Return={baseline_return}%, MaxDD={baseline_maxdd}%\n\n")
        if improved_runs:
            f.write(f"✅ **개선 조합 존재:** {len(improved_runs)} runs\n\n")
            f.write("개선 기준: `Return > -5.35%` OR `MaxDD < 5.21%`\n\n")
            # 최고 개선 조합 3개 표시
            best_improved = sorted(
                improved_runs,
                key=lambda x: (
                    x.get("metrics", {}).get("final_return", 0),
                    -x.get("metrics", {}).get("max_drawdown", 0),
                ),
                reverse=True,
            )[:3]
            f.write("### 최고 개선 조합 (상위 3개)\n\n")
            f.write("| RunID | Return | MaxDD | BlockRate | Window | Policy |\n")
            f.write("|-------|--------|-------|-----------|--------|--------|\n")
            for r in best_improved:
                f.write(
                    f"|{escape_markdown(r.get('run_id', 'N/A'))}|"
                    f"{escape_markdown(f\"{r.get('metrics', {}).get('final_return', 0):.2f}%\")}|"
                    f"{escape_markdown(f\"{r.get('metrics', {}).get('max_drawdown', 0):.2f}%\")}|"
                    f"{escape_markdown(f\"{r.get('metrics', {}).get('block_rate', 0):.4f}\")}|"
                    f"{escape_markdown(str(r.get('guard_params', {}).get('recent_trades_window', 'N/A')))}|"
                    f"{escape_markdown(r.get('guard_params', {}).get('insufficient_sample_policy', 'N/A'))}|\n"
                )
            f.write("\n")
        else:
            f.write("❌ **개선 조합 없음**\n\n")
            f.write("모든 조합에서 1차 Top 결과(Return=-5.35%, MaxDD=5.21%) 대비 개선이 없습니다.\n\n")
            f.write("### 결론\n\n")
            f.write("Guard v1(win_rate/avg_return 기반)로는 성과 개선이 어려운 것으로 판단됩니다.\n\n")
            f.write("### 다음 단계 제안\n\n")
            f.write("- Guard 지표 재설계 후보:\n")
            f.write("  1. **최근 N개 트레이드의 연속 손실 횟수 기반**: 연속 손실이 K회 이상이면 BLOCK\n")
            f.write("  2. **최근 N개 트레이드의 최대 손실폭 기반**: 최대 손실이 임계값을 초과하면 BLOCK\n")
            f.write("  3. **최근 N개 트레이드의 샤프 비율 기반**: 샤프 비율이 음수이고 절대값이 임계값 이상이면 BLOCK\n")
            f.write("\n")
        
        f.write("## 상위 N개 결과\n\n")
        f.write(
            "| RunID | Window | Policy | MinWR | MinAR | BlockRemain | Trades | BlockRate | Return | MaxDD | Sharpe | DecisionSample |\n"
        )
        f.write(
            "|-------|--------|--------|-------|-------|-------------|--------|-----------|--------|-------|--------|----------------|\n"
        )
        
        for r in top_runs:
            decision_sample = r.get("last_decision_sample", "")
            # DECISION 로그에서 핵심 정보만 추출
            if decision_sample:
                # 마지막 100자만 표시
                decision_sample = decision_sample[-100:] if len(decision_sample) > 100 else decision_sample
            else:
                decision_sample = "N/A"
            
            metrics = r.get("metrics", {})
            guard_params = r.get("guard_params", {})
            
            block_rate_val = metrics.get("block_rate", 0)
            total_return_val = metrics.get("final_return", 0)
            max_drawdown_val = metrics.get("max_drawdown", 0)
            sharpe_val = metrics.get("sharpe", 0)
            
            block_rate_str = f"{block_rate_val:.4f}"
            total_return_str = f"{total_return_val:.2f}%"
            max_drawdown_str = f"{max_drawdown_val:.2f}%"
            sharpe_str = f"{sharpe_val:.4f}"
            
            f.write(
                f"|{escape_markdown(r.get('run_id', 'N/A'))}|"
                f"{escape_markdown(str(guard_params.get('recent_trades_window', 'N/A')))}|"
                f"{escape_markdown(guard_params.get('insufficient_sample_policy', 'N/A'))}|"
                f"{escape_markdown(str(guard_params.get('min_win_rate', 'N/A')))}|"
                f"{escape_markdown(str(guard_params.get('min_avg_return', 'N/A')))}|"
                f"{escape_markdown(str(guard_params.get('min_block_trades', 'N/A')))}|"
                f"{escape_markdown(str(metrics.get('total_trades', 'N/A')))}|"
                f"{escape_markdown(block_rate_str)}|"
                f"{escape_markdown(total_return_str)}|"
                f"{escape_markdown(max_drawdown_str)}|"
                f"{escape_markdown(sharpe_str)}|"
                f"{escape_markdown(decision_sample)}|\n"
            )
    
    print(f"✓ 리포트 생성 완료: {out_path}")
    print(f"  총 실행: {len(results)} runs")
    print(f"  필터 통과: {len(filtered_results)} runs")
    print(f"  상위 {top_n}개 요약 완료")
    if improved_runs:
        print(f"  ✅ 1차 대비 개선: {len(improved_runs)} runs")
    else:
        print(f"  ❌ 1차 대비 개선: 없음")


def main():
    parser = argparse.ArgumentParser(description="StrategyGuard 2차 스윕 결과 요약")
    parser.add_argument("--results-dir", type=Path, required=True, help="스윕 결과 디렉토리")
    parser.add_argument("--out", type=Path, required=True, help="출력 리포트 경로")
    parser.add_argument("--top-n", type=int, default=10, help="상위 N개 (기본값: 10)")
    parser.add_argument("--baseline-return", type=float, default=-5.35, help="1차 Top Return 기준값")
    parser.add_argument("--baseline-maxdd", type=float, default=5.21, help="1차 Top MaxDD 기준값")
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1
    
    summarize_sweep_v2_results(
        results_dir=args.results_dir,
        out_path=args.out,
        top_n=args.top_n,
        baseline_return=args.baseline_return,
        baseline_maxdd=args.baseline_maxdd,
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
