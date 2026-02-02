#!/usr/bin/env python3
"""
StrategyGuard 파라미터 스윕 결과 요약 리포트 생성.

Usage:
    python scripts/summarize_guard_sweep.py \
        --results data/experiments/guard_sweep_20250120_123456/results.json \
        --out docs/stage2_guard_param_sweep.md \
        --top-n 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def escape_markdown(text: str) -> str:
    """Markdown 테이블에서 특수문자 escape"""
    if text == "" or text is None:
        return "N/A"
    return str(text).replace("|", "\\|").replace("\n", " ")


def summarize_sweep_results(
    results_path: Path,
    out_path: Path,
    top_n: int = 5,
) -> None:
    """스윕 결과를 요약하여 리포트 생성"""
    # 결과 로드
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # block_rate > 0인 run만 필터
    block_runs = [r for r in results if r.get("block_rate", 0) > 0]
    
    if not block_runs:
        # BLOCK이 발생한 run이 없으면 리포트에 기록
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# StrategyGuard 파라미터 스윕 결과 요약\n\n")
            f.write("**생성일:** " + str(Path(results_path).parent.name) + "\n\n")
            f.write("## 요약\n\n")
            f.write("⚠️ **BLOCK이 발생한 run이 없습니다.**\n\n")
            f.write("모든 파라미터 조합에서 `block_rate = 0`입니다.\n\n")
            f.write("## 전체 결과\n\n")
            f.write("| RunID | Window | Policy | MinWR | MinAR | Trades | BlockRate | Return | MaxDD |\n")
            f.write("|-------|--------|--------|-------|-------|--------|-----------|--------|-------|\n")
            for r in results[:10]:  # 상위 10개만 표시
                block_rate_val = r.get("block_rate", 0)
                total_return_val = r.get("total_return", 0)
                max_drawdown_val = r.get("max_drawdown", 0)
                block_rate_str = f"{block_rate_val:.4f}"
                total_return_str = f"{total_return_val:.2f}%"
                max_drawdown_str = f"{max_drawdown_val:.2f}%"
                
                f.write(
                    f"|{escape_markdown(r.get('run_id', 'N/A'))}|"
                    f"{escape_markdown(str(r.get('recent_trades_window', 'N/A')))}|"
                    f"{escape_markdown(r.get('insufficient_sample_policy', 'N/A'))}|"
                    f"{escape_markdown(str(r.get('min_win_rate', 'N/A')))}|"
                    f"{escape_markdown(str(r.get('min_avg_return', 'N/A')))}|"
                    f"{escape_markdown(str(r.get('total_trades', 'N/A')))}|"
                    f"{escape_markdown(block_rate_str)}|"
                    f"{escape_markdown(total_return_str)}|"
                    f"{escape_markdown(max_drawdown_str)}|\n"
                )
        return
    
    # 우선순위 정렬: final_return desc, max_drawdown asc
    block_runs_sorted = sorted(
        block_runs,
        key=lambda x: (x.get("total_return", 0), -x.get("max_drawdown", 0)),
        reverse=True,
    )
    
    # 상위 N개 선택
    top_runs = block_runs_sorted[:top_n]
    
    # 리포트 생성
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# StrategyGuard 파라미터 스윕 결과 요약\n\n")
        f.write("**생성일:** " + str(Path(results_path).parent.name) + "\n\n")
        f.write("## 요약\n\n")
        f.write(f"- 총 실행: {len(results)} runs\n")
        f.write(f"- BLOCK 발생: {len(block_runs)} runs\n")
        f.write(f"- 상위 {top_n}개 요약 (block_rate > 0, final_return desc, max_drawdown asc)\n\n")
        f.write("## 상위 N개 결과\n\n")
        f.write(
            "| RunID | Window | Policy | MinWR | MinAR | Trades | BlockRate | Return | MaxDD | Sharpe | DecisionSample |\n"
        )
        f.write(
            "|-------|--------|--------|-------|-------|--------|-----------|--------|-------|--------|----------------|\n"
        )
        
        for r in top_runs:
            decision_sample = r.get("decision_sample", "")
            # DECISION 로그에서 핵심 정보만 추출
            if decision_sample:
                # 마지막 100자만 표시
                decision_sample = decision_sample[-100:] if len(decision_sample) > 100 else decision_sample
            else:
                decision_sample = "N/A"
            
            block_rate_str = f"{r.get('block_rate', 0):.4f}"
            total_return_str = f"{r.get('total_return', 0):.2f}%"
            max_drawdown_str = f"{r.get('max_drawdown', 0):.2f}%"
            sharpe_str = f"{r.get('sharpe', 0):.4f}"
            
            f.write(
                f"|{escape_markdown(r.get('run_id', 'N/A'))}|"
                f"{escape_markdown(str(r.get('recent_trades_window', 'N/A')))}|"
                f"{escape_markdown(r.get('insufficient_sample_policy', 'N/A'))}|"
                f"{escape_markdown(str(r.get('min_win_rate', 'N/A')))}|"
                f"{escape_markdown(str(r.get('min_avg_return', 'N/A')))}|"
                f"{escape_markdown(str(r.get('total_trades', 'N/A')))}|"
                f"{escape_markdown(block_rate_str)}|"
                f"{escape_markdown(total_return_str)}|"
                f"{escape_markdown(max_drawdown_str)}|"
                f"{escape_markdown(sharpe_str)}|"
                f"{escape_markdown(decision_sample)}|\n"
            )
    
    print(f"✓ 리포트 생성 완료: {out_path}")
    print(f"  BLOCK 발생 run: {len(block_runs)}/{len(results)}")
    print(f"  상위 {top_n}개 요약 완료")


def main():
    parser = argparse.ArgumentParser(description="StrategyGuard 스윕 결과 요약")
    parser.add_argument("--results", type=Path, required=True, help="results.json 경로")
    parser.add_argument("--out", type=Path, required=True, help="출력 리포트 경로")
    parser.add_argument("--top-n", type=int, default=5, help="상위 N개 (기본값: 5)")
    
    args = parser.parse_args()
    
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        return 1
    
    summarize_sweep_results(
        results_path=args.results,
        out_path=args.out,
        top_n=args.top_n,
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
