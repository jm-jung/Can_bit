#!/usr/bin/env python3
"""
StrategyGuard 2차 파라미터 스윕 중간 요약 리포트 생성.

스윕 진행 중에 중간 결과를 요약하여 조기 탐색을 지원합니다.

Usage:
    python scripts/summarize_guard_sweep_v2_partial.py \
        --results-dir data/experiments/guard_sweep_v2_20260121_144412 \
        --out docs/stage2_guard_param_sweep_v2_partial.md \
        --baseline-return -5.35 \
        --baseline-maxdd 5.21
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


def load_partial_results(results_dir: Path) -> list[dict]:
    """진행 중인 스윕 결과 파일들 로드 (크기>0인 파일만)"""
    results = []
    
    # 전체 results.json이 있으면 사용
    results_json = results_dir / "results.json"
    if results_json.exists() and results_json.stat().st_size > 0:
        try:
            with open(results_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results.extend(loaded)
                else:
                    results.append(loaded)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Failed to parse results.json: {e}")
    
    # 개별 run 디렉토리에서 result.json 로드 (크기>0인 것만)
    for result_file in results_dir.glob("*/result.json"):
        if result_file.stat().st_size > 0:
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                    # 중복 방지: run_id로 체크
                    if not any(r.get("run_id") == result.get("run_id") for r in results):
                        results.append(result)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Failed to parse {result_file}: {e}")
    
    return results


def summarize_partial_results(
    results_dir: Path,
    out_path: Path,
    baseline_return: float = -5.35,
    baseline_maxdd: float = 5.21,
    max_candidates: int = 10,
) -> None:
    """중간 결과를 요약하여 리포트 생성"""
    # 결과 로드
    results = load_partial_results(results_dir)
    
    if not results:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# StrategyGuard 2차 파라미터 스윕 중간 요약\n\n")
            f.write("⚠️ **완료된 run이 없습니다.**\n\n")
            f.write(f"디렉토리: {results_dir}\n\n")
        print("⚠️ 완료된 run이 없습니다.")
        return
    
    # baseline 대비 개선 후보 필터링
    improved_candidates = []
    for r in results:
        metrics = r.get("metrics", {})
        return_val = metrics.get("final_return", 0)
        maxdd_val = metrics.get("max_drawdown", 0)
        
        # 개선 조건: (return > baseline_return) OR (max_drawdown < baseline_maxdd)
        if return_val > baseline_return or maxdd_val < baseline_maxdd:
            improved_candidates.append(r)
    
    # 정렬: return desc, max_drawdown asc
    improved_candidates_sorted = sorted(
        improved_candidates,
        key=lambda x: (
            x.get("metrics", {}).get("final_return", 0),
            -x.get("metrics", {}).get("max_drawdown", 0),
        ),
        reverse=True,
    )
    
    # 상위 N개 선택
    top_candidates = improved_candidates_sorted[:max_candidates]
    
    # 리포트 생성
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# StrategyGuard 2차 파라미터 스윕 중간 요약\n\n")
        f.write(f"**생성일:** {results_dir.name}\n\n")
        f.write("## 현재 진행 상황\n\n")
        f.write(f"- 완료된 runs: {len(results)} 개\n")
        f.write(f"- Baseline 대비 개선 후보: {len(improved_candidates)} 개\n")
        f.write(f"- 상위 {len(top_candidates)}개 표시\n\n")
        f.write(f"**Baseline 기준:** Return={baseline_return}%, MaxDD={baseline_maxdd}%\n\n")
        f.write("**개선 조건:** `Return > -5.35%` OR `MaxDD < 5.21%`\n\n")
        
        if top_candidates:
            f.write("## Baseline 대비 개선 후보 (상위 10개)\n\n")
            f.write(
                "| RunID | Window | Policy | MinWR | MinAR | BlockRemain | Return | MaxDD | BlockRate |\n"
            )
            f.write(
                "|-------|--------|--------|-------|-------|-------------|--------|-------|-----------|\n"
            )
            
            for r in top_candidates:
                metrics = r.get("metrics", {})
                guard_params = r.get("guard_params", {})
                
                return_val = metrics.get("final_return", 0)
                maxdd_val = metrics.get("max_drawdown", 0)
                block_rate_val = metrics.get("block_rate", 0)
                
                return_str = f"{return_val:.2f}%"
                maxdd_str = f"{maxdd_val:.2f}%"
                block_rate_str = f"{block_rate_val:.4f}"
                
                f.write(
                    f"|{escape_markdown(r.get('run_id', 'N/A'))}|"
                    f"{escape_markdown(str(guard_params.get('recent_trades_window', 'N/A')))}|"
                    f"{escape_markdown(guard_params.get('insufficient_sample_policy', 'N/A'))}|"
                    f"{escape_markdown(str(guard_params.get('min_win_rate', 'N/A')))}|"
                    f"{escape_markdown(str(guard_params.get('min_avg_return', 'N/A')))}|"
                    f"{escape_markdown(str(guard_params.get('min_block_trades', 'N/A')))}|"
                    f"{escape_markdown(return_str)}|"
                    f"{escape_markdown(maxdd_str)}|"
                    f"{escape_markdown(block_rate_str)}|\n"
                )
        else:
            f.write("## Baseline 대비 개선 후보\n\n")
            f.write("❌ **개선 후보 없음**\n\n")
            f.write("현재까지 완료된 runs 중 baseline 대비 개선된 조합이 없습니다.\n\n")
    
    print(f"✓ 중간 요약 리포트 생성 완료: {out_path}")
    print(f"  완료된 runs: {len(results)} 개")
    print(f"  Baseline 대비 개선 후보: {len(improved_candidates)} 개")
    if top_candidates:
        print(f"  상위 {len(top_candidates)}개 표시:")
        for i, r in enumerate(top_candidates[:5], 1):
            metrics = r.get("metrics", {})
            print(
                f"    {i}. {r.get('run_id', 'N/A')}: "
                f"Return={metrics.get('final_return', 0):.2f}%, "
                f"MaxDD={metrics.get('max_drawdown', 0):.2f}%"
            )


def main():
    parser = argparse.ArgumentParser(description="StrategyGuard 2차 스윕 중간 요약")
    parser.add_argument("--results-dir", type=Path, required=True, help="스윕 결과 디렉토리")
    parser.add_argument("--out", type=Path, required=True, help="출력 리포트 경로")
    parser.add_argument("--baseline-return", type=float, default=-5.35, help="1차 Top Return 기준값")
    parser.add_argument("--baseline-maxdd", type=float, default=5.21, help="1차 Top MaxDD 기준값")
    parser.add_argument("--max-candidates", type=int, default=10, help="최대 후보 개수")
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1
    
    summarize_partial_results(
        results_dir=args.results_dir,
        out_path=args.out,
        baseline_return=args.baseline_return,
        baseline_maxdd=args.baseline_maxdd,
        max_candidates=args.max_candidates,
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
