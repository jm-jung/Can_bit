#!/usr/bin/env python3
"""
Stage-2 CAP 임계값 미니 스윕 스크립트

목적: CAP 적용 빈도(현재 cap=1.0 비중 70%)를 낮춰서 성능 개선 폭을 키우는 것
방법: entropy/p_diff 임계값을 완화하여 CAP 적용 빈도 증가
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

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================================
# 스윕 그리드 설계
# ======================================================================
# 원칙: "조금만 완화"해서 cap=1.0 비중을 70% → 50~60% 정도로만 낮추기
# 기본값: high_entropy_th=0.66, mid_entropy_th=0.64, tiny_pdiff_th=0.002, small_pdiff_th=0.004

SWEEP_GRID = [
    # 기본값 (baseline)
    {
        "run_id": "baseline",
        "high_entropy_th": 0.66,
        "mid_entropy_th": 0.64,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.004,
    },
    # entropy만 완화 (mid_entropy_th만)
    {
        "run_id": "entropy_mid_62",
        "high_entropy_th": 0.66,
        "mid_entropy_th": 0.62,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.004,
    },
    {
        "run_id": "entropy_mid_60",
        "high_entropy_th": 0.66,
        "mid_entropy_th": 0.60,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.004,
    },
    # entropy 완화 (high도)
    {
        "run_id": "entropy_high_64_mid_62",
        "high_entropy_th": 0.64,
        "mid_entropy_th": 0.62,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.004,
    },
    {
        "run_id": "entropy_high_62_mid_60",
        "high_entropy_th": 0.62,
        "mid_entropy_th": 0.60,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.004,
    },
    # p_diff만 완화
    {
        "run_id": "pdiff_small_005",
        "high_entropy_th": 0.66,
        "mid_entropy_th": 0.64,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.005,
    },
    {
        "run_id": "pdiff_small_006",
        "high_entropy_th": 0.66,
        "mid_entropy_th": 0.64,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.006,
    },
    # entropy + p_diff 조합
    {
        "run_id": "entropy_mid_62_pdiff_005",
        "high_entropy_th": 0.66,
        "mid_entropy_th": 0.62,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.005,
    },
    {
        "run_id": "entropy_mid_60_pdiff_005",
        "high_entropy_th": 0.66,
        "mid_entropy_th": 0.60,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.005,
    },
]

# ======================================================================
# 백테스트 실행 함수
# ======================================================================
def run_backtest_with_cap_thresholds(
    run_id: str,
    high_entropy_th: float,
    mid_entropy_th: float,
    tiny_pdiff_th: float,
    small_pdiff_th: float,
) -> dict:
    """
    CAP 임계값을 설정하고 백테스트를 실행합니다.
    
    방법: CLI 옵션으로 임계값을 전달합니다.
    """
    # 백테스트 실행
    cmd = [
        sys.executable, "-m", "src.backtest.run_ml_xgb_backtest",
        "--strategy", "ml_tcn",
        "--symbol", "BTCUSDT",
        "--timeframe", "5m",
        "--direction", "long",
        "--start-date", "2025-01-01",
        "--end-date", "2025-03-01",
        "--use-optimized-threshold",
        "--signal-confirmation-bars", "1",
        "--min-hold-bars", "12",
        "--cooldown-bars", "12",
        "--use-strategy-guard",
        "--strategy-guard-v2",
        "--strategy-guard-v2-mode", "soft",
        "--strategy-guard-v2-scale-floor", "0.02",
        "--use-stage2",
        "--stage2-cap-entropy-high-th", str(high_entropy_th),
        "--stage2-cap-entropy-mid-th", str(mid_entropy_th),
        "--stage2-cap-pdiff-tiny-th", str(tiny_pdiff_th),
        "--stage2-cap-pdiff-small-th", str(small_pdiff_th),
        "--no-save",
    ]
    
    logger.info(f"[{run_id}] 백테스트 실행 중...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    
    if result.returncode != 0:
        logger.error(f"[{run_id}] 백테스트 실패: {result.stderr}")
        return {
            "run_id": run_id,
            "error": result.stderr,
            "returncode": result.returncode,
        }
    
    # 로그 파싱
    log_output = result.stdout + result.stderr
    metrics = parse_backtest_log(log_output, run_id)
    
    return metrics


# ======================================================================
# 로그 파싱 함수
# ======================================================================
def parse_backtest_log(log_output: str, run_id: str) -> dict:
    """백테스트 로그에서 메트릭을 추출합니다."""
    metrics = {
        "run_id": run_id,
        "total_return": None,
        "max_drawdown": None,
        "total_trades": None,
        "win_rate": None,
        "avg_holding": None,
        "cap_distribution": {
            "cap_1_0": 0,
            "cap_0_8": 0,
            "cap_0_6": 0,
        },
        "final_scale_mean": None,
        "final_scale_median": None,
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
    
    # CAP 분포 파싱
    cap_logs = re.findall(r"\[STAGE2\]\[CAP\].*?cap=([\d.]+)", log_output)
    for cap_str in cap_logs:
        cap_val = float(cap_str)
        if cap_val == 1.0:
            metrics["cap_distribution"]["cap_1_0"] += 1
        elif cap_val == 0.8:
            metrics["cap_distribution"]["cap_0_8"] += 1
        elif cap_val == 0.6:
            metrics["cap_distribution"]["cap_0_6"] += 1
    
    # final_scale 통계 파싱
    final_scale_match = re.search(
        r"final_scale_mean: ([\d.]+)",
        log_output,
    )
    if final_scale_match:
        metrics["final_scale_mean"] = float(final_scale_match.group(1))
    
    final_scale_median_match = re.search(
        r"final_scale_median: ([\d.]+)",
        log_output,
    )
    if final_scale_median_match:
        metrics["final_scale_median"] = float(final_scale_median_match.group(1))
    
    return metrics


# ======================================================================
# 메인 실행
# ======================================================================
def main():
    """스윕 실행 및 결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "data" / "experiments" / f"stage2_cap_sweep_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, params in enumerate(SWEEP_GRID, 1):
        run_id = params["run_id"]
        logger.info(f"[{i}/{len(SWEEP_GRID)}] {run_id} 실행 중...")
        
        result = run_backtest_with_cap_thresholds(
            run_id=run_id,
            high_entropy_th=params["high_entropy_th"],
            mid_entropy_th=params["mid_entropy_th"],
            tiny_pdiff_th=params["tiny_pdiff_th"],
            small_pdiff_th=params["small_pdiff_th"],
        )
        
        result.update(params)  # 파라미터도 결과에 포함
        results.append(result)
        
        # 중간 저장
        results_file = results_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"[{run_id}] 완료: Return={result.get('total_return')}%, MaxDD={result.get('max_drawdown')}%")
    
    # 최종 요약 리포트 생성
    generate_summary_report(results, results_dir)
    
    logger.info(f"스윕 완료. 결과: {results_dir}")


def generate_summary_report(results: list[dict], results_dir: Path):
    """요약 리포트 생성"""
    # baseline 찾기
    baseline = next((r for r in results if r["run_id"] == "baseline"), None)
    
    # 필터링: 에러 없는 결과만
    valid_results = [r for r in results if "error" not in r and r.get("total_return") is not None]
    
    # 정렬: Return desc, MaxDD asc
    sorted_results = sorted(
        valid_results,
        key=lambda x: (x.get("total_return", -999), -x.get("max_drawdown", 999)),
        reverse=True,
    )
    
    # Top 5 선택
    top_n = sorted_results[:5]
    
    # Markdown 리포트 생성
    report_path = PROJECT_ROOT / "docs" / "stage2_cap_threshold_sweep.md"
    with open(report_path, "w") as f:
        f.write("# Stage-2 CAP 임계값 스윕 결과\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**스윕 기간**: 2025-01-01 ~ 2025-03-01 (단기 구간)\n\n")
        
        if baseline:
            f.write(f"## Baseline (현재 완화 CAP)\n\n")
            f.write(f"- Total Return: {baseline.get('total_return', 'N/A')}%\n")
            f.write(f"- Max Drawdown: {baseline.get('max_drawdown', 'N/A')}%\n")
            f.write(f"- Total Trades: {baseline.get('total_trades', 'N/A')}\n")
            f.write(f"- Win Rate: {baseline.get('win_rate', 'N/A')}%\n")
            cap_dist = baseline.get("cap_distribution", {})
            total_cap = sum(cap_dist.values())
            if total_cap > 0:
                f.write(f"- CAP 분포: cap=1.0 ({cap_dist.get('cap_1_0', 0) * 100 // total_cap}%), "
                       f"cap=0.8 ({cap_dist.get('cap_0_8', 0) * 100 // total_cap}%), "
                       f"cap=0.6 ({cap_dist.get('cap_0_6', 0) * 100 // total_cap}%)\n")
            f.write("\n")
        
        f.write("## Top 5 결과\n\n")
        f.write("| Run ID | Entropy (High/Mid) | p_diff (Tiny/Small) | ")
        f.write("Return | MaxDD | Trades | Win Rate | CAP 분포 (1.0/0.8/0.6) |\n")
        f.write("|--------|-------------------|---------------------|")
        f.write("-------|-------|--------|----------|------------------------|\n")
        
        for r in top_n:
            cap_dist = r.get("cap_distribution", {})
            total_cap = sum(cap_dist.values())
            cap_pct = (
                f"{cap_dist.get('cap_1_0', 0) * 100 // total_cap if total_cap > 0 else 0}%/"
                f"{cap_dist.get('cap_0_8', 0) * 100 // total_cap if total_cap > 0 else 0}%/"
                f"{cap_dist.get('cap_0_6', 0) * 100 // total_cap if total_cap > 0 else 0}%"
            )
            
            f.write(
                f"| {r['run_id']} | "
                f"{r.get('high_entropy_th', 'N/A')}/{r.get('mid_entropy_th', 'N/A')} | "
                f"{r.get('tiny_pdiff_th', 'N/A')}/{r.get('small_pdiff_th', 'N/A')} | "
                f"{r.get('total_return', 'N/A')}% | "
                f"{r.get('max_drawdown', 'N/A')}% | "
                f"{r.get('total_trades', 'N/A')} | "
                f"{r.get('win_rate', 'N/A')}% | "
                f"{cap_pct} |\n"
            )
        
        f.write("\n## 유망 후보 추천\n\n")
        if baseline:
            improved = [
                r for r in valid_results
                if r["run_id"] != "baseline"
                and (
                    r.get("total_return", -999) > baseline.get("total_return", -999)
                    or r.get("max_drawdown", 999) < baseline.get("max_drawdown", 999)
                )
            ]
            if improved:
                f.write("다음 후보들이 baseline 대비 개선을 보입니다:\n\n")
                for r in improved[:3]:
                    f.write(f"- **{r['run_id']}**: Return={r.get('total_return')}%, MaxDD={r.get('max_drawdown')}%\n")
            else:
                f.write("baseline 대비 개선된 후보가 없습니다.\n")
    
    logger.info(f"요약 리포트 생성: {report_path}")


if __name__ == "__main__":
    main()
