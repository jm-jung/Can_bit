#!/usr/bin/env python3
"""
Stage-2 CAP 임계값 장기 검증 스크립트

3개 run을 장기 구간(2023-01-01~2024-12-31)으로 실행하고 결과를 수집합니다.
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

# ======================================================================
# 장기 검증 대상 3개 run
# ======================================================================
LONGRUN_RUNS = [
    {
        "run_id": "baseline",
        "description": "현재 완화 CAP (기본값)",
        "high_entropy_th": 0.66,
        "mid_entropy_th": 0.64,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.004,
    },
    {
        "run_id": "pdiff_small_005",
        "description": "p_diff small 임계값 완화 (0.004 → 0.005)",
        "high_entropy_th": 0.66,
        "mid_entropy_th": 0.64,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.005,
    },
    {
        "run_id": "entropy_high_64_mid_62",
        "description": "entropy 임계값 완화 (high=0.64, mid=0.62)",
        "high_entropy_th": 0.64,
        "mid_entropy_th": 0.62,
        "tiny_pdiff_th": 0.002,
        "small_pdiff_th": 0.004,
    },
]

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
    "--use-stage2",
    "--no-save",
]


def run_backtest(run_id: str, params: dict) -> dict:
    """백테스트 실행 및 결과 추출"""
    cmd = [
        sys.executable, "-m", "src.backtest.run_ml_xgb_backtest"
    ] + COMMON_ARGS + [
        "--stage2-cap-entropy-high-th", str(params["high_entropy_th"]),
        "--stage2-cap-entropy-mid-th", str(params["mid_entropy_th"]),
        "--stage2-cap-pdiff-tiny-th", str(params["tiny_pdiff_th"]),
        "--stage2-cap-pdiff-small-th", str(params["small_pdiff_th"]),
    ]
    
    logger.info(f"[{run_id}] 백테스트 실행 중...")
    logger.info(f"  파라미터: high={params['high_entropy_th']}, mid={params['mid_entropy_th']}, "
                f"tiny={params['tiny_pdiff_th']}, small={params['small_pdiff_th']}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    
    if result.returncode != 0:
        logger.error(f"[{run_id}] 백테스트 실패: {result.stderr[:500]}")
        return {
            "run_id": run_id,
            "error": result.stderr[:500],
            "returncode": result.returncode,
        }
    
    # 로그 파싱
    log_output = result.stdout + result.stderr
    metrics = parse_backtest_log(log_output, run_id, params)
    
    return metrics


def parse_backtest_log(log_output: str, run_id: str, params: dict) -> dict:
    """백테스트 로그에서 메트릭을 추출합니다."""
    metrics = {
        "run_id": run_id,
        "params": params,
        "total_return": None,
        "max_drawdown": None,
        "total_trades": None,
        "win_rate": None,
        "cap_distribution": None,
        "final_scale_stats": None,
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
    
    # CAP 분포 파싱 (로그에서)
    cap_dist_match = re.search(
        r"\[요청 2-1\] CAP 분포.*?cap=1\.0: (\d+) \(([\d.]+)%\).*?"
        r"cap=0\.8: (\d+) \(([\d.]+)%\).*?"
        r"cap=0\.6: (\d+) \(([\d.]+)%\)",
        log_output,
        re.DOTALL,
    )
    if cap_dist_match:
        metrics["cap_distribution"] = {
            "cap_1_0": int(cap_dist_match.group(1)),
            "cap_1_0_pct": float(cap_dist_match.group(2)),
            "cap_0_8": int(cap_dist_match.group(3)),
            "cap_0_8_pct": float(cap_dist_match.group(4)),
            "cap_0_6": int(cap_dist_match.group(5)),
            "cap_0_6_pct": float(cap_dist_match.group(6)),
            "total_entries": int(cap_dist_match.group(1)) + int(cap_dist_match.group(3)) + int(cap_dist_match.group(5)),
        }
    
    # final_scale 통계 파싱
    final_scale_match = re.search(
        r"\[요청 1\] Final Scale 분포.*?"
        r"final_scale_min: ([\d.]+).*?"
        r"final_scale_median: ([\d.]+).*?"
        r"final_scale_p90: ([\d.]+).*?"
        r"final_scale_max: ([\d.]+).*?"
        r"final_scale_mean: ([\d.]+)",
        log_output,
        re.DOTALL,
    )
    if final_scale_match:
        metrics["final_scale_stats"] = {
            "min": float(final_scale_match.group(1)),
            "median": float(final_scale_match.group(2)),
            "p90": float(final_scale_match.group(3)),
            "max": float(final_scale_match.group(4)),
            "mean": float(final_scale_match.group(5)),
        }
    
    return metrics


def generate_report(results: list[dict], report_path: Path):
    """장기 검증 리포트 생성"""
    with open(report_path, "w") as f:
        f.write("# Stage-2 CAP 임계값 장기 검증 결과\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**검증 기간**: 2023-01-01 ~ 2024-12-31 (장기 구간)\n\n")
        f.write("## 비교 대상\n\n")
        f.write("| Run ID | 설명 | Entropy (High/Mid) | p_diff (Tiny/Small) |\n")
        f.write("|--------|------|-------------------|---------------------|\n")
        
        for r in results:
            params = r.get("params", {})
            description = params.get("description", "N/A")
            f.write(
                f"| {r['run_id']} | {description} | "
                f"{params.get('high_entropy_th', 'N/A')}/{params.get('mid_entropy_th', 'N/A')} | "
                f"{params.get('tiny_pdiff_th', 'N/A')}/{params.get('small_pdiff_th', 'N/A')} |\n"
            )
        
        f.write("\n## 결과 비교\n\n")
        f.write("| Run ID | Return | MaxDD | Trades | Win Rate | ")
        f.write("CAP 분포 (1.0/0.8/0.6) | final_scale (min/median/mean/p90/max) |\n")
        f.write("|--------|--------|-------|--------|----------|")
        f.write("------------------------|--------------------------------------|\n")
        
        for r in results:
            if "error" in r:
                f.write(f"| {r['run_id']} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |\n")
                continue
            
            cap_dist = r.get("cap_distribution", {})
            cap_str = (
                f"{cap_dist.get('cap_1_0', 0)}/{cap_dist.get('cap_0_8', 0)}/{cap_dist.get('cap_0_6', 0)} "
                f"({cap_dist.get('cap_1_0_pct', 0):.1f}%/{cap_dist.get('cap_0_8_pct', 0):.1f}%/{cap_dist.get('cap_0_6_pct', 0):.1f}%)"
                if cap_dist else "N/A"
            )
            
            fs_stats = r.get("final_scale_stats", {})
            fs_str = (
                f"{fs_stats.get('min', 0):.3f}/{fs_stats.get('median', 0):.3f}/"
                f"{fs_stats.get('mean', 0):.3f}/{fs_stats.get('p90', 0):.3f}/{fs_stats.get('max', 0):.3f}"
                if fs_stats else "N/A"
            )
            
            f.write(
                f"| {r['run_id']} | "
                f"{r.get('total_return', 'N/A')}% | "
                f"{r.get('max_drawdown', 'N/A')}% | "
                f"{r.get('total_trades', 'N/A')} | "
                f"{r.get('win_rate', 'N/A')}% | "
                f"{cap_str} | "
                f"{fs_str} |\n"
            )
        
        f.write("\n## 결론\n\n")
        baseline = next((r for r in results if r["run_id"] == "baseline"), None)
        if baseline and "error" not in baseline:
            f.write(f"**Baseline (현재 완화 CAP)**:\n")
            f.write(f"- Return: {baseline.get('total_return', 'N/A')}%\n")
            f.write(f"- MaxDD: {baseline.get('max_drawdown', 'N/A')}%\n")
            f.write(f"- Total Trades: {baseline.get('total_trades', 'N/A')}\n")
            f.write(f"- Win Rate: {baseline.get('win_rate', 'N/A')}%\n\n")
            
            improved = [
                r for r in results
                if r["run_id"] != "baseline" and "error" not in r
                and (
                    (r.get("total_return") is not None and baseline.get("total_return") is not None
                     and r.get("total_return") > baseline.get("total_return"))
                    or (r.get("max_drawdown") is not None and baseline.get("max_drawdown") is not None
                        and r.get("max_drawdown") < baseline.get("max_drawdown"))
                )
            ]
            
            if improved:
                f.write("**Baseline 대비 개선된 후보**:\n\n")
                for r in improved:
                    f.write(f"- **{r['run_id']}**: Return={r.get('total_return')}%, MaxDD={r.get('max_drawdown')}%\n")
            else:
                f.write("**Baseline 대비 개선된 후보 없음**\n")


def main():
    """메인 실행"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "data" / "experiments" / f"stage2_cap_longrun_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, run_config in enumerate(LONGRUN_RUNS, 1):
        run_id = run_config["run_id"]
        logger.info(f"[{i}/{len(LONGRUN_RUNS)}] {run_id} 실행 중...")
        
        result = run_backtest(run_id, run_config)
        results.append(result)
        
        # 중간 저장
        results_file = results_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"[{run_id}] 완료: Return={result.get('total_return')}%, MaxDD={result.get('max_drawdown')}%")
    
    # 리포트 생성
    report_path = PROJECT_ROOT / "docs" / "stage2_cap_threshold_sweep_longrun.md"
    generate_report(results, report_path)
    
    logger.info(f"장기 검증 완료. 결과: {results_dir}")
    logger.info(f"리포트: {report_path}")


if __name__ == "__main__":
    main()
