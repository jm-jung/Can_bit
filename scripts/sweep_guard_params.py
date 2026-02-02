#!/usr/bin/env python3
"""
StrategyGuard 파라미터 스윕 실행 스크립트.

Usage:
    python scripts/sweep_guard_params.py \
        --strategy ml_tcn \
        --symbol BTCUSDT \
        --timeframe 5m \
        --start-date 2023-01-01 \
        --end-date 2024-12-31 \
        --direction long
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import json
import re


def parse_backtest_log(log_path: Path) -> dict:
    """백테스트 로그에서 메트릭 추출"""
    result = {
        "total_trades": 0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "total_blocks": 0,
        "block_rate": 0.0,
        "decision_sample": "",
    }
    
    if not log_path.exists():
        return result
    
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Total Trades
    match = re.search(r"Total Trades:\s*(\d+)", content)
    if match:
        result["total_trades"] = int(match.group(1))
    
    # Total Return
    match = re.search(r"Total Return:\s*([-+]?\d+\.?\d*)%", content)
    if match:
        result["total_return"] = float(match.group(1))
    
    # Max Drawdown
    match = re.search(r"Max Drawdown:\s*([-+]?\d+\.?\d*)%", content)
    if match:
        result["max_drawdown"] = float(match.group(1))
    
    # Win Rate
    match = re.search(r"Win Rate:\s*(\d+\.?\d*)%", content)
    if match:
        result["win_rate"] = float(match.group(1))
    
    # Sharpe Ratio
    match = re.search(r"Sharpe Ratio:\s*([-+]?\d+\.?\d*)", content)
    if match:
        result["sharpe"] = float(match.group(1))
    
    # Guard BLOCK count
    match = re.search(r"BLOCK=(\d+)", content)
    if match:
        result["total_blocks"] = int(match.group(1))
    
    # Block rate 계산
    if result["total_trades"] > 0:
        result["block_rate"] = result["total_blocks"] / result["total_trades"]
    
    # DECISION 로그 마지막 샘플
    decision_lines = re.findall(
        r"\[STRATEGY_GUARD\]\[DECISION\].*",
        content,
    )
    if decision_lines:
        result["decision_sample"] = decision_lines[-1]
    
    return result


def run_sweep(
    strategy: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    direction: str = "long",
    out_dir: Path | None = None,
) -> list[dict]:
    """파라미터 스윕 실행"""
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"data/experiments/guard_sweep_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 스윕 파라미터 정의
    recent_trades_windows = [5, 10, 20, 40]
    insufficient_sample_policies = ["allow", "block", "defer"]
    min_win_rates = [0.4, 0.6, 0.8]
    min_avg_returns = [-0.01, 0.0, 0.005]
    
    results = []
    
    total_runs = (
        len(recent_trades_windows)
        * len(insufficient_sample_policies)
        * len(min_win_rates)
        * len(min_avg_returns)
    )
    
    run_idx = 0
    
    for window in recent_trades_windows:
        for policy in insufficient_sample_policies:
            for min_wr in min_win_rates:
                for min_ar in min_avg_returns:
                    run_idx += 1
                    run_id = f"{run_idx:04d}_w{window}_p{policy}_wr{min_wr}_ar{min_ar}"
                    
                    print(f"[{run_idx}/{total_runs}] Running: {run_id}")
                    
                    log_path = out_dir / f"{run_id}.log"
                    csv_path = out_dir / f"{run_id}.csv"
                    
                    # CLI 명령 구성
                    cmd = [
                        sys.executable,
                        "-m",
                        "src.backtest.run_ml_xgb_backtest",
                        "--strategy",
                        strategy,
                        "--symbol",
                        symbol,
                        "--timeframe",
                        timeframe,
                        "--direction",
                        direction,
                        "--start-date",
                        start_date,
                        "--end-date",
                        end_date,
                        "--use-optimized-threshold",
                        "--signal-confirmation-bars",
                        "1",
                        "--use-strategy-guard",
                        "--strategy-guard-recent-trades-window",
                        str(window),
                        "--strategy-guard-insufficient-sample-policy",
                        policy,
                        "--strategy-guard-min-win-rate",
                        str(min_wr),
                        "--strategy-guard-min-avg-return",
                        str(min_ar),
                        "--strategy-guard-min-block-trades",
                        "3",
                        "--no-save",
                        "--dump-trades",
                        "--dump-trades-path",
                        str(csv_path),
                    ]
                    
                    # 실행
                    with open(log_path, "w", encoding="utf-8") as log_file:
                        proc = subprocess.run(
                            cmd,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                    
                    # 결과 파싱
                    metrics = parse_backtest_log(log_path)
                    metrics["run_id"] = run_id
                    metrics["recent_trades_window"] = window
                    metrics["insufficient_sample_policy"] = policy
                    metrics["min_win_rate"] = min_wr
                    metrics["min_avg_return"] = min_ar
                    metrics["exit_code"] = proc.returncode
                    
                    results.append(metrics)
                    
                    # 중간 결과 저장
                    results_path = out_dir / "results.json"
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)
                    
                    print(f"  → Trades: {metrics['total_trades']}, BLOCK: {metrics['total_blocks']}, Return: {metrics['total_return']:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="StrategyGuard 파라미터 스윕")
    parser.add_argument("--strategy", default="ml_tcn", help="전략명")
    parser.add_argument("--symbol", default="BTCUSDT", help="심볼")
    parser.add_argument("--timeframe", default="5m", help="타임프레임")
    parser.add_argument("--start-date", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--direction", default="long", choices=["long", "short", "both"], help="거래 방향")
    parser.add_argument("--out-dir", type=Path, default=None, help="출력 디렉토리")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("StrategyGuard Parameter Sweep")
    print("=" * 80)
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
    print(f"Period: {args.start_date} ~ {args.end_date}")
    print(f"Direction: {args.direction}")
    print("=" * 80)
    
    results = run_sweep(
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        direction=args.direction,
        out_dir=args.out_dir,
    )
    
    print("\n" + "=" * 80)
    print("Sweep Complete")
    print("=" * 80)
    print(f"Total runs: {len(results)}")
    print(f"Results saved to: {args.out_dir or 'data/experiments/guard_sweep_*'}")
    
    # block_rate > 0인 run 개수
    block_runs = [r for r in results if r["block_rate"] > 0]
    print(f"Runs with BLOCK > 0: {len(block_runs)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
