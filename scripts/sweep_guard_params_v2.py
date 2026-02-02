#!/usr/bin/env python3
"""
StrategyGuard 2차 파라미터 스윕 실행 스크립트.

1차 스윕에서 성과 개선이 없던 원인을 검증하고,
"성과(수익률/드로우다운) 개선이 가능한 Guard 조합"이 존재하는지 최종 판별.

Usage:
    python scripts/sweep_guard_params_v2.py \
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
        "total_entries": 0,
        "total_exits": 0,
        "total_trades": 0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "total_blocks": 0,
        "block_rate": 0.0,
        "last_decision_sample": "",
        "guard_summary_sample": "",
    }
    
    if not log_path.exists():
        return result
    
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Total Entries (TRADE COUNT SUMMARY)
    match = re.search(r"total_entries=(\d+)", content)
    if match:
        result["total_entries"] = int(match.group(1))
    
    # Total Exits (TRADE COUNT SUMMARY)
    match = re.search(r"total_exits=(\d+)", content)
    if match:
        result["total_exits"] = int(match.group(1))
    
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
        result["last_decision_sample"] = decision_lines[-1]
    
    # SUMMARY 로그 마지막 샘플
    summary_lines = re.findall(
        r"\[STRATEGY_GUARD\]\[SUMMARY\].*",
        content,
    )
    if summary_lines:
        result["guard_summary_sample"] = summary_lines[-1]
    
    return result


def run_sweep_v2(
    strategy: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    direction: str = "long",
    out_dir: Path | None = None,
) -> list[dict]:
    """2차 파라미터 스윕 실행"""
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"data/experiments/guard_sweep_v2_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2차 스윕 파라미터 정의
    insufficient_sample_policies = ["block", "defer"]  # allow 제외
    recent_trades_windows = [3, 5, 7, 10]
    min_win_rates = [0.2, 0.3, 0.4, 0.5, 0.6]
    min_avg_returns = [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005]
    min_block_trades_list = [1, 3, 5]
    
    results = []
    
    total_runs = (
        len(insufficient_sample_policies)
        * len(recent_trades_windows)
        * len(min_win_rates)
        * len(min_avg_returns)
        * len(min_block_trades_list)
    )
    
    run_idx = 0
    
    for policy in insufficient_sample_policies:
        for window in recent_trades_windows:
            for min_wr in min_win_rates:
                for min_ar in min_avg_returns:
                    for min_bt in min_block_trades_list:
                        run_idx += 1
                        run_id = f"{run_idx:04d}_w{window}_p{policy}_wr{min_wr}_ar{min_ar}_br{min_bt}"
                        
                        print(f"[{run_idx}/{total_runs}] Running: {run_id}")
                        
                        run_dir = out_dir / run_id
                        run_dir.mkdir(exist_ok=True)
                        
                        log_path = run_dir / f"{run_id}.log"
                        csv_path = run_dir / f"{run_id}.csv"
                        
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
                            str(min_bt),
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
                        
                        # 결과 JSON 구성
                        result = {
                            "run_id": run_id,
                            "strategy": strategy,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "direction": direction,
                            "start_date": start_date,
                            "end_date": end_date,
                            "guard_params": {
                                "recent_trades_window": window,
                                "insufficient_sample_policy": policy,
                                "min_win_rate": min_wr,
                                "min_avg_return": min_ar,
                                "min_block_trades": min_bt,
                            },
                            "metrics": {
                                "total_entries": metrics["total_entries"],
                                "total_exits": metrics["total_exits"],
                                "total_trades": metrics["total_trades"],
                                "total_blocks": metrics["total_blocks"],
                                "block_rate": metrics["block_rate"],
                                "final_return": metrics["total_return"],
                                "max_drawdown": metrics["max_drawdown"],
                                "sharpe": metrics["sharpe"],
                            },
                            "last_decision_sample": metrics["last_decision_sample"] or None,
                            "guard_summary_sample": metrics["guard_summary_sample"] or None,
                            "exit_code": proc.returncode,
                        }
                        
                        results.append(result)
                        
                        # 개별 run 결과 저장
                        result_path = run_dir / "result.json"
                        with open(result_path, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2)
                        
                        # 중간 결과 저장 (전체)
                        results_path = out_dir / "results.json"
                        with open(results_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2)
                        
                        print(
                            f"  → Trades: {metrics['total_trades']}, "
                            f"BLOCK: {metrics['total_blocks']}, "
                            f"Return: {metrics['total_return']:.2f}%, "
                            f"MaxDD: {metrics['max_drawdown']:.2f}%"
                        )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="StrategyGuard 2차 파라미터 스윕")
    parser.add_argument("--strategy", default="ml_tcn", help="전략명")
    parser.add_argument("--symbol", default="BTCUSDT", help="심볼")
    parser.add_argument("--timeframe", default="5m", help="타임프레임")
    parser.add_argument("--start-date", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--direction", default="long", choices=["long", "short", "both"], help="거래 방향")
    parser.add_argument("--out-dir", type=Path, default=None, help="출력 디렉토리")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("StrategyGuard Parameter Sweep V2")
    print("=" * 80)
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
    print(f"Period: {args.start_date} ~ {args.end_date}")
    print(f"Direction: {args.direction}")
    print("=" * 80)
    print("Sweep Parameters:")
    print("  - insufficient_sample_policy: [block, defer] (allow 제외)")
    print("  - recent_trades_window: [3, 5, 7, 10]")
    print("  - min_win_rate: [0.2, 0.3, 0.4, 0.5, 0.6]")
    print("  - min_avg_return: [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005]")
    print("  - min_block_trades: [1, 3, 5]")
    print("=" * 80)
    
    results = run_sweep_v2(
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        direction=args.direction,
        out_dir=args.out_dir,
    )
    
    print("\n" + "=" * 80)
    print("Sweep V2 Complete")
    print("=" * 80)
    print(f"Total runs: {len(results)}")
    print(f"Results saved to: {args.out_dir or 'data/experiments/guard_sweep_v2_*'}")
    
    # block_rate > 0인 run 개수
    block_runs = [r for r in results if r["metrics"]["block_rate"] > 0]
    print(f"Runs with BLOCK > 0: {len(block_runs)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
