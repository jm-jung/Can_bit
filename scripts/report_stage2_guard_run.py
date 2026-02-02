#!/usr/bin/env python3
"""
Stage-2 + StrategyGuard 백테스트 결과를 리포트 문서에 자동 기록하는 스크립트.

Usage:
    python scripts/report_stage2_guard_run.py \
        --log data/backtest_logs/guard_blockrun_20250122_123456.log \
        --csv data/backtest_dumps/trades_guard_blockrun_20250122_123456.csv \
        --run-id 20250122_123456 \
        --period "2023-01-01~2024-12-31" \
        --case "GUARD_BLOCKRUN" \
        --out docs/stage2_guard_eval_longrun.md
"""
from __future__ import annotations

import argparse
import re
import os
from pathlib import Path


def parse_backtest_summary(log_content: str) -> dict:
    """Backtest Summary 블록에서 지표 추출"""
    result = {
        "total_return": "N/A",
        "win_rate": "N/A",
        "max_drawdown": "N/A",
        "sharpe": "N/A",
        "total_trades": "N/A",
    }
    
    # Total Return
    match = re.search(r"Total Return:\s*([-+]?\d+\.?\d*)", log_content)
    if match:
        result["total_return"] = match.group(1)
    
    # Win Rate
    match = re.search(r"Win Rate:\s*(\d+\.?\d*)", log_content)
    if match:
        result["win_rate"] = match.group(1)
    
    # Max Drawdown
    match = re.search(r"Max Drawdown:\s*([-+]?\d+\.?\d*)", log_content)
    if match:
        result["max_drawdown"] = match.group(1)
    
    # Sharpe Ratio
    match = re.search(r"Sharpe Ratio:\s*([-+]?\d+\.?\d*)", log_content)
    if match:
        result["sharpe"] = match.group(1)
    
    # Total Trades
    match = re.search(r"Total Trades:\s*(\d+)", log_content)
    if match:
        result["total_trades"] = match.group(1)
    
    return result


def parse_block_reasons(log_content: str) -> dict:
    """Block reasons 딕셔너리 추출"""
    result = {
        "hysteresis": "N/A",
        "stage2_no_trade": "N/A",
        "strategy_guard": "N/A",
        "others": "N/A",
    }
    
    # Block reasons 섹션 찾기
    block_section_match = re.search(
        r"\[ANTI-OVERTRADING\] Block reasons:.*?(\{.*?\})",
        log_content,
        re.DOTALL,
    )
    if block_section_match:
        block_str = block_section_match.group(1)
        
        # 각 키-값 추출
        for key in ["hysteresis", "stage2_no_trade", "strategy_guard"]:
            match = re.search(rf"'{key}':\s*(\d+)", block_str)
            if match:
                result[key] = match.group(1)
        
        # others (direction_filter, cooldown 등)
        others_keys = ["direction_filter", "cooldown", "confirmation", "margin_zone"]
        others_sum = 0
        for key in others_keys:
            match = re.search(rf"'{key}':\s*(\d+)", block_str)
            if match:
                others_sum += int(match.group(1))
        if others_sum > 0:
            result["others"] = str(others_sum)
    
    return result


def parse_stage2_stats(log_content: str) -> dict:
    """Stage-2 통계 추출"""
    result = {
        "trade": "N/A",
        "no_trade": "N/A",
        "exit_on_flat": "N/A",
    }
    
    # Stage-2 통계 섹션 찾기
    stage2_match = re.search(
        r"\[ANTI-OVERTRADING\] Stage-2:.*?trade=(\d+).*?no_trade=(\d+).*?exit_on_flat=(\d+)",
        log_content,
        re.DOTALL,
    )
    if stage2_match:
        result["trade"] = stage2_match.group(1)
        result["no_trade"] = stage2_match.group(2)
        result["exit_on_flat"] = stage2_match.group(3)
    else:
        # 개별 라인으로 찾기
        trade_match = re.search(r"stage2_trade_count=(\d+)", log_content)
        if trade_match:
            result["trade"] = trade_match.group(1)
        
        no_trade_match = re.search(r"stage2_no_trade_count=(\d+)", log_content)
        if no_trade_match:
            result["no_trade"] = no_trade_match.group(1)
    
    return result


def parse_guard_stats(log_content: str) -> dict:
    """StrategyGuard 통계 추출"""
    result = {
        "total_checks": "N/A",
        "allow": "N/A",
        "block": "N/A",
        "current_decision": "N/A",
        "recent_trades_tracked": "N/A",
        "decision_reason_sample": "",  # Decision 근거 샘플 (로그에서 파싱)
    }
    
    # StrategyGuard Statistics 섹션 찾기
    guard_section_match = re.search(
        r"\[STRATEGY GUARD\] StrategyGuard Statistics.*?Total checks: (\d+).*?ALLOW=(\d+).*?BLOCK=(\d+)",
        log_content,
        re.DOTALL,
    )
    if guard_section_match:
        result["total_checks"] = guard_section_match.group(1)
        result["allow"] = guard_section_match.group(2)
        result["block"] = guard_section_match.group(3)
    
    # Current decision
    decision_match = re.search(r"Current decision: (\w+)", log_content)
    if decision_match:
        result["current_decision"] = decision_match.group(1)
    
    # Recent trades tracked
    recent_match = re.search(r"Recent trades tracked: (\d+)", log_content)
    if recent_match:
        result["recent_trades_tracked"] = recent_match.group(1)
    
    # Decision 근거 샘플 파싱 ([STRATEGY_GUARD][DECISION] 라인)
    # 새 형식: [STRATEGY_GUARD][DECISION] event=ENTRY/EXIT trade_index=... idx=... ts=... decision=... recent_trades_count=... win_rate=... avg_return=... min_win_rate=... min_avg_return=... min_block_trades=... is_blocked=... block_trades_remaining=...
    decision_lines = re.findall(
        r"\[STRATEGY_GUARD\]\[DECISION\].*?event=(\w+).*?trade_index=(\d+).*?idx=(\d+).*?ts=([^\s]+).*?decision=(\w+).*?recent_trades_count=(\d+).*?win_rate=([\d.]+).*?avg_return=([\d.e-]+).*?min_win_rate=([\d.]+).*?min_avg_return=([\d.e-]+).*?min_block_trades=(\d+).*?is_blocked=(\w+).*?block_trades_remaining=(\d+)",
        log_content,
    )
    if decision_lines:
        # 마지막 샘플 사용
        last_sample = decision_lines[-1]
        result["decision_reason_sample"] = (
            f"event={last_sample[0]} decision={last_sample[4]} "
            f"recent_trades_count={last_sample[5]} win_rate={last_sample[6]} avg_return={last_sample[7]} "
            f"min_win_rate={last_sample[8]} min_avg_return={last_sample[9]} "
            f"is_blocked={last_sample[11]} block_trades_remaining={last_sample[12]}"
        )
    
    return result


def escape_markdown(text: str) -> str:
    """Markdown 테이블에서 특수문자 escape"""
    if text == "N/A" or text == "":
        return text
    return str(text).replace("|", "\\|").replace("\n", " ")


def format_block_reasons(block_reasons: dict) -> str:
    """Block reasons를 문자열로 포맷"""
    parts = []
    for key in ["hysteresis", "stage2_no_trade", "strategy_guard", "others"]:
        val = block_reasons.get(key, "N/A")
        if val != "N/A" and val != "0":
            parts.append(f"{key}={val}")
    if not parts:
        return "N/A"
    return "; ".join(parts)


def format_stage2(stage2: dict, case: str = "") -> str:
    """Stage-2 통계를 문자열로 포맷"""
    # GUARD_BLOCKRUN은 Stage2 OFF이므로 "OFF (N/A)"로 표시
    if case == "GUARD_BLOCKRUN":
        return "OFF (N/A)"
    
    parts = []
    for key in ["trade", "no_trade", "exit_on_flat"]:
        val = stage2.get(key, "N/A")
        if val != "N/A":
            parts.append(f"{key}={val}")
    if not parts:
        return "N/A"
    return "; ".join(parts)


def format_guard(guard: dict) -> str:
    """Guard 통계를 문자열로 포맷"""
    parts = []
    if guard.get("total_checks") != "N/A":
        parts.append(f"Checks={guard['total_checks']}")
    if guard.get("allow") != "N/A":
        parts.append(f"ALLOW={guard['allow']}")
    if guard.get("block") != "N/A":
        parts.append(f"BLOCK={guard['block']}")
    if guard.get("current_decision") != "N/A":
        parts.append(f"Decision={guard['current_decision']}")
    if guard.get("recent_trades_tracked") != "N/A":
        parts.append(f"Tracked={guard['recent_trades_tracked']}")
    if not parts:
        return "N/A"
    return "; ".join(parts)


def append_to_report(
    report_path: str,
    run_id: str,
    period: str,
    case: str,
    direction: str,
    summary: dict,
    block_reasons: dict,
    stage2: dict,
    guard: dict,
    csv_path: str,
    notes: str = "",
):
    """리포트 파일에 표 한 줄 추가"""
    report_file = Path(report_path)
    
    # 리포트 파일이 없으면 생성
    if not report_file.exists():
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# Stage-2 + StrategyGuard 장기구간 평가 리포트\n\n")
            f.write("**생성일:** 2025-01-22\n")
            f.write("**목적:** Guard BLOCK 유도 및 장기구간 성능 평가\n\n")
            f.write("## 평가 케이스\n\n")
            f.write("### 공통 설정\n")
            f.write("- **전략:** ml_tcn\n")
            f.write("- **심볼:** BTCUSDT\n")
            f.write("- **타임프레임:** 5m\n")
            f.write("- **방향:** long-only\n")
            f.write("- **최적화 임계값:** ON\n")
            f.write("- **Signal confirmation bars:** 1\n\n")
            f.write("## 실행 결과\n\n")
            f.write("| RunID | Period | Case | Direction | Trades | TotalReturn | WinRate | MaxDD | Sharpe | BlockReasons | Stage2 | Guard | Notes | CSVPaths |\n")
            f.write("|-------|--------|------|-----------|--------|-------------|---------|-------|--------|--------------|--------|-------|-------|----------|\n")
    
    # 기존 내용 읽기
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 동일 RunID가 있으면 제거
    lines = content.split("\n")
    new_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if f"|{run_id}|" in line:
            skip_next = True
            continue
        if skip_next and line.startswith("|") and "---" not in line:
            skip_next = False
            continue
        if not skip_next:
            new_lines.append(line)
    
    # 새 행 추가
    block_reasons_str = format_block_reasons(block_reasons)
    stage2_str = format_stage2(stage2, case=case)
    guard_str = format_guard(guard)
    
    csv_filename = os.path.basename(csv_path) if csv_path else "N/A"
    
    new_row = (
        f"|{escape_markdown(run_id)}|"
        f"{escape_markdown(period)}|"
        f"{escape_markdown(case)}|"
        f"{escape_markdown(direction)}|"
        f"{escape_markdown(summary.get('total_trades', 'N/A'))}|"
        f"{escape_markdown(summary.get('total_return', 'N/A'))}|"
        f"{escape_markdown(summary.get('win_rate', 'N/A'))}|"
        f"{escape_markdown(summary.get('max_drawdown', 'N/A'))}|"
        f"{escape_markdown(summary.get('sharpe', 'N/A'))}|"
        f"{escape_markdown(block_reasons_str)}|"
        f"{escape_markdown(stage2_str)}|"
        f"{escape_markdown(guard_str)}|"
        f"{escape_markdown(notes)}|"
        f"{escape_markdown(csv_filename)}|"
    )
    
    new_lines.append(new_row)
    
    # 파일 쓰기
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Stage-2 + StrategyGuard 결과 리포트 자동 기록")
    parser.add_argument("--log", required=True, help="백테스트 로그 파일 경로")
    parser.add_argument("--csv", required=True, help="Trade dump CSV 파일 경로")
    parser.add_argument("--run-id", required=True, help="Run ID (예: 20250122_123456)")
    parser.add_argument("--period", required=True, help="평가 기간 (예: 2023-01-01~2024-12-31)")
    parser.add_argument("--case", required=True, help="케이스명 (예: GUARD_BLOCKRUN)")
    parser.add_argument("--out", required=True, help="리포트 파일 경로")
    parser.add_argument("--direction", default="long", help="거래 방향 (기본값: long)")
    parser.add_argument("--notes", default="", help="추가 메모")
    
    args = parser.parse_args()
    
    # 로그 파일 읽기
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {args.log}")
        return 1
    
    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()
    
    # CSV 파일 확인
    csv_path = Path(args.csv)
    csv_exists = csv_path.exists()
    csv_size = csv_path.stat().st_size if csv_exists else 0
    
    # 파싱
    summary = parse_backtest_summary(log_content)
    block_reasons = parse_block_reasons(log_content)
    stage2 = parse_stage2_stats(log_content)
    guard = parse_guard_stats(log_content)
    
    # Notes에 파싱 실패 항목 추가
    notes_parts = [args.notes] if args.notes else []
    
    if summary["total_trades"] == "N/A":
        notes_parts.append("TotalTrades 파싱 실패")
    if block_reasons["hysteresis"] == "N/A":
        notes_parts.append("BlockReasons 파싱 실패")
    if stage2["trade"] == "N/A" and args.case != "GUARD_BLOCKRUN":
        # GUARD_BLOCKRUN은 Stage2 OFF이므로 N/A가 정상
        notes_parts.append("Stage2 통계 파싱 실패")
    if guard["total_checks"] == "N/A":
        notes_parts.append("Guard 통계 파싱 실패")
    
    if not csv_exists:
        notes_parts.append(f"CSV 파일 없음: {args.csv}")
    elif csv_size == 0:
        notes_parts.append(f"CSV 파일 크기 0: {args.csv}")
    
    # Guard decision debug 정보 추가
    decision_log_count = len(re.findall(r"\[STRATEGY_GUARD\]\[DECISION\]", log_content))
    if decision_log_count == 0:
        notes_parts.append("No DECISION log lines found (patch not executed or pattern mismatch)")
    elif guard.get("decision_reason_sample"):
        notes_parts.append(f"Guard decision debug enabled; Sample: {guard['decision_reason_sample']}")
        # BLOCK=0이면 근거값 추가
        if guard.get("block") == "0" or guard.get("block") == 0:
            sample_match = re.search(
                r"win_rate=([\d.]+).*?avg_return=([\d.e-]+).*?min_win_rate=([\d.]+).*?min_avg_return=([\d.e-]+)",
                guard["decision_reason_sample"],
            )
            if sample_match:
                win_rate = float(sample_match.group(1))
                avg_return = float(sample_match.group(2))
                min_win_rate = float(sample_match.group(3))
                min_avg_return = float(sample_match.group(4))
                notes_parts.append(
                    f"BLOCK=0 유지. 근거값: win_rate={win_rate:.4f} (th={min_win_rate:.4f}), "
                    f"avg_return={avg_return:.6f} (th={min_avg_return:.6f})"
                )
    elif guard.get("total_checks") != "N/A":
        notes_parts.append(f"Guard decision debug enabled (로그 {decision_log_count}줄 발견, 파싱 실패)")
    
    notes = "; ".join(notes_parts) if notes_parts else ""
    
    # 리포트에 추가
    append_to_report(
        report_path=args.out,
        run_id=args.run_id,
        period=args.period,
        case=args.case,
        direction=args.direction,
        summary=summary,
        block_reasons=block_reasons,
        stage2=stage2,
        guard=guard,
        csv_path=args.csv if csv_exists else "",
        notes=notes,
    )
    
    print(f"✓ 리포트 업데이트 완료: {args.out}")
    print(f"  RunID: {args.run_id}")
    print(f"  Total Trades: {summary.get('total_trades', 'N/A')}")
    print(f"  Guard BLOCK: {guard.get('block', 'N/A')}")
    
    return 0


if __name__ == "__main__":
    exit(main())

