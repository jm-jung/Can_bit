#!/usr/bin/env python3
"""
로그 파일에서 CAP 디버그 정보를 추출하여 리포트를 생성하는 스크립트
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_cap_debug_from_log(log_file: Path) -> dict:
    """로그 파일에서 CAP 디버그 정보 추출"""
    result = {
        "cap_debug_logs": [],
        "cap_entry_exit_links": [],
        "cap_trigger_stats": {
            "total_checks": 0,
            "entries_attempted": 0,
            "entries_executed": 0,
            "cap_1_0_count": 0,
            "cap_0_8_count": 0,
            "cap_0_6_count": 0,
            "triggered_by_entropy_only": 0,
            "triggered_by_pdiff_only": 0,
            "triggered_by_both": 0,
            "triggered_by_none": 0,
        },
        "total_return": None,
        "max_drawdown": None,
        "total_trades": None,
        "win_rate": None,
    }
    
    if not log_file.exists():
        logger.error(f"로그 파일이 없습니다: {log_file}")
        return result
    
    with open(log_file, "r") as f:
        content = f.read()
    
    # [STAGE2][CAP] JSON 로그 추출 (한 줄 JSON)
    cap_log_pattern = r'\[STAGE2\]\[CAP\]\s+(\{.*?\})\s*$'
    for line in content.split('\n'):
        match = re.search(cap_log_pattern, line)
        if match:
            try:
                json_str = match.group(1)
                log_entry = json.loads(json_str)
                result["cap_debug_logs"].append(log_entry)
            except json.JSONDecodeError:
                continue
    
    # [ENTRY][CAPLINK] JSON 로그 추출
    entry_link_pattern = r'\[ENTRY\]\[CAPLINK\]\s+(\{.*?\})\s*$'
    for line in content.split('\n'):
        match = re.search(entry_link_pattern, line)
        if match:
            try:
                json_str = match.group(1)
                log_entry = json.loads(json_str)
                result["cap_entry_exit_links"].append(log_entry)
            except json.JSONDecodeError:
                continue
    
    # [EXIT][CAPLINK] JSON 로그 추출 및 ENTRY 링크에 병합
    exit_link_pattern = r'\[EXIT\]\[CAPLINK\]\s+(\{.*?\})\s*$'
    for line in content.split('\n'):
        match = re.search(exit_link_pattern, line)
        if match:
            try:
                json_str = match.group(1)
                exit_link = json.loads(json_str)
                trade_id = exit_link.get("trade_id")
                if trade_id:
                    # ENTRY 링크 찾아서 병합
                    for entry_link in result["cap_entry_exit_links"]:
                        if entry_link.get("trade_id") == trade_id and entry_link.get("event") == "ENTRY":
                            entry_link.update(exit_link)
                            break
            except json.JSONDecodeError:
                continue
    
    # [CAP DEBUG SUMMARY] 추출
    summary_pattern = r'\[CAP DEBUG SUMMARY\].*?total_checks:\s+(\d+).*?entries_attempted:\s+(\d+).*?entries_executed:\s+(\d+).*?cap_1_0_count:\s+(\d+).*?cap_0_8_count:\s+(\d+).*?cap_0_6_count:\s+(\d+).*?triggered_by_entropy_only:\s+(\d+).*?triggered_by_pdiff_only:\s+(\d+).*?triggered_by_both:\s+(\d+).*?triggered_by_none:\s+(\d+)'
    summary_match = re.search(summary_pattern, content, re.DOTALL)
    if summary_match:
        result["cap_trigger_stats"]["total_checks"] = int(summary_match.group(1))
        result["cap_trigger_stats"]["entries_attempted"] = int(summary_match.group(2))
        result["cap_trigger_stats"]["entries_executed"] = int(summary_match.group(3))
        result["cap_trigger_stats"]["cap_1_0_count"] = int(summary_match.group(4))
        result["cap_trigger_stats"]["cap_0_8_count"] = int(summary_match.group(5))
        result["cap_trigger_stats"]["cap_0_6_count"] = int(summary_match.group(6))
        result["cap_trigger_stats"]["triggered_by_entropy_only"] = int(summary_match.group(7))
        result["cap_trigger_stats"]["triggered_by_pdiff_only"] = int(summary_match.group(8))
        result["cap_trigger_stats"]["triggered_by_both"] = int(summary_match.group(9))
        result["cap_trigger_stats"]["triggered_by_none"] = int(summary_match.group(10))
    
    # Total Return, Max Drawdown, Total Trades 추출
    summary_match = re.search(
        r"Total Return: ([\d.-]+)%\s+"
        r"Win Rate: ([\d.-]+)%\s+"
        r"Max Drawdown: ([\d.-]+)%\s+"
        r"Total Trades: (\d+)",
        content,
    )
    if summary_match:
        result["total_return"] = float(summary_match.group(1))
        result["win_rate"] = float(summary_match.group(2))
        result["max_drawdown"] = float(summary_match.group(3))
        result["total_trades"] = int(summary_match.group(4))
    
    return result


def generate_report(result: dict, output_dir: Path):
    """CAP 디버그 리포트 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1) stage2_cap_debug_summary_<ts>.md
    summary_path = output_dir / f"stage2_cap_debug_summary_{timestamp}.md"
    with open(summary_path, "w") as f:
        f.write("# Stage-2 CAP 로그 기반 최종 확정 리포트\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 실행 커맨드/설정
        f.write("## 실행 커맨드/설정\n\n")
        f.write("- Guard v2 + Stage2 v2.2 CAP\n")
        f.write("- pdiff_small_th=0.005 (기본값 승격)\n")
        f.write("- final_scale = guard_scale * stage2_cap\n\n")
        
        # 성능 요약
        if result.get("total_return") is not None:
            f.write("## 성능 요약\n\n")
            f.write(f"- Total Return: {result['total_return']:.2f}%\n")
            f.write(f"- Max Drawdown: {result['max_drawdown']:.2f}%\n")
            f.write(f"- Total Trades: {result['total_trades']}\n")
            f.write(f"- Win Rate: {result['win_rate']:.2f}%\n\n")
        
        # CAP 분포
        trigger_stats = result.get("cap_trigger_stats", {})
        f.write("## CAP 분포 + trigger_by_rule\n\n")
        f.write("| 항목 | 값 | 비율 |\n")
        f.write("|------|-----|------|\n")
        total_checks = trigger_stats.get("total_checks", 0)
        if total_checks > 0:
            f.write(f"| total_checks | {total_checks} | 100% |\n")
            f.write(f"| entries_attempted | {trigger_stats.get('entries_attempted', 0)} | {trigger_stats.get('entries_attempted', 0) * 100 / total_checks:.1f}% |\n")
            f.write(f"| entries_executed | {trigger_stats.get('entries_executed', 0)} | {trigger_stats.get('entries_executed', 0) * 100 / total_checks:.1f}% |\n")
            f.write(f"| cap_1_0_count | {trigger_stats.get('cap_1_0_count', 0)} | {trigger_stats.get('cap_1_0_count', 0) * 100 / total_checks:.1f}% |\n")
            f.write(f"| cap_0_8_count | {trigger_stats.get('cap_0_8_count', 0)} | {trigger_stats.get('cap_0_8_count', 0) * 100 / total_checks:.1f}% |\n")
            f.write(f"| cap_0_6_count | {trigger_stats.get('cap_0_6_count', 0)} | {trigger_stats.get('cap_0_6_count', 0) * 100 / total_checks:.1f}% |\n")
        f.write("\n")
        f.write("| Trigger 원인 | 값 |\n")
        f.write("|-------------|-----|\n")
        f.write(f"| triggered_by_entropy_only | {trigger_stats.get('triggered_by_entropy_only', 0)} |\n")
        f.write(f"| triggered_by_pdiff_only | {trigger_stats.get('triggered_by_pdiff_only', 0)} |\n")
        f.write(f"| triggered_by_both | {trigger_stats.get('triggered_by_both', 0)} |\n")
        f.write(f"| triggered_by_none | {trigger_stats.get('triggered_by_none', 0)} |\n\n")
        
        # cap=0.6 케이스 로그 발췌
        debug_logs = result.get("cap_debug_logs", [])
        cap_0_6_logs = [log for log in debug_logs if abs(log.get("stage2_cap", 1.0) - 0.6) < 0.001]
        f.write("## cap=0.6 케이스 로그 발췌 (5개)\n\n")
        if cap_0_6_logs:
            for i, log in enumerate(cap_0_6_logs[:5], 1):
                f.write(f"### 케이스 {i}\n\n")
                f.write(f"- ts: {log.get('ts')}\n")
                f.write(f"- entropy: {log.get('entropy')}\n")
                f.write(f"- p_diff: {log.get('p_diff')}\n")
                f.write(f"- cap_reason: {log.get('cap_reason')}\n")
                f.write(f"- trigger_type: {log.get('trigger_type')}\n\n")
        else:
            f.write("cap=0.6 케이스가 이번 구간에 발생하지 않았습니다.\n\n")
        
        # ENTRY/EXIT 연결 샘플
        links = result.get("cap_entry_exit_links", [])
        entry_links = [l for l in links if l.get("event") == "ENTRY" or "entry_ts" in l]
        connected_trades = [l for l in entry_links if "exit_ts" in l and "scaled_profit" in l]
        f.write("## ENTRY/EXIT 연결 샘플 (cap별 성과 비교)\n\n")
        f.write("| cap | trade_count | mean_profit | mean_holding |\n")
        f.write("|-----|-------------|-------------|--------------|\n")
        for cap_val in [1.0, 0.8, 0.6]:
            cap_trades = [l for l in connected_trades if abs(l.get("stage2_cap", 1.0) - cap_val) < 0.001]
            if cap_trades:
                profits = [t.get("scaled_profit", 0) for t in cap_trades]
                holdings = [t.get("holding_bars", 0) for t in cap_trades]
                f.write(f"| {cap_val} | {len(cap_trades)} | {sum(profits) / len(profits):.6f} | {sum(holdings) / len(holdings):.1f} |\n")
            else:
                f.write(f"| {cap_val} | 0 | N/A | N/A |\n")
        
        # 결론
        f.write("\n## 결론\n\n")
        f.write("### 왜 cap=1.0이 70%인가?\n\n")
        total_checks = trigger_stats.get("total_checks", 0)
        cap_1_0_pct = trigger_stats.get("cap_1_0_count", 0) * 100 / total_checks if total_checks > 0 else 0
        f.write(f"- cap=1.0 비율: {cap_1_0_pct:.1f}%\n")
        f.write(f"- triggered_by_none: {trigger_stats.get('triggered_by_none', 0)} (대부분의 경우 CAP 조건을 만족하지 않음)\n")
        f.write(f"- entropy/p_diff 임계값이 보수적이어서 대부분의 경우 CAP이 적용되지 않음\n\n")
        
        f.write("### cap=0.6 유지/제거 추천\n\n")
        cap_0_6_count = trigger_stats.get("cap_0_6_count", 0)
        if cap_0_6_count > 0:
            f.write(f"- cap=0.6 발생 횟수: {cap_0_6_count} (샘플 수 충분)\n")
            f.write(f"- 유지 추천: 샘플 수가 충분하고 성과 분석 필요\n")
        else:
            f.write(f"- cap=0.6 발생 횟수: {cap_0_6_count} (샘플 수 부족)\n")
            f.write(f"- 제거 고려: 샘플 수가 부족하여 효과 검증 어려움\n")
        f.write("\n")
        
        f.write("### entropy vs p_diff 중 무엇을 조정할지 추천\n\n")
        entropy_only = trigger_stats.get("triggered_by_entropy_only", 0)
        pdiff_only = trigger_stats.get("triggered_by_pdiff_only", 0)
        both = trigger_stats.get("triggered_by_both", 0)
        f.write(f"- triggered_by_entropy_only: {entropy_only}\n")
        f.write(f"- triggered_by_pdiff_only: {pdiff_only}\n")
        f.write(f"- triggered_by_both: {both}\n")
        if both > entropy_only and both > pdiff_only:
            f.write(f"- 추천: entropy와 p_diff 임계값을 함께 완화 (both가 가장 많음)\n")
        elif entropy_only > pdiff_only:
            f.write(f"- 추천: entropy 임계값 완화 (entropy_only가 더 많음)\n")
        else:
            f.write(f"- 추천: p_diff 임계값 완화 (pdiff_only가 더 많음)\n")
        f.write("\n")
    
    # 2) stage2_cap_debug_samples_<ts>.jsonl
    samples_path = output_dir / f"stage2_cap_debug_samples_{timestamp}.jsonl"
    debug_logs = result.get("cap_debug_logs", [])
    with open(samples_path, "w") as f:
        for log in debug_logs:
            f.write(json.dumps(log, default=str) + "\n")
    
    # 3) stage2_cap_entry_exit_link_<ts>.csv
    csv_path = output_dir / f"stage2_cap_entry_exit_link_{timestamp}.csv"
    links = result.get("cap_entry_exit_links", [])
    entry_links = [l for l in links if l.get("event") == "ENTRY" or "entry_ts" in l]
    connected_trades = [l for l in entry_links if "exit_ts" in l]
    with open(csv_path, "w") as f:
        # 헤더
        f.write("trade_id,entry_ts,exit_ts,side,entry_price,exit_price,holding_bars,")
        f.write("p_long,margin,entropy,p_diff,guard_scale,stage2_cap,final_scale,cap_reason,")
        f.write("realized_profit,scaled_profit\n")
        # 데이터
        for link in connected_trades:
            f.write(f"{link.get('trade_id')},{link.get('entry_ts')},{link.get('exit_ts')},")
            f.write(f"{link.get('side')},{link.get('entry_price')},{link.get('exit_price')},{link.get('holding_bars')},")
            f.write(f"{link.get('p_long')},{link.get('margin')},{link.get('entropy')},{link.get('p_diff')},")
            f.write(f"{link.get('guard_scale')},{link.get('stage2_cap')},{link.get('final_scale')},{link.get('cap_reason')},")
            f.write(f"{link.get('realized_profit')},{link.get('scaled_profit')}\n")
    
    logger.info(f"리포트 생성 완료:")
    logger.info(f"  - {summary_path}")
    logger.info(f"  - {samples_path} ({len(debug_logs)} lines)")
    logger.info(f"  - {csv_path} ({len([l for l in entry_links if 'exit_ts' in l])} trades)")
    
    return summary_path, samples_path, csv_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True, help="로그 파일 경로")
    parser.add_argument("--output-dir", type=str, default=None, help="출력 디렉토리 (기본값: data/backtest_reports)")
    
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "data" / "backtest_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = extract_cap_debug_from_log(log_file)
    generate_report(result, output_dir)
