#!/usr/bin/env python3
"""
Stage-2 CAP 로그 기반 최종 확정 리포트 생성 스크립트

백테스트 결과에서 CAP 디버그 로그를 추출하여 리포트를 생성합니다.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
        for i, log in enumerate(cap_0_6_logs[:5], 1):
            f.write(f"### 케이스 {i}\n\n")
            f.write(f"- ts: {log.get('ts')}\n")
            f.write(f"- entropy: {log.get('entropy')}\n")
            f.write(f"- p_diff: {log.get('p_diff')}\n")
            f.write(f"- cap_reason: {log.get('cap_reason')}\n")
            f.write(f"- trigger_type: {log.get('trigger_type')}\n\n")
        
        # ENTRY/EXIT 연결 샘플
        links = result.get("cap_entry_exit_links", [])
        entry_links = [l for l in links if l.get("event") == "ENTRY"]
        f.write("## ENTRY/EXIT 연결 샘플 (cap별 성과 비교)\n\n")
        f.write("| cap | trade_count | mean_profit | mean_holding |\n")
        f.write("|-----|-------------|-------------|--------------|\n")
        for cap_val in [1.0, 0.8, 0.6]:
            cap_trades = [l for l in entry_links if abs(l.get("stage2_cap", 1.0) - cap_val) < 0.001 and "scaled_profit" in l]
            if cap_trades:
                profits = [t.get("scaled_profit", 0) for t in cap_trades]
                holdings = [t.get("holding_bars", 0) for t in cap_trades]
                f.write(f"| {cap_val} | {len(cap_trades)} | {sum(profits) / len(profits):.6f} | {sum(holdings) / len(holdings):.1f} |\n")
        
        # 결론
        f.write("\n## 결론\n\n")
        f.write("### 왜 cap=1.0이 70%인가?\n\n")
        f.write("(분석 결과 기반으로 작성)\n\n")
        f.write("### cap=0.6 유지/제거 추천\n\n")
        f.write("(샘플수, 성과, 케이스 로그 기반으로 작성)\n\n")
        f.write("### entropy vs p_diff 중 무엇을 조정할지 추천\n\n")
        f.write("(trigger_by_rule 분포 기반으로 작성)\n\n")
    
    # 2) stage2_cap_debug_samples_<ts>.jsonl
    samples_path = output_dir / f"stage2_cap_debug_samples_{timestamp}.jsonl"
    debug_logs = result.get("cap_debug_logs", [])
    with open(samples_path, "w") as f:
        for log in debug_logs:
            f.write(json.dumps(log) + "\n")
    
    # 3) stage2_cap_entry_exit_link_<ts>.csv
    csv_path = output_dir / f"stage2_cap_entry_exit_link_{timestamp}.csv"
    links = result.get("cap_entry_exit_links", [])
    entry_links = [l for l in links if l.get("event") == "ENTRY"]
    with open(csv_path, "w") as f:
        # 헤더
        f.write("trade_id,entry_ts,exit_ts,side,entry_price,exit_price,holding_bars,")
        f.write("p_long,margin,entropy,p_diff,guard_scale,stage2_cap,final_scale,cap_reason,")
        f.write("realized_profit,scaled_profit\n")
        # 데이터
        for link in entry_links:
            if "exit_ts" in link:  # EXIT 정보가 있는 것만
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
    # 백테스트 결과를 읽어서 리포트 생성
    # 실제로는 백테스트 실행 후 result를 전달받아야 함
    # 여기서는 예시로 빈 dict를 사용
    result = {}
    output_dir = PROJECT_ROOT / "data" / "backtest_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_report(result, output_dir)
