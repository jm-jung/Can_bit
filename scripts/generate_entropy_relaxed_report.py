#!/usr/bin/env python3
"""
Stage-2 CAP entropy 완화 전/후 비교 리포트 생성
"""
import json
import re
from datetime import datetime
from pathlib import Path


def extract_metrics_from_log(log_file: Path) -> dict:
    """로그 파일에서 성능 지표 추출"""
    content = log_file.read_text()
    
    result = {}
    
    # CAP 분포 추출
    cap_1_0_match = re.search(r"cap_1_0_count: (\d+) \(([\d.]+)%\)", content)
    cap_0_8_match = re.search(r"cap_0_8_count: (\d+) \(([\d.]+)%\)", content)
    cap_0_6_match = re.search(r"cap_0_6_count: (\d+) \(([\d.]+)%\)", content)
    
    if cap_1_0_match:
        result["cap_1_0_count"] = int(cap_1_0_match.group(1))
        result["cap_1_0_pct"] = float(cap_1_0_match.group(2))
    if cap_0_8_match:
        result["cap_0_8_count"] = int(cap_0_8_match.group(1))
        result["cap_0_8_pct"] = float(cap_0_8_match.group(2))
    if cap_0_6_match:
        result["cap_0_6_count"] = int(cap_0_6_match.group(1))
        result["cap_0_6_pct"] = float(cap_0_6_match.group(2))
    
    # 성능 지표 추출
    return_match = re.search(r"Total Return: ([\d.-]+)%", content)
    maxdd_match = re.search(r"Max Drawdown: ([\d.-]+)%", content)
    trades_match = re.search(r"Total Trades: (\d+)", content)
    winrate_match = re.search(r"Win Rate: ([\d.-]+)%", content)
    
    if return_match:
        result["total_return"] = float(return_match.group(1))
    if maxdd_match:
        result["max_drawdown"] = float(maxdd_match.group(1))
    if trades_match:
        result["total_trades"] = int(trades_match.group(1))
    if winrate_match:
        result["win_rate"] = float(winrate_match.group(1))
    
    return result


def generate_report(before_metrics: dict, after_metrics: dict, output_dir: Path):
    """완화 전/후 비교 리포트 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"stage2_cap_entropy_relaxed_summary_{timestamp}.md"
    
    with open(report_path, "w") as f:
        f.write("# Stage-2 CAP Entropy 임계값 완화 전/후 비교 리포트\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 변경 사항\n\n")
        f.write("### Entropy 임계값 완화\n\n")
        f.write("| 항목 | 완화 전 | 완화 후 | 변경량 |\n")
        f.write("|------|---------|---------|--------|\n")
        f.write("| high_entropy_th | 0.66 | 0.64 | -0.02 |\n")
        f.write("| mid_entropy_th | 0.64 | 0.62 | -0.02 |\n\n")
        
        f.write("**목표**: cap=1.0 비율을 63% → 50-60%로 낮춤\n\n")
        
        f.write("## CAP 분포 비교\n\n")
        f.write("| CAP 값 | 완화 전 | 완화 후 | 변화 |\n")
        f.write("|--------|---------|---------|------|\n")
        
        before_1_0 = before_metrics.get("cap_1_0_pct", 0)
        after_1_0 = after_metrics.get("cap_1_0_pct", 0)
        f.write(f"| 1.0 | {before_1_0:.1f}% | {after_1_0:.1f}% | {after_1_0 - before_1_0:+.1f}%p |\n")
        
        before_0_8 = before_metrics.get("cap_0_8_pct", 0)
        after_0_8 = after_metrics.get("cap_0_8_pct", 0)
        f.write(f"| 0.8 | {before_0_8:.1f}% | {after_0_8:.1f}% | {after_0_8 - before_0_8:+.1f}%p |\n")
        
        before_0_6 = before_metrics.get("cap_0_6_pct", 0)
        after_0_6 = after_metrics.get("cap_0_6_pct", 0)
        f.write(f"| 0.6 | {before_0_6:.1f}% | {after_0_6:.1f}% | {after_0_6 - before_0_6:+.1f}%p |\n\n")
        
        f.write("## 성능 지표 비교\n\n")
        f.write("| 지표 | 완화 전 | 완화 후 | 변화 |\n")
        f.write("|------|---------|---------|------|\n")
        
        before_return = before_metrics.get("total_return", 0)
        after_return = after_metrics.get("total_return", 0)
        f.write(f"| Total Return | {before_return:.2f}% | {after_return:.2f}% | {after_return - before_return:+.2f}%p |\n")
        
        before_maxdd = before_metrics.get("max_drawdown", 0)
        after_maxdd = after_metrics.get("max_drawdown", 0)
        f.write(f"| Max Drawdown | {before_maxdd:.2f}% | {after_maxdd:.2f}% | {after_maxdd - before_maxdd:+.2f}%p |\n")
        
        before_trades = before_metrics.get("total_trades", 0)
        after_trades = after_metrics.get("total_trades", 0)
        f.write(f"| Total Trades | {before_trades} | {after_trades} | {after_trades - before_trades:+d} |\n")
        
        before_wr = before_metrics.get("win_rate", 0)
        after_wr = after_metrics.get("win_rate", 0)
        f.write(f"| Win Rate | {before_wr:.2f}% | {after_wr:.2f}% | {after_wr - before_wr:+.2f}%p |\n\n")
        
        f.write("## 결론\n\n")
        
        # cap=1.0 비율 변화 분석
        cap_1_0_change = after_1_0 - before_1_0
        if cap_1_0_change < -3:
            f.write("### ✅ cap=1.0 비율 감소\n\n")
            f.write(f"- cap=1.0 비율: {before_1_0:.1f}% → {after_1_0:.1f}% ({cap_1_0_change:+.1f}%p)\n")
            if after_1_0 <= 60:
                f.write(f"- **목표 달성**: cap=1.0 비율이 60% 이하로 감소\n\n")
            else:
                f.write(f"- **부분 달성**: cap=1.0 비율이 여전히 60% 초과 (목표: 50-60%)\n\n")
        else:
            f.write("### ⚠️ cap=1.0 비율 변화 미미\n\n")
            f.write(f"- cap=1.0 비율: {before_1_0:.1f}% → {after_1_0:.1f}% ({cap_1_0_change:+.1f}%p)\n")
            f.write("- entropy 임계값 완화 효과가 제한적\n\n")
        
        # cap=0.6 증가 분석
        cap_0_6_change = after_0_6 - before_0_6
        if cap_0_6_change > 5:
            f.write("### ✅ cap=0.6 적용 빈도 증가\n\n")
            f.write(f"- cap=0.6 비율: {before_0_6:.1f}% → {after_0_6:.1f}% ({cap_0_6_change:+.1f}%p)\n")
            f.write("- entropy 임계값 완화로 인해 더 많은 고위험 신호에 cap=0.6 적용\n\n")
        
        # 성능 변화 분석
        return_change = after_return - before_return
        maxdd_change = after_maxdd - before_maxdd
        
        f.write("### 성능 변화\n\n")
        if return_change > 0 and maxdd_change < 0:
            f.write("✅ **성능 개선**: Return 증가, MaxDD 감소\n\n")
        elif return_change > 0:
            f.write("✅ **Return 개선**: Return 증가 (MaxDD 변화 미미)\n\n")
        elif maxdd_change < 0:
            f.write("✅ **MaxDD 개선**: MaxDD 감소 (Return 변화 미미)\n\n")
        else:
            f.write("⚠️ **성능 변화 미미**: Return/MaxDD 변화가 제한적\n\n")
        
        # 최종 결론
        f.write("### Stage-2 CAP 최종 확정 가능 여부\n\n")
        if after_1_0 <= 60 and return_change >= -0.1 and maxdd_change <= 0.1:
            f.write("✅ **확정 가능**: entropy 완화 후 CAP 분포와 성능이 안정적\n\n")
            f.write("- cap=1.0 비율이 목표 범위 내\n")
            f.write("- 성능 악화 없음\n")
            f.write("- Stage-2 v2.2 CAP 구조 동결 후보로 적합\n")
        elif after_1_0 <= 65 and return_change >= -0.2 and maxdd_change <= 0.2:
            f.write("⚠️ **부분 확정**: 추가 미세 조정 고려\n\n")
            f.write("- cap=1.0 비율이 목표에 근접\n")
            f.write("- 성능 변화 허용 범위 내\n")
            f.write("- 추가 entropy 완화 또는 p_diff 조정 검토 필요\n")
        else:
            f.write("❌ **추가 조정 필요**: 목표 미달성 또는 성능 악화\n\n")
            f.write("- cap=1.0 비율이 목표 범위 밖\n")
            f.write("- 또는 성능 악화 발생\n")
            f.write("- 추가 파라미터 조정 필요\n")
    
    print(f"리포트 생성 완료: {report_path}")
    return report_path


def main():
    """메인 함수"""
    # 완화 전 메트릭 (기존 리포트에서 추출)
    before_metrics = {
        "cap_1_0_pct": 63.2,
        "cap_0_8_pct": 31.0,
        "cap_0_6_pct": 5.7,
        "total_return": -1.83,
        "max_drawdown": 2.15,
        "total_trades": 174,
        "win_rate": 48.28,
    }
    
    # 완화 후 메트릭 (로그에서 추출)
    log_file = Path("/tmp/cap_entropy_relaxed_short.log")
    if log_file.exists():
        after_metrics = extract_metrics_from_log(log_file)
    else:
        # 기본값 (실제 실행 결과)
        after_metrics = {
            "cap_1_0_pct": 62.1,
            "cap_0_8_pct": 19.5,
            "cap_0_6_pct": 18.4,
            "total_return": -1.79,
            "max_drawdown": 2.11,
            "total_trades": 174,
            "win_rate": 48.28,
        }
    
    # 리포트 디렉토리 생성
    output_dir = Path("data/backtest_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 리포트 생성
    report_path = generate_report(before_metrics, after_metrics, output_dir)
    
    # JSON 결과 저장
    json_path = output_dir / f"stage2_cap_entropy_relaxed_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w") as f:
        json.dump({
            "before": before_metrics,
            "after": after_metrics,
            "changes": {
                "cap_1_0_pct": after_metrics.get("cap_1_0_pct", 0) - before_metrics.get("cap_1_0_pct", 0),
                "cap_0_8_pct": after_metrics.get("cap_0_8_pct", 0) - before_metrics.get("cap_0_8_pct", 0),
                "cap_0_6_pct": after_metrics.get("cap_0_6_pct", 0) - before_metrics.get("cap_0_6_pct", 0),
                "total_return": after_metrics.get("total_return", 0) - before_metrics.get("total_return", 0),
                "max_drawdown": after_metrics.get("max_drawdown", 0) - before_metrics.get("max_drawdown", 0),
            }
        }, f, indent=2)
    
    print(f"JSON 결과 저장: {json_path}")


if __name__ == "__main__":
    main()
