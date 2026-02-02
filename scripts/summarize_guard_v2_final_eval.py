#!/usr/bin/env python3
"""
Guard v2 실전 가능 여부 평가 결과 요약

STEP별 결과를 정리하고 최종 결론을 생성합니다.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments" / "guard_v2_final_eval"
DOCS_DIR = PROJECT_ROOT / "docs"

# 기준 결과
BASELINE_RETURN = -13.89
BASELINE_MAXDD = 13.77
CURRENT_RETURN = -1.99
CURRENT_MAXDD = 2.32

def load_results(run_id: str) -> dict:
    """실험 결과 로드"""
    summary_file = EXPERIMENTS_DIR / run_id / "summary.json"
    
    if not summary_file.exists():
        print(f"요약 파일이 없습니다: {summary_file}")
        sys.exit(1)
    
    with open(summary_file, "r") as f:
        return json.load(f)

def format_pct(value):
    """퍼센트 포맷"""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"

def format_num(value):
    """숫자 포맷"""
    if value is None:
        return "N/A"
    return f"{value:,}"

def generate_report(run_id: str) -> str:
    """최종 보고서 생성"""
    results = load_results(run_id)
    
    # STEP 1 결과
    step1 = results.get("step1", {})
    cost_on = step1.get("cost_on", {})
    cost_off = step1.get("cost_off", {})
    
    # STEP 2 결과
    step2 = results.get("step2", {})
    
    # STEP 3 결과
    step3 = results.get("step3", {})
    
    # 커맨드 추출
    commands = []
    if cost_on.get("command"):
        commands.append(("STEP 1-1: Guard v2 ON, 비용 ON", cost_on["command"]))
    if cost_off.get("command"):
        commands.append(("STEP 1-2: Guard v2 ON, 비용 OFF", cost_off["command"]))
    
    # STEP 2 커맨드 샘플
    if step2:
        first_run = list(step2.values())[0]
        if first_run.get("command"):
            commands.append(("STEP 2: margin/entropy 스윕 (샘플)", first_run["command"]))
    
    # STEP 3 커맨드
    if step3:
        for key, value in step3.items():
            if value.get("command"):
                commands.append((f"STEP 3: {key}", value["command"]))
    
    # 결론 생성
    conclusions = []
    recommendations = []
    
    # STEP 1 결론
    if cost_off.get("total_return") is not None:
        cost_impact = cost_off.get("total_return", 0) - cost_on.get("total_return", 0)
        if cost_off.get("total_return", 0) > 0 or abs(cost_off.get("total_return", 0)) < 1:
            conclusions.append("**비용 OFF에서 수익 또는 0 근처** → 거래 빈도/비용 최적화가 다음 과제")
            recommendations.append("거래 빈도 추가 감소 (min_hold/cooldown 증가) 또는 수수료율 재검토")
        else:
            conclusions.append("**비용 OFF에서도 음수** → 신호/Stage2/threshold가 다음 과제")
            recommendations.append("Stage-2/threshold 재검증 또는 신호 품질 개선")
    
    # STEP 2 결론
    if step2:
        best_runs = sorted(
            [(k, v) for k, v in step2.items() if v.get("total_return") is not None],
            key=lambda x: (x[1].get("total_return", -999), -x[1].get("max_drawdown", 999)),
            reverse=True
        )[:3]
        
        if best_runs:
            best = best_runs[0][1]
            if best.get("total_return", 0) > CURRENT_RETURN:
                conclusions.append(f"**STEP 2 최적 조합이 현재보다 개선** (Return: {format_pct(best.get('total_return'))})")
                recommendations.append(f"최적 파라미터 적용: min_margin={best.get('min_margin')}, max_entropy={best.get('max_entropy')}")
            else:
                conclusions.append("**STEP 2 모든 조합이 현재보다 악화 또는 유사**")
    
    # STEP 3 결론
    if step3:
        stage2_off = step3.get("stage2_off", {})
        stage2_on = step3.get("stage2_on", {})
        if stage2_off.get("total_return") and stage2_on.get("total_return"):
            if stage2_on.get("total_return", 0) > stage2_off.get("total_return", 0):
                conclusions.append("**Stage-2 ON이 OFF보다 개선** → Stage-2 활성화 권장")
            else:
                conclusions.append("**Stage-2 효과 제한적** → Stage-2 비활성화 유지")
    
    # 최종 판정
    # STEP 2 최적 조합이 있으면 그것을 기준으로 판정
    best_step2 = None
    if step2:
        best_runs = sorted(
            [(k, v) for k, v in step2.items() if v.get("total_return") is not None],
            key=lambda x: (x[1].get("total_return", -999), -x[1].get("max_drawdown", 999)),
            reverse=True
        )
        if best_runs:
            best_step2 = best_runs[0][1]
    
    final_return = best_step2.get("total_return") if best_step2 else cost_on.get("total_return", CURRENT_RETURN)
    final_maxdd = best_step2.get("max_drawdown") if best_step2 else cost_on.get("max_drawdown", CURRENT_MAXDD)
    
    is_production_ready = (
        final_return > 0 or
        (final_return > BASELINE_RETURN and final_maxdd < BASELINE_MAXDD and abs(final_return) < 1.0)
    )
    
    bottleneck = "미확정"
    if cost_off.get("total_return", 0) > 0 or abs(cost_off.get("total_return", 999)) < 0.5:
        bottleneck = "비용/빈도"
    elif cost_off.get("total_return", 0) < -5:
        bottleneck = "신호/Stage2"
    else:
        bottleneck = "비용+신호 혼합"
    
    next_action = "추가 스윕 필요"
    if is_production_ready:
        if best_step2:
            next_action = f"최적 파라미터 적용 (min_margin={best_step2.get('min_margin')}, max_entropy={best_step2.get('max_entropy')}) 및 실전 모니터링"
        else:
            next_action = "실전용 파라미터 고정 및 모니터링"
    elif bottleneck == "비용/빈도":
        next_action = "거래 빈도 추가 감소 (min_hold/cooldown 증가)"
    elif bottleneck == "신호/Stage2":
        next_action = "Stage-2/threshold 재검증 또는 모델 교체"
    else:
        if best_step2 and best_step2.get("total_return", 0) > cost_on.get("total_return", -999):
            next_action = f"최적 파라미터 적용: min_margin={best_step2.get('min_margin')}, max_entropy={best_step2.get('max_entropy')}"
        else:
            next_action = recommendations[0] if recommendations else "추가 분석 필요"
    
    md = f"""# Guard v2 실전 가능 여부 평가 최종 보고서

**실행 ID:** {run_id}  
**생성일:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 실행 커맨드

"""
    
    for title, cmd in commands:
        md += f"### {title}\n\n```bash\n{cmd}\n```\n\n"
    
    md += f"""---

## STEP 1: 거래 비용 OFF 실험

### 비용 ON vs OFF 비교

| 지표 | 비용 ON | 비용 OFF | 차이 |
|------|---------|----------|------|
| Total Return | {format_pct(cost_on.get('total_return'))} | {format_pct(cost_off.get('total_return'))} | {format_pct((cost_off.get('total_return') or 0) - (cost_on.get('total_return') or 0))} |
| Max Drawdown | {format_pct(cost_on.get('max_drawdown'))} | {format_pct(cost_off.get('max_drawdown'))} | {format_pct((cost_off.get('max_drawdown') or 0) - (cost_on.get('max_drawdown') or 0))} |
| Total Trades | {format_num(cost_on.get('total_trades'))} | {format_num(cost_off.get('total_trades'))} | {format_num((cost_off.get('total_trades') or 0) - (cost_on.get('total_trades') or 0))} |
| Avg Holding | {cost_on.get('avg_holding_bars', 'N/A')} bars | {cost_off.get('avg_holding_bars', 'N/A')} bars | - |
| Win Rate | {format_pct(cost_on.get('win_rate'))} | {format_pct(cost_off.get('win_rate'))} | {format_pct((cost_off.get('win_rate') or 0) - (cost_on.get('win_rate') or 0))} |

### 결론

{chr(10).join(f"- {c}" for c in conclusions if "비용" in c or "비용" in str(c))}

---

## STEP 2: margin/entropy 미니 스윕 (12-run)

### 상위 3개 조합

"""
    
    if step2:
        best_runs = sorted(
            [(k, v) for k, v in step2.items() if v.get("total_return") is not None],
            key=lambda x: (x[1].get("total_return", -999), -x[1].get("max_drawdown", 999)),
            reverse=True
        )[:3]
        
        md += "| 순위 | min_margin | max_entropy | Total Return | Max Drawdown | Total Trades | Median Scale | P90 Scale |\n"
        md += "|------|------------|-------------|--------------|--------------|--------------|--------------|-----------|\n"
        
        for idx, (run_name, metrics) in enumerate(best_runs, 1):
            scale_stats = metrics.get("scale_stats", {})
            md += f"| {idx} | {metrics.get('min_margin', 'N/A')} | {metrics.get('max_entropy', 'N/A')} | "
            md += f"{format_pct(metrics.get('total_return'))} | {format_pct(metrics.get('max_drawdown'))} | "
            md += f"{format_num(metrics.get('total_trades'))} | "
            md += f"{scale_stats.get('median', 0):.3f} | {scale_stats.get('p90', 0):.3f} |\n"
        
        md += "\n### Baseline 및 현재 대비\n\n"
        md += "| 기준 | Total Return | Max Drawdown |\n"
        md += "|------|--------------|--------------|\n"
        md += f"| Baseline (Guard OFF) | {format_pct(BASELINE_RETURN)} | {format_pct(BASELINE_MAXDD)} |\n"
        md += f"| 현재 (Guard v2) | {format_pct(CURRENT_RETURN)} | {format_pct(CURRENT_MAXDD)} |\n"
        
        if best_runs:
            best = best_runs[0][1]
            md += f"| STEP 2 최적 | {format_pct(best.get('total_return'))} | {format_pct(best.get('max_drawdown'))} |\n"
    
    md += f"""

---

## STEP 3: Stage-2/threshold 재검증

"""
    
    if step3:
        md += "### Stage-2 ON vs OFF 비교\n\n"
        md += "| 지표 | Stage-2 OFF | Stage-2 ON | 차이 |\n"
        md += "|------|------------|------------|------|\n"
        
        stage2_off = step3.get("stage2_off", {})
        stage2_on = step3.get("stage2_on", {})
        
        md += f"| Total Return | {format_pct(stage2_off.get('total_return'))} | {format_pct(stage2_on.get('total_return'))} | "
        md += f"{format_pct((stage2_on.get('total_return') or 0) - (stage2_off.get('total_return') or 0))} |\n"
        md += f"| Max Drawdown | {format_pct(stage2_off.get('max_drawdown'))} | {format_pct(stage2_on.get('max_drawdown'))} | "
        md += f"{format_pct((stage2_on.get('max_drawdown') or 0) - (stage2_off.get('max_drawdown') or 0))} |\n"
        
        threshold_default = step3.get("threshold_default", {})
        if threshold_default.get("total_return") is not None:
            md += "\n### Threshold 비교 (optimized vs default)\n\n"
            md += "| 지표 | Optimized | Default (0.5) | 차이 |\n"
            md += "|------|-----------|---------------|------|\n"
            md += f"| Total Return | {format_pct(cost_on.get('total_return'))} | {format_pct(threshold_default.get('total_return'))} | "
            md += f"{format_pct((threshold_default.get('total_return') or 0) - (cost_on.get('total_return') or 0))} |\n"
    else:
        md += "*STEP 3는 조건 미충족으로 실행되지 않았습니다.*\n"
    
    md += f"""

---

## 종합 결론

{chr(10).join(f"- {c}" for c in conclusions)}

---

## 최종 판정

### Guard v2 실전 가능 여부

**{'✅ YES' if is_production_ready else '❌ NO'}**

- 현재 Return: {format_pct(final_return)}
- 현재 MaxDD: {format_pct(final_maxdd)}
- Baseline 대비: Return {format_pct(final_return - BASELINE_RETURN)}, MaxDD {format_pct(final_maxdd - BASELINE_MAXDD)}

### 현재 병목

**{bottleneck}**

### 다음 액션

**{next_action}**

---

**참고:** 전체 실험 결과는 `data/experiments/guard_v2_final_eval/{run_id}/` 디렉토리에 저장되어 있습니다.
"""
    
    return md

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        # 최신 run_id 찾기
        if EXPERIMENTS_DIR.exists():
            run_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()], reverse=True)
            if run_dirs:
                run_id = run_dirs[0].name
                print(f"최신 run_id 사용: {run_id}")
            else:
                print("실험 결과가 없습니다. 먼저 scripts/run_guard_v2_final_eval.py를 실행하세요.")
                sys.exit(1)
        else:
            print("실험 결과가 없습니다. 먼저 scripts/run_guard_v2_final_eval.py를 실행하세요.")
            sys.exit(1)
    else:
        run_id = sys.argv[1]
    
    md_content = generate_report(run_id)
    
    # 문서 저장
    output_file = DOCS_DIR / "guard_v2_final_eval.md"
    with open(output_file, "w") as f:
        f.write(md_content)
    
    print(f"최종 보고서 생성 완료: {output_file}")

if __name__ == "__main__":
    main()
