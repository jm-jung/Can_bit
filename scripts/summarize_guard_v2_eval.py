#!/usr/bin/env python3
"""
Guard v2 평가 결과 요약 스크립트

실험 결과를 읽어서 markdown 요약 문서를 생성합니다.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments" / "guard_v2_eval"
DOCS_DIR = PROJECT_ROOT / "docs"

def load_results(run_id: str) -> dict:
    """실험 결과 로드"""
    summary_file = EXPERIMENTS_DIR / run_id / "summary.json"
    
    if not summary_file.exists():
        # 개별 실험 결과 로드
        results = {}
        for exp_id in ["exp1_guard_on", "exp2_guard_on_cost_off", "exp3_guard_off"]:
            metrics_file = EXPERIMENTS_DIR / run_id / exp_id / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    results[exp_id] = json.load(f)
        return results
    
    with open(summary_file, "r") as f:
        return json.load(f)

def format_percentage(value):
    """퍼센트 포맷"""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"

def format_number(value):
    """숫자 포맷"""
    if value is None:
        return "N/A"
    return f"{value:,}"

def generate_summary(run_id: str) -> str:
    """요약 문서 생성"""
    results = load_results(run_id)
    
    # 실험 커맨드 추출
    exp1_cmd = results.get("exp1", {}).get("command", "N/A")
    exp2_cmd = results.get("exp2", {}).get("command", "N/A")
    exp3_cmd = results.get("exp3", {}).get("command", "N/A")
    
    # 실험 1 결과
    exp1 = results.get("exp1", {})
    exp1_scale_stats = exp1.get("scale_stats", {})
    exp1_decisions = exp1.get("decision_counts", {})
    
    # 실험 2 결과
    exp2 = results.get("exp2", {})
    
    # 실험 3 결과
    exp3 = results.get("exp3", {})
    
    # 결론 생성
    conclusions = []
    recommendations = []
    
    # 실험 2 결론 (비용 영향)
    # Note: 현재 실험 2는 비용 OFF 옵션이 없어 실험 1과 동일하게 실행됨
    # 실제 비용 영향 분석은 추후 commission/slippage 옵션 추가 후 재실험 필요
    if exp1.get("total_return") is not None and exp2.get("total_return") is not None:
        cost_impact = exp2.get("total_return", 0) - exp1.get("total_return", 0)
        if abs(cost_impact) < 0.01:
            conclusions.append("**실험 2는 비용 OFF 옵션 미지원으로 실험 1과 동일 실행됨** → 비용 영향 분석은 추후 재실험 필요")
        elif cost_impact > 50:
            conclusions.append("**비용 OFF에서 크게 개선** → 비용/빈도 문제가 주요 원인")
            recommendations.append("거래 빈도 추가 감소 (min_hold/cooldown 증가) 또는 수수료율 재검토")
        elif exp2.get("total_return", 0) < -50:
            conclusions.append("**비용 OFF에서도 여전히 큰 손실** → 신호/전략 자체 문제")
            recommendations.append("Stage-2/threshold 재검증 또는 신호 품질 개선 필요")
        else:
            conclusions.append("**비용 영향 중간** → 비용과 신호 품질 모두 개선 필요")
    
    # 실험 3 결론 (Baseline 비교)
    if exp1.get("total_return") is not None and exp3.get("total_return") is not None:
        return_diff = exp1.get("total_return", 0) - exp3.get("total_return", 0)
        dd_diff = exp1.get("max_drawdown", 0) - exp3.get("max_drawdown", 0)
        
        if return_diff > 5 or dd_diff < -5:
            conclusions.append("**Guard v2가 Baseline 대비 개선** → 유지/튜닝 가치 있음")
            recommendations.append("margin/entropy 파라미터 미니 스윕 (12-run)으로 최적화")
        else:
            conclusions.append("**Guard v2가 Baseline 대비 악화 또는 유사** → Guard v2 구조 재검토 필요")
            recommendations.append("Guard v2 접근 방식 재설계 또는 Stage-2/threshold 우선 개선")
    
    # 실험 1 결론 (scale 분포)
    if exp1_scale_stats:
        scale_range = exp1_scale_stats.get("max", 0) - exp1_scale_stats.get("min", 0)
        if scale_range < 0.1:
            conclusions.append("**scale 분포가 좁음** → Guard v2 조절 효과 제한적")
            recommendations.append("scale_floor 낮추기 또는 margin/entropy 파라미터 조정")
        else:
            conclusions.append("**scale 분포가 적절함** → Guard v2가 의미 있게 조절 중")
    
    # 추가 추천 (실험 결과 기반)
    if exp1.get("total_return") is not None and exp3.get("total_return") is not None:
        if exp1.get("total_return", 0) > exp3.get("total_return", 0) and exp1.get("max_drawdown", 100) < exp3.get("max_drawdown", 100):
            recommendations.append("Guard v2가 return과 drawdown 모두 개선 → 현재 설정 유지하며 파라미터 미세 조정")
        elif exp1.get("total_trades", 0) > exp3.get("total_trades", 0) * 5:
            recommendations.append("Guard v2가 trades를 크게 증가시킴 → min_hold/cooldown 추가 증가 또는 Guard v2 BLOCK 임계값 강화")
    
    # 기본 추천 (추천이 부족한 경우)
    if len(recommendations) < 3:
        recommendations.append("Stage-2/threshold 재검증으로 신호 품질 개선")
    
    md = f"""# Guard v2 평가 실험 요약

**실행 ID:** {run_id}  
**생성일:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 실험 실행 커맨드

### 실험 1: Guard v2 ON (비용 ON)

```bash
{exp1_cmd}
```

### 실험 2: Guard v2 ON (비용 OFF)

```bash
{exp2_cmd}
```

### 실험 3: Guard OFF (Baseline)

```bash
{exp3_cmd}
```

---

## 실험 1: Guard 개입 정도 분포 확인

### 결과 요약

| 지표 | 값 |
|------|-----|
| Total Return | {format_percentage(exp1.get('total_return'))} |
| Max Drawdown | {format_percentage(exp1.get('max_drawdown'))} |
| Total Trades | {format_number(exp1.get('total_trades'))} |
| Avg Holding | {exp1.get('avg_holding_bars', 'N/A')} bars |
| Win Rate | {format_percentage(exp1.get('win_rate'))} |

### Scale 통계

"""
    
    if exp1_scale_stats:
        md += f"""| 통계 | 값 |
|------|-----|
| Count | {exp1_scale_stats.get('count', 'N/A')} |
| Min | {exp1_scale_stats.get('min', 0):.4f} |
| P10 | {exp1_scale_stats.get('p10', 0):.4f} |
| Median | {exp1_scale_stats.get('median', 0):.4f} |
| P90 | {exp1_scale_stats.get('p90', 0):.4f} |
| Max | {exp1_scale_stats.get('max', 0):.4f} |

### Scale 히스토그램 (10 bins)

"""
        histogram = exp1.get("scale_histogram", {})
        if histogram:
            counts = histogram.get("counts", [])
            bin_min = histogram.get("min", 0)
            bin_max = histogram.get("max", 1)
            bin_width = (bin_max - bin_min) / len(counts) if counts else 1
            
            md += "| Bin Range | Count |\n"
            md += "|-----------|-------|\n"
            for i, count in enumerate(counts):
                bin_start = bin_min + i * bin_width
                bin_end = bin_min + (i + 1) * bin_width
                md += f"| {bin_start:.3f} ~ {bin_end:.3f} | {count} |\n"
    
    md += f"""
### Decision 카운트

| Decision | Count |
|----------|-------|
| ALLOW | {exp1_decisions.get('ALLOW', 0)} |
| BLOCK | {exp1_decisions.get('BLOCK', 0)} |
| DEFER | {exp1_decisions.get('DEFER', 0)} |

---

## 실험 2: 거래 비용 vs 신호 엣지 분리

### 비용 ON vs OFF 비교

| 지표 | 비용 ON (실험1) | 비용 OFF (실험2) | 차이 |
|------|----------------|------------------|------|
| Total Return | {format_percentage(exp1.get('total_return'))} | {format_percentage(exp2.get('total_return'))} | {format_percentage((exp2.get('total_return') or 0) - (exp1.get('total_return') or 0))} |
| Max Drawdown | {format_percentage(exp1.get('max_drawdown'))} | {format_percentage(exp2.get('max_drawdown'))} | {format_percentage((exp2.get('max_drawdown') or 0) - (exp1.get('max_drawdown') or 0))} |
| Total Trades | {format_number(exp1.get('total_trades'))} | {format_number(exp2.get('total_trades'))} | {format_number((exp2.get('total_trades') or 0) - (exp1.get('total_trades') or 0))} |
| Avg Holding | {exp1.get('avg_holding_bars', 'N/A')} bars | {exp2.get('avg_holding_bars', 'N/A')} bars | - |
| Win Rate | {format_percentage(exp1.get('win_rate'))} | {format_percentage(exp2.get('win_rate'))} | {format_percentage((exp2.get('win_rate') or 0) - (exp1.get('win_rate') or 0))} |

### 결론

{chr(10).join(f"- {c}" for c in conclusions if "비용" in c or "비용" in str(c))}

---

## 실험 3: Baseline(Guard OFF) 공정 비교

### Guard ON vs OFF 비교

| 지표 | Guard ON (실험1) | Guard OFF (실험3) | 차이 |
|------|------------------|-------------------|------|
| Total Return | {format_percentage(exp1.get('total_return'))} | {format_percentage(exp3.get('total_return'))} | {format_percentage((exp1.get('total_return') or 0) - (exp3.get('total_return') or 0))} |
| Max Drawdown | {format_percentage(exp1.get('max_drawdown'))} | {format_percentage(exp3.get('max_drawdown'))} | {format_percentage((exp1.get('max_drawdown') or 0) - (exp3.get('max_drawdown') or 0))} |
| Total Trades | {format_number(exp1.get('total_trades'))} | {format_number(exp3.get('total_trades'))} | {format_number((exp1.get('total_trades') or 0) - (exp3.get('total_trades') or 0))} |
| Avg Holding | {exp1.get('avg_holding_bars', 'N/A')} bars | {exp3.get('avg_holding_bars', 'N/A')} bars | - |
| Win Rate | {format_percentage(exp1.get('win_rate'))} | {format_percentage(exp3.get('win_rate'))} | {format_percentage((exp1.get('win_rate') or 0) - (exp3.get('win_rate') or 0))} |

### 결론

{chr(10).join(f"- {c}" for c in conclusions if "Baseline" in c or "Guard v2" in c)}

---

## 종합 결론

{chr(10).join(f"- {c}" for c in conclusions)}

---

## 다음 액션 추천

{chr(10).join(f"1. **추천 {i+1}:** {r}" for i, r in enumerate(recommendations[:3]))}

---

**참고:** 전체 실험 결과는 `data/experiments/guard_v2_eval/{run_id}/` 디렉토리에 저장되어 있습니다.
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
                print("실험 결과가 없습니다. 먼저 scripts/run_guard_v2_eval.py를 실행하세요.")
                sys.exit(1)
        else:
            print("실험 결과가 없습니다. 먼저 scripts/run_guard_v2_eval.py를 실행하세요.")
            sys.exit(1)
    else:
        run_id = sys.argv[1]
    
    md_content = generate_summary(run_id)
    
    # 문서 저장
    output_file = DOCS_DIR / "guard_v2_eval_summary.md"
    with open(output_file, "w") as f:
        f.write(md_content)
    
    print(f"요약 문서 생성 완료: {output_file}")

if __name__ == "__main__":
    main()
