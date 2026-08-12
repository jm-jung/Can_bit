# Phase D13 Implementation Report: Directional edge + flat margin + confidence

## 1. 변경 파일 목록

| 파일 | 변경 내용 |
|------|------------|
| `src/strategies/ml_signal_policy.py` | `decide_action_directional_edge_margin_conf()` 추가 (D13 룰: edge + flat_margin + confidence) |
| `src/backtest/ml_backtest_engine_impl.py` | `decision_mode`에 `directional_edge_margin_conf` 추가, `min_confidence` 인자 추가, D13 분기 및 avg_long_edge_entry/avg_short_edge_entry/avg_pflat_rejected 집계 |
| `scripts/run_tcn_label_sweep_v2.py` | `run_backtest_7d`에 `min_confidence` 인자 추가 및 엔진 전달 |
| `scripts/run_d12_directional_strategy_sweep.py` | `--decision-mode`에 `directional_edge_margin_conf` 추가, `--min-confidence` 추가, run_one 결과에 min_confidence/avg_* 필드 포함, 요약 테이블 컬럼 확장 |
| `scripts/run_phase_d13_directional_margin_conf.py` | **신규** D13 전용 스윕 스크립트 (baseline 옵션, edge/flat/conf sweep) |
| `data/reports/phase_d13_implementation_report.md` | 본 문서 |

## 2. Decision modes (최종)

| 모드 | 설명 | legacy entry gates |
|------|------|---------------------|
| **argmax** | 기존: signal = argmax(p), flat_max_th/margin_th 등 gating | 적용 |
| **directional_edge** | D12: long_edge/short_edge + (optional) p_side > p_flat, margin | 옵션 |
| **directional_edge_gated** | D12: 동일 시그널 + min_max_proba/max_entropy 적용 | 적용 |
| **directional_edge_margin_conf** | D13: (p_side - p_opposite) ≥ edge, (p_side - p_flat) ≥ flat_margin, p_side ≥ confidence | **기본 OFF** (순수 룰 효과 측정) |

## 3. D13 decision rule 상세

**LONG**
- `(p_long - p_short) ≥ min_directional_edge`
- `(p_long - p_flat) ≥ min_flat_margin`
- `p_long ≥ min_confidence`

**SHORT**
- `(p_short - p_long) ≥ min_directional_edge`
- `(p_short - p_flat) ≥ min_flat_margin`
- `p_short ≥ min_confidence`

둘 다 만족하지 않으면 signal = FLAT.

## 4. CLI 인자 (D12 스크립트 확장)

- `--decision-mode`: `argmax` | `directional_edge` | `directional_edge_gated` | `directional_edge_margin_conf`
- `--min-directional-edge`: float (D13 포함)
- `--min-side-flat-margin`: float (D13에서 min_flat_margin으로 사용)
- `--min-confidence`: float (D13 전용)
- `--disable-legacy-entry-gates`: directional 모드에서 min_max_proba/max_entropy 미적용

## 5. D13 전용 스크립트 실행

```bash
# 가상환경
.venv/bin/python

# 도움말
python -m scripts.run_phase_d13_directional_margin_conf --help

# baseline + confidence sweep (720d만, 4 run)
python -m scripts.run_phase_d13_directional_margin_conf --run-baseline --days 720 --sweep-mode conf

# edge sweep (edge=0.04,0.08,0.12,0.16, flat=0.03, conf=0.40) × 180/365/720d
python -m scripts.run_phase_d13_directional_margin_conf --sweep-mode edge

# 전체 sweep (edge + flat + conf) × 180/365/720d
python -m scripts.run_phase_d13_directional_margin_conf --run-baseline --sweep-mode all
```

## 6. 생성 결과 파일

- `data/backtests/phase_d13_results.json`: results, baseline_720, created_at
- `data/reports/phase_d13_summary.md`: baseline 요약 + 전체 runs 표 (run_id, days, decision_mode, min_directional_edge, min_flat_margin, min_confidence, cost_on, MDD, trades, entries_attempted, entries_executed, signal_long/short/flat, directional_rule_reject, avg_long_edge_entry, avg_short_edge_entry, avg_pflat_rejected) + 해석 문장

## 7. 성공 판단 기준 (D13 목표)

- 720d cost_on 개선 (baseline 대비)
- trades 감소하되 과도한 붕괴 없음
- entries_attempted 감소
- entries_executed 300~600 구간 이상적
- mean pnl per trade 상승, MDD 감소

## 8. 핵심 비교

- **Baseline (argmax)**: cost_on ≈ -0.112, MDD ≈ 0.152, trades ≈ 1491, entries_attempted ≈ 81905, entries_executed ≈ 745
- **D13 후보**: 위 메트릭 delta로 비교하여 “argmax와 directional_edge 사이의 중간 전략” 후보 선정
