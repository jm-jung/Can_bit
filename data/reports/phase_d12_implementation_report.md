# Phase D12 Implementation Report: Directional Edge Decision Rule

## 1. 변경 파일 목록

| 파일 | 변경 내용 |
|------|------------|
| `src/strategies/ml_signal_policy.py` | `decide_action_directional_edge()` 추가, ReasonCode에 DIRECTIONAL_EDGE_LONG/SHORT, DIRECTIONAL_RULE_REJECT 추가 |
| `src/backtest/ml_backtest_engine_impl.py` | `generate_signals`에 decision_mode/min_directional_edge/require_direction_gt_flat/min_side_flat_margin 추가, directional 분기 및 `_last_signal_stats` 저장; `run_backtest`에 D12 인자 추가, result에 signal stats 병합 |
| `src/backtest/ml_backtest_engines.py` | `execute_trades`에 `use_legacy_entry_gates` 추가, min_max_proba/max_entropy 적용 조건 분기 |
| `scripts/run_tcn_label_sweep_v2.py` | `run_backtest_7d`에 decision_mode, min_directional_edge, require_direction_gt_flat, min_side_flat_margin, use_legacy_entry_gates_with_directional 인자 추가 및 엔진 전달 |
| `scripts/run_d12_directional_strategy_sweep.py` | **신규** D12 전용 스윕/CLI 스크립트 |
| `data/reports/phase_d12_implementation_report.md` | 본 문서 |

## 2. 구현한 decision modes

| 모드 | 설명 | legacy entry gates (min_max_proba, max_entropy) |
|------|------|---------------------------------------------------|
| **argmax** | 기존: signal = argmax(p), 방향은 argmax, flat_max_th/margin_th 등으로 gating | 적용 |
| **directional_edge** | long_edge = p_long - p_short, short_edge = p_short - p_long; LONG if long_edge >= min_directional_edge and p_long > p_flat; SHORT 동일; else FLAT | 기본 OFF (순수 실험용), 옵션으로 ON 가능 |
| **directional_edge_gated** | directional_edge와 동일한 시그널 생성 + execute_trades에서 min_max_proba/max_entropy 적용 | 적용 |

## 3. 실제 실행 커맨드

```bash
# 가상환경
.venv/bin/python

# CLI 도움말 (D12 스크립트)
python scripts/run_d12_directional_strategy_sweep.py --help

# Baseline만 (argmax, 180/365/720d)
python scripts/run_d12_directional_strategy_sweep.py --sweep none

# 단일 실행: directional_edge, edge=0.04, 180d
python scripts/run_d12_directional_strategy_sweep.py --decision-mode directional_edge --min-directional-edge 0.04 --days 180

# 샘플 스윕 (baseline + D12-A 0.04 + D12-B 0.04 + D12-C 일부)
python scripts/run_d12_directional_strategy_sweep.py --sweep sample

# 전체 스윕 (D12-A 5개 edge, D12-B 3개 edge, D12-C 조합)
python scripts/run_d12_directional_strategy_sweep.py --sweep full
```

**참고**: 첫 실행 시 720d OHLCV 로드 및 TCN 추론으로 수 분 소요될 수 있음. 샌드박스 환경에서는 numpy 로딩 시 exit 139가 날 수 있으므로, 로컬에서 `required_permissions: all` 또는 일반 터미널 실행 권장.

## 4. 검증 결과

- **코드**: Lint 에러 없음.
- **Baseline 회귀**: `decision_mode=argmax`(기본) 시 기존과 동일한 경로(decide_action_3class + flat_max_th/margin_th)로 시그널 생성, `use_legacy_entry_gates=True`로 min_max_proba/max_entropy 적용 유지.
- **실행**: `--sweep none` 실행 시 720d 로드 후 180/365/720d baseline 3회 실행. 환경에 따라 첫 실행이 오래 걸릴 수 있음.

## 5. 생성된 결과 파일 경로

- `data/backtests/phase_d12_results.json` — runs 배열, baseline_720, created_at
- `data/reports/phase_d12_summary.md` — 표 요약 (run_id, days, decision_mode, min_directional_edge, cost_on, MDD, trades, entries_attempted, entries_executed, signal_long/short/flat, directional_rule_reject)

## 6. Baseline 대비 핵심 변화

- **시그널 생성**: argmax 모드는 변경 없음. directional 모드에서는 **방향 결정 규칙**이 argmax → edge+flat 우위 규칙으로 교체됨.
- **entries_attempted**: directional 모드에서는 시그널 자체가 줄어들 수 있어, D10/D11과 달리 attempted부터 baseline과 다를 수 있음.
- **진단 통계**: result에 `decision_mode`, `signal_long_count`, `signal_short_count`, `signal_flat_count`, `directional_rule_reject_count`, `min_directional_edge`, `require_direction_gt_flat`, `min_side_flat_margin` 등이 포함됨.

## 7. D12 1차 결론 (실행 후 채울 항목)

- directional mode가 argmax 대비 entries_attempted를 실제로 줄였는가
- entries_executed / trades가 어떻게 변했는가
- 720d cost_on / MDD 개선 여부
- pure directional_edge vs directional_edge_gated 중 어느 쪽이 더 유망한가

(위 항목은 실제 스윕 실행 후 `phase_d12_summary.md` 및 `phase_d12_results.json`을 보고 채우면 됨.)

## 8. 남은 후속 작업 (최소한)

- 180/365/720d 기준으로 D12-A/B/C 스윕 실행 후 표·해석 업데이트
- 필요 시 `avg_directional_edge_on_executed_entries`, `avg_pflat_on_rejected_candidates` 등 추가 통계 확장
- (선택) entry examples 10~20개 샘플링 로그
