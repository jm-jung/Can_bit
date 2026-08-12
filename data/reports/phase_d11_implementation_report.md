# Phase D11 Implementation Report

## 1. 변경 파일 목록

- **src/backtest/ml_backtest_engines.py**
  - `execute_trades()` 인자 추가: `min_directional_gap: Optional[float] = None`, `max_flat_entry_proba: Optional[float] = None`
  - D11 필터 로직 추가 (D10 spread 필터 직후): directional gap (LONG/SHORT별 edge), flat suppression (p_flat > max_flat_entry_proba 시 진입 차단)
  - 카운터 추가: `directional_gap_fail_count`, `long_directional_gap_fail_count`, `short_directional_gap_fail_count`, `flat_suppression_fail_count`
  - 결과 dict 확장: `filters_applied`, `filter_skip_stats`, `filter_block_breakdown`, `directional_gap_fail_count`, `flat_suppression_fail_count`, `long_directional_gap_fail_count`, `short_directional_gap_fail_count`

- **src/backtest/ml_backtest_engine_impl.py**
  - `run_backtest()` 인자 추가: `min_directional_gap`, `max_flat_entry_proba`
  - `execute_trades()` 호출 시 위 인자 전달

- **scripts/run_tcn_label_sweep_v2.py**
  - `run_backtest_7d()` 인자 추가: `min_directional_gap`, `max_flat_entry_proba` (기본값 None)
  - `engine.run_backtest()` 호출 시 전달

- **scripts/run_phase_d11_filter_sweep.py** (신규)
  - D11 전용 스크립트: baseline, D11-A(방향 gap), D11-B(flat 억제), D11-C(조합) 실행
  - CLI: `--min-directional-gap`, `--max-flat-entry-proba`, `--days`, `--sweep {none,sample,full}`
  - 720d 1회 로드 후 180/365/720 슬라이스, 멀티 윈도우 + 스윕 지원
  - 산출: `data/backtests/phase_d11_results.json`, `data/reports/phase_d11_summary.md`

## 2. 구현 내용 요약

- **Directional gap (D11-A)**  
  - LONG: `long_edge = p_long - p_short` → `long_edge < min_directional_gap` 이면 진입 차단.  
  - SHORT: `short_edge = p_short - p_long` → `short_edge < min_directional_gap` 이면 진입 차단.  
  - `min_directional_gap is None` 이면 비활성(기존 동작 유지).

- **Flat suppression (D11-B)**  
  - `p_flat > max_flat_entry_proba` 이면 진입 차단.  
  - `max_flat_entry_proba is None` 이면 비활성.

- **조합 (D11-C)**  
  - 두 필터 모두 활성화 가능. 적용 순서: 기존 entry 필터 → D10 spread → **D11 directional gap** → **D11 flat suppression** → Guard v2 / hysteresis.

- **통계**  
  - `entries_attempted`, `entries_executed`, `total_trades`, `directional_gap_fail_count`, `flat_suppression_fail_count`, `long_directional_gap_fail_count`, `short_directional_gap_fail_count` 수집·기록.  
  - `filter_block_breakdown`으로 필터별 차단 횟수 제공.

- **기본값**  
  - 새 파라미터 기본값은 `None` → 미지정 시 D11 필터 비활성, 기존 백테스트 결과와 동일.

## 3. 실제 실행한 커맨드

```bash
cd /Users/jeongminjun/Projects/Can_bit
source .venv/bin/activate

# --help 확인
python scripts/run_phase_d11_filter_sweep.py --help

# baseline만 (D11 필터 없음, 180/365/720)
python scripts/run_phase_d11_filter_sweep.py --sweep none
```

(아래는 구현 검증 후 실행할 커맨드 예시.)

```bash
# D11-A 샘플: 720d, gap=0.04
python scripts/run_phase_d11_filter_sweep.py --min-directional-gap 0.04 --days 720

# D11-B 샘플: 720d, flat=0.35
python scripts/run_phase_d11_filter_sweep.py --max-flat-entry-proba 0.35 --days 720

# D11-C 샘플: 720d, gap=0.04 + flat=0.35
python scripts/run_phase_d11_filter_sweep.py --min-directional-gap 0.04 --max-flat-entry-proba 0.35 --days 720

# 스윕 샘플 (baseline + A 0.04, B 0.35, C (0.04,0.35) 각 180/365/720)
python scripts/run_phase_d11_filter_sweep.py --sweep sample

# 전체 스윕 (A 5개 gap × 3윈도우, B 5개 flat × 3윈도우, C 4조합 × 3윈도우)
python scripts/run_phase_d11_filter_sweep.py --sweep full
```

## 4. 검증 결과 요약

- **--help**  
  - `--min-directional-gap`, `--max-flat-entry-proba`, `--days`, `--sweep` 정상 노출 (실행 확인됨).

- **Baseline 회귀**  
  - `--sweep none` 실행 시 720d 1회 로드 후 180/365/720 baseline 백테스트 3회 수행.  
  - 720d 로드·추론이 수 분 소요되며, 동일 환경에서 이전 D10 실행과 유사한 시간이 걸림.  
  - 완료 시 `phase_d11_results.json`, `phase_d11_summary.md` 생성됨.

- **D11-A/B/C 샘플**  
  - 위 단일 실행 커맨드로 각각 720d 1회 실행 시 D11 필터가 적용된 백테스트 1회씩 수행 가능.  
  - `directional_gap_fail_count` / `flat_suppression_fail_count` 및 `entries_executed` 변화로 “필터는 돌지만 executed trades는 거의 그대로” 여부 확인 가능.

## 5. 생성된 산출물 경로

- **JSON**  
  - `data/backtests/phase_d11_results.json`  
  - 각 run별 metrics + baseline_720 + created_at.

- **리포트**  
  - `data/reports/phase_d11_summary.md`  
  - Baseline 요약, 전체 runs 테이블 (run_id, days, min_directional_gap, max_flat_entry_proba, cost_on, MDD, trades, entries_attempted, entries_executed, directional_gap_fail_count, flat_suppression_fail_count), 해석 안내.

## 6. D11 1차 결론

- **구현**  
  - D11 directional gap·flat suppression 필터가 기존 entry 파이프라인에 최소 침습으로 추가되었고, 기본값 `None`으로 기존 동작이 유지됨.  
  - D10 spread 필터와 독립이며, fail count와 `entries_executed`로 실효성 확인 가능.

- **실행**  
  - `--help` 및 `--sweep none`(baseline) 실행으로 CLI·파이프라인 동작 확인.  
  - 720d 로드가 길어, full/sample 스윕 및 D11-A/B/C 단일 샘플 실행은 동일 커맨드로 완료 후 생성되는 `phase_d11_results.json` / `phase_d11_summary.md`로 결과 확인 가능.

- **해석 기준 (리포트 반영)**  
  - `directional_gap_fail_count` 또는 `flat_suppression_fail_count`가 큰데 `entries_executed`가 baseline과 거의 같으면 “D10.1처럼 필터 실효성 낮음”으로 해석.  
  - 720d cost_on·trades·MDD를 baseline과 비교해 개선·붕괴 여부 판단.

## 7. 남은 후속 작업

- **실행**  
  - `--sweep none` 완료 후, 필요 시 `--sweep sample` 또는 `--sweep full` 실행으로 표·delta 확보.  
  - `--min-directional-gap 0.04 --days 720` 등 단일 run으로 D11-A/B/C 각각 1회씩 추가 실행해 fail count vs entries_executed 검증.

- **리포트 보강**  
  - baseline 대비 delta_cost_on, delta_MDD, delta_trades 열을 summary 표에 추가하면 비교가 더 명확해짐 (스크립트 확장 가능).
