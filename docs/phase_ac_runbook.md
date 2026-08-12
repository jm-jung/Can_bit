# Phase A / Phase C 실행 가이드

## 목적

- **Phase A**: 비용 시나리오 점검 (base vs conservative commission/slippage). 판정: **PASS** / **FLAG**.
- **Phase C**: Flat-proba gate (진입 시 proba_flat > max_flat_proba 이면 skip). 판정: **SUCCESS** / **NO-IMPROVE** / **OVERFILTER** 경고.

## 한 줄 실행 (전체)

```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate && ./scripts/run_phase_checks.sh ALL
```

환경 변수(segfault 방지)는 스크립트 내부에서 설정됨. torch import 테스트 후 Phase A → Phase C 순으로 실행.  
(로컬에서 torch segfault 시 `SKIP_TORCH_TEST=1 ./scripts/run_phase_checks.sh ALL` 로 torch 검사만 건너뛸 수 있음.)

## 개별 Phase

```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

./scripts/run_phase_checks.sh A    # Phase A만 (phase_a_base → phase_a_conservative → compare_phase_a_costs)
./scripts/run_phase_checks.sh C    # Phase C만 (phase_a_base 없으면 먼저 실행 → phase_c_flatgate_045 → compare_baseline_vs_flatgate)
./scripts/run_phase_checks.sh ALL  # A 후 C
```

## 결과 확인

- **diagnostics JSON**: `data/diagnostics/` 아래
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_a_base.json`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_a_conservative.json`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c_flatgate_045.json`
- **run_id**: 각 파일의 `meta.run_id` 또는 파일명 접미사로 식별.
- **실행 로그**: `data/diagnostics/phase_runs/phase_run_YYYYMMDD_HHMMSS.log`

## 판정 규칙

### Phase A
- **PASS**: 두 run 존재 + 비교표 출력.
- **FLAG**: 보수 시나리오(phase_a_conservative)에서 365d cost_on이 base 대비 0.010 이상 악화 **또는** MDD가 0.010 이상 악화.

### Phase C
- **SUCCESS**: 365d cost_on이 baseline(phase_a_base) 대비 +0.003 이상 개선.
- **NO-IMPROVE**: 그 외.
- **OVERFILTER**: trades가 baseline 대비 -5% 이상 감소 시 경고 표시.

## 수동 비교 (스크립트만)

```bash
python -m scripts.compare_phase_a_costs --base-run-id phase_a_base --other-run-id phase_a_conservative
python -m scripts.compare_baseline_vs_flatgate --baseline-run-id phase_a_base --flatgate-run-id phase_c_flatgate_045
```
