# Phase A / Phase C 요약 (GPT 공유용)

Can_bit 레포에서 **Phase A(비용 시나리오 점검)** + **Phase C(flat-proba gate)** 를 자동 실행·비교하도록 구현한 내용 요약.

---

## 1. Phase A (비용 시나리오 점검)

- **목적**: 동일 전략으로 commission/slippage만 바꿔서 두 시나리오를 돌리고, 보수 시나리오에서 수익·MDD가 크게 나빠지지 않는지 점검. (개선이 아니라 견고성/문서화.)
- **실행**:
  - `phase_a_base`: commission=0.0009, slippage=0.0001
  - `phase_a_conservative`: commission=0.0012, slippage=0.00015
- **판정**:
  - **PASS**: 두 run 다 있고 비교표 출력됨.
  - **FLAG**: conservative에서 365d **cost_on_return**이 base 대비 0.010 이상 악화 **또는** **max_drawdown**이 0.010 이상 악화.

---

## 2. Phase C (Flat-proba gate)

- **가설**: 3-class( long / short / flat )에서 **proba_flat**이 크면 애매한 예측이므로, 그 구간에서는 진입을 막는다.
- **로직**: `proba_flat = 1 - proba_long - proba_short`. 진입 직전에 `proba_flat > max_flat_proba` 이면 해당 진입 스킵.
- **옵션**: `--entry-flat-gate on`(기본 off), `--max-flat-proba 0.45`. 검증 run_id: **phase_c_flatgate_045**.
- **결과에 추가된 통계**: `entries_blocked_by_flat_gate`, `pct_flat_gate_block`, `max_flat_proba` (백테스트 result / JSON meta에 저장).
- **판정**:
  - **SUCCESS**: 365d cost_on이 baseline(phase_a_base) 대비 **+0.003 이상** 개선.
  - **NO-IMPROVE**: 그 외.
  - **OVERFILTER**: trades가 baseline 대비 **-5% 이상** 감소 시 경고.

---

## 3. 공통 기준 파라미터 (고정)

- id=h15_t0p004, symbol=BTCUSDT, timeframe=5m  
- min_max_proba=0.58, max_entropy=1.35, min_hold=48, cooldown=24  
- regime-filter=off, position-scaling=off  
- early-exit=on, lookback=12, p_floor=0.55, bad_k=8  
- days-list=30,365  

---

## 4. 실행 방법

- **전체 (A → C)**  
  `cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate && ./scripts/run_phase_checks.sh ALL`
- **A만**: `./scripts/run_phase_checks.sh A`  
- **C만** (A 결과 이미 있으면): `./scripts/run_phase_checks.sh C`  
- torch segfault 시: `SKIP_TORCH_TEST=1 ./scripts/run_phase_checks.sh ALL` (또는 C)

---

## 5. 결과 위치

- **JSON** (run_id별): `data/diagnostics/`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_a_base.json`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_a_conservative.json`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c_flatgate_045.json`
- **실행 로그**: `data/diagnostics/phase_runs/phase_run_YYYYMMDD_HHMMSS.log`
- **비교 스크립트** (표·판정만 다시 보기):
  - Phase A: `python -m scripts.compare_phase_a_costs --base-run-id phase_a_base --other-run-id phase_a_conservative`
  - Phase C: `python -m scripts.compare_baseline_vs_flatgate --baseline-run-id phase_a_base --flatgate-run-id phase_c_flatgate_045`

---

## 6. 변경/추가된 파일 (참고)

- **백테스트**: `src/backtest/ml_backtest_engines.py`, `ml_backtest_engine_impl.py` — flat-gate 진입 스킵 + result 통계.
- **검증 CLI**: `scripts/run_tcn_candidate_validation.py` — `--entry-flat-gate`, `--max-flat-proba`, meta/row에 flat-gate 필드.
- **비교**: `scripts/compare_phase_a_costs.py`, `scripts/compare_baseline_vs_flatgate.py`
- **자동 실행**: `scripts/run_phase_checks.sh` (A / C / ALL)
- **실행 가이드**: `docs/phase_ac_runbook.md` (상세), `docs/phase_ac_summary.md` (이 요약)

위 내용을 GPT에게 붙여넣으면 Phase A/C가 뭔지, 어떻게 돌리고, 결과/판정을 어디서 보는지 설명하는 데 쓸 수 있음.

---

## 7. compare 출력 전문 (의사결정용)

### compare_phase_a_costs — PASS/FLAG 결론

```
========================================================================
Phase A: base vs conservative
========================================================================
base:        tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_a_base.json
conservative: tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_a_conservative.json

metric                       30d_base    30d_other    365d_base   365d_other
------------------------------------------------------------------------
cost_on_return                 0.0178       0.0140      -0.0526      -0.0993
cost_off_return                0.0284       0.0283       0.0943       0.0943
max_drawdown                   0.0079       0.0083       0.1168       0.1454
trades                             51           51          719          719

판정: FLAG (MDD 0.010+ 악화)
```

### compare_baseline_vs_flatgate 결과 전문

```
========================================================================
Baseline (phase_a_base) vs Flat-gate (phase_c_flatgate_045) — 30d / 365d
========================================================================
baseline:  tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_a_base.json
flatgate:  tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c_flatgate_045.json

metric                     30d_baseline   30d_flatgate  365d_baseline  365d_flatgate
----------------------------------------------------------------------------------
cost_on_return                   0.0178         0.0208        -0.0526        -0.0536
cost_off_return                  0.0284         0.0310         0.0943         0.0927
max_drawdown                     0.0079         0.0079         0.1168         0.1168
trades                               51             49            719            717

Flat-gate stats (365d)
  pct_flat_gate_block: 0.0, entries_blocked_by_flat_gate: 0, max_flat_proba: 0.45

판정: NO-IMPROVE
```
