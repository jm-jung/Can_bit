# ML TCN(h15_t0p004) 전략 모델 개선 로드맵

목표: 365d cost_on을 **구조적으로** 개선할 수 있는 방향을 단계별(Phase)로 정의.  
단기 튜닝이 아닌, 한 Phase당 **하나의 가설만** 검증한다.

---

## [0] 현재 확정된 운영 파라미터 (고정)

| 항목 | 값 |
|------|-----|
| id | h15_t0p004 |
| symbol | BTCUSDT |
| timeframe | 5m |
| min_max_proba | 0.58 |
| max_entropy | 1.35 |
| min_hold | 48 |
| cooldown | 24 |
| regime_filter | off |
| position_scaling | off |
| **early_exit** | **ON** (채택) |
| early_exit_lookback | 12 |
| early_exit_p_floor | 0.55 |
| early_exit_bad_k | 8 |

---

## [1] 현재 성능 스냅샷 (참고)

- baseline(early_exit OFF) 365d cost_on = **-0.0573**
- early_exit(ON) 365d cost_on = **-0.0553** (개선)
- trades ≈ 719, MDD 0.1179 → 0.1168 (소폭 개선)

**고정 기준 run (Baseline)**  
아래 커맨드를 “Baseline(early_exit ON)” 기준으로 사용하며, 각 Phase는 이 baseline에 **변경 1개**만 추가해 실행한다.

```bash
python -X faulthandler -m scripts.run_tcn_candidate_validation \
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \
  --regime-filter off \
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8 \
  --days-list 30,365
```

---

## [2] 공통 검증 환경 (실행 전 항상 적용)

```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

---

## [3] 로드맵 공통 성공 기준

| 조건 | 기준 |
|------|------|
| 365d cost_on | baseline 대비 **+0.003 이상** 개선 → 성공 |
| max_drawdown | baseline 대비 **+0.005 이상** 악화 → 실패 |
| trades | baseline 대비 **-5% 이하** 감소 → 실패 (과도한 보수화) |
| 30d cost_on | baseline 대비 **20% 이상** 악화 → 실패 |

실패 시: 해당 Phase 변경은 **롤백**하고 다음 Phase로 넘어간다.

---

## [4] Phase별 정의

### Phase A: 데이터/라벨/슬리피지-수수료 모델 점검

| 항목 | 내용 |
|------|------|
| **가설** | 현재 수수료(0.0009)/슬리피지(0.0001) 가정이 보수적이지 않으면 365d cost_on이 과대평가된다. 보수적 시나리오(commission↑, slippage↑)로 점검하면 실거래 대비 견고성을 확인할 수 있다. |
| **구현 범위** | `scripts/run_tcn_candidate_validation.py`: CLI에 `--commission`, `--slippage` 추가 후, 기존 고정값 대신 사용. (이미 `run_backtest_7d`는 인자로 받음.) |
| **검증 커맨드** | 1) Baseline(기본 수수료): `--run-id phase_a_baseline` 으로 위 고정 기준 run 1회. 2) 보수적 수수료: 동일 옵션 + `--commission 0.0012 --slippage 0.00015 --run-id phase_a_commission_0012` 1회. |
| **성공 기준** | (1) 두 run 모두 정상 완료. (2) 보수적 run(0.0012/0.00015)에서 365d cost_on이 baseline run 대비 -0.003 이상 악화되어도 “점검 완료”로 간주(문서화). (3) 보수적 run에서 MDD +0.005 초과 악화 또는 trades -5% 이하 감소 시, 수수료/슬리피지 가정 재검토. |
| **산출물** | `data/diagnostics/` 에 `phase_a_baseline`, `phase_a_commission_0012` JSON/MD. `scripts/compare_phase_results.py` 로 Phase A 비교표 출력. |

---

### Phase B: 확률 캘리브레이션 + threshold 재최적화

| 항목 | 내용 |
|------|------|
| **가설** | raw model proba가 과신되어 있어, temperature scaling 또는 isotonic 보정 후 threshold를 재최적화하면 365d cost_on이 개선된다. |
| **구현 범위** | (1) 추론 파이프라인에서 보정 옵션 추가: `src/dl/` 또는 inference 래퍼에서 temperature scaling / isotonic 적용. (2) `run_tcn_candidate_validation` 또는 별 스크립트에서 `--calibration temperature --T 1.2` 등 1개 시나리오만 추가. (3) 기존 threshold(0.58 등) 고정 또는 소규모 그리드 탐색(동일 Phase 내 1개 변경만). |
| **검증 커맨드** | Baseline(고정 기준 run) + `--calibration temperature --calibration-T 1.2 --run-id phase_b_temp_1p2` (구현 후 실제 인자명에 맞게 수정). |
| **성공 기준** | 로드맵 공통 기준. 365d cost_on +0.003 이상 개선, MDD/trades/30d 악화 시 실패·롤백. |
| **산출물** | `phase_b_*` JSON/MD. Phase B 비교표. |

---

### Phase C: 진입 품질 강화 (엔트로피/마진/uncertainty, 레짐 아님)

| 항목 | 내용 |
|------|------|
| **가설** | 레짐 필터 대신, **엔트로피 상한·마진·클래스 불균형·uncertainty** 기반 진입 필터 1종을 재설계하면 진입 품질이 올라가 365d cost_on이 개선된다. |
| **구현 범위** | `src/backtest/ml_backtest_engines.py` 또는 필터 레이어: 예) `min_entropy_upper`(진입 시 entropy < X만 허용), 또는 flat 확률 구간 제거, 또는 uncertainty(1 - max_proba) 기반 게이트 1개. entry 조건만 추가, regime 미사용. |
| **검증 커맨드** | Baseline + `--entry-quality-filter entropy_upper --max-entropy-entry 1.25 --run-id phase_c_entropy_1p25` (실제 인자명은 구현에 맞게). |
| **성공 기준** | 로드맵 공통 기준. |
| **산출물** | `phase_c_*` JSON/MD. Phase C 비교표. |

---

### Phase D: 학습 파이프라인 개선 (time split / walk-forward / leakage / feature norm)

| 항목 | 내용 |
|------|------|
| **가설** | 학습 시 **time split 강화, walk-forward 검증, leakage 제거, feature normalization 고정**을 적용하면 OOS 성능(365d cost_on)이 구조적으로 개선된다. |
| **구현 범위** | `src/dl/train/` 등: train/val split을 엄격한 시계열 분리, walk-forward 또는 rolling window 검증, 학습/추론 시 feature 정규화 계수 고정(학습 구간 기준). 재학습 후 새 체크포인트(예: h15_t0p004_v2) 저장. |
| **검증 커맨드** | 동일 백테스트 파이프라인으로 **새 모델** 검증: `--id h15_t0p004_v2 ... --run-id phase_d_v2` (나머지 옵션은 고정 기준과 동일). |
| **성공 기준** | 로드맵 공통 기준. 새 모델이 baseline(기존 h15_t0p004) 대비 365d cost_on +0.003 이상 개선. |
| **산출물** | `phase_d_*` JSON/MD. Phase D 비교표. 새 모델 체크포인트. |

---

### Phase E: 전략 레벨 개선 (trade 관리 1종: early_exit 확장 / partial TP / time stop)

| 항목 | 내용 |
|------|------|
| **가설** | early_exit 외에 **partial take-profit**, **time stop**, **trailing stop** 중 **1개만** 추가하면 MDD 감소 또는 365d cost_on 개선이 가능하다. |
| **구현 범위** | `src/backtest/ml_backtest_engines.py`: 기존 early_exit와 동일하게 “옵션 1개”만 추가(기본 OFF). 예: `--partial-tp-ratio 0.5 --partial-tp-bars 24` 또는 `--time-stop-bars 96`. |
| **검증 커맨드** | Baseline + `--partial-tp-ratio 0.5 --partial-tp-bars 24 --run-id phase_e_partial_tp` (또는 time-stop 등 구현에 맞게). |
| **성공 기준** | 로드맵 공통 기준. |
| **산출물** | `phase_e_*` JSON/MD. Phase E 비교표. |

---

## [5] Phase 요약 표

| Phase | 가설 요약 | 변경 1개 | 성공 시 | 실패 시 |
|-------|-----------|----------|---------|---------|
| A | 수수료/슬리피지 보수 시나리오 점검 | --commission/--slippage | 점검 완료·문서화 | 가정 재검토 |
| B | 확률 캘리브레이션 + threshold 재최적화 | --calibration (temperature 등) | 365d cost_on +0.003↑ | 롤백 |
| C | 진입 품질(엔트로피/마진/uncertainty) 필터 1종 | entry-quality 필터 옵션 | 365d cost_on +0.003↑ | 롤백 |
| D | 학습 파이프라인(time split/leakage/feature norm) | 새 모델 id (재학습) | 365d cost_on +0.003↑ | 롤백 |
| E | trade 관리 1종(partial TP / time stop 등) | 전략 옵션 1개 | 365d cost_on 또는 MDD 개선 | 롤백 |

---

## [6] 다음 액션 3개

1. **로드맵 문서 생성** — 이 문서(`docs/roadmap_model_improvement.md`) 확정. Phase별 커맨드는 `scripts/run_phase_checks.sh` 에 모여 있음.
2. **Phase A 실행** — 터미널에서:
   ```bash
   cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate
   export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
   ./scripts/run_phase_checks.sh phase_a
   ```
   또는 `phase_a_base`(baseline만), `phase_a_comm`(보수적 수수료만) 실행.
3. **Phase A 결과 비교표 출력** — Phase A 2회 run 완료 후:
   ```bash
   python -m scripts.compare_phase_results --phase A
   ```
   로 diagnostics JSON을 읽어 30d/365d cost_on, MDD, trades 비교표 출력.

이후 Phase B~E는 각 Phase 구현이 완료된 뒤 `run_phase_checks.sh phase_b` 등으로 실행하고 `compare_phase_results --phase B` 로 비교표를 생성하면 된다.
