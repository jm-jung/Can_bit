# Phase C2: Flat-gate sweep 실행 가이드

## 목적

- **가설**: flat-gate로 애매한 구간만 걸러서 진입 품질을 높인다. max_flat_proba만 스윕해 365d cost_on을 baseline 대비 +0.003 이상 개선하는 값을 찾는다.
- **Baseline**: early_exit ON, regime OFF, **entry-flat-gate OFF** (phase_c2_base).
- **스윕 값**: max_flat_proba = 0.45, 0.40, 0.35, 0.30.

## 고정 파라미터 (변경 금지)

- id=h15_t0p004, symbol=BTCUSDT, timeframe=5m  
- min_max_proba=0.58, max_entropy=1.35, min_hold=48, cooldown=24  
- regime-filter=off, position-scaling=off  
- early-exit=on, early-exit-lookback=12, early-exit-p-floor=0.55, early-exit-bad-k=8  
- days-list=30,365  
- commission/slippage=기본값  

## 한 줄 실행

```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate && ./scripts/run_phase_checks.sh C2
```

- torch segfault(exit 139) 시: 스크립트가 자동으로 `SKIP_TORCH_TEST=1`로 재실행. 수동으로 하려면:
  `SKIP_TORCH_TEST=1 ./scripts/run_phase_checks.sh C2`

## 실행 순서

1. phase_c2_base (entry-flat-gate off)  
2. phase_c2_flatgate_045 (max_flat_proba=0.45)  
3. phase_c2_flatgate_040 (0.40)  
4. phase_c2_flatgate_035 (0.35)  
5. phase_c2_flatgate_030 (0.30)  
6. compare_baseline_vs_flatgate_sweep  

## 결과 위치

- **JSON/MD**: `data/diagnostics/`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c2_base.json` / `.md`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c2_flatgate_045.json` / `.md` (040, 035, 030 동일)
- **로그**: `data/diagnostics/phase_runs/phase_run_YYYYMMDD_HHMMSS.log`

## 판정 규칙 (compare 출력)

- **SUCCESS**: 365d cost_on >= baseline + 0.003, MDD 악화 없음(+0.005 이내), trades -5% 미만, 30d cost_on 20% 이상 악화 없음.  
- **FAIL**: MDD +0.005 초과 악화, 또는 trades -5% 초과, 또는 30d cost_on 20% 이상 악화.  
- **NO-IMPROVE**: SUCCESS 미충족, FAIL 아님.  

## 요약 출력 (마지막 3줄)

- BEST run_id / max_flat_proba / 365d cost_on / trades / MDD  
- baseline 대비 delta(cost_on, trades, MDD)  
- **최종 판정**: ADOPT(채택) 또는 KEEP_BASELINE(유지)  

## 수동 비교만 실행

```bash
python -m scripts.compare_baseline_vs_flatgate_sweep \
  --baseline-run-id phase_c2_base \
  --other-run-ids phase_c2_flatgate_045,phase_c2_flatgate_040,phase_c2_flatgate_035,phase_c2_flatgate_030
```

## Segfault(exit 139) 시

1. 스크립트가 torch 실패 시 자동으로 `SKIP_TORCH_TEST=1`로 재실행.  
2. 그래도 139면: env(OMP_NUM_THREADS=1 등) 적용·`-X faulthandler` 유지.  
3. 계속 139면: CPU wheel 재설치 안내 —  
   `pip uninstall -y torch torchvision torchaudio`  
   `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`  
