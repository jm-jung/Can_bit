# Phase C3: Flat-aware Early Exit 실행 가이드

## 목적

- **가설**: flat 확률이 높은 구간에서는 모델이 애매하다는 신호이므로, early_exit를 더 공격적으로 동작시킨다 (effective_bad_k 감소).
- **구조**: 기존 early_exit 유지. `flat_exit_aware` ON 시, 현재 bar의 proba_flat > threshold 이면 effective_bad_k = max(1, bad_k - flat_exit_badk_delta).
- **목표**: 365d cost_on을 baseline 대비 +0.003 이상 개선 가능한지 검증.

## 고정 파라미터 (변경 금지)

- id=h15_t0p004, symbol=BTCUSDT, timeframe=5m  
- min_max_proba=0.58, max_entropy=1.35, min_hold=48, cooldown=24  
- regime-filter=off, position-scaling=off  
- early-exit=on, early-exit-lookback=12, early-exit-p-floor=0.55, early-exit-bad-k=8  
- days-list=30,365  

## CLI 옵션

- `--flat-exit-aware {on,off}` (default off)  
- `--flat-exit-threshold` float (default 0.30)  
- `--flat-exit-badk-delta` int (default 2)  

## 한 줄 실행

```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate && ./scripts/run_phase_checks.sh C3
```

- torch segfault 시: `SKIP_TORCH_TEST=1 ./scripts/run_phase_checks.sh C3`

## 실행 순서

1. phase_c3_base (flat-exit-aware off)  
2. phase_c3_flat_030_d2 (threshold=0.30, badk_delta=2)  
3. phase_c3_flat_035_d2 (threshold=0.35, badk_delta=2)  
4. compare_baseline_vs_flat_exit_aware  

## 결과 위치

- **JSON/MD**: `data/diagnostics/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c3_base.json` (및 flat_030_d2, flat_035_d2)
- **로그**: `data/diagnostics/phase_runs/phase_run_YYYYMMDD_HHMMSS.log`

## 판정 규칙 (compare)

- **SUCCESS**: 365d cost_on >= baseline + 0.003, MDD <= baseline + 0.005, trades >= baseline * 0.95  
- **FAIL**: MDD +0.005 초과 악화 OR trades -5% 초과 감소  
- **NO-IMPROVE**: 그 외  

## 요약 출력 (마지막 3줄)

- BEST run_id / 365d cost_on / trades / MDD  
- baseline 대비 delta(cost_on, trades, MDD)  
- 최종 판정: ADOPT 또는 KEEP_BASELINE  

## 수동 비교만 실행

```bash
python -m scripts.compare_baseline_vs_flat_exit_aware \
  --baseline-run-id phase_c3_base \
  --other-run-ids phase_c3_flat_030_d2,phase_c3_flat_035_d2
```

## Segfault(exit 139) 시

- thread env 설정 (OMP_NUM_THREADS=1 등), -X faulthandler 유지.  
- 그래도 실패 시 torch CPU wheel 재설치:  
  `pip uninstall -y torch torchvision torchaudio`  
  `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
