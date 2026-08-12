# Phase B1: Probability Calibration (Temperature Scaling) 실행 가이드

## 목적

- **가설**: raw proba가 과신/미보정일 수 있음. Temperature scaling으로 proba 분포를 완화하면 비용 민감 구간이 줄어 365d cost_on이 개선될 수 있음.
- **단일 변경**: 오직 temperature scaling만 적용. threshold(0.58/1.35 등), early_exit, regime off, position_scaling off는 고정.

## 무엇을 바꿨는지

- **Calibration**: `--calibration {off,temp}` (기본 off). `temp` 시 pseudo-logit → softmax(logits/T) 로 3-class(pl, ps, pf) 보정.
- **T=1.0** 이면 원본과 동일(회귀 없음).
- 적용 위치: `run_tcn_candidate_validation` 내부에서 proba 배열을 얻은 직후, 백테스트 엔진에 넘기기 전.

## 고정 파라미터 (절대 변경 금지)

- id=h15_t0p004, symbol=BTCUSDT, timeframe=5m  
- min_max_proba=0.58, max_entropy=1.35, min_hold=48, cooldown=24  
- regime-filter=off, position-scaling=off  
- early-exit=on, early-exit-lookback=12, early-exit-p-floor=0.55, early-exit-bad-k=8  
- days-list=30,365  

## 실행 방법

### 한 줄 실행 (전체)

```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate && ./scripts/run_phase_b1_calibration.sh
```

- 스크립트 내부에서 segfault 방지 env 설정 + (선택) torch import 테스트 수행.
- torch segfault 시: `SKIP_TORCH_TEST=1 ./scripts/run_phase_b1_calibration.sh`  
  그래도 run 중 exit 139가 나면:
  - (S1) 위 thread env vars 적용 + faulthandler로 재실행
  - (S2) `python -c "import torch"` 단독에서 터지면: `pip show torch`, `python -V` 확인, macOS(MPS/CPU) 확인
  - (S3) CPU wheel 재설치:  
    `pip uninstall -y torch torchvision torchaudio`  
    `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
  - (S4) 새 venv 생성 후 requirements 재설치(필요 시 사용자 판단)

### 수동 실행 (개별 run)

```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Baseline (calibration off)
python -X faulthandler -m scripts.run_tcn_candidate_validation \
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \
  --regime-filter off --position-scaling off \
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8 \
  --calibration off --days-list 30,365 --run-id phase_b1_base

# T=1.1
python -X faulthandler -m scripts.run_tcn_candidate_validation \
  ... 동일 공통 옵션 ... \
  --calibration temp --calibration-T 1.1 --days-list 30,365 --run-id phase_b1_t11

# T=1.2, T=1.4 동일 방식 (--calibration-T 1.2 / 1.4, --run-id phase_b1_t12 / phase_b1_t14)
```

### 비교만 다시 실행

```bash
python -m scripts.compare_baseline_vs_calibration --baseline-run-id phase_b1_base --other-run-ids phase_b1_t11,phase_b1_t12,phase_b1_t14
```

## 판정 기준

Baseline(phase_b1_base) 대비 각 calibration run별:

- **SUCCESS**
  - 365d cost_on_return >= baseline + 0.003  
  - 365d max_drawdown <= baseline + 0.005  
  - 365d trades >= baseline * 0.95  
  - 30d cost_on_return >= baseline_30d * 0.8 (20% 이상 악화 금지)
- **FAIL**
  - MDD가 baseline + 0.005 초과 악화  
  - trades가 baseline 대비 -5% 초과 감소  
  - 또는 실행 실패/JSON 미생성
- **NO-IMPROVE**
  - 위 SUCCESS 미충족이지만 FAIL도 아님  

비용 참고용으로 cost_off_return도 표에 표시(판정에는 필수 아님).

## 결과 위치

- **JSON/MD**: `data/diagnostics/`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_b1_base.json` / `.md`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_b1_t11.json` / `.md`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_b1_t12.json` / `.md`
  - `tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_b1_t14.json` / `.md`
- **실행 로그**: `data/diagnostics/phase_runs/phase_b1_YYYYMMDD_HHMMSS.log`
- **meta**: 각 JSON의 `meta.calibration`, `meta.calibration_T`, (선택) `calibration_pl_mean_before/after` 등

## 결과 붙이는 섹션 (표)

실행 후 `compare_baseline_vs_calibration` 출력을 그대로 이 문서나 별도 리포트에 붙여넣으면 됨.  
표 내용: 30d/365d cost_on_return, cost_off_return, max_drawdown, trades (baseline + t11, t12, t14) 및 Calibration T, 최고 365d cost_on run, 판정(SUCCESS/NO-IMPROVE/FAIL).
