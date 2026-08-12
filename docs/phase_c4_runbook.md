# Phase C4: Time-stop(최대 보유시간 강제청산) 실행 가이드

## 목적

- **가설**: 너무 오래 끌고 가는 포지션을 time-stop으로 잘라내면 tail risk(드로우다운)를 줄이면서 cost_on이 개선되거나 유지된다.
- **구현**: holding_bars >= time_stop_bars 도달 시 시장가 청산(기존 exit 비용/슬리피지 동일 적용).
- **옵션**: time_stop 독립 토글, 기본 OFF. Flat-aware early exit(C3 best)는 C4-Combo에서만 ON(ts96 + flat035d2).

## 고정 파라미터 (변경 금지)

- id=h15_t0p004, symbol=BTCUSDT, timeframe=5m  
- min_max_proba=0.58, max_entropy=1.35, min_hold=48, cooldown=24  
- regime-filter=off, position-scaling=off  
- early-exit=on, early-exit-lookback=12, early-exit-p-floor=0.55, early-exit-bad-k=8  
- days-list=30,365  

## CLI 옵션

- `--time-stop {on,off}` (default off)  
- `--time-stop-bars` INT (default 96; 5m 기준 72=6h, 96=8h, 144=12h)

## 운영 권장 프리셋 (채택 확정)

- **권장**: time_stop=on, time_stop_bars=72 (Phase C4 채택. C5 micro sweep 후 최종 bars는 docs/phase_c5_adopt.md 참고.)
- 코드 기본값은 OFF 유지(회귀 방지). 파이프라인/실전에서는 위 프리셋 사용.  

## 한 줄 실행

```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate && ./scripts/run_phase_checks.sh C4
```

- torch segfault 시: `SKIP_TORCH_TEST=1 ./scripts/run_phase_checks.sh C4`

## 실행 순서 (C4)

1. phase_c4_base (time_stop off)  
2. phase_c4_ts72 (time_stop_bars=72)  
3. phase_c4_ts96 (96)  
4. phase_c4_ts144 (144)  
5. phase_c4_combo_ts96_flat035d2 (flat-exit-aware on 0.35/2 + time_stop 96)  
6. compare_baseline_vs_time_stop_sweep  

## 결과 위치

- **JSON/MD**: `data/diagnostics/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c4_*.json` (base, ts72, ts96, ts144, combo_ts96_flat035d2)
- **로그**: `data/diagnostics/phase_runs/phase_run_YYYYMMDD_HHMMSS.log`
- **요약**: `data/diagnostics/phase_c4_time_stop_sweep_summary.md` (compare 실행 후 생성)

## 지표 정의

- **time_stop_exit_count**: time_stop 조건으로 실제 청산된 횟수.
- **pct_time_stop_exits**: time_stop_exit_count / trades (해당 기간 총 체결 트레이드 수).
- **30d (short-term)**: 요약본에 30d cost_on, 30d MDD, 30d trades 한 줄 표기 — 30d에서 MDD↑ 등이 크면 short-term 리스크 참고.
- **exit_reason breakdown**: time_stop / early_exit / normal_exit 별 count 및 avg_pnl (어떤 청산이 손익에 기여했는지 확인용).

## 판정 규칙 (compare)

- **SUCCESS**: 365d cost_on >= baseline + 0.003, MDD <= baseline + 0.005, trades >= baseline * 0.95  
- **FAIL**: MDD가 baseline 대비 +0.005 초과 악화  
- **OVERFILTER**: trades가 baseline 대비 -5% 이상 감소  
- **NO-IMPROVE**: 그 외  
- **Best run**: cost_on 우선, 동점이면 MDD 낮은 쪽  

## 수동 비교만 실행

```bash
python -m scripts.compare_baseline_vs_time_stop_sweep \
  --baseline-run-id phase_c4_base \
  --other-run-ids phase_c4_ts72,phase_c4_ts96,phase_c4_ts144,phase_c4_combo_ts96_flat035d2
```

## 스팟체크 (선택)

- 동일 설정으로 **run_id만 바꿔** 1회 재실행해 보면 결과·time_stop_exit_count가 비슷한지 확인 가능 (우연성 감소).
- time_stop_exit_count가 19처럼 적당히 작을 때, 1회 더 돌려보는 것이 안정적.
- 예: `--run-id phase_c4_ts72_spot` 으로 한 번 더 실행 후, 기존 phase_c4_ts72와 cost_on / time_stop_exit_count 비교.

## Segfault(exit 139) 시

- SKIP_TORCH_TEST=1, OMP/MKL/VECLIB/NUMEXPR=1 유지.  
- torch CPU wheel 재설치: `pip uninstall -y torch torchvision torchaudio` 후 `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
