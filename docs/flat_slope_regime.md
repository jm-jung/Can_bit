# Flat Slope Regime (횡보 차단)

## 정의

- **개념**: EMA200 기울기(EMA slope)가 거의 수평이면 ‘횡보’로 판단하여 Long 진입을 차단.
- **차단 조건**: `flat = abs(ema_slope) <= slope_threshold` 이면 진입 스킵.
- **ema_slope**: `compute_ema_slope(ema_series, lookback=regime_slope_lookback)` 재사용.  
  `ema_slope[t] = ema[t] - ema[t - lookback]` (가격 단위, 비정규화).

## CLI 예시

```bash
# baseline (OFF)
python -m scripts.run_tcn_candidate_validation \
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \
  --regime-filter off \
  --days-list 30,365

# flat_slope (1차 검증)
python -m scripts.run_tcn_candidate_validation \
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \
  --regime-filter flat_slope \
  --regime-ema-span 200 --regime-slope-lookback 48 \
  --slope-threshold 0.00001 \
  --days-list 30,365
```

## 결과 JSON/MD 필드

- **meta**: `slope_threshold`
- **results[]**: `slope_threshold`, `pct_flat_slope_block` (bar 기준 flat 비율), `entries_blocked_by_regime_flat_slope`, `blocked_ratio`
- **summary**: `overblock_warning` (blocked_ratio >= 0.9), `overblock_nogo` (blocked_ratio >= 0.95)

## 1차 판정 기준

- **365d**: cost_on_return이 baseline 대비 개선, MDD 크게 악화 없음, trades 과도 감소 없음. `overblock_nogo == True`면 실패.
- **30d**: cost_on이 baseline 대비 ±20% 이내 유지 권장.
- **과차단**: 30d trades가 baseline의 30% 미만이면 slope_threshold 스케일/완화 검토.
