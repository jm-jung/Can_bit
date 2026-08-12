# Early Exit (조기 청산) 가이드

## 개요

진입 필터를 더 건드리지 않고, **나쁜 구간에서 빨리 빠지는 Exit** 옵션으로 cost_on·MDD 개선을 시도하는 기능입니다. 기본값은 **OFF**로 두어 기존 동작과 완전 동일(회귀 방지)합니다.

## 규칙 (proba 기반 1종)

- **min_hold 준수**: 진입 후 보유 bar 수 < min_hold 이면 조기 청산 **금지**.
- **min_hold 만족 이후**: 최근 `early_exit_lookback` bar 안에, model proba(롱 시그널 확률)가 `p_floor` 미만인 bar가 `early_exit_bad_k`개 이상이면 **즉시 청산**.
- proba는 캐시된 proba series 사용. 현재 **LONG 포지션만** 적용 (SHORT는 미적용).

## CLI 예시

```bash
# Baseline (Early Exit OFF)
python -X faulthandler -m scripts.run_tcn_candidate_validation \
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \
  --regime-filter off \
  --early-exit off \
  --days-list 30,365

# Early Exit ON (기본 파라미터)
python -X faulthandler -m scripts.run_tcn_candidate_validation \
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \
  --regime-filter off \
  --early-exit on \
  --early-exit-lookback 12 \
  --early-exit-p-floor 0.55 \
  --early-exit-bad-k 8 \
  --days-list 30,365

# 비교
python -m scripts.compare_baseline_vs_early_exit
```

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--early-exit` | off | on \| off |
| `--early-exit-lookback` | 12 | 최근 N bar (5m 기준 12 = 1시간) |
| `--early-exit-p-floor` | 0.55 | 이 값 미만이면 "bad" bar로 카운트 |
| `--early-exit-bad-k` | 8 | lookback 안에 bad bar가 K개 이상이면 청산 |

## 성공 기준 (1차)

- **365d**
  - cost_on_return ≥ baseline − 0.001 (악화하지 않거나 개선)
  - max_drawdown 유지 또는 감소 (0.002 이상 개선이면 strong)
  - trades 과도 감소 없음 (≤10% 감소 허용)
- **30d**
  - cost_on_return이 baseline 대비 ±20% 이내 유지

## 튜닝 포인트 (실패 시, 2회만)

- **trades 너무 줄어듦 / 성과 악화**
  - `early_exit_bad_k` 8 → 10 (덜 민감)
  - 또는 `early_exit_lookback` 12 → 18
- **early_exit가 거의 안 걸림 (early_exit_rate 매우 낮음, 개선 없음)**
  - `early_exit_p_floor` 0.55 → 0.56
  - 또는 `early_exit_bad_k` 8 → 6 (더 민감)

## 결과 필드

- `early_exit_enabled` (bool)
- `early_exit_count` (int)
- `early_exit_rate` = early_exit_count / exits (trades)
- `early_exit_lookback`, `early_exit_p_floor`, `early_exit_bad_k`

span 결과 row 및 MD/JSON 메타에도 동일 키가 포함됩니다.
