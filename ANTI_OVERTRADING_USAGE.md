# Anti-Overtrading 기능 사용 가이드

## 개요

ML LSTM-Attn 백테스트에서 플립/과다매매를 억제하기 위한 기능입니다.

## 주요 기능

1. **히스테리시스 (Enter/Exit Threshold 분리)**
   - 진입 임계값과 청산 임계값을 분리하여 불필요한 진입/청산을 방지
   - 예: `enter_long_th=0.65`, `exit_long_th=0.52` → 진입은 어렵게, 청산은 쉽게

2. **최소 보유 기간 (Min Hold Bars)**
   - 포지션 진입 후 일정 기간 동안은 반대 신호/홀드 신호가 와도 청산 금지
   - 강제 청산 조건(TP/SL/max_holding)은 예외

3. **재진입 쿨다운 (Cooldown Bars)**
   - 포지션 청산 후 일정 기간 동안은 신규 진입 금지

4. **Signal Confirmation 강화**
   - `signal-confirmation-bars`를 전환(FLIP)에도 적용
   - 롱↔숏 전환 시에도 연속 N바 확인 필요

5. **No-Trade Zone 확대**
   - `flat_max_th`: `max(proba_long, proba_short) < flat_max_th` 이면 FLAT 유지
   - `margin_th`: `|proba_long - proba_short| < margin_th` 이면 FLAT 유지

## 실행 예시

### 1. 보수적 설정 (거래수 크게 감소)

```bash
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_lstm_attn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --use-optimized-threshold \
  --signal-confirmation-bars 8 \
  --min-hold-bars 12 \
  --cooldown-bars 8 \
  --enter-long-th 0.70 \
  --exit-long-th 0.55 \
  --enter-short-th 0.70 \
  --exit-short-th 0.55 \
  --margin-th 0.08
```

**설정 설명:**
- `signal-confirmation-bars=8`: 8바 연속 확인 필요 (5m 기준 40분)
- `min-hold-bars=12`: 최소 12바 보유 (5m 기준 60분)
- `cooldown-bars=8`: 청산 후 8바 대기 (5m 기준 40분)
- `enter-long-th=0.70`: LONG 진입은 proba_long >= 0.70
- `exit-long-th=0.55`: LONG 청산은 proba_long < 0.55
- `margin-th=0.08`: proba 차이가 0.08 미만이면 FLAT

**예상 효과:**
- 거래 수 대폭 감소 (기존 18,582 → 수백~수천 개 수준)
- 플립 억제
- 비용 감소

### 2. 중간 설정 (검증용)

```bash
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_lstm_attn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --use-optimized-threshold \
  --signal-confirmation-bars 5 \
  --min-hold-bars 6 \
  --cooldown-bars 4 \
  --enter-long-th 0.65 \
  --exit-long-th 0.52 \
  --enter-short-th 0.65 \
  --exit-short-th 0.52 \
  --margin-th 0.06
```

**설정 설명:**
- `signal-confirmation-bars=5`: 5바 연속 확인 (5m 기준 25분)
- `min-hold-bars=6`: 최소 6바 보유 (5m 기준 30분)
- `cooldown-bars=4`: 청산 후 4바 대기 (5m 기준 20분)
- `enter-long-th=0.65`: LONG 진입은 proba_long >= 0.65
- `exit-long-th=0.52`: LONG 청산은 proba_long < 0.52
- `margin-th=0.06`: proba 차이가 0.06 미만이면 FLAT

**예상 효과:**
- 거래 수 중간 수준 감소
- 플립 억제
- 비용 중간 수준 감소

## CLI 파라미터 설명

### 히스테리시스 파라미터
- `--enter-long-th`: LONG 진입 임계값 (기본값: `long_threshold`와 동일)
- `--exit-long-th`: LONG 청산 임계값 (기본값: `enter_long_th * 0.95`)
- `--enter-short-th`: SHORT 진입 임계값 (기본값: `short_threshold`와 동일)
- `--exit-short-th`: SHORT 청산 임계값 (기본값: `enter_short_th * 0.95`)

### 최소 보유/쿨다운 파라미터
- `--min-hold-bars`: 최소 보유 기간 (바 단위, 기본값: None)
- `--cooldown-bars`: 재진입 쿨다운 (바 단위, 기본값: None)

### No-Trade Zone 파라미터
- `--flat-max-th`: `max(proba_long, proba_short) < flat_max_th` 이면 FLAT (기본값: None)
- `--margin-th`: `|proba_long - proba_short| < margin_th` 이면 FLAT (기본값: None)

### Signal Confirmation 파라미터
- `--apply-confirmation-to-flips`: 전환(FLIP)에도 confirmation 적용 (기본값: True)
- `--no-confirmation-to-flips`: 전환(FLIP)에 confirmation 적용하지 않음

## 검증 로그

백테스트 종료 시 다음 정보가 출력됩니다:

```
[ANTI-OVERTRADING] Anti-Overtrading Statistics
================================================================================
[ANTI-OVERTRADING] Entry counts: LONG=XXX, SHORT=XXX, Total entries=XXX
[ANTI-OVERTRADING] Exit count: XXX
[ANTI-OVERTRADING] Flip count: XXX (LONG↔SHORT transitions)
[ANTI-OVERTRADING] Block reasons: {'min_hold': XXX, 'cooldown': XXX, 'confirmation': XXX, 'margin_zone': XXX}
[ANTI-OVERTRADING] Fee decay estimate: roundtrip_fee=0.001800 (0.1800%), trades=XXX, exp_decay=XXX, compound_decay=XXX
[ANTI-OVERTRADING] ⚠️  If fee_decay < 0.1, costs alone would reduce balance to <10% of initial. Overtrading is likely the main issue.
```

## 주의사항

1. **비용 모델**: 수수료/슬리피지는 현재 trade별 profit에 ratio로 반영됩니다. `total_fees`는 절대값이지만 실제 balance 업데이트에는 사용되지 않습니다.

2. **단위 일관성**: `balance=1.0` 구조와 `total_fees`(절대값)의 단위 불일치는 로그에 경고로 표시되지만, 실제 계산에는 영향이 없습니다.

3. **파라미터 튜닝**: 과도하게 보수적인 설정은 거래 기회를 놓칠 수 있습니다. 백테스트 결과를 확인하며 적절한 균형을 찾아야 합니다.

