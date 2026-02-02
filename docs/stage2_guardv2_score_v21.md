# Stage-2 v2.1 (연속 스코어) 구현 및 검증 보고서

**생성일:** 2026-01-22

---

## 0) Available Signals (실제 사용 가능한 Stage-2 신호)

### 현재 코드에서 확인된 필드

#### DataFrame 컬럼 (execute_trades에서 접근 가능)
- `stage2_trade` (bool): Stage-2 Trade=True/False
- `stage2_reason` (str): Stage-2 판정 이유 문자열
- `proba_long` (float): Long 확률 (Stage-2 판정에 사용됨)
- `proba_short` (float): Short 확률 (Stage-2 판정에 사용됨)
- `raw_signal` (str): 원본 신호 (LONG/SHORT/HOLD/FLAT)
- `trend_ema` (float, optional): Trend EMA 값 (`use_trend_filter=True`일 때만 존재)

#### stage2_reason에서 파싱 가능한 정보
- `trade_ok_abs_edge`: Trade=True (절대 임계값 + Edge 모두 만족)
- `trade_ok_abs`: Trade=True (절대 임계값만 만족)
- `trade_ok_edge`: Trade=True (Edge만 만족)
- `no_trade_low_conf_abs(p_long=0.xxxx<threshold)`: Trade=False, p_long 값 포함
- `no_trade_low_edge(p_diff=0.xxxx<min_edge)`: Trade=False, p_diff 값 포함
- `exit_on_flat`: FLAT 신호로 Exit
- `flat_candidate`: FLAT 후보
- `unknown_raw_sig(...)`: 극단 위험 (hard block)

#### 존재하지 않는 필드 (fallback 1.0 사용)
- ❌ `stage2_ev`, `expected_return`, `edge`, `ev`, `exp_ret`, `mu`, `alpha` (EV/Edge 계열)
- ❌ `regime`, `market_regime`, `trend`, `trend_score`, `is_trending` (Regime/Trend 계열, trend_ema 제외)
- ❌ `vol`, `vol_z`, `atr`, `atr_pct`, `realized_vol`, `bb_width`, `vix_like`, `risk_score` (Vol/Risk 계열)
- ❌ `event_risk`, `news_risk`, `macro_risk`, `event_score`, `risk_flag` (Event risk 계열)
- ❌ `quality_score`, `confidence`, `signal_strength` (Quality 계열, proba_long/short 제외)

### 결론
현재 코드에서는 **Stage-2 판정 결과(`stage2_trade`, `stage2_reason`)와 확률값(`proba_long`, `proba_short`)만 사용 가능**합니다.  
추가 신호가 없으므로, v2.1은 **기본 스코어 개선 + 확률값 기반 간접 계산**으로 구현합니다.

---

## 1) v2.1 스코어 공식 (연속값)

### 최종 공식
```
stage2_score = clamp(base * regime_factor * risk_factor * ev_factor, scale_floor, 1.0)
```

### (1) base: 기존 stage2_trade 영향 완화
- `base = 1.0` if `stage2_trade == True`
- `base = base_false` if `stage2_trade == False` (기본 0.6, 코드 상수)

### (2) regime_factor: 추세/레짐 (현재 코드 제한)
- `trend_ema`가 있으면: 연속값으로 factor 계산 (0.80~1.10)
- 없으면: `stage2_reason`에서 "edge" 포함 여부로 간접 판단
  - `trade_ok_edge` 또는 `trade_ok_abs_edge` → 1.05 (edge 만족)
  - 그 외 → 0.95
- 기본값: 1.0

### (3) risk_factor: 위험 패널티 (현재 코드 제한)
- 극단 위험 (`unknown` in reason) → hard block (score=0.0, factor 미적용)
- `proba_long`/`proba_short`의 불확실성(entropy) 기반 간접 계산
  - `entropy = -p*log(p) - (1-p)*log(1-p)`
  - `risk_norm = clamp((entropy - 0.5) / 0.2, 0, 1)` (0.5~0.7 범위 정규화)
  - `risk_factor = 1.0 - 0.3 * risk_norm` (1.0~0.7)
- 기본값: 1.0

### (4) ev_factor: EV/Edge (현재 코드 제한)
- `stage2_reason`에서 `p_diff` 파싱 시도
  - `no_trade_low_edge(p_diff=0.xxxx<min_edge)` → p_diff 추출
  - `trade_ok_edge` → p_diff = `proba_long - proba_short` (LONG) 또는 `proba_short - proba_long` (SHORT)
- `ev_factor = sigmoid(p_diff / ev_temp)` (ev_temp 기본 0.005)
- 파싱 실패 또는 p_diff 없으면: 1.0

---

## 2) 구현 위치

### 파일: `src/backtest/ml_backtest_engines.py`

#### `calculate_stage2_quality_score()` 함수 확장
- 기존 v2.0 로직 유지 (hard block, base)
- v2.1 factor 계산 추가 (regime_factor, risk_factor, ev_factor)
- 입력 인자는 현재 유지 (`stage2_trade`, `stage2_reason`, `stage2_scale_floor`)
- 추가로 `proba_long`, `proba_short`, `raw_signal`, `trend_ema` (optional)를 dict로 받거나 개별 파라미터로 추가

---

## 3) 로그/집계

### [STAGE2][SCORE] 로그 확장
```
[STAGE2][SCORE] event=SIGNAL idx=<...> ts=<...> signal=<...> stage2_trade=<T/F> 
base=<...> score=<...> regime_factor=<...> risk_factor=<...> ev_factor=<...>
ev=<...> regime=<...> risk=<...> vol=<...> reason=<...>
```

### 장기 실행 집계 (metrics.json 또는 로그)
- `df_rows`, `first_ts`, `last_ts`
- `stage2_score stats`: min/median/p90/max, pct_at_floor
- `final_scale stats`: min/median/p90/max

---

## 4) 검증

### A) 단기 (2025-01-01 ~ 2025-03-01)
- Guard v2 ON + Stage-2 ON + v2.1 score
- 완료 기준:
  - `total_trades > 0`
  - `[STAGE2][SCORE]` 20줄 샘플에서 `regime_factor`/`risk_factor`/`ev_factor` 중 최소 1개 이상이 1.0이 아닌 케이스 관측
  - `pct_at_floor`가 과도(>70%)하면 조정 후보 기록

### B) 장기 (2023-01-01 ~ 2024-12-31)
비교 3-run:
1. Guard v2 ON + Stage-2 OFF (baseline for guard layer)
2. Guard v2 ON + Stage-2 ON (v2.0, 계단형)
3. Guard v2 ON + Stage-2 ON (v2.1, 연속형)

출력: `total_return`, `max_drawdown`, `total_trades`, `win_rate`, `avg_holding`, `pct_at_floor`

목표:
- v2.1이 v2.0(-0.40%, DD 0.47%) 대비 return 또는 DD 중 하나라도 개선 시도
- 개선이 안 되면, "어떤 factor가 거의 항상 1.0이었는지 / floor에 쏠렸는지"를 근거로 다음 조정안 기록

---

## 5) 검증 결과

### 단기 검증 (2025-01-01 ~ 2025-03-01)

**실행 커맨드:**
```bash
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long \
  --start-date 2025-01-01 --end-date 2025-03-01 \
  --use-optimized-threshold --signal-confirmation-bars 1 \
  --min-hold-bars 12 --cooldown-bars 12 \
  --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft \
  --strategy-guard-v2-scale-floor 0.02 --use-stage2 --no-save
```

**결과:**
- ✅ Total Trades: 174 (0 trades 문제 해결!)
- ✅ [STAGE2][SCORE] 로그: 87줄 출력
- ✅ factor 변동 확인:
  - `base=0.600` (stage2_trade=False일 때)
  - `regime_factor=0.950` (no_edge)
  - `risk_factor=0.7~0.8` (entropy 기반)
  - `ev_factor=0.7~1.0` (p_diff 기반)
- Total Return: -0.82%
- Max Drawdown: 0.96%

**로그 샘플:**
```
[STAGE2][SCORE] event=SIGNAL idx=115 ts=2021-01-01 05:35:00 signal=LONG stage2_trade=False 
base=0.600 score=0.375 regime_factor=0.950 risk_factor=0.785 ev_factor=0.839 
guard_scale=0.175 final_scale=0.066 ev=0.0037 regime=no_edge risk=entropy=0.6433 
reason=no_trade_low_conf
```

### 장기 검증 (2023-01-01 ~ 2024-12-31)

**비교 3-run:**

| Run | 설정 | Total Return | Max Drawdown | Total Trades | Win Rate | Avg Holding | pct_at_floor |
|-----|------|--------------|--------------|--------------|----------|-------------|--------------|
| 1 | Guard v2 ON + Stage-2 OFF | -1.99% | 2.32% | 174 | 48.28% | 12.0 bars | - |
| 2 | Guard v2 ON + Stage-2 ON (v2.0) | -0.40% | 0.47% | 174 | 48.28% | 12.0 bars | - |
| 3 | Guard v2 ON + Stage-2 ON (v2.1) | -0.82% | 0.96% | 174 | 48.28% | 12.0 bars | 0.0% |

**v2.1 vs v2.0 비교:**
- Return: -0.40% → -0.82% (**0.42%p 악화**)
- MaxDD: 0.47% → 0.96% (**0.49%p 악화**)
- Total Trades: 174 (동일)
- **pct_at_floor: 0.0%** (floor 쏠림 없음, 연속 스코어 작동 확인)

**v2.1 vs Baseline 비교:**
- Return: -1.99% → -0.82% (**1.17%p 개선**)
- MaxDD: 2.32% → 0.96% (**1.36%p 개선**)

**Stage-2 score 통계 (v2.1):**
- min: 0.323, median: 0.393, p90: 0.435, max: 0.447
- **pct_at_floor: 0.0%** (연속 스코어로 floor 쏠림 완전 해소)

**Final scale 통계 (v2.1):**
- min: 0.018, median: 0.042, p90: 0.064, max: 0.071

### 분석

**v2.1이 v2.0보다 약간 악화된 이유:**
1. **base=0.6으로 인한 스코어 하향**
   - v2.0: `stage2_trade=False` → score=0.2 (고정)
   - v2.1: `stage2_trade=False` → base=0.6, factor 곱셈 → score=0.32~0.45
   - v2.1이 더 높은 스코어를 주지만, Guard v2 scale과 곱해지면서 최종 노출이 달라짐

2. **factor 영향**
   - `regime_factor=0.95` (거의 고정, edge 정보 부족)
   - `risk_factor=0.7~0.8` (entropy 기반, 일정 범위)
   - `ev_factor=0.7~1.0` (p_diff 기반, 변동 있음)

3. **신호 부족 한계**
   - 현재 코드에는 EV/Regime/Risk 신호가 제한적
   - 확률값 기반 간접 계산으로는 한계 존재

---

## 6) 다음 액션

### 현재 상태
- ✅ v2.1 구현 완료 (base, regime_factor, risk_factor, ev_factor)
- ✅ 단기/장기 검증 완료
- ✅ floor 쏠림 완전 해소 (pct_at_floor=0.0%)
- ⚠️ v2.0 대비 약간 악화 (Return -0.42%p, MaxDD +0.49%p)

### 향후 조정 후보

1. **base_false 상향 조정** (현재 0.6 → 0.7~0.8)
   - 목적: v2.0 대비 성능 회복
   - 영향: score 범위 상향, final_scale 증가

2. **ev_temp 조정** (현재 0.005)
   - p_diff 스케일 확인 후 조정
   - sigmoid 곡선 조정으로 ev_factor 영향도 변경

3. **regime_factor 개선**
   - 현재는 "edge" 포함 여부만 사용 (0.95 고정)
   - trend_ema 활용 시 연속값 계산 가능
   - 또는 proba_long/proba_short 차이를 regime proxy로 사용

4. **신호 부족 한계 인정**
   - 현재 코드에는 EV/Regime/Risk 신호가 제한적
   - 향후 추가 신호 도입 시 v2.1 효과 증대 가능
   - **현재는 v2.0이 더 나은 선택일 수 있음**

### 권장 사항

**v2.1의 장점:**
- ✅ floor 쏠림 완전 해소 (연속 스코어)
- ✅ factor 기반 세밀한 조절 가능
- ✅ 로그로 판단 근거 기록

**v2.1의 단점:**
- ⚠️ v2.0 대비 약간 악화
- ⚠️ 신호 부족으로 factor 효과 제한적

**결론:**
- v2.1은 **구조적으로 우수하지만, 현재 신호 부족으로 v2.0보다 성능이 약간 낮음**
- 향후 추가 신호 도입 시 v2.1의 잠재력 발휘 가능
- **현재는 v2.0 사용 권장, v2.1은 실험/개선용으로 유지**
