# Can_bit TCN 진단·Feature Research 통합 요약 (GPT 보낼용)

**목적**: base 모델(h15_t0p004) → Signal Density / Viability / D14 / Retrain Round 1 → Feature Research Round 1(extended_safe_v1)까지의 맥락과 결과를 한 문서로 정리.

**공통 설정**: symbol=BTCUSDT, timeframe=5m, seq_len=60, horizon=15, threshold=±0.004, 720d (일부 365d).

---

# Part A. Base 모델 시그널 검증

## A1. Signal Density Test (base h15_t0p004)

**결론: NO_SIGNAL** — base feature만으로는 방향성 edge가 통계적으로 없음.

### 1. 전체 directional accuracy
- TCN: accuracy=0.4938
- Random: accuracy=0.4979

### 2. Confidence bucket accuracy
- 0.33–0.40: count=30401, direction_accuracy=0.5012 (mean_return=0.000116)
- 0.40–0.45: count=43171, direction_accuracy=0.4892 (mean_return=0.000031)
- 0.45–0.50: count=40372, direction_accuracy=0.4884 (mean_return=0.000022)
- 0.50–0.55: count=31678, direction_accuracy=0.4952 (mean_return=-0.000031)
- 0.55–0.60: count=23724, direction_accuracy=0.4869 (mean_return=0.000055)
- 0.60–0.65: count=17206, direction_accuracy=0.5105 (mean_return=-0.000069)
- 0.65+: count=21130, direction_accuracy=0.4966 (mean_return=-0.000101)

### 3. Signal density
- total_rows=207682, signal_count=41882, signal_density=0.2017
- mean_return_signal=-0.000067, mean_return_all=0.000013

### 4. vol_q2 subset signal density
- vol_q2: total_rows=69227, signal_density=0.2185, direction_accuracy=0.4899, mean_return_all=0.000068, mean_return_signal=0.000094

### 5. 모델 signal이 실제 edge인지 여부
- **최종 판정: NO_SIGNAL**

---

## A2. TCN Viability Final Diagnostic (base h15_t0p004)

**결론: TCN_STILL_WORTH_PUSHING** — permutation importance는 의미 있으나, usable signal 극소수 구간은 없고, regime subset에서만 개선.

### 1. 사용 모델
- model_id: h15_t0p004, seq_len=60, horizon=15, threshold=±0.004, symbol=BTCUSDT, timeframe=5m, days=720

### 2. Feature importance 요약 (상위)
- close_sma_ratio, rolling_mean_20, low, close_open_pct, close_ema_ratio, close_open, close, high, ema_20, rsi_14. Event feature 기여: 거의 없음.

### 3. Decile / Top bucket
- max_proba_top, entropy_bottom, long_edge_top 등 상위 극소수 구간에서 usable signal(양수 cost_adj) 없음.

### 4. Regime-filtered backtest
- full baseline: cost_on=-0.1121, MDD=0.1523, trades=1491
- trend_q2: cost_on=-0.0347; vol_q0: cost_on=-0.0262; vol_q2: cost_on=0.0032
- 특정 regime에서 baseline 대비 개선: 있음

### 5. 최종 판정
- **TCN_STILL_WORTH_PUSHING** (perm_meaningful=True, usable_signal_top=False, regime_better=True)

---

## A3. Phase D14-Final (rule/threshold 종료)

**결론: RULE_BRANCH_EXHAUSTED_RETRAIN_RECOMMENDED** — max_proba/entropy 극소수 구간에서 실전성 있는 cost-adjusted 양수 없음.

- Threshold final check: top 0.1%~5%, entropy bottom 0.1%~5%, intersection에서 mean_cost_adj_argmax 0 수렴 또는 표본 부족.
- 재학습 추천 후보: h20_t0p005, h30_t0p006, h15_t0p005.

---

## A4. Retrain Round 1 (후보 3종 vs baseline)

**결론: 주력 모델 h15_t0p004 유지.** 후보 3개(h20_t0p005, h15_t0p005, h30_t0p006)는 720d cost_on·MDD 모두 baseline보다 열세.

- h15_t0p004 720d: cost_on=-0.1120, MDD=0.1523, trades=1491
- h20_t0p005 / h15_t0p005 / h30_t0p006: cost_on -0.29 ~ -0.59, MDD 0.34~0.60
- 다음 step: feature/데이터 보강 후 재학습 검토 → Feature Research Round 1으로 이어짐.

---

# Part B. Feature Research Round 1 (extended_safe_v1)

**목표**: base만으로는 시그널 없음 → 저비용 feature 확장(ATR, volume z-score, candle structure, realized vol, multi-TF trend)으로 정보 부족 문제인지 검증.

**모델**: h15_extsafe_v1 (동일 TCN, seq_len=60, horizon=15, threshold=±0.004, preset=extended_safe_v1).

## B1. 새 feature 추가 목록
- Volatility: atr_14, true_range, range_pct, range_ma_ratio
- Volume: volume_zscore_20/50, volume_ma_ratio, volume_spike_flag
- Candle structure: body_size, body_ratio, upper/lower_wick, bullish/bearish_flag
- Realized vol: realized_vol_12/24/48, rv_ratio_short
- Multi-TF trend: 15m, 1h ema20_tf, close_ema_ratio_tf, trend_flag_tf

## B2. Feature importance (상위)
- feat_vol_volatility_20, range_pct, feat_vol_atr_norm, feat_vol_range_norm, volume, feat_volu_ma_20, feat_vol_tr, high_low, rolling_std_20, feat_trend_sma20_over_close

## B3. Alignment 변화
- direction_accuracy: **0.6089**
- spearman_long_short_vs_return: **0.2249**

## B4. Signal density 변화
- signal_density: **0.6189**
- mean_return_signal: **0.000129**

## B5. Backtest 결과 비교
- Baseline (h15_t0p004): cost_on=-0.1121, MDD=0.1523, trades=1491
- h15_extsafe_v1: cost_on=1.8176, MDD=0.5340, trades=5119

## B6. 최종 판정
**FEATURE_SET_IMPROVED**

---

*원본 산출물 경로*
- Signal Density: `data/diagnostics/signal_density/signal_density_summary.md`
- TCN Viability: `data/diagnostics/tcn_viability/tcn_viability_summary.md`
- D14 Final: `data/diagnostics/d14_final/d14_final_summary.md`
- Retrain Round 1: `data/diagnostics/retrain_round1/retrain_round1_summary.md`
- Feature Research Round 1: `data/diagnostics/feature_research_round1/feature_research_round1_summary.md`
