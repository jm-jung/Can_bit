# Guard v2 평가 실험 요약

**실행 ID:** 20260122_115918  
**생성일:** 2026-01-22 12:00:54

---

## 실험 실행 커맨드

### 실험 1: Guard v2 ON (비용 ON)

```bash
/Users/jeongminjun/Projects/Can_bit/.venv/bin/python -m src.backtest.run_ml_xgb_backtest --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long --start-date 2023-01-01 --end-date 2024-12-31 --use-optimized-threshold --signal-confirmation-bars 1 --min-hold-bars 12 --cooldown-bars 12 --no-save --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft --strategy-guard-v2-scale-floor 0.02
```

### 실험 2: Guard v2 ON (비용 OFF)

```bash
/Users/jeongminjun/Projects/Can_bit/.venv/bin/python -m src.backtest.run_ml_xgb_backtest --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long --start-date 2023-01-01 --end-date 2024-12-31 --use-optimized-threshold --signal-confirmation-bars 1 --min-hold-bars 12 --cooldown-bars 12 --no-save --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft --strategy-guard-v2-scale-floor 0.02
```

### 실험 3: Guard OFF (Baseline)

```bash
/Users/jeongminjun/Projects/Can_bit/.venv/bin/python -m src.backtest.run_ml_xgb_backtest --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long --start-date 2023-01-01 --end-date 2024-12-31 --use-optimized-threshold --signal-confirmation-bars 1 --min-hold-bars 12 --cooldown-bars 12 --no-save
```

---

## 실험 1: Guard 개입 정도 분포 확인

### 결과 요약

| 지표 | 값 |
|------|-----|
| Total Return | -1.99% |
| Max Drawdown | 2.32% |
| Total Trades | 174 |
| Avg Holding | 12.0 bars |
| Win Rate | 48.28% |

### Scale 통계

| 통계 | 값 |
|------|-----|
| Count | 294 |
| Min | 0.0350 |
| P10 | 0.0370 |
| Median | 0.0710 |
| P90 | 0.1430 |
| Max | 0.1770 |

### Scale 히스토그램 (10 bins)

| Bin Range | Count |
|-----------|-------|
| 0.035 ~ 0.049 | 110 |
| 0.049 ~ 0.063 | 18 |
| 0.063 ~ 0.078 | 29 |
| 0.078 ~ 0.092 | 18 |
| 0.092 ~ 0.106 | 19 |
| 0.106 ~ 0.120 | 25 |
| 0.120 ~ 0.134 | 17 |
| 0.134 ~ 0.149 | 37 |
| 0.149 ~ 0.163 | 11 |
| 0.163 ~ 0.177 | 10 |

### Decision 카운트

| Decision | Count |
|----------|-------|
| ALLOW | 260 |
| BLOCK | 56 |
| DEFER | 0 |

---

## 실험 2: 거래 비용 vs 신호 엣지 분리

### 비용 ON vs OFF 비교

| 지표 | 비용 ON (실험1) | 비용 OFF (실험2) | 차이 |
|------|----------------|------------------|------|
| Total Return | -1.99% | -1.99% | 0.00% |
| Max Drawdown | 2.32% | 2.32% | 0.00% |
| Total Trades | 174 | 174 | 0 |
| Avg Holding | 12.0 bars | 12.0 bars | - |
| Win Rate | 48.28% | 48.28% | 0.00% |

### 결론

- **실험 2는 비용 OFF 옵션 미지원으로 실험 1과 동일 실행됨** → 비용 영향 분석은 추후 재실험 필요

---

## 실험 3: Baseline(Guard OFF) 공정 비교

### Guard ON vs OFF 비교

| 지표 | Guard ON (실험1) | Guard OFF (실험3) | 차이 |
|------|------------------|-------------------|------|
| Total Return | -1.99% | -13.89% | 11.90% |
| Max Drawdown | 2.32% | 13.77% | -11.45% |
| Total Trades | 174 | 20 | 154 |
| Avg Holding | 12.0 bars | 24.4 bars | - |
| Win Rate | 48.28% | 10.00% | 38.28% |

### 결론

- **Guard v2가 Baseline 대비 개선** → 유지/튜닝 가치 있음
- **scale 분포가 적절함** → Guard v2가 의미 있게 조절 중

---

## 종합 결론

- **실험 2는 비용 OFF 옵션 미지원으로 실험 1과 동일 실행됨** → 비용 영향 분석은 추후 재실험 필요
- **Guard v2가 Baseline 대비 개선** → 유지/튜닝 가치 있음
- **scale 분포가 적절함** → Guard v2가 의미 있게 조절 중

---

## 다음 액션 추천

1. **추천 1:** margin/entropy 파라미터 미니 스윕 (12-run)으로 최적화
1. **추천 2:** Guard v2가 return과 drawdown 모두 개선 → 현재 설정 유지하며 파라미터 미세 조정
1. **추천 3:** Stage-2/threshold 재검증으로 신호 품질 개선

---

**참고:** 전체 실험 결과는 `data/experiments/guard_v2_eval/20260122_115918/` 디렉토리에 저장되어 있습니다.
