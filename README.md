# 🪙 Can_bit — Bitcoin Auto Trading System  
### FastAPI + CCXT + Strategy Engine + Backtest + Realtime + XGBoost + LSTM/Attention + Risk Management + Backoffice + Dashboard

Can_bit은 **이벤트 피처를 기반으로 완전 자동화된 비트코인 알고리즘 트레이딩 시스템**입니다.

> **데이터 수집 → 피처 생성 → ML(XGB) / DL(LSTM+Attention) 학습 → Proba 캐시 → Threshold 최적화 → 백테스트 → 실시간 업데이트 → 자동매매 → 리스크 관리 → 대시보드**
까지 모든 파이프라인을 한 번에 제공합니다.

---

# 🚀 주요 구성 요소 (Features)

---

## 1. Binance OHLCV 데이터 수집
- CCXT 기반 Binance 1분봉 자동 수집  
- 로컬 CSV(`src/data/btc_ohlcv.csv`)에 저장  
- 실시간 업데이트 엔진과 자동 연동  
- 재시도 및 오류 로깅 내장  

```
python -m src.data.fetch_binance_ohlcv
```

---

## 2. 지표 계산 (Indicators)
위치: `src/indicators/basic.py`

- EMA20, SMA20  
- RSI(14)  
- 고저 스프레드, close-open 파생 변수  
- 이벤트 기반 피처와 자동 매핑  

---

## 3. ML 전략 — XGBoost Strategy (ML/XGB)

### ✔ 특징
- horizon=5 수익률 예측 (Up/Down Clf)
- feature preset 기반 자동 피처 생성
- proba 기반 LONG/SHORT/HOLD 라벨링
- threshold 자동 최적화 지원

### 학습
```
python -m src.ml.train_xgb --feature-preset extended_safe
```

### Proba Cache 생성
```
python -m src.optimization.ml_proba_cache --strategy ml_xgb --symbol BTCUSDT --timeframe 1m
```


### Threshold 최적화
```
python -m src.optimization.optimize_ml_threshold
```

### API

`GET /debug/strategy/xgb-ml`
`GET /debug/backtest/xgb-ml`

---

## 4. 딥러닝 전략 — LSTM + Attention (3-Class DL Strategy)
📌 **2025.12 대규모 리팩토링 완료 — 완전자동 파이프라인 구축**

### ✔ 모델 구성
- Sequence length: **60**
- Feature count: **32개**
  - 기본 OHLCV 파생 피처: 14개  
  - Event Feature: **18개**
- LSTM 2-layer  
- Attention Layer  
- Fully Connected Classifier  
- Output: **3-class (FLAT=0, LONG=1, SHORT=2)**

### ✔ 레이블링 기준
- future return > +0.001 → LONG  
- future return < –0.001 → SHORT  
- 그 사이 → FLAT  

---

## 🔥 DL 학습 파이프라인 핵심 기능

### ✔ Event Feature 파이프라인 완전 자동화  
(인덱스 정렬, NaN 안전 처리, 데이터 정합성)

### ✔ Class Weighting 자동 계산  
(FLAT/LONG/SHORT 비중으로 weight 적용)

### ✔ Collapse Detection  
- 특정 클래스 90% 이상 예측 시 경고  
- 한 클래스 미예측 시 경고  

### ✔ Debug Small Overfit 모드  
- 64 샘플 stratified subset  
- 100 epochs aggressive overfit test  
- class weight off  
- checkpoint 저장 off  

### ✔ Early Stopping (Production)  
- patience=15  
- min_delta=1e-4  
- 기본 epoch=200  

### 학습 실행
```
python -m src.dl.train.train_lstm_attn
```


### Proba Cache 생성
🚨 **DL은 반드시 1m timeframe으로 생성**
```
python -m src.optimization.ml_proba_cache --strategy ml_lstm_attn --symbol BTCUSDT --timeframe 1m
```

### API
`GET /debug/strategy/dl-lstm-attn`
`GET /debug/backtest/dl-lstm-attn`

---

## 5. Backtest Engine (전략 공통)
위치: `src/backtest/engine.py`

### 기능
- 매매 시뮬레이션
- 포지션 관리
- 총 수익률 / MDD / 승률 / 연속 손익  
- equity_curve.json 자동 저장
- ML/DL 전략 백테스트 통합

### API
`GET /debug/backtest/xgb-ml`
`GET /debug/backtest/dl-lstm-attn`

---

### 🔄 Long-Only / Short-Only 분리 실험

Long-Only / Short-Only 실험은 성능 향상을 목적으로 하지 않는다.

이 실험의 목적은:
- 신호 방향별 기여도 분해
- Stage-2 Guard(CAP)의 작동 구간 분석
- Guard 판단 근거 로그의 정합성 검증

특히 Long-Only 실험은 전략 선택을 위한 것이 아니라,
**전략 구조를 해부하기 위한 분석 도구**로 사용된다.

---

## 6. 실시간 업데이트 엔진 (Realtime Updater)
`src/realtime/updater.py`

- 1분마다 최신 OHLCV 수집  
- CSV 업데이트 → 지표 업데이트 → 전략 실행 자동화  
- FastAPI Background Task 기반  

`GET /realtime/last`

---

## 7. 자동매매 엔진 (SIM Trading)

`src/trading/binance_client.py`

- 전략 신호 기반 자동 매매  
- dry-run 모드  
- 포지션/Order history 관리  
- 리스크 매니저와 연동  

API:
`GET /trade/step`
`GET /trade/position`

---

## 8. 실거래 아키텍처 (REAL Trading Skeleton)

`src/trading/binance_real_client.py`

- SIM / REAL 모드 스위치  
- 실거래 주문 구조만 준비  
- 안정성 문제로 기본은 dry-run  

---

## 9. Risk Manager — 리스크 관리 엔진

`src/trading/risk.py`

- 주문 쿨다운  
- 1회 리스크 비중 제한  
- 일일 손실 제한  
- Equity 변화 감시  
- 거래 중지 이유 기록  

`GET /risk/status`
`POST /risk/reset-day`

---

## 9-1. StrategyGuard v2 — 신호 안정화 & 과매매 방지 계층

StrategyGuard v2는 Risk Manager 이전 단계에서 작동하는
**신호 품질 안정화 계층**이다.

Risk Manager가 손실을 제한하는 역할이라면,
StrategyGuard는 애초에 **왜 이 신호가 나왔는지 설명 가능한 상태인지**를 검증한다.

### 구조
- Stage-1 Guard
  - 신호 confidence 필터
  - trade density / cooldown 제어
  - 극단적 연속 시그널 차단

- Stage-2 Guard
  - CAP 기반 장기 구간 contribution 제어
  - 특정 조건 또는 구간에 수익이 과도하게 집중되는 현상 방지

StrategyGuard는 거래를 막기 위해 존재하지 않는다.
**차단 실패(BLOCK=0) 또한 실험 결과로 기록**되며,
이를 통해 Guard 파라미터와 전략 구조를 지속적으로 개선한다.


---

## 10. Backoffice 로그 / 모니터링

`src/backoffice/`

- trade.log  
- error.log  
- risk.log  
- equity curve 관리  
- daily report 생성  
- 전체 시스템 헬스 체크  

`GET /backoffice/monitor`
`GET /backoffice/daily-report`

---

## 🧪 Stage-2 Guard — CAP 기반 Long-Run Contribution Control

Stage-2 Guard는 단일 트레이드의 손익을 제한하지 않는다.
대신 **장기 구간(Long-Run)에서 특정 조건이 전체 성과에 과도하게 기여하는 것을 제한**한다.

### CAP의 의미
- 단일 거래 수익 상한 ❌
- 장기 구간에서의 누적 기여도 상한 ⭕
- 수익이 특정 시점·조건·신호에 몰리는 현상 방지

### 도입 배경
- 단기 백테스트에서는 좋아 보이나
  실제로는 일부 구간에만 성과가 집중되는 전략 다수 존재
- 이는 실전에서 가장 위험한 착시 패턴

### 실험 결과 요약
- CAP ↓ : trade 수 감소, 안정성 증가
- CAP ↑ : contribution 집중 위험 증가
- CAP 값은 단기 성능과 장기 안정성 사이의 트레이드오프를 형성

Stage-2 Guard의 모든 실험 결과는
`data/backtest_reports/`에 자동 저장되며,
BLOCK 발생 여부 및 판단 근거 로그를 함께 기록한다.

---
# 📊 Next.js Frontend Dashboard

`/frontend/`

### 기술 스택
- Next.js 14 (App Router)
- TailwindCSS
- React Query
- Axios
- Recharts

### 기능 페이지
- 실시간 신호 모니터  
- 전략별 백테스트 시각화  
- Risk Dashboard  
- Backoffice Log Viewer  

---

# 📂 Project Structure
```
src/
├── main.py
├── core/
│ └── config.py
├── services/
│ └── ohlcv_service.py
├── indicators/
│ └── basic.py
├── ml/
│ ├── features.py
│ ├── train_xgb.py
│ └── xgb_model.py
├── dl/
│ ├── train/
│ │ └── train_lstm_attn.py
│ ├── models/
│ │ └── lstm_attn.py
│ └── utils.py
├── strategies/
│ ├── basic.py
│ ├── ml_xgb.py
│ └── dl_lstm_attn.py
├── optimization/
│ ├── ml_proba_cache.py
│ ├── optimize_ml_threshold.py
│ └── utils.py
├── realtime/
│ └── updater.py
├── backtest/
│ └── engine.py
├── trading/
│ ├── engine.py
│ ├── binance_client.py
│ ├── binance_real_client.py
│ ├── router.py
│ └── risk.py
└── backoffice/
├── logs.py
├── utils.py
├── equity_manager.py
└── router.py
```
---

# 🔌 실행 방법

## 백엔드 실행
```
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Swagger:
`http://127.0.0.1:8000/docs`

## 프론트 실행
```
cd frontend
cp .env.example .env.local
npm install
npm run dev
```


---
---

# 🗑️ Deprecated / Abandoned Strategies  
이 프로젝트는 다양한 전략 실험을 통해 발전해왔으며,  
아래 전략들은 실제 구현과 테스트를 진행했으나 **명확한 한계 때문에 중단**되었습니다.

> ⚠️ 주의: 여기에는 “일시 보류 전략(예: 수익률 Regression)”은 포함되지 않습니다.  
> 아래 목록은 **명확하게 포기한 전략만** 기록합니다.

---

## 1. 단순 Threshold 기반 전략 (지표만 사용)
### 🔍 시도 내용
- 기본 지표(EMA, RSI 등)에 threshold만 조합해서 LONG/SHORT 결정
- 예: RSI < 30 → LONG / RSI > 70 → SHORT

### ❗ 문제점
- BTC 1분봉은 대부분 노이즈 → threshold만으로 방향성을 잡기 거의 불가능  
- 변동성이 높아 rule-based threshold가 지나치게 **민감하거나 둔감**해짐  
- 기간 바뀌면 threshold가 깨지는 **overfitting 현상** 발생  
- 백테스트 결과: 일관성 부족·승률 불안정

### ❌ 이유
- threshold 단독 전략은 시장 구조를 반영하지 못함  
→ ML/DL 기반 모델로 전환하며 완전히 폐기

---

## 2. 단독 Threshold 최적화(Sharpe 기반)  
### 🔍 시도 내용
- 전략은 단순 조건으로 두고  
- threshold만 grid search로 최적화하여 성능 개선을 시도

### ❗ 문제점
- 전략 자체가 비선형 구조를 학습하지 못하므로 threshold 조정으로는 한계  
- 특정 구간에서는 Sharpe가 높아 보이지만  
  시장 변동성 달라지면 성능 무너짐  
- threshold는 시장의 구조적 변화를 반영할 수 없음

### ❌ 이유
- threshold만 최적화해선 근본적으로 신호 품질이 개선되지 않음  
→ XGBoost → LSTM Attention으로 아키텍처 업그레이드하며 폐기  

---

## 3. XGBoost 확률을 그대로 LONG/SHORT로 사용하는 전략  
### 🔍 시도 내용
- proba_up만 기반으로  
  - ≥0.5 → LONG  
  - <0.5 → SHORT  
  같은 단순 분류 기반 전략 시도

### ❗ 문제점
- ML 확률값이 시장 상황에 따라 **스케일이 크게 흔들림**  
- threshold 없이 사용 시 LONG/SHORT가 과도하게 많이 발생하거나 반대로 거의 발생하지 않음  
- 분포가 bias되면 collapse 형태로 변함 (예: 80% FLAT에도 LONG만 나옴)

### ❌ 이유
- XGBoost는 “확률 해석”이 모델·시장에 따라 불안정  
→ **Threshold Auto-Optimization을 필수적으로 넣어야만 안정화 가능**

단순 proba 컷오프 전략은 폐기.

---

## 4. 2-Class LSTM 전략 (UP vs DOWN)
### 🔍 시도 내용
- LSTM이 Up/Down만 분류하는 구조 사용  
- FLAT 구간은 모두 UP 또는 DOWN으로 흡수

### ❗ 문제점
- BTC의 대부분은 FLAT(변동성 0~0.1%) → FLAT 없어지면 데이터 왜곡  
- train/valid에서 **class imbalance 심각**  
- collapse(한 클래스만 예측) 반복  
- 3-Class 전략 대비 극도로 낮은 성능

### ❌ 이유
- 2-Class는 시장 구조적으로 불가능  
- FLAT이 반드시 있어야 안정적 Long/Short 분리 가능  
→ 3-Class LSTM + Attention 구조로 완전 전환

---

## 5. Rule-based 매매 + Threshold 조합 전략
### 🔍 시도 내용
- 지표 기반 기본 전략에 threshold를 추가해서 조건을 강화하는 방식  
예:  
- 가격 > EMA20 AND RSI < 70 AND return > 0.0005 → LONG

### ❗ 문제점
- 조건이 복잡해질수록 overfitting 증가  
- 시장 변동성이 바뀌면 즉시 성능 붕괴  
- Long/Short 신호가 지나치게 적어짐 → 백테스트 신뢰도 하락

### ❌ 이유
- rule 기반 전략은 비선형적 가격 움직임을 포착하기 어려움  
→ ML/DL 기반 전략 등장 후 완전히 폐기

---
## 6. Guard 미적용 전략의 공통 문제
StrategyGuard를 적용하지 않은 전략들은 공통적인 문제를 보였다.

- trade density 폭증
- 단기 Sharpe 착시
- 특정 구간 수익 집중
- 실전 리스크 급증

이로 인해 Can_bit에서는
**모든 전략이 StrategyGuard를 통과해야만 실전 후보로 간주된다.**


# 📌 정리  
포기한 전략들은 아래 같은 공통 이유를 가짐:

| 폐기 이유 | 설명 |
|----------|------|
| 시장 노이즈가 너무 커서 rule-based로는 대응 불가 | BTC 1분봉은 threshold 기반 전략을 붕괴시킴 |
| collapse 현상 | 2-class 또는 threshold-only에서 자주 발생 |
| 일반화 실패 | 특정 기간에서는 돌아가지만 다른 구간에서는 깨짐 |
| 신호가 너무 적거나 너무 많음 | 실전 트레이딩 불가능 |

---

# 🚫 포함되지 않은 전략 (보류 상태)
아래 전략들은 **포기한 전략이 아니라, 나중에 다시 시도할 가능성 있음**:

- **Regression 기반 수익률 예측 모델**  
  (보류: 3-Class 안정화 후 Ensemble에 다시 포함 예정)

- **Transformer 기반 DL 모델**  
  (CPU 환경 성능 문제 → GPU 환경 구축 후 재도입 가능)  

---


# 📈 향후 확장 계획

- Transformer 기반 DL 전략 추가  
- Multi-model Ensemble (XGB + LSTM + Transformer)  
- 슬리피지·수수료 시뮬레이터  
- 실전 Binance REAL 주문 적용  
- Redis 기반 캐싱  
- Docker / Kubernetes 배포  
- WebSocket 실시간 대시보드  

---

# 📜 License  
MIT License
