# 🪙 Can_bit — Bitcoin Auto Trading System  
### FastAPI + CCXT + Strategy Engine + Backtest + Realtime + XGBoost + LSTM/Attention + Risk Management + Backoffice + Dashboard

Can_bit은 **완전 자동화된 비트코인 알고리즘 트레이딩 시스템**입니다.

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

`python -m src.data.fetch_binance_ohlcv`

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
`python -m src.ml.train_xgb --feature-preset extended_safe`

### Proba Cache 생성
`python -m src.optimization.ml_proba_cache --strategy ml_xgb --symbol BTCUSDT --timeframe 1m`


### Threshold 최적화
`python -m src.optimization.optimize_ml_threshold`

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
`python -m src.dl.train.train_lstm_attn`


### Proba Cache 생성
🚨 **DL은 반드시 1m timeframe으로 생성**
`python -m src.optimization.ml_proba_cache --strategy ml_lstm_attn --symbol BTCUSDT --timeframe 1m`

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
`pip install -r requirements.txt`
`uvicorn src.main:app --reload`

Swagger:
`http://127.0.0.1:8000/docs`

## 프론트 실행
`cd frontend`
`cp .env.example .env.local`
`npm install`
`npm run dev`


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