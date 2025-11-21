# 🪙 Can_bit — Bitcoin Auto Trading System
### FastAPI + CCXT + Strategy Engine + Backtest + Realtime + ML/DL + Risk Management + Backoffice + Dashboard

이 프로젝트는 비트코인 자동매매 시스템으로,  
**데이터 수집 → 지표 계산 → ML/XGB → LSTM+Attention → 전략 → 백테스트 → 실시간 업데이트 → 자동매매 엔진 → 리스크 관리 → 백오피스 로그/모니터링 → 프론트 대시보드**  
까지 포함된 완성형 서버 애플리케이션입니다.

---

## 🚀 주요 기능 (Features)

### ✓ 1. Binance OHLCV 데이터 수집
- CCXT 기반 1분봉 캔들 자동 수집
- CSV 저장 (`src/data/btc_ohlcv.csv`)
- 실시간 업데이트 엔진과 연동
- 연결 실패 대비 재시도 및 로깅

---

### ✓ 2. 지표 계산 (Indicators)
- EMA(20)
- SMA(20)
- RSI(14)
- 기본 파생 지표 자동 생성  
→ `indicators/basic.py`에서 처리

---

### ✓ 3. 전략 엔진 (Strategy Engine)

#### 기본 전략: **EMA + RSI Strategy**
조건:
- 가격 > EMA20 AND RSI < 70 → **LONG**
- 가격 < EMA20 AND RSI > 30 → **SHORT**
- 나머지 → **HOLD**

`GET /debug/strategy/simple`

---

#### ML 전략: **XGBoost Strategy**
- 5분 뒤 수익률 > 0 여부를 예측하는 이진 분류 모델
- 학습: `python -m src.ml.train_xgb`
- 전략: `GET /debug/strategy/xgb-ml`
- 백테스트: `GET /debug/backtest/xgb-ml`

Prediction rule:
- proba_up ≥ 0.55 → LONG  
- proba_up ≤ 0.45 → SHORT  
- 그 사이 → HOLD  

---

#### 딥러닝 전략: **LSTM + Attention (Deep Learning Strategy)**
🆕 **2025.11 — 전체 파이프라인 대규모 리팩토링 & 디버깅 완료**

구성:
- 60개 시퀀스(window), 32개 특징(feature)
- LSTM 2-layer + Attention layer + FC classifier
- BCE/FocalLoss 선택 가능
- Event Feature 18종 포함
- 정규화/feature 일관성 체크 로직 추가
- 라벨 분포 및 threshold 분석 기능 추가
- collapse(상수 출력) 진단 기능 포함

학습 실행:
`python -m src.dl.train.train_lstm_attn`

모델 호출:
`GET /debug/strategy/dl-lstm-attn`

백테스트:
`GET /debug/backtest/dl-lstm-attn`

디버깅 기능 포함:
- Gradient norm 로깅
- last layer weight 변화 모니터링
- prob_up 분포 collapse 감지
- 소규모 Overfit 모드(`DEBUG_SMALL_OVERFIT=True`)

---

### ✓ 4. 백테스트 엔진 (Backtest Engine)
📁 `backtest/engine.py`

기능:
- 매매 시뮬레이션
- 진입/청산 처리
- 총 수익률 / 승률 / MDD / 연속 승/패
- equity_curve.json 자동 생성

예시:
`GET /debug/backtest/simple`

---

### ✓ 5. 실시간 업데이트 엔진 (Realtime Updater)

📁 `realtime/updater.py`

- 1분마다 최신 OHLCV 가져옴
- CSV 갱신 → 지표 갱신 → 전략 계산 자동화
- FastAPI Background Task로 24시간 동작

`GET /realtime/last`

---

### ✓ 6. 모의 자동매매 엔진 (Dummy Trader)

📁 `trading/binance_client.py`

- BUY / SELL / CLOSE
- 포지션 메모리 관리
- dry-run 모드 기반 모의 주문
- 전략 신호 기반 자동 트레이딩 테스트

엔드포인트:
`GET /trade/step`  
`GET /trade/position`

---

### ✓ 7. 실전 주문 아키텍처 (Real Trading Structure — Dry Run)

📁 `trading/binance_real_client.py`

- SIM / REAL 모드 스위치
- REAL도 현재는 안전을 위해 dry-run
- Binance 실전 주문 구조만 잡아둠

---

### ✓ 8. 리스크 관리 엔진 (Risk Manager)

📁 `trading/risk.py`

- 최대 포지션 크기
- 1회 리스크 비중
- 일일 최대손실 제한
- 주문 쿨다운
- Equity 추적 및 전일 대비 변화 로깅
- 거래 중지 사유 기록/관리

엔드포인트:
`GET /risk/status`  
`POST /risk/reset-day`

---

### ✓ 9. 백오피스 모니터링 API (Backoffice)

📁 `src/backoffice/`

기능:
- 거래 로그 (`trades.log`)
- 에러 로그 (`errors.log`)
- 리스크 로그 (`risk.log`)
- Equity curve 관리
- Daily report 생성
- 전체 시스템 헬스 체크

---

### ✓ 10. Next.js 프론트 대시보드

📁 `/frontend`

구성:
- Next.js 14
- TailwindCSS
- React Query
- Axios
- Recharts 대시보드

기능 페이지:
- 실시간 시그널
- Trades 테이블
- Risk Dashboard
- Backoffice Logs

---

## 📂 프로젝트 구조 (Project Structure)

src/  
├── main.py  
├── core/  
│   └── config.py  
├── services/  
│   └── ohlcv_service.py  
├── indicators/  
│   └── basic.py  
├── strategies/  
│   ├── basic.py  
│   ├── ml_xgb.py  
│   └── dl_lstm_attn.py  
├── ml/  
│   ├── features.py  
│   ├── train_xgb.py  
│   └── xgb_model.py  
├── dl/  
│   ├── train/  
│   │   └── train_lstm_attn.py  
│   ├── models/  
│   │   └── lstm_attn.py  
│   └── utils.py  
├── realtime/  
│   └── updater.py  
├── backtest/  
│   └── engine.py  
├── trading/  
│   ├── engine.py  
│   ├── binance_client.py  
│   ├── binance_real_client.py  
│   ├── router.py  
│   └── risk.py  
└── backoffice/  
    ├── logs.py  
    ├── utils.py  
    ├── equity_manager.py  
    └── router.py  

---

## 🔌 FastAPI 주요 엔드포인트

### ⭐ 데이터 & 전략
`GET /realtime/last`  
`GET /debug/strategy/simple`  
`GET /debug/strategy/xgb-ml`  
`GET /debug/strategy/dl-lstm-attn`

### ⭐ 백테스트
`GET /debug/backtest/simple`  
`GET /debug/backtest/xgb-ml`  
`GET /debug/backtest/dl-lstm-attn`

### ⭐ 자동매매 엔진
`GET /trade/step`  
`GET /trade/position`

### ⭐ 모드 관리
`GET /trade/mode`  
`POST /trade/mode/{SIM|REAL}`

### ⭐ 리스크
`GET /risk/status`  
`POST /risk/reset-day`

### ⭐ 백오피스
`GET /backoffice/logs/*`  
`GET /backoffice/equity-curve`  
`GET /backoffice/daily-report`  
`GET /backoffice/monitor`

---

## 🔧 설치 & 실행

### 백엔드
`pip install -r requirements.txt`
`uvicorn src.main:app --reload`


Swagger:
`http://127.0.0.1:8000/docs`

---

### 프론트엔드
`cd frontend`
`cp .env.example .env.local`
`npm install`
`npm run dev`

`http://localhost:3000`

---

## 📈 향후 확장 계획

- 딥러닝 전략 고도화 (Transformer 기반)
- 슬리피지·수수료 모델링
- 실전 Binance REAL 주문 API 활성화
- Docker + Kubernetes 배포
- WebSocket 기반 실시간 대시보드
- Redis 캐싱
- AutoML 기반 전략 탐색기

---

## 📜 라이선스
MIT License
