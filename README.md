# 🪙 Can_bit — Bitcoin Auto Trading System  
### FastAPI + CCXT + Strategy Engine + Backtest + Realtime + Risk Management + Backoffice + Dashboard

이 프로젝트는 비트코인 자동매매 시스템으로,  
**데이터 수집 → 지표 계산 → 전략 → 백테스트 → 실시간 업데이트 → 자동매매 엔진 → 리스크 관리 → 백오피스 로그/모니터링 → 프론트 대시보드**  
까지 모두 포함된 완성형 서버 애플리케이션입니다.

---

## 🚀 주요 기능 (Features)

### ✓ 1. Binance OHLCV 데이터 수집
- CCXT 기반 1분봉 캔들 자동 수집  
- CSV 저장 (`src/data/btc_ohlcv.csv`)  
- 실시간 업데이트 엔진과 연동  
- 오류/연결 실패 대비 로깅 처리  

---

### ✓ 2. 지표 계산 (Indicators)
- EMA(20)  
- SMA(20)  
- RSI(14)  
→ `indicators/basic.py`에서 자동 계산

---

### ✓ 3. 전략 엔진 (Strategy Engine)

#### 기본 전략: **EMA + RSI Strategy**

조건:  
- 가격 > EMA20 AND RSI < 70 → **LONG**  
- 가격 < EMA20 AND RSI > 30 → **SHORT**  
- 나머지 → **HOLD**

엔드포인트:
`GET /debug/strategy/simple`

#### ML 전략: **XGBoost Strategy**

XGBoost 기반 머신러닝 전략으로 다음 5분 수익률 > 0 여부를 이진 분류합니다.

- 학습 실행: `python -m src.ml.train_xgb`
- 전략 호출: `GET /debug/strategy/xgb-ml`
- 백테스트: `GET /debug/backtest/xgb-ml`

규칙:
- proba_up >= 0.55 → **LONG**
- proba_up <= 0.45 → **SHORT**
- 0.45 < proba_up < 0.55 → **HOLD**

---

### ✓ 4. 백테스트 엔진 (Backtest Engine)

Core file: `backtest/engine.py`

기능:  
- 전체 데이터에 전략 적용  
- 매매 진입/청산 시뮬레이션  
- 총 수익률 / 승률 / MDD 계산  
- equity_curve.json 자동 생성  

엔드포인트:
`GET /debug/backtest/simple`

---

### ✓ 5. 실시간 업데이트 엔진 (Realtime Updater)

`realtime/updater.py`

- 1분마다 Binance OHLCV 최신 캔들 가져옴  
- CSV 자동 업데이트  
- 지표 / 전략 값 자동 갱신  
- FastAPI Background Task로 24시간 동작  

엔드포인트:
`GET /realtime/last`

---

### ✓ 6. 모의 자동매매 엔진 (Dummy Trader)

`trading/binance_client.py`

기능:  
- BUY/SELL/청산 지원  
- 포지션 상태 메모리 기반 관리  
- dry-run 기반 모의 주문  
- 전략 신호 기반 자동매매 테스트용  

엔드포인트:
`GET /trade/step`
`GET /trade/position`

---

### ✓ 7. 실전 주문 아키텍처 (Real Trading Structure)

`trading/binance_real_client.py` + `trading/router.py`

- SIM / REAL 모드 스위치  
- REAL 모드도 현재는 dry-run (안전)  
- 실전 주문 로직 구조만 존재 (실제 주문 X)  

엔드포인트:
`GET /trade/mode`
`POST /trade/mode/{SIM|REAL}`

---

### ✓ 8. 리스크 관리 엔진 (Risk Manager)

`trading/risk.py`

자동매매 시스템 보호 기능:

- 최대 포지션 크기 제한  
- 1회 리스크 비중 계산  
- 일일 최대 손실 제한 (Daily Max Loss %)  
- 주문 간 최소 간격(쿨다운)  
- Equity 추적  
- 마지막 거래 PnL 기록  
- 거래 중지 사유(trading_disabled_reason) 관리  

엔드포인트:
`GET /risk/status`
`POST /risk/reset-day`

---

### ✓ 9. 백오피스 모니터링 API (Backoffice)

📁 `src/backoffice/`

지원 기능:

- 거래 로그 (`trades.log`)  
- 에러 로그 (`errors.log`)  
- 리스크 로그 (`risk.log`)  
- Equity Curve JSON 관리  
- 일일 리포트  
- 전체 시스템 상태 모니터링  

엔드포인트:
`GET /backoffice/logs/trades`
`GET /backoffice/logs/errors`
`GET /backoffice/logs/risk`
`GET /backoffice/equity-curve`
`GET /backoffice/daily-report`
`GET /backoffice/monitor`

---

### ✓ 10. Next.js 프론트 대시보드 (Dashboard)

📁 `/frontend`

구성:

- Next.js 14 (App Router)
- TailwindCSS  
- React Query  
- Axios API Layer  
- Recharts 그래프  

페이지:

- Dashboard (실시간 시그널 / 가격 / 모드 전환)
- Trades 로그 테이블
- Risk 상태 모니터
- Backoffice 로그 조회

---

## 📂 프로젝트 구조 (Project Structure)

src/
├── main.py # FastAPI entry
├── core/
│ └── config.py
├── services/
│ └── ohlcv_service.py # CSV Loader
├── indicators/
│ └── basic.py # EMA, RSI, SMA
├── strategies/
│ ├── basic.py # EMA+RSI strategy
│ └── ml_xgb.py # XGBoost ML strategy
├── ml/
│ ├── features.py # Feature engineering
│ ├── train_xgb.py # Model training script
│ └── xgb_model.py # Model loader
├── realtime/
│ └── updater.py # Live OHLCV updater
├── backtest/
│ └── engine.py # Backtester
├── trading/
│ ├── engine.py # Auto trading engine
│ ├── binance_client.py # Dummy trader (mock)
│ ├── binance_real_client.py # Real trading structure (dry-run)
│ ├── router.py # SIM / REAL mode router
│ └── risk.py # Risk management engine
└── backoffice/
├── logs.py
├── utils.py
├── equity_manager.py
└── router.py


---

## 🔌 FastAPI 주요 엔드포인트 목록

### 데이터 & 전략
`GET /realtime/last`
`GET /debug/strategy/simple`
`GET /debug/strategy/xgb-ml`

### 백테스트
`GET /debug/backtest/simple`
`GET /debug/backtest/xgb-ml`

### 자동매매 엔진
`GET /trade/step`
`GET /trade/position`

### 모드 관리 (SIM / REAL)
`GET /trade/mode`
`POST /trade/mode/{SIM|REAL}`

### 리스크 관리
`GET /risk/status`
`POST /risk/reset-day`

### 백오피스
`GET /backoffice/logs/*`
`GET /backoffice/equity-curve`
`GET /backoffice/daily-report`
`GET /backoffice/monitor`

---

## 🔧 설치 & 실행

### 백엔드
`pip install -r requirements.txt`
`uvicorn src.main:app --reload`


Swagger 문서:
`http://127.0.0.1:8000/docs`


---

### 프론트엔드
`cd frontend`
`cp .env.example .env.local`
`npm install`
`npm run dev`

브라우저 접속:
`http://localhost:3000`

---

## 📈 향후 확장 계획

- ML/DL 기반 고급 전략 추가  
- 이벤트 기반 Feature Engineering  
- 슬리피지·수수료 모델링  
- 실전 주문 API 구현 (REAL 모드 활성화)  
- 자동 리포트 생성  
- 대시보드 실시간 WebSocket 적용  
- Redis 기반 실시간 캐싱  
- Docker/Kubernetes 배포  

---

## 📜 라이선스
MIT License
