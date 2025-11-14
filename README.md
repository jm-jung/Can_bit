# Can_bit# 🪙 Bitcoin Auto Trading System  
### FastAPI + CCXT + Strategy Engine + Backtest + Realtime + Risk Management

이 프로젝트는 비트코인 자동매매 시스템으로,  
**데이터 수집 → 지표 계산 → 전략 → 백테스트 → 실시간 업데이트 → 자동매매 엔진 → 리스크 관리**  
까지 모두 포함된 완성형 서버 애플리케이션입니다.

---

## 🚀 주요 기능 (Features)

### ✓ 1. Binance OHLCV 데이터 수집
- CCXT 기반 1분봉 캔들 자동 수집
- CSV 저장 (`src/data/btc_ohlcv.csv`)
- 오류/연결 실패 대비 로깅 처리

---

### ✓ 2. 지표 계산 (Indicators)
- EMA(20)
- SMA(20)
- RSI(14)

→ `indicators/basic.py`에서 자동 계산

---

### ✓ 3. 전략 엔진 (Strategy Engine)
기본 전략: **EMA + RSI Strategy**

조건:
- 가격 > EMA20 AND RSI < 70 → **LONG**
- 가격 < EMA20 AND RSI > 30 → **SHORT**
- 나머지 → **HOLD**

엔드포인트:
GET /debug/strategy/simple

yaml
코드 복사

---

### ✓ 4. 백테스트 엔진 (Backtest Engine)

Core file: `backtest/engine.py`

기능:
- 전체 데이터에 전략 적용
- 매매 진입/청산 시뮬레이션
- 총 수익률, 승률, MDD 계산
- equity curve 자동 생성

엔드포인트:
GET /debug/backtest/simple

yaml
코드 복사

---

### ✓ 5. 실시간 업데이트 엔진 (Realtime Updater)

`realtime/updater.py`

- 1분마다 Binance OHLCV 최신 캔들 가져옴
- CSV 자동 업데이트
- 지표 / 전략 값 자동 갱신
- FastAPI 백그라운드 Task로 24시간 동작

엔드포인트:
GET /realtime/last

yaml
코드 복사

---

### ✓ 6. 모의 자동매매 엔진 (Dummy Trader)

`trading/binance_client.py`

- BUY/SELL/청산 지원
- 포지션 상태 메모리 기반 관리
- 모의 주문 → dry-run 형태로 기록
- 테스트 시각화에 적합

엔드포인트:
GET /trade/step
GET /trade/position

yaml
코드 복사

---

### ✓ 7. 실전 주문 아키텍처 (Real Trading Structure)

`trading/binance_real_client.py` + `trading/router.py`

- SIM / REAL 모드 스위치
- REAL 모드도 현재는 dry-run (안전)
- 실전 주문 코드는 구조만 존재 (아직 실행되지 않음)
- 심플한 모드 전환 API 제공

엔드포인트:
GET /trade/mode
POST /trade/mode/{SIM|REAL}

yaml
코드 복사

---

### ✓ 8. 리스크 관리 엔진 (Risk Manager)

`trading/risk.py`

자동매매의 안전장치:

- 최대 포지션 크기 제한  
- 1회 리스크 비중 계산  
- 일일 최대 손실 제한 (daily max drawdown)  
- 주문 쿨다운(연속 주문 간 최소 시간)  
- equity 추적  
- 마지막 거래 수익률(pnl) 기록  
- 거래 중지 사유(trading_disabled_reason) 관리  

엔드포인트:
GET /risk/status
POST /risk/reset-day

yaml
코드 복사

---

## 📂 프로젝트 구조 (Project Structure)

src/
├── main.py # FastAPI entry
├── services/
│ └── ohlcv_service.py # CSV Loader
├── indicators/
│ └── basic.py # EMA, RSI, SMA
├── strategies/
│ └── basic.py # EMA+RSI strategy
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
└── data/
└── btc_ohlcv.csv # Market data

yaml
코드 복사

---

## 🔌 FastAPI 주요 엔드포인트 목록

### 데이터 & 전략
GET /realtime/last
GET /debug/strategy/simple

shell
코드 복사

### 백테스트
GET /debug/backtest/simple

shell
코드 복사

### 자동매매 엔진
GET /trade/step
GET /trade/position

shell
코드 복사

### 모드 관리 (SIM / REAL)
GET /trade/mode
POST /trade/mode/{SIM|REAL}

shell
코드 복사

### 리스크 관리
GET /risk/status
POST /risk/reset-day

yaml
코드 복사

---

## 🔧 설치 & 실행

pip install -r requirements.txt
uvicorn src.main:app --reload

코드 복사

브라우저에서 Swagger 확인:
http://127.0.0.1:8000/docs

yaml
코드 복사

---

## 📈 향후 확장 계획

- MACD, Bollinger 등 고급 전략 추가
- 실전 주문 API 구현 (REAL 모드 활성화)
- 주문 슬리피지 / 수수료 반영
- 자동 리포트 생성
- 대시보드 시각화 (Streamlit or React)
- Redis 기반 실시간 캐싱
- Kubernetes / Docker 배포

---

## 📜 라이선스
MIT License