"""
FastAPI ML Prediction Web Application
Main entry point for the application
"""
import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging import setup_logging
from src.ml.predictor import load_model
from src.api.routes_prediction import router as prediction_router
from src.api.routes_status import router as status_router
from src.schemas.ohlcv import OHLCVCandle
from src.services.ohlcv_service import get_last_candle, get_recent_candles
from src.strategies.basic import simple_ema_rsi_strategy
from src.strategies.ml_xgb import ml_xgb_strategy
from src.strategies.dl_lstm_attn import get_lstm_attn_signal
from src.backtest.engine import run_backtest, run_backtest_with_ml, run_backtest_with_dl_lstm_attn
from src.realtime.updater import update_latest_candle
from src.trading.engine import trading_step
from src.trading.router import trading_router
from src.trading.risk import risk_manager
from src.trading.binance_real_client import binance_real
from src.backoffice.router import router as backoffice_router
from src.backoffice.logs import log_error_event

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


async def candle_updater():
    """Background task that periodically fetches new candles."""
    while True:
        try:
            updated = update_latest_candle()
            if updated:
                logger.info("📈 New candle appended & indicators refreshed.")
            else:
                logger.info("⏳ No new candle yet.")
        except Exception as exc:
            logger.error(f"❌ Realtime update error: {exc}")
            log_error_event({"event": "candle_update_failed", "details": str(exc)})
        await asyncio.sleep(60)


async def auto_trader():
    """Automated trading loop that executes the strategy every minute."""
    while True:
        try:
            result = trading_step()
            logger.info("🤖 Trade step: %s", result)
        except Exception as exc:
            logger.error(f"❌ Trading engine error: {exc}")
            log_error_event({"event": "trading_step_failed", "details": str(exc)})
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app
    Handles startup and shutdown events
    """
    # Startup: Load model and store in app.state
    logger.info("Loading ML model...")
    
    try:
        model = load_model()
        app.state.model = model

        if model is None:
            logger.warning("⚠️ Model file not found. App running without model.")
        else:
            logger.info("✅ Model loaded successfully.")

    except Exception as exc:
        logger.warning(f"Failed to load model: {exc}. App will continue without model.")
        app.state.model = None
    
    candle_task = asyncio.create_task(candle_updater())
    trading_task = asyncio.create_task(auto_trader())

    try:
        yield
    finally:
        for task in (candle_task, trading_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        logger.info("Shutting down application...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="ML Prediction Web API",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(status_router, tags=["status"])
app.include_router(prediction_router, prefix="/api/v1", tags=["prediction"])
app.include_router(backoffice_router, prefix="/backoffice", tags=["backoffice"])


@app.get("/debug/ohlcv/last", response_model=OHLCVCandle)
def read_last_candle():
    """Return the latest BTC/USDT OHLCV candle."""
    return get_last_candle()


@app.get("/debug/ohlcv/recent", response_model=List[OHLCVCandle])
def read_recent_candles(limit: int = Query(100, ge=1, le=1000)):
    """Return recent BTC/USDT OHLCV candles."""
    return get_recent_candles(limit=limit)


@app.get("/debug/strategy/simple")
def read_simple_strategy():
    """
    단순 EMA + RSI 전략의 신호(LONG/SHORT/HOLD)와 지표 값을 반환
    """
    return simple_ema_rsi_strategy()


@app.get("/debug/backtest/simple")
def read_backtest_simple():
    """
    EMA+RSI 기반 전략의 전체 백테스트 리포트 반환
    """
    return run_backtest()


@app.get("/debug/strategy/xgb-ml")
def read_xgb_ml_strategy():
    """
    XGBoost ML 전략의 신호와 확률을 반환
    """
    result = ml_xgb_strategy()
    if result["proba_up"] is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="ML model not available")
    return result


@app.get("/debug/backtest/xgb-ml")
def read_backtest_xgb_ml():
    """
    XGBoost ML 기반 전략의 전체 백테스트 리포트 반환
    """
    return run_backtest_with_ml()


@app.get("/debug/strategy/dl-lstm-attn")
def read_dl_lstm_attn_strategy():
    """
    LSTM + Attention 딥러닝 전략의 신호와 확률을 반환
    """
    result = get_lstm_attn_signal()
    if result["proba_up"] is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="DL model not available")
    return result


@app.get("/debug/backtest/dl-lstm-attn")
def read_backtest_dl_lstm_attn():
    """
    LSTM + Attention 딥러닝 기반 전략의 전체 백테스트 리포트 반환
    """
    return run_backtest_with_dl_lstm_attn()


@app.get("/realtime/last")
def realtime_last_candle():
    """
    실시간 갱신 엔진이 업데이트한 최신 캔들과 전략 결과를 반환
    """
    from src.strategies.basic import simple_ema_rsi_strategy as get_strategy
    from src.services.ohlcv_service import get_last_candle as fetch_last_candle

    return {
        "latest_candle": fetch_last_candle(),
        "strategy": get_strategy(),
    }


@app.get("/trade/step")
def trade_step():
    """
    1회 트레이딩 스텝 실행:
    전략 → 진입 or 청산 판단 → 포지션 업데이트
    """
    return trading_step()


@app.get("/trade/position")
def trade_position():
    """
    현재 보유 포지션 조회
    """
    return trading_router.get_client().get_position()


@app.get("/trade/mode")
def get_trade_mode():
    return {"mode": trading_router.mode}


@app.post("/trade/mode/{mode}")
def set_trade_mode(mode: str):
    """
    trading mode 변경 API (SIM / REAL)
    REAL 모드는 dry-run 상태이며 실제 주문은 실행되지 않음.
    """
    mode_upper = mode.upper()
    if mode_upper not in ("SIM", "REAL"):
        return {"error": "mode must be SIM or REAL"}

    if mode_upper == "REAL":
        trading_router.set_real()
    else:
        trading_router.set_sim()

    return {"status": "ok", "mode": trading_router.mode}


@app.get("/risk/status")
def get_risk_status():
    """
    현재 리스크 상태 및 설정값을 조회하는 엔드포인트
    """
    return risk_manager.status()


@app.post("/risk/reset-day")
def reset_risk_day():
    """
    강제로 '오늘' 기준을 초기화 (테스트용)
    """
    risk_manager.start_equity_today = risk_manager.equity
    risk_manager.today = datetime.utcnow().date()
    risk_manager.trading_disabled_reason = None
    return {"status": "ok", "equity": risk_manager.equity}


@app.get("/trade/live-mode")
def get_live_mode():
    """
    실전 Binance 주문 live_mode 조회용 엔드포인트.
    
    True일 경우 실제 Binance API로 주문을 시도할 수 있으므로 매우 주의가 필요하다.
    이 값은 trade/mode가 REAL일 때만 의미가 있다.
    """
    return {
        "live_mode": binance_real.get_live_mode(),
        "warning": "live_mode가 True이면 실제 주문이 실행될 수 있습니다. 신중히 사용하세요.",
    }


@app.post("/trade/live-mode/{flag}")
def set_live_mode(flag: str):
    """
    실전 Binance 주문 live_mode 설정 엔드포인트.
    
    - flag = "on"  → live_mode = True  (실제 주문 시도 가능)
    - flag = "off" → live_mode = False (dry-run 모드)
    
    실제 운영 시에는 반드시 소액 및 sandbox 모드에서 충분히 검증 후 사용하는 것을 권장한다.
    
    주의사항:
    - 이 API는 매우 위험할 수 있습니다.
    - 실제 자금이 사용될 수 있으므로 신중하게 사용하세요.
    - sandbox 모드에서 먼저 테스트하는 것을 강력히 권장합니다.
    """
    flag = flag.lower()
    if flag == "on":
        binance_real.enable_live_mode()
    elif flag == "off":
        binance_real.disable_live_mode()
    else:
        return {"error": "flag must be 'on' or 'off'"}
    
    return {
        "live_mode": binance_real.get_live_mode(),
        "message": f"Live mode has been turned {flag}",
        "warning": "live_mode가 True이면 실제 주문이 실행될 수 있습니다." if binance_real.get_live_mode() else None,
    }

