"""
FastAPI ML Prediction Web Application
Main entry point for the application
"""
import asyncio
import logging
import subprocess
import sys
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging import setup_logging
from src.ml.predictor import load_model
from src.api.routes_prediction import router as prediction_router
from src.api.routes_status import router as status_router
from src.schemas.ohlcv import OHLCVCandle
from src.services.ohlcv_service import get_last_candle, get_recent_candles, load_ohlcv_df
from src.strategies.basic import simple_ema_rsi_strategy
from src.strategies.ml_xgb import ml_xgb_strategy
from src.strategies.dl_lstm_attn import get_lstm_attn_signal
from src.backtest.engine import (
    run_backtest,
    run_backtest_with_ml,
    run_backtest_with_dl_lstm_attn,
    run_backtest_compare,
)
from src.optimization.optimize_ml_threshold import run_threshold_optimization_for_ml_strategy
from src.optimization.threshold_optimizer import load_threshold_result
from src.realtime.updater import update_latest_candle
from src.trading.engine import trading_step
from src.trading.router import trading_router
from src.trading.risk import risk_manager
from src.trading.binance_real_client import binance_real
from src.backoffice.router import router as backoffice_router
from src.backoffice.logs import log_error_event
from src.events.dataset import (
    build_event_feature_df,
    load_event_features,
    load_processed_events,
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


async def candle_updater():
    """Background task that periodically fetches new candles. Runs sync I/O in thread to avoid blocking /docs and API."""
    while True:
        try:
            updated = await asyncio.to_thread(update_latest_candle)
            if updated:
                logger.info("📈 New candle appended & indicators refreshed.")
            else:
                logger.info("⏳ No new candle yet.")
        except Exception as exc:
            logger.error(f"❌ Realtime update error: {exc}")
            log_error_event({"event": "candle_update_failed", "details": str(exc)})
        await asyncio.sleep(60)


async def auto_trader():
    """Automated trading loop that executes the strategy every minute. Runs sync in thread to avoid blocking /docs and API."""
    while True:
        try:
            result = await asyncio.to_thread(trading_step)
            logger.info("🤖 Trade step: %s", result)
        except Exception as exc:
            logger.error(f"❌ Trading engine error: {exc}")
            log_error_event({"event": "trading_step_failed", "details": str(exc)})
        await asyncio.sleep(60)


async def meta_updater():
    """
    Meta Layer를 trading_step과 독립적으로 주기 평가하여
    meta_state_snapshot/state_log가 '반드시' 갱신되도록 보장한다.
    (evaluate() 자체가 내부 rate-limit을 가지므로 폴링은 가볍게/자주 해도 안전)
    """
    from src.trading.engine import _meta_layer  # local import to avoid import-time side effects

    poll_s = float(getattr(settings, "META_EVAL_POLL_SECONDS", 300.0))
    while True:
        try:
            snap = await asyncio.to_thread(_meta_layer.evaluate)
            # skipped=True면 rate-limit으로 평가 생략된 상태
            logger.info(
                "🧭 Meta eval: state=%s mult=%s skipped=%s last_eval_ts=%s",
                snap.get("current_state"),
                snap.get("position_multiplier"),
                snap.get("skipped"),
                snap.get("last_eval_ts"),
            )
        except Exception as exc:
            logger.error(f"❌ Meta eval error: {exc}")
            log_error_event({"event": "meta_eval_failed", "details": str(exc)})
        await asyncio.sleep(poll_s)


async def meta_metrics_refresher():
    """
    FR2 rolling metrics 공급자(meta_metrics_source_latest.json)를 주기적으로 갱신.

    중요 순서:
    - metrics refresh -> meta evaluate
    - 둘은 같은 프로세스에서 돌아가지만 파일 교체(atomic rename)로 인해 race condition을 최소화.
    """
    refresh_enabled = bool(getattr(settings, "META_METRICS_REFRESH_ENABLED", True))
    if not refresh_enabled:
        logger.info("Meta metrics refresh disabled (META_METRICS_REFRESH_ENABLED=False).")
        return

    refresh_s = float(getattr(settings, "META_METRICS_REFRESH_SECONDS", 3600.0))
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "refresh_fr2_meta_metrics.py"
    mode = str(getattr(settings, "META_METRICS_REFRESH_MODE", "operational_latest")).strip()

    # Prevent overlap
    refresh_lock = asyncio.Lock()

    if not script_path.exists():
        logger.error("Meta metrics refresh script not found: %s", script_path)
        return

    while True:
        try:
            async with refresh_lock:
                logger.info("🔄 Refreshing FR2 meta rolling metrics: %s", script_path)
                args = []
                if mode == "legacy_last_row":
                    args.append("--use-legacy-last-row")
                elif mode == "operational_latest":
                    args.append("--latest-eval")
                elif mode == "research_step":
                    pass
                await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, str(script_path), *args],
                    check=False,
                )
        except Exception as exc:
            logger.error("Meta metrics refresh failed: %s", exc)
            try:
                log_error_event({"event": "meta_metrics_refresh_failed", "details": str(exc)})
            except Exception:
                pass
        await asyncio.sleep(refresh_s)



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
    
    # 시작 시 기본값: SIM + live 주문 비활성화 (실수 방지)
    trading_router.set_sim()
    binance_real.disable_live_mode()
    logger.info("Trading mode: SIM, live orders: disabled (default).")

    candle_task = asyncio.create_task(candle_updater())
    meta_task = asyncio.create_task(meta_updater())
    meta_metrics_task = asyncio.create_task(meta_metrics_refresher())
    tasks_to_cancel = [candle_task, meta_task, meta_metrics_task]
    if getattr(settings, "AUTO_TRADING_ENABLED", True):
        trading_task = asyncio.create_task(auto_trader())
        tasks_to_cancel.append(trading_task)
        logger.info("Auto-trading loop enabled (AUTO_TRADING_ENABLED=True).")
    else:
        logger.info("Auto-trading loop disabled (AUTO_TRADING_ENABLED=False). Use /trade/step or POST /trading/step to run manually.")

    try:
        yield
    finally:
        for task in tasks_to_cancel:
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
    # 로깅은 get_lstm_attn_signal 내부에서 처리됨
    return result


@app.get("/optimization/threshold/ml-xgb")
def optimize_ml_threshold(
    metric: str = Query("sharpe", description="Metric to optimize: 'total_return' or 'sharpe'"),
    save: bool = Query(True, description="Save result to JSON file"),
):
    """
    Run threshold optimization for ML XGBoost strategy.
    
    This endpoint performs grid search over threshold candidates and selects
    the best combination based on the specified metric.
    """
    try:
        result = run_threshold_optimization_for_ml_strategy(
            metric_name=metric,
            save_result=save,
        )
        return {
            "status": "success",
            "best_long_threshold": result.best_long_threshold,
            "best_short_threshold": result.best_short_threshold,
            "best_metric_value": result.best_metric_value,
            "metric_name": result.metric_name,
            "total_trials": len(result.trials),
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/optimization/threshold/ml-xgb/load")
def load_optimized_threshold(
    strategy: str = Query("ml_xgb", description="Strategy name"),
    symbol: str = Query("BTCUSDT", description="Trading symbol"),
    timeframe: str = Query("1m", description="Timeframe"),
):
    """
    Load optimized thresholds from JSON file.
    """
    from pathlib import Path
    threshold_path = Path("data/thresholds") / f"{strategy}_{symbol}_{timeframe}.json"
    
    if not threshold_path.exists():
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Optimized thresholds not found at {threshold_path}"
        )
    
    try:
        result = load_threshold_result(threshold_path)
        return {
            "status": "success",
            "best_long_threshold": result.best_long_threshold,
            "best_short_threshold": result.best_short_threshold,
            "best_metric_value": result.best_metric_value,
            "metric_name": result.metric_name,
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to load thresholds: {str(e)}")


@app.get("/debug/backtest/dl-lstm-attn")
def read_backtest_dl_lstm_attn(
    threshold_up: float | None = Query(
        None,
        description="Optional override for LSTM up-threshold (e.g. 0.52)",
    ),
    threshold_down: float | None = Query(
        None,
        description="Optional override for LSTM down-threshold (e.g. 0.48)",
    ),
):
    """
    LSTM + Attention 딥러닝 기반 전략의 전체 백테스트 리포트 반환
    
    NOTE: threshold_up/down can be overridden via API query params now.
    """
    logger.info(
        "[DEBUG] /debug/backtest/dl-lstm-attn called (threshold_up=%s, threshold_down=%s)",
        threshold_up,
        threshold_down,
    )
    return run_backtest_with_dl_lstm_attn(
        threshold_up=threshold_up,
        threshold_down=threshold_down,
    )


@app.get("/debug/backtest/compare")
def read_backtest_compare(
    threshold_up: float | None = Query(
        None,
        description="Optional override for LSTM up-threshold",
    ),
    threshold_down: float | None = Query(
        None,
        description="Optional override for LSTM down-threshold",
    ),
):
    """
    simple / XGB-ML / DL-LSTM-Attn 3가지 전략의 백테스트 성능을 한 번에 비교해서 반환.
    """
    logger.info(
        "[DEBUG] /debug/backtest/compare called (threshold_up=%s, threshold_down=%s)",
        threshold_up,
        threshold_down,
    )
    return run_backtest_compare(
        threshold_up=threshold_up,
        threshold_down=threshold_down,
    )


@app.get("/debug/events/latest")
def read_debug_events_latest(limit: int = Query(50, ge=1, le=500)):
    """
    최근 분류된 이벤트 목록을 반환
    """
    try:
        events = load_processed_events(limit=limit)
        return {
            "count": len(events),
            "events": [event.model_dump() for event in events]
        }
    except Exception as exc:
        logger.error("이벤트 로드 실패: %s", exc, exc_info=True)
        return {
            "count": 0,
            "events": [],
            "error": str(exc)
        }


@app.get("/debug/events/features")
def read_debug_event_features(
    limit: int = Query(200, ge=1, le=2000),
    refresh: bool = False,
):
    """
    이벤트 피처 집계결과를 반환 (옵션: refresh로 재생성)
    """
    try:
        feature_df = load_event_features()
        if feature_df.empty or refresh:
            logger.info("이벤트 피처를 재생성합니다...")
            ohlcv_df = load_ohlcv_df()
            if ohlcv_df.empty:
                return {
                    "count": 0,
                    "features": [],
                    "error": "OHLCV 데이터가 없어 이벤트 피처를 생성할 수 없습니다."
                }
            feature_df = build_event_feature_df(ohlcv_df, save=True)
        
        if feature_df.empty:
            return {
                "count": 0,
                "features": [],
                "message": "이벤트 피처가 비어있습니다."
            }
        
        payload = (
            feature_df.sort_index()
            .tail(limit)
            .reset_index()
            .rename(columns={"index": "timestamp"})
            .to_dict("records")
        )
        return {
            "count": len(payload),
            "shape": list(feature_df.shape),
            "columns": list(feature_df.columns),
            "features": payload
        }
    except Exception as exc:
        logger.error("이벤트 피처 로드/생성 실패: %s", exc, exc_info=True)
        return {
            "count": 0,
            "features": [],
            "error": str(exc)
        }


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

