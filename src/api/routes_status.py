"""
Status and health check API routes
"""
import asyncio
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path

from src.core.config import settings
from src.trading.engine import get_strategy_type, set_strategy_type, trading_step, StrategyType
from src.trading.router import trading_router
from src.trading.engine import _meta_layer  # type: ignore

router = APIRouter()

# Prevent concurrent /trading/step executions from Swagger double-clicks, etc.
_trading_step_lock = asyncio.Lock()

# Allow strategy names for POST /strategy/select
STRATEGY_MAP = {
    "RULE": StrategyType.RULE,
    "ML": StrategyType.ML,
    "HYBRID": StrategyType.HYBRID,
    "FR2_B_WITH_META": StrategyType.FR2_B_WITH_META,
}


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str


class VersionResponse(BaseModel):
    """Version information response model"""
    version: str
    project_name: str
    environment: str


@router.get("/ops/config")
async def ops_config():
    """운영 확인용: 서버가 실제로 읽은 설정값 노출 (민감정보 제외)."""
    return {
        "AUTO_TRADING_ENABLED": getattr(settings, "AUTO_TRADING_ENABLED", None),
        "TRADING_STEP_TIMEOUT_SECONDS": getattr(settings, "TRADING_STEP_TIMEOUT_SECONDS", None),
        "trading_mode": getattr(trading_router, "mode", None),
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with status and timestamp
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/version", response_model=VersionResponse)
async def get_version():
    """
    Get API version information
    
    Returns:
        VersionResponse with version details
    """
    return VersionResponse(
        version=settings.VERSION,
        project_name=settings.PROJECT_NAME,
        environment=settings.ENVIRONMENT
    )


@router.get("/strategy/meta-state")
async def get_meta_state():
    """
    Return current B_with_meta meta-layer status for dashboard/ops.
    Paper-first: also includes SIM/REAL and live trading flag when available.
    """
    snap = _meta_layer.snapshot()
    # Rolling metrics snapshot (health check용)
    metrics_snap_path = Path("data/diagnostics/fr2/meta_metrics_snapshot.json")
    metrics_snap = {}
    try:
        if metrics_snap_path.exists():
            metrics_snap = json.loads(metrics_snap_path.read_text(encoding="utf-8"))
    except Exception:
        metrics_snap = {}

    # Flatten frequently requested fields for ops UI
    def _get(key: str, default=None):
        return metrics_snap.get(key, default)

    # Attach router mode and real-client live flag if possible
    live_enabled = None
    try:
        if trading_router.mode == "REAL":
            client = trading_router.get_client()
            live_enabled = getattr(client, "get_live_mode", lambda: None)()
    except Exception:
        live_enabled = None

    return {
        "package": snap.get("strategy_name", "B_with_meta"),
        "strategy_type": str(get_strategy_type().value),
        "trading_mode": trading_router.mode,
        "live_orders_enabled": live_enabled,
        # Required fields for "why reduced/off" understanding
        "current_state": snap.get("current_state"),
        "current_multiplier": snap.get("position_multiplier"),
        "cost_on_60d": _get("cost_on_60d"),
        "cost_on_90d": _get("cost_on_90d"),
        "alpha_fee_ratio_60d": _get("alpha_fee_ratio_60d"),
        "alpha_fee_ratio_90d": _get("alpha_fee_ratio_90d"),
        "score": _get("score"),
        "alpha_score": _get("alpha_score"),
        "trades_60d": _get("trades_60d"),
        "metrics_timestamp": _get("metrics_timestamp"),
        "metrics_stale_flag": _get("metrics_stale_flag"),
        "metrics_stale_duration_minutes": _get("stale_duration_minutes"),
        "transition_reason": snap.get("transition_reason"),
        "meta": snap,
        "metrics": metrics_snap,
    }


class StrategySelectRequest(BaseModel):
    """Request body for strategy selection."""
    strategy: str  # RULE | ML | HYBRID | FR2_B_WITH_META


class StrategySelectResponse(BaseModel):
    """Response for strategy selection."""
    status: str
    strategy_type: str
    message: str


@router.post("/strategy/select", response_model=StrategySelectResponse)
async def select_strategy(body: StrategySelectRequest):
    """
    운영 전략 타입 전환 (서버 재시작 없이).
    - FR2_B_WITH_META: B(0.6 off) + Meta Layer (paper-first, reduced-first)
    - RULE: EMA+RSI
    - ML: XGBoost ML
    - HYBRID: RULE+ML 일치 시만
    """
    key = body.strategy.strip().upper()
    if key not in STRATEGY_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"strategy must be one of: {list(STRATEGY_MAP.keys())}",
        )
    set_strategy_type(STRATEGY_MAP[key])
    return {
        "status": "ok",
        "strategy_type": key,
        "message": f"Strategy set to {key}. Use GET /strategy/meta-state to verify.",
    }


@router.post(
    "/trading/step",
    responses={200: {"description": "Trading step result (status, risk, order/trade/position 등)", "content": {"application/json": {"schema": {"type": "object"}}}}},
)
async def run_trading_step():
    """
    1회 트레이딩 스텝 실행 (paper 검증용).
    전략 시그널 → meta gate → risk → 진입/청산 판단.
    반환값으로 blocked_by_meta, position_multiplier, status 확인.
    AUTO_TRADING_ENABLED=true 이면 수동 호출 비활성화(503).
    """
    if getattr(settings, "AUTO_TRADING_ENABLED", True):
        raise HTTPException(
            status_code=503,
            detail="AUTO_TRADING_ENABLED=true. 수동 POST /trading/step 비활성화. 상태는 GET /strategy/meta-state, 로그는 state_log/trades.log로 확인.",
        )
    # Avoid concurrent executions and make hangs visible as timeout instead of infinite loading.
    if _trading_step_lock.locked():
        raise HTTPException(status_code=409, detail="trading_step already running; try again in a moment.")

    async with _trading_step_lock:
        timeout_s = float(getattr(settings, "TRADING_STEP_TIMEOUT_SECONDS", 90.0))
        try:
            # trading_step() can do heavy sync I/O/compute; run in thread.
            return await asyncio.wait_for(asyncio.to_thread(trading_step), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=f"trading_step timed out after {timeout_s:.0f}s (still computing or stuck).",
            ) from exc


@router.post("/strategy/meta-evaluate-now")
async def meta_evaluate_now():
    """메타 레이어를 즉시 평가해서 metrics_snapshot/state_log/snapshot을 갱신합니다(검증용)."""
    fn = getattr(_meta_layer, "evaluate_force", None)
    if callable(fn):
        result = await asyncio.to_thread(fn)
    else:
        result = await asyncio.to_thread(_meta_layer.evaluate)
    return result

