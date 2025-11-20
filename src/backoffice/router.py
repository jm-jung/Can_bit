"""Backoffice FastAPI router."""
from __future__ import annotations

from fastapi import APIRouter, Query

from src.backoffice.utils import (
    compute_daily_report,
    read_error_logs,
    read_risk_logs,
    read_trade_logs,
)
from src.backoffice.equity_manager import load_equity_curve
from src.trading.binance_real_client import binance_real
from src.trading.engine import get_last_signal, get_last_trade
from src.strategies.ml_xgb import ml_xgb_strategy
from src.trading.router import trading_router
from src.trading.risk import risk_manager

router = APIRouter()


@router.get("/logs/trades")
def get_trade_logs(limit: int = Query(100, ge=1, le=1000)):
    return read_trade_logs(limit)


@router.get("/logs/errors")
def get_error_logs(limit: int = Query(100, ge=1, le=1000)):
    return read_error_logs(limit)


@router.get("/logs/risk")
def get_risk_logs(limit: int = Query(100, ge=1, le=1000)):
    return read_risk_logs(limit)


@router.get("/equity-curve")
def get_equity_curve():
    return load_equity_curve()


@router.get("/equity-chart")
def get_equity_chart():
    return load_equity_curve()


@router.get("/daily-report")
def get_daily_report():
    trades = read_trade_logs(limit=5000)
    return compute_daily_report(trades)


@router.get("/monitor")
def monitor_state():
    client = trading_router.get_client()
    risk_status = risk_manager.status()
    
    # Get ML strategy signal
    ml_result = ml_xgb_strategy()
    
    return {
        "position": client.get_position(),
        "last_signal": get_last_signal(),
        "last_ml_signal": ml_result.get("signal"),
        "last_ml_proba_up": ml_result.get("proba_up"),
        "risk": {
            "equity": risk_status.get("equity"),
            "drawdown_today": risk_status.get("drawdown_today"),
            "trading_disabled": bool(risk_status.get("trading_disabled_reason")),
            "reason": risk_status.get("trading_disabled_reason"),
        },
        "last_trade": get_last_trade(),
        "live_mode": binance_real.get_live_mode(),
        "trade_mode": trading_router.mode,
    }
