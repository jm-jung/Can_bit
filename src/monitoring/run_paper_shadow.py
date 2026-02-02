#!/usr/bin/env python3
"""
Paper/Shadow 실행 CLI

Guard v2 + Stage-2 v2.2 CAP 로직으로 최신 데이터에서 paper/shadow(로깅-only) 실행합니다.
실주문은 절대 하지 않으며, 모니터링 summary JSON만 생성합니다.

Usage:
    python -m src.monitoring.run_paper_shadow --symbol BTCUSDT --timeframe 5m --lookback-days 30 --update-data 1
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
from src.backtest.backtest_report import print_backtest_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def update_ohlcv_data(symbol: str, timeframe: str) -> bool:
    """OHLCV 데이터 업데이트"""
    try:
        logger.info(f"[Paper/Shadow] OHLCV 데이터 업데이트 시작: {symbol}, {timeframe}")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.data.update_ohlcv",
                "--symbol",
                symbol,
                "--timeframe",
                timeframe,
                "--end",
                "now",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"[Paper/Shadow] OHLCV 데이터 업데이트 완료")
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[Paper/Shadow] OHLCV 데이터 업데이트 실패: {e}")
        if e.stderr:
            logger.error(e.stderr)
        return False
    except Exception as e:
        logger.error(f"[Paper/Shadow] OHLCV 데이터 업데이트 오류: {e}")
        return False


def run_paper_shadow(
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    lookback_days: int = 30,
    update_data: bool = True,
    strategy: str = "ml_tcn",
    direction: str = "long",
) -> bool:
    """
    Paper/Shadow 실행
    
    Args:
        symbol: 거래 심볼
        timeframe: 타임프레임
        lookback_days: 최근 N일 데이터 사용
        update_data: 데이터 업데이트 여부
        strategy: 전략 이름
        direction: 방향 (long/short)
    
    Returns:
        성공 여부
    """
    # 데이터 업데이트
    if update_data:
        if not update_ohlcv_data(symbol, timeframe):
            logger.warning("[Paper/Shadow] 데이터 업데이트 실패했지만 계속 진행합니다.")
    
    # 최신 데이터 범위 계산
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"[Paper/Shadow] Paper/Shadow 실행 시작:")
    logger.info(f"  - 심볼: {symbol}")
    logger.info(f"  - 타임프레임: {timeframe}")
    logger.info(f"  - 전략: {strategy}")
    logger.info(f"  - 방향: {direction}")
    logger.info(f"  - 기간: {start_date_str} ~ {end_date_str} (최근 {lookback_days}일)")
    logger.info(f"  - 모드: paper (실주문 없음, 모니터링만)")
    
    # 백테스트 엔진 직접 호출 (모니터링 로거가 자동으로 summary 생성)
    try:
        engine = get_ml_backtest_engine(
            strategy_name=strategy,
            symbol=symbol,
            timeframe=timeframe,
            feature_preset="extended_safe",
        )
        
        # 날짜 범위 필터링을 위한 index_mask 생성
        # 엔진이 데이터를 로드한 후 필터링해야 하므로, 
        # 먼저 데이터를 로드하여 날짜 범위에 맞는 mask 생성
        import pandas as pd
        from src.services.ohlcv_service import load_ohlcv_df
        
        df_all = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str).replace(hour=23, minute=59, second=59)
        
        index_mask = (df_all["timestamp"] >= start_dt) & (df_all["timestamp"] <= end_dt)
        index_mask_array = index_mask.values
        
        logger.info(f"[Paper/Shadow] 데이터 필터링: 전체 {len(df_all)} rows 중 {index_mask_array.sum()} rows 사용")
        
        # 백테스트 실행 (모니터링 로거가 자동으로 summary 생성)
        result = engine.run_backtest(
            long_threshold=None,
            short_threshold=None,
            use_optimized_threshold=True,
            long_only=(direction == "long"),
            short_only=(direction == "short"),
            signal_confirmation_bars=1,
            min_hold_bars=12,
            cooldown_bars=12,
            use_strategy_guard=True,
            use_strategy_guard_v2=True,
            strategy_guard_v2_mode="soft",
            strategy_guard_v2_scale_floor=0.02,
            use_stage2=True,
            index_mask=index_mask_array,
        )
        
        # 결과 출력
        print_backtest_summary(result, strategy_name=strategy)
        
        # result가 BacktestResult 객체인지 dict인지 확인
        if isinstance(result, dict):
            total_return = result.get("total_return", 0.0)
            total_trades = result.get("total_trades", 0)
        else:
            total_return = result.total_return
            total_trades = result.total_trades
        
        logger.info("[Paper/Shadow] ✅ Paper/Shadow 실행 완료")
        logger.info(f"[Paper/Shadow] 결과: Return={total_return:.2%}, Trades={total_trades}")
        logger.info("[Paper/Shadow] 모니터링 summary JSON이 생성되었습니다:")
        logger.info(f"  - data/monitoring/monitor_guard_stage2_summary_*.json")
        
        return True
        
    except Exception as e:
        logger.error(f"[Paper/Shadow] ❌ Paper/Shadow 실행 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Paper/Shadow 실행 (Guard v2 + Stage-2 v2.2 CAP, 로깅-only)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="거래 심볼 (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5m",
        help="타임프레임 (default: 5m)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="최근 N일 데이터 사용 (default: 30)",
    )
    parser.add_argument(
        "--update-data",
        type=int,
        default=1,
        choices=[0, 1],
        help="데이터 업데이트 여부 (1=업데이트, 0=스킵, default: 1)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ml_tcn",
        choices=["ml_xgb", "ml_lstm_attn", "ml_tcn"],
        help="전략 이름 (default: ml_tcn)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="long",
        choices=["long", "short"],
        help="방향 (default: long)",
    )
    
    args = parser.parse_args()
    
    success = run_paper_shadow(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        update_data=bool(args.update_data),
        strategy=args.strategy,
        direction=args.direction,
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
