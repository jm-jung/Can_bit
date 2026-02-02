#!/usr/bin/env python3
"""
OHLCV 증분 업데이트 CLI

로컬 데이터의 마지막 timestamp 이후부터 현재까지 OHLCV를 증분 fetch하여 병합합니다.

Usage:
    python -m src.data.update_ohlcv --symbol BTCUSDT --timeframe 5m --end now
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.fetch_binance_ohlcv import fetch_full_history, load_existing_csv
from src.services.ohlcv_service import OHLCV_CSV_PATHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_ohlcv_path(symbol: str, timeframe: str) -> Path:
    """OHLCV 파일 경로 결정"""
    # OHLCV_CSV_PATHS에서 찾기
    key = (symbol.upper(), timeframe.lower())
    if key in OHLCV_CSV_PATHS:
        path = OHLCV_CSV_PATHS[key]
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path
    
    # 기본 경로 (1m인 경우)
    if timeframe.lower() == "1m":
        return PROJECT_ROOT / "src" / "data" / "btc_ohlcv.csv"
    
    # 5m인 경우
    if timeframe.lower() == "5m":
        return PROJECT_ROOT / "data" / "ohlcv" / f"{symbol.upper()}_{timeframe.lower()}_full.csv"
    
    # 기타
    return PROJECT_ROOT / "data" / "ohlcv" / f"{symbol.upper()}_{timeframe.lower()}_full.csv"


def update_ohlcv(
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    end: str = "now",
    insecure_ssl: bool = False,
    fallback_direct_klines: bool = False,
    timeout_ms: int = 30000,
) -> dict:
    """
    OHLCV 증분 업데이트
    
    Args:
        symbol: 거래 심볼 (예: BTCUSDT)
        timeframe: 타임프레임 (예: 5m)
        end: 종료 시점 ("now" 또는 "YYYY-MM-DD")
    
    Returns:
        업데이트 결과 딕셔너리
    """
    # CCXT 심볼 형식으로 변환 (BTC/USDT)
    ccxt_symbol = symbol.replace("USDT", "/USDT") if not "/" in symbol else symbol
    
    # OHLCV 파일 경로
    csv_path = get_ohlcv_path(symbol, timeframe)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 기존 데이터 로드
    existing_df = load_existing_csv(csv_path)
    
    before_last_ts = None
    before_last_ts_dt = None
    if len(existing_df) > 0:
        # _ts_dt 컬럼이 있으면 사용, 없으면 timestamp 파싱
        if "_ts_dt" in existing_df.columns:
            before_last_ts_dt = existing_df["_ts_dt"].max()
            before_last_ts = before_last_ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            before_last_ts = existing_df["timestamp"].max()
            before_last_ts_dt = pd.to_datetime(before_last_ts, utc=True)
        logger.info(f"[Update] 기존 데이터: {len(existing_df)} rows, 마지막 timestamp: {before_last_ts}")
    else:
        logger.info("[Update] 기존 데이터 없음. 전체 히스토리 fetch 필요.")
        return {
            "status": "error",
            "message": "기존 데이터가 없습니다. fetch_binance_ohlcv.py를 사용하여 초기 데이터를 먼저 가져오세요.",
        }
    
    # since 계산 (기존 데이터의 마지막 timestamp 이후)
    since_dt = before_last_ts_dt if before_last_ts_dt is not None else pd.to_datetime(before_last_ts, utc=True)
    since_str = since_dt.strftime("%Y-%m-%d %H:%M")
    
    # end 계산 (UTC-aware로 통일)
    if end.lower() == "now":
        end_dt = pd.Timestamp.now(tz="UTC")
    else:
        try:
            end_dt_naive = datetime.strptime(end, "%Y-%m-%d")
            end_dt = pd.Timestamp(end_dt_naive, tz="UTC")
        except ValueError:
            try:
                end_dt_naive = datetime.strptime(end, "%Y-%m-%d %H:%M")
                end_dt = pd.Timestamp(end_dt_naive, tz="UTC")
            except ValueError:
                logger.error(f"[Update] 잘못된 end 형식: {end}")
                return {"status": "error", "message": f"잘못된 end 형식: {end}"}
    
    # since가 end보다 늦으면 업데이트 불필요
    if since_dt >= end_dt:
        logger.info(f"[Update] 이미 최신 데이터입니다. (마지막: {since_dt}, 요청: {end_dt})")
        return {
            "status": "skipped",
            "before_last_ts": str(before_last_ts),
            "after_last_ts": str(before_last_ts),
            "fetched_rows": 0,
            "merged_total_rows": len(existing_df),
        }
    
    logger.info(f"[Update] 증분 업데이트 시작: {since_str} ~ {end_dt.strftime('%Y-%m-%d %H:%M')} (UTC)")
    
    # fetch_full_history 호출 (append 모드)
    try:
        fetch_full_history(
            symbol=ccxt_symbol,
            timeframe=timeframe,
            since=since_str,
            append=True,
            outfile=csv_path,
        )
        
        # 업데이트된 데이터 로드
        updated_df = load_existing_csv(csv_path)
        
        # 최신 timestamp 계산
        if "_ts_dt" in updated_df.columns:
            after_last_ts_dt = updated_df["_ts_dt"].max()
            after_last_ts = after_last_ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            after_last_ts = updated_df["timestamp"].max()
            after_last_ts_dt = pd.to_datetime(after_last_ts, utc=True)
        
        # 신규 row 계산 (before_last_ts_dt 이후의 데이터)
        if "_ts_dt" in updated_df.columns and before_last_ts_dt is not None:
            new_rows = len(updated_df[updated_df["_ts_dt"] > before_last_ts_dt])
        else:
            # fallback: 전체 row 차이
            new_rows = len(updated_df) - len(existing_df)
        
        # 전체 fetched rows (fetch_full_history에서 가져온 row 수)
        # fetch_full_history의 로그에서 확인하거나, 여기서는 new_rows로 근사
        fetched_rows_approx = new_rows
        
        # 운영 안전장치: 실제로 새로운 데이터가 있어야 하는데 반영되지 않았을 때만 에러
        # fetch_full_history에서 이미 검증하므로, 여기서는 정보만 출력
        if new_rows == 0 and before_last_ts_dt is not None and after_last_ts_dt is not None:
            if after_last_ts_dt <= before_last_ts_dt:
                # 가져온 데이터가 모두 중복이거나 이미 존재하는 범위 → 정상
                logger.info(
                    f"[Update] No new rows appended (all fetched data were duplicates or already exists). "
                    f"before={before_last_ts}, after={after_last_ts}, new_rows={new_rows}"
                )
        
        result = {
            "status": "success",
            "before_last_ts": str(before_last_ts),
            "after_last_ts": str(after_last_ts),
            "fetched_rows": fetched_rows_approx,
            "new_rows": new_rows,
            "merged_total_rows": len(updated_df),
            "csv_path": str(csv_path),
        }
        
        logger.info(f"[Update] ✅ 업데이트 완료:")
        logger.info(f"  - 이전 마지막: {before_last_ts}")
        logger.info(f"  - 현재 마지막: {after_last_ts}")
        logger.info(f"  - 신규 반영 행 수: {new_rows}")
        logger.info(f"  - 총 행 수: {len(updated_df)}")
        logger.info(f"  - 저장 경로: {csv_path}")
        
        # 모니터링 로그 저장 (선택)
        monitoring_dir = PROJECT_ROOT / "data" / "monitoring"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        log_file = monitoring_dir / f"ohlcv_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        return result
        
    except Exception as e:
        logger.error(f"[Update] ❌ 업데이트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
        }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="OHLCV 증분 업데이트 (로컬 데이터의 마지막 timestamp 이후부터 현재까지)"
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
        "--end",
        type=str,
        default="now",
        help="종료 시점: 'now' 또는 'YYYY-MM-DD' (default: now)",
    )
    parser.add_argument(
        "--insecure-ssl",
        type=int,
        default=0,
        choices=[0, 1],
        help="SSL 인증서 검증 비활성화 (비추천, 필요한 경우만) (default: 0)",
    )
    parser.add_argument(
        "--fallback-direct-klines",
        type=int,
        default=0,
        choices=[0, 1],
        help="CCXT 대신 직접 klines API 사용 (최후 수단) (default: 0)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="네트워크 타임아웃 (밀리초) (default: 30000)",
    )
    
    args = parser.parse_args()
    
    # 환경변수 확인 (CLI 인자보다 우선순위 낮음)
    import os
    insecure_ssl_env = os.getenv("CANBIT_INSECURE_SSL", "0") == "1"
    insecure_ssl = bool(args.insecure_ssl) or insecure_ssl_env
    
    result = update_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        end=args.end,
        insecure_ssl=insecure_ssl,
        fallback_direct_klines=bool(args.fallback_direct_klines),
        timeout_ms=args.timeout_ms,
    )
    
    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
