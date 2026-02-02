"""
OHLCV CSV 기간 분리 도구

이 스크립트는 OHLCV CSV 파일을 지정된 기간으로 잘라서 새로운 CSV 파일로 저장합니다.
OOS(Out-of-Sample) 검증을 위한 기간별 데이터셋 생성에 사용됩니다.

사용 예시:
    python tools/slice_ohlcv_csv.py \
        --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
        --out-csv data/ohlcv/BTCUSDT_5m_2024.csv \
        --start 2024-01-01 \
        --end 2024-12-31
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def slice_ohlcv_csv(
    in_csv: Path,
    out_csv: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """
    OHLCV CSV를 지정된 기간으로 잘라서 저장합니다.
    
    Args:
        in_csv: 입력 CSV 파일 경로
        out_csv: 출력 CSV 파일 경로
        start_date: 시작 날짜 (YYYY-MM-DD 형식, 포함)
        end_date: 종료 날짜 (YYYY-MM-DD 형식, 포함)
    """
    logger.info(f"Loading OHLCV CSV from: {in_csv}")
    
    # CSV 로드
    df = pd.read_csv(in_csv)
    
    # timestamp 컬럼 확인 및 변환
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
        if "timestamp" not in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
    else:
        raise ValueError("CSV must have 'timestamp' or 'datetime' column")
    
    logger.info(f"Loaded {len(df)} rows from input CSV")
    logger.info(f"Original timestamp range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 기간 필터링
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df = df[df["timestamp"] >= start_dt]
        logger.info(f"Filtered by start_date >= {start_date}: {len(df)} rows remaining")
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        # end_date 포함 (다음 날 00:00:00 이전)
        end_dt = end_dt + pd.Timedelta(days=1)
        df = df[df["timestamp"] < end_dt]
        logger.info(f"Filtered by end_date < {end_dt} (inclusive of {end_date}): {len(df)} rows remaining")
    
    if len(df) == 0:
        raise ValueError(
            f"No data found in the specified date range "
            f"(start={start_date}, end={end_date})"
        )
    
    # timestamp 컬럼을 원래 형식으로 변환 (CSV 저장 전)
    # 원본 CSV의 timestamp 형식 유지
    if "timestamp" in df.columns:
        # datetime을 문자열로 변환 (ISO 형식)
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # 출력 디렉토리 생성
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV 저장
    logger.info(f"Saving sliced CSV to: {out_csv}")
    df.to_csv(out_csv, index=False)
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("Slicing completed successfully")
    logger.info(f"Output file: {out_csv}")
    logger.info(f"Rows: {len(df):,}")
    if start_date:
        logger.info(f"Start date: {start_date}")
    if end_date:
        logger.info(f"End date: {end_date}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OHLCV CSV 기간 분리 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 2024년 전체 데이터 추출
  python tools/slice_ohlcv_csv.py \\
      --in-csv data/ohlcv/BTCUSDT_5m_full.csv \\
      --out-csv data/ohlcv/BTCUSDT_5m_2024.csv \\
      --start 2024-01-01 \\
      --end 2024-12-31
  
  # 2023년 전체 데이터 추출
  python tools/slice_ohlcv_csv.py \\
      --in-csv data/ohlcv/BTCUSDT_5m_full.csv \\
      --out-csv data/ohlcv/BTCUSDT_5m_2023.csv \\
      --start 2023-01-01 \\
      --end 2023-12-31
        """
    )
    
    parser.add_argument(
        "--in-csv",
        type=Path,
        required=True,
        help="입력 OHLCV CSV 파일 경로",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="출력 CSV 파일 경로",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="시작 날짜 (YYYY-MM-DD 형식, 포함). 기본값: None (제한 없음)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="종료 날짜 (YYYY-MM-DD 형식, 포함). 기본값: None (제한 없음)",
    )
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not args.in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.in_csv}")
    
    # 기간 검증
    if args.start is None and args.end is None:
        logger.warning("Neither --start nor --end specified. Output will be identical to input.")
    
    # 실행
    try:
        slice_ohlcv_csv(
            in_csv=args.in_csv,
            out_csv=args.out_csv,
            start_date=args.start,
            end_date=args.end,
        )
    except Exception as e:
        logger.exception(f"Error during slicing: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()

