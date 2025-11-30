"""
테스트/디버깅 전용: 가짜 이벤트 삽입 및 피처 생성 확인 스크립트

이 스크립트는 이벤트 파이프라인의 동작을 확인하기 위해:
1. 가짜 이벤트를 raw 이벤트 저장소에 삽입
2. build_event_feature_frame을 호출하여 피처 생성 확인

주의사항:
- 이 스크립트는 테스트/디버깅 전용입니다.
- 여기서 넣은 가짜 이벤트는 `data/events/raw/BTCUSDT_1m_events.parquet`에 append됩니다.
- 실제 운영 전에 이 테스트 데이터를 지우고 싶으면:
  - `data/events/raw/BTCUSDT_1m_events.parquet` 파일을 직접 삭제하면 됩니다.
- 프로덕션 코드(src/ml, src/backtest 등)는 수정하지 않습니다.

실행 예시:
(.venv) PS C:\Canbit\Can_bit> python -m src.events.dev_event_pipeline_sanity
"""
from __future__ import annotations

import logging

import pandas as pd

from src.events.aggregator import build_event_feature_frame
from src.events.store import append_raw_events, load_raw_events
from src.services.ohlcv_service import load_ohlcv_df

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 테스트용 공통 상수
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("[Test] Inserting fake raw events and building event features...")
    logger.info("[Test] Using dynamic test window from actual OHLCV data.")
    logger.info("=" * 60)

    # 1. OHLCV 데이터 로딩
    logger.info("[Test] Loading OHLCV data...")
    ohlcv_df = load_ohlcv_df()
    logger.info(f"[Test] OHLCV shape: {ohlcv_df.shape}")

    # timestamp 컬럼이 있으면 인덱스로 설정
    if "timestamp" in ohlcv_df.columns:
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"])
        ohlcv_df = ohlcv_df.set_index("timestamp").sort_index()

    # 인덱스가 DatetimeIndex인지 확인
    if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
        logger.error(
            "[Test] OHLCV index is not DatetimeIndex. "
            "Cannot proceed with event feature building."
        )
        logger.info("=" * 60)
        logger.info("[Test] Event pipeline sanity check completed.")
        logger.info("=" * 60)
        exit(1)

    # 2. OHLCV 인덱스 범위 로깅 및 동적 윈도우 계산
    min_ts = ohlcv_df.index.min()
    max_ts = ohlcv_df.index.max()
    logger.info(f"[Test] OHLCV index min: {min_ts}, max: {max_ts}")

    # For safety, if there are not enough rows, handle that gracefully
    if min_ts is None or max_ts is None:
        logger.warning("[Test] OHLCV index has no data. Aborting test.")
        logger.info("=" * 60)
        logger.info("[Test] Event pipeline sanity check completed.")
        logger.info("=" * 60)
        exit(1)

    # Define a small window somewhere in the beginning portion of the data
    # Example: window starts 30 minutes after min_ts, spans 4 hours
    test_start = min_ts + pd.Timedelta(minutes=30)
    test_end = test_start + pd.Timedelta(hours=4)

    # Clamp test_end to not exceed max_ts
    if test_end > max_ts:
        test_end = max_ts

    logger.info(f"[Test] Using dynamic test window: start={test_start}, end={test_end}")

    # 3. OHLCV 슬라이스
    ohlcv_slice = ohlcv_df[
        (ohlcv_df.index >= test_start) & (ohlcv_df.index <= test_end)
    ].copy()
    logger.info(
        f"[Test] OHLCV slice shape: {ohlcv_slice.shape} "
        f"(from {test_start} to {test_end})"
    )

    if ohlcv_slice.empty:
        logger.warning(
            "[Test] OHLCV slice is empty. "
            "Cannot build event features. Check dynamic window logic."
        )
        logger.info("=" * 60)
        logger.info("[Test] Event pipeline sanity check completed.")
        logger.info("=" * 60)
        exit(1)

    # 4. 가짜 이벤트 생성 (동적 윈도우 내에)
    logger.info("[Test] Generating fake events inside dynamic window...")

    base = test_start
    offsets = [30, 40, 70, 105, 140]  # minutes from base

    fake_timestamps = []
    for offset in offsets:
        ts = base + pd.Timedelta(minutes=offset)
        if ts <= test_end:
            fake_timestamps.append(ts)

    # If after clamping we have fewer than 5 timestamps, just use as many as possible
    if not fake_timestamps:
        logger.warning(
            "[Test] No valid timestamps for fake events inside test window. Aborting."
        )
        logger.info("=" * 60)
        logger.info("[Test] Event pipeline sanity check completed.")
        logger.info("=" * 60)
        exit(1)

    # Build the fake events DataFrame
    categories = ["INFLUENCER", "INSTITUTION", "MACRO_POLICY", "REGULATION", "GEOPOLITICAL"]
    sentiments = [0.6, -0.3, 0.9, -0.5, 0.2]
    intensities = [0.7, 0.4, 0.95, 0.6, 0.3]

    # Trim lists to match the number of fake_timestamps
    n = len(fake_timestamps)
    categories = categories[:n]
    sentiments = sentiments[:n]
    intensities = intensities[:n]

    fake_events = pd.DataFrame({
        "timestamp": fake_timestamps,
        "category": categories,
        "sentiment_score": sentiments,
        "intensity": intensities,
        "source_type": ["TEST"] * n,
        "symbol": [SYMBOL] * n,
    })

    logger.info("[Test] Creating fake events inside dynamic window:")
    logger.info("\n%s", fake_events)

    # 5. 가짜 이벤트 삽입
    logger.info("[Test] Appending fake events to raw event store...")
    before_df = load_raw_events(SYMBOL, TIMEFRAME)
    logger.info(f"[Test] Raw events before insert: {len(before_df)}")

    append_raw_events(SYMBOL, TIMEFRAME, fake_events)

    after_df = load_raw_events(SYMBOL, TIMEFRAME)
    inserted = len(after_df) - len(before_df)
    logger.info(f"[Test] Raw events after insert: {len(after_df)} (inserted={inserted})")

    # 6. 이벤트 피처 프레임 빌드
    logger.info("[Test] Building event feature frame...")
    event_features = build_event_feature_frame(
        ohlcv_df=ohlcv_slice,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        lookback_minutes=None,  # use default from settings
    )

    logger.info(f"[Test] Event feature frame shape: {event_features.shape}")

    # 7. Non-zero counts 계산 및 로깅
    non_zero_counts = (event_features != 0).sum()
    logger.info("[Test] Non-zero counts per event feature column:")
    logger.info("\n%s", non_zero_counts)

    # 8. 샘플 데이터 출력
    logger.info("[Test] Sample of event features (head 10):")
    logger.info("\n%s", event_features.head(10))

    # 9. 검증: 이벤트 피처가 실제로 non-zero 값을 갖는지 확인
    total_non_zero = non_zero_counts.sum()
    if total_non_zero == 0:
        logger.error(
            "[Test] ⚠️  WARNING: All event features are zero! "
            "Fake events may not have been processed correctly."
        )
    else:
        logger.info(
            f"[Test] ✓ Verification: {total_non_zero} non-zero values found "
            f"across all event columns. Event features are being generated."
        )

    logger.info("=" * 60)
    logger.info("[Test] Event pipeline sanity check completed.")
    logger.info("=" * 60)
