"""이벤트 피처 생성 CLI."""
from __future__ import annotations

import logging

from src.core.config import settings
from src.events.dataset import build_event_feature_df
from src.services.ohlcv_service import load_ohlcv_df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("OHLCV 데이터를 로드합니다...")
    df = load_ohlcv_df()
    if df.empty:
        logger.error("OHLCV 데이터가 비어있습니다. 이벤트 피처를 생성할 수 없습니다.")
        return
    logger.info("총 %d개의 캔들이 로드되었습니다.", len(df))

    logger.info("이벤트 피처를 생성합니다 (lookback=%d분)...", settings.EVENTS_LOOKBACK_MINUTES)
    try:
        feature_df = build_event_feature_df(df, save=True)
        if feature_df.empty:
            logger.warning("생성된 이벤트 피처가 비어있습니다.")
        else:
            logger.info("생성된 이벤트 피처 shape=%s", feature_df.shape)
    except Exception as exc:
        logger.error("이벤트 피처 생성 중 오류 발생: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()

