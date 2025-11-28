"""
디버깅 스크립트: LSTM-Attn 전략 백테스트 파이프라인 점검
"""
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.backtest.engine import run_backtest_with_ml
from src.core.config import settings
from src.dl.lstm_attn_model import get_lstm_attn_model
from src.indicators.basic import get_df_with_indicators
from src.strategies.ml_thresholds import resolve_ml_thresholds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("LSTM-Attn Backtest Debug Test")
    logger.info("=" * 60)

    # 1. 데이터 확인
    logger.info("\n[1] Checking data...")
    try:
        df = get_df_with_indicators()
        logger.info("  ✓ Data loaded: %d rows", len(df))
        logger.info("  ✓ Columns: %s", list(df.columns))
        logger.info("  ✓ Date range: %s to %s", df["timestamp"].min(), df["timestamp"].max())
    except Exception as exc:
        logger.error("  ✗ Failed to load data: %s", exc)
        return

    # 2. 모델 확인
    logger.info("\n[2] Checking LSTM model...")
    try:
        model = get_lstm_attn_model()
        if model is None:
            logger.error("  ✗ Model instance is None")
            return
        if not model.is_loaded():
            logger.error("  ✗ Model not loaded. Path: %s", model.model_path)
            logger.error("  ✗ Path exists: %s", model.model_path.exists())
            return
        logger.info("  ✓ Model loaded: %s", model.model_path)
        logger.info("  ✓ Window size: %d", model.window_size)
    except Exception as exc:
        logger.error("  ✗ Failed to get model: %s", exc)
        return

    # 3. 예측 테스트
    logger.info("\n[3] Testing predictions...")
    try:
        start_idx = model.window_size
        test_indices = [
            start_idx,
            start_idx + 20,
            start_idx + 50,
            min(start_idx + 200, len(df) - 1),
        ]
        for idx in test_indices:
            if idx >= len(df):
                continue
            df_slice = df.iloc[: idx + 1]
            try:
                prob = model.predict_proba_latest(df_slice)
                logger.info("  ✓ Index %d: prob_up = %.4f", idx, prob)
            except Exception as exc:
                logger.warning("  ✗ Index %d prediction failed: %s", idx, exc)
    except Exception as exc:
        logger.error("  ✗ Prediction test failed: %s", exc)
        return

    # 4. Threshold 확인 및 백테스트 실행
    logger.info("\n[4] Resolving thresholds & running backtest...")
    use_optimized = getattr(settings, "USE_OPTIMIZED_THRESHOLDS", False)
    resolved_long, resolved_short = resolve_ml_thresholds(
        long_threshold=None,
        short_threshold=None,
        use_optimized_thresholds=use_optimized,
        strategy_name="ml_lstm_attn",
        default_long=settings.LSTM_ATTN_THRESHOLD_UP,
        default_short=settings.LSTM_ATTN_THRESHOLD_DOWN,
    )
    logger.info(
        "  ✓ Resolved thresholds (USE_OPTIMIZED_THRESHOLDS=%s): long=%.3f, short=%s",
        use_optimized,
        resolved_long,
        f"{resolved_short:.3f}" if resolved_short is not None else "None",
    )
    try:
        result = run_backtest_with_ml(
            long_threshold=resolved_long,
            short_threshold=resolved_short,
            use_optimized_thresholds=False,
            strategy_name="ml_lstm_attn",
        )

        logger.info("  ✓ Backtest completed")
        logger.info("  - Total trades: %d", result["total_trades"])
        logger.info("  - Long trades: %d", result["long_trades"])
        logger.info("  - Short trades: %d", result["short_trades"])
        logger.info("  - Total return: %.4f", result["total_return"])
        logger.info("  - Win rate: %.4f", result["win_rate"])
        logger.info("  - Max drawdown: %.4f", result["max_drawdown"])

        if result["total_trades"] == 0:
            logger.error("\n  ✗ PROBLEM: No trades generated!")
        else:
            logger.info("\n  ✓ Trades were generated successfully")

    except Exception as exc:
        logger.error("  ✗ Backtest failed: %s", exc, exc_info=True)
        return

    logger.info("\n" + "=" * 60)
    logger.info("Debug test complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

