"""
디버깅 스크립트: ML 백테스트가 거래를 생성하지 않는 문제 진단
"""
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.backtest.engine import run_backtest_with_ml
from src.core.config import settings
from src.ml.xgb_model import get_xgb_model
from src.indicators.basic import get_df_with_indicators
from src.strategies.ml_thresholds import resolve_ml_thresholds

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("ML Backtest Debug Test")
    logger.info("=" * 60)
    
    # 1. 데이터 확인
    logger.info("\n[1] Checking data...")
    try:
        df = get_df_with_indicators()
        logger.info(f"  ✓ Data loaded: {len(df)} rows")
        logger.info(f"  ✓ Columns: {list(df.columns)}")
        logger.info(f"  ✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"  ✓ Has NaN in indicators: {df[['ema_20', 'rsi_14', 'sma_20']].isna().any().any()}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load data: {e}")
        return
    
    # 2. 모델 확인
    logger.info("\n[2] Checking model...")
    try:
        model = get_xgb_model()
        if model is None:
            logger.error("  ✗ Model instance is None")
            return
        if not model.is_loaded():
            logger.error(f"  ✗ Model not loaded. Path: {model.model_path}")
            logger.error(f"  ✗ Path exists: {model.model_path.exists()}")
            return
        logger.info(f"  ✓ Model loaded: {model.model_path}")
    except Exception as e:
        logger.error(f"  ✗ Failed to get model: {e}")
        return
    
    # 3. 예측 테스트
    logger.info("\n[3] Testing predictions...")
    try:
        # 처음 몇 개 행에 대해 예측 시도
        test_indices = [20, 50, 100, min(200, len(df)-1)]
        for idx in test_indices:
            if idx >= len(df):
                continue
            df_slice = df.iloc[:idx+1]
            try:
                proba = model.predict_proba_latest(df_slice)
                logger.info(f"  ✓ Index {idx}: proba_up = {proba:.4f}")
            except Exception as e:
                logger.warning(f"  ✗ Index {idx}: Prediction failed - {e}")
    except Exception as e:
        logger.error(f"  ✗ Prediction test failed: {e}")
        return
    
    # 4. Threshold 확인 및 백테스트 실행
    logger.info("\n[4] Resolving thresholds & running backtest...")
    resolved_long, resolved_short = resolve_ml_thresholds(
        long_threshold=None,
        short_threshold=None,
        use_optimized_thresholds=getattr(settings, "USE_OPTIMIZED_THRESHOLDS", False),
    )
    logger.info(
        "  ✓ Resolved thresholds (USE_OPTIMIZED_THRESHOLDS=%s): long=%.3f, short=%s",
        getattr(settings, "USE_OPTIMIZED_THRESHOLDS", False),
        resolved_long,
        f"{resolved_short:.3f}" if resolved_short is not None else "None",
    )
    try:
        result = run_backtest_with_ml(
            long_threshold=None,
            short_threshold=None
        )
        
        logger.info(f"  ✓ Backtest completed")
        logger.info(f"  - Total trades: {result['total_trades']}")
        logger.info(f"  - Long trades: {result['long_trades']}")
        logger.info(f"  - Short trades: {result['short_trades']}")
        logger.info(f"  - Total return: {result['total_return']:.4f}")
        logger.info(f"  - Win rate: {result['win_rate']:.4f}")
        logger.info(f"  - Max drawdown: {result['max_drawdown']:.4f}")
        
        if result['total_trades'] == 0:
            logger.error("\n  ✗ PROBLEM: No trades generated!")
            logger.error("  This confirms the issue. Check logs above for details.")
        else:
            logger.info("\n  ✓ Trades were generated successfully")
            
    except Exception as e:
        logger.error(f"  ✗ Backtest failed: {e}", exc_info=True)
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("Debug test complete")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

