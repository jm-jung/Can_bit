"""
Long-period backtest entry point for ML XGBoost strategy.

Usage:
    python -m src.backtest.run_ml_xgb_backtest --symbol BTCUSDT --timeframe 1m --start-date 2024-01-01 --end-date 2024-06-30 --use-optimized-threshold
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from src.backtest.backtest_report import print_backtest_summary, save_backtest_report
from src.core.config import settings
from src.research.ml_strategy_research import run_all_experiments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run long-period backtest for ML XGBoost strategy"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ml_xgb",
        choices=["ml_xgb", "ml_lstm_attn", "ml_tcn"],
        help="Strategy name (default: ml_xgb)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Timeframe (default: 1m)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If not provided, uses all available data.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). If not provided, uses all available data.",
    )
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=None,
        help="Long threshold override (default: from optimized thresholds or settings)",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=None,
        help="Short threshold override (default: from optimized thresholds or settings)",
    )
    parser.add_argument(
        "--use-optimized-threshold",
        action="store_true",
        help="Use optimized ML thresholds from data/thresholds folder",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save backtest report to file",
    )
    parser.add_argument(
        "--research-mode",
        action="store_true",
        default=False,
        help="Run research experiments instead of normal backtest",
    )
    parser.add_argument(
        "--signal-confirmation-bars",
        type=int,
        default=1,
        help="Number of consecutive bars required to confirm signal (default: 1, no smoothing)",
    )
    parser.add_argument(
        "--use-trend-filter",
        action="store_true",
        default=False,
        help="Apply EMA-based trend filter (default: False)",
    )
    parser.add_argument(
        "--trend-ema-window",
        type=int,
        default=200,
        help="EMA window for trend filter (default: 200)",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=None,
        help="Take profit percentage (e.g., 0.003 = 0.3%%), None to disable (default: None)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Stop loss percentage (e.g., 0.002 = 0.2%%), None to disable (default: None)",
    )
    parser.add_argument(
        "--feature-preset",
        type=str,
        default="extended_safe",
        choices=["base", "extended_safe", "extended_full"],
        help="Feature preset for ml_xgb strategy (default: extended_safe)",
    )
    
    # ======================================================================
    # Anti-overtrading parameters (플립/과다매매 억제)
    # ======================================================================
    parser.add_argument(
        "--enter-long-th",
        type=float,
        default=None,
        help="LONG 진입 임계값 (proba_long >= enter_long_th). 기본값: long_threshold와 동일",
    )
    parser.add_argument(
        "--exit-long-th",
        type=float,
        default=None,
        help="LONG 청산 임계값 (proba_long < exit_long_th). 기본값: enter_long_th보다 낮게 설정 권장",
    )
    parser.add_argument(
        "--enter-short-th",
        type=float,
        default=None,
        help="SHORT 진입 임계값 (proba_short >= enter_short_th). 기본값: short_threshold와 동일",
    )
    parser.add_argument(
        "--exit-short-th",
        type=float,
        default=None,
        help="SHORT 청산 임계값 (proba_short < exit_short_th). 기본값: enter_short_th보다 낮게 설정 권장",
    )
    parser.add_argument(
        "--min-hold-bars",
        type=int,
        default=None,
        help="최소 보유 기간 (바). 포지션 진입 후 이 기간 동안은 반대 신호/홀드 신호가 와도 청산 금지. 기본값: None (비활성화)",
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=None,
        help="재진입 쿨다운 (바). 포지션 청산 후 이 기간 동안은 신규 진입 금지. 기본값: None (비활성화)",
    )
    parser.add_argument(
        "--flat-max-th",
        type=float,
        default=None,
        help="FLAT 영역 확대: max(proba_long, proba_short) < flat_max_th 이면 FLAT 유지. 기본값: None (비활성화)",
    )
    parser.add_argument(
        "--margin-th",
        type=float,
        default=None,
        help="FLAT 영역 확대 (마진 기반): |proba_long - proba_short| < margin_th 이면 FLAT 유지. 기본값: None (비활성화)",
    )
    parser.add_argument(
        "--apply-confirmation-to-flips",
        action="store_true",
        default=True,
        help="signal-confirmation-bars를 전환(FLIP)에도 적용 (기본값: True)",
    )
    parser.add_argument(
        "--no-confirmation-to-flips",
        dest="apply_confirmation_to_flips",
        action="store_false",
        help="signal-confirmation-bars를 전환(FLIP)에 적용하지 않음",
    )
    
    # ======================================================================
    # Stage-2 (2단계 게이팅) 옵션
    # ======================================================================
    parser.add_argument(
        "--use-stage2",
        action="store_true",
        default=False,
        help="Stage-2 거래 게이팅 활성화: 방향 후보가 실제로 거래할만한 확신이 있는지 이진 판정. "
             "Trade=False면 포지션 유지, Trade=True면 기존 execution 규칙 적용. "
             "기본값: False (기존 로직과 동일하게 동작)",
    )
    parser.add_argument(
        "--stage2-trade-th",
        type=float,
        default=None,
        help="Stage-2 절대 임계값: p_long >= stage2_trade_th (LONG) 또는 p_short >= stage2_trade_th (SHORT) "
             "일 때 Trade=True. 기본값: None (비활성)",
    )
    parser.add_argument(
        "--stage2-min-edge",
        type=float,
        default=0.0,
        help="Stage-2 마진 임계값: (p_long - p_short) >= stage2_min_edge (LONG) 또는 "
             "(p_short - p_long) >= stage2_min_edge (SHORT) 일 때 Trade=True. 기본값: 0.0",
    )
    parser.add_argument(
        "--stage2-exit-on-flat",
        action="store_true",
        default=False,
        help="Stage-1이 FLAT/HOLD 후보일 때 강제 청산(Exit-to-flat) 수행. 기본값: False (포지션 유지)",
    )
    parser.add_argument(
        "--stage2-allow-flip",
        action="store_true",
        default=True,
        help="Stage-2에서 포지션 전환(FLIP) 허용. 기본값: True",
    )
    parser.add_argument(
        "--no-stage2-allow-flip",
        dest="stage2_allow_flip",
        action="store_false",
        help="Stage-2에서 포지션 전환(FLIP) 금지",
    )
    parser.add_argument(
        "--stage2-cooldown-bars",
        type=int,
        default=0,
        help="Stage-2로 거래한 직후 추가 쿨다운 기간(바). 기본값: 0 (비활성)",
    )
    parser.add_argument(
        "--stage2-block-if-final-scale-below",
        type=float,
        default=0.0,
        help="Stage-2 v2.2: final_scale이 이 값보다 낮으면 ENTRY 차단. 기본값: 0.0 (비활성)",
    )
    parser.add_argument(
        "--stage2-cap-entropy-high-th",
        type=float,
        default=0.64,  # 완화: 0.66 → 0.64
        help="Stage-2 CAP: high entropy 임계값 (cap=0.6 조건). 기본값: 0.64",
    )
    parser.add_argument(
        "--stage2-cap-entropy-mid-th",
        type=float,
        default=0.62,  # 완화: 0.64 → 0.62
        help="Stage-2 CAP: mid entropy 임계값 (cap=0.8 조건). 기본값: 0.62",
    )
    parser.add_argument(
        "--stage2-cap-pdiff-tiny-th",
        type=float,
        default=0.002,
        help="Stage-2 CAP: tiny p_diff 임계값 (cap=0.6 조건). 기본값: 0.002",
    )
    parser.add_argument(
        "--stage2-cap-pdiff-small-th",
        type=float,
        default=0.005,
        help="Stage-2 CAP: small p_diff 임계값 (cap=0.8 조건). 기본값: 0.005 (pdiff_small_005 승격)",
    )
    
    # ======================================================================
    # Direction filter (LONG-only / SHORT-only)
    # ======================================================================
    parser.add_argument(
        "--direction",
        type=str,
        choices=["both", "long", "short"],
        default="both",
        help="거래 방향 필터: 'both' (LONG+SHORT 모두), 'long' (LONG-only), 'short' (SHORT-only). "
             "기본값: both (기존 동작 유지)",
    )
    
    # ======================================================================
    # StrategyGuard (전략 실행 허용/차단)
    # ======================================================================
    parser.add_argument(
        "--use-strategy-guard",
        action="store_true",
        default=False,
        help="StrategyGuard 활성화: 최근 트레이드 성능을 기반으로 실행 허용/차단. "
             "기본값: False (기존 동작 유지)",
    )
    
    # ======================================================================
    # StrategyGuard Phase-2 옵션 (UNBLOCK + 히스테리시스)
    # ======================================================================
    parser.add_argument(
        "--strategy-guard-min-win-rate",
        type=float,
        default=None,
        help="BLOCK 조건: win_rate < 이 값이면 BLOCK. "
             "기본값: 0.4 (기존 MVP와 동일하게 유지하려면 미지정)",
    )
    parser.add_argument(
        "--strategy-guard-min-avg-return",
        type=float,
        default=None,
        help="BLOCK 조건: avg_return < 이 값이면 BLOCK. "
             "기본값: -0.01 (기존 MVP와 동일하게 유지하려면 미지정)",
    )
    parser.add_argument(
        "--strategy-guard-unblock-win-rate",
        type=float,
        default=None,
        help="UNBLOCK 조건: win_rate >= 이 값이면 BLOCK에서 ALLOW로 복귀. "
             "기본값: 0.45 (기존 MVP와 동일하게 유지하려면 미지정)",
    )
    parser.add_argument(
        "--strategy-guard-unblock-avg-return",
        type=float,
        default=None,
        help="UNBLOCK 조건: avg_return >= 이 값이면 BLOCK에서 ALLOW로 복귀. "
             "기본값: 0.0 (기존 MVP와 동일하게 유지하려면 미지정)",
    )
    parser.add_argument(
        "--strategy-guard-min-block-trades",
        type=int,
        default=None,
        help="BLOCK 최소 유지 트레이드 수 (히스테리시스). "
             "BLOCK 발생 후 이 수만큼 트레이드가 완료되어야 UNBLOCK 조건 체크 가능. "
             "기본값: 5 (기존 MVP와 동일하게 유지하려면 미지정)",
    )
    parser.add_argument(
        "--strategy-guard-recent-trades-window",
        type=int,
        default=None,
        help="최근 N개 트레이드 기준. "
             "기본값: 20 (기존 MVP와 동일하게 유지하려면 미지정)",
    )
    parser.add_argument(
        "--strategy-guard-insufficient-sample-policy",
        type=str,
        choices=["allow", "block", "defer"],
        default=None,
        help="표본 부족 정책: recent_trades_count < recent_trades_window일 때 정책. "
             "'allow' (기본값, 기존 동작): ALLOW 반환. "
             "'block': BLOCK 반환. "
             "'defer': DEFER 반환 (판단 스킵). "
             "기본값: allow (기존 동작 유지)",
    )
    
    # ======================================================================
    # StrategyGuard v2 옵션 (신호 품질/불확실성 기반)
    # ======================================================================
    parser.add_argument(
        "--strategy-guard-v2",
        action="store_true",
        default=False,
        help="StrategyGuard v2 활성화: 신호 품질/불확실성 기반 판단 (SOFT 모드: position_scale 지원). "
             "기본값: False (기존 v1 동작 유지)",
    )
    parser.add_argument(
        "--strategy-guard-v2-mode",
        type=str,
        choices=["soft", "hard"],
        default=None,
        help="Guard v2 모드: 'soft' (position_scale 0.0~1.0, 기본값), 'hard' (ALLOW/BLOCK만). "
             "기본값: soft (--strategy-guard-v2 사용 시)",
    )
    parser.add_argument(
        "--strategy-guard-v2-window-signal-stats",
        type=int,
        default=None,
        help="신호 통계 윈도우 (bars). 최근 N개 신호의 평균 margin/entropy 계산에 사용. "
             "기본값: 200",
    )
    parser.add_argument(
        "--strategy-guard-v2-min-margin",
        type=float,
        default=None,
        help="최소 결정 마진 (|p_long - 0.5| 또는 p_long - threshold). "
             "기본값: 0.02",
    )
    parser.add_argument(
        "--strategy-guard-v2-max-entropy",
        type=float,
        default=None,
        help="최대 엔트로피 (0~0.693, 0.693은 완전 불확실). "
             "기본값: 0.65",
    )
    parser.add_argument(
        "--strategy-guard-v2-scale-floor",
        type=float,
        default=None,
        help="최소 position_scale (0~1). "
             "기본값: 0.2",
    )
    parser.add_argument(
        "--strategy-guard-v2-block-if-scale-below",
        type=float,
        default=None,
        help="hard 모드: position_scale이 이 값보다 낮으면 BLOCK. "
             "기본값: 0.05",
    )
    
    # ======================================================================
    # Trade dump options
    # ======================================================================
    
    # ======================================================================
    # SHORT Strategy MVP
    # ======================================================================
    parser.add_argument(
        "--enable-short-strategy",
        action="store_true",
        default=False,
        help="SHORT 전략 MVP 활성화: 기존 ml_tcn 전략의 SHORT 실행 모드. "
             "요구사항: direction=both, Stage-2=ON, StrategyGuard=ON. "
             "기본값: False (기존 동작 유지)",
    )
    
    # ======================================================================
    # Trade dump options (트레이드 이벤트 덤프)
    # ======================================================================
    parser.add_argument(
        "--dump-trades",
        action="store_true",
        default=False,
        help="트레이드 이벤트(ENTRY/EXIT)를 CSV 파일로 덤프. 기본값: False",
    )
    parser.add_argument(
        "--dump-trades-path",
        type=str,
        default=None,
        help="덤프 파일 경로. 기본값: data/backtest_dumps/trades_{strategy}_{symbol}_{timeframe}_{direction}_{start}_{end}.csv",
    )
    
    # ======================================================================
    # Commission and Slippage (for cost analysis)
    # ======================================================================
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=None,
        help="Commission rate (e.g., 0.0004 = 0.04%%). 기본값: 엔진 기본값 사용",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=None,
        help="Slippage rate (e.g., 0.0005 = 0.05%%). 기본값: 엔진 기본값 사용",
    )
    
    return parser.parse_args()


def filter_data_by_date(
    df,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Filter DataFrame by date range.
    
    Args:
        df: DataFrame with timestamp column
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        Filtered DataFrame
    """
    if start_date is None and end_date is None:
        return df
    
    if "timestamp" not in df.columns:
        logger.warning("DataFrame does not have 'timestamp' column. Skipping date filter.")
        return df
    
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        df = df[df["timestamp"] >= start_dt]
        logger.info(f"Filtered data: start_date >= {start_date}")
    
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        # Include the entire end date
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
        df = df[df["timestamp"] <= end_dt]
        logger.info(f"Filtered data: end_date <= {end_date}")
    
    logger.info(f"Filtered data shape: {df.shape}")
    return df


def main():
    """Main entry point for long-period backtest."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Long-Period ML Backtest")
    logger.info("=" * 60)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
    if args.strategy == "ml_xgb":
        logger.info(f"Feature preset: {args.feature_preset}")
    logger.info(f"Start date: {args.start_date or 'All available'}")
    logger.info(f"End date: {args.end_date or 'All available'}")
    logger.info("=" * 60)
    
    # Log threshold usage flag
    logger.info(
        "[ML Backtest] use_optimized_threshold=%s",
        args.use_optimized_threshold
    )
    
    # Research mode: run experiments instead of normal backtest
    if args.research_mode:
        logger.info("[ML Backtest] Research mode enabled - running experiments...")
        results = run_all_experiments(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            use_optimized_threshold=args.use_optimized_threshold,
        )
        return results
    
    # Normal backtest mode
    # Use strategy-aware backtest engine
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    
    engine = get_ml_backtest_engine(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        feature_preset=args.feature_preset,
    )
    
    # Convert direction to long_only/short_only for backward compatibility
    long_only = (args.direction == "long")
    short_only = (args.direction == "short")
    
    # Log direction setting
    logger.info(f"[ML Backtest] Direction filter: {args.direction} (long_only={long_only}, short_only={short_only})")
    
    # Generate dump file path if dump is enabled
    dump_trades_path = None
    if args.dump_trades:
        if args.dump_trades_path:
            dump_trades_path = args.dump_trades_path
        else:
            # Generate default path
            import os
            from datetime import datetime
            dump_dir = "data/backtest_dumps"
            os.makedirs(dump_dir, exist_ok=True)
            
            start_str = args.start_date.replace("-", "") if args.start_date else "all"
            end_str = args.end_date.replace("-", "") if args.end_date else "all"
            dump_trades_path = os.path.join(
                dump_dir,
                f"trades_{args.strategy}_{args.symbol}_{args.timeframe}_{args.direction}_{start_str}_{end_str}.csv"
            )
        logger.info(f"[ML Backtest] Trade dump enabled: {dump_trades_path}")
    
    # Run backtest using engine
    result = engine.run_backtest(
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        use_optimized_threshold=args.use_optimized_threshold,
        signal_confirmation_bars=args.signal_confirmation_bars,
        use_trend_filter=args.use_trend_filter,
        trend_ema_window=args.trend_ema_window,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        # Direction filter
        long_only=long_only,
        short_only=short_only,
        # Anti-overtrading parameters
        enter_long_th=args.enter_long_th,
        exit_long_th=args.exit_long_th,
        enter_short_th=args.enter_short_th,
        exit_short_th=args.exit_short_th,
        min_hold_bars=args.min_hold_bars,
        cooldown_bars=args.cooldown_bars,
        flat_max_th=args.flat_max_th,
        margin_th=args.margin_th,
        apply_confirmation_to_flips=args.apply_confirmation_to_flips,
        # Stage-2 options
        use_stage2=args.use_stage2,
        stage2_trade_th=args.stage2_trade_th,
        stage2_min_edge=args.stage2_min_edge,
        stage2_exit_on_flat=args.stage2_exit_on_flat,
        stage2_allow_flip=args.stage2_allow_flip,
        stage2_cooldown_bars=args.stage2_cooldown_bars,
        stage2_block_if_final_scale_below=args.stage2_block_if_final_scale_below,
        # Stage-2 CAP 임계값 (스윕용)
        stage2_cap_entropy_high_th=args.stage2_cap_entropy_high_th,
        stage2_cap_entropy_mid_th=args.stage2_cap_entropy_mid_th,
        stage2_cap_pdiff_tiny_th=args.stage2_cap_pdiff_tiny_th,
        stage2_cap_pdiff_small_th=args.stage2_cap_pdiff_small_th,
        # StrategyGuard
        use_strategy_guard=args.use_strategy_guard,
        # StrategyGuard Phase-2 options
        strategy_guard_min_win_rate=args.strategy_guard_min_win_rate,
        strategy_guard_min_avg_return=args.strategy_guard_min_avg_return,
        strategy_guard_unblock_win_rate=args.strategy_guard_unblock_win_rate,
        strategy_guard_unblock_avg_return=args.strategy_guard_unblock_avg_return,
        strategy_guard_min_block_trades=args.strategy_guard_min_block_trades,
        strategy_guard_recent_trades_window=args.strategy_guard_recent_trades_window,
        strategy_guard_insufficient_sample_policy=args.strategy_guard_insufficient_sample_policy,
        # StrategyGuard v2
        use_strategy_guard_v2=args.strategy_guard_v2,
        strategy_guard_v2_mode=args.strategy_guard_v2_mode,
        strategy_guard_v2_window_signal_stats=args.strategy_guard_v2_window_signal_stats,
        strategy_guard_v2_min_margin=args.strategy_guard_v2_min_margin,
        strategy_guard_v2_max_entropy=args.strategy_guard_v2_max_entropy,
        strategy_guard_v2_scale_floor=args.strategy_guard_v2_scale_floor,
        strategy_guard_v2_block_if_scale_below=args.strategy_guard_v2_block_if_scale_below,
        # Trade dump
        dump_trades_path=dump_trades_path,
        # Commission and slippage
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
    )
    
    # Print summary
    print_backtest_summary(result, strategy_name=args.strategy)
    
    # Save report
    if not args.no_save:
        report_path = save_backtest_report(
            result=result,
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            # Stage-2 parameters
            use_stage2=args.use_stage2,
            stage2_trade_th=args.stage2_trade_th,
            stage2_min_edge=args.stage2_min_edge,
            stage2_exit_on_flat=args.stage2_exit_on_flat,
            stage2_allow_flip=args.stage2_allow_flip,
            stage2_cooldown_bars=args.stage2_cooldown_bars,
            stage2_block_if_final_scale_below=args.stage2_block_if_final_scale_below,
            # Stage-2 CAP 임계값 (스윕용)
            stage2_cap_entropy_high_th=args.stage2_cap_entropy_high_th,
            stage2_cap_entropy_mid_th=args.stage2_cap_entropy_mid_th,
            stage2_cap_pdiff_tiny_th=args.stage2_cap_pdiff_tiny_th,
            stage2_cap_pdiff_small_th=args.stage2_cap_pdiff_small_th,
            # Direction filter
            direction=args.direction,
            # StrategyGuard
            use_strategy_guard=args.use_strategy_guard,
            # StrategyGuard Phase-2 options
            strategy_guard_unblock_win_rate=args.strategy_guard_unblock_win_rate,
            strategy_guard_unblock_avg_return=args.strategy_guard_unblock_avg_return,
            strategy_guard_min_block_trades=args.strategy_guard_min_block_trades,
            # SHORT Strategy MVP
            enable_short_strategy=args.enable_short_strategy,
        )
        logger.info(f"Backtest report saved to: {report_path}")
    
    return result


if __name__ == "__main__":
    import pandas as pd
    main()

