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
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
from src.backtest.backtest_report import print_backtest_summary
from src.optimization.ml_proba_cache import _get_cache_path, get_or_build_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_ROOT / "data/monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)


def _as_iso(ts_obj: Any) -> str | None:
    if ts_obj is None:
        return None
    try:
        return str(ts_obj)
    except Exception:
        return None


def _safe_ts_minmax(df) -> tuple[str | None, str | None]:
    if df is None or "timestamp" not in df.columns or len(df) == 0:
        return None, None
    return _as_iso(df["timestamp"].min()), _as_iso(df["timestamp"].max())


def _load_cache_with_meta(cache_path: Path) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    cache_df = pd.read_parquet(cache_path)
    ts_min, ts_max = _safe_ts_minmax(cache_df)
    return cache_df, {
        "cache_source_path": str(cache_path),
        "proba_source_path": str(cache_path),
        "proba_len": int(len(cache_df)),
        "proba_ts_min": ts_min,
        "proba_ts_max": ts_max,
        "cache_mtime": datetime.fromtimestamp(cache_path.stat().st_mtime).isoformat(),
    }


def _inject_summary_meta(run_id: str | None, payload: dict[str, Any]) -> Path | None:
    if not run_id:
        return None
    summary_path = MONITORING_DIR / f"monitor_guard_stage2_summary_{run_id}.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    summary.update(payload)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def _find_latest_run_id_from_summary() -> str | None:
    candidates = sorted(MONITORING_DIR.glob("monitor_guard_stage2_summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates[:10]:
        try:
            summary = json.loads(p.read_text(encoding="utf-8"))
            rid = summary.get("run_id")
            if rid:
                return str(rid)
        except Exception:
            continue
    return None


def _extract_eval_ts_from_jsonl(run_id: str) -> tuple[str | None, str | None, str | None, str | None]:
    jsonl_path = MONITORING_DIR / f"monitor_guard_stage2_{run_id}.jsonl"
    if not jsonl_path.exists():
        return None, None, None, None
    ts_all: list[str] = []
    entry_ts: list[str] = []
    exit_ts: list[str] = []
    try:
        for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            ts = ev.get("ts")
            if ts:
                ts_all.append(str(ts))
            et = ev.get("event")
            if et == "ENTRY" and ts:
                entry_ts.append(str(ts))
            elif et == "EXIT" and ts:
                exit_ts.append(str(ts))
    except Exception:
        return None, None, None, None
    return (
        min(ts_all) if ts_all else None,
        max(ts_all) if ts_all else None,
        min(entry_ts) if entry_ts else None,
        max(exit_ts) if exit_ts else None,
    )


def _write_failure_summary(run_id: str, payload: dict[str, Any]) -> Path:
    summary_path = MONITORING_DIR / f"monitor_guard_stage2_summary_{run_id}.json"
    summary = {
        "run_id": run_id,
        "mode": "backtest",
        "timestamp": datetime.now().isoformat(),
        "evaluation_status": "FAILED_INDEX_MASK_MISMATCH",
        "pipeline_status": "NO_VALID_EVAL",
        "total_checks": 0,
        "entries_attempted": 0,
        "entries_executed": 0,
        "total_trades": 0,
        "blocked_by_min_hold": 0,
        "blocked_by_cooldown": 0,
        "blocked_by_guard_hard": 0,
        "total_return": None,
        "win_rate": None,
        "max_drawdown": None,
        "avg_profit": None,
        "equity_final": None,
    }
    summary.update(payload)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


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
    force_refresh_proba: bool = False,
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
        force_refresh_proba: proba/cache 강제 재생성 여부
    
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
        import pandas as pd
        from src.services.ohlcv_service import load_ohlcv_df

        engine = get_ml_backtest_engine(
            strategy_name=strategy,
            symbol=symbol,
            timeframe=timeframe,
            feature_preset="extended_safe",
        )

        # 날짜 범위 필터링을 위한 index_mask 생성
        df_all = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str).replace(hour=23, minute=59, second=59)

        index_mask = (df_all["timestamp"] >= start_dt) & (df_all["timestamp"] <= end_dt)
        index_mask_array = index_mask.values

        ohlcv_ts_min = _as_iso(df_all["timestamp"].min()) if len(df_all) else None
        ohlcv_ts_max = _as_iso(df_all["timestamp"].max()) if len(df_all) else None
        index_mask_true_count = int(index_mask_array.sum())

        logger.info(f"[Paper/Shadow] 데이터 필터링: 전체 {len(df_all)} rows 중 {index_mask_array.sum()} rows 사용")

        # ------------------------------------------------------------------
        # Preflight: engine이 실제 사용할 cache/proba 기준으로 정합 점검
        # ------------------------------------------------------------------
        use_events: bool | None = None
        tcn_preset: str | None = getattr(engine, "tcn_preset", None)
        if strategy == "ml_tcn" and not tcn_preset:
            import os

            ev = os.environ.get("USE_EVENTS_FOR_TCN", "").strip().lower()
            use_events = ev not in ("0", "false", "no")
        cache_path = _get_cache_path(
            strategy_name=strategy,
            symbol=symbol,
            timeframe=timeframe,
            feature_preset="base",
            use_events=use_events,
            tcn_preset=tcn_preset if strategy == "ml_tcn" else None,
        )
        if not cache_path.exists() or force_refresh_proba:
            logger.warning(
                "[Paper/Shadow] proba cache rebuild requested (missing=%s, force=%s): %s",
                (not cache_path.exists()),
                force_refresh_proba,
                cache_path,
            )
            get_or_build_predictions(
                strategy_name=strategy,
                symbol=symbol,
                timeframe=timeframe,
                feature_preset="base",
                force_rebuild=True,
                tcn_preset=tcn_preset if strategy == "ml_tcn" else None,
            )

        # engine.load_predictions() 결과를 기준으로 mask 생성해야 length mismatch를 피할 수 있음
        proba_long_arr, proba_short_arr, engine_df = engine.load_predictions()
        engine_df_len = int(len(engine_df))
        engine_df_ts_min, engine_df_ts_max = _safe_ts_minmax(engine_df)
        cache_meta = {
            "cache_source_path": str(cache_path),
            "proba_source_path": str(cache_path),
            "proba_len": int(len(proba_long_arr)),
            "proba_ts_min": engine_df_ts_min,
            "proba_ts_max": engine_df_ts_max,
            "cache_mtime": datetime.fromtimestamp(cache_path.stat().st_mtime).isoformat() if cache_path.exists() else None,
        }

        # 평가용 mask는 engine_df timestamp 기준으로 재계산
        if "timestamp" not in engine_df.columns:
            raise ValueError("[Paper/Shadow] engine_df has no timestamp column")
        engine_ts = pd.to_datetime(engine_df["timestamp"])
        eval_mask = (engine_ts >= start_dt) & (engine_ts <= end_dt)
        index_mask_array = eval_mask.values
        index_mask_true_count = int(index_mask_array.sum())
        index_mask_len = int(len(index_mask_array))
        preflight_mismatch = index_mask_len != engine_df_len

        # stale 판단: engine/proba max가 OHLCV max보다 크게 뒤쳐지면 실패 처리
        cache_stale = False
        cache_lag_hours = None
        try:
            if ohlcv_ts_max and engine_df_ts_max:
                ohlcv_ts_max_obj = pd.to_datetime(ohlcv_ts_max)
                engine_ts_max_obj = pd.to_datetime(engine_df_ts_max)
                cache_lag_hours = float((ohlcv_ts_max_obj - engine_ts_max_obj).total_seconds() / 3600.0)
                cache_stale = cache_lag_hours > 24.0
        except Exception:
            cache_stale = False

        mask_apply_reason = "ok"
        if preflight_mismatch:
            mask_apply_reason = "index_mask_len_mismatch_vs_engine_df"
        elif index_mask_true_count <= 0:
            mask_apply_reason = "index_mask_true_count_zero"
        elif cache_stale:
            mask_apply_reason = "stale_proba_cache_over_24h"

        base_diag = {
            "lookback_days": int(lookback_days),
            "ohlcv_path": str(PROJECT_ROOT / "data/ohlcv" / f"{symbol}_{timeframe}_full.csv"),
            "ohlcv_len": int(len(df_all)),
            "ohlcv_ts_min": ohlcv_ts_min,
            "ohlcv_ts_max": ohlcv_ts_max,
            "engine_df_len": engine_df_len,
            "engine_df_ts_min": engine_df_ts_min,
            "engine_df_ts_max": engine_df_ts_max,
            "index_mask_len": index_mask_len,
            "index_mask_true_count": index_mask_true_count,
            "mask_applied": False,
            "mask_apply_reason": mask_apply_reason,
            "backtest_eval_ts_min": None,
            "backtest_eval_ts_max": None,
            "first_trade_entry_ts": None,
            "last_trade_exit_ts": None,
            "evaluation_status": None,
            "pipeline_status": None,
            "cache_stale_detected": cache_stale,
            "cache_lag_hours": cache_lag_hours,
        }
        base_diag.update(cache_meta)

        logger.info("[PAPER_SHADOW][DATA_DIAG] %s", json.dumps(base_diag, ensure_ascii=False))

        # mismatch/빈 mask는 정상 성과를 만들지 않고 실패 summary 기록
        if preflight_mismatch or index_mask_true_count <= 0 or cache_stale:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            fail_payload = dict(base_diag)
            fail_payload.update(
                {
                    "run_id": run_id,
                    "evaluation_status": (
                        "FAILED_INDEX_MASK_MISMATCH"
                        if preflight_mismatch
                        else ("FAILED_EMPTY_INDEX_MASK" if index_mask_true_count <= 0 else "FAILED_STALE_PROBA_CACHE")
                    ),
                    "pipeline_status": "NO_VALID_EVAL",
                }
            )
            summary_path = _write_failure_summary(run_id, fail_payload)
            logger.error(
                "[Paper/Shadow] NO_VALID_EVAL: preflight failed (mismatch=%s, mask_true=%s). summary=%s",
                preflight_mismatch,
                index_mask_true_count,
                summary_path,
            )
            return False

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
            proba_long_cache=proba_long_arr,
            proba_short_cache=proba_short_arr,
            df_with_proba=engine_df,
            index_mask=index_mask_array,
        )

        # 결과 출력
        print_backtest_summary(result, strategy_name=strategy)

        # result가 BacktestResult 객체인지 dict인지 확인
        if isinstance(result, dict):
            total_return = result.get("total_return", 0.0)
            total_trades = result.get("total_trades", 0)
            run_id = result.get("run_id")
            trades = result.get("trades", []) or []
            eval_df = result.get("df", None)
            backtest_eval_ts_min = _as_iso(eval_df["timestamp"].min()) if isinstance(eval_df, pd.DataFrame) and "timestamp" in eval_df.columns and len(eval_df) else None
            backtest_eval_ts_max = _as_iso(eval_df["timestamp"].max()) if isinstance(eval_df, pd.DataFrame) and "timestamp" in eval_df.columns and len(eval_df) else None
        else:
            total_return = result.total_return
            total_trades = result.total_trades
            run_id = getattr(result, "run_id", None)
            trades = getattr(result, "trades", []) or []
            eval_df = getattr(result, "df", None)
            backtest_eval_ts_min = _as_iso(eval_df["timestamp"].min()) if isinstance(eval_df, pd.DataFrame) and "timestamp" in eval_df.columns and len(eval_df) else None
            backtest_eval_ts_max = _as_iso(eval_df["timestamp"].max()) if isinstance(eval_df, pd.DataFrame) and "timestamp" in eval_df.columns and len(eval_df) else None

        if not run_id:
            run_id = _find_latest_run_id_from_summary()

        first_trade_entry_ts = None
        last_trade_exit_ts = None
        if trades:
            first_trade_entry_ts = _as_iso(getattr(trades[0], "entry_time", None) or (trades[0].get("entry_time") if isinstance(trades[0], dict) else None))
            last_trade_exit_ts = _as_iso(getattr(trades[-1], "exit_time", None) or (trades[-1].get("exit_time") if isinstance(trades[-1], dict) else None))
        if run_id:
            jsonl_eval_min, jsonl_eval_max, jsonl_first_entry, jsonl_last_exit = _extract_eval_ts_from_jsonl(run_id)
            if backtest_eval_ts_min is None:
                backtest_eval_ts_min = jsonl_eval_min
            if backtest_eval_ts_max is None:
                backtest_eval_ts_max = jsonl_eval_max
            if first_trade_entry_ts is None:
                first_trade_entry_ts = jsonl_first_entry
            if last_trade_exit_ts is None:
                last_trade_exit_ts = jsonl_last_exit

        diag_payload = dict(base_diag)
        diag_payload.update(
            {
                "run_id": run_id,
                "mask_applied": True,
                "mask_apply_reason": "applied",
                "backtest_eval_ts_min": backtest_eval_ts_min,
                "backtest_eval_ts_max": backtest_eval_ts_max,
                "first_trade_entry_ts": first_trade_entry_ts,
                "last_trade_exit_ts": last_trade_exit_ts,
                "evaluation_status": "OK",
                "pipeline_status": "OK",
            }
        )
        summary_path = _inject_summary_meta(run_id, diag_payload)
        logger.info("[PAPER_SHADOW][DATA_DIAG] %s", json.dumps(diag_payload, ensure_ascii=False))
        if summary_path:
            logger.info("[Paper/Shadow] DATA_DIAG injected into summary: %s", summary_path)

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
    parser.add_argument(
        "--force-refresh-proba",
        type=int,
        default=0,
        choices=[0, 1],
        help="proba/cache 강제 재생성 여부 (1=강제, default: 0)",
    )
    
    args = parser.parse_args()
    
    success = run_paper_shadow(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        update_data=bool(args.update_data),
        strategy=args.strategy,
        direction=args.direction,
        force_refresh_proba=bool(args.force_refresh_proba),
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
