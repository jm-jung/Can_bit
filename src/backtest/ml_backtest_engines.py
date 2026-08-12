"""
Strategy-aware ML backtest engines.

This module provides abstract and concrete implementations of backtest engines
for different ML strategies (XGBoost, LSTM-Attention).
"""
from __future__ import annotations

import json
import logging
import math
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.ml_backtest_types import BacktestResult, Signal, Trade
from src.core.config import settings

logger = logging.getLogger(__name__)


def resolve_profit_lock_intrabar(
    direction: str,
    entry_price: float,
    current_high: float,
    current_low: float,
    take_profit: float,
    stop_loss: float,
    *,
    conservative: bool,
) -> tuple[str | None, float | None]:
    """
    현재 봉의 high/low만 보고 profit lock TP/SL 여부를 판별한다.
    LONG: TP는 상단 터치(high), SL은 하단 터치(low) 시 해당 극값으로 청산 가정.
    SHORT: 반대. 동일 바 TP·SL 동시 충족 시 conservative=True면 SL 우선.
    """
    if entry_price <= 0:
        return None, None
    d = (direction or "LONG").upper()
    sl_mag = abs(float(stop_loss))

    if d == "LONG":
        tp_px = entry_price * (1.0 + float(take_profit))
        sl_px = entry_price * (1.0 + float(stop_loss))
        tp_hit = current_high >= tp_px
        sl_hit = current_low <= sl_px
        if tp_hit and sl_hit:
            if conservative:
                return "profit_lock_sl", float(current_low)
            return "profit_lock_tp", float(current_high)
        if sl_hit:
            return "profit_lock_sl", float(current_low)
        if tp_hit:
            return "profit_lock_tp", float(current_high)
        return None, None

    # SHORT
    tp_px = entry_price / (1.0 + float(take_profit))
    sl_px = entry_price / (1.0 + sl_mag)
    tp_hit = current_low <= tp_px
    sl_hit = current_high >= sl_px
    if tp_hit and sl_hit:
        if conservative:
            return "profit_lock_sl", float(current_high)
        return "profit_lock_tp", float(current_low)
    if sl_hit:
        return "profit_lock_sl", float(current_high)
    if tp_hit:
        return "profit_lock_tp", float(current_low)
    return None, None


def overtrading_reentry_block(
    bar_index: int,
    signal: str,
    last_exit_bar_index: int | None,
    last_exit_direction: str | None,
    last_exit_reason: str | None,
    *,
    enable_reentry_cooldown: bool,
    reentry_cooldown_bars: int,
    enable_directional_cooldown: bool,
    directional_cooldown_bars: int,
    enable_post_tp_block: bool,
    post_tp_block_bars: int,
    enable_post_sl_block: bool,
    post_sl_block_bars: int,
) -> tuple[bool, str | None, str | None]:
    """
    재진입 억제 필터. 우선순위(로깅): post_sl > post_tp > directional > global.
    반환: (blocked, cooldown_type, skip_reason)
    """
    if signal not in ("LONG", "SHORT"):
        return False, None, None
    if last_exit_bar_index is None:
        return False, None, None

    bars_since = bar_index - int(last_exit_bar_index)
    ler = last_exit_reason or ""

    if enable_post_sl_block and ler == "profit_lock_sl" and bars_since < int(post_sl_block_bars):
        return True, "post_sl", "post_sl_block"

    if enable_post_tp_block and ler == "profit_lock_tp" and bars_since < int(post_tp_block_bars):
        return True, "post_tp", "post_tp_block"

    if (
        enable_directional_cooldown
        and last_exit_direction is not None
        and signal == last_exit_direction
        and bars_since < int(directional_cooldown_bars)
    ):
        return True, "directional", "directional_cooldown"

    if enable_reentry_cooldown and bars_since < int(reentry_cooldown_bars):
        return True, "global", "reentry_cooldown"

    return False, None, None


# 모니터링 디렉토리
MONITORING_DIR = Path("data/monitoring")
MONITORING_DIR.mkdir(parents=True, exist_ok=True)


class MonitoringLogger:
    """실전 모니터링 로거 (shadow mode, logging-only)"""
    
    def __init__(self, run_id: str | None = None, mode: str = "backtest"):
        """
        모니터링 로거 초기화
        
        Args:
            run_id: 실행 단위 UUID (None이면 자동 생성)
            mode: 실행 모드 (live / paper / backtest)
        """
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mode = mode
        self.log_file = MONITORING_DIR / f"monitor_guard_stage2_{self.run_id}.jsonl"
        self.summary_file = MONITORING_DIR / f"monitor_guard_stage2_summary_{self.run_id}.json"
        self._file_handle = None
        self._check_count = 0
        self._entry_count = 0
        self._exit_count = 0
        
        # 집계용 데이터 (샘플링된 CHECK 이벤트용)
        self.guard_scales: list[float] = []
        self.stage2_caps: list[float] = []
        self.final_scales: list[float] = []
        
        # 전체 bar 기준 분포 집계 (100% 수집, 샘플링 없음)
        self.guard_scale_all_checks: list[float] = []
        self.stage2_cap_all_checks: list[float] = []
        self.final_scale_all_checks: list[float] = []
        self.cap_reason_all_checks: list[str] = []
        
    def _to_python_type(self, val):
        """NumPy 타입을 Python 기본 타입으로 변환"""
        if val is None:
            return None
        if isinstance(val, (np.integer, np.floating)):
            return float(val) if isinstance(val, np.floating) else int(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val
    
    def _write_log(self, event_data: dict):
        """JSONL 파일에 로그 기록"""
        if self._file_handle is None:
            self._file_handle = open(self.log_file, "a", encoding="utf-8")
        
        json_line = json.dumps(event_data, default=str, ensure_ascii=False)
        self._file_handle.write(json_line + "\n")
        self._file_handle.flush()  # 즉시 디스크에 쓰기
    
    def log_check(
        self,
        ts: str,
        symbol: str,
        timeframe: str,
        bar_index: int | None,
        guard_scale: float | None,
        guard_components: dict | None,
        stage2_cap: float | None,
        cap_reason: str | None,
        final_scale: float | None,
        signal: str | None = None,
        sample_rate: int = 10,  # N bar 중 1회만 상세 로그
    ):
        """
        CHECK 시점 로그 기록
        
        Args:
            ts: 타임스탬프
            symbol: 심볼
            timeframe: 타임프레임
            bar_index: 바 인덱스
            guard_scale: Guard v2 scale
            guard_components: Guard v2 입력 요소 (p_long, margin, entropy 등)
            stage2_cap: Stage-2 CAP 값
            cap_reason: CAP 적용 이유
            final_scale: 최종 scale
            signal: 신호 (LONG/SHORT/HOLD/FLAT)
            sample_rate: 샘플링 비율 (N bar 중 1회)
        """
        self._check_count += 1
        
        # 전체 bar 기준 분포 집계 (100% 수집, 샘플링 없음)
        if guard_scale is not None:
            self.guard_scale_all_checks.append(guard_scale)
        if stage2_cap is not None:
            self.stage2_cap_all_checks.append(stage2_cap)
        if final_scale is not None:
            self.final_scale_all_checks.append(final_scale)
        if cap_reason is not None:
            self.cap_reason_all_checks.append(cap_reason)
        
        # 샘플링: sample_rate bar 중 1회만 상세 로그
        if self._check_count % sample_rate != 0:
            return
        
        event_data = {
            "event": "CHECK",
            "ts": ts,
            "run_id": self.run_id,
            "mode": self.mode,
            "symbol": symbol,
            "timeframe": timeframe,
            "bar_index": bar_index,
            "signal": signal,
            "guard_scale": self._to_python_type(guard_scale),
            "guard_components": {
                k: self._to_python_type(v) for k, v in (guard_components or {}).items()
            },
            "stage2_cap": self._to_python_type(stage2_cap),
            "cap_reason": cap_reason,
            "final_scale": self._to_python_type(final_scale),
        }
        
        self._write_log(event_data)
        
        # 샘플링된 데이터 수집 (기존 호환성 유지)
        if guard_scale is not None:
            self.guard_scales.append(guard_scale)
        if stage2_cap is not None:
            self.stage2_caps.append(stage2_cap)
        if final_scale is not None:
            self.final_scales.append(final_scale)
    
    def log_entry(
        self,
        ts: str,
        symbol: str,
        timeframe: str,
        bar_index: int | None,
        trade_id: int,
        side: str,
        entry_price: float,
        guard_scale: float | None,
        guard_components: dict | None,
        stage2_cap: float | None,
        cap_reason: str | None,
        final_scale: float | None,
        position_size: float | None = None,
        position_notional: float | None = None,
        position_size_source: str | None = None,
    ):
        """ENTRY 시점 로그 기록"""
        self._entry_count += 1
        
        event_data = {
            "event": "ENTRY",
            "ts": ts,
            "run_id": self.run_id,
            "mode": self.mode,
            "symbol": symbol,
            "timeframe": timeframe,
            "bar_index": bar_index,
            "trade_id": trade_id,
            "side": side,
            "entry_price": self._to_python_type(entry_price),
            "guard_scale": self._to_python_type(guard_scale),
            "guard_components": {
                k: self._to_python_type(v) for k, v in (guard_components or {}).items()
            },
            "stage2_cap": self._to_python_type(stage2_cap),
            "cap_reason": cap_reason,
            "final_scale": self._to_python_type(final_scale),
            "position_size": self._to_python_type(position_size),
        }
        
        # position_size null 보강
        if position_notional is not None:
            event_data["position_notional"] = self._to_python_type(position_notional)
        if position_size_source is not None:
            event_data["position_size_source"] = position_size_source
        elif position_size is None:
            # position_size가 null이고 source도 없으면 원인 명시
            event_data["position_size_source"] = "unavailable"
        
        self._write_log(event_data)
    
    def log_exit(
        self,
        ts: str,
        symbol: str,
        timeframe: str,
        bar_index: int | None,
        trade_id: int,
        exit_price: float,
        realized_profit: float,
        holding_bars: int,
        mfe: float | None = None,
        mae: float | None = None,
    ):
        """EXIT 시점 로그 기록"""
        self._exit_count += 1
        
        event_data = {
            "event": "EXIT",
            "ts": ts,
            "run_id": self.run_id,
            "mode": self.mode,
            "symbol": symbol,
            "timeframe": timeframe,
            "bar_index": bar_index,
            "trade_id": trade_id,
            "exit_price": self._to_python_type(exit_price),
            "realized_profit": self._to_python_type(realized_profit),
            "holding_bars": holding_bars,
            "mfe": self._to_python_type(mfe),
            "mae": self._to_python_type(mae),
        }
        
        self._write_log(event_data)
    
    def save_summary(
        self,
        total_checks: int,
        entries_attempted: int,
        entries_executed: int,
        total_trades: int,
        blocked_by_min_hold: int = 0,
        blocked_by_cooldown: int = 0,
        blocked_by_guard_hard: int = 0,
        config_snapshot: dict | None = None,
        *,
        total_return: float | None = None,
        max_drawdown: float | None = None,
        win_rate: float | None = None,
        avg_profit: float | None = None,
        equity_final: float | None = None,
    ):
        """집계 요약 스냅샷 저장 (리스크 통계 + 선택적 백테스트 성과 지표)"""
        import numpy as np
        from collections import Counter
        
        summary = {
            "run_id": self.run_id,
            "mode": self.mode,
            "timestamp": datetime.now().isoformat(),
            "total_checks": total_checks,
            "entries_attempted": entries_attempted,
            "entries_executed": entries_executed,
            "total_trades": total_trades,
            "blocked_by_min_hold": blocked_by_min_hold,
            "blocked_by_cooldown": blocked_by_cooldown,
            "blocked_by_guard_hard": blocked_by_guard_hard,
        }
        
        # 백테스트 성과 지표 (Discord 리포트 등에서 사용)
        if total_return is not None or max_drawdown is not None or win_rate is not None or avg_profit is not None or equity_final is not None:
            summary.update({
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "equity_final": equity_final,
            })
        
        # Guard scale 통계 (샘플링된 데이터, 기존 호환성)
        if self.guard_scales:
            scales = np.array(self.guard_scales)
            summary["guard_scale_stats"] = {
                "count": len(self.guard_scales),
                "mean": float(np.mean(scales)),
                "median": float(np.median(scales)),
                "p10": float(np.percentile(scales, 10)),
                "p90": float(np.percentile(scales, 90)),
                "min": float(np.min(scales)),
                "max": float(np.max(scales)),
            }
        
        # Stage-2 CAP 분포 (샘플링된 데이터, 기존 호환성)
        if self.stage2_caps:
            cap_counts = {1.0: 0, 0.8: 0, 0.6: 0}
            for cap in self.stage2_caps:
                if abs(cap - 1.0) < 0.001:
                    cap_counts[1.0] += 1
                elif abs(cap - 0.8) < 0.001:
                    cap_counts[0.8] += 1
                elif abs(cap - 0.6) < 0.001:
                    cap_counts[0.6] += 1
            
            total_caps = len(self.stage2_caps)
            summary["stage2_cap_distribution"] = {
                "cap_1_0": cap_counts[1.0],
                "cap_1_0_pct": round(cap_counts[1.0] / total_caps * 100, 1) if total_caps > 0 else 0,
                "cap_0_8": cap_counts[0.8],
                "cap_0_8_pct": round(cap_counts[0.8] / total_caps * 100, 1) if total_caps > 0 else 0,
                "cap_0_6": cap_counts[0.6],
                "cap_0_6_pct": round(cap_counts[0.6] / total_caps * 100, 1) if total_caps > 0 else 0,
                "total": total_caps,
            }
        
        # Final scale 통계 (샘플링된 데이터, 기존 호환성)
        if self.final_scales:
            final_scales = np.array(self.final_scales)
            summary["final_scale_stats"] = {
                "count": len(self.final_scales),
                "mean": float(np.mean(final_scales)),
                "median": float(np.median(final_scales)),
                "p10": float(np.percentile(final_scales, 10)),
                "p90": float(np.percentile(final_scales, 90)),
                "min": float(np.min(final_scales)),
                "max": float(np.max(final_scales)),
            }
        
        # ======================================================================
        # 전체 bar 기준 분포 집계 (100% 수집, 샘플링 없음)
        # ======================================================================
        
        # Guard scale 히스토그램 (버킷 0.0~1.0, step 0.1)
        if self.guard_scale_all_checks:
            guard_histogram = {}
            for bucket_start in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                bucket_end = bucket_start + 0.1
                count = sum(1 for s in self.guard_scale_all_checks if bucket_start <= s < bucket_end)
                guard_histogram[f"{bucket_start:.1f}~{bucket_end:.1f}"] = count
            # 1.0은 별도 처리
            count_1_0 = sum(1 for s in self.guard_scale_all_checks if abs(s - 1.0) < 0.001)
            guard_histogram["1.0"] = count_1_0
            summary["guard_scale_histogram"] = guard_histogram
        
        # Stage-2 CAP 히스토그램 (1.0/0.8/0.6 카운트)
        if self.stage2_cap_all_checks:
            cap_histogram = {1.0: 0, 0.8: 0, 0.6: 0, "other": 0}
            for cap in self.stage2_cap_all_checks:
                if abs(cap - 1.0) < 0.001:
                    cap_histogram[1.0] += 1
                elif abs(cap - 0.8) < 0.001:
                    cap_histogram[0.8] += 1
                elif abs(cap - 0.6) < 0.001:
                    cap_histogram[0.6] += 1
                else:
                    cap_histogram["other"] += 1
            total_all_caps = len(self.stage2_cap_all_checks)
            summary["stage2_cap_histogram"] = {
                "cap_1_0": cap_histogram[1.0],
                "cap_1_0_pct": round(cap_histogram[1.0] / total_all_caps * 100, 1) if total_all_caps > 0 else 0,
                "cap_0_8": cap_histogram[0.8],
                "cap_0_8_pct": round(cap_histogram[0.8] / total_all_caps * 100, 1) if total_all_caps > 0 else 0,
                "cap_0_6": cap_histogram[0.6],
                "cap_0_6_pct": round(cap_histogram[0.6] / total_all_caps * 100, 1) if total_all_caps > 0 else 0,
                "other": cap_histogram["other"],
                "total": total_all_caps,
            }
        
        # Final scale 히스토그램 (버킷 0.0~1.0, step 0.1)
        if self.final_scale_all_checks:
            final_histogram = {}
            for bucket_start in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                bucket_end = bucket_start + 0.1
                count = sum(1 for s in self.final_scale_all_checks if bucket_start <= s < bucket_end)
                final_histogram[f"{bucket_start:.1f}~{bucket_end:.1f}"] = count
            # 1.0은 별도 처리
            count_1_0 = sum(1 for s in self.final_scale_all_checks if abs(s - 1.0) < 0.001)
            final_histogram["1.0"] = count_1_0
            summary["final_scale_histogram"] = final_histogram
        
        # Cap reason 카운트 (상위 N개만)
        if self.cap_reason_all_checks:
            reason_counter = Counter(self.cap_reason_all_checks)
            top_reasons = reason_counter.most_common(10)  # 상위 10개
            summary["cap_reason_counts"] = {
                reason: count for reason, count in top_reasons
            }
        
        # Config snapshot 추가
        if config_snapshot:
            summary["config_snapshot"] = config_snapshot
        
        # JSON 파일로 저장
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[MONITOR] 요약 스냅샷 저장: {self.summary_file}")
    
    def close(self):
        """파일 핸들 닫기"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


def calculate_stage2_cap(
    stage2_reason: str,
    proba_long: float | None = None,
    proba_short: float | None = None,
    raw_signal: str | None = None,
    # v2.2 cap 임계값 (코드 상수)
    high_entropy_th: float = 0.64,  # 완화: 0.66 → 0.64
    mid_entropy_th: float = 0.62,   # 완화: 0.64 → 0.62
    tiny_pdiff_th: float = 0.002,
    small_pdiff_th: float = 0.005,  # 기본값 승격: pdiff_small_005 (기존 0.004)
    # v2.2 cap 값 (완화된 룰: {0.6, 0.8, 1.0})
    cap_low: float = 0.6,   # 기존 0.35 → 0.6 (과도한 축소 방지)
    cap_mid: float = 0.8,   # 기존 0.55 → 0.8 (중간 위험 구간 완화)
    cap_default: float = 1.0,  # 정상 구간 (변경 없음)
) -> tuple[float, dict]:
    """
    Stage-2 v2.2 cap 계산 (조건부 캡 방식).
    
    Args:
        stage2_reason: Stage-2 판정 이유
        proba_long: Long 확률 (optional)
        proba_short: Short 확률 (optional)
        raw_signal: 원본 신호 (LONG/SHORT/HOLD/FLAT, optional)
        high_entropy_th: High entropy 임계값
        mid_entropy_th: Mid entropy 임계값
        tiny_pdiff_th: Tiny p_diff 임계값
        small_pdiff_th: Small p_diff 임계값
    
    Returns:
        (cap, debug_info): cap 값 (0.35/0.55/1.0), debug_info 딕셔너리
    """
    import math
    
    cap = 1.0  # default
    cap_reason = "default"
    
    debug_info = {
        "cap": 1.0,
        "p_diff": None,
        "entropy": None,
        "cap_reason": "default",
        "cap_low": cap_low,
        "cap_mid": cap_mid,
    }
    
    # p_diff 추출
    p_diff = None
    import re
    p_diff_match = re.search(r"p_diff=([\d.-]+)", stage2_reason)
    if p_diff_match:
        try:
            p_diff = float(p_diff_match.group(1))
        except ValueError:
            pass
    
    # 파싱 실패 시 proba_long/proba_short에서 직접 계산
    if p_diff is None and proba_long is not None and proba_short is not None:
        if raw_signal == "LONG":
            p_diff = proba_long - proba_short
        elif raw_signal == "SHORT":
            p_diff = proba_short - proba_long
        else:
            p_diff = abs(proba_long - proba_short)
    
    debug_info["p_diff"] = p_diff
    
    # entropy 계산
    entropy = None
    if proba_long is not None and proba_short is not None:
        if raw_signal == "LONG":
            p = proba_long
        elif raw_signal == "SHORT":
            p = proba_short
        else:
            p = max(proba_long, proba_short)
        
        if 0 < p < 1:
            entropy = -p * math.log(p) - (1 - p) * math.log(1 - p)
    
    debug_info["entropy"] = entropy
    
    # cap 계산 (조건부)
    if entropy is not None and p_diff is not None:
        abs_pdiff = abs(p_diff)
        
        if entropy >= high_entropy_th and abs_pdiff <= tiny_pdiff_th:
            cap = cap_low  # 완화된 룰: 0.6 (기존 0.35)
            cap_reason = f"high_entropy({entropy:.4f})_tiny_pdiff({abs_pdiff:.4f})"
        elif entropy >= mid_entropy_th and abs_pdiff <= small_pdiff_th:
            cap = cap_mid  # 완화된 룰: 0.8 (기존 0.55)
            cap_reason = f"mid_entropy({entropy:.4f})_small_pdiff({abs_pdiff:.4f})"
        else:
            cap_reason = "no_cap"
    else:
        cap_reason = "no_data"
    
    debug_info["cap"] = cap
    debug_info["cap_reason"] = cap_reason
    
    return cap, debug_info
    """
    Stage-2 v2.2 cap 계산 (조건부 캡 방식).
    
    Args:
        stage2_reason: Stage-2 판정 이유
        proba_long: Long 확률 (optional)
        proba_short: Short 확률 (optional)
        raw_signal: 원본 신호 (LONG/SHORT/HOLD/FLAT, optional)
        high_entropy_th: 높은 엔트로피 임계값
        mid_entropy_th: 중간 엔트로피 임계값
        tiny_pdiff_th: 매우 작은 p_diff 임계값
        small_pdiff_th: 작은 p_diff 임계값
    
    Returns:
        (cap, reason): cap 값 (0.35/0.55/1.0), 적용 이유
    """
    import math
    
    cap = 1.0  # default
    cap_reason = "default"
    
    # p_diff 추출
    p_diff = None
    import re
    p_diff_match = re.search(r"p_diff=([\d.-]+)", stage2_reason)
    if p_diff_match:
        try:
            p_diff = float(p_diff_match.group(1))
        except ValueError:
            pass
    
    # 파싱 실패 시 proba_long/proba_short에서 직접 계산
    if p_diff is None and proba_long is not None and proba_short is not None:
        if raw_signal == "LONG":
            p_diff = proba_long - proba_short
        elif raw_signal == "SHORT":
            p_diff = proba_short - proba_long
        else:
            p_diff = abs(proba_long - proba_short)
    
    # entropy 계산
    entropy = None
    if proba_long is not None and proba_short is not None:
        if raw_signal == "LONG":
            p = proba_long
        elif raw_signal == "SHORT":
            p = proba_short
        else:
            p = max(proba_long, proba_short)
        
        if 0 < p < 1:
            entropy = -p * math.log(p) - (1 - p) * math.log(1 - p)
    
    # cap 계산 (조건부)
    if entropy is not None and p_diff is not None:
        abs_pdiff = abs(p_diff)
        
        if entropy >= high_entropy_th and abs_pdiff <= tiny_pdiff_th:
            cap = cap_low  # 완화된 룰: 0.6 (기존 0.35)
            cap_reason = f"high_entropy({entropy:.4f})_tiny_pdiff({abs_pdiff:.4f})"
        elif entropy >= mid_entropy_th and abs_pdiff <= small_pdiff_th:
            cap = cap_mid  # 완화된 룰: 0.8 (기존 0.55)
            cap_reason = f"mid_entropy({entropy:.4f})_small_pdiff({abs_pdiff:.4f})"
        else:
            cap_reason = "no_cap"
    else:
        cap_reason = "no_data"
    
    return cap, cap_reason

def calculate_stage2_quality_score(
    stage2_trade: bool,
    stage2_reason: str,
    stage2_scale_floor: float = 0.2,
    # v2.1 추가 파라미터 (optional, fallback 사용)
    proba_long: float | None = None,
    proba_short: float | None = None,
    raw_signal: str | None = None,
    trend_ema: float | None = None,
) -> tuple[float, dict]:
    """
    Stage-2 quality score 계산 (Guard v2 친화적 soft gating, v2.1 연속 스코어).
    
    Args:
        stage2_trade: Stage-2 Trade=True/False
        stage2_reason: Stage-2 판정 이유
        stage2_scale_floor: Trade=False일 때 최소 스케일 (0~1)
        proba_long: Long 확률 (optional)
        proba_short: Short 확률 (optional)
        raw_signal: 원본 신호 (LONG/SHORT/HOLD/FLAT, optional)
        trend_ema: Trend EMA 값 (optional)
    
    Returns:
        (quality_score, debug_info): 
        - quality_score: 0.0 (hard block) ~ 1.0 (full scale)
        - debug_info: factor 값들 (regime_factor, risk_factor, ev_factor, base)
    """
    import math
    
    debug_info = {
        "base": 1.0,
        "regime_factor": 1.0,
        "risk_factor": 1.0,
        "ev_factor": 1.0,
        "ev": None,
        "regime": None,
        "risk": None,
        "vol": None,
    }
    
    # 극단적 위험/결측: hard block
    if "unknown" in stage2_reason.lower() or "error" in stage2_reason.lower():
        debug_info["reason"] = f"HARD_BLOCK: {stage2_reason}"
        return 0.0, debug_info
    
    # (1) base: 기존 stage2_trade 영향 완화
    base_false = 0.6  # 코드 상수 (나중에 CLI 옵션화 가능)
    if stage2_trade:
        base = 1.0
    else:
        base = base_false
    debug_info["base"] = base
    
    # (2) regime_factor: 추세/레짐
    regime_factor = 1.0
    regime_info = None
    
    # trend_ema가 있으면 연속값으로 계산
    if trend_ema is not None:
        # trend_ema는 가격 대비 상대값으로 정규화 필요 (간단히 절대값 사용)
        # 실제로는 이전 EMA와 비교하거나, 가격 대비 비율을 사용해야 함
        # 여기서는 기본값 유지 (향후 개선 가능)
        regime_info = f"trend_ema={trend_ema:.4f}"
    else:
        # stage2_reason에서 "edge" 포함 여부로 간접 판단
        if "edge" in stage2_reason.lower():
            regime_factor = 1.05  # edge 만족
            regime_info = "edge_ok"
        else:
            regime_factor = 0.95  # edge 미만족
            regime_info = "no_edge"
    
    debug_info["regime_factor"] = regime_factor
    debug_info["regime"] = regime_info
    
    # (3) risk_factor: 위험 패널티 (entropy 기반)
    risk_factor = 1.0
    risk_info = None
    
    if proba_long is not None and proba_short is not None:
        # 현재 신호에 맞는 확률 선택
        if raw_signal == "LONG":
            p = proba_long
        elif raw_signal == "SHORT":
            p = proba_short
        else:
            p = max(proba_long, proba_short)  # fallback
        
        # Binary entropy 계산
        if 0 < p < 1:
            entropy = -p * math.log(p) - (1 - p) * math.log(1 - p)
            # entropy 범위: 0 (확실) ~ 0.693 (불확실)
            # 0.5~0.7 범위를 0~1로 정규화
            risk_norm = max(0.0, min(1.0, (entropy - 0.5) / 0.2))
            risk_factor = 1.0 - 0.3 * risk_norm  # 1.0~0.7
            risk_info = f"entropy={entropy:.4f}"
        else:
            risk_info = "entropy_invalid"
    
    debug_info["risk_factor"] = risk_factor
    debug_info["risk"] = risk_info
    
    # (4) ev_factor: EV/Edge (p_diff 기반)
    ev_factor = 1.0
    ev_value = None
    
    # stage2_reason에서 p_diff 파싱 시도
    import re
    p_diff = None
    
    # "p_diff=0.xxxx" 패턴 찾기
    p_diff_match = re.search(r"p_diff=([\d.-]+)", stage2_reason)
    if p_diff_match:
        try:
            p_diff = float(p_diff_match.group(1))
        except ValueError:
            pass
    
    # 파싱 실패 시 proba_long/proba_short에서 직접 계산
    if p_diff is None and proba_long is not None and proba_short is not None:
        if raw_signal == "LONG":
            p_diff = proba_long - proba_short
        elif raw_signal == "SHORT":
            p_diff = proba_short - proba_long
        else:
            p_diff = abs(proba_long - proba_short)
    
    if p_diff is not None:
        ev_value = p_diff
        ev_temp = 0.005  # 기본값 (수익률 단위)
        # sigmoid: 1 / (1 + exp(-x/t))
        ev_factor = 1.0 / (1.0 + math.exp(-p_diff / ev_temp))
        # ev_factor를 0.5~1.0 범위로 조정 (p_diff=0일 때 0.5, 양수일 때 증가)
        ev_factor = 0.5 + 0.5 * ev_factor
    
    debug_info["ev_factor"] = ev_factor
    debug_info["ev"] = ev_value
    
    # 최종 스코어 계산
    score = base * regime_factor * risk_factor * ev_factor
    score = max(stage2_scale_floor, min(1.0, score))  # clamp
    
    return score, debug_info


def _position_scale_from_proba(
    p: float,
    mode: str,
    p_floor: float,
    p_full: float,
    size_min: float,
    size_max: float,
    p_mid: float = 0.60,
    k: float = 25.0,
) -> float:
    """Proba-based position scale: linear (piecewise) or sigmoid. Returns 0..size_max."""
    if mode == "linear":
        if p < p_floor:
            return 0.0
        denom = p_full - p_floor
        raw = (p - p_floor) / denom if denom > 0 else 1.0
        raw = max(0.0, min(1.0, raw))
        return size_min + (size_max - size_min) * raw
    else:  # sigmoid
        if p < p_floor:
            return 0.0
        x = k * (p - p_mid)
        sig = 1.0 / (1.0 + math.exp(-x)) if x < 500 else 1.0
        return size_min + (size_max - size_min) * sig


def _scale_bin_key(scale: float) -> str:
    if scale <= 0:
        return "[0.0]"
    if scale <= 0.4:
        return "[0.25-0.4]"
    if scale <= 0.6:
        return "[0.4-0.6]"
    if scale <= 0.8:
        return "[0.6-0.8]"
    return "[0.8-1.0]"


# Default commission and slippage rates
DEFAULT_COMMISSION_RATE = getattr(settings, "COMMISSION_RATE", 0.0004)
DEFAULT_SLIPPAGE_RATE = getattr(settings, "SLIPPAGE_RATE", 0.0005)

# Cache directory for prediction probabilities
CACHE_DIR = Path("data/cache/ml_predictions")


class MLBacktestEngine(ABC):
    """
    Abstract base class for ML strategy backtest engines.
    
    Each concrete implementation handles strategy-specific:
    - Probability loading/caching
    - Signal generation from probabilities
    - Trade execution logic
    """
    
    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        feature_preset: str = "extended_safe",
        tcn_preset: str | None = None,
    ):
        """
        Initialize backtest engine.

        Args:
            strategy_name: Strategy identifier (e.g., "ml_xgb", "ml_lstm_attn")
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "5m")
            feature_preset: Feature preset (for XGBoost, ignored for LSTM)
            tcn_preset: For ml_tcn only: "base" or "calendar_e0" to select cache file.
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.feature_preset = feature_preset
        self.tcn_preset = tcn_preset
        self.log_prefix = f"[ML Backtest][{self.get_engine_name()}]"
    
    @abstractmethod
    def get_engine_name(self) -> str:
        """Return engine name for logging (e.g., 'XGBoost', 'LSTM-Attn')."""
        pass
    
    @abstractmethod
    def load_predictions(
        self,
        proba_long_cache: Optional[np.ndarray] = None,
        proba_short_cache: Optional[np.ndarray] = None,
        df_with_proba: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load or compute prediction probabilities.
        
        Args:
            proba_long_cache: Optional pre-computed LONG probabilities
            proba_short_cache: Optional pre-computed SHORT probabilities
            df_with_proba: Optional DataFrame aligned with probabilities
        
        Returns:
            Tuple of (proba_long_arr, proba_short_arr, df_aligned)
        """
        pass
    
    @abstractmethod
    def generate_signals(
        self,
        proba_long_arr: np.ndarray,
        proba_short_arr: np.ndarray,
        df: pd.DataFrame,
        long_threshold: float,
        short_threshold: Optional[float],
        long_only: bool = False,
        short_only: bool = False,
        signal_confirmation_bars: int = 1,
        use_trend_filter: bool = False,
        trend_ema_window: int = 200,
        flat_threshold: Optional[float] = None,
        confidence_margin: float = 0.0,
        min_proba_dominance: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate trading signals from probabilities.
        
        Args:
            proba_long_arr: LONG probabilities array
            proba_short_arr: SHORT probabilities array
            df: DataFrame with OHLCV data
            long_threshold: Threshold for LONG signals
            short_threshold: Threshold for SHORT signals
            long_only: If True, only generate LONG signals
            short_only: If True, only generate SHORT signals
            signal_confirmation_bars: Number of consecutive bars for signal confirmation
            use_trend_filter: Whether to apply EMA trend filter
            trend_ema_window: EMA window for trend filter
        
        Returns:
            DataFrame with 'signal' column added
        """
        pass
    
    def execute_trades(
        self,
        df: pd.DataFrame,
        commission_rate: Optional[float] = None,
        slippage_rate: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        max_holding_bars: Optional[int] = None,
        use_confidence_filter: bool = False,
        confidence_quantile: float = 0.85,
        daily_loss_limit: Optional[float] = None,
        # Anti-overtrading parameters
        enter_long_th: Optional[float] = None,
        exit_long_th: Optional[float] = None,
        enter_short_th: Optional[float] = None,
        exit_short_th: Optional[float] = None,
        min_hold_bars: Optional[int] = None,
        cooldown_bars: Optional[int] = None,
        proba_long_arr: Optional[np.ndarray] = None,
        proba_short_arr: Optional[np.ndarray] = None,
        # Direction filter
        long_only: bool = False,
        short_only: bool = False,
        # StrategyGuard
        use_strategy_guard: bool = False,
        # StrategyGuard Phase-2 options
        strategy_guard_min_win_rate: float | None = None,
        strategy_guard_min_avg_return: float | None = None,
        strategy_guard_unblock_win_rate: float | None = None,
        strategy_guard_unblock_avg_return: float | None = None,
        strategy_guard_min_block_trades: int | None = None,
        strategy_guard_recent_trades_window: int | None = None,
        strategy_guard_insufficient_sample_policy: str | None = None,
        # StrategyGuard v2 options
        use_strategy_guard_v2: bool = False,
        strategy_guard_v2_mode: str | None = None,
        strategy_guard_v2_window_signal_stats: int | None = None,
        strategy_guard_v2_min_margin: float | None = None,
        strategy_guard_v2_max_entropy: float | None = None,
        strategy_guard_v2_scale_floor: float | None = None,
        strategy_guard_v2_block_if_scale_below: float | None = None,
        # SHORT Strategy MVP
        enable_short_strategy: bool = False,
        # Trade dump
        dump_trades_path: str | None = None,
        # Stage-2 (for dump accuracy)
        use_stage2: bool = False,
        # Stage-2 v2.2 options
        stage2_block_if_final_scale_below: float = 0.0,
        # Stage-2 CAP 임계값 (스윕용)
        stage2_cap_entropy_high_th: float = 0.64,  # 완화: 0.66 → 0.64
        stage2_cap_entropy_mid_th: float = 0.62,   # 완화: 0.64 → 0.62
        stage2_cap_pdiff_tiny_th: float = 0.002,
        stage2_cap_pdiff_small_th: float = 0.005,  # 기본값 승격: pdiff_small_005 (기존 0.004)
        # Entry filters (minimal invasive: optional)
        min_max_proba: Optional[float] = None,
        max_entropy: Optional[float] = None,
        min_proba_gap: float = 0.0,
        # Phase D11: directional edge + flat suppression (optional, default off)
        min_directional_gap: Optional[float] = None,
        max_flat_entry_proba: Optional[float] = None,
        min_hold_bars_override: Optional[int] = None,
        cooldown_bars_override: Optional[int] = None,
        # D12: when False (directional_edge pure), skip min_max_proba / max_entropy entry gates
        use_legacy_entry_gates: bool = True,
        # Regime filter (EMA200: Long entry only when close > ema200; default OFF)
        regime_filter_enabled: bool = False,
        regime_ema_span: int = 200,
        regime_rule: str = "ema_only",  # "ema_only" | ... | "downtrend_block" | "proba_quantile"
        regime_slope_lookback: int = 48,
        vol_window: int = 48,
        vol_threshold: float = 0.0010,
        breakout_lookback: int = 96,
        breakout_mode: str = "ema_high",  # "ema" | "ema_high"
        slope_threshold: float = 1e-5,  # flat_slope: skip entry when abs(ema_slope) <= this
        q_window: int = 8640,  # proba_quantile: rolling window (bars), default 30d @ 5m
        q: float = 0.90,  # proba_quantile: quantile threshold (e.g. 0.90 = top 10%)
        p_floor: float = 0.55,  # proba_quantile: minimum p gate
        # Position scaling (proba-based size): default OFF
        position_scaling_enabled: bool = False,
        position_scaling_mode: str = "linear",  # "linear" | "sigmoid"
        position_p_floor: float = 0.55,
        position_p_full: float = 0.65,
        position_size_min: float = 0.25,
        position_size_max: float = 1.0,
        position_p_mid: float = 0.60,  # sigmoid
        position_k: float = 25.0,  # sigmoid steepness
        # Early exit (proba-based): default OFF
        early_exit_enabled: bool = False,
        early_exit_lookback: int = 12,
        early_exit_p_floor: float = 0.55,
        early_exit_bad_k: int = 8,
        # Entry flat-proba gate (Phase C): skip entry when proba_flat > max_flat_proba; default OFF
        entry_flat_gate_enabled: bool = False,
        max_flat_proba: float = 0.45,
        # Flat-aware early exit (Phase C3): when proba_flat > threshold, use lower effective_bad_k; default OFF
        flat_exit_aware_enabled: bool = False,
        flat_exit_threshold: float = 0.30,
        flat_exit_badk_delta: int = 2,
        # Time-stop (Phase C4): force exit when holding_bars >= time_stop_bars; default OFF
        time_stop_enabled: bool = False,
        time_stop_bars: int = 96,
        # Partial take-profit (Phase C7): lock part of profit at threshold; default OFF
        partial_tp_enabled: bool = False,
        partial_tp_threshold: float = 0.0025,
        partial_tp_ratio: float = 0.5,
        # Partial TP fee correction (Phase C8 audit): if True, use correct fee on partial close/remaining exit; default False for backward compat
        partial_tp_fee_correct: bool = False,
        # Break-even stop (Phase C9): when unrealized return >= be_threshold, move stop to entry; default OFF
        break_even_stop_enabled: bool = False,
        be_threshold: float = 0.0,
        # Trade log (candidate_validation sanity check): when True, result will include trade_events for CSV export
        emit_trade_log: bool = False,
        # signal_exit_th only (does not touch early_exit / time_stop / SHORT logic):
        # - trailing_window_bars=1: 기존과 동일 (현재 바 proba만 사용).
        # - trailing_window_bars>1: 최근 W바 구간의 min(방향 proba) < exit_*_th 이면 청산 (min_hold 이후 약세 기억 청산).
        signal_exit_th_trailing_window_bars: int = 1,
        # 진입 시점 방향 proba 대비 추가 하락폭(Δ < -x)이면 signal_exit_th (EXP_B)
        signal_exit_th_delta_from_entry: Optional[float] = None,
        # 손실형 선별 보조: 임계 기반 signal_exit_th 에서만 직전 바 close 로 청산가 조정 (δ_p, MAE_cut)
        signal_exit_th_loss_aux_delta_p: Optional[float] = None,
        signal_exit_th_loss_aux_mae_cut: Optional[float] = None,
        signal_exit_th_loss_aux_short_only: bool = False,
        signal_exit_th_loss_aux_min_bars: Optional[int] = None,
        # 손실 보조 보호 게이트 (선택): 직전 바 close 조정을 건너뜀
        signal_exit_th_loss_aux_gate_delta_unreal_positive: bool = False,
        signal_exit_th_loss_aux_gate_mfe_min: Optional[float] = None,
        # Adaptive ENTRY: require state_log unique_round_trips >= N; None = off
        entry_min_unique_round_trips: Optional[int] = None,
        # CSV path for load_latest_window_stats; None -> data/diagnostics/fr2/state_log.csv
        entry_urt_state_log_path: Optional[Path] = None,
        # URT from load_latest_window_stats taken before state_log append (see LstmAttn run_backtest)
        entry_urt_state_log_snapshot: Optional[int] = None,
        # When set (e.g. compare script), overrides default "pre_append_snapshot" in result metadata
        entry_urt_gate_value_source_override: Optional[str] = None,
        # Profit lock exit (optional; default OFF — 동일 파라미터면 기존 결과 유지)
        enable_profit_lock_exit: bool = False,
        profit_lock_take_profit: float = 0.0015,
        profit_lock_stop_loss: float = -0.0030,
        profit_lock_max_bars: int = 12,
        profit_lock_conservative_intrabar: bool = True,
        # Overtrading / re-entry control (optional; default OFF)
        enable_reentry_cooldown: bool = False,
        reentry_cooldown_bars: int = 12,
        enable_directional_cooldown: bool = False,
        directional_cooldown_bars: int = 12,
        enable_post_tp_block: bool = False,
        post_tp_block_bars: int = 6,
        enable_post_sl_block: bool = False,
        post_sl_block_bars: int = 12,
        # Regime ablation (diagnostics): entry-only skip; default OFF
        enable_regime_filter: bool = False,
        regime_filter_mode: str = "none",
        regime_combo_block_keys: tuple[str, ...] | None = None,
        # Positive regime (allow-list): block when bar not in allowed regime; default OFF
        enable_positive_regime_filter: bool = False,
        positive_regime_filter_mode: str = "none",
        # Adaptive positive regime: gate ON일 때만 positive filter 적용; default OFF
        enable_adaptive_positive_regime_filter: bool = False,
        adaptive_positive_regime_mode: str = "none",
        enable_confidence_filter: bool = False,
        confidence_filter_mode: str = "none",
    ) -> BacktestResult:
        """
        Execute trades based on signals in DataFrame.
        
        This is a generic implementation that works for both XGBoost and LSTM.
        Strategy-specific position management can be overridden if needed.
        
        Args:
            df: DataFrame with 'signal' column
            commission_rate: Override commission rate
            slippage_rate: Override slippage rate
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            max_holding_bars: Maximum bars to hold a position
            use_confidence_filter: Whether to use confidence filter
            confidence_quantile: Confidence quantile threshold
            daily_loss_limit: Daily loss limit (kill switch)
            dump_trades_path: Path to dump trade events CSV (None to disable)
        
        Returns:
            BacktestResult
        """
        from src.backtest.engine import _compute_trade_stats
        from src.backtest.strategy_guard import (
            StrategyGuard,
            StrategyGuardConfig,
            StrategyGuardV2,
            StrategyGuardV2Config,
        )
        
        # Initialize StrategyGuard if enabled
        guard: StrategyGuard | None = None
        guard_v2: StrategyGuardV2 | None = None
        
        if use_strategy_guard_v2:
            # Guard v2 초기화
            logger.info(f"{self.log_prefix}[StrategyGuardV2] Initializing Guard v2...")
            config_v2 = StrategyGuardV2Config()
            config_v2.enable_v2 = True
            if strategy_guard_v2_mode is not None:
                if strategy_guard_v2_mode not in ["soft", "hard"]:
                    raise ValueError(f"Invalid strategy_guard_v2_mode: {strategy_guard_v2_mode}")
                config_v2.mode = strategy_guard_v2_mode
            if strategy_guard_v2_window_signal_stats is not None:
                config_v2.window_signal_stats = strategy_guard_v2_window_signal_stats
            if strategy_guard_v2_min_margin is not None:
                config_v2.min_margin = strategy_guard_v2_min_margin
            if strategy_guard_v2_max_entropy is not None:
                config_v2.max_entropy = strategy_guard_v2_max_entropy
            if strategy_guard_v2_scale_floor is not None:
                config_v2.scale_floor = strategy_guard_v2_scale_floor
            if strategy_guard_v2_block_if_scale_below is not None:
                config_v2.block_if_scale_below = strategy_guard_v2_block_if_scale_below
            
            guard_v2 = StrategyGuardV2(config=config_v2)
            logger.info(
                f"{self.log_prefix}[StrategyGuardV2] ✓ Guard v2 instance created. "
                f"Config: mode={config_v2.mode}, window_signal_stats={config_v2.window_signal_stats}, "
                f"min_margin={config_v2.min_margin}, max_entropy={config_v2.max_entropy}, "
                f"scale_floor={config_v2.scale_floor}, block_if_scale_below={config_v2.block_if_scale_below}"
            )
        
        if use_strategy_guard:
            # Phase-2: CLI 옵션으로 설정 오버라이드
            config = StrategyGuardConfig()
            # BLOCK 조건 오버라이드
            if strategy_guard_min_win_rate is not None:
                config.min_win_rate = strategy_guard_min_win_rate
            if strategy_guard_min_avg_return is not None:
                config.min_avg_return = strategy_guard_min_avg_return
            # UNBLOCK 조건 오버라이드
            if strategy_guard_unblock_win_rate is not None:
                config.unblock_win_rate = strategy_guard_unblock_win_rate
            if strategy_guard_unblock_avg_return is not None:
                config.unblock_avg_return = strategy_guard_unblock_avg_return
            # 히스테리시스 오버라이드
            if strategy_guard_min_block_trades is not None:
                config.min_block_trades = strategy_guard_min_block_trades
            # recent_trades_window 오버라이드
            if strategy_guard_recent_trades_window is not None:
                config.recent_trades_window = strategy_guard_recent_trades_window
            # insufficient_sample_policy 오버라이드
            if strategy_guard_insufficient_sample_policy is not None:
                if strategy_guard_insufficient_sample_policy not in ["allow", "block", "defer"]:
                    raise ValueError(f"Invalid insufficient_sample_policy: {strategy_guard_insufficient_sample_policy}")
                config.insufficient_sample_policy = strategy_guard_insufficient_sample_policy
            
            guard = StrategyGuard(config=config)
            logger.info(
                f"{self.log_prefix}[StrategyGuard] Enabled (Phase-2: UNBLOCK + 히스테리시스). "
                f"Config: recent_trades_window={guard.config.recent_trades_window}, "
                f"insufficient_sample_policy={guard.config.insufficient_sample_policy}, "
                f"BLOCK: min_win_rate={guard.config.min_win_rate}, min_avg_return={guard.config.min_avg_return}, "
                f"UNBLOCK: unblock_win_rate={guard.config.unblock_win_rate}, unblock_avg_return={guard.config.unblock_avg_return}, "
                f"min_block_trades={guard.config.min_block_trades}"
            )
        
        # SHORT 전략 활성화 로그
        if enable_short_strategy:
            logger.info(
                f"{self.log_prefix}[SHORT STRATEGY] Enabled: SHORT 전략이 활성화되었습니다. "
                f"SHORT 진입 조건: signal=SHORT, Stage-2 통과, StrategyGuard=ALLOW"
            )
        else:
            logger.info(
                f"{self.log_prefix}[SHORT STRATEGY] Disabled: SHORT 전략이 비활성화되어 있습니다. "
                f"(기본 동작: SHORT 신호는 기존 로직대로 처리)"
            )
        
        trades: list[Trade] = []
        equity_curve: list[float] = []
        position: dict | None = None
        balance = 1.0
        entries_attempted = 0
        entries_blocked_by_regime_ablation = 0
        entries_blocked_by_positive_regime = 0
        positive_regime_allowed_entries = 0
        positive_regime_blocked_entries = 0
        adaptive_positive_gate_on_count = 0
        entries_blocked_by_adaptive_positive = 0
        adaptive_positive_allowed_entries = 0
        adaptive_positive_blocked_entries = 0
        entries_blocked_by_confidence = 0
        exits_executed = 0
        tp_exits = 0
        sl_exits = 0
        profit_lock_tp_exits = 0
        profit_lock_sl_exits = 0

        # Additional tracking
        position_entry_bar_index: int | None = None
        daily_balance_start: float = 1.0
        current_date: str | None = None
        trading_disabled: bool = False
        
        # Confidence filter tracking
        confidence_proba_history: list[float] = []
        confidence_window = 100
        
        effective_commission_rate = commission_rate if commission_rate is not None else DEFAULT_COMMISSION_RATE
        effective_slippage_rate = slippage_rate if slippage_rate is not None else DEFAULT_SLIPPAGE_RATE
        effective_min_hold_bars = min_hold_bars_override if min_hold_bars_override is not None else min_hold_bars
        effective_cooldown_bars = cooldown_bars_override if cooldown_bars_override is not None else cooldown_bars

        # Track trade events for logging
        trade_events: list[dict] = []  # Entry/Exit/Flip events
        signal_exit_th_trade_count = 0
        signal_exit_th_scaled_profit_sum = 0.0
        signal_exit_th_loss_aux_applied = 0
        
        # Track first 10 trades for detailed debugging
        trade_count = 0
        MAX_DEBUG_TRADES = 10
        
        # Track first 20 entry attempts for signal/direction mapping verification
        entry_sample_count = 0
        MAX_ENTRY_SAMPLES = 20
        
        # ======================================================================
        # Anti-overtrading state tracking
        # ======================================================================
        last_exit_bar_index: int | None = None  # For cooldown tracking
        last_exit_direction: str | None = None
        last_exit_reason: str | None = None
        entry_count_long = 0
        entry_count_short = 0
        exit_count = 0
        flip_count = 0  # LONG->SHORT or SHORT->LONG transitions
        
        # Block reason tracking (for logging)
        block_reasons: dict[str, int] = {
            "strategy_guard": 0,  # StrategyGuard에서 BLOCK
            "direction_filter": 0,
            "min_hold": 0,
            "cooldown": 0,
            "confirmation": 0,
            "margin_zone": 0,
            "hysteresis": 0,
            "stage2_no_trade": 0,  # Stage-2에서 Trade=False로 차단
            "regime_ablation": 0,
            "positive_regime": 0,
        }
        skipped_by_min_max_proba = 0
        skipped_by_max_entropy = 0
        skipped_by_proba_gap = 0
        spread_rejected_list: list[float] = []
        spread_executed_list: list[float] = []
        # Phase D11: directional gap + flat suppression fail counts
        directional_gap_fail_count = 0
        long_directional_gap_fail_count = 0
        short_directional_gap_fail_count = 0
        flat_suppression_fail_count = 0

        # Stage-2 statistics
        stage2_no_trade_count = 0
        stage2_trade_count = 0
        stage2_exit_on_flat_count = 0
        
        logger.debug(f"{self.log_prefix} Starting trade execution loop: {len(df)} rows to process")

        # ======================================================================
        # [FILTER DISTRIBUTION] max_proba / entropy 비율 (필터 원인 분석용)
        # ======================================================================
        if proba_long_arr is not None and proba_short_arr is not None and len(proba_long_arr) > 0:
            n_bars = min(len(proba_long_arr), len(proba_short_arr), len(df))
            cnt_ge_045 = cnt_ge_050 = cnt_ge_055 = cnt_ge_060 = 0
            cnt_le_160 = cnt_le_155 = cnt_le_150 = cnt_le_145 = 0
            for idx in range(n_bars):
                pl = float(proba_long_arr[idx]) if idx < len(proba_long_arr) else 0.0
                ps = float(proba_short_arr[idx]) if idx < len(proba_short_arr) else 0.0
                pf = max(0.0, min(1.0, 1.0 - pl - ps))
                mp = max(pl, ps, pf)
                h = 0.0
                for p in (pl, ps, pf):
                    if p > 1e-12:
                        h -= p * math.log2(p)
                if mp >= 0.45:
                    cnt_ge_045 += 1
                if mp >= 0.50:
                    cnt_ge_050 += 1
                if mp >= 0.55:
                    cnt_ge_055 += 1
                if mp >= 0.60:
                    cnt_ge_060 += 1
                if h <= 1.60:
                    cnt_le_160 += 1
                if h <= 1.55:
                    cnt_le_155 += 1
                if h <= 1.50:
                    cnt_le_150 += 1
                if h <= 1.45:
                    cnt_le_145 += 1
            ratio_ge_045 = cnt_ge_045 / n_bars if n_bars else 0
            ratio_ge_050 = cnt_ge_050 / n_bars if n_bars else 0
            ratio_ge_055 = cnt_ge_055 / n_bars if n_bars else 0
            ratio_ge_060 = cnt_ge_060 / n_bars if n_bars else 0
            ratio_le_160 = cnt_le_160 / n_bars if n_bars else 0
            ratio_le_155 = cnt_le_155 / n_bars if n_bars else 0
            ratio_le_150 = cnt_le_150 / n_bars if n_bars else 0
            ratio_le_145 = cnt_le_145 / n_bars if n_bars else 0
            logger.info(
                f"{self.log_prefix}[FILTER DISTRIBUTION] n_bars={n_bars} "
                f"ratio_max_proba_ge_045={ratio_ge_045:.4f} ge_050={ratio_ge_050:.4f} "
                f"ge_055={ratio_ge_055:.4f} ge_060={ratio_ge_060:.4f} | "
                f"ratio_entropy_le_160={ratio_le_160:.4f} le_155={ratio_le_155:.4f} "
                f"le_150={ratio_le_150:.4f} le_145={ratio_le_145:.4f}"
            )

        # ======================================================================
        # [DEBUG] Signal 컬럼명 및 값 매핑 확인 (처음 200 bars)
        # ======================================================================
        if len(df) > 0:
            signal_col = "signal"
            if signal_col in df.columns:
                unique_signals = sorted(df[signal_col].unique())
                logger.info(
                    f"{self.log_prefix}[SIGNAL DEBUG] Signal column: '{signal_col}', "
                    f"unique values: {unique_signals}"
                )
                
                # 처음 200 bars에서 signal != FLAT/HOLD인 샘플 확인
                debug_df = df.head(200)
                non_flat_mask = ~debug_df[signal_col].isin(["FLAT", "HOLD"])
                non_flat_count = non_flat_mask.sum()
                if non_flat_count > 0:
                    non_flat_samples = debug_df[non_flat_mask][signal_col].head(10).tolist()
                    logger.info(
                        f"{self.log_prefix}[SIGNAL DEBUG] First 200 bars: "
                        f"non-FLAT/HOLD signals={non_flat_count}, "
                        f"samples={non_flat_samples}"
                    )
                else:
                    logger.info(
                        f"{self.log_prefix}[SIGNAL DEBUG] First 200 bars: "
                        f"all signals are FLAT/HOLD"
                    )
            else:
                logger.warning(
                    f"{self.log_prefix}[SIGNAL DEBUG] Signal column '{signal_col}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )
        
        # Track guard decision for each row (for dump)
        guard_decision_by_idx: dict[int, str] = {}
        
        # ======================================================================
        # [실전 모니터링] 모니터링 로거 초기화 (shadow mode, logging-only)
        # ======================================================================
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        monitor = MonitoringLogger(run_id=run_id, mode="backtest")
        
        # ======================================================================
        # [정밀 분석] 통계 수집 (요청 1-3)
        # ======================================================================
        final_scale_at_entry: list[float] = []  # ENTRY 시점 final_scale 수집
        stage2_cap_at_entry: list[float] = []  # ENTRY 시점 stage2_cap 수집 (CAP 분포용)
        stage2_hard_block_count = 0  # cap == 0.0으로 ENTRY 차단
        stage2_soft_gated_count = 0  # cap < 1.0 이지만 ENTRY 허용
        stage2_score_samples: list[dict] = []  # ENTRY 시점 샘플 (최대 10개)
        stage2_cap_applied_count = 0  # v2.2: cap < 1.0 인 entry 수
        stage2_entry_blocked_by_scale_count = 0  # v2.2: final_scale 기반 ENTRY 차단
        stage2_cap_sample_count = 0  # [STAGE2][CAP] 로그 샘플 카운터 (최대 20개)
        cap_breakdown_records: list[dict] = []  # CAP 값별 성적 분해용 (ENTRY 기준)
        
        # [Guard v2 관측] 통계 수집
        guard_scale_at_entry: list[float] = []  # ENTRY 시점 guard_scale 수집
        guard_scale_all_checks: list[float] = []  # 전체 checks 기준 guard_scale 수집 (분포용)
        guard_entry_exit_links: list[dict] = []  # ENTRY/EXIT 연결 (guard_scale 성과 분석용)
        guard_scale_sample_count = 0  # [GUARD][SCALE] 로그 샘플 카운터 (최대 100개)
        
        # [DEBUG] CAP 로그 기반 확정용 집계
        cap_debug_logs: list[dict] = []  # 상세 CAP 로그 (30줄 이상)
        cap_entry_exit_links: list[dict] = []  # ENTRY/EXIT 연결 (20개 이상)
        cap_trigger_stats = {
            "total_checks": 0,
            "entries_attempted": 0,
            "entries_executed": 0,
            "cap_1_0_count": 0,
            "cap_0_8_count": 0,
            "cap_0_6_count": 0,
            "triggered_by_entropy_only": 0,
            "triggered_by_pdiff_only": 0,
            "triggered_by_both": 0,
            "triggered_by_none": 0,
        }
        next_trade_id = 1  # trade_id 생성용
        
        # Overtrading 방지 집계
        blocked_by_min_hold = 0
        blocked_by_cooldown = 0
        blocked_by_guard_hard = 0
        blocked_by_ot_global = 0
        blocked_by_ot_directional = 0
        blocked_by_ot_post_tp = 0
        blocked_by_ot_post_sl = 0
        
        # Regime filter: Long entry gate only (EMA or Vol Compression)
        entries_blocked_by_regime = 0
        entries_blocked_by_regime_price = 0
        entries_blocked_by_regime_slope = 0
        entries_blocked_by_regime_vol = 0
        entries_blocked_by_regime_vol_slope = 0
        entries_blocked_by_regime_vol_breakout = 0
        entries_allowed_on_decompress = 0
        entries_blocked_by_regime_flat_slope = 0
        entries_blocked_by_regime_downtrend = 0
        entries_blocked_by_regime_proba_quantile = 0
        entries_blocked_by_flat_gate = 0
        entries_blocked_by_urt_gate = 0
        urt_gate_block_samples: list[dict[str, int]] = []
        # Entry URT: prefer snapshot (CSV before append_state_log in run_backtest); else read file here.
        entry_urt_gate_active = False
        entry_urt_from_state_log: int | None = None
        entry_urt_state_log_resolved: Path | None = None
        entry_urt_gate_value_source: str | None = None
        if entry_min_unique_round_trips is not None:
            entry_urt_state_log_resolved = (
                Path(entry_urt_state_log_path)
                if entry_urt_state_log_path is not None
                else Path("data/diagnostics/fr2/state_log.csv")
            )
            # Only snapshot from run_backtest (CSV before append_state_log). Re-reading the file here
            # would see runtime_activation_probe rows appended immediately before execute_trades.
            if entry_urt_state_log_snapshot is not None:
                entry_urt_from_state_log = int(entry_urt_state_log_snapshot)
                entry_urt_gate_active = True
                entry_urt_gate_value_source = (
                    entry_urt_gate_value_source_override
                    if entry_urt_gate_value_source_override is not None
                    else "pre_append_snapshot"
                )
        # Position scaling (proba-based) stats
        entries_scaled_count = 0
        entries_scaled_applied_count = 0  # 실제로 scale 적용된 진입 수 (skip 제외)
        scale_sum = 0.0
        scale_sq_sum = 0.0
        scale_min_val = 1.0
        scale_max_val = 0.0
        scale_bins_counts: dict[str, int] = {"[0.0]": 0, "[0.25-0.4]": 0, "[0.4-0.6]": 0, "[0.6-0.8]": 0, "[0.8-1.0]": 0}
        early_exit_count = 0
        total_exit_evaluations = 0  # Phase C3: bars where we evaluated early exit
        flat_exit_adjusted_count = 0  # Phase C3: evaluations where we used reduced effective_bad_k
        flat_exit_trigger_count = 0  # Phase C3: early exits that used reduced effective_bad_k
        time_stop_exit_count = 0  # Phase C4: exits due to time_stop
        partial_tp_count = 0  # Phase C7: number of trades that had a partial TP
        partial_tp_pnl_list: list[float] = []  # Phase C7: realized return from each partial TP
        break_even_exit_count = 0  # Phase C9: exits due to break-even stop
        ema_series: Optional[pd.Series] = None
        ema_slope_series: Optional[pd.Series] = None
        vol_series: Optional[pd.Series] = None
        compress_series: Optional[pd.Series] = None
        decompress_event_series: Optional[pd.Series] = None
        rolling_high_series: Optional[pd.Series] = None
        q_series: Optional[pd.Series] = None
        if regime_filter_enabled and "close" in df.columns:
            if regime_rule in ("ema_only", "ema_plus_slope"):
                from src.strategy_filters.regime_filter import compute_ema, compute_ema_slope
                ema_series = compute_ema(df["close"], span=regime_ema_span)
                if regime_rule == "ema_plus_slope":
                    ema_slope_series = compute_ema_slope(ema_series, lookback=regime_slope_lookback)
            elif regime_rule == "vol_compress":
                from src.strategy_filters.regime_filter import compute_vol_compress
                vol_series = compute_vol_compress(df["close"], window=vol_window)
            elif regime_rule == "vol_slope":
                from src.strategy_filters.regime_filter import compute_ema, compute_ema_slope, compute_vol_compress
                ema_series = compute_ema(df["close"], span=regime_ema_span)
                ema_slope_series = compute_ema_slope(ema_series, lookback=regime_slope_lookback)
                vol_series = compute_vol_compress(df["close"], window=vol_window)
            elif regime_rule == "vol_breakout":
                from src.strategy_filters.regime_filter import compute_ema, compute_vol_compress
                vol_series = compute_vol_compress(df["close"], window=vol_window)
                compress_series = vol_series < vol_threshold
                decompress_event_series = (compress_series.shift(1) == True) & (compress_series == False)
                ema_series = compute_ema(df["close"], span=regime_ema_span)
                rolling_high_series = df["close"].rolling(breakout_lookback).max().shift(1)
            elif regime_rule == "flat_slope":
                from src.strategy_filters.regime_filter import compute_ema, compute_ema_slope
                ema_series = compute_ema(df["close"], span=regime_ema_span)
                ema_slope_series = compute_ema_slope(ema_series, lookback=regime_slope_lookback)
            elif regime_rule == "downtrend_block":
                from src.strategy_filters.regime_filter import compute_ema, compute_ema_slope
                ema_series = compute_ema(df["close"], span=regime_ema_span)
                ema_slope_series = compute_ema_slope(ema_series, lookback=regime_slope_lookback)
            elif regime_rule == "proba_quantile" and proba_long_arr is not None and len(proba_long_arr) > 0:
                # Rolling quantile of p (no lookahead: shift(1) then rolling)
                n_align = min(len(df), len(proba_long_arr))
                p_series_proba = pd.Series(proba_long_arr[:n_align], index=range(n_align))
                min_periods = max(50, q_window // 10)
                q_series = p_series_proba.shift(1).rolling(q_window, min_periods=min_periods).quantile(q)

        regime_ablation_block_mask: np.ndarray | None = None
        if enable_regime_filter and regime_filter_mode not in (None, "", "none"):
            from src.strategy_filters.regime_ablation import compute_regime_ablation_block_mask

            regime_ablation_block_mask = compute_regime_ablation_block_mask(
                df,
                regime_filter_mode,
                regime_combo_block_keys=regime_combo_block_keys,
            )

        positive_regime_block_mask: np.ndarray | None = None
        positive_regime_key_arr: np.ndarray | None = None
        use_static_positive = enable_positive_regime_filter and not enable_adaptive_positive_regime_filter
        if use_static_positive and positive_regime_filter_mode not in (None, "", "none"):
            from src.strategy_filters.regime_ablation import (
                compute_positive_regime_allow_mask,
                compute_regime_components,
            )

            *_, positive_regime_key_arr = compute_regime_components(df)
            positive_regime_block_mask = ~compute_positive_regime_allow_mask(
                df, positive_regime_filter_mode
            )

        adaptive_gate_on_mask: np.ndarray | None = None
        adaptive_positive_final_block: np.ndarray | None = None
        adaptive_regime_key_arr: np.ndarray | None = None
        if enable_adaptive_positive_regime_filter and adaptive_positive_regime_mode not in (None, "", "none"):
            from src.strategy_filters.regime_ablation import (
                compute_adaptive_positive_block_mask,
                compute_regime_components as _arc,
            )

            *_, adaptive_regime_key_arr = _arc(df)
            adaptive_gate_on_mask, _, adaptive_positive_final_block = compute_adaptive_positive_block_mask(
                df, adaptive_positive_regime_mode
            )

        for i, row in enumerate(df.itertuples()):
            signal: Signal = getattr(row, "signal")
            current_price = float(getattr(row, "close"))
            current_high = float(getattr(row, "high")) if hasattr(row, "high") else current_price
            current_low = float(getattr(row, "low")) if hasattr(row, "low") else current_price
            row_timestamp = getattr(row, "timestamp")
            
            # Stage-2 정보 추출
            raw_signal = getattr(row, "raw_signal", signal) if hasattr(row, "raw_signal") else signal
            # Stage-2 OFF일 때는 빈값으로, ON일 때만 실제 값 사용
            if use_stage2:
                stage2_trade = getattr(row, "stage2_trade", True) if hasattr(row, "stage2_trade") else True
                stage2_reason = getattr(row, "stage2_reason", "") if hasattr(row, "stage2_reason") else ""
            else:
                # Stage-2 OFF: dump에 빈값 기록
                stage2_trade = None
                stage2_reason = ""
            
            # ======================================================================
            # [STRATEGY GUARD] 실행 허용/차단 판정 (Phase-2: UNBLOCK + 히스테리시스)
            # ======================================================================
            current_guard_decision = ""
            if guard is not None:
                # 신호 업데이트 (Stage-2 통과 여부)
                guard.update_signal(stage2_trade=stage2_trade, timestamp=str(row_timestamp))
                
                # Guard 판정 (trade_index 전달)
                guard_decision = guard.check(trade_index=i, timestamp=str(row_timestamp))
                current_guard_decision = guard_decision
                guard_decision_by_idx[i] = guard_decision
                
                if guard_decision == "BLOCK":
                    # BLOCK 상태: 모든 signal을 HOLD로 처리
                    signal = "HOLD"
                    block_reasons["strategy_guard"] = block_reasons.get("strategy_guard", 0) + 1
                    # HOLD 신호는 포지션 유지 (아무 행동 안 함)
                    continue
                elif guard_decision == "COOLING":
                    # COOLING 상태: BLOCK과 동일하게 처리 (향후 확장 가능)
                    signal = "HOLD"
                    block_reasons["strategy_guard"] = block_reasons.get("strategy_guard", 0) + 1
                    continue
                elif guard_decision == "DEFER":
                    # DEFER 상태: 판단 스킵, signal은 그대로 유지하되 Guard는 개입하지 않음
                    # (기존 로직대로 진행)
                    pass
            else:
                guard_decision_by_idx[i] = ""
            
            # ======================================================================
            # [SHORT STRATEGY MVP] SHORT 전략 활성 조건 체크
            # ======================================================================
            if enable_short_strategy and signal == "SHORT":
                # SHORT 전략 활성 조건 체크
                short_strategy_allowed = True
                short_block_reason = ""
                
                # 조건 1: StrategyGuard 상태 == ALLOW
                if guard is not None:
                    guard_decision = guard.state.current_decision if hasattr(guard, 'state') else "ALLOW"
                    if guard_decision == "BLOCK":
                        short_strategy_allowed = False
                        short_block_reason = "strategy_guard_block"
                        block_reasons["short_strategy_guard"] = block_reasons.get("short_strategy_guard", 0) + 1
                
                # 조건 2: Stage-2 통과 (이미 위에서 체크됨, stage2_trade == True)
                if short_strategy_allowed and not stage2_trade:
                    short_strategy_allowed = False
                    short_block_reason = "stage2_no_trade"
                    block_reasons["short_strategy_stage2"] = block_reasons.get("short_strategy_stage2", 0) + 1
                
                # 조건 3: p_short edge >= threshold (기존 threshold 재사용)
                # 주의: 이 조건은 이미 Stage-2에서 체크되므로 여기서는 추가 체크 불필요
                # (Stage-2가 통과했다면 p_short edge는 이미 만족)
                
                if not short_strategy_allowed:
                    # SHORT 전략 비활성: signal을 HOLD로 변환
                    signal = "HOLD"
                    logger.debug(
                        f"{self.log_prefix}[SHORT STRATEGY] SHORT signal blocked: {short_block_reason}"
                    )
                else:
                    logger.debug(
                        f"{self.log_prefix}[SHORT STRATEGY] SHORT signal allowed (stage2_trade={stage2_trade}, guard={guard_decision if guard is not None else 'N/A'})"
                    )
            
            # ======================================================================
            # [DIRECTION FILTER] long_only/short_only 필터링 (안전장치)
            # ======================================================================
            if long_only and signal == "SHORT":
                signal = "HOLD"  # SHORT 신호를 HOLD로 변환
                block_reasons["direction_filter"] = block_reasons.get("direction_filter", 0) + 1
            elif short_only and signal == "LONG":
                signal = "HOLD"  # LONG 신호를 HOLD로 변환
                block_reasons["direction_filter"] = block_reasons.get("direction_filter", 0) + 1
            
            # ======================================================================
            # [STAGE-2 v2.2] 조건부 캡 방식 (곱셈 패널티 해결)
            # ======================================================================
            stage2_cap = 1.0  # 기본값 (Stage-2 OFF일 때)
            stage2_cap_debug_info = {}  # v2.2 cap debug 정보
            if use_stage2:
                # 추가 정보 추출 (optional)
                proba_long_val = None
                proba_short_val = None
                trend_ema_val = None
                
                if proba_long_arr is not None and i < len(proba_long_arr):
                    proba_long_val = proba_long_arr[i]
                if proba_short_arr is not None and i < len(proba_short_arr):
                    proba_short_val = proba_short_arr[i]
                if hasattr(row, "trend_ema"):
                    try:
                        trend_ema_val = float(getattr(row, "trend_ema"))
                    except (ValueError, TypeError):
                        pass
                
                # v2.2: cap 방식으로 변경
                stage2_cap, stage2_cap_debug_info = calculate_stage2_cap(
                    stage2_reason=stage2_reason,
                    proba_long=proba_long_val,
                    proba_short=proba_short_val,
                    raw_signal=raw_signal,
                )
                
                # Hard block 조건: 극단적 위험/결측만 차단 (v2.2에서는 cap==0.0이면 hard block)
                if stage2_cap == 0.0:
                    stage2_no_trade_count += 1
                    block_reasons["stage2_no_trade"] = block_reasons.get("stage2_no_trade", 0) + 1
                    if signal == "HOLD":
                        continue
                    signal = "HOLD"
                    continue
                else:
                    stage2_trade_count += 1
                    if stage2_reason == "exit_on_flat":
                        stage2_exit_on_flat_count += 1
            
            # Daily loss limit (kill switch)
            if daily_loss_limit is not None:
                row_date = str(row_timestamp).split(" ")[0] if isinstance(row_timestamp, str) else str(row_timestamp).split("T")[0]
                if current_date is None:
                    current_date = row_date
                    daily_balance_start = balance
                elif row_date != current_date:
                    current_date = row_date
                    daily_balance_start = balance
                    trading_disabled = False
                
                daily_return = (balance - daily_balance_start) / daily_balance_start
                if daily_return <= -daily_loss_limit:
                    trading_disabled = True
                    if position is not None:
                        # Force close position
                        exit_price = current_price
                        entry_price = position["entry_price"]
                        direction = position["side"]
                        
                        entry_cost = entry_price * (effective_commission_rate + effective_slippage_rate)
                        exit_cost = exit_price * (effective_commission_rate + effective_slippage_rate)
                        
                        if direction == "LONG":
                            effective_entry = entry_price + entry_cost
                            effective_exit = exit_price - exit_cost
                            profit = (effective_exit - effective_entry) / effective_entry
                        else:
                            effective_entry = entry_price - entry_cost
                            effective_exit = exit_price + exit_cost
                            profit = (effective_entry - effective_exit) / effective_entry
                        
                        balance *= 1 + profit
                        trade = Trade(
                                entry_time=position["entry_time"],
                                exit_time=str(row_timestamp),
                                entry_price=entry_price,
                                exit_price=exit_price,
                                direction=direction,
                                profit=profit,
                            )
                        trades.append(trade)
                        equity_curve.append(balance)
                        exits_executed += 1
                        
                        # Update StrategyGuard with completed trade
                        if guard is not None and profit is not None:
                            guard.update_trade(profit=profit, timestamp=str(row_timestamp))
                            # Decision 근거 로그 출력 (트레이드 이벤트 시점: EXIT from daily loss limit)
                            guard.check(trade_index=i, timestamp=str(row_timestamp))
                            snapshot = guard.debug_snapshot()
                            logger.info(
                                f"[STRATEGY_GUARD][DECISION] "
                                f"event=EXIT_DAILY_LOSS trade_index={len(trades)} idx={i} ts={str(row_timestamp)} "
                                f"decision={snapshot['decision']} recent_trades_count={snapshot['recent_trades_count']} "
                                f"recent_trades_window={snapshot['recent_trades_window']} "
                                f"win_rate={snapshot['win_rate']:.4f} avg_return={snapshot['avg_return']:.6f} "
                                f"min_win_rate={snapshot['min_win_rate']:.4f} min_avg_return={snapshot['min_avg_return']:.6f} "
                                f"min_block_trades={snapshot['min_block_trades']} "
                                f"is_blocked={snapshot['is_blocked']} block_trades_remaining={snapshot['block_trades_remaining']}"
                            )
                        
                        position = None
                        position_entry_bar_index = None
                        last_exit_bar_index = i
                        last_exit_direction = direction
                        last_exit_reason = "daily_loss_limit"
                    continue
            
            # Check for exit conditions if position exists
            exit_reason: str | None = None
            exit_price: float | None = None
            blocked_by_min_hold_flag = False  # boolean 플래그
            blocked_by_cooldown_flag = False  # boolean 플래그 (아래에서 재정의됨)
            blocked_by_confirmation = False
            blocked_by_margin_zone = False
            
            if position is not None:
                sth_crossed_for_aux = False  # 임계/트레일링 기반 signal_exit_th 여부 (손실 보조용)
                entry_price = position["entry_price"]
                direction = position["side"]
                bars_held = i - position_entry_bar_index if position_entry_bar_index is not None else 0
                # MFE/MAE (bar high/low) for trade log / sanity check
                if direction == "LONG":
                    ret_high = (current_high - entry_price) / entry_price
                    ret_low = (current_low - entry_price) / entry_price
                else:
                    ret_high = (entry_price - current_low) / entry_price
                    ret_low = (entry_price - current_high) / entry_price
                mfe_cur = position.get("max_favorable_excursion")
                mae_cur = position.get("max_adverse_excursion")
                position["max_favorable_excursion"] = max(mfe_cur if mfe_cur is not None else ret_high, ret_high)
                position["max_adverse_excursion"] = min(mae_cur if mae_cur is not None else ret_low, ret_low)
                
                # ======================================================================
                # [ANTI-OVERTRADING] 최소 보유 기간 체크
                # ======================================================================
                if effective_min_hold_bars is not None and bars_held < effective_min_hold_bars:
                    # 강제 청산 조건(TP/SL/max_holding)은 예외
                    # 하지만 신호 기반 청산은 min_hold 동안 금지
                    blocked_by_min_hold_flag = True

                # Profit lock exit (optional): 현재 봉 intrabar TP/SL, max_bars 이내만 (max_holding/time 이전에 평가)
                if (
                    exit_reason is None
                    and enable_profit_lock_exit
                    and position_entry_bar_index is not None
                    and profit_lock_max_bars >= 1
                ):
                    bh_pl = i - int(position_entry_bar_index)
                    if 1 <= bh_pl <= int(profit_lock_max_bars):
                        pl_r, pl_px = resolve_profit_lock_intrabar(
                            direction=direction,
                            entry_price=float(entry_price),
                            current_high=float(current_high),
                            current_low=float(current_low),
                            take_profit=float(profit_lock_take_profit),
                            stop_loss=float(profit_lock_stop_loss),
                            conservative=bool(profit_lock_conservative_intrabar),
                        )
                        if pl_r is not None and pl_px is not None:
                            exit_reason = pl_r
                            exit_price = float(pl_px)
                            blocked_by_min_hold_flag = False
                
                # Max holding bars (강제 청산 - min_hold 예외 없음)
                if max_holding_bars is not None and position_entry_bar_index is not None:
                    if bars_held >= max_holding_bars:
                        exit_reason = "max_holding"
                        exit_price = current_price
                        blocked_by_min_hold_flag = False  # 강제 청산은 min_hold 무시
                
                # Partial take-profit (Phase C7): 수익 구간에서 tp_ratio만큼 먼저 청산 (1 trade당 1회)
                # [AUDIT] 부분 청산 시 수수료: 청산 notional = ratio*size*price → fee in return = ratio*(price/entry)*fee_ratio.
                # 기존: partial_fee_impact = fee_ratio*(1+price/entry) (비율 미적용·과다). partial_tp_fee_correct=True 시 정식 적용.
                if (
                    exit_reason is None
                    and partial_tp_enabled
                    and not position.get("partial_tp_done", False)
                ):
                    if direction == "LONG":
                        current_return = (current_price - entry_price) / entry_price
                    else:
                        current_return = (entry_price - current_price) / entry_price
                    if current_return >= partial_tp_threshold:
                        fee_ratio = effective_commission_rate + effective_slippage_rate
                        partial_gross = current_return * partial_tp_ratio
                        if partial_tp_fee_correct:
                            partial_fee_impact = partial_tp_ratio * (current_price / entry_price) * fee_ratio
                        else:
                            partial_fee_impact = fee_ratio * (1.0 + current_price / entry_price)
                        partial_net_return = partial_gross - partial_fee_impact
                        position_scale_pt = position.get("position_scale", 1.0)
                        balance *= 1.0 + partial_net_return * position_scale_pt
                        position["partial_tp_done"] = True
                        position["remaining_ratio"] = 1.0 - partial_tp_ratio
                        partial_tp_count += 1
                        partial_tp_pnl_list.append(partial_net_return)
                        if trade_events is not None:
                            trade_events.append({
                                "event": "PARTIAL_TP",
                                "ts": str(row_timestamp),
                                "entry_price": entry_price,
                                "exit_price": current_price,
                                "side": direction,
                                "partial_tp_ratio": partial_tp_ratio,
                                "partial_net_return": partial_net_return,
                                "profit": partial_net_return,
                            })
                
                # Break-even stop (Phase C9): max_unrealized >= be_threshold 이면 stop = entry, 가격이 entry로 돌아오면 청산
                if exit_reason is None and break_even_stop_enabled and be_threshold > 0 and position is not None:
                    if direction == "LONG":
                        current_return = (current_price - entry_price) / entry_price
                    else:
                        current_return = (entry_price - current_price) / entry_price
                    position["max_unrealized_return"] = max(
                        position.get("max_unrealized_return", current_return),
                        current_return,
                    )
                    if position["max_unrealized_return"] >= be_threshold:
                        if not position.get("be_stop_triggered", False):
                            position["be_arm_ts"] = str(row_timestamp)
                        position["be_stop_triggered"] = True
                    if position.get("be_stop_triggered", False):
                        if direction == "LONG" and current_low <= entry_price:
                            exit_reason = "break_even"
                            exit_price = entry_price
                            break_even_exit_count += 1
                        elif direction == "SHORT" and current_high >= entry_price:
                            exit_reason = "break_even"
                            exit_price = entry_price
                            break_even_exit_count += 1
                
                # Time-stop (Phase C4): holding_bars >= time_stop_bars 이면 강제 청산
                if exit_reason is None and time_stop_enabled and time_stop_bars > 0 and position_entry_bar_index is not None:
                    if bars_held >= time_stop_bars:
                        exit_reason = "time_stop"
                        exit_price = current_price
                        time_stop_exit_count += 1
                        blocked_by_min_hold_flag = False
                
                # Calculate returns for TP/SL (only if not already exiting)
                if exit_reason is None:
                    if direction == "LONG":
                        current_return_tp = (current_high / entry_price) - 1
                        current_return_sl = (current_low / entry_price) - 1
                    else:
                        current_return_tp = (entry_price / current_low) - 1
                        current_return_sl = (entry_price / current_high) - 1
                    
                    # Check TP (강제 청산 - min_hold 예외 없음)
                    if take_profit_pct is not None and current_return_tp >= take_profit_pct:
                        exit_reason = "tp"
                        exit_price = current_high if direction == "LONG" else current_low
                        blocked_by_min_hold_flag = False  # 강제 청산은 min_hold 무시
                    
                    # Check SL (강제 청산 - min_hold 예외 없음)
                    if exit_reason is None:
                        if stop_loss_pct is not None and current_return_sl <= -stop_loss_pct:
                            exit_reason = "sl"
                            exit_price = current_low if direction == "LONG" else current_high
                            blocked_by_min_hold_flag = False  # 강제 청산은 min_hold 무시
                
                # ======================================================================
                # [EARLY EXIT] min_hold 만족 후 최근 N바에서 proba < p_floor 인 바가 K개 이상이면 청산
                # [Phase C3] flat_exit_aware: when proba_flat > threshold, use effective_bad_k = max(1, bad_k - delta)
                # ======================================================================
                if exit_reason is None and not blocked_by_min_hold_flag and early_exit_enabled and direction == "LONG" and proba_long_arr is not None:
                    total_exit_evaluations += 1
                    effective_bad_k = early_exit_bad_k
                    if flat_exit_aware_enabled and proba_short_arr is not None and i < len(proba_long_arr) and i < len(proba_short_arr):
                        p_long_i = float(proba_long_arr[i])
                        p_short_i = float(proba_short_arr[i])
                        p_flat_i = max(0.0, min(1.0, 1.0 - p_long_i - p_short_i))
                        if p_flat_i > flat_exit_threshold:
                            effective_bad_k = max(1, early_exit_bad_k - flat_exit_badk_delta)
                            flat_exit_adjusted_count += 1
                    start_j = max(0, i - early_exit_lookback + 1)
                    bad_count = 0
                    for j in range(start_j, i + 1):
                        if j < len(proba_long_arr) and float(proba_long_arr[j]) < early_exit_p_floor:
                            bad_count += 1
                    if bad_count >= effective_bad_k:
                        exit_reason = "early_exit"
                        exit_price = current_price
                        early_exit_count += 1
                        if effective_bad_k != early_exit_bad_k:
                            flat_exit_trigger_count += 1
                
                # ======================================================================
                # [ANTI-OVERTRADING] 히스테리시스 기반 신호 체크
                # ======================================================================
                if exit_reason is None and not blocked_by_min_hold_flag:
                    # 히스테리시스: enter/exit threshold 분리
                    if proba_long_arr is not None and proba_short_arr is not None:
                        p_long = float(proba_long_arr[i]) if i < len(proba_long_arr) else 0.0
                        p_short = float(proba_short_arr[i]) if i < len(proba_short_arr) else 0.0
                        
                        # Exit threshold 체크 (enter보다 낮은 임계값)
                        if direction == "LONG":
                            # LONG 포지션: proba_long < exit_long_th 이면 청산
                            if exit_long_th is not None:
                                W = max(1, int(signal_exit_th_trailing_window_bars))
                                crossed_trailing = False
                                if position_entry_bar_index is not None and W > 1:
                                    j0 = max(int(position_entry_bar_index), i - W + 1)
                                    seg = proba_long_arr[j0 : i + 1]
                                    crossed_trailing = bool(float(np.min(seg)) < float(exit_long_th))
                                else:
                                    crossed_trailing = p_long < exit_long_th
                                delta_hit = False
                                if signal_exit_th_delta_from_entry is not None and position is not None:
                                    ep = position.get("entry_dir_proba")
                                    if ep is not None:
                                        delta_hit = (p_long - float(ep)) < -float(signal_exit_th_delta_from_entry)
                                if crossed_trailing or delta_hit:
                                    exit_reason = "signal_exit_th"
                                    exit_price = current_price
                                    sth_crossed_for_aux = crossed_trailing
                            else:
                                # exit_long_th가 없으면 기존 로직 사용
                                exit_reason = self._should_exit_position(position, signal)
                                if exit_reason is not None and exit_price is None:
                                    exit_price = current_price
                        elif direction == "SHORT":
                            # SHORT 포지션: proba_short < exit_short_th 이면 청산
                            if exit_short_th is not None:
                                W = max(1, int(signal_exit_th_trailing_window_bars))
                                crossed_trailing = False
                                if position_entry_bar_index is not None and W > 1:
                                    j0 = max(int(position_entry_bar_index), i - W + 1)
                                    seg = proba_short_arr[j0 : i + 1]
                                    crossed_trailing = bool(float(np.min(seg)) < float(exit_short_th))
                                else:
                                    crossed_trailing = p_short < exit_short_th
                                delta_hit = False
                                if signal_exit_th_delta_from_entry is not None and position is not None:
                                    ep = position.get("entry_dir_proba")
                                    if ep is not None:
                                        delta_hit = (p_short - float(ep)) < -float(signal_exit_th_delta_from_entry)
                                if crossed_trailing or delta_hit:
                                    exit_reason = "signal_exit_th"
                                    exit_price = current_price
                                    sth_crossed_for_aux = crossed_trailing
                            else:
                                # exit_short_th가 없으면 기존 로직 사용
                                exit_reason = self._should_exit_position(position, signal)
                                if exit_reason is not None and exit_price is None:
                                    exit_price = current_price
                    else:
                        # proba 배열이 없으면 기존 로직 사용
                        exit_reason = self._should_exit_position(position, signal)
                    if exit_reason is not None and exit_price is None:
                        exit_price = current_price
                
                # 손실형 signal_exit_th 보조: (entry_proba - proba[i-1]) > δ_p 및 MAE < mae_cut 일 때 직전 바 close 로 청산
                if (
                    position is not None
                    and exit_reason == "signal_exit_th"
                    and exit_price is not None
                    and signal_exit_th_loss_aux_delta_p is not None
                    and signal_exit_th_loss_aux_mae_cut is not None
                    and sth_crossed_for_aux
                    and i > 0
                ):
                    ep = position.get("entry_dir_proba")
                    mae_val = position.get("max_adverse_excursion")
                    bh = i - position_entry_bar_index if position_entry_bar_index is not None else 0
                    ok_dir = (not signal_exit_th_loss_aux_short_only) or direction == "SHORT"
                    ok_bars = (
                        signal_exit_th_loss_aux_min_bars is None
                        or bh >= int(signal_exit_th_loss_aux_min_bars)
                    )
                    if ep is not None and mae_val is not None and ok_dir and ok_bars:
                        p_prev: float | None = None
                        drop = 0.0
                        if direction == "LONG" and proba_long_arr is not None:
                            p_prev = float(proba_long_arr[i - 1])
                            drop = float(ep) - p_prev
                        elif direction == "SHORT" and proba_short_arr is not None:
                            p_prev = float(proba_short_arr[i - 1])
                            drop = float(ep) - p_prev
                        if (
                            p_prev is not None
                            and drop > float(signal_exit_th_loss_aux_delta_p)
                            and float(mae_val) < float(signal_exit_th_loss_aux_mae_cut)
                        ):
                            apply_loss_aux_price = True
                            # 청산 직전 4바 Δunrealized (winners/losers 분석과 동일 정의)
                            entry_bar = position_entry_bar_index
                            if entry_bar is not None and entry_bar >= 0 and i < len(df):
                                cx = float(df["close"].iloc[i])
                                j4 = max(int(entry_bar), i - 4)
                                cj4 = float(df["close"].iloc[j4])
                                if direction == "LONG":
                                    u_exit = (cx - entry_price) / entry_price
                                    u_prev4 = (cj4 - entry_price) / entry_price
                                else:
                                    u_exit = (entry_price - cx) / entry_price
                                    u_prev4 = (entry_price - cj4) / entry_price
                                d_unreal_4 = u_exit - u_prev4
                            else:
                                d_unreal_4 = 0.0
                            if signal_exit_th_loss_aux_gate_delta_unreal_positive and d_unreal_4 > 0.0:
                                apply_loss_aux_price = False
                            mfe_tracked = position.get("max_favorable_excursion")
                            if (
                                apply_loss_aux_price
                                and signal_exit_th_loss_aux_gate_mfe_min is not None
                                and mfe_tracked is not None
                                and float(mfe_tracked) >= float(signal_exit_th_loss_aux_gate_mfe_min)
                            ):
                                apply_loss_aux_price = False
                            if apply_loss_aux_price:
                                exit_price = float(df["close"].iloc[i - 1])
                                signal_exit_th_loss_aux_applied += 1
                
                # Execute exit if needed
                if exit_reason is not None:
                    # Safety check: exit_price must be assigned
                    if exit_price is None:
                        logger.warning(
                            f"{self.log_prefix} exit_reason={exit_reason} but exit_price is None. "
                            f"Using current_price={current_price} as fallback."
                        )
                        exit_price = current_price
                    
                    # ======================================================================
                    # [FEE DEBUG] 수수료/슬리피지 계산 상세
                    # ======================================================================
                    entry_cost = entry_price * (effective_commission_rate + effective_slippage_rate)
                    exit_cost = exit_price * (effective_commission_rate + effective_slippage_rate)
                    
                    # Calculate raw return (before fees)
                    if direction == "LONG":
                        raw_return = (exit_price - entry_price) / entry_price
                        effective_entry = entry_price + entry_cost
                        effective_exit = exit_price - exit_cost
                        profit = (effective_exit - effective_entry) / effective_entry
                    else:
                        raw_return = (entry_price - exit_price) / entry_price
                        effective_entry = entry_price - entry_cost
                        effective_exit = exit_price + exit_cost
                        profit = (effective_entry - effective_exit) / effective_entry
                    
                    # Phase C7: 부분 익절 후 잔여 포지션만 청산 시 잔여 비율 기준 수익률 적용
                    # [AUDIT] 잔여 청산 시 entry 수수료는 진입 시 이미 지불됨. exit만 잔여 notional 기준: remaining_ratio*(exit/entry)*fee_ratio.
                    # partial_tp_fee_correct=True 시 위 정식 적용; 기본값은 기존 결과 호환 유지.
                    remaining_ratio = position.get("remaining_ratio", 1.0)
                    if remaining_ratio < 1.0:
                        fee_ratio = effective_commission_rate + effective_slippage_rate
                        if partial_tp_fee_correct:
                            fee_impact_remaining = remaining_ratio * (exit_price / entry_price) * fee_ratio
                        else:
                            fee_impact_remaining = fee_ratio * (1.0 + exit_price / entry_price)
                        if direction == "LONG":
                            profit = (exit_price - entry_price) / entry_price * remaining_ratio - fee_impact_remaining
                        else:
                            profit = (entry_price - exit_price) / entry_price * remaining_ratio - fee_impact_remaining
                    
                    # Calculate fees and slippage
                    total_fee = entry_cost + exit_cost
                    slippage_cost = abs(exit_price - current_price) if exit_price != current_price else 0.0
                    
                    # Fee as ratio of price (for unit check)
                    entry_fee_ratio = entry_cost / entry_price if entry_price > 0 else 0.0
                    exit_fee_ratio = exit_cost / exit_price if exit_price > 0 else 0.0
                    total_fee_ratio = entry_fee_ratio + exit_fee_ratio
                    
                    # Determine trade event type
                    trade_event = "EXIT_TO_FLAT"
                    if exit_reason == "tp":
                        trade_event = "EXIT_TP"
                    elif exit_reason == "sl":
                        trade_event = "EXIT_SL"
                    elif exit_reason == "max_holding":
                        trade_event = "EXIT_MAX_HOLDING"
                    elif exit_reason == "time_stop":
                        trade_event = "EXIT_TIME_STOP"
                    elif exit_reason == "break_even":
                        trade_event = "EXIT_BREAK_EVEN"
                    elif exit_reason == "profit_lock_tp":
                        trade_event = "EXIT_PROFIT_LOCK_TP"
                    elif exit_reason == "profit_lock_sl":
                        trade_event = "EXIT_PROFIT_LOCK_SL"
                    elif exit_reason == "signal_opposite" or exit_reason == "signal_exit_th":
                        # Check if we're entering opposite position (flip)
                        if signal in ("LONG", "SHORT") and signal != direction:
                            trade_event = "FLIP"
                            flip_count += 1
                        else:
                            trade_event = "EXIT_TO_FLAT"
                    
                    # position_scale 적용: profit을 스케일링하여 실제 노출 조절
                    position_scale = position.get("position_scale", 1.0) if position else 1.0
                    scaled_profit = profit * position_scale
                    
                    balance_before = balance
                    balance *= 1 + scaled_profit  # 스케일링된 profit 적용
                    balance_after = balance
                    pnl_change = balance_after - balance_before
                    
                    # Create trade record (scaled_profit을 실제 반영된 수익률로 저장)
                    trade = Trade(
                        entry_time=position["entry_time"],
                        exit_time=str(row_timestamp),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=direction,
                        profit=scaled_profit,  # 실제 반영된 수익률 (원본 profit * position_scale)
                    )
                    trades.append(trade)
                    if exit_reason == "signal_exit_th":
                        signal_exit_th_trade_count += 1
                        signal_exit_th_scaled_profit_sum += scaled_profit
                    
                    # [MONITOR][EXIT] 로그 기록
                    trade_id = position.get("trade_id") if position else None
                    if trade_id is not None:
                        monitor.log_exit(
                            ts=str(row_timestamp),
                            symbol=self.symbol,
                            timeframe=self.timeframe,
                            bar_index=i,
                            trade_id=trade_id,
                            exit_price=exit_price,
                            realized_profit=profit,  # 원본 profit
                            holding_bars=i - position_entry_bar_index if position_entry_bar_index is not None else 0,
                            mfe=None,  # MFE는 현재 계산되지 않음
                            mae=None,  # MAE는 현재 계산되지 않음
                        )
                    
                    # EXIT 시점 CAP link 로그
                    if use_stage2 and trade_id is not None and len(cap_entry_exit_links) < 50:
                        import json
                        import numpy as np
                        
                        def to_python_type(val):
                            if val is None:
                                return None
                            if isinstance(val, (np.integer, np.floating)):
                                return float(val) if isinstance(val, np.floating) else int(val)
                            if isinstance(val, np.ndarray):
                                return val.tolist()
                            return val
                        
                        # ENTRY 링크 찾기
                        entry_link = next((link for link in cap_entry_exit_links if link.get("trade_id") == trade_id and link.get("event") == "ENTRY"), None)
                        if entry_link:
                            # Guard v2 EXIT link도 업데이트
                            guard_entry_link = next((link for link in guard_entry_exit_links if link.get("trade_id") == trade_id and link.get("event") == "ENTRY"), None)
                            if guard_entry_link:
                                guard_entry_link.update({
                                    "event": "EXIT",
                                    "exit_ts": str(row_timestamp),
                                    "exit_price": exit_price,
                                    "holding_bars": i - position_entry_bar_index if position_entry_bar_index is not None else 0,
                                    "realized_profit": profit,
                                    "scaled_profit": scaled_profit,
                                })
                                logger.info(f"[EXIT][GUARDLINK] {json.dumps(guard_entry_link)}")
                        
                        if entry_link:
                            exit_link = {
                                "trade_id": trade_id,
                                "event": "EXIT",
                                "exit_ts": str(row_timestamp),
                                "exit_price": float(exit_price),
                                "holding_bars": i - position_entry_bar_index if position_entry_bar_index is not None else 0,
                                "realized_profit": to_python_type(profit) if profit is not None else None,  # 원본 profit
                                "scaled_profit": to_python_type(scaled_profit) if scaled_profit is not None else None,  # 스케일링된 profit
                                "entry_cap": entry_link.get("stage2_cap"),
                                "entry_final_scale": entry_link.get("final_scale"),
                            }
                            # ENTRY 링크에 EXIT 정보 병합
                            entry_link.update(exit_link)
                            logger.info(f"[EXIT][CAPLINK] {json.dumps(exit_link)}")
                    
                    cap_trigger_stats["entries_executed"] += 1
                    
                    # CAP breakdown 기록 (ENTRY 기준 cap_bucket으로 그룹화)
                    if use_stage2:
                        stage2_cap_val = position.get("stage2_cap", 1.0)
                        # cap_bucket 결정 (0.6, 0.8, 1.0 중 하나)
                        if abs(stage2_cap_val - 0.6) < 0.001:
                            cap_bucket = 0.6
                        elif abs(stage2_cap_val - 0.8) < 0.001:
                            cap_bucket = 0.8
                        else:
                            cap_bucket = 1.0
                        
                        cap_breakdown_records.append({
                            "cap_bucket": cap_bucket,
                            "stage2_cap": stage2_cap_val,
                            "final_scale": position.get("final_scale", 1.0),
                            "scaled_profit": scaled_profit,
                            "bars_held": i - position_entry_bar_index if position_entry_bar_index is not None else 0,
                            "entry_idx": position.get("entry_idx", i),
                            "exit_idx": i,
                        })
                    
                    # Debug: 원본 profit과 scaled_profit 로깅 (첫 10개만)
                    if trade_count < MAX_DEBUG_TRADES:
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] position_scale={position_scale:.3f}, "
                            f"profit_original={profit:.6f}, profit_scaled={scaled_profit:.6f}"
                        )
                    
                    # Update StrategyGuard with completed trade (원본 profit 사용, Guard는 실제 트레이드 성과 평가)
                    if guard is not None and profit is not None:
                        guard.update_trade(profit=profit, timestamp=str(row_timestamp))
                        # Phase-2: BLOCK 이후 트레이드 수 증가 (히스테리시스)
                        if guard.state.current_decision == "BLOCK":
                            guard.state.trades_since_block += 1
                        # Decision 근거 로그 출력 (트레이드 이벤트 시점: EXIT)
                        guard.check(trade_index=i, timestamp=str(row_timestamp))
                        snapshot = guard.debug_snapshot()
                        logger.info(
                            f"[STRATEGY_GUARD][DECISION] "
                            f"event=EXIT trade_index={len(trades)} idx={i} ts={str(row_timestamp)} "
                            f"decision={snapshot['decision']} recent_trades_count={snapshot['recent_trades_count']} "
                            f"recent_trades_window={snapshot['recent_trades_window']} "
                            f"win_rate={snapshot['win_rate']:.4f} avg_return={snapshot['avg_return']:.6f} "
                            f"min_win_rate={snapshot['min_win_rate']:.4f} min_avg_return={snapshot['min_avg_return']:.6f} "
                            f"min_block_trades={snapshot['min_block_trades']} "
                            f"is_blocked={snapshot['is_blocked']} block_trades_remaining={snapshot['block_trades_remaining']}"
                        )
                    
                    # ======================================================================
                    # [TRADE DEBUG] 첫 10개 트레이드 상세 로그
                    # ======================================================================
                    if trade_count < MAX_DEBUG_TRADES:
                        logger.info("=" * 80)
                        logger.info(f"{self.log_prefix}[TRADE DEBUG] Trade #{trade_count + 1}")
                        logger.info("=" * 80)
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] trade_index={trade_count + 1}, "
                            f"direction={direction}, exit_reason={exit_reason}"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] entry_price={entry_price:.2f}, "
                            f"exit_price={exit_price:.2f}, current_price={current_price:.2f}"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] raw_return={raw_return:.6f} "
                            f"(=(exit-entry)/entry for LONG, (entry-exit)/entry for SHORT)"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] entry_cost={entry_cost:.4f} "
                            f"(=entry_price * (commission+slippage) = {entry_price:.2f} * {effective_commission_rate + effective_slippage_rate:.6f})"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] exit_cost={exit_cost:.4f} "
                            f"(=exit_price * (commission+slippage) = {exit_price:.2f} * {effective_commission_rate + effective_slippage_rate:.6f})"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] total_fee={total_fee:.4f} "
                            f"(=entry_cost + exit_cost, 절대값 단위)"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] entry_fee_ratio={entry_fee_ratio:.6f}, "
                            f"exit_fee_ratio={exit_fee_ratio:.6f}, total_fee_ratio={total_fee_ratio:.6f}"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] effective_entry={effective_entry:.2f}, "
                            f"effective_exit={effective_exit:.2f}"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] profit={profit:.6f} "
                            f"(=(effective_exit-effective_entry)/effective_entry)"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] balance_before={balance_before:.6f}, "
                            f"balance_after={balance_after:.6f}, pnl_change={pnl_change:.6f}"
                        )
                        logger.info(
                            f"{self.log_prefix}[TRADE DEBUG] balance_update: {balance_before:.6f} * (1 + {profit:.6f}) = {balance_after:.6f}"
                        )
                        logger.info("=" * 80)
                        trade_count += 1
                    
                    # Log trade event (with dump fields)
                    # Guard decision: Guard OFF면 빈값, ON이면 실제 판정
                    guard_decision_for_dump = ""
                    if use_strategy_guard:
                        guard_decision_for_dump = guard_decision_by_idx.get(i, "")
                    
                    # Stage-2 trade: Stage-2 OFF면 None, ON이면 실제 값
                    stage2_trade_for_dump = None
                    stage2_reason_for_dump = ""
                    if use_stage2:
                        stage2_trade_for_dump = stage2_trade
                        stage2_reason_for_dump = stage2_reason
                    
                    bars_held_exit = i - position_entry_bar_index if position_entry_bar_index is not None else 0
                    _exit_rule = (
                        "profit_lock"
                        if exit_reason in ("profit_lock_tp", "profit_lock_sl")
                        else None
                    )
                    trade_events.append({
                        "event": trade_event,
                        "idx": i,
                        "ts": str(row_timestamp),
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "profit": profit,
                        "fee": total_fee,
                        "slippage": slippage_cost,
                        "pnl_change": pnl_change,
                        "balance_after": balance_after,
                        "exit_reason": exit_reason,
                        "exit_rule": _exit_rule,
                        "exit_bar_index": i,
                        "profit_lock_take_profit": (
                            float(profit_lock_take_profit)
                            if exit_reason in ("profit_lock_tp", "profit_lock_sl")
                            else None
                        ),
                        "profit_lock_stop_loss": (
                            float(profit_lock_stop_loss)
                            if exit_reason in ("profit_lock_tp", "profit_lock_sl")
                            else None
                        ),
                        "bars_held": bars_held_exit,
                        "entry_ts": position.get("entry_time"),
                        "exit_ts": str(row_timestamp),
                        "holding_bars": bars_held_exit,
                        "net_return": profit,
                        "position_scale": position.get("position_scale", 1.0),
                        "max_favorable_excursion": position.get("max_favorable_excursion"),
                        "max_adverse_excursion": position.get("max_adverse_excursion"),
                        "be_armed": position.get("be_stop_triggered", False),
                        "be_arm_ts": position.get("be_arm_ts"),
                        # Dump fields
                        "raw_signal": raw_signal,
                        "final_signal": signal,
                        "stage2_trade": stage2_trade_for_dump,
                        "stage2_reason": stage2_reason_for_dump,
                        "guard_decision": guard_decision_for_dump,
                        "trade_index": len(trades) + 1,  # 1-based trade index
                        "top1_proba": position.get("entry_top1_proba"),
                        "top2_proba": position.get("entry_top2_proba"),
                        "proba_gap": position.get("entry_proba_gap"),
                    })
                    
                    logger.debug(
                        f"{self.log_prefix}[EXECUTION] {trade_event}: idx={i} ts={row_timestamp} "
                        f"direction={direction} entry={entry_price:.2f} exit={exit_price:.2f} "
                        f"profit={profit:.4f} fee={total_fee:.4f} slippage={slippage_cost:.4f} "
                        f"pnl_change={pnl_change:.6f} balance={balance_after:.6f} "
                        f"exit_reason={exit_reason} bars_held={i - position_entry_bar_index if position_entry_bar_index is not None else 0}"
                    )
                    # NOTE: Do not append a second Trade for the same exit (scaled_profit row already recorded above).
                    equity_curve.append(balance)
                    exits_executed += 1
                    if exit_reason == "tp":
                        tp_exits += 1
                    elif exit_reason == "sl":
                        sl_exits += 1
                    elif exit_reason == "profit_lock_tp":
                        profit_lock_tp_exits += 1
                    elif exit_reason == "profit_lock_sl":
                        profit_lock_sl_exits += 1
                    position = None
                    position_entry_bar_index = None
                    last_exit_bar_index = i  # Track exit for cooldown
                    last_exit_direction = direction
                    last_exit_reason = exit_reason if exit_reason is not None else "unknown"
                    exit_count += 1
            
            # ======================================================================
            # [ANTI-OVERTRADING] Entry logic with cooldown and hysteresis
            # ======================================================================
            if position is None and not trading_disabled:
                blocked_by_regime_ablation_flag = False
                blocked_by_positive_regime_flag = False
                # Entry 시도 카운팅: signal이 LONG/SHORT인 순간부터 카운팅
                if signal in ("LONG", "SHORT"):
                    entries_attempted += 1
                
                # Cooldown 체크
                blocked_by_cooldown_flag = False
                if effective_cooldown_bars is not None and last_exit_bar_index is not None:
                    bars_since_exit = i - last_exit_bar_index
                    if bars_since_exit < effective_cooldown_bars:
                        blocked_by_cooldown_flag = True
                        blocked_by_cooldown += 1  # 모니터링용 카운터
                        if signal in ("LONG", "SHORT"):
                            block_reasons["cooldown"] += 1
                            logger.debug(
                                f"{self.log_prefix}[ENTRY BLOCKED] Cooldown: "
                                f"signal={signal}, bars_since_exit={bars_since_exit}, "
                                f"cooldown_bars={effective_cooldown_bars}"
                            )

                # 재진입 억제 (profit lock 등 조기 청산 후 과매매 완화); 기본 OFF
                blocked_by_ot_flag = False
                ot_cooldown_type: str | None = None
                ot_skip_reason: str | None = None
                any_ot_enabled = (
                    enable_reentry_cooldown
                    or enable_directional_cooldown
                    or enable_post_tp_block
                    or enable_post_sl_block
                )
                if signal in ("LONG", "SHORT"):
                    blocked_by_ot_flag, ot_cooldown_type, ot_skip_reason = overtrading_reentry_block(
                        bar_index=i,
                        signal=signal,
                        last_exit_bar_index=last_exit_bar_index,
                        last_exit_direction=last_exit_direction,
                        last_exit_reason=last_exit_reason,
                        enable_reentry_cooldown=enable_reentry_cooldown,
                        reentry_cooldown_bars=reentry_cooldown_bars,
                        enable_directional_cooldown=enable_directional_cooldown,
                        directional_cooldown_bars=directional_cooldown_bars,
                        enable_post_tp_block=enable_post_tp_block,
                        post_tp_block_bars=post_tp_block_bars,
                        enable_post_sl_block=enable_post_sl_block,
                        post_sl_block_bars=post_sl_block_bars,
                    )
                    if blocked_by_ot_flag:
                        if ot_cooldown_type == "global":
                            blocked_by_ot_global += 1
                        elif ot_cooldown_type == "directional":
                            blocked_by_ot_directional += 1
                        elif ot_cooldown_type == "post_tp":
                            blocked_by_ot_post_tp += 1
                        elif ot_cooldown_type == "post_sl":
                            blocked_by_ot_post_sl += 1
                        if emit_trade_log and any_ot_enabled:
                            trade_events.append({
                                "event": "ENTRY_SKIPPED_OT",
                                "idx": i,
                                "ts": str(row_timestamp),
                                "signal": signal,
                                "skip_due_to_cooldown": True,
                                "cooldown_type": ot_cooldown_type,
                                "skip_reason": ot_skip_reason,
                                "bars_since_exit": (
                                    i - last_exit_bar_index
                                    if last_exit_bar_index is not None
                                    else None
                                ),
                                "last_exit_reason": last_exit_reason,
                                "last_exit_direction": last_exit_direction,
                            })
                
                if (
                    enable_regime_filter
                    and regime_filter_mode not in (None, "", "none")
                    and regime_ablation_block_mask is not None
                    and signal in ("LONG", "SHORT")
                ):
                    if bool(regime_ablation_block_mask[i]):
                        blocked_by_regime_ablation_flag = True
                        entries_blocked_by_regime_ablation += 1
                        block_reasons["regime_ablation"] = block_reasons.get("regime_ablation", 0) + 1

                blocked_by_adaptive_positive_flag = False
                if (
                    use_static_positive
                    and positive_regime_filter_mode not in (None, "", "none")
                    and positive_regime_block_mask is not None
                    and signal in ("LONG", "SHORT")
                    and not blocked_by_cooldown_flag
                    and not blocked_by_ot_flag
                    and not blocked_by_regime_ablation_flag
                ):
                    if bool(positive_regime_block_mask[i]):
                        blocked_by_positive_regime_flag = True
                        entries_blocked_by_positive_regime += 1
                        positive_regime_blocked_entries += 1
                        block_reasons["positive_regime"] = block_reasons.get("positive_regime", 0) + 1
                        if emit_trade_log:
                            rk_parts = ["", "", ""]
                            if positive_regime_key_arr is not None and i < len(positive_regime_key_arr):
                                rk = str(positive_regime_key_arr[i])
                                parts = rk.split("_", 2)
                                if len(parts) >= 3:
                                    rk_parts = [parts[0], parts[1], parts[2]]
                            trade_events.append(
                                {
                                    "event": "ENTRY_SKIPPED_POSITIVE_REGIME",
                                    "idx": i,
                                    "ts": str(row_timestamp),
                                    "signal": signal,
                                    "skip_reason": "positive_regime_filter",
                                    "positive_regime_filter_mode": positive_regime_filter_mode,
                                    "ema200_side": rk_parts[0],
                                    "trend_state": rk_parts[1],
                                    "vol_bucket": rk_parts[2],
                                }
                            )
                    else:
                        positive_regime_allowed_entries += 1

                if (
                    enable_adaptive_positive_regime_filter
                    and adaptive_positive_regime_mode not in (None, "", "none")
                    and adaptive_positive_final_block is not None
                    and signal in ("LONG", "SHORT")
                    and not blocked_by_cooldown_flag
                    and not blocked_by_ot_flag
                    and not blocked_by_regime_ablation_flag
                    and not blocked_by_positive_regime_flag
                ):
                    if adaptive_gate_on_mask is not None and bool(adaptive_gate_on_mask[i]):
                        adaptive_positive_gate_on_count += 1
                    if bool(adaptive_positive_final_block[i]):
                        blocked_by_adaptive_positive_flag = True
                        entries_blocked_by_adaptive_positive += 1
                        adaptive_positive_blocked_entries += 1
                        block_reasons["adaptive_positive"] = block_reasons.get("adaptive_positive", 0) + 1
                        if emit_trade_log:
                            rk_parts = ["", "", ""]
                            if adaptive_regime_key_arr is not None and i < len(adaptive_regime_key_arr):
                                rk = str(adaptive_regime_key_arr[i])
                                parts = rk.split("_", 2)
                                if len(parts) >= 3:
                                    rk_parts = [parts[0], parts[1], parts[2]]
                            trade_events.append(
                                {
                                    "event": "ENTRY_SKIPPED_ADAPTIVE_POSITIVE_REGIME",
                                    "idx": i,
                                    "ts": str(row_timestamp),
                                    "signal": signal,
                                    "skip_reason": "adaptive_positive_regime_filter",
                                    "adaptive_positive_regime_mode": adaptive_positive_regime_mode,
                                    "adaptive_gate_on": True,
                                    "ema200_side": rk_parts[0],
                                    "trend_state": rk_parts[1],
                                    "vol_bucket": rk_parts[2],
                                }
                            )
                    else:
                        adaptive_positive_allowed_entries += 1

                # Entry 허용 체크
                if (
                    signal in ("LONG", "SHORT")
                    and not blocked_by_cooldown_flag
                    and not blocked_by_ot_flag
                    and not blocked_by_regime_ablation_flag
                    and not blocked_by_positive_regime_flag
                    and not blocked_by_adaptive_positive_flag
                ):
                        # 히스테리시스: enter threshold 체크
                        entry_allowed = False
                        blocked_by_hysteresis = False
                        p_long = None
                        p_short = None
                        hysteresis_reason = ""
                        guard_scale = 1.0  # 기본값 (Guard v2 OFF일 때)
                        guard_v2_decision = "ALLOW"  # 기본값
                        skip_entry_filter = False  # min_max_proba / max_entropy 필터 (옵션)
                        entry_proba_gap_val = None
                        entry_top1_val = None
                        entry_top2_val = None

                        # Guard v2 디버그: proba 배열 확인
                        if guard_v2 is not None and i < 5:
                            logger.info(
                                f"{self.log_prefix}[StrategyGuardV2][DEBUG] "
                                f"idx={i}, signal={signal}, proba_long_arr={'None' if proba_long_arr is None else f'len={len(proba_long_arr)}'}, "
                                f"proba_short_arr={'None' if proba_short_arr is None else f'len={len(proba_short_arr)}'}"
                            )
                        
                        # Guard v2가 있지만 proba 배열이 None인 경우 경고
                        if guard_v2 is not None and (proba_long_arr is None or proba_short_arr is None) and i < 5:
                            logger.warning(
                                f"{self.log_prefix}[StrategyGuardV2][WARNING] "
                                f"Guard v2 enabled but proba arrays are None! "
                                f"proba_long_arr={'None' if proba_long_arr is None else 'OK'}, "
                                f"proba_short_arr={'None' if proba_short_arr is None else 'OK'}"
                            )
                        
                        if proba_long_arr is not None and proba_short_arr is not None:
                            p_long = proba_long_arr[i] if i < len(proba_long_arr) else 0.0
                            p_short = proba_short_arr[i] if i < len(proba_short_arr) else 0.0
                            p_flat = float(1.0 - p_long - p_short)
                            p_flat = max(0.0, min(1.0, p_flat))

                            # [ENTRY FILTER] flat-proba gate (Phase C): skip when proba_flat > max_flat_proba
                            if entry_flat_gate_enabled and p_flat > max_flat_proba:
                                skip_entry_filter = True
                                entries_blocked_by_flat_gate += 1

                            # [ENTRY FILTER] min_max_proba / max_entropy (optional; D12: skip when use_legacy_entry_gates=False)
                            if use_legacy_entry_gates and (min_max_proba is not None or max_entropy is not None):
                                current_max_proba = max(p_long, p_short, p_flat)
                                _h = 0.0
                                for _p in (p_long, p_short, p_flat):
                                    if _p > 1e-12:
                                        _h -= _p * math.log2(_p)
                                current_entropy = float(_h)
                                if min_max_proba is not None and current_max_proba < min_max_proba:
                                    skip_entry_filter = True
                                    skipped_by_min_max_proba += 1
                                if max_entropy is not None and current_entropy > max_entropy:
                                    skip_entry_filter = True
                                    skipped_by_max_entropy += 1

                            # [ENTRY FILTER] confidence spread (Phase D10): spread = top1 - top2; require spread >= min_proba_gap
                            _sorted_probs = sorted([p_long, p_short, p_flat], reverse=True)
                            entry_top1_val = _sorted_probs[0]
                            entry_top2_val = _sorted_probs[1]
                            entry_proba_gap_val = entry_top1_val - entry_top2_val
                            if min_proba_gap is not None and min_proba_gap > 0 and entry_proba_gap_val < min_proba_gap:
                                skip_entry_filter = True
                                skipped_by_proba_gap += 1
                                spread_rejected_list.append(entry_proba_gap_val)

                            # [ENTRY FILTER] Phase D11: directional gap (long_edge / short_edge)
                            if min_directional_gap is not None and signal == "LONG":
                                long_edge = float(p_long - p_short)
                                if long_edge < min_directional_gap:
                                    skip_entry_filter = True
                                    directional_gap_fail_count += 1
                                    long_directional_gap_fail_count += 1
                            if min_directional_gap is not None and signal == "SHORT":
                                short_edge = float(p_short - p_long)
                                if short_edge < min_directional_gap:
                                    skip_entry_filter = True
                                    directional_gap_fail_count += 1
                                    short_directional_gap_fail_count += 1

                            # [ENTRY FILTER] Phase D11: flat suppression (p_flat <= max_flat_entry_proba)
                            if max_flat_entry_proba is not None and p_flat > max_flat_entry_proba:
                                skip_entry_filter = True
                                flat_suppression_fail_count += 1

                            # [ENTRY FILTER] Signal quality / confidence filter (diagnostics)
                            if enable_confidence_filter and confidence_filter_mode not in (None, "", "none"):
                                _cf_mode = confidence_filter_mode
                                _cf_max_p = max(p_long, p_short)
                                _cf_sorted = sorted([p_long, p_short, p_flat], reverse=True)
                                _cf_margin = _cf_sorted[0] - _cf_sorted[1]
                                _cf_h = 0.0
                                for _cp in (p_long, p_short, p_flat):
                                    if _cp > 1e-12:
                                        _cf_h -= _cp * math.log(_cp)
                                _cf_pass = True
                                if _cf_mode == "max_proba_055":
                                    _cf_pass = _cf_max_p >= 0.55
                                elif _cf_mode == "max_proba_060":
                                    _cf_pass = _cf_max_p >= 0.60
                                elif _cf_mode == "max_proba_065":
                                    _cf_pass = _cf_max_p >= 0.65
                                elif _cf_mode == "max_proba_070":
                                    _cf_pass = _cf_max_p >= 0.70
                                elif _cf_mode == "max_proba_075":
                                    _cf_pass = _cf_max_p >= 0.75
                                elif _cf_mode == "margin_005":
                                    _cf_pass = _cf_margin >= 0.05
                                elif _cf_mode == "margin_010":
                                    _cf_pass = _cf_margin >= 0.10
                                elif _cf_mode == "margin_015":
                                    _cf_pass = _cf_margin >= 0.15
                                elif _cf_mode == "margin_020":
                                    _cf_pass = _cf_margin >= 0.20
                                elif _cf_mode == "entropy_120":
                                    _cf_pass = _cf_h <= 1.20
                                elif _cf_mode == "entropy_110":
                                    _cf_pass = _cf_h <= 1.10
                                elif _cf_mode == "entropy_100":
                                    _cf_pass = _cf_h <= 1.00
                                elif _cf_mode == "entropy_090":
                                    _cf_pass = _cf_h <= 0.90
                                elif _cf_mode == "hybrid_mp060_mg010":
                                    _cf_pass = _cf_max_p >= 0.60 and _cf_margin >= 0.10
                                elif _cf_mode == "hybrid_mp065_et110":
                                    _cf_pass = _cf_max_p >= 0.65 and _cf_h <= 1.10
                                elif _cf_mode == "hybrid_mg010_et110":
                                    _cf_pass = _cf_margin >= 0.10 and _cf_h <= 1.10
                                if not _cf_pass:
                                    skip_entry_filter = True
                                    entries_blocked_by_confidence += 1

                            # Guard v2 체크 (신호 발생 시점)
                            if guard_v2 is not None:
                                # threshold는 long_threshold 또는 short_threshold 사용
                                threshold = None
                                if signal == "LONG":
                                    threshold = enter_long_th if enter_long_th is not None else 0.5
                                    p_for_guard = p_long
                                else:  # SHORT
                                    threshold = enter_short_th if enter_short_th is not None else 0.5
                                    # SHORT의 경우 p_short를 p_long로 변환 (1 - p_short)
                                    p_for_guard = 1.0 - p_short
                                
                                guard_v2.update_signal(
                                    p_long=p_for_guard,
                                    threshold=threshold,
                                    timestamp=str(row_timestamp),
                                )
                                guard_v2_decision, position_scale = guard_v2.check(
                                    p_long=p_for_guard,
                                    threshold=threshold,
                                    signal=signal,
                                    trade_index=i,
                                    timestamp=str(row_timestamp),
                                )
                                
                                # Guard v2 scale 수집 (전체 checks 기준)
                                guard_scale_all_checks.append(position_scale)
                                
                                # Guard v2 로그 출력 (신호 발생 시점 - 샘플링)
                                snapshot = guard_v2.debug_snapshot()
                                
                                # [MONITOR][CHECK] 로그 기록
                                guard_components = {
                                    "p_long": p_for_guard,
                                    "margin": snapshot.get("margin"),
                                    "entropy": snapshot.get("entropy"),
                                    "recent_mean_margin": snapshot.get("recent_mean_margin"),
                                    "recent_mean_entropy": snapshot.get("recent_mean_entropy"),
                                }
                                
                                # Stage-2 CAP은 아직 계산 전이므로 None
                                monitor.log_check(
                                    ts=str(row_timestamp),
                                    symbol=self.symbol,
                                    timeframe=self.timeframe,
                                    bar_index=i,
                                    guard_scale=position_scale,
                                    guard_components=guard_components,
                                    stage2_cap=None,  # ENTRY 블록에서 계산됨
                                    cap_reason=None,
                                    final_scale=None,  # ENTRY 블록에서 계산됨
                                    signal=signal,
                                    sample_rate=10,  # 10 bar 중 1회만 상세 로그
                                )
                                
                                # [GUARD][SCALE] 로그 (샘플링: 최대 100개)
                                if guard_scale_sample_count < 100:
                                    import json
                                    import numpy as np
                                    
                                    def to_python_type(val):
                                        if val is None:
                                            return None
                                        if isinstance(val, (np.integer, np.floating)):
                                            return float(val) if isinstance(val, np.floating) else int(val)
                                        if isinstance(val, np.ndarray):
                                            return val.tolist()
                                        return val
                                    
                                    guard_log_entry = {
                                        "event": "SIGNAL",
                                        "ts": str(row_timestamp),
                                        "symbol": self.symbol,
                                        "timeframe": self.timeframe,
                                        "bar_index": i,
                                        "signal": signal,
                                        "p_long": to_python_type(p_for_guard),
                                        "margin": to_python_type(snapshot.get("margin")),
                                        "entropy": to_python_type(snapshot.get("entropy")),
                                        "guard_scale": to_python_type(position_scale),
                                        "decision": snapshot.get("decision", "N/A"),
                                        "mode": snapshot.get("mode", "N/A"),
                                        "reason": snapshot.get("reason", "N/A"),
                                        "recent_mean_margin": to_python_type(snapshot.get("recent_mean_margin")),
                                        "recent_mean_entropy": to_python_type(snapshot.get("recent_mean_entropy")),
                                    }
                                    logger.info(f"[GUARD][SCALE] {json.dumps(guard_log_entry)}")
                                    guard_scale_sample_count += 1
                                p_long_str = f"{snapshot['p_long']:.4f}" if snapshot['p_long'] is not None else "N/A"
                                margin_str = f"{snapshot['margin']:.4f}" if snapshot['margin'] is not None else "N/A"
                                entropy_str = f"{snapshot['entropy']:.4f}" if snapshot['entropy'] is not None else "N/A"
                                mean_margin_str = f"{snapshot['recent_mean_margin']:.4f}" if snapshot['recent_mean_margin'] is not None else "N/A"
                                mean_entropy_str = f"{snapshot['recent_mean_entropy']:.4f}" if snapshot['recent_mean_entropy'] is not None else "N/A"
                                logger.info(
                                    f"[STRATEGY_GUARD][DECISION] "
                                    f"v2_enabled=True event=SIGNAL idx={i} ts={str(row_timestamp)} "
                                    f"signal={signal} decision={snapshot['decision']} scale={snapshot['position_scale']:.3f} "
                                    f"p_long={p_long_str} margin={margin_str} entropy={entropy_str} "
                                    f"mean_margin={mean_margin_str} mean_entropy={mean_entropy_str} "
                                    f"mode={snapshot['mode']} reason={snapshot['reason']}"
                                )
                                
                                # Guard v2가 BLOCK이면 entry_allowed = False
                                if guard_v2_decision == "BLOCK":
                                    entry_allowed = False
                                    blocked_by_hysteresis = True
                                    hysteresis_reason = f"GuardV2_BLOCK: {guard_v2.state.last_reason}"
                                else:
                                    # Guard v2가 ALLOW면 기존 로직 계속
                                    entry_allowed = True
                                    # Guard v2 scale 저장 (ENTRY 블록에서 사용)
                                    guard_scale = position_scale
                            else:
                                # Guard v2가 없으면 기존 로직
                                entry_allowed = True
                                # Guard v2 OFF면 guard_scale = 1.0
                                guard_scale = 1.0
                            
                            if entry_allowed and guard_v2 is None:
                                # Guard v2가 없을 때만 기존 hysteresis 체크
                                if signal == "LONG":
                                    if enter_long_th is not None:
                                        entry_allowed = p_long >= enter_long_th
                                        if not entry_allowed:
                                            blocked_by_hysteresis = True
                                            hysteresis_reason = f"p_long={p_long:.4f} < enter_long_th={enter_long_th:.4f}"
                                    else:
                                        # enter_long_th가 없으면 기존 로직 (signal만 체크)
                                        entry_allowed = True
                                elif signal == "SHORT":
                                    if enter_short_th is not None:
                                        entry_allowed = p_short >= enter_short_th
                                        if not entry_allowed:
                                            blocked_by_hysteresis = True
                                            hysteresis_reason = f"p_short={p_short:.4f} < enter_short_th={enter_short_th:.4f}"
                                    else:
                                        # enter_short_th가 없으면 기존 로직 (signal만 체크)
                                        entry_allowed = True
                        else:
                            # proba 배열이 없으면 기존 로직
                            entry_allowed = True
                        
                        # ======================================================================
                        # [SAMPLE DEBUG] 진입 20개 샘플 로깅
                        # ======================================================================
                        if entry_allowed and entry_sample_count < MAX_ENTRY_SAMPLES:
                            logger.info("=" * 80)
                            logger.info(f"{self.log_prefix}[SAMPLE DEBUG] Entry Attempt #{entry_sample_count + 1}")
                            logger.info("=" * 80)
                            logger.info(
                                f"{self.log_prefix}[SAMPLE DEBUG] idx={i}, ts={row_timestamp}"
                            )
                            logger.info(
                                f"{self.log_prefix}[SAMPLE DEBUG] signal={signal} (from DataFrame)"
                            )
                            if p_long is not None and p_short is not None:
                                logger.info(
                                    f"{self.log_prefix}[SAMPLE DEBUG] proba_long={p_long:.6f}, proba_short={p_short:.6f}"
                                )
                            logger.info(
                                f"{self.log_prefix}[SAMPLE DEBUG] enter_long_th={enter_long_th}, "
                                f"exit_long_th={exit_long_th}, enter_short_th={enter_short_th}, "
                                f"exit_short_th={exit_short_th}"
                            )
                            logger.info(
                                f"{self.log_prefix}[SAMPLE DEBUG] entry_allowed={entry_allowed}, "
                                f"blocked_by_hysteresis={blocked_by_hysteresis}, "
                                f"blocked_by_cooldown={blocked_by_cooldown_flag}"
                            )
                            if blocked_by_hysteresis:
                                logger.info(
                                    f"{self.log_prefix}[SAMPLE DEBUG] Hysteresis reason: {hysteresis_reason}"
                                )
                            entry_sample_count += 1
                        
                        # Hysteresis threshold로 막힌 경우 카운팅
                        if blocked_by_hysteresis:
                            block_reasons["hysteresis"] = block_reasons.get("hysteresis", 0) + 1
                            logger.debug(
                                f"{self.log_prefix}[ENTRY BLOCKED] Hysteresis: "
                                f"signal={signal}, p_long={p_long:.4f}, p_short={p_short:.4f}, "
                                f"enter_long_th={enter_long_th}, enter_short_th={enter_short_th}"
                            )
                        
                        if skip_entry_filter:
                            entry_allowed = False
                        if (
                            entry_allowed
                            and entry_min_unique_round_trips is not None
                            and entry_urt_gate_active
                            and entry_urt_from_state_log is not None
                            and int(entry_urt_from_state_log) < int(entry_min_unique_round_trips)
                        ):
                            entry_allowed = False
                            entries_blocked_by_urt_gate += 1
                            block_reasons["urt_gate"] = block_reasons.get("urt_gate", 0) + 1
                            if len(urt_gate_block_samples) < 10:
                                urt_gate_block_samples.append(
                                    {
                                        "bar_index": int(i),
                                        "urt": int(entry_urt_from_state_log),
                                        "threshold": int(entry_min_unique_round_trips),
                                    }
                                )
                        if entry_allowed:
                            # ======================================================================
                            # [PATCH 1] stage2_cap 계산 스코프 통일 (ENTRY 시점에서 항상 재계산)
                            # ======================================================================
                            # Guard v2 ON/OFF와 무관하게 stage2_cap을 항상 재계산
                            if use_stage2:
                                stage2_cap, stage2_cap_debug_info = calculate_stage2_cap(
                                    stage2_reason=stage2_reason,
                                    proba_long=p_long,
                                    proba_short=p_short,
                                    raw_signal=raw_signal,
                                    high_entropy_th=stage2_cap_entropy_high_th,
                                    mid_entropy_th=stage2_cap_entropy_mid_th,
                                    tiny_pdiff_th=stage2_cap_pdiff_tiny_th,
                                    small_pdiff_th=stage2_cap_pdiff_small_th,
                                )
                            else:
                                stage2_cap = 1.0
                                stage2_cap_debug_info = {"cap": 1.0, "p_diff": None, "entropy": None, "cap_reason": "stage2_off"}
                            
                            # ======================================================================
                            # [PATCH 2] final_scale 계산 (Guard v2 scale * Stage-2 cap)
                            # ======================================================================
                            # Guard v2 scale 결정
                            # Guard v2가 있으면 이미 계산된 position_scale을 guard_scale로 사용
                            # (위에서 guard_v2.check()가 호출되어 position_scale이 계산됨)
                            # Guard v2가 없으면 guard_scale = 1.0
                            # 주의: position_scale은 Guard v2 블록 내부에서만 정의되므로,
                            #       여기서는 guard_scale 변수를 사용하여 명확하게 분리
                            
                            # final_scale = guard_scale * stage2_cap
                            final_scale = guard_scale * stage2_cap
                            
                            # ======================================================================
                            # [PATCH 3] [STAGE2][CAP] 로그 강제 출력 (최대 20개 샘플)
                            # ======================================================================
                            if stage2_cap_sample_count < 20:
                                cap_reason_str = stage2_cap_debug_info.get("cap_reason", "N/A") if isinstance(stage2_cap_debug_info, dict) else "N/A"
                                cap_low_val = stage2_cap_debug_info.get("cap_low", "N/A") if isinstance(stage2_cap_debug_info, dict) else "N/A"
                                cap_mid_val = stage2_cap_debug_info.get("cap_mid", "N/A") if isinstance(stage2_cap_debug_info, dict) else "N/A"
                                logger.info(
                                    f"[STAGE2][CAP] event=ENTRY idx={i} ts={str(row_timestamp)} "
                                    f"cap={stage2_cap:.3f} guard_scale={guard_scale:.3f} "
                                    f"final_scale={final_scale:.3f} cap_low={cap_low_val} cap_mid={cap_mid_val} reason={cap_reason_str}"
                                )
                                stage2_cap_sample_count += 1
                            
                            # ======================================================================
                            # [DEBUG] CAP 상세 로그 (30줄 이상 확보용)
                            # ======================================================================
                            if use_stage2 and len(cap_debug_logs) < 100:  # 충분히 확보
                                import json
                                import numpy as np
                                
                                # numpy 타입을 Python 기본 타입으로 변환
                                def to_python_type(val):
                                    if val is None:
                                        return None
                                    if isinstance(val, (np.integer, np.floating)):
                                        return float(val) if isinstance(val, np.floating) else int(val)
                                    if isinstance(val, np.ndarray):
                                        return val.tolist()
                                    return val
                                
                                p_diff_val = to_python_type(stage2_cap_debug_info.get("p_diff") if isinstance(stage2_cap_debug_info, dict) else None)
                                entropy_val = to_python_type(stage2_cap_debug_info.get("entropy") if isinstance(stage2_cap_debug_info, dict) else None)
                                cap_reason_val = stage2_cap_debug_info.get("cap_reason", "N/A") if isinstance(stage2_cap_debug_info, dict) else "N/A"
                                
                                # margin 계산
                                p_long_float = to_python_type(p_long) if p_long is not None else None
                                margin_val = abs(p_long_float - 0.5) if p_long_float is not None else None
                                
                                # trigger 원인 분석
                                triggered_by_entropy = False
                                triggered_by_pdiff = False
                                if entropy_val is not None and p_diff_val is not None:
                                    abs_pdiff = abs(p_diff_val)
                                    if entropy_val >= stage2_cap_entropy_high_th and abs_pdiff <= stage2_cap_pdiff_tiny_th:
                                        triggered_by_entropy = True
                                        triggered_by_pdiff = True
                                    elif entropy_val >= stage2_cap_entropy_mid_th and abs_pdiff <= stage2_cap_pdiff_small_th:
                                        triggered_by_entropy = True
                                        triggered_by_pdiff = True
                                    elif entropy_val >= stage2_cap_entropy_high_th:
                                        triggered_by_entropy = True
                                    elif abs_pdiff <= stage2_cap_pdiff_tiny_th:
                                        triggered_by_pdiff = True
                                
                                # trigger_by_rule 분류
                                if triggered_by_entropy and triggered_by_pdiff:
                                    trigger_type = "both"
                                elif triggered_by_entropy:
                                    trigger_type = "entropy_only"
                                elif triggered_by_pdiff:
                                    trigger_type = "pdiff_only"
                                else:
                                    trigger_type = "none"
                                
                                cap_log_entry = {
                                    "event": "ENTRY",
                                    "ts": str(row_timestamp),
                                    "symbol": self.symbol,
                                    "timeframe": self.timeframe,
                                    "side": signal if entry_allowed else "N/A",  # direction 대신 signal 사용
                                    "bar_index": i,
                                    "p_long": to_python_type(p_long) if p_long is not None else None,
                                    "p_short": to_python_type(p_short) if p_short is not None else None,
                                    "margin": margin_val,
                                    "entropy": entropy_val,
                                    "p_diff": p_diff_val,
                                    "abs_pdiff": abs(p_diff_val) if p_diff_val is not None else None,
                                    "guard_scale": guard_scale,
                                    "stage2_cap": stage2_cap,
                                    "final_scale": final_scale,
                                    "cap_reason": cap_reason_val,
                                    "trigger_type": trigger_type,
                                    "rule_inputs": {
                                        "entropy": entropy_val,
                                        "p_diff": p_diff_val,
                                        "abs_pdiff": abs(p_diff_val) if p_diff_val is not None else None,
                                    },
                                    "thresholds": {
                                        "high_entropy_th": stage2_cap_entropy_high_th,
                                        "mid_entropy_th": stage2_cap_entropy_mid_th,
                                        "tiny_pdiff_th": stage2_cap_pdiff_tiny_th,
                                        "small_pdiff_th": stage2_cap_pdiff_small_th,
                                    },
                                }
                                cap_debug_logs.append(cap_log_entry)
                                
                                # JSON 형식으로 로그 출력 (오류 발생 시 스킵)
                                try:
                                    logger.info(f"[STAGE2][CAP] {json.dumps(cap_log_entry, default=str)}")
                                except (TypeError, ValueError) as e:
                                    logger.warning(f"[STAGE2][CAP] JSON 직렬화 실패: {e}, cap={stage2_cap}")
                                
                                # 집계 카운터 업데이트
                                cap_trigger_stats["total_checks"] += 1
                                if abs(stage2_cap - 1.0) < 0.001:
                                    cap_trigger_stats["cap_1_0_count"] += 1
                                elif abs(stage2_cap - 0.8) < 0.001:
                                    cap_trigger_stats["cap_0_8_count"] += 1
                                elif abs(stage2_cap - 0.6) < 0.001:
                                    cap_trigger_stats["cap_0_6_count"] += 1
                                
                                if trigger_type == "both":
                                    cap_trigger_stats["triggered_by_both"] += 1
                                elif trigger_type == "entropy_only":
                                    cap_trigger_stats["triggered_by_entropy_only"] += 1
                                elif trigger_type == "pdiff_only":
                                    cap_trigger_stats["triggered_by_pdiff_only"] += 1
                                else:
                                    cap_trigger_stats["triggered_by_none"] += 1
                            
                            # [정밀 분석] soft gated 카운트 (ENTRY 시점, Stage-2 ON이고 cap < 1.0일 때)
                            if use_stage2 and stage2_cap < 1.0:
                                stage2_soft_gated_count += 1
                            
                            # Confidence filter check
                            if use_confidence_filter:
                                # This would need proba arrays - for now, skip if not available
                                # Can be enhanced later
                                pass
                            
                            # Determine trade event type and direction
                            # CRITICAL: direction must match signal
                            direction = signal  # signal이 "LONG"이면 direction도 "LONG", "SHORT"이면 "SHORT"
                            
                            # ======================================================================
                            # [SAMPLE DEBUG] 진입 시 signal/direction 매핑 검증
                            # ======================================================================
                            if entry_sample_count <= MAX_ENTRY_SAMPLES:
                                logger.info(
                                    f"{self.log_prefix}[SAMPLE DEBUG] ✓ ENTRY ALLOWED: "
                                    f"signal={signal} -> direction={direction}"
                                )
                                # Assert: signal과 direction이 일치해야 함
                                if signal != direction:
                                    logger.error(
                                        f"{self.log_prefix}[SAMPLE DEBUG] ⚠️  MAPPING MISMATCH: "
                                        f"signal={signal} but direction={direction}!"
                                    )
                            
                            trade_event = f"ENTRY_{direction}"
                            
                            # [Stage-2 v2.2] 선택적 ENTRY 게이트 (final_scale 기반)
                            if stage2_block_if_final_scale_below > 0 and final_scale < stage2_block_if_final_scale_below:
                                entry_allowed = False
                                stage2_entry_blocked_by_cap_count += 1
                                blocked_by_hysteresis = True
                                hysteresis_reason = f"Stage2_CAP_BLOCK: final_scale={final_scale:.3f} < {stage2_block_if_final_scale_below:.3f}"
                                continue  # ENTRY 차단, 다음 루프로
                            
                            # [Regime filter] Long entry gate (EMA or Vol Compression)
                            if regime_filter_enabled and signal == "LONG":
                                if regime_rule in ("ema_only", "ema_plus_slope") and ema_series is not None and i < len(ema_series):
                                    close_val = current_price
                                    ema_i = float(ema_series.iloc[i])
                                    if close_val <= ema_i:
                                        entries_blocked_by_regime_price += 1
                                        entries_blocked_by_regime += 1
                                        continue
                                    if regime_rule == "ema_plus_slope" and ema_slope_series is not None and i < len(ema_slope_series):
                                        slope_i = ema_slope_series.iloc[i]
                                        if pd.isna(slope_i) or float(slope_i) <= 0:
                                            entries_blocked_by_regime_slope += 1
                                            entries_blocked_by_regime += 1
                                            continue
                                elif regime_rule == "vol_compress" and vol_series is not None and i < len(vol_series):
                                    vol_i = vol_series.iloc[i]
                                    if pd.isna(vol_i) or float(vol_i) <= vol_threshold:
                                        entries_blocked_by_regime_vol += 1
                                        entries_blocked_by_regime += 1
                                        continue
                                elif regime_rule == "vol_slope" and vol_series is not None and ema_slope_series is not None and i < len(vol_series) and i < len(ema_slope_series):
                                    vol_i = vol_series.iloc[i]
                                    slope_i = ema_slope_series.iloc[i]
                                    compress = pd.isna(vol_i) or float(vol_i) <= vol_threshold
                                    slope_ok = not pd.isna(slope_i) and float(slope_i) > 0
                                    allow_entry = (not compress) or slope_ok
                                    if not allow_entry:
                                        entries_blocked_by_regime_vol_slope += 1
                                        entries_blocked_by_regime += 1
                                        continue
                                elif regime_rule == "vol_breakout" and compress_series is not None and decompress_event_series is not None and ema_series is not None and i < len(compress_series) and i < len(decompress_event_series) and i < len(ema_series):
                                    compress_i = bool(compress_series.iloc[i])
                                    decompress_i = bool(decompress_event_series.iloc[i])
                                    if compress_i:
                                        entries_blocked_by_regime_vol += 1
                                        entries_blocked_by_regime += 1
                                        continue
                                    if decompress_i:
                                        ema_i = ema_series.iloc[i]
                                        ema_ok = not pd.isna(ema_i) and current_price > float(ema_i)
                                        if breakout_mode == "ema":
                                            breakout_ok = ema_ok
                                        else:
                                            rh_i = rolling_high_series.iloc[i] if rolling_high_series is not None and i < len(rolling_high_series) else None
                                            high_break = rh_i is not None and not pd.isna(rh_i) and current_price > float(rh_i)
                                            breakout_ok = ema_ok and high_break
                                        if not breakout_ok:
                                            entries_blocked_by_regime_vol_breakout += 1
                                            entries_blocked_by_regime += 1
                                            continue
                                        entries_allowed_on_decompress += 1
                                    # else: non-compress and not decompress_event -> allow as usual
                                elif regime_rule == "flat_slope" and ema_slope_series is not None and i < len(ema_slope_series):
                                    slope_i = ema_slope_series.iloc[i]
                                    if pd.isna(slope_i) or abs(float(slope_i)) <= slope_threshold:
                                        entries_blocked_by_regime_flat_slope += 1
                                        entries_blocked_by_regime += 1
                                        continue
                                elif regime_rule == "downtrend_block" and ema_series is not None and ema_slope_series is not None and i < len(ema_series) and i < len(ema_slope_series):
                                    ema_i = ema_series.iloc[i]
                                    slope_i = ema_slope_series.iloc[i]
                                    below_ema = pd.isna(ema_i) or current_price <= float(ema_i)
                                    slope_non_positive = not pd.isna(slope_i) and float(slope_i) <= 0
                                    is_downtrend = below_ema and slope_non_positive
                                    if is_downtrend:
                                        entries_blocked_by_regime_downtrend += 1
                                        entries_blocked_by_regime += 1
                                        continue
                                elif regime_rule == "proba_quantile" and proba_long_arr is not None and q_series is not None and i < len(proba_long_arr) and i < len(q_series):
                                    p_i = float(proba_long_arr[i])
                                    q_th_i = q_series.iloc[i]
                                    if pd.isna(q_th_i):
                                        gate = p_floor
                                    else:
                                        gate = max(p_floor, float(q_th_i))
                                    if p_i < gate:
                                        entries_blocked_by_regime_proba_quantile += 1
                                        entries_blocked_by_regime += 1
                                        continue
                            
                            # [정밀 분석] ENTRY 시점 final_scale 및 stage2_cap 수집 (entry_allowed일 때만)
                            if entry_allowed:
                                final_scale_at_entry.append(final_scale)
                                guard_scale_at_entry.append(guard_scale)  # Guard v2 scale 수집
                                if use_stage2:
                                    stage2_cap_at_entry.append(stage2_cap)
                            
                            # [정밀 분석] ENTRY 시점 샘플 수집 (최대 10개, Stage-2 ON일 때만, v2.2 cap 정보)
                            if use_stage2 and len(stage2_score_samples) < 10:
                                sample = {
                                    "idx": i,
                                    "ts": str(row_timestamp),
                                    "signal": signal,
                                    "stage2_trade": stage2_trade,
                                    "stage2_reason": stage2_reason,
                                    "p_diff": stage2_cap_debug_info.get("p_diff") if isinstance(stage2_cap_debug_info, dict) else None,
                                    "entropy": stage2_cap_debug_info.get("entropy") if isinstance(stage2_cap_debug_info, dict) else None,
                                    "stage2_cap": stage2_cap,
                                    "cap_reason": stage2_cap_debug_info.get("cap_reason", "N/A") if isinstance(stage2_cap_debug_info, dict) else "N/A",
                                    "guard_scale": guard_scale,  # 위에서 계산한 guard_scale 사용
                                    "final_scale": final_scale,
                                }
                                stage2_score_samples.append(sample)
                            
                            # [Position scaling] Proba-based scale (LONG only); combine with final_scale
                            effective_position_scale = final_scale
                            if position_scaling_enabled and direction == "LONG":
                                p_for_scale = float(p_long) if p_long is not None else 0.0
                                proba_scale = _position_scale_from_proba(
                                    p_for_scale,
                                    position_scaling_mode,
                                    position_p_floor,
                                    position_p_full,
                                    position_size_min,
                                    position_size_max,
                                    position_p_mid,
                                    position_k,
                                )
                                entries_scaled_count += 1
                                bin_key = _scale_bin_key(proba_scale)
                                scale_bins_counts[bin_key] = scale_bins_counts.get(bin_key, 0) + 1
                                if proba_scale <= 0.0:
                                    # 방어적 no-op: entry skip, 통계만 기록
                                    continue
                                entries_scaled_applied_count += 1
                                scale_sum += proba_scale
                                scale_sq_sum += proba_scale * proba_scale
                                scale_min_val = min(scale_min_val, proba_scale)
                                scale_max_val = max(scale_max_val, proba_scale)
                                effective_position_scale = final_scale * proba_scale
                            
                            # trade_id 할당 (scale<=0 continue 이후이므로 실제 진입만 ID 부여)
                            trade_id = next_trade_id
                            next_trade_id += 1
                            
                            if entry_proba_gap_val is not None:
                                spread_executed_list.append(entry_proba_gap_val)
                            entry_dir_proba_val: float | None = None
                            if direction == "LONG" and proba_long_arr is not None and i < len(proba_long_arr):
                                entry_dir_proba_val = float(proba_long_arr[i])
                            elif direction == "SHORT" and proba_short_arr is not None and i < len(proba_short_arr):
                                entry_dir_proba_val = float(proba_short_arr[i])
                            position = {
                                "side": direction,  # direction 사용 (signal과 동일해야 함)
                                "entry_price": current_price,
                                "entry_time": str(row_timestamp),
                                "entry_dir_proba": entry_dir_proba_val,
                                "position_scale": effective_position_scale,  # Guard v2 * Stage-2 * proba_scale(if enabled)
                                "stage2_cap": stage2_cap if use_stage2 else 1.0,  # CAP breakdown용
                                "final_scale": final_scale,  # CAP breakdown용
                                "guard_scale": guard_scale,  # Guard v2 breakdown용
                                "entry_idx": i,  # CAP breakdown용
                                "trade_id": trade_id,  # ENTRY/EXIT 연결용
                                "partial_tp_done": False,  # Phase C7: 1 trade당 1회 부분 익절
                                "remaining_ratio": 1.0,  # Phase C7: 부분 익절 후 잔여 비율
                                "max_unrealized_return": 0.0,  # Phase C9: break-even용
                                "be_stop_triggered": False,  # Phase C9: break-even stop 활성화 여부
                                "max_favorable_excursion": None,  # trade log MFE
                                "max_adverse_excursion": None,  # trade log MAE
                                "entry_top1_proba": entry_top1_val,
                                "entry_top2_proba": entry_top2_val,
                                "entry_proba_gap": entry_proba_gap_val,
                            }
                            
                            # [MONITOR][ENTRY] 로그 기록
                            cap_reason_str = stage2_cap_debug_info.get("cap_reason", "N/A") if isinstance(stage2_cap_debug_info, dict) else "N/A"
                            guard_components_entry = {
                                "p_long": p_long,
                                "margin": abs(p_long - 0.5) if p_long is not None else None,
                                "entropy": stage2_cap_debug_info.get("entropy") if isinstance(stage2_cap_debug_info, dict) else None,
                            }
                            if guard_v2 is not None:
                                snapshot_entry = guard_v2.debug_snapshot()
                                guard_components_entry.update({
                                    "recent_mean_margin": snapshot_entry.get("recent_mean_margin"),
                                    "recent_mean_entropy": snapshot_entry.get("recent_mean_entropy"),
                                })
                            
                            stage2_cap_val = stage2_cap if use_stage2 else 1.0
                            
                            # position_size null 보강: 백테스트에서는 실제 주문 수량이 없으므로 unavailable로 표시
                            position_size_source = "unavailable"  # 백테스트에서는 실제 주문 수량 없음
                            
                            monitor.log_entry(
                                ts=str(row_timestamp),
                                symbol=self.symbol,
                                timeframe=self.timeframe,
                                bar_index=i,
                                trade_id=trade_id,
                                side=direction,
                                entry_price=current_price,
                                guard_scale=guard_scale,
                                guard_components=guard_components_entry,
                                stage2_cap=stage2_cap_val,
                                cap_reason=cap_reason_str if use_stage2 else None,
                                final_scale=final_scale,
                                position_size=None,  # 실제 주문 수량은 계산되지 않음 (백테스트)
                                position_size_source=position_size_source,
                            )
                            
                            # 전체 bar 기준 분포 집계 (ENTRY 시점에서 stage2_cap과 final_scale 수집)
                            # CHECK 시점에서는 None이었지만, ENTRY 시점에서 실제 값이 계산됨
                            monitor.stage2_cap_all_checks.append(stage2_cap_val)
                            monitor.final_scale_all_checks.append(final_scale)
                            if cap_reason_str and cap_reason_str != "N/A":
                                monitor.cap_reason_all_checks.append(cap_reason_str)
                            
                            # 샘플링된 데이터 수집 (기존 호환성 유지)
                            if use_stage2:
                                monitor.stage2_caps.append(stage2_cap_val)
                            monitor.final_scales.append(final_scale)
                            
                            # ENTRY 시점 CAP link 로그 (direction 정의 이후)
                            if use_stage2 and len(cap_entry_exit_links) < 50:  # 충분히 확보
                                import json
                                import numpy as np
                                
                                def to_python_type(val):
                                    if val is None:
                                        return None
                                    if isinstance(val, (np.integer, np.floating)):
                                        return float(val) if isinstance(val, np.floating) else int(val)
                                    if isinstance(val, np.ndarray):
                                        return val.tolist()
                                    return val
                                
                                p_diff_val = to_python_type(stage2_cap_debug_info.get("p_diff") if isinstance(stage2_cap_debug_info, dict) else None)
                                entropy_val = to_python_type(stage2_cap_debug_info.get("entropy") if isinstance(stage2_cap_debug_info, dict) else None)
                                cap_reason_val = stage2_cap_debug_info.get("cap_reason", "N/A") if isinstance(stage2_cap_debug_info, dict) else "N/A"
                                p_long_float = to_python_type(p_long) if p_long is not None else None
                                margin_val = abs(p_long_float - 0.5) if p_long_float is not None else None
                                
                                entry_link = {
                                    "trade_id": trade_id,
                                    "event": "ENTRY",
                                    "entry_ts": str(row_timestamp),
                                    "entry_price": float(current_price),
                                    "side": direction,
                                    "p_long": p_long_float,
                                    "margin": margin_val,
                                    "entropy": entropy_val,
                                    "p_diff": p_diff_val,
                                    "guard_scale": to_python_type(guard_scale) if guard_scale is not None else None,
                                    "stage2_cap": to_python_type(stage2_cap) if stage2_cap is not None else None,
                                    "final_scale": to_python_type(final_scale) if final_scale is not None else None,
                                    "cap_reason": cap_reason_val,
                                }
                                cap_entry_exit_links.append(entry_link)
                                logger.info(f"[ENTRY][CAPLINK] {json.dumps(entry_link)}")
                            
                            cap_trigger_stats["entries_attempted"] += 1
                            position_entry_bar_index = i
                            
                            if direction == "LONG":
                                entry_count_long += 1
                            else:
                                entry_count_short += 1
                                # SHORT 전략 진입 로그
                                if enable_short_strategy:
                                    logger.info(
                                        f"{self.log_prefix}[SHORT STRATEGY] SHORT entry: idx={i}, ts={row_timestamp}, "
                                        f"price={current_price:.2f}, stage2_trade={stage2_trade}, "
                                        f"guard={guard.state.current_decision if guard is not None else 'N/A'}"
                                    )
                            
                            # Log entry event (with dump fields)
                            # Guard decision: Guard OFF면 빈값, ON이면 실제 판정
                            guard_decision_for_dump = ""
                            if use_strategy_guard:
                                guard_decision_for_dump = guard_decision_by_idx.get(i, "")
                            elif use_strategy_guard_v2:
                                guard_decision_for_dump = guard_v2_decision
                            
                            # Stage-2 trade: Stage-2 OFF면 None, ON이면 실제 값
                            stage2_trade_for_dump = None
                            stage2_reason_for_dump = ""
                            if use_stage2:
                                stage2_trade_for_dump = stage2_trade
                                stage2_reason_for_dump = stage2_reason
                            
                            # entry_cost 계산 (position_scale은 profit에만 적용, 비용은 원본 유지)
                            entry_cost = current_price * (effective_commission_rate + effective_slippage_rate)
                            trade_events.append({
                                "event": trade_event,
                                "idx": i,
                                "ts": str(row_timestamp),
                                "direction": direction,
                                "entry_price": current_price,
                                "fee": entry_cost,
                                "balance_before": balance,
                                # Dump fields
                                "raw_signal": raw_signal,
                                "final_signal": signal,
                                "stage2_trade": stage2_trade_for_dump,
                                "stage2_reason": stage2_reason_for_dump,
                                "guard_decision": guard_decision_for_dump,
                                "trade_index": entry_count_long + entry_count_short,  # 1-based entry count
                            })
                            
                            # Decision 근거 로그 출력 (트레이드 이벤트 시점: ENTRY)
                            if guard is not None:
                                guard.check(trade_index=i, timestamp=str(row_timestamp))
                                snapshot = guard.debug_snapshot()
                                logger.info(
                                    f"[STRATEGY_GUARD][DECISION] "
                                    f"event=ENTRY trade_index={entry_count_long + entry_count_short} idx={i} ts={str(row_timestamp)} "
                                    f"decision={snapshot['decision']} recent_trades_count={snapshot['recent_trades_count']} "
                                    f"recent_trades_window={snapshot['recent_trades_window']} "
                                    f"win_rate={snapshot['win_rate']:.4f} avg_return={snapshot['avg_return']:.6f} "
                                    f"min_win_rate={snapshot['min_win_rate']:.4f} min_avg_return={snapshot['min_avg_return']:.6f} "
                                    f"min_block_trades={snapshot['min_block_trades']} "
                                    f"is_blocked={snapshot['is_blocked']} block_trades_remaining={snapshot['block_trades_remaining']}"
                                )
                            elif guard_v2 is not None:
                                snapshot = guard_v2.debug_snapshot()
                                p_long_str = f"{snapshot['p_long']:.4f}" if snapshot['p_long'] is not None else "N/A"
                                margin_str = f"{snapshot['margin']:.4f}" if snapshot['margin'] is not None else "N/A"
                                entropy_str = f"{snapshot['entropy']:.4f}" if snapshot['entropy'] is not None else "N/A"
                                mean_margin_str = f"{snapshot['recent_mean_margin']:.4f}" if snapshot['recent_mean_margin'] is not None else "N/A"
                                mean_entropy_str = f"{snapshot['recent_mean_entropy']:.4f}" if snapshot['recent_mean_entropy'] is not None else "N/A"
                                final_scale_str = f"{final_scale:.3f}" if 'final_scale' in locals() else f"{snapshot['position_scale']:.3f}"
                                logger.info(
                                    f"[STRATEGY_GUARD][DECISION] "
                                    f"v2_enabled=True event=ENTRY trade_index={entry_count_long + entry_count_short} idx={i} ts={str(row_timestamp)} "
                                    f"decision={snapshot['decision']} scale={snapshot['position_scale']:.3f} "
                                    f"final_scale={final_scale_str} "
                                    f"p_long={p_long_str} margin={margin_str} entropy={entropy_str} "
                                    f"mean_margin={mean_margin_str} mean_entropy={mean_entropy_str} "
                                    f"mode={snapshot['mode']} reason={snapshot['reason']}"
                                )
                        
                        if entry_allowed:
                            logger.debug(
                                f"{self.log_prefix}[EXECUTION] {trade_event}: idx={i} ts={row_timestamp} "
                                f"direction={direction} entry_price={current_price:.2f} fee={entry_cost:.4f} "
                                f"balance={balance:.6f}"
                            )
                elif signal in ("LONG", "SHORT") and (
                    blocked_by_cooldown_flag
                    or blocked_by_ot_flag
                    or blocked_by_regime_ablation_flag
                    or blocked_by_positive_regime_flag
                ):
                    # Entry blocked by cooldown, overtrading, regime ablation, or positive-regime gate
                    pass
        
        # Close remaining position at end (forced EOD close)
        if position is not None:
            last_row = df.iloc[-1]
            forced_exit_price = float(last_row["close"])
            entry_price = position["entry_price"]
            direction = position["side"]
            
            logger.info(
                f"{self.log_prefix} Forced EOD close: position={direction}, "
                f"entry_price={entry_price:.2f}, exit_price={forced_exit_price:.2f}, "
                f"entry_bar={position_entry_bar_index}, exit_bar={len(df)-1}"
            )
            
            entry_cost = entry_price * (effective_commission_rate + effective_slippage_rate)
            exit_cost = forced_exit_price * (effective_commission_rate + effective_slippage_rate)
            
            if direction == "LONG":
                effective_entry = entry_price + entry_cost
                effective_exit = forced_exit_price - exit_cost
                profit = (effective_exit - effective_entry) / effective_entry
            else:
                effective_entry = entry_price - entry_cost
                effective_exit = forced_exit_price + exit_cost
                profit = (effective_entry - effective_exit) / effective_entry
            
            # position_scale 적용 (forced EOD close도 동일하게)
            position_scale = position.get("position_scale", 1.0) if position else 1.0
            scaled_profit = profit * position_scale
            
            balance *= 1 + scaled_profit
            trade = Trade(
                    entry_time=position["entry_time"],
                    exit_time=str(last_row["timestamp"]),
                    entry_price=entry_price,
                    exit_price=forced_exit_price,
                    direction=direction,
                    profit=scaled_profit,  # 실제 반영된 수익률
                )
            trades.append(trade)
            
            # CAP breakdown 기록 (Forced EOD close도 동일하게)
            if use_stage2 and position:
                stage2_cap_val = position.get("stage2_cap", 1.0)
                if abs(stage2_cap_val - 0.6) < 0.001:
                    cap_bucket = 0.6
                elif abs(stage2_cap_val - 0.8) < 0.001:
                    cap_bucket = 0.8
                else:
                    cap_bucket = 1.0
                
                cap_breakdown_records.append({
                    "cap_bucket": cap_bucket,
                    "stage2_cap": stage2_cap_val,
                    "final_scale": position.get("final_scale", 1.0),
                    "scaled_profit": scaled_profit,
                    "bars_held": len(df) - 1 - position_entry_bar_index if position_entry_bar_index is not None else 0,
                    "entry_idx": position.get("entry_idx", position_entry_bar_index if position_entry_bar_index is not None else 0),
                    "exit_idx": len(df) - 1,
                })
            
            equity_curve.append(balance)
            exits_executed += 1
            last_exit_bar_index = len(df) - 1
            last_exit_direction = direction
            last_exit_reason = "eod_forced"
            
            # Update StrategyGuard with completed trade
            if guard is not None and profit is not None:
                guard.update_trade(profit=profit, timestamp=str(last_row["timestamp"]))
                # Decision 근거 로그 출력 (트레이드 이벤트 시점: EXIT forced EOD)
                guard.check(trade_index=len(df) - 1, timestamp=str(last_row["timestamp"]))
                snapshot = guard.debug_snapshot()
                logger.info(
                    f"[STRATEGY_GUARD][DECISION] "
                    f"event=EXIT_EOD trade_index={len(trades)} idx={len(df) - 1} ts={str(last_row['timestamp'])} "
                    f"decision={snapshot['decision']} recent_trades_count={snapshot['recent_trades_count']} "
                    f"recent_trades_window={snapshot['recent_trades_window']} "
                    f"win_rate={snapshot['win_rate']:.4f} avg_return={snapshot['avg_return']:.6f} "
                    f"min_win_rate={snapshot['min_win_rate']:.4f} min_avg_return={snapshot['min_avg_return']:.6f} "
                    f"min_block_trades={snapshot['min_block_trades']} "
                    f"is_blocked={snapshot['is_blocked']} block_trades_remaining={snapshot['block_trades_remaining']}"
                )
        
        # Compute statistics
        from src.backtest.engine import _compute_trade_stats, dedupe_trades_round_trips

        stats = _compute_trade_stats(trades)
        deduped_trades = dedupe_trades_round_trips(trades)
        unique_round_trips = len(deduped_trades)
        duplicate_trade_rows = len(trades) - unique_round_trips
        
        # Calculate max drawdown
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0.0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0.0
        
        # Calculate win rate
        if trades:
            winning_trades = [t for t in trades if t["profit"] is not None and t["profit"] > 0]
            win_rate = len(winning_trades) / len(trades)
        else:
            win_rate = 0.0
        
        total_return = balance - 1.0
        
        # ======================================================================
        # [EXECUTION DEBUG] 트레이드 실행 요약 로그
        # ======================================================================
        logger.info("=" * 60)
        logger.info(f"{self.log_prefix}[EXECUTION DEBUG] Trade Execution Summary")
        logger.info("=" * 60)
        
        # Count trade events by type
        event_counts: dict[str, int] = {}
        total_fees = 0.0
        total_slippage = 0.0
        total_fees_ratio = 0.0  # Track fee as ratio for unit consistency check
        
        for event in trade_events:
            event_type = event["event"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            if "fee" in event:
                total_fees += event["fee"]
                # Calculate fee ratio if entry_price available
                if "entry_price" in event and event["entry_price"] > 0:
                    fee_ratio = event["fee"] / event["entry_price"]
                    total_fees_ratio += fee_ratio
            if "slippage" in event:
                total_slippage += event["slippage"]
        
        # Log summary with special handling for zero trades
        if stats['total_trades'] == 0:
            logger.warning(
                f"{self.log_prefix}[EXECUTION DEBUG] NO TRADES EXECUTED. "
                f"entries_attempted={entries_attempted}, "
                f"stage2_no_trade={stage2_no_trade_count}, "
                f"hysteresis_block={block_reasons.get('hysteresis', 0)}, "
                f"signals may be too restrictive."
            )
        else:
            logger.info(
                f"{self.log_prefix}[EXECUTION DEBUG] Total trades: {stats['total_trades']}, "
                f"entries={entries_attempted} (LONG={entry_count_long}, SHORT={entry_count_short}), "
                f"exits={exits_executed}, tp={tp_exits}, sl={sl_exits}, flips={flip_count}"
            )
            if long_only:
                logger.info(
                    f"{self.log_prefix}[EXECUTION DEBUG] Direction filter: LONG-only mode active. "
                    f"SHORT entries blocked: {block_reasons.get('direction_filter', 0)}"
                )
            elif short_only:
                logger.info(
                    f"{self.log_prefix}[EXECUTION DEBUG] Direction filter: SHORT-only mode active. "
                    f"LONG entries blocked: {block_reasons.get('direction_filter', 0)}"
                )
            if enable_short_strategy:
                logger.info(
                    f"{self.log_prefix}[EXECUTION DEBUG] SHORT Strategy: enabled=True, "
                    f"SHORT entries={entry_count_short}, "
                    f"blocked_by_guard={block_reasons.get('short_strategy_guard', 0)}, "
                    f"blocked_by_stage2={block_reasons.get('short_strategy_stage2', 0)}"
                )
            logger.info(
                f"{self.log_prefix}[EXECUTION DEBUG] Performance: "
                f"total_return={total_return:.4f}, win_rate={win_rate:.4f}, "
                f"max_drawdown={max_dd:.4f}"
            )
            logger.info(
                f"{self.log_prefix}[EXECUTION DEBUG] Costs: "
                f"total_fees={total_fees:.6f} (절대값 누적, price 단위), "
                f"total_fees_ratio={total_fees_ratio:.6f} (비율 누적), "
                f"total_slippage={total_slippage:.6f}"
            )
            logger.info(
                f"{self.log_prefix}[EXECUTION DEBUG] ⚠️  Fee Unit Check: "
                f"total_fees는 price 기반 절대값입니다. "
                f"balance=1.0 구조와 단위 불일치 가능성 있음."
            )
            logger.info(
                f"{self.log_prefix}[EXECUTION DEBUG] Trade events: {event_counts}"
            )
            
            # Calculate average holding period
            if trade_events:
                holding_periods = [
                    e.get("bars_held", 0) for e in trade_events
                    if "bars_held" in e and e["bars_held"] > 0
                ]
                if holding_periods:
                    avg_holding = sum(holding_periods) / len(holding_periods)
                    logger.info(
                        f"{self.log_prefix}[EXECUTION DEBUG] Average holding period: {avg_holding:.1f} bars"
                    )
        
        # ======================================================================
        # [ANTI-OVERTRADING] 검증 로그 (거래가 0개여도 출력)
        # ======================================================================
        logger.info("=" * 60)
        logger.info(f"{self.log_prefix}[ANTI-OVERTRADING] Anti-Overtrading Statistics")
        logger.info("=" * 60)
        logger.info(
            f"{self.log_prefix}[ANTI-OVERTRADING] Entry attempts: {entries_attempted} "
            f"(LONG={entry_count_long}, SHORT={entry_count_short})"
        )
        logger.info(
            f"{self.log_prefix}[ANTI-OVERTRADING] Exit count: {exit_count}"
        )
        logger.info(
            f"{self.log_prefix}[ANTI-OVERTRADING] Flip count: {flip_count} "
            f"(LONG↔SHORT transitions)"
        )
        logger.info(
            f"{self.log_prefix}[ANTI-OVERTRADING] Block reasons: {block_reasons}"
        )
        
        # Stage-2 statistics
        if stage2_no_trade_count > 0 or stage2_trade_count > 0:
            logger.info(
                f"{self.log_prefix}[ANTI-OVERTRADING] Stage-2: "
                f"trade={stage2_trade_count}, no_trade={stage2_no_trade_count}, "
                f"exit_on_flat={stage2_exit_on_flat_count}"
            )
        
        # Fee decay estimate
        roundtrip_fee = (effective_commission_rate + effective_slippage_rate) * 2  # Entry + Exit
        if stats['total_trades'] > 0:
            # Exponential decay approximation
            fee_decay_estimate_exp = np.exp(-roundtrip_fee * stats['total_trades'])
            # Compound decay approximation
            fee_decay_estimate_compound = (1 - roundtrip_fee) ** stats['total_trades']
            
            logger.info(
                f"{self.log_prefix}[ANTI-OVERTRADING] Fee decay estimate: "
                f"roundtrip_fee={roundtrip_fee:.6f} ({roundtrip_fee*100:.4f}%), "
                f"trades={stats['total_trades']}, "
                f"exp_decay={fee_decay_estimate_exp:.6f}, "
                f"compound_decay={fee_decay_estimate_compound:.6f}"
            )
            logger.info(
                f"{self.log_prefix}[ANTI-OVERTRADING] ⚠️  If fee_decay < 0.1, "
                f"costs alone would reduce balance to <10% of initial. "
                f"Overtrading is likely the main issue."
            )
        elif entries_attempted > 0:
            logger.info(
                f"{self.log_prefix}[ANTI-OVERTRADING] No trades executed despite {entries_attempted} entry attempts. "
                f"Block reasons: {block_reasons}"
            )
        
        # ======================================================================
        # [TRADE COUNT SUMMARY] 트레이드 수 집계 검증 로그
        # ======================================================================
        logger.info("=" * 60)
        logger.info(f"{self.log_prefix}[TRADE COUNT SUMMARY] Trade Count Verification")
        logger.info("=" * 60)
        logger.info(
            f"{self.log_prefix}[TRADE COUNT SUMMARY] total_trades={stats['total_trades']} "
            f"(round-trip 기준, ENTRY+EXIT 완료된 트레이드 수)"
        )
        logger.info(
            f"{self.log_prefix}[TRADE COUNT SUMMARY] total_entries={entry_count_long + entry_count_short} "
            f"(LONG={entry_count_long}, SHORT={entry_count_short})"
        )
        logger.info(
            f"{self.log_prefix}[TRADE COUNT SUMMARY] total_exits={exits_executed} "
            f"(signal_exit={exits_executed - tp_exits - sl_exits - profit_lock_tp_exits - profit_lock_sl_exits}, "
            f"tp={tp_exits}, sl={sl_exits}, profit_lock_tp={profit_lock_tp_exits}, profit_lock_sl={profit_lock_sl_exits})"
        )
        logger.info("=" * 60)
        
        # ======================================================================
        # [STRATEGY GUARD] 통계 로깅
        # ======================================================================
        if guard is not None:
            guard_stats = guard.get_stats()
            guard_snapshot = guard.debug_snapshot()
            logger.info("=" * 60)
            logger.info(f"{self.log_prefix}[STRATEGY GUARD] StrategyGuard Statistics")
            logger.info("=" * 60)
            logger.info(
                f"{self.log_prefix}[STRATEGY GUARD] Total checks: {guard_stats['total_checks']}, "
                f"ALLOW={guard_stats['allow_count']}, BLOCK={guard_stats['block_count']}"
            )
            logger.info(
                f"{self.log_prefix}[STRATEGY GUARD] Current decision: {guard_stats['current_decision']}"
            )
            if guard_stats['block_reason']:
                logger.info(
                    f"{self.log_prefix}[STRATEGY GUARD] Last block reason: {guard_stats['block_reason']}"
            )
            logger.info(
                f"{self.log_prefix}[STRATEGY GUARD] Recent trades tracked: {guard_stats['recent_trades_count']}, "
                f"recent signals tracked: {guard_stats['recent_signals_count']}"
            )
            logger.info("=" * 60)
            
            # Guard 요약 로그 (집계 검증용)
            max_recent_trades_count = guard_stats['recent_trades_count']
            logger.info(
                f"[STRATEGY_GUARD][SUMMARY] total_trades={stats['total_trades']} "
                f"recent_trades_window={guard_snapshot['recent_trades_window']} "
                f"max_recent_trades_count={max_recent_trades_count}"
            )
        
        logger.info("=" * 60)
        
        result = BacktestResult(
            total_return=total_return,
            win_rate=win_rate,
            max_drawdown=max_dd,
            trades=trades,
            equity_curve=equity_curve if equity_curve else [1.0],
            total_trades=stats["total_trades"],
            long_trades=stats["long_trades"],
            short_trades=stats["short_trades"],
            avg_profit=stats["avg_profit"],
            median_profit=stats["median_profit"],
            avg_win=stats["avg_win"],
            avg_loss=stats["avg_loss"],
            max_consecutive_wins=stats["max_consecutive_wins"],
            max_consecutive_losses=stats["max_consecutive_losses"],
        )
        result["unique_round_trips"] = unique_round_trips
        result["duplicate_trade_rows"] = duplicate_trade_rows
        result["signal_exit_th_count"] = signal_exit_th_trade_count
        result["signal_exit_th_total_profit"] = signal_exit_th_scaled_profit_sum
        result["signal_exit_th_loss_aux_applied"] = signal_exit_th_loss_aux_applied

        # Add Stage-2 statistics (optional fields)
        result["stage2_no_trade_count"] = stage2_no_trade_count
        result["stage2_trade_count"] = stage2_trade_count
        result["stage2_exit_on_flat_count"] = stage2_exit_on_flat_count
        result["block_reasons"] = block_reasons
        result["filters_applied"] = {
            "min_max_proba": min_max_proba,
            "max_entropy": max_entropy,
            "min_proba_gap": min_proba_gap,
            "min_directional_gap": min_directional_gap,
            "max_flat_entry_proba": max_flat_entry_proba,
            "min_hold": effective_min_hold_bars,
            "cooldown": effective_cooldown_bars,
        }
        result["filter_skip_stats"] = {
            "by_min_max_proba": skipped_by_min_max_proba,
            "by_max_entropy": skipped_by_max_entropy,
            "by_proba_gap": skipped_by_proba_gap,
            "by_directional_gap": directional_gap_fail_count,
            "by_flat_suppression": flat_suppression_fail_count,
        }
        result["directional_gap_fail_count"] = directional_gap_fail_count
        result["flat_suppression_fail_count"] = flat_suppression_fail_count
        result["long_directional_gap_fail_count"] = long_directional_gap_fail_count
        result["short_directional_gap_fail_count"] = short_directional_gap_fail_count
        result["filter_block_breakdown"] = {
            "by_min_max_proba": skipped_by_min_max_proba,
            "by_max_entropy": skipped_by_max_entropy,
            "by_proba_gap": skipped_by_proba_gap,
            "by_directional_gap": directional_gap_fail_count,
            "by_flat_suppression": flat_suppression_fail_count,
            "long_directional_gap_fail": long_directional_gap_fail_count,
            "short_directional_gap_fail": short_directional_gap_fail_count,
        }
        result["filtered_trade_count"] = skipped_by_proba_gap
        total_entry_attempts = entries_attempted
        result["filtered_trade_ratio"] = (skipped_by_proba_gap / total_entry_attempts) if total_entry_attempts and total_entry_attempts > 0 else 0.0
        result["mean_spread_of_executed_trades"] = float(sum(spread_executed_list) / len(spread_executed_list)) if spread_executed_list else None
        result["mean_spread_of_rejected_trades"] = float(sum(spread_rejected_list) / len(spread_rejected_list)) if spread_rejected_list else None

        # Regime filter (EMA200) 통계
        result["regime_enabled"] = regime_filter_enabled
        result["regime_span"] = regime_ema_span
        result["regime_rule"] = regime_rule if regime_filter_enabled else None
        result["regime_slope_lookback"] = regime_slope_lookback if regime_filter_enabled and regime_rule in ("ema_only", "ema_plus_slope", "vol_slope", "flat_slope", "downtrend_block") else None
        result["vol_window"] = vol_window if regime_filter_enabled and regime_rule in ("vol_compress", "vol_slope", "vol_breakout") else None
        result["vol_threshold"] = vol_threshold if regime_filter_enabled and regime_rule in ("vol_compress", "vol_slope", "vol_breakout") else None
        result["breakout_lookback"] = breakout_lookback if regime_filter_enabled and regime_rule == "vol_breakout" else None
        result["breakout_mode"] = breakout_mode if regime_filter_enabled and regime_rule == "vol_breakout" else None
        result["slope_threshold"] = slope_threshold if regime_filter_enabled and regime_rule == "flat_slope" else None
        result["q_window"] = q_window if regime_filter_enabled and regime_rule == "proba_quantile" else None
        result["q"] = q if regime_filter_enabled and regime_rule == "proba_quantile" else None
        result["p_floor"] = p_floor if regime_filter_enabled and regime_rule == "proba_quantile" else None
        result["q_threshold_mean"] = float(q_series.dropna().mean()) if regime_filter_enabled and regime_rule == "proba_quantile" and q_series is not None and q_series.notna().any() else None
        result["entry_price_basis"] = "close" if regime_filter_enabled else None
        result["ema_align_basis"] = "same_bar" if regime_filter_enabled and regime_rule in ("ema_only", "ema_plus_slope") else None
        if regime_filter_enabled and ema_series is not None and len(ema_series) == len(df):
            pct_above = float((df["close"] > ema_series).sum() / len(df))
            result["pct_above_ema200"] = pct_above
        else:
            result["pct_above_ema200"] = None
        if regime_filter_enabled and regime_rule == "vol_compress" and vol_series is not None and len(vol_series) == len(df):
            result["pct_compress"] = float((vol_series <= vol_threshold).sum() / len(df))
        elif regime_filter_enabled and regime_rule == "vol_breakout" and vol_series is not None and len(vol_series) == len(df):
            result["pct_compress"] = float((vol_series < vol_threshold).sum() / len(df))
        else:
            result["pct_compress"] = None
        if regime_filter_enabled and regime_rule == "vol_slope" and vol_series is not None and ema_slope_series is not None and len(vol_series) == len(df) and len(ema_slope_series) == len(df):
            compress = vol_series <= vol_threshold
            slope_positive = ema_slope_series > 0
            block_mask = compress & (~slope_positive)
            result["pct_vol_slope_block"] = float(block_mask.sum() / len(df))
        else:
            result["pct_vol_slope_block"] = None
        if regime_filter_enabled and regime_rule == "flat_slope" and ema_slope_series is not None and len(ema_slope_series) == len(df):
            flat_mask = ema_slope_series.isna() | (ema_slope_series.abs() <= slope_threshold)
            result["pct_flat_slope_block"] = float(flat_mask.sum() / len(df))
        else:
            result["pct_flat_slope_block"] = None
        if regime_filter_enabled and regime_rule == "downtrend_block" and ema_series is not None and ema_slope_series is not None and len(ema_series) == len(df) and len(ema_slope_series) == len(df):
            below_ema = df["close"] <= ema_series
            downtrend_mask = below_ema & (~ema_slope_series.isna()) & (ema_slope_series <= 0)
            result["pct_downtrend_block"] = float(downtrend_mask.sum() / len(df))
        else:
            result["pct_downtrend_block"] = None
        result["entries_blocked_by_regime"] = entries_blocked_by_regime
        result["enable_regime_filter"] = enable_regime_filter
        result["regime_filter_mode"] = regime_filter_mode if enable_regime_filter else "none"
        result["regime_combo_block_keys"] = (
            regime_combo_block_keys
            if enable_regime_filter and regime_filter_mode == "combo_f5_worst_keys"
            else None
        )
        result["entries_blocked_by_regime_ablation"] = entries_blocked_by_regime_ablation
        result["enable_positive_regime_filter"] = enable_positive_regime_filter
        result["positive_regime_filter_mode"] = (
            positive_regime_filter_mode if enable_positive_regime_filter else "none"
        )
        result["entries_blocked_by_positive_regime"] = entries_blocked_by_positive_regime
        result["positive_regime_allowed_entries"] = positive_regime_allowed_entries
        result["positive_regime_blocked_entries"] = positive_regime_blocked_entries
        result["enable_adaptive_positive_regime_filter"] = enable_adaptive_positive_regime_filter
        result["adaptive_positive_regime_mode"] = (
            adaptive_positive_regime_mode if enable_adaptive_positive_regime_filter else "none"
        )
        result["adaptive_positive_gate_on_count"] = adaptive_positive_gate_on_count
        result["entries_blocked_by_adaptive_positive"] = entries_blocked_by_adaptive_positive
        result["adaptive_positive_allowed_entries"] = adaptive_positive_allowed_entries
        result["adaptive_positive_blocked_entries"] = adaptive_positive_blocked_entries
        result["enable_confidence_filter"] = enable_confidence_filter
        result["confidence_filter_mode"] = confidence_filter_mode if enable_confidence_filter else "none"
        result["entries_blocked_by_confidence"] = entries_blocked_by_confidence
        result["entries_blocked_by_regime_price"] = entries_blocked_by_regime_price if regime_filter_enabled else None
        result["entries_blocked_by_regime_slope"] = entries_blocked_by_regime_slope if regime_filter_enabled else None
        result["entries_blocked_by_regime_vol"] = entries_blocked_by_regime_vol if regime_filter_enabled else None
        result["entries_blocked_by_regime_vol_slope"] = entries_blocked_by_regime_vol_slope if regime_filter_enabled else None
        result["entries_blocked_by_regime_vol_breakout"] = entries_blocked_by_regime_vol_breakout if regime_filter_enabled else None
        result["entries_allowed_on_decompress"] = entries_allowed_on_decompress if regime_filter_enabled else None
        result["entries_blocked_by_regime_flat_slope"] = entries_blocked_by_regime_flat_slope if regime_filter_enabled else None
        result["entries_blocked_by_regime_downtrend"] = entries_blocked_by_regime_downtrend if regime_filter_enabled else None
        result["entries_blocked_by_regime_proba_quantile"] = entries_blocked_by_regime_proba_quantile if regime_filter_enabled else None
        result["entry_flat_gate_enabled"] = entry_flat_gate_enabled
        result["entries_blocked_by_flat_gate"] = entries_blocked_by_flat_gate if entry_flat_gate_enabled else None
        result["pct_flat_gate_block"] = (entries_blocked_by_flat_gate / entries_attempted) if entry_flat_gate_enabled and entries_attempted > 0 else None
        result["entries_blocked_by_urt_gate"] = (
            None
            if entry_min_unique_round_trips is not None and not entry_urt_gate_active
            else (entries_blocked_by_urt_gate if entry_min_unique_round_trips is not None else None)
        )
        result["pct_urt_gate_block"] = (
            None
            if entry_min_unique_round_trips is not None and not entry_urt_gate_active
            else (
                (entries_blocked_by_urt_gate / entries_attempted)
                if entry_min_unique_round_trips is not None and entries_attempted > 0
                else None
            )
        )
        result["entry_urt_from_state_log"] = (
            entry_urt_from_state_log if entry_min_unique_round_trips is not None else None
        )
        result["entry_urt_gate_active"] = (
            bool(entry_urt_gate_active) if entry_min_unique_round_trips is not None else None
        )
        result["entry_urt_state_log_path"] = (
            str(entry_urt_state_log_resolved)
            if entry_min_unique_round_trips is not None and entry_urt_state_log_resolved is not None
            else None
        )
        result["entry_urt_gate_value_source"] = (
            entry_urt_gate_value_source if entry_min_unique_round_trips is not None else None
        )
        result["entry_urt_gate_block_samples"] = (
            urt_gate_block_samples if entry_min_unique_round_trips is not None else None
        )
        result["max_flat_proba"] = max_flat_proba if entry_flat_gate_enabled else None
        attempted_before_regime = entries_blocked_by_regime + entry_count_long
        result["blocked_ratio"] = (
            entries_blocked_by_regime / attempted_before_regime
            if attempted_before_regime > 0 else 0.0
        )
        result["pct_proba_quantile_block"] = (
            entries_blocked_by_regime_proba_quantile / attempted_before_regime
            if regime_filter_enabled and regime_rule == "proba_quantile" and attempted_before_regime > 0 else None
        )
        # Position scaling (proba-based) result keys
        result["position_scaling"] = position_scaling_mode if position_scaling_enabled else "off"
        result["position_p_floor"] = position_p_floor if position_scaling_enabled else None
        result["position_p_full"] = position_p_full if position_scaling_enabled else None
        result["position_size_min"] = position_size_min if position_scaling_enabled else None
        result["position_size_max"] = position_size_max if position_scaling_enabled else None
        result["position_p_mid"] = position_p_mid if position_scaling_enabled and position_scaling_mode == "sigmoid" else None
        result["position_k"] = position_k if position_scaling_enabled and position_scaling_mode == "sigmoid" else None
        result["entries_scaled_count"] = entries_scaled_count if position_scaling_enabled else None
        result["entries_scaled_applied_count"] = entries_scaled_applied_count if position_scaling_enabled else None
        result["scale_mean"] = (scale_sum / entries_scaled_applied_count) if position_scaling_enabled and entries_scaled_applied_count > 0 else None
        result["scale_min"] = scale_min_val if position_scaling_enabled and entries_scaled_applied_count > 0 else None
        result["scale_max"] = scale_max_val if position_scaling_enabled and entries_scaled_applied_count > 0 else None
        result["scale_bins"] = dict(scale_bins_counts) if position_scaling_enabled else None
        # Early exit stats
        result["early_exit_enabled"] = early_exit_enabled
        result["early_exit_count"] = early_exit_count if early_exit_enabled else None
        num_exits = len(trades)
        result["early_exit_rate"] = (early_exit_count / num_exits) if early_exit_enabled and num_exits > 0 else None
        result["early_exit_lookback"] = early_exit_lookback if early_exit_enabled else None
        result["early_exit_p_floor"] = early_exit_p_floor if early_exit_enabled else None
        result["early_exit_bad_k"] = early_exit_bad_k if early_exit_enabled else None
        result["flat_exit_trigger_count"] = flat_exit_trigger_count if (early_exit_enabled and flat_exit_aware_enabled) else None
        result["pct_flat_exit_adjusted"] = (flat_exit_adjusted_count / total_exit_evaluations) if (early_exit_enabled and flat_exit_aware_enabled and total_exit_evaluations > 0) else None
        result["time_stop_enabled"] = time_stop_enabled
        result["time_stop_bars"] = time_stop_bars if time_stop_enabled else None
        result["time_stop_exit_count"] = time_stop_exit_count if time_stop_enabled else None
        result["pct_time_stop_exits"] = (time_stop_exit_count / num_exits) if time_stop_enabled and num_exits > 0 else None

        # Partial take-profit (Phase C7): 타입 정규화 (docs/phase_c7_summary.md)
        # partial_tp=on: count(int>=0), pct(float), avg_pnl(float or 0.0). partial_tp=off: count=0, pct=0.0, avg_pnl=None
        result["partial_tp_enabled"] = partial_tp_enabled
        result["partial_tp_threshold"] = partial_tp_threshold if partial_tp_enabled else None
        result["partial_tp_ratio"] = partial_tp_ratio if partial_tp_enabled else None
        result["partial_tp_count"] = partial_tp_count  # 항상 int (off 시 0)
        result["pct_partial_tp"] = (partial_tp_count / num_exits) if num_exits > 0 else 0.0  # 항상 float (off 시 0.0)
        result["partial_tp_avg_pnl"] = (
            (sum(partial_tp_pnl_list) / len(partial_tp_pnl_list))
            if partial_tp_pnl_list
            else (0.0 if partial_tp_enabled else None)  # on & count==0 -> 0.0, off -> None
        )

        # Break-even stop (Phase C9)
        result["break_even_stop_enabled"] = break_even_stop_enabled
        result["be_threshold"] = be_threshold if break_even_stop_enabled else None
        result["break_even_exit_count"] = break_even_exit_count if break_even_stop_enabled else None
        result["pct_break_even_exits"] = (
            (break_even_exit_count / num_exits) if break_even_stop_enabled and num_exits > 0 else None
        )
        result["profit_lock_enabled"] = enable_profit_lock_exit
        result["profit_lock_take_profit"] = profit_lock_take_profit if enable_profit_lock_exit else None
        result["profit_lock_stop_loss"] = profit_lock_stop_loss if enable_profit_lock_exit else None
        result["profit_lock_max_bars"] = profit_lock_max_bars if enable_profit_lock_exit else None
        result["profit_lock_conservative_intrabar"] = (
            profit_lock_conservative_intrabar if enable_profit_lock_exit else None
        )
        result["profit_lock_tp_exits"] = profit_lock_tp_exits
        result["profit_lock_sl_exits"] = profit_lock_sl_exits
        result["overtrading_control"] = {
            "enable_reentry_cooldown": enable_reentry_cooldown,
            "reentry_cooldown_bars": reentry_cooldown_bars if enable_reentry_cooldown else None,
            "enable_directional_cooldown": enable_directional_cooldown,
            "directional_cooldown_bars": directional_cooldown_bars if enable_directional_cooldown else None,
            "enable_post_tp_block": enable_post_tp_block,
            "post_tp_block_bars": post_tp_block_bars if enable_post_tp_block else None,
            "enable_post_sl_block": enable_post_sl_block,
            "post_sl_block_bars": post_sl_block_bars if enable_post_sl_block else None,
            "blocked_entry_global": blocked_by_ot_global,
            "blocked_entry_directional": blocked_by_ot_directional,
            "blocked_entry_post_tp": blocked_by_ot_post_tp,
            "blocked_entry_post_sl": blocked_by_ot_post_sl,
        }
        if emit_trade_log:
            result["trade_events"] = trade_events

        # Exit reason breakdown: time_stop / early_exit / break_even / normal_exit (상호배타)
        exit_reason_counts: dict[str, int] = {"time_stop": 0, "early_exit": 0, "break_even": 0, "normal_exit": 0}
        exit_reason_pnls: dict[str, list[float]] = {"time_stop": [], "early_exit": [], "break_even": [], "normal_exit": []}
        for event in trade_events:
            if event.get("event", "").startswith("EXIT") and "exit_reason" in event:
                reason = event.get("exit_reason") or "normal_exit"
                if reason == "time_stop":
                    bucket = "time_stop"
                elif reason == "early_exit":
                    bucket = "early_exit"
                elif reason == "break_even":
                    bucket = "break_even"
                else:
                    bucket = "normal_exit"
                exit_reason_counts[bucket] = exit_reason_counts.get(bucket, 0) + 1
                if "profit" in event and event["profit"] is not None:
                    exit_reason_pnls.setdefault(bucket, []).append(float(event["profit"]))
        result["exit_reason_counts"] = exit_reason_counts
        result["exit_reason_avg_pnl"] = {
            k: (sum(v) / len(v) if v else None) for k, v in exit_reason_pnls.items()
        }

        # ======================================================================
        # [정밀 분석] 통계 출력 (요청 1-3)
        # ======================================================================
        logger.info("=" * 80)
        logger.info(f"{self.log_prefix}[PRECISION ANALYSIS] Stage-2 v2.1 정밀 분석 통계")
        logger.info("=" * 80)
        
        # [요청 1] final_scale 분포 (ENTRY 시점 기준)
        if final_scale_at_entry:
            final_scale_sorted = sorted(final_scale_at_entry)
            n = len(final_scale_sorted)
            logger.info(f"[요청 1] Final Scale 분포 (ENTRY 시점 기준, n={n}):")
            logger.info(f"  final_scale_min: {min(final_scale_sorted):.6f}")
            logger.info(f"  final_scale_median: {final_scale_sorted[int(n * 0.5)]:.6f}")
            logger.info(f"  final_scale_p90: {final_scale_sorted[int(n * 0.9)]:.6f}")
            logger.info(f"  final_scale_max: {max(final_scale_sorted):.6f}")
            final_scale_mean = sum(final_scale_sorted) / n
            logger.info(f"  final_scale_mean: {final_scale_mean:.6f}")
            # v2.0(OFF) 대비 mean 변화도 출력 (참고: v2.0 mean=0.106913)
            v20_mean = 0.106913
            logger.info(f"  v2.0(OFF) 대비 mean 변화: {final_scale_mean:.6f} vs {v20_mean:.6f} (차이: {final_scale_mean - v20_mean:+.6f}, 비율: {final_scale_mean / v20_mean * 100:.1f}%)")
            
            result["final_scale_stats"] = {
                "count": n,
                "min": min(final_scale_sorted),
                "median": final_scale_sorted[int(n * 0.5)],
                "p90": final_scale_sorted[int(n * 0.9)],
                "max": max(final_scale_sorted),
                "mean": sum(final_scale_sorted) / n,
            }
        else:
            logger.info("[요청 1] Final Scale 분포: ENTRY가 없어 통계 없음")
            result["final_scale_stats"] = None
        
        # [요청 2] Stage-2 차단/완화 통계
        logger.info(f"[요청 2] Stage-2 차단/완화 통계:")
        logger.info(f"  stage2_hard_block_count: {stage2_hard_block_count} (cap==0.0으로 ENTRY 차단)")
        logger.info(f"  stage2_soft_gated_count: {stage2_soft_gated_count} (cap<1.0이지만 ENTRY 허용)")
        logger.info(f"  stage2_total_entries: {len(final_scale_at_entry)} (실제 ENTRY 발생)")
        logger.info(f"  entries_attempted: {entries_attempted} (ENTRY 시도 총 횟수)")
        logger.info(f"  stage2_cap_applied_count: {stage2_cap_applied_count} (v2.2: cap<1.0인 entry 수)")
        logger.info(f"  stage2_entry_blocked_by_scale_count: {stage2_entry_blocked_by_scale_count} (v2.2: final_scale 기반 ENTRY 차단)")
        
        result["stage2_hard_block_count"] = stage2_hard_block_count
        result["stage2_soft_gated_count"] = stage2_soft_gated_count
        result["stage2_total_entries"] = len(final_scale_at_entry)
        result["entries_attempted"] = entries_attempted
        result["stage2_cap_applied_count"] = stage2_cap_applied_count
        result["stage2_entry_blocked_by_scale_count"] = stage2_entry_blocked_by_scale_count
        
        # [요청 2-1] CAP 분포 (ENTRY 시점 기준)
        if stage2_cap_at_entry:
            cap_1_0_count = sum(1 for cap in stage2_cap_at_entry if abs(cap - 1.0) < 0.001)
            cap_0_8_count = sum(1 for cap in stage2_cap_at_entry if abs(cap - 0.8) < 0.001)
            cap_0_6_count = sum(1 for cap in stage2_cap_at_entry if abs(cap - 0.6) < 0.001)
            total_cap_entries = len(stage2_cap_at_entry)
            logger.info(f"[요청 2-1] CAP 분포 (ENTRY 시점 기준, n={total_cap_entries}):")
            logger.info(f"  cap=1.0: {cap_1_0_count} ({cap_1_0_count * 100 / total_cap_entries:.1f}%)")
            logger.info(f"  cap=0.8: {cap_0_8_count} ({cap_0_8_count * 100 / total_cap_entries:.1f}%)")
            logger.info(f"  cap=0.6: {cap_0_6_count} ({cap_0_6_count * 100 / total_cap_entries:.1f}%)")
            
            result["cap_distribution"] = {
                "total_entries": total_cap_entries,
                "cap_1_0": cap_1_0_count,
                "cap_1_0_pct": cap_1_0_count * 100 / total_cap_entries if total_cap_entries > 0 else 0.0,
                "cap_0_8": cap_0_8_count,
                "cap_0_8_pct": cap_0_8_count * 100 / total_cap_entries if total_cap_entries > 0 else 0.0,
                "cap_0_6": cap_0_6_count,
                "cap_0_6_pct": cap_0_6_count * 100 / total_cap_entries if total_cap_entries > 0 else 0.0,
            }
        else:
            logger.info("[요청 2-1] CAP 분포: ENTRY가 없어 통계 없음")
            result["cap_distribution"] = None
        
        # [요청 3] Stage-2 v2.1 스코어 계산 근거 샘플 (10개)
        logger.info(f"[요청 3] Stage-2 v2.1 스코어 계산 근거 샘플 (n={len(stage2_score_samples)}):")
        for idx, sample in enumerate(stage2_score_samples, 1):
            logger.info(f"  샘플 {idx}:")
            logger.info(f"    idx={sample['idx']}, ts={sample['ts']}, signal={sample['signal']}")
            logger.info(f"    stage2_trade={sample['stage2_trade']}, stage2_reason={sample['stage2_reason']}")
            logger.info(f"    파싱된 입력값: p_diff={sample.get('p_diff', 'N/A')}, entropy={sample.get('entropy', 'N/A')}")
            logger.info(f"    stage2_cap={sample.get('stage2_cap', 'N/A')}, cap_reason={sample.get('cap_reason', 'N/A')}, "
                       f"guard_scale={sample.get('guard_scale', 'N/A')}, final_scale={sample.get('final_scale', 'N/A')}")
        
        result["stage2_score_samples"] = stage2_score_samples
        
        # [DEBUG] CAP 로그 기반 확정용 요약 출력
        logger.info("=" * 80)
        logger.info(f"{self.log_prefix}[CAP DEBUG SUMMARY] CAP 로그 기반 확정 요약")
        logger.info("=" * 80)
        logger.info(f"  total_checks: {cap_trigger_stats['total_checks']}")
        logger.info(f"  entries_attempted: {cap_trigger_stats['entries_attempted']}")
        logger.info(f"  entries_executed: {cap_trigger_stats['entries_executed']}")
        if cap_trigger_stats['total_checks'] > 0:
            logger.info(f"  cap_1_0_count: {cap_trigger_stats['cap_1_0_count']} ({cap_trigger_stats['cap_1_0_count'] * 100 / cap_trigger_stats['total_checks']:.1f}%)")
            logger.info(f"  cap_0_8_count: {cap_trigger_stats['cap_0_8_count']} ({cap_trigger_stats['cap_0_8_count'] * 100 / cap_trigger_stats['total_checks']:.1f}%)")
            logger.info(f"  cap_0_6_count: {cap_trigger_stats['cap_0_6_count']} ({cap_trigger_stats['cap_0_6_count'] * 100 / cap_trigger_stats['total_checks']:.1f}%)")
        logger.info(f"  triggered_by_entropy_only: {cap_trigger_stats['triggered_by_entropy_only']}")
        logger.info(f"  triggered_by_pdiff_only: {cap_trigger_stats['triggered_by_pdiff_only']}")
        logger.info(f"  triggered_by_both: {cap_trigger_stats['triggered_by_both']}")
        logger.info(f"  triggered_by_none: {cap_trigger_stats['triggered_by_none']}")
        logger.info(f"  cap_debug_logs_count: {len(cap_debug_logs)}")
        logger.info(f"  cap_entry_exit_links_count: {len([l for l in cap_entry_exit_links if l.get('event') == 'ENTRY'])}")
        logger.info("=" * 80)
        
        # result에 추가
        result["cap_debug_logs"] = cap_debug_logs
        result["cap_entry_exit_links"] = cap_entry_exit_links
        result["cap_trigger_stats"] = cap_trigger_stats
        
        # Guard v2 통계 추가
        if use_strategy_guard_v2:
            result["guard_v2_scale_at_entry"] = guard_scale_at_entry
            result["guard_v2_scale_all_checks"] = guard_scale_all_checks
            result["guard_v2_entry_exit_links"] = guard_entry_exit_links
            
            # Guard v2 scale 분포 요약
            if guard_scale_all_checks:
                import numpy as np
                scales = np.array(guard_scale_all_checks)
                logger.info(f"[GUARD][SCALE] 분포 요약 (전체 checks, n={len(guard_scale_all_checks)}):")
                logger.info(f"  mean: {np.mean(scales):.4f}, median: {np.median(scales):.4f}")
                logger.info(f"  p10: {np.percentile(scales, 10):.4f}, p25: {np.percentile(scales, 25):.4f}")
                logger.info(f"  p75: {np.percentile(scales, 75):.4f}, p90: {np.percentile(scales, 90):.4f}")
                logger.info(f"  min: {np.min(scales):.4f}, max: {np.max(scales):.4f}")
            
            if guard_scale_at_entry:
                scales_entry = np.array(guard_scale_at_entry)
                logger.info(f"[GUARD][SCALE] ENTRY 시점 분포 (n={len(guard_scale_at_entry)}):")
                logger.info(f"  mean: {np.mean(scales_entry):.4f}, median: {np.median(scales_entry):.4f}")
                logger.info(f"  min: {np.min(scales_entry):.4f}, max: {np.max(scales_entry):.4f}")
        
        # [요청 4] CAP 값별 성적 분해 (ENTRY 기준 cap_bucket별 집계)
        if cap_breakdown_records:
            breakdown_stats = {}
            for cap_bucket in [1.0, 0.8, 0.6]:
                bucket_records = [r for r in cap_breakdown_records if abs(r["cap_bucket"] - cap_bucket) < 0.001]
                if bucket_records:
                    profits = [r["scaled_profit"] for r in bucket_records]
                    holdings = [r["bars_held"] for r in bucket_records]
                    final_scales = [r["final_scale"] for r in bucket_records]
                    
                    wins = [p for p in profits if p > 0]
                    losses = [p for p in profits if p < 0]
                    
                    breakdown_stats[cap_bucket] = {
                        "trade_count": len(bucket_records),
                        "win_rate": len(wins) / len(bucket_records) if bucket_records else 0.0,
                        "mean_profit": sum(profits) / len(profits),
                        "median_profit": sorted(profits)[len(profits) // 2] if profits else 0.0,
                        "mean_holding_bars": sum(holdings) / len(holdings) if holdings else 0.0,
                        "total_contribution": sum(profits),  # sum(scaled_profit)
                        "final_scale_mean": sum(final_scales) / len(final_scales) if final_scales else 0.0,
                        "final_scale_median": sorted(final_scales)[len(final_scales) // 2] if final_scales else 0.0,
                        "final_scale_p90": sorted(final_scales)[int(len(final_scales) * 0.9)] if final_scales and len(final_scales) > 0 else 0.0,
                    }
                else:
                    breakdown_stats[cap_bucket] = None
            
            logger.info(f"[요청 4] CAP 값별 성적 분해 (ENTRY 기준, n={len(cap_breakdown_records)}):")
            for cap_bucket in [1.0, 0.8, 0.6]:
                stats = breakdown_stats.get(cap_bucket)
                if stats:
                    logger.info(f"  cap={cap_bucket}: trades={stats['trade_count']}, "
                              f"win_rate={stats['win_rate']:.2%}, "
                              f"mean_profit={stats['mean_profit']:.6f}, "
                              f"median_profit={stats['median_profit']:.6f}, "
                              f"mean_holding={stats['mean_holding_bars']:.1f}, "
                              f"total_contribution={stats['total_contribution']:.6f}")
                else:
                    logger.info(f"  cap={cap_bucket}: 해당 구간에 발생하지 않음")
            
            result["cap_breakdown"] = breakdown_stats
        else:
            logger.info("[요청 4] CAP 값별 성적 분해: 기록 없음")
            result["cap_breakdown"] = None
        
        logger.info("=" * 80)
        
        # Add StrategyGuard statistics (Phase-2: UNBLOCK + 히스테리시스)
        if guard is not None:
            guard_stats = guard.get_stats()
            result["strategy_guard_stats"] = {
                "total_checks": guard_stats["total_checks"],
                "block_count": guard_stats["block_count"],
                "unblock_count": guard_stats["unblock_count"],
                "allow_count": guard_stats["allow_count"],
                "current_decision": guard_stats["current_decision"],
                "block_reason": guard_stats["block_reason"],
                "unblock_reason": guard_stats["unblock_reason"],
                "trades_since_block": guard_stats["trades_since_block"],
                "state_history": guard_stats["state_history"],
            }
        
        # Add SHORT Strategy statistics (MVP)
        if enable_short_strategy:
            short_trades_count = stats.get("short_trades", 0)
            result["short_strategy_enabled"] = True
            result["short_trades_count"] = short_trades_count
            logger.info(
                f"{self.log_prefix}[SHORT STRATEGY] SHORT Strategy Summary: "
                f"enabled=True, short_trades={short_trades_count}, "
                f"blocked_by_guard={block_reasons.get('short_strategy_guard', 0)}, "
                f"blocked_by_stage2={block_reasons.get('short_strategy_stage2', 0)}"
            )
        else:
            result["short_strategy_enabled"] = False
            result["short_trades_count"] = 0
        
        # ======================================================================
        # [TRADE DUMP] CSV 파일로 트레이드 이벤트 덤프
        # ======================================================================
        if dump_trades_path is not None:
            import csv
            import os
            
            # 디렉토리 생성
            dump_dir = os.path.dirname(dump_trades_path)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            
            # CSV 헤더 정의
            fieldnames = [
                "event_type",
                "trade_index",
                "idx",
                "timestamp",
                "direction",
                "price",
                "raw_signal",
                "final_signal",
                "stage2_trade",
                "stage2_reason",
                "guard_decision",
                "exit_reason",
                "pnl",
            ]
            
            # CSV 파일 작성
            with open(dump_trades_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for event in trade_events:
                    # event_type 결정
                    event_type = event.get("event", "")
                    
                    # price 결정 (ENTRY는 entry_price, EXIT는 exit_price)
                    if "entry_price" in event and event["entry_price"] is not None:
                        price = event["entry_price"]
                    elif "exit_price" in event and event["exit_price"] is not None:
                        price = event["exit_price"]
                    else:
                        price = ""
                    
                    # pnl 결정 (EXIT 이벤트에만)
                    pnl = event.get("profit", "") if "profit" in event and event.get("profit") is not None else ""
                    
                    # exit_reason 결정 (EXIT 이벤트에만)
                    exit_reason = event.get("exit_reason", "") if "exit_reason" in event else ""
                    
                    # stage2_trade를 문자열로 변환 (True/False -> "True"/"False", None이면 빈값)
                    stage2_trade_str = ""
                    if "stage2_trade" in event:
                        stage2_trade_val = event["stage2_trade"]
                        if stage2_trade_val is True:
                            stage2_trade_str = "True"
                        elif stage2_trade_val is False:
                            stage2_trade_str = "False"
                        # None이면 빈값 유지
                    
                    # stage2_reason (Stage-2 ON일 때만 값이 있음)
                    stage2_reason_str = event.get("stage2_reason", "")
                    
                    row = {
                        "event_type": event_type,
                        "trade_index": event.get("trade_index", ""),
                        "idx": event.get("idx", ""),
                        "timestamp": event.get("ts", ""),
                        "direction": event.get("direction", ""),
                        "price": price,
                        "raw_signal": event.get("raw_signal", ""),
                        "final_signal": event.get("final_signal", ""),
                        "stage2_trade": stage2_trade_str,
                        "stage2_reason": stage2_reason_str,
                        "guard_decision": event.get("guard_decision", ""),
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                    }
                    writer.writerow(row)
            
            logger.info(
                f"{self.log_prefix}[TRADE DUMP] Dumped {len(trade_events)} trade events to {dump_trades_path}"
            )
        
        # ======================================================================
        # [실전 모니터링] 집계 요약 저장 및 로거 종료
        # ======================================================================
        total_checks = len(guard_scale_all_checks) if use_strategy_guard_v2 else len(df)
        entries_attempted = cap_trigger_stats.get("entries_attempted", 0)
        entries_executed = cap_trigger_stats.get("entries_executed", 0)
        
        # stats 계산 (result 생성 전이므로 직접 계산)
        from src.backtest.engine import _compute_trade_stats
        stats = _compute_trade_stats(trades)
        
        # Config snapshot 구성 (실행 시점 실제 사용 값)
        config_snapshot = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": monitor.mode,
            "run_id": monitor.run_id,
            "stage2": {
                "use_stage2": use_stage2,
                "cap_entropy_high_th": stage2_cap_entropy_high_th,
                "cap_entropy_mid_th": stage2_cap_entropy_mid_th,
                "cap_pdiff_tiny_th": stage2_cap_pdiff_tiny_th,
                "cap_pdiff_small_th": stage2_cap_pdiff_small_th,
                "cap_low": 0.6,  # 하드코딩된 값 (calculate_stage2_cap 기본값)
                "cap_mid": 0.8,  # 하드코딩된 값
                "cap_default": 1.0,  # 하드코딩된 값
                "block_if_final_scale_below": stage2_block_if_final_scale_below,
            },
            "anti_overtrading": {
                "min_hold_bars": min_hold_bars,
                "cooldown_bars": cooldown_bars,
                "min_hold_effective": effective_min_hold_bars,
                "cooldown_effective": effective_cooldown_bars,
            },
            "guard_v2": {
                "use_strategy_guard_v2": use_strategy_guard_v2,
                "mode": strategy_guard_v2_mode if use_strategy_guard_v2 else None,
                "scale_floor": strategy_guard_v2_scale_floor if use_strategy_guard_v2 else None,
                "block_if_scale_below": strategy_guard_v2_block_if_scale_below if use_strategy_guard_v2 else None,
                "min_margin": strategy_guard_v2_min_margin if use_strategy_guard_v2 else None,
                "max_entropy": strategy_guard_v2_max_entropy if use_strategy_guard_v2 else None,
                "window_signal_stats": strategy_guard_v2_window_signal_stats if use_strategy_guard_v2 else None,
            },
        }
        
        # 백테스트 성과 지표 (summary JSON 및 Discord 리포트용)
        equity_curve = result.get("equity_curve") if isinstance(result, dict) else getattr(result, "equity_curve", None)
        final_equity = float(equity_curve[-1]) if equity_curve and len(equity_curve) > 0 else None
        monitor.save_summary(
            total_checks=total_checks,
            entries_attempted=entries_attempted,
            entries_executed=entries_executed,
            total_trades=stats["total_trades"],
            blocked_by_min_hold=blocked_by_min_hold,
            blocked_by_cooldown=blocked_by_cooldown,
            blocked_by_guard_hard=blocked_by_guard_hard,
            config_snapshot=config_snapshot,
            total_return=result.get("total_return") if isinstance(result, dict) else getattr(result, "total_return", None),
            max_drawdown=result.get("max_drawdown") if isinstance(result, dict) else getattr(result, "max_drawdown", None),
            win_rate=result.get("win_rate") if isinstance(result, dict) else getattr(result, "win_rate", None),
            avg_profit=result.get("avg_profit") if isinstance(result, dict) else getattr(result, "avg_profit", None),
            equity_final=final_equity,
        )
        monitor.close()
        
        logger.info(f"[MONITOR] 모니터링 로그 저장 완료: {monitor.log_file}")
        logger.info(f"[MONITOR] 요약 스냅샷 저장 완료: {monitor.summary_file}")
        
        return result
    
    def _should_exit_position(self, position: dict, signal: Signal) -> Optional[str]:
        """
        Determine if position should be exited based on signal.
        
        Default implementation: exit if signal is HOLD or opposite direction.
        Can be overridden for strategy-specific logic.
        
        Args:
            position: Current position dict
            signal: Current signal
        
        Returns:
            Exit reason string or None
        """
        position_side = position["side"]
        
        if signal == "HOLD":
            return "signal_hold"
        elif signal != position_side:
            return "signal_opposite"
        
        return None
    
    @abstractmethod
    def run_backtest(
        self,
        long_threshold: float,
        short_threshold: Optional[float],
        use_optimized_threshold: bool = False,
        proba_long_cache: Optional[np.ndarray] = None,
        proba_short_cache: Optional[np.ndarray] = None,
        df_with_proba: Optional[pd.DataFrame] = None,
        index_mask: Optional[np.ndarray] = None,
        commission_rate: Optional[float] = None,
        slippage_rate: Optional[float] = None,
        long_only: bool = False,
        short_only: bool = False,
        signal_confirmation_bars: int = 1,
        use_trend_filter: bool = False,
        trend_ema_window: int = 200,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        max_holding_bars: Optional[int] = None,
        use_confidence_filter: bool = False,
        confidence_quantile: float = 0.85,
        daily_loss_limit: Optional[float] = None,
        flat_threshold: Optional[float] = None,
        confidence_margin: float = 0.0,
        min_proba_dominance: float = 0.0,
    ) -> BacktestResult:
        """
        Run complete backtest.
        
        Args:
            long_threshold: Threshold for LONG signals
            short_threshold: Threshold for SHORT signals
            use_optimized_threshold: Whether to load optimized thresholds
            proba_long_cache: Optional pre-computed LONG probabilities
            proba_short_cache: Optional pre-computed SHORT probabilities
            df_with_proba: Optional DataFrame aligned with probabilities
            index_mask: Optional boolean mask for in-sample/out-of-sample splits
            commission_rate: Override commission rate
            slippage_rate: Override slippage rate
            long_only: If True, only execute LONG trades
            short_only: If True, only execute SHORT trades
            signal_confirmation_bars: Number of consecutive bars for signal confirmation
            use_trend_filter: Whether to apply EMA trend filter
            trend_ema_window: EMA window for trend filter
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            max_holding_bars: Maximum bars to hold a position
            use_confidence_filter: Whether to use confidence filter
            confidence_quantile: Confidence quantile threshold
            daily_loss_limit: Daily loss limit (kill switch)
        
        Returns:
            BacktestResult
        """
        pass

