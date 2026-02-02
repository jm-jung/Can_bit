"""
StrategyGuard: 전략 실행 허용/차단 관리 (MVP)

StrategyGuard는 전략 로직을 변경하지 않고, 실행만 허용/차단합니다.

Guard v2: 신호 품질/불확실성 기반 판단 (SOFT 모드: position_scale 지원)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

GuardDecision = Literal["ALLOW", "BLOCK", "COOLING", "DEFER"]


@dataclass
class StrategyGuardConfig:
    """StrategyGuard 설정 (Phase-2: UNBLOCK + 히스테리시스 추가)"""
    
    # 최근 N 트레이드 기준
    recent_trades_window: int = 20  # 최근 20개 트레이드 기준
    
    # 표본 부족 정책 (insufficient sample policy)
    insufficient_sample_policy: Literal["allow", "block", "defer"] = "allow"  # 표본 부족 시 정책
    
    # BLOCK 임계값 (기존 MVP)
    min_win_rate: float = 0.4  # win_rate < 0.4면 BLOCK
    min_avg_return: float = -0.01  # avg_return < -0.01 (즉, -1%)면 BLOCK
    
    # UNBLOCK 임계값 (Phase-2 추가)
    unblock_win_rate: float = 0.45  # win_rate >= 0.45면 UNBLOCK
    unblock_avg_return: float = 0.0  # avg_return >= 0.0면 UNBLOCK
    
    # 히스테리시스 (진동 방지)
    min_block_trades: int = 5  # BLOCK 상태 최소 유지 트레이드 수
    
    # Stage-2 pass rate 임계값 (선택)
    min_stage2_pass_rate: float = 0.3  # stage2_pass_rate < 0.3이면 BLOCK (선택)
    use_stage2_check: bool = False  # Stage-2 체크 활성화 여부


@dataclass
class StrategyGuardState:
    """StrategyGuard 상태 추적 (Phase-2: UNBLOCK + 히스테리시스)"""
    
    # 최근 트레이드 기록
    recent_trades: list[dict]  # {"profit": float, "timestamp": str}
    
    # 최근 신호 기록 (Stage-2 통과 여부)
    recent_signals: list[dict]  # {"stage2_trade": bool, "timestamp": str}
    
    # 현재 상태
    current_decision: GuardDecision = "ALLOW"
    block_reason: str = ""
    unblock_reason: str = ""
    
    # 히스테리시스 추적
    last_block_trade_index: int | None = None  # 마지막 BLOCK 발생 시 트레이드 인덱스
    trades_since_block: int = 0  # BLOCK 이후 트레이드 수
    
    # 통계
    total_checks: int = 0
    block_count: int = 0
    unblock_count: int = 0
    allow_count: int = 0
    
    # 상태 전이 이력
    state_history: list[dict] = None  # [{"decision": "BLOCK", "reason": "...", "trade_index": 123, "timestamp": "..."}, ...]
    
    def __post_init__(self):
        """초기화 후 state_history를 빈 리스트로 설정"""
        if self.state_history is None:
            self.state_history = []


class StrategyGuard:
    """
    전략 실행 허용/차단 관리 (MVP)
    
    최근 트레이드 성능과 신호 품질을 기반으로 실행을 허용/차단합니다.
    """
    
    def __init__(self, config: StrategyGuardConfig | None = None):
        """
        Initialize StrategyGuard.
        
        Args:
            config: Guard 설정 (None이면 기본값 사용)
        """
        self.config = config if config is not None else StrategyGuardConfig()
        self.state = StrategyGuardState(
            recent_trades=[],
            recent_signals=[],
        )
        self.log_prefix = "[StrategyGuard]"
    
    def update_trade(self, profit: float, timestamp: str | None = None) -> None:
        """
        트레이드 완료 시 호출하여 상태 업데이트.
        
        Args:
            profit: 트레이드 수익률 (ratio)
            timestamp: 트레이드 타임스탬프 (선택)
        """
        self.state.recent_trades.append({
            "profit": profit,
            "timestamp": timestamp or "",
        })
        
        # 윈도우 크기 제한
        if len(self.state.recent_trades) > self.config.recent_trades_window * 2:
            # 최근 N개만 유지
            self.state.recent_trades = self.state.recent_trades[-self.config.recent_trades_window:]
    
    def update_signal(self, stage2_trade: bool, timestamp: str | None = None) -> None:
        """
        신호 생성 시 호출하여 Stage-2 통과 여부 기록.
        
        Args:
            stage2_trade: Stage-2에서 Trade=True인지 여부
            timestamp: 신호 타임스탬프 (선택)
        """
        if not self.config.use_stage2_check:
            return
        
        self.state.recent_signals.append({
            "stage2_trade": stage2_trade,
            "timestamp": timestamp or "",
        })
        
        # 윈도우 크기 제한
        if len(self.state.recent_signals) > self.config.recent_trades_window * 2:
            self.state.recent_signals = self.state.recent_signals[-self.config.recent_trades_window:]
    
    def check(self, trade_index: int | None = None, timestamp: str | None = None) -> GuardDecision:
        """
        현재 상태를 기반으로 ALLOW/BLOCK 판정 (Phase-2: UNBLOCK + 히스테리시스).
        
        Args:
            trade_index: 현재 트레이드 인덱스 (히스테리시스 추적용)
            timestamp: 현재 타임스탬프 (상태 전이 이력용)
        
        Returns:
            "ALLOW", "BLOCK", 또는 "COOLING"
        """
        self.state.total_checks += 1
        
        # 최근 트레이드가 충분하지 않으면 표본 부족 정책 적용
        if len(self.state.recent_trades) < self.config.recent_trades_window:
            recent_trades_count = len(self.state.recent_trades)
            
            if self.config.insufficient_sample_policy == "allow":
                # 기존 동작: ALLOW
                if self.state.current_decision != "ALLOW":
                    self._log_state_transition("ALLOW", "insufficient_trades", trade_index, timestamp)
                    self.state.current_decision = "ALLOW"
                    self.state.last_block_trade_index = None
                    self.state.trades_since_block = 0
                self.state.allow_count += 1
                return "ALLOW"
            elif self.config.insufficient_sample_policy == "block":
                # BLOCK 정책: 표본 부족 시 BLOCK
                if self.state.current_decision != "BLOCK":
                    self._log_state_transition("BLOCK", f"insufficient_trades (count={recent_trades_count} < window={self.config.recent_trades_window})", trade_index, timestamp)
                    self.state.current_decision = "BLOCK"
                    self.state.block_reason = f"INSUFFICIENT_SAMPLE (count={recent_trades_count} < window={self.config.recent_trades_window})"
                    self.state.block_count += 1
                    self.state.last_block_trade_index = trade_index
                    self.state.trades_since_block = 0
                return "BLOCK"
            elif self.config.insufficient_sample_policy == "defer":
                # DEFER 정책: 판단 스킵
                if self.state.current_decision != "DEFER":
                    self._log_state_transition("DEFER", f"insufficient_trades (count={recent_trades_count} < window={self.config.recent_trades_window})", trade_index, timestamp)
                    self.state.current_decision = "DEFER"
                # DEFER는 별도 카운터가 없으므로 allow_count에 포함하지 않음
                return "DEFER"
            else:
                # 기본값은 allow
                if self.state.current_decision != "ALLOW":
                    self._log_state_transition("ALLOW", "insufficient_trades", trade_index, timestamp)
                    self.state.current_decision = "ALLOW"
                    self.state.last_block_trade_index = None
                    self.state.trades_since_block = 0
                self.state.allow_count += 1
                return "ALLOW"
        
        # 최근 N개 트레이드 성능 계산
        recent = self.state.recent_trades[-self.config.recent_trades_window:]
        profits = [t["profit"] for t in recent if t["profit"] is not None]
        
        if not profits:
            # profit 정보가 없으면 ALLOW
            if self.state.current_decision != "ALLOW":
                self._log_state_transition("ALLOW", "no_profit_data", trade_index, timestamp)
                self.state.current_decision = "ALLOW"
                self.state.last_block_trade_index = None
                self.state.trades_since_block = 0
            self.state.allow_count += 1
            return "ALLOW"
        
        # Win rate 계산
        wins = [p for p in profits if p > 0]
        win_rate = len(wins) / len(profits) if profits else 0.0
        
        # Average return 계산
        avg_return = sum(profits) / len(profits) if profits else 0.0
        
        # 현재 상태에 따른 분기
        if self.state.current_decision == "BLOCK":
            # BLOCK 상태: UNBLOCK 조건 체크 (히스테리시스 포함)
            decision = self._check_unblock(win_rate, avg_return, trade_index, timestamp)
        else:
            # ALLOW 상태: BLOCK 조건 체크
            decision = self._check_block(win_rate, avg_return, recent, trade_index, timestamp)
        
        return decision
    
    def _check_block(
        self,
        win_rate: float,
        avg_return: float,
        recent: list[dict],
        trade_index: int | None,
        timestamp: str | None,
    ) -> GuardDecision:
        """ALLOW 상태에서 BLOCK 조건 체크"""
        block_reasons = []
        
        if win_rate < self.config.min_win_rate:
            block_reasons.append(f"win_rate={win_rate:.3f}<{self.config.min_win_rate:.3f}")
        
        if avg_return < self.config.min_avg_return:
            block_reasons.append(f"avg_return={avg_return:.4f}<{self.config.min_avg_return:.4f}")
        
        # Stage-2 pass rate 체크 (선택)
        if self.config.use_stage2_check and self.state.recent_signals:
            recent_signals = self.state.recent_signals[-self.config.recent_trades_window:]
            stage2_passed = sum(1 for s in recent_signals if s["stage2_trade"])
            stage2_pass_rate = stage2_passed / len(recent_signals) if recent_signals else 0.0
            
            if stage2_pass_rate < self.config.min_stage2_pass_rate:
                block_reasons.append(
                    f"stage2_pass_rate={stage2_pass_rate:.3f}<{self.config.min_stage2_pass_rate:.3f}"
                )
        
        if block_reasons:
            # BLOCK 전이
            self.state.current_decision = "BLOCK"
            self.state.block_reason = "; ".join(block_reasons)
            self.state.block_count += 1
            self.state.last_block_trade_index = trade_index
            self.state.trades_since_block = 0
            
            self._log_state_transition("BLOCK", self.state.block_reason, trade_index, timestamp)
            
            logger.warning(
                f"{self.log_prefix} BLOCK decision: {self.state.block_reason} "
                f"(recent_trades={len(recent)}, win_rate={win_rate:.3f}, avg_return={avg_return:.4f})"
            )
            return "BLOCK"
        else:
            # ALLOW 유지
            self.state.current_decision = "ALLOW"
            self.state.allow_count += 1
            return "ALLOW"
    
    def _check_unblock(
        self,
        win_rate: float,
        avg_return: float,
        trade_index: int | None,
        timestamp: str | None,
    ) -> GuardDecision:
        """BLOCK 상태에서 UNBLOCK 조건 체크 (히스테리시스 포함)"""
        # 히스테리시스: 최소 유지 트레이드 수 체크
        # 주의: trades_since_block은 execute_trades()에서 트레이드 완료 시(update_trade() 호출 후) 증가하므로,
        # 여기서는 현재 값을 체크만 수행
        
        # 최소 유지 트레이드 수 미달 시 BLOCK 유지
        if self.state.trades_since_block < self.config.min_block_trades:
            return "BLOCK"
        
        # UNBLOCK 조건 체크
        unblock_reasons = []
        
        if win_rate >= self.config.unblock_win_rate:
            unblock_reasons.append(f"win_rate={win_rate:.3f}>={self.config.unblock_win_rate:.3f}")
        
        if avg_return >= self.config.unblock_avg_return:
            unblock_reasons.append(f"avg_return={avg_return:.4f}>={self.config.unblock_avg_return:.4f}")
        
        # UNBLOCK 조건 모두 만족 시
        if len(unblock_reasons) >= 2:  # win_rate와 avg_return 모두 만족
            # UNBLOCK 전이
            self.state.current_decision = "ALLOW"
            self.state.unblock_reason = "; ".join(unblock_reasons)
            self.state.unblock_count += 1
            self.state.last_block_trade_index = None
            self.state.trades_since_block = 0
            
            self._log_state_transition("ALLOW", f"UNBLOCK: {self.state.unblock_reason}", trade_index, timestamp)
            
            logger.info(
                f"{self.log_prefix} UNBLOCK decision: {self.state.unblock_reason} "
                f"(win_rate={win_rate:.3f}, avg_return={avg_return:.4f}, "
                f"trades_since_block={self.state.trades_since_block}, min_block_trades={self.config.min_block_trades})"
            )
            return "ALLOW"
        else:
            # BLOCK 유지
            return "BLOCK"
    
    def debug_snapshot(self) -> dict:
        """
        현재 Guard 상태를 읽기 전용으로 스냅샷 반환 (트레이드 이벤트 시점 로깅용).
        
        Returns:
            dict with keys: decision, recent_trades_count, recent_trades_window, win_rate, avg_return,
            min_win_rate, min_avg_return, min_block_trades, is_blocked, block_trades_remaining
        """
        recent_trades_count = len(self.state.recent_trades)
        
        # 최근 트레이드가 충분하지 않으면 기본값 반환
        if recent_trades_count < self.config.recent_trades_window:
            return {
                "decision": self.state.current_decision,
                "recent_trades_count": recent_trades_count,
                "recent_trades_window": self.config.recent_trades_window,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "min_win_rate": self.config.min_win_rate,
                "min_avg_return": self.config.min_avg_return,
                "min_block_trades": self.config.min_block_trades,
                "is_blocked": (self.state.current_decision == "BLOCK"),
                "block_trades_remaining": max(0, self.config.min_block_trades - self.state.trades_since_block) if self.state.current_decision == "BLOCK" else 0,
            }
        
        # 최근 N개 트레이드 성능 계산
        recent = self.state.recent_trades[-self.config.recent_trades_window:]
        profits = [t["profit"] for t in recent if t["profit"] is not None]
        
        if not profits:
            return {
                "decision": self.state.current_decision,
                "recent_trades_count": recent_trades_count,
                "recent_trades_window": self.config.recent_trades_window,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "min_win_rate": self.config.min_win_rate,
                "min_avg_return": self.config.min_avg_return,
                "min_block_trades": self.config.min_block_trades,
                "is_blocked": (self.state.current_decision == "BLOCK"),
                "block_trades_remaining": max(0, self.config.min_block_trades - self.state.trades_since_block) if self.state.current_decision == "BLOCK" else 0,
            }
        
        # Win rate 계산
        wins = [p for p in profits if p > 0]
        win_rate = len(wins) / len(profits) if profits else 0.0
        
        # Average return 계산
        avg_return = sum(profits) / len(profits) if profits else 0.0
        
        return {
            "decision": self.state.current_decision,
            "recent_trades_count": recent_trades_count,
            "recent_trades_window": self.config.recent_trades_window,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "min_win_rate": self.config.min_win_rate,
            "min_avg_return": self.config.min_avg_return,
            "min_block_trades": self.config.min_block_trades,
            "is_blocked": (self.state.current_decision == "BLOCK"),
            "block_trades_remaining": max(0, self.config.min_block_trades - self.state.trades_since_block) if self.state.current_decision == "BLOCK" else 0,
        }
    
    def _log_state_transition(
        self,
        new_decision: GuardDecision,
        reason: str,
        trade_index: int | None,
        timestamp: str | None,
    ) -> None:
        """상태 전이 이력 기록"""
        self.state.state_history.append({
            "decision": new_decision,
            "reason": reason,
            "trade_index": trade_index,
            "timestamp": timestamp or "",
        })
    
    def get_stats(self) -> dict:
        """
        Guard 통계 반환 (Phase-2: UNBLOCK + 히스테리시스).
        
        Returns:
            통계 딕셔너리
        """
        return {
            "total_checks": self.state.total_checks,
            "block_count": self.state.block_count,
            "unblock_count": self.state.unblock_count,
            "allow_count": self.state.allow_count,
            "current_decision": self.state.current_decision,
            "block_reason": self.state.block_reason,
            "unblock_reason": self.state.unblock_reason,
            "recent_trades_count": len(self.state.recent_trades),
            "recent_signals_count": len(self.state.recent_signals),
            "trades_since_block": self.state.trades_since_block,
            "state_history": self.state.state_history,
        }


# ============================================================================
# StrategyGuard v2: 신호 품질/불확실성 기반 판단
# ============================================================================

@dataclass
class StrategyGuardV2Config:
    """StrategyGuard v2 설정: 신호 품질/불확실성 기반"""
    
    # v2 활성화
    enable_v2: bool = False
    
    # 모드: soft (position_scale) 또는 hard (ALLOW/BLOCK)
    mode: Literal["soft", "hard"] = "soft"
    
    # 신호 통계 윈도우 (bars)
    window_signal_stats: int = 200
    
    # 임계값
    min_margin: float = 0.02  # 최소 결정 마진 (|p_long - 0.5| 또는 p_long - threshold)
    max_entropy: float = 0.65  # 최대 엔트로피 (0~0.693, 0.693은 완전 불확실)
    
    # position_scale 관련
    scale_floor: float = 0.2  # 최소 스케일 (0~1)
    block_if_scale_below: float = 0.05  # hard 모드: scale이 이 값보다 낮으면 BLOCK


@dataclass
class StrategyGuardV2State:
    """StrategyGuard v2 상태 추적"""
    
    # 최근 신호 기록 (p_long, margin, entropy)
    recent_signals: list[dict]  # {"p_long": float, "margin": float, "entropy": float, "timestamp": str}
    
    # 현재 상태
    current_scale: float = 1.0  # 현재 position_scale (0~1)
    current_decision: GuardDecision = "ALLOW"
    last_reason: str = ""


class StrategyGuardV2:
    """
    StrategyGuard v2: 신호 품질/불확실성 기반 판단
    
    - 트레이드 결과 누적이 아닌 매 시그널 시점에 계산 가능한 입력으로 판단
    - SOFT 모드: position_scale (0.0~1.0) 반환
    - HARD 모드: ALLOW/BLOCK 반환 (기존 호환)
    """
    
    def __init__(self, config: StrategyGuardV2Config | None = None):
        """
        Initialize StrategyGuardV2.
        
        Args:
            config: Guard v2 설정 (None이면 기본값 사용)
        """
        self.config = config if config is not None else StrategyGuardV2Config()
        self.state = StrategyGuardV2State(
            recent_signals=[],
        )
        self.log_prefix = "[StrategyGuardV2]"
    
    def update_signal(
        self,
        p_long: float,
        threshold: float | None = None,
        timestamp: str | None = None,
    ) -> None:
        """
        신호 발생 시 호출하여 신호 품질 기록.
        
        Args:
            p_long: 모델이 롱일 확률 (0~1)
            threshold: 롱 신호 임계값 (None이면 0.5 사용)
            timestamp: 신호 타임스탬프 (선택)
        """
        if threshold is None:
            threshold = 0.5
        
        # 결정 마진 계산: threshold와 분리하여 "확신(quality)"로 정의
        # margin = abs(p_long - 0.5) (항상 0.5 기준으로 확신 측정)
        # threshold는 signal/entry gate 용도로만 사용, scale 계산에서는 margin을 0.5 기준으로 사용
        margin = abs(p_long - 0.5)
        
        # 엔트로피 계산: -p*log(p) - (1-p)*log(1-p)
        # p=0 또는 p=1일 때는 0, p=0.5일 때 최대값 0.693
        if p_long <= 0.0 or p_long >= 1.0:
            entropy = 0.0
        else:
            entropy = -(p_long * math.log(p_long) + (1.0 - p_long) * math.log(1.0 - p_long))
        
        self.state.recent_signals.append({
            "p_long": p_long,
            "margin": margin,
            "entropy": entropy,
            "timestamp": timestamp or "",
        })
        
        # 윈도우 크기 제한
        if len(self.state.recent_signals) > self.config.window_signal_stats * 2:
            self.state.recent_signals = self.state.recent_signals[-self.config.window_signal_stats:]
    
    def check(
        self,
        p_long: float,
        threshold: float | None = None,
        signal: str | None = None,
        trade_index: int | None = None,
        timestamp: str | None = None,
    ) -> tuple[GuardDecision, float]:
        """
        현재 신호 품질을 기반으로 판정.
        
        Args:
            p_long: 모델이 롱일 확률 (0~1)
            threshold: 롱 신호 임계값 (None이면 0.5 사용)
            signal: 현재 신호 ("LONG", "SHORT", "HOLD")
            trade_index: 현재 트레이드 인덱스 (로깅용)
            timestamp: 현재 타임스탬프 (로깅용)
        
        Returns:
            (decision, position_scale) 튜플
            - decision: "ALLOW" 또는 "BLOCK"
            - position_scale: 0.0~1.0 (SOFT 모드에서 사용)
        """
        if threshold is None:
            threshold = 0.5
        
        # 현재 신호 품질 계산: threshold와 분리하여 "확신(quality)"로 정의
        # margin = abs(p_long - 0.5) (항상 0.5 기준으로 확신 측정)
        # threshold는 signal/entry gate 용도로만 사용, scale 계산에서는 margin을 0.5 기준으로 사용
        margin = abs(p_long - 0.5)
        
        if p_long <= 0.0 or p_long >= 1.0:
            entropy = 0.0
        else:
            entropy = -(p_long * math.log(p_long) + (1.0 - p_long) * math.log(1.0 - p_long))
        
        # 최근 신호 통계 계산
        if len(self.state.recent_signals) >= min(10, self.config.window_signal_stats // 10):
            recent = self.state.recent_signals[-self.config.window_signal_stats:]
            recent_margins = [s["margin"] for s in recent]
            recent_entropies = [s["entropy"] for s in recent]
            mean_margin = sum(recent_margins) / len(recent_margins) if recent_margins else 0.0
            mean_entropy = sum(recent_entropies) / len(recent_entropies) if recent_entropies else 0.0
        else:
            # 표본 부족: 현재 값만 사용
            mean_margin = margin
            mean_entropy = entropy
        
        # position_scale 계산 (SOFT 모드 기본)
        # - margin이 높고 entropy가 낮으면 scale↑
        # - margin이 낮거나 entropy가 높으면 scale↓
        
        # margin 기반 스케일 (0~1)
        margin_scale = min(1.0, max(0.0, (margin - self.config.min_margin) / (0.5 - self.config.min_margin)))
        
        # entropy 기반 스케일 (0~1, entropy가 낮을수록 높음)
        entropy_scale = min(1.0, max(0.0, 1.0 - (entropy / self.config.max_entropy)))
        
        # 최종 scale: margin과 entropy의 가중 평균
        position_scale = (margin_scale * 0.6 + entropy_scale * 0.4)
        
        # scale_floor 적용
        position_scale = max(self.config.scale_floor, position_scale)
        
        # 최근 통계 반영 (가중 평균)
        if len(self.state.recent_signals) >= 10:
            recent_mean_margin_scale = min(1.0, max(0.0, (mean_margin - self.config.min_margin) / (0.5 - self.config.min_margin)))
            recent_mean_entropy_scale = min(1.0, max(0.0, 1.0 - (mean_entropy / self.config.max_entropy)))
            recent_scale = (recent_mean_margin_scale * 0.6 + recent_mean_entropy_scale * 0.4)
            recent_scale = max(self.config.scale_floor, recent_scale)
            
            # 현재와 최근의 가중 평균 (현재 70%, 최근 30%)
            position_scale = position_scale * 0.7 + recent_scale * 0.3
        
        # 모드에 따른 decision 결정
        if self.config.mode == "soft":
            # SOFT 모드: scale 기반으로 decision 결정
            if position_scale < self.config.block_if_scale_below:
                decision = "BLOCK"
                reason = f"scale={position_scale:.3f} < block_threshold={self.config.block_if_scale_below:.3f}"
            else:
                decision = "ALLOW"
                reason = f"scale={position_scale:.3f} (margin={margin:.4f}, entropy={entropy:.4f})"
        else:
            # HARD 모드: margin/entropy 임계값 기반
            if margin < self.config.min_margin or entropy > self.config.max_entropy:
                decision = "BLOCK"
                reason = f"margin={margin:.4f} < {self.config.min_margin:.4f} OR entropy={entropy:.4f} > {self.config.max_entropy:.4f}"
                position_scale = 0.0  # HARD 모드에서는 scale 무시
            else:
                decision = "ALLOW"
                reason = f"margin={margin:.4f} >= {self.config.min_margin:.4f} AND entropy={entropy:.4f} <= {self.config.max_entropy:.4f}"
                position_scale = 1.0  # HARD 모드에서는 scale 무시
        
        self.state.current_scale = position_scale
        self.state.current_decision = decision
        self.state.last_reason = reason
        
        return (decision, position_scale)
    
    def debug_snapshot(self) -> dict:
        """
        현재 Guard v2 상태를 읽기 전용으로 스냅샷 반환.
        
        Returns:
            dict with keys: v2_enabled, mode, p_long, margin, entropy, recent_mean_margin,
            recent_mean_entropy, position_scale, decision, reason
        """
        if not self.state.recent_signals:
            return {
                "v2_enabled": True,
                "mode": self.config.mode,
                "p_long": None,
                "margin": None,
                "entropy": None,
                "recent_mean_margin": None,
                "recent_mean_entropy": None,
                "position_scale": self.state.current_scale,
                "decision": self.state.current_decision,
                "reason": self.state.last_reason,
            }
        
        last_signal = self.state.recent_signals[-1]
        p_long = last_signal["p_long"]
        margin = last_signal["margin"]
        entropy = last_signal["entropy"]
        
        # 최근 통계
        if len(self.state.recent_signals) >= 10:
            recent = self.state.recent_signals[-self.config.window_signal_stats:]
            recent_margins = [s["margin"] for s in recent]
            recent_entropies = [s["entropy"] for s in recent]
            recent_mean_margin = sum(recent_margins) / len(recent_margins) if recent_margins else 0.0
            recent_mean_entropy = sum(recent_entropies) / len(recent_entropies) if recent_entropies else 0.0
        else:
            recent_mean_margin = margin
            recent_mean_entropy = entropy
        
        return {
            "v2_enabled": True,
            "mode": self.config.mode,
            "p_long": p_long,
            "margin": margin,
            "entropy": entropy,
            "recent_mean_margin": recent_mean_margin,
            "recent_mean_entropy": recent_mean_entropy,
            "position_scale": self.state.current_scale,
            "decision": self.state.current_decision,
            "reason": self.state.last_reason,
        }

