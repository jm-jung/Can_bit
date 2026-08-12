"""
Decision Engine: 최근 N일 집계 메트릭으로 실전 후보 / 개선 필요 / 폐기 / 데이터 부족 판정.
우선순위: 데이터 부족 > 폐기 > 실전 후보 > 개선 필요.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.decision.decision_types import DecisionResult, DecisionThresholds
from src.decision.decision_utils import (
    aggregate_returns_for_sharpe,
    normalize_ratio,
    safe_float,
)


def decide(metrics: Dict[str, Any], thresholds: DecisionThresholds) -> DecisionResult:
    """
    metrics: total_trades, total_return, max_drawdown, win_rate, sharpe (선택)
    - total_return / max_drawdown: 비율(0.05=5%) 또는 퍼센트(5) 자동 판별.
    - None/누락 시 해당 조건에서 불리 또는 "metric missing" reasons.
    """
    reasons: List[str] = []
    # 정규화: 비율로 통일
    raw_trades = metrics.get("total_trades") or metrics.get("trades")
    trades = int(safe_float(raw_trades) or 0)
    total_return = normalize_ratio(metrics.get("total_return") or metrics.get("return"))
    mdd_raw = metrics.get("max_drawdown") or metrics.get("mdd")
    mdd = normalize_ratio(mdd_raw)
    if mdd is not None and mdd > 0:
        mdd = -abs(mdd)  # MDD는 보통 음수 또는 양수로 저장될 수 있음; 판정에서는 절대값 사용
    win_rate = safe_float(metrics.get("win_rate"))
    sharpe = safe_float(metrics.get("sharpe"))
    collapse = metrics.get("collapse") is True

    metrics_snapshot = {
        "total_trades": trades,
        "total_return": total_return,
        "max_drawdown": mdd_raw if mdd_raw is not None else mdd,
        "win_rate": win_rate,
        "sharpe": sharpe,
    }

    # A) 데이터 부족(보류)
    if trades < thresholds.min_trades:
        return DecisionResult(
            label="데이터 부족(보류)",
            reasons=[f"trades 부족: {trades} < {thresholds.min_trades}"],
            metrics_snapshot=metrics_snapshot,
            thresholds_used=_thresholds_to_dict(thresholds),
        )

    # 보조: 누락 메트릭 표시
    if total_return is None:
        reasons.append("total_return missing")
    if mdd is None and mdd_raw is None:
        pass  # mdd_raw might be in snapshot
    if win_rate is None:
        reasons.append("win_rate missing")

    # B) 폐기 (mdd는 비율 절대값으로 비교: 0.12 = 12%)
    discard_reasons: List[str] = []
    if collapse:
        discard_reasons.append("collapse 플래그 설정됨")
    mdd_abs = 0.0
    if mdd_raw is not None:
        mdd_val = normalize_ratio(mdd_raw)
        mdd_abs = abs(mdd_val) if mdd_val is not None else 0.0
    elif mdd is not None:
        mdd_abs = abs(mdd)
    if mdd_abs >= thresholds.discard_mdd:
        discard_reasons.append(f"mdd({mdd_abs:.2%}) >= discard_mdd({thresholds.discard_mdd:.2%})")
    ret = total_return if total_return is not None else 0.0
    if ret <= thresholds.discard_return and (sharpe or 0) <= thresholds.discard_sharpe:
        discard_reasons.append(
            f"total_return({ret:.2%}) <= discard_return({thresholds.discard_return:.2%}) "
            f"and sharpe({sharpe}) <= discard_sharpe({thresholds.discard_sharpe})"
        )
    if discard_reasons:
        return DecisionResult(
            label="폐기",
            reasons=discard_reasons,
            metrics_snapshot=metrics_snapshot,
            thresholds_used=_thresholds_to_dict(thresholds),
        )

    # C) 실전 후보
    if (total_return is not None and total_return > 0) and (sharpe is not None and sharpe > 0):
        mdd_ok = mdd_abs < thresholds.candidate_mdd
        wr_ok = (win_rate is not None) and (win_rate >= thresholds.candidate_win_rate)
        if mdd_ok and wr_ok:
            return DecisionResult(
                label="실전 후보",
                reasons=["요건 충족"],
                metrics_snapshot=metrics_snapshot,
                thresholds_used=_thresholds_to_dict(thresholds),
            )

    # D) 개선 필요
    improve_reasons: List[str] = []
    if win_rate is not None and win_rate < thresholds.improve_win_rate:
        improve_reasons.append(f"win_rate({win_rate:.2%}) < improve_win_rate({thresholds.improve_win_rate:.2%})")
    if sharpe is not None and sharpe < 0:
        improve_reasons.append("sharpe < 0")
    if total_return is not None and total_return < 0:
        improve_reasons.append("total_return < 0")
    if not improve_reasons:
        improve_reasons.append("실전 후보/폐기 요건 미충족")

    return DecisionResult(
        label="개선 필요",
        reasons=improve_reasons,
        metrics_snapshot=metrics_snapshot,
        thresholds_used=_thresholds_to_dict(thresholds),
    )


def _thresholds_to_dict(t: DecisionThresholds) -> Dict[str, Any]:
    return {
        "min_trades": t.min_trades,
        "discard_mdd": t.discard_mdd,
        "discard_return": t.discard_return,
        "discard_sharpe": t.discard_sharpe,
        "candidate_mdd": t.candidate_mdd,
        "candidate_win_rate": t.candidate_win_rate,
        "improve_win_rate": t.improve_win_rate,
    }
