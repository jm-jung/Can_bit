"""Decision Engine 타입 정의 (비율 단위: 0.05 = 5%)"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DecisionThresholds:
    """판단 기준값 (모두 ratio: 0.05 = 5%, win_rate 0.45 = 45%)"""
    min_trades: int = 100
    discard_mdd: float = 0.12
    discard_return: float = -0.02
    discard_sharpe: float = -0.2
    candidate_mdd: float = 0.05
    candidate_win_rate: float = 0.45
    improve_win_rate: float = 0.40


@dataclass
class DecisionResult:
    """판단 결과 (단일 진실 소스)"""
    label: str  # "실전 후보" | "개선 필요" | "폐기" | "데이터 부족(보류)"
    reasons: List[str]
    metrics_snapshot: Dict[str, Any]
    thresholds_used: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "reasons": self.reasons,
            "metrics_snapshot": self.metrics_snapshot,
            "thresholds_used": self.thresholds_used,
        }
