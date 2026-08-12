"""정규화/검증: 퍼센트 vs 비율, None/NaN 처리"""
from __future__ import annotations

import math
from typing import Any, Optional


def safe_float(value: Any) -> Optional[float]:
    """None/NaN/비문자열 처리 후 float 또는 None"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    try:
        x = float(value)
        return None if (math.isnan(x) or math.isinf(x)) else x
    except (TypeError, ValueError):
        return None


def normalize_ratio(value: Optional[float]) -> Optional[float]:
    """
    return / mdd 등: abs(value) > 1.5 이면 퍼센트로 간주하고 /100.
    그 외는 비율 그대로. None이면 None.
    """
    v = safe_float(value)
    if v is None:
        return None
    if abs(v) > 1.5:
        return v / 100.0
    return v


def aggregate_returns_for_sharpe(returns: list) -> Optional[float]:
    """
    returns 리스트(비율)로 Sharpe 근사: mean(r)/std(r)*sqrt(n), n>=2, std>0.
    없으면 None.
    """
    valid = [r for r in returns if safe_float(r) is not None]
    if len(valid) < 2:
        return None
    mean_r = sum(valid) / len(valid)
    variance = sum((x - mean_r) ** 2 for x in valid) / (len(valid) - 1)
    std_r = math.sqrt(variance)
    if std_r <= 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(len(valid))
