"""Decision 기준값: 기본값 + env/CLI override (하드코딩 없음)"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.decision.decision_types import DecisionThresholds


def get_default_thresholds() -> DecisionThresholds:
    return DecisionThresholds(
        min_trades=100,
        discard_mdd=0.12,
        discard_return=-0.02,
        discard_sharpe=-0.2,
        candidate_mdd=0.05,
        candidate_win_rate=0.45,
        improve_win_rate=0.40,
    )


def thresholds_from_env() -> DecisionThresholds:
    """환경변수 DECISION_MIN_TRADES 등으로 override (있을 때만)"""
    t = get_default_thresholds()
    v = os.getenv("DECISION_MIN_TRADES")
    if v is not None:
        try:
            t.min_trades = int(v)
        except ValueError:
            pass
    for key, attr, cast in [
        ("DECISION_DISCARD_MDD", "discard_mdd", float),
        ("DECISION_DISCARD_RETURN", "discard_return", float),
        ("DECISION_DISCARD_SHARPE", "discard_sharpe", float),
        ("DECISION_CANDIDATE_MDD", "candidate_mdd", float),
        ("DECISION_CANDIDATE_WIN_RATE", "candidate_win_rate", float),
        ("DECISION_IMPROVE_WIN_RATE", "improve_win_rate", float),
    ]:
        val = os.getenv(key)
        if val is not None:
            try:
                setattr(t, attr, cast(val))
            except ValueError:
                pass
    return t


def thresholds_from_dict(d: Optional[Dict[str, Any]]) -> DecisionThresholds:
    """딕셔너리(CLI args 등)에서 override"""
    t = get_default_thresholds()
    if not d:
        return t
    if "min_trades" in d and d["min_trades"] is not None:
        t.min_trades = int(d["min_trades"])
    for attr, key in [
        ("discard_mdd", "discard_mdd"),
        ("discard_return", "discard_return"),
        ("discard_sharpe", "discard_sharpe"),
        ("candidate_mdd", "candidate_mdd"),
        ("candidate_win_rate", "candidate_win_rate"),
        ("improve_win_rate", "improve_win_rate"),
    ]:
        if key in d and d[key] is not None:
            setattr(t, attr, float(d[key]))
    return t
