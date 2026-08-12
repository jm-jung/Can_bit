"""Metrics aggregation for prospective observation."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from canbit_equity.regime.prospective.config import PRIMARY_HORIZON, PROSP_ROOT


def extract_shadow_summary(shadow_result: Dict[str, Any]) -> Dict[str, Any]:
    base = (shadow_result.get("metrics") or {}).get("BASE") or {}
    c = base.get("candidate") or {}
    d = base.get("dual") or {}
    b = base.get("buy_hold") or {}
    if not c:
        return {
            "candidate_shadow_return": None,
            "dual_shadow_return": None,
            "buy_hold_return": None,
            "candidate_sharpe": None,
            "dual_sharpe": None,
            "candidate_mdd": None,
            "dual_mdd": None,
            "candidate_exposure": None,
            "candidate_trade_count": None,
        }
    return {
        "candidate_shadow_return": c.get("total_return"),
        "dual_shadow_return": d.get("total_return"),
        "buy_hold_return": b.get("total_return"),
        "candidate_sharpe": c.get("Sharpe") if (c.get("sessions") or 0) >= 20 else None,
        "dual_sharpe": d.get("Sharpe") if (d.get("sessions") or 0) >= 20 else None,
        "candidate_mdd": c.get("max_drawdown"),
        "dual_mdd": d.get("max_drawdown"),
        "candidate_exposure": c.get("long_exposure"),
        "candidate_trade_count": c.get("trade_count"),
    }


def outcome_counts(outcomes: pd.DataFrame) -> Dict[str, int]:
    if len(outcomes) == 0 or "horizon" not in outcomes.columns:
        return {"matured_5d": 0, "matured_10d": 0, "matured_20d": 0}
    return {
        "matured_5d": int((outcomes["horizon"] == 5).sum()),
        "matured_10d": int((outcomes["horizon"] == 10).sum()),
        "matured_20d": int((outcomes["horizon"] == PRIMARY_HORIZON).sum()),
    }


def maybe_write_review_packages(strict_n: int, matured20: int, coverage: float, state: Dict[str, Any]) -> Optional[str]:
    if strict_n >= 252 and matured20 >= 120 and coverage >= 0.90:
        name = "qqq_regime_target_review_package"
        verdict = "MANUAL_FORENSIC_REVIEW_REQUIRED"
    elif strict_n >= 126 and matured20 >= 60 and coverage >= 0.90:
        name = "qqq_regime_minimum_review_package"
        verdict = "MANUAL_FORENSIC_REVIEW_REQUIRED"
    else:
        return None
    payload = {**state, "review_verdict": verdict, "auto_promotion": False, "production_ready": False, "promotion_ready": False}
    import json
    from pathlib import Path

    for root in (PROSP_ROOT / "reports",):
        (root / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        (root / f"{name}.md").write_text(
            f"# {name}\n\n`{verdict}`\n\nNo automatic PROMISING promotion.\nproduction_ready=false promotion_ready=false\n"
        )
    return name
