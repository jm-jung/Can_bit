"""Daily / cumulative reporting."""
from __future__ import annotations

from typing import Any, Dict

from canbit_equity.regime.prospective.config import DIAG_ROOT, PROSP_ROOT
from canbit_equity.regime.prospective.file_lock import atomic_write_json


def write_reports(state: Dict[str, Any], day: str, la: Dict[str, Any], ma: Dict[str, Any]) -> None:
    atomic_write_json(PROSP_ROOT / "state/prospective_state.json", state)
    atomic_write_json(PROSP_ROOT / "metrics/prospective_metrics_latest.json", state)
    atomic_write_json(PROSP_ROOT / f"daily_snapshots/{day}.json", state)
    md = (
        f"# QQQ Regime Prospective Daily\n\n"
        f"- operational: `{state['operational_status']}`\n"
        f"- readiness: `{state['research_readiness']}`\n"
        f"- latest completed: {state['latest_completed_qqq_session']}\n"
        f"- first unseen: {state['first_unseen_session']}\n"
        f"- new prediction: {state['new_prediction']}\n"
        f"- action: `{state['current_action']}`\n"
        f"- production_ready=false promotion_ready=false execution_enabled=false\n"
    )
    (PROSP_ROOT / f"daily_snapshots/{day}.md").write_text(md)
    (PROSP_ROOT / "reports/qqq_regime_prospective_latest.md").write_text(md)
    atomic_write_json(PROSP_ROOT / "reports/qqq_regime_prospective_latest.json", state)

    compact = {
        "qqq_prospective_implementation_verdict": (
            "QQQ_PROSPECTIVE_OBSERVER_READY_WAITING"
            if state["research_readiness"] == "WAITING_FOR_FIRST_UNSEEN_SESSION"
            else "QQQ_PROSPECTIVE_OBSERVER_ACTIVE"
        ),
        "mode": "PROSPECTIVE_OBSERVATION_ONLY",
        **{k: state[k] for k in state if k not in ("note",)},
    }
    atomic_write_json(DIAG_ROOT / "reports/qqq_regime_prospective_compact.json", compact)
    atomic_write_json(
        DIAG_ROOT / "reports/qqq_regime_prospective_implementation_report.json",
        {**compact, "lock_audit_detail": la, "model_audit_detail": ma, "implementation_note": state.get("note")},
    )
    (DIAG_ROOT / "reports/qqq_regime_prospective_implementation_report.md").write_text(
        f"# QQQ Prospective Implementation\n\n"
        f"**Verdict:** `{compact['qqq_prospective_implementation_verdict']}`\n\n"
        f"Lock PASS · Model `{ma.get('verdict')}` · fixed threshold={state.get('fixed_threshold')}\n\n"
        f"Latest completed QQQ: {state['latest_completed_qqq_session']}\n"
        f"First unseen: {state['first_unseen_session']}\n"
        f"Readiness: `{state['research_readiness']}`\n"
        f"Action: `{state['current_action']}`\n\n"
        f"No historical re-research. No daily retrain. Original lock untouched.\n"
    )
