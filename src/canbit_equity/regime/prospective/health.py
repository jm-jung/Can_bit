"""Readiness / operational health helpers."""
from __future__ import annotations

from typing import Any, Dict


def readiness_from_counts(strict: int, matured20: int, coverage: float, first_unseen_complete: bool) -> str:
    if not first_unseen_complete:
        return "WAITING_FOR_FIRST_UNSEEN_SESSION"
    if strict <= 0:
        return "COLLECTING_EARLY"
    if strict < 63:
        return "COLLECTING_EARLY"
    if strict < 126:
        return "COLLECTING_MINIMUM_APPROACHING"
    if strict >= 252 and matured20 >= 120 and coverage >= 0.90:
        return "TARGET_RESEARCH_REVIEW_READY"
    if strict >= 126 and matured20 >= 60 and coverage >= 0.90:
        return "MINIMUM_RESEARCH_REVIEW_READY"
    if strict >= 126 and (matured20 < 60 or coverage < 0.90):
        return "OUTCOME_SAMPLE_INSUFFICIENT"
    if strict >= 252:
        return "TARGET_SESSION_TARGET_REACHED"
    if strict >= 126:
        return "MINIMUM_SESSION_TARGET_REACHED"
    return "COLLECTING_EARLY"


def current_action(readiness: str, op: str) -> str:
    if op == "CRITICAL":
        return "FIX_PROSPECTIVE_PREDICTION_PIPELINE"
    if readiness == "WAITING_FOR_FIRST_UNSEEN_SESSION":
        return "WAITING_FOR_FIRST_UNSEEN_SESSION"
    if readiness in ("MINIMUM_RESEARCH_REVIEW_READY", "TARGET_RESEARCH_REVIEW_READY"):
        return "MANUAL_FORENSIC_REVIEW_REQUIRED"
    return "KEEP_COLLECTING_PROSPECTIVE_OBSERVATIONS"


def operational_status(
    *,
    lock_ok: bool,
    model_ok: bool,
    data_ok: bool,
    align_ok: bool,
    conflict: bool,
    warning: bool,
) -> str:
    if not lock_ok or not model_ok or conflict or not align_ok:
        return "CRITICAL"
    if warning or not data_ok:
        return "WARNING"
    return "NORMAL"
