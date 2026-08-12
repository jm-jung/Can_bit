"""Thin alignment re-export for prospective package layout."""
from __future__ import annotations

from typing import Any, Dict

from canbit_equity.regime.alignment import build_aligned_panel


def prospective_alignment_audit(qqq_df) -> Dict[str, Any]:
    _, audit = build_aligned_panel(qqq_df)
    return audit
