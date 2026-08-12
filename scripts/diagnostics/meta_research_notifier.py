"""
Discord/webhook notifier for Meta research daily pipeline (diagnostics only).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("meta_research_notifier")


def _post_webhook(payload: Dict[str, Any], *, dry_run: bool = False) -> bool:
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if dry_run:
        logger.info("[Discord dry-run] %s", payload.get("content", "")[:500])
        return True
    if not webhook:
        logger.warning("DISCORD_WEBHOOK_URL missing")
        return False
    try:
        resp = requests.post(webhook, json=payload, timeout=15)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.error("Discord webhook failed: %s", exc)
        return False


def send_success_alert(report: Dict[str, Any], *, dry_run: bool = False) -> bool:
    milestone = report.get("milestone", {})
    content = (
        "**[CAN_BIT META DAILY]**\n"
        f"status: **{report.get('status', 'PASS')}**\n"
        f"rows_total: {report.get('total_rows', 0)}\n"
        f"rows_added: {report.get('rows_added', 0)}\n"
        f"milestone: {milestone.get('current', 0)} / {milestone.get('target_300', 300)}\n"
        f"Q2_BDI_MDD: {report.get('q2_bdi_mdd', 'N/A')}\n"
        f"false_high: {report.get('false_high', 'N/A')}\n"
        f"schema_drift: {report.get('schema_drift', False)}\n"
        f"duplicates: {report.get('duplicate_trade_ids', 0)}\n"
        f"audit: **{report.get('audit_status', 'PASS')}**\n"
        f"next_action: {report.get('next_action', 'accumulate')}"
    )
    shadow = report.get("meta_shadow")
    if isinstance(shadow, dict):
        content += (
            "\n\n"
            "**[CAN_BIT META SHADOW]**\n"
            f"date: {shadow.get('date', 'N/A')}\n"
            f"rows_added: {shadow.get('rows_added', 0)}\n"
            f"regime: {shadow.get('regime', 'N/A')}\n"
            f"meta_forward_score: {shadow.get('meta_forward_score', 'N/A')}\n"
            f"bucket_monotonicity: {shadow.get('bucket_monotonicity', False)}\n"
            f"routing_consistency: {shadow.get('routing_consistency', False)}\n"
            f"false_high_events: {shadow.get('false_high_events', 0)}\n"
            f"activation_policy_best: {shadow.get('activation_policy_best', 'N/A')}\n"
            f"Q2_vs_meta_shadow_MDD: {shadow.get('Q2_vs_meta_shadow_MDD', 'N/A')}\n"
            "promotion_ready: false"
        )
    cwce = report.get("risk_aware_tcn_shadow")
    if isinstance(cwce, dict):
        content += (
            "\n\n"
            "**[CAN_BIT RISK-AWARE TCN SHADOW]**\n"
            f"status: {cwce.get('status', 'PASS')}\n"
            f"candidate: {cwce.get('candidate', 'RiskAwareTCN_CWCE')}\n"
            f"LONG_ECE: {cwce.get('LONG_ECE_baseline', 'N/A')} -> {cwce.get('LONG_ECE_CWCE', 'N/A')}\n"
            f"false_high_gap: {cwce.get('false_high_gap_baseline', 'N/A')} -> {cwce.get('false_high_gap_CWCE', 'N/A')}\n"
            f"Q2_MDD: {cwce.get('Q2_MDD_baseline', 'N/A')} -> {cwce.get('Q2_MDD_CWCE', 'N/A')}\n"
            f"routing_consistency: {cwce.get('routing_consistency_CWCE', False)}\n"
            f"collapse: {cwce.get('confidence_collapse_detected', False)}\n"
            "promotion_ready: false"
        )
    return _post_webhook({"content": content}, dry_run=dry_run)


def send_failure_alert(
    reason: str,
    detail: Optional[str] = None,
    *,
    dry_run: bool = False,
) -> bool:
    content = (
        "**[CAN_BIT META DAILY — FAILURE]**\n"
        f"reason: **{reason}**\n"
    )
    if detail:
        content += f"detail: {detail[:900]}\n"
    content += "action: check ops log + dataset integrity"
    return _post_webhook({"content": content}, dry_run=dry_run)
