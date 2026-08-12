"""QQQ prospective Discord notifier — separate from BTC notifier."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from canbit_equity.regime.prospective.config import DIAG_ROOT, PROSP_ROOT, WEBHOOK_DEFAULT, ensure_dirs
from canbit_equity.regime.prospective.file_lock import atomic_write_json

KST = ZoneInfo("Asia/Seoul")


def _webhook_path() -> Path:
    override = os.environ.get("CANBIT_QQQ_REGIME_DISCORD_WEBHOOK_FILE")
    return Path(override) if override else WEBHOOK_DEFAULT


def load_webhook_url() -> Optional[str]:
    p = _webhook_path()
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip()


def _append_outbox(event: Dict[str, Any]) -> None:
    path = PROSP_ROOT / "outbox/discord_outbox.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def _dedupe_path() -> Path:
    return DIAG_ROOT / "discord/dedupe_keys.json"


def load_dedupe_keys() -> Dict[str, Any]:
    p = _dedupe_path()
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def prediction_dedupe_key(state: Dict[str, Any]) -> str:
    session = state.get("prediction_session") or state.get("latest_valid_session") or "NA"
    signal = state.get("locked_signal") or state.get("latest_valid_signal") or "NA"
    prob = state.get("probability")
    if prob is None:
        prob = state.get("latest_valid_probability")
    dual = state.get("dual_signal") or state.get("latest_valid_dual") or "NA"
    prov = state.get("prediction_provenance") or state.get("latest_valid_provenance") or "NA"
    model = state.get("model_hash") or "NA"
    return f"QQQ_PREDICTION_V2:{session}:{signal}:{prob}:{dual}:{prov}:{model}"


def matured_dedupe_key(session: str, matured_n: int) -> str:
    return f"QQQ_MATURED_20D_V1:{session}:{matured_n}"


def warning_dedupe_key(code: str, state_hash: str) -> str:
    return f"QQQ_OPERATIONAL_WARNING_V1:{code}:{state_hash}"


def build_message(state: Dict[str, Any]) -> str:
    """Build user-facing Discord content. Never render SIGNAL n/a + PROBABILITY None combo."""
    action = state.get("notification_action")
    pred_ok = bool(state.get("prediction_available")) or (
        state.get("locked_signal") in ("LONG", "FLAT") and state.get("probability") is not None
    )
    session = (
        state.get("prediction_session")
        or state.get("latest_valid_session")
        or state.get("latest_completed_qqq_session")
    )
    if action == "NO_STRICT_PREDICTION_WARNING" and not pred_ok:
        return (
            "QQQ REGIME PROSPECTIVE (research shadow — not an investment order)\n\n"
            f"SESSION: {session}\n"
            f"OPERATIONAL: {state.get('operational_status')}\n"
            f"READINESS: {state.get('research_readiness')}\n\n"
            "SIGNAL FOR NEXT SESSION: NOT_AVAILABLE\n"
            "REASON: NO_STRICT_PREDICTION_GENERATED\n"
            f"THRESHOLD: {state.get('threshold')}\n\n"
            f"CURRENT REFRESH: {state.get('current_refresh_status')}\n\n"
            f"STRICT SESSIONS: {state.get('strict_sessions')} / {state.get('minimum_sessions')} / {state.get('target_sessions')}\n"
            f"MATURED 20D: {state.get('matured_20d')}\n\n"
            f"ACTION: {state.get('current_action')}\n"
            "production_ready=false · promotion_ready=false · execution_enabled=false\n"
        )

    signal = state.get("locked_signal") or state.get("latest_valid_signal")
    prob = state.get("probability")
    if prob is None:
        prob = state.get("latest_valid_probability")
    dual = state.get("dual_signal") or state.get("latest_valid_dual")
    prov = state.get("prediction_provenance") or state.get("latest_valid_provenance")
    features = "COMPLETE" if pred_ok else (state.get("feature_completeness") or "NOT_AVAILABLE")
    refresh = state.get("current_refresh_status")

    lines = [
        "QQQ REGIME PROSPECTIVE (research shadow — not an investment order)",
        "",
        f"SESSION: {session}",
        f"OPERATIONAL: {state.get('operational_status')}",
        f"READINESS: {state.get('research_readiness')}",
        "",
        f"SIGNAL FOR NEXT SESSION: {signal if signal in ('LONG', 'FLAT') else 'NOT_AVAILABLE'}",
        f"PROBABILITY: {prob if prob is not None else 'NOT_AVAILABLE'}",
        f"THRESHOLD: {state.get('threshold')}",
        f"DUAL: {dual if dual in ('LONG', 'FLAT') else 'NOT_AVAILABLE'}",
        f"PROVENANCE: {prov if prov else 'NOT_AVAILABLE'}",
        "",
        f"DATA: {state.get('data_quality')}  ALIGNMENT: {state.get('alignment')}",
        f"FEATURES: {features}",
    ]
    if refresh and refresh not in ("COMPLETE", None):
        lines.append(f"CURRENT REFRESH: {refresh}")
    lines.extend(
        [
            "",
            f"STRICT SESSIONS: {state.get('strict_sessions')} / {state.get('minimum_sessions')} / {state.get('target_sessions')}",
            f"MATURED 20D: {state.get('matured_20d')}",
            "",
            f"ACTION: {state.get('current_action')}",
            "production_ready=false · promotion_ready=false · execution_enabled=false",
            "",
        ]
    )
    return "\n".join(lines)


def send_discord(content: str, dedupe_key: str, dry_run: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    keys = load_dedupe_keys()
    if dedupe_key in keys:
        return {"status": "ALREADY_SENT_NO_ACTION", "dedupe_key": dedupe_key, "webhook_secret_exposed": False}

    msg_hash = hashlib.sha256(content.encode()).hexdigest()
    if dry_run:
        return {
            "status": "DRY_RUN",
            "dedupe_key": dedupe_key,
            "message_hash": msg_hash,
            "webhook_secret_exposed": False,
            "content_preview_len": len(content),
            "content_preview": content[:800],
        }

    url = load_webhook_url()
    if not url:
        event = {"dedupe_key": dedupe_key, "content": content, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _append_outbox(event)
        return {"status": "NO_WEBHOOK_OUTBOXED", "dedupe_key": dedupe_key, "webhook_secret_exposed": False}

    body = json.dumps({"content": content}).encode()
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "canbit-observation-notifier/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            status = int(getattr(resp, "status", None) or resp.getcode())
    except Exception as exc:
        _append_outbox(
            {
                "dedupe_key": dedupe_key,
                "content": content,
                "error": type(exc).__name__,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        return {
            "status": "SEND_FAILED_OUTBOXED",
            "error": type(exc).__name__,
            "dedupe_key": dedupe_key,
            "webhook_secret_exposed": False,
        }

    keys[dedupe_key] = {
        "sent_at_utc": datetime.now(timezone.utc).isoformat(),
        "message_hash": msg_hash,
        "http_status": status,
    }
    atomic_write_json(_dedupe_path(), keys)
    return {
        "status": "SENT_OK",
        "http_status": status,
        "dedupe_key": dedupe_key,
        "message_hash": msg_hash,
        "webhook_secret_exposed": False,
    }


def _final_retry_exhausted(now_kst: Optional[datetime] = None) -> bool:
    now_kst = now_kst or datetime.now(KST)
    return (now_kst.hour > 15) or (now_kst.hour == 15 and now_kst.minute >= 20)


def decide_notification(state: Dict[str, Any], now_kst: Optional[datetime] = None) -> Tuple[str, bool, Optional[str]]:
    """
    Returns (notification_action, should_send, dedupe_key).
    Policies: existing strict prediction + transient pending => no Discord;
    new prediction / matured / critical / exhausted no-prediction warning only.
    """
    now_kst = now_kst or datetime.now(KST)
    pred_ok = bool(state.get("prediction_available")) or (
        state.get("locked_signal") in ("LONG", "FLAT") and state.get("probability") is not None
    )
    refresh = state.get("current_refresh_status") or state.get("feature_completeness")
    op = state.get("operational_status")

    if bool(state.get("new_prediction")) and pred_ok:
        return "NEW_STRICT_PREDICTION", True, prediction_dedupe_key(state)

    if int(state.get("matured_20d_new") or 0) > 0:
        sess = str(state.get("latest_completed_qqq_session") or "NA")
        key = matured_dedupe_key(sess, int(state.get("matured_20d") or 0))
        return "MATURED_20D", True, key

    if op == "CRITICAL":
        code = str(state.get("warning_code") or "CRITICAL")
        h = hashlib.sha256(
            json.dumps(
                {
                    "op": op,
                    "session": state.get("latest_completed_qqq_session"),
                    "code": code,
                    "lock": state.get("lock_audit"),
                    "model": state.get("model_audit"),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:16]
        return "CRITICAL", True, warning_dedupe_key(code, h)

    # Existing valid prediction: never Discord for transient pending / source-revision WARNING spam
    if pred_ok and not bool(state.get("new_prediction")):
        if refresh in (
            "PENDING_OR_MISSING",
            "IDEMPOTENT_REFRESH_NO_NEW_PREDICTION",
            "COMPLETE",
            "STATUS_READ_ONLY_OVERLAY",
        ):
            return "EXISTING_PREDICTION_NO_ACTION", False, None
        if op == "WARNING":
            return "EXISTING_PREDICTION_NO_ACTION", False, None

    # No prediction yet for pending session
    pending = state.get("pending_signal_session")
    if pending and not pred_ok:
        if _final_retry_exhausted(now_kst):
            code = "NO_STRICT_PREDICTION_AVAILABLE"
            h = hashlib.sha256(f"{pending}:{code}".encode()).hexdigest()[:16]
            return "NO_STRICT_PREDICTION_WARNING", True, warning_dedupe_key(code, h)
        return "WAITING_FOR_FEATURES_NO_DISCORD", False, None

    # Prediction exists for a prior session but current pending session has none yet
    if pending and pred_ok and str(state.get("prediction_session")) != str(pending):
        # Still waiting for features for the new session; do not null-render prior prediction
        if _final_retry_exhausted(now_kst):
            # Only warn if the pending session itself has no ledger row (caller should set)
            if state.get("pending_session_has_prediction") is False:
                code = "NO_STRICT_PREDICTION_AVAILABLE"
                h = hashlib.sha256(f"{pending}:{code}".encode()).hexdigest()[:16]
                return "NO_STRICT_PREDICTION_WARNING", True, warning_dedupe_key(code, h)
        return "WAITING_FOR_FEATURES_NO_DISCORD", False, None

    # Recurring WARNING (e.g. source_revision) without new material event
    if op == "WARNING":
        return "TRANSIENT_PENDING_EXISTING_PREDICTION_NO_ACTION" if pred_ok else "NO_EVENT_NO_SEND", False, None

    return "NO_EVENT_NO_SEND", False, None


def maybe_notify_from_state(state: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    action, should, dedupe = decide_notification(state)
    out: Dict[str, Any] = {
        "notification_action": action,
        "should_send": should,
        "webhook_secret_exposed": False,
        "prediction_available": bool(state.get("prediction_available"))
        or (state.get("locked_signal") in ("LONG", "FLAT") and state.get("probability") is not None),
        "signal": state.get("locked_signal") or state.get("latest_valid_signal"),
        "probability": state.get("probability")
        if state.get("probability") is not None
        else state.get("latest_valid_probability"),
        "dual": state.get("dual_signal") or state.get("latest_valid_dual"),
        "provenance": state.get("prediction_provenance") or state.get("latest_valid_provenance"),
        "session": state.get("prediction_session")
        or state.get("latest_valid_session")
        or state.get("latest_completed_qqq_session"),
        "current_refresh_status": state.get("current_refresh_status"),
    }
    if not should or not dedupe:
        out["status"] = action
        return out

    content = build_message({**state, "notification_action": action})
    # Guard: never send the old broken n/a + None combo
    if "SIGNAL FOR NEXT SESSION: n/a" in content or "PROBABILITY: None" in content:
        out["status"] = "BLOCKED_INVALID_RENDER"
        out["notification_action"] = "BLOCKED_INVALID_RENDER"
        return out

    res = send_discord(content, dedupe, dry_run=dry_run)
    out.update(res)
    out["notification_action"] = action
    return out


def send_test() -> Dict[str, Any]:
    content = (
        "[TEST] QQQ NOTIFIER EXISTING-PREDICTION FALLBACK PASS\n\n"
        "SESSION: 2026-08-04\n"
        "LATEST VALID SIGNAL: FLAT\n"
        "CURRENT REFRESH: PENDING_OR_MISSING\n"
        "EXPECTED ACTION: EXISTING_PREDICTION_NO_ACTION\n"
        "production_ready=false promotion_ready=false execution_enabled=false\n"
    )
    return send_discord(content, dedupe_key="TEST_MESSAGE_V2", dry_run=False)


def flush_outbox() -> Dict[str, Any]:
    path = PROSP_ROOT / "outbox/discord_outbox.jsonl"
    if not path.exists():
        return {"status": "EMPTY", "sent": 0}
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    sent = 0
    remain = []
    for ln in lines:
        ev = json.loads(ln)
        res = send_discord(ev["content"], ev["dedupe_key"], dry_run=False)
        if res.get("status") in ("SENT_OK", "ALREADY_SENT_NO_ACTION"):
            sent += 1
        else:
            remain.append(ln)
    path.write_text("\n".join(remain) + ("\n" if remain else ""))
    return {"status": "FLUSHED", "sent": sent, "remaining": len(remain), "webhook_secret_exposed": False}
