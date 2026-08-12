"""Status freshness helpers — read-only, no provider/model side effects."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from canbit_equity.regime.prospective.config import DIAG_ROOT, LAUNCHD_LABEL, PROSP_ROOT

KST = ZoneInfo("Asia/Seoul")


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _mtime_utc(path: Path) -> Optional[datetime]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def launchd_probe() -> Dict[str, Any]:
    uid = os.getuid()
    label = LAUNCHD_LABEL
    out: Dict[str, Any] = {
        "launchd_label": label,
        "launchd_loaded": False,
        "launchd_enabled": None,
        "launchd_last_exit_code": None,
        "launchd_run_count": None,
        "launchd_state": None,
    }
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = proc.stdout or ""
        out["launchd_loaded"] = proc.returncode == 0 and "state =" in text
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("state ="):
                out["launchd_state"] = s.split("=", 1)[1].strip()
            elif s.startswith("last exit code ="):
                raw = s.split("=", 1)[1].strip()
                out["launchd_last_exit_code"] = raw
            elif s.startswith("runs ="):
                try:
                    out["launchd_run_count"] = int(s.split("=", 1)[1].strip())
                except Exception:
                    pass
        # enabled probe
        en = subprocess.run(
            ["launchctl", "print-disabled", f"gui/{uid}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        disabled = False
        for line in (en.stdout or "").splitlines():
            if label in line and "disabled" in line.lower() and "true" in line.lower():
                disabled = True
        out["launchd_enabled"] = not disabled if en.returncode == 0 else None
    except Exception as exc:
        out["launchd_probe_error"] = type(exc).__name__
    log = DIAG_ROOT / "launchd/stderr.log"
    out["launchd_log_last_modified_utc"] = _mtime_utc(log).isoformat() if _mtime_utc(log) else None
    return out


def build_status_payload(*, compact: bool = True) -> Dict[str, Any]:
    """Build status response with freshness metadata. Does not update state or download data."""
    now_utc = datetime.now(timezone.utc)
    now_kst = now_utc.astimezone(KST)

    compact_path = DIAG_ROOT / "reports/qqq_regime_prospective_compact.json"
    state_path = PROSP_ROOT / "state/prospective_state.json"
    source_path = compact_path if compact_path.exists() and compact_path.stat().st_size else state_path
    if not source_path.exists() or source_path.stat().st_size == 0:
        return {
            "status": "NOT_RUN",
            "status_checked_at_utc": now_utc.isoformat(),
            "status_checked_at_kst": now_kst.isoformat(),
            "state_fresh": False,
            "status_was_stale_snapshot": True,
            "production_ready": False,
            "promotion_ready": False,
            "execution_enabled": False,
        }

    try:
        underlying = json.loads(source_path.read_text())
    except json.JSONDecodeError:
        underlying = json.loads(state_path.read_text()) if state_path.exists() else {"status": "CORRUPT"}

    # Overlay latest valid STRICT prediction from immutable ledger (never trust null runtime fields).
    try:
        import pandas as pd

        from canbit_equity.regime.prospective.discord_notify import decide_notification
        from canbit_equity.regime.prospective.prediction import get_latest_valid_strict_prediction

        ledger_path = PROSP_ROOT / "predictions/strict_predictions.parquet"
        if ledger_path.exists():
            strict = pd.read_parquet(ledger_path)
            latest_valid = get_latest_valid_strict_prediction(strict)
            if latest_valid is not None:
                underlying = {
                    **underlying,
                    "prediction_available": True,
                    "latest_valid_session": latest_valid["signal_session"],
                    "latest_valid_signal": latest_valid["signal"],
                    "latest_valid_probability": latest_valid["probability"],
                    "latest_valid_dual": latest_valid["dual"],
                    "latest_valid_provenance": latest_valid["provenance"],
                    "prediction_session": latest_valid["signal_session"],
                    "locked_signal": latest_valid["signal"],
                    "probability": latest_valid["probability"],
                    "dual_signal": latest_valid["dual"],
                    "prediction_provenance": latest_valid["provenance"],
                    "feature_completeness": "COMPLETE",
                    "strict_sessions": int(len(strict[strict["provenance_tier"] == "STRICT_PROSPECTIVE"]))
                    if "provenance_tier" in strict.columns
                    else int(len(strict)),
                }
                if not underlying.get("current_refresh_status"):
                    underlying["current_refresh_status"] = "STATUS_READ_ONLY_OVERLAY"
                action, _, _ = decide_notification(underlying)
                underlying["notification_action"] = action
    except Exception as exc:
        underlying = {**underlying, "ledger_overlay_error": type(exc).__name__}

    state_gen = _parse_iso(underlying.get("generated_at_utc"))
    age_hours = None
    if state_gen is not None:
        age_hours = (now_utc - state_gen).total_seconds() / 3600.0
    stale = age_hours is None or age_hours >= 24.0

    launchd = launchd_probe()
    automation_stale = stale or (
        str(launchd.get("launchd_last_exit_code") or "") not in ("0", "0.", "(never exited)", "None", "")
        and launchd.get("launchd_last_exit_code") not in (0, None, "(never exited)")
    )
    # treat last exit 1 as automation stale
    try:
        if int(str(launchd.get("launchd_last_exit_code")).split()[0]) != 0:
            automation_stale = True
    except Exception:
        pass

    eligible = underlying.get("strict_sessions", 0) + underlying.get("late_replays", 0) + underlying.get(
        "missing_sessions", 0
    )
    coverage = underlying.get("strict_coverage")
    coverage_reason = None
    if int(underlying.get("strict_sessions") or 0) == 0 and int(underlying.get("late_replays") or 0) == 0 and int(
        underlying.get("missing_sessions") or 0
    ) == 0:
        # waiting / no eligible — avoid misleading 1.0
        if underlying.get("research_readiness") == "WAITING_FOR_FIRST_UNSEEN_SESSION" or eligible == 0:
            # Only override display if coverage was the placeholder 1.0 with zero activity
            if coverage == 1.0 and int(underlying.get("strict_sessions") or 0) == 0:
                coverage = None
                coverage_reason = "NO_ELIGIBLE_SESSIONS_YET"

    payload = {
        **underlying,
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_kst": now_kst.isoformat(),
        "status_checked_at_utc": now_utc.isoformat(),
        "status_checked_at_kst": now_kst.isoformat(),
        "underlying_state_generated_at_utc": underlying.get("generated_at_utc"),
        "underlying_state_generated_at_kst": underlying.get("generated_at_kst"),
        "state_generated_at_utc": underlying.get("generated_at_utc"),
        "state_generated_at_kst": underlying.get("generated_at_kst"),
        "state_age_hours": age_hours,
        "state_fresh": (not stale),
        "status_was_stale_snapshot": stale,
        "status_freshness_verdict": "STATUS_STALE_SNAPSHOT" if stale else "STATUS_FRESH",
        "latest_completed_session_from_state": underlying.get("latest_completed_qqq_session"),
        "status_source_path": str(source_path.resolve()),
        "status_does_not_run_update": True,
        "automation_stale": automation_stale,
        "strict_coverage": coverage,
        "strict_coverage_reason": coverage_reason,
        "production_ready": False,
        "promotion_ready": False,
        "execution_enabled": False,
        "private_calls": 0,
        "order_calls": 0,
        **launchd,
    }
    if not compact:
        payload["underlying_state"] = underlying
    return payload
