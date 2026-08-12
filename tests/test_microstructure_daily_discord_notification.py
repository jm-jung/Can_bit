"""Tests for microstructure daily Discord observation notifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/diagnostics"))

import microstructure_daily_discord_notification as n


def _base_compact(**overrides):
    base = {
        "latest_complete_day_utc": "2026-07-29",
        "latest_complete_day_coverage_pct": 100.0,
        "collector_health": "HEALTHY",
        "ws_state": "HEALTHY",
        "staleness": "STALENESS_OK",
        "CURRENT_COLLECTION_STABILITY": "STABLE",
        "CURRENT_15M_CORE_PERSIST_COVERAGE": 100.0,
        "CURRENT_60M_CORE_PERSIST_COVERAGE": 100.0,
        "KEEP_AWAKE_GUARD_ACTIVE": True,
        "KEEP_AWAKE_GUARD_VERDICT": "KEEP_AWAKE_GUARD_ACTIVE",
        "IDLE_SLEEP_ASSERTION_ACTIVE": True,
        "SYSTEM_SLEEP_ASSERTION_ACTIVE": True,
        "COLLECTOR_WRAPPED_BY_CAFFEINATE": True,
        "DUPLICATE_CAFFEINATE_GUARDS": 0,
        "ACTIVE_CONNECTION_GENERATIONS": 1,
        "pending_gaps": 0,
        "failed_gaps": 0,
        "contamination": 0,
        "POWER_SOURCE": "AC",
        "production_ready": False,
        "promotion_ready": False,
        "readiness": "QUALITY_REVIEW_REQUIRED",
        "primary_markers": 48,
        "basis_markers": 5,
        "trusted_taker_markers": 3,
        "readiness_bottleneck": "basis_markers",
        "recent_7d_coverage_pct": 30.0,
        "recent_7d_pass_days": "1/7",
        "partial_day_strict_coverage_pct": 100.0,
        "current_partial_day_utc": "2026-07-30",
        "CURRENT_15M_RECONNECT_CYCLES": 0,
        "CURRENT_60M_RECONNECT_CYCLES": 0,
        "hard_stale_incidents_7d": 0,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
    }
    base.update(overrides)
    return base


def test_expected_complete_day_utc():
    from datetime import datetime, timezone

    # KST 2026-07-30 09:05 = UTC 2026-07-30 00:05
    ts = datetime(2026, 7, 30, 0, 5, tzinfo=timezone.utc)
    assert n.expected_complete_day_utc(ts) == "2026-07-29"


def test_normal_day():
    ev = n.evaluate_operational(_base_compact(), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "NORMAL"
    assert ev["readiness"] == "QUALITY_REVIEW_REQUIRED"


def test_warning_coverage():
    ev = n.evaluate_operational(_base_compact(latest_complete_day_coverage_pct=92.0), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "WARNING"


def test_critical_coverage():
    ev = n.evaluate_operational(_base_compact(latest_complete_day_coverage_pct=70.0), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_collector_stale_critical():
    ev = n.evaluate_operational(_base_compact(staleness="STALE", collector_health="NOT_HEALTHY"), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_writer_stalled_critical():
    ev = n.evaluate_operational(_base_compact(CURRENT_COLLECTION_STABILITY="HARD_STALE"), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_keepawake_missing_critical():
    ev = n.evaluate_operational(
        _base_compact(KEEP_AWAKE_GUARD_ACTIVE=False, KEEP_AWAKE_GUARD_VERDICT="KEEP_AWAKE_GUARD_MISSING"),
        daily_ok=True,
        compact_ok=True,
        target_day="2026-07-29",
        expected_day="2026-07-29",
    )
    assert ev["operational_severity"] == "CRITICAL"


def test_duplicate_guard_critical():
    ev = n.evaluate_operational(_base_compact(DUPLICATE_CAFFEINATE_GUARDS=2), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_pending_gap_critical():
    ev = n.evaluate_operational(_base_compact(pending_gaps=1), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_failed_gap_critical():
    ev = n.evaluate_operational(_base_compact(failed_gaps=1), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_contamination_critical():
    ev = n.evaluate_operational(_base_compact(contamination=1), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_private_call_critical():
    ev = n.evaluate_operational(_base_compact(private_endpoint_calls=1), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_production_ready_critical():
    ev = n.evaluate_operational(_base_compact(production_ready=True), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_readiness_quality_review_operational_normal():
    ev = n.evaluate_operational(_base_compact(readiness="QUALITY_REVIEW_REQUIRED"), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "NORMAL"
    payload = n.build_discord_payload(ev, target_day="2026-07-29")
    assert payload["embeds"][0]["color"] == n.COLOR_NORMAL


def test_ready_transition_text():
    ev = n.evaluate_operational(
        _base_compact(readiness="READY_FOR_FORMAL_EVALUATION"),
        daily_ok=True,
        compact_ok=True,
        target_day="2026-07-29",
        expected_day="2026-07-29",
    )
    payload = n.build_discord_payload(ev, target_day="2026-07-29", ready_transition=True)
    assert "정식" in payload["embeds"][0]["description"] or "🎯" in payload["embeds"][0]["description"]


def test_daily_update_failure_critical():
    ev = n.evaluate_operational(_base_compact(), daily_ok=False, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_compact_invalid_critical():
    ev = n.evaluate_operational({}, daily_ok=True, compact_ok=False, target_day="2026-07-29", expected_day="2026-07-29")
    assert ev["operational_severity"] == "CRITICAL"


def test_webhook_url_validation(tmp_path):
    bad = tmp_path / "wh"
    bad.write_text("not-a-url\n")
    bad.chmod(0o600)
    url, meta = n.load_webhook_url(bad)
    assert url is None
    assert meta["error"] == "WEBHOOK_SECRET_INVALID_FORMAT"


def test_secret_missing(tmp_path):
    url, meta = n.load_webhook_url(tmp_path / "missing")
    assert url is None
    assert meta["error"] == "WEBHOOK_SECRET_MISSING"


def test_redact_never_leaks():
    secret = "https://discord.com/api/webhooks/123/abcSECRETTOKEN"
    text = n.redact_text(f"posted to {secret}", [secret])
    assert "abcSECRETTOKEN" not in text
    assert "***REDACTED***" in text


def test_send_webhook_204(monkeypatch):
    class Resp:
        status = 204

        def read(self, n=-1):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def getcode(self):
            return 204

    monkeypatch.setattr(n.urllib.request, "urlopen", lambda *a, **k: Resp())
    result = n.send_webhook("https://discord.com/api/webhooks/1/token", {"content": "x"})
    assert result["success"] is True
    assert result["http_status"] == 204


def test_send_webhook_429_then_ok(monkeypatch):
    calls = {"n": 0}

    class Ok:
        status = 204

        def read(self, n=-1):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def getcode(self):
            return 204

    def fake_urlopen(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise n.urllib.error.HTTPError(
                "https://discord.com/api/webhooks/1/t",
                429,
                "rate",
                hdrs=type("H", (), {"get": lambda self, k, d=None: "0"})(),
                fp=None,
            )
        return Ok()

    monkeypatch.setattr(n.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(n.time, "sleep", lambda s: None)
    result = n.send_webhook("https://discord.com/api/webhooks/1/token", {"content": "x"})
    assert result["success"] is True


def test_send_webhook_500_retries(monkeypatch):
    def fake_urlopen(*a, **k):
        raise n.urllib.error.HTTPError(
            "https://discord.com/api/webhooks/1/t",
            500,
            "err",
            hdrs=type("H", (), {"get": lambda self, k, d=None: None})(),
            fp=None,
        )

    monkeypatch.setattr(n.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(n.time, "sleep", lambda s: None)
    result = n.send_webhook("https://discord.com/api/webhooks/1/token", {"content": "x"})
    assert result["success"] is False
    assert result["error_category"] in {"SERVER_ERROR", "HTTP_500"}
    assert len(result["attempts"]) == n.MAX_RETRIES


def test_send_webhook_permanent_4xx(monkeypatch):
    def fake_urlopen(*a, **k):
        raise n.urllib.error.HTTPError(
            "https://discord.com/api/webhooks/1/t",
            400,
            "bad",
            hdrs=type("H", (), {"get": lambda self, k, d=None: None})(),
            fp=None,
        )

    monkeypatch.setattr(n.urllib.request, "urlopen", fake_urlopen)
    result = n.send_webhook("https://discord.com/api/webhooks/1/token", {"content": "x"})
    assert result.get("permanent") is True
    assert result["success"] is False


def test_duplicate_and_outbox(tmp_path, monkeypatch):
    # isolate state dirs
    monkeypatch.setattr(n, "ROOT", tmp_path)
    monkeypatch.setattr(n, "STATE", tmp_path / "state")
    monkeypatch.setattr(n, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(n, "OUTBOX_ARCHIVED", tmp_path / "outbox/archived")
    monkeypatch.setattr(n, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(n, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(n, "VALIDATION", tmp_path / "validation")
    n.ensure_dirs()

    secret = tmp_path / "secret"
    secret.write_text("https://discord.com/api/webhooks/123456789012345678/abcdefghijklmnopqrstuvwx_YZ-0123456789abcdef\n")
    secret.chmod(0o600)

    monkeypatch.setattr(n, "send_webhook", lambda url, payload: {"success": True, "http_status": 204, "attempts": [{"attempt": 1}]})
    monkeypatch.setattr(n, "run_suite", lambda *a, **k: {"exit_code": 0, "parsed": _base_compact(), "parse_error": None})

    r1 = n.run_notification(skip_daily_update=True, compact_override=_base_compact(), secret_path=secret)
    assert r1["verdict"] == "SENT_OK"
    r2 = n.run_notification(skip_daily_update=True, compact_override=_base_compact(), secret_path=secret)
    assert r2["verdict"] == "ALREADY_SENT_NO_ACTION"
    assert r2["duplicate_send_prevented"] is True


def test_outbox_retry_success(tmp_path, monkeypatch):
    monkeypatch.setattr(n, "ROOT", tmp_path)
    monkeypatch.setattr(n, "STATE", tmp_path / "state")
    monkeypatch.setattr(n, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(n, "OUTBOX_ARCHIVED", tmp_path / "outbox/archived")
    monkeypatch.setattr(n, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(n, "LOGS", tmp_path / "logs")
    n.ensure_dirs()
    payload = {"embeds": [{"title": "x"}]}
    path = n.outbox_write("2026-07-29", "DAILY_SUMMARY", payload, {"attempts": [], "http_status": 500, "error_category": "SERVER_ERROR"}, "CRITICAL")
    assert path.exists()
    monkeypatch.setattr(n, "send_webhook", lambda url, payload: {"success": True, "http_status": 204, "attempts": []})
    result = n.retry_outbox("https://discord.com/api/webhooks/1/token")
    assert any(r.get("action") == "sent" for r in result["retried"])
    assert result["pending_outbox_count"] == 0


def test_payload_size_reasonable():
    ev = n.evaluate_operational(_base_compact(), daily_ok=True, compact_ok=True, target_day="2026-07-29", expected_day="2026-07-29")
    payload = n.build_discord_payload(ev, target_day="2026-07-29")
    raw = json.dumps(payload)
    assert len(raw) < 6000
    assert "ChatGPT" in raw or "CAN_BIT DAILY" in raw


def test_partial_day_not_driving_critical():
    # low partial should not alone make critical if complete day ok
    ev = n.evaluate_operational(
        _base_compact(partial_day_strict_coverage_pct=5.0),
        daily_ok=True,
        compact_ok=True,
        target_day="2026-07-29",
        expected_day="2026-07-29",
    )
    assert ev["operational_severity"] == "NORMAL"
