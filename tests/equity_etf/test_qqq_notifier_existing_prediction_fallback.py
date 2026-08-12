"""Notifier fallback: existing STRICT prediction must not be rendered as n/a on retry."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.regime.prospective.config import DIAG_ROOT, ORIG_LOCK, PROSP_ROOT
from canbit_equity.regime.prospective.discord_notify import (
    build_message,
    decide_notification,
    maybe_notify_from_state,
    prediction_dedupe_key,
)
from canbit_equity.regime.prospective.prediction import (
    get_latest_valid_prediction_for_session,
    get_latest_valid_strict_prediction,
)
from canbit_equity.regime.prospective.status import build_status_payload

KST = ZoneInfo("Asia/Seoul")
EXPECTED_PROB = 0.3589586195705251
MODEL_HASH = "50bf31647dbe1abafc10286d298f841b81aa777341af86f8fdc027fcb4d2d45e"
FEATURE_HASH = "a2c6da49fb35191a8ac43eccd6aff1921d70f6e926ee684b67b847e1d3090d02"
LOCK_HASH = "f0f1cfab7e3ab69215e393447e59172c8e983167d11cdd23f1bf0a1e4fa9ee3a"


def _ledger() -> pd.DataFrame:
    return pd.read_parquet(PROSP_ROOT / "predictions/strict_predictions.parquet")


def test_existing_strict_plus_pending_no_discord():
    pred = get_latest_valid_prediction_for_session(_ledger(), "2026-08-04")
    assert pred is not None
    state = {
        "new_prediction": False,
        "prediction_available": True,
        "prediction_session": pred["signal_session"],
        "locked_signal": pred["signal"],
        "probability": pred["probability"],
        "dual_signal": pred["dual"],
        "prediction_provenance": pred["provenance"],
        "current_refresh_status": "PENDING_OR_MISSING",
        "feature_completeness": "COMPLETE",
        "operational_status": "WARNING",
        "model_hash": MODEL_HASH,
        "threshold": 0.45,
        "strict_sessions": 3,
        "matured_20d": 0,
        "minimum_sessions": 126,
        "target_sessions": 252,
        "current_action": "KEEP_COLLECTING_PROSPECTIVE_OBSERVATIONS",
    }
    action, should, _ = decide_notification(state)
    assert action == "EXISTING_PREDICTION_NO_ACTION"
    assert should is False
    res = maybe_notify_from_state(state, dry_run=True)
    assert res["notification_action"] == "EXISTING_PREDICTION_NO_ACTION"
    assert res["signal"] == "FLAT"
    assert abs(float(res["probability"]) - EXPECTED_PROB) < 1e-12
    msg = build_message(state)
    assert "SIGNAL FOR NEXT SESSION: n/a" not in msg
    assert "PROBABILITY: None" not in msg
    assert "FLAT" in msg


def test_no_prediction_0920_pending_no_discord():
    state = {
        "new_prediction": False,
        "prediction_available": False,
        "locked_signal": None,
        "probability": None,
        "pending_signal_session": "2026-08-06",
        "pending_session_has_prediction": False,
        "current_refresh_status": "PENDING_OR_MISSING",
        "operational_status": "WARNING",
        "latest_completed_qqq_session": "2026-08-06",
        "matured_20d": 0,
    }
    action, should, _ = decide_notification(state, now_kst=datetime(2026, 8, 6, 9, 20, tzinfo=KST))
    assert action == "WAITING_FOR_FEATURES_NO_DISCORD"
    assert should is False


def test_no_prediction_1220_still_waiting():
    state = {
        "new_prediction": False,
        "prediction_available": False,
        "pending_signal_session": "2026-08-06",
        "pending_session_has_prediction": False,
        "current_refresh_status": "PENDING_OR_MISSING",
        "operational_status": "WARNING",
        "matured_20d": 0,
    }
    action, should, _ = decide_notification(state, now_kst=datetime(2026, 8, 6, 12, 20, tzinfo=KST))
    assert action == "WAITING_FOR_FEATURES_NO_DISCORD"
    assert should is False


def test_no_prediction_final_retry_warning():
    state = {
        "new_prediction": False,
        "prediction_available": False,
        "pending_signal_session": "2026-08-06",
        "pending_session_has_prediction": False,
        "current_refresh_status": "PENDING_OR_MISSING",
        "operational_status": "WARNING",
        "latest_completed_qqq_session": "2026-08-06",
        "threshold": 0.45,
        "strict_sessions": 3,
        "matured_20d": 0,
        "minimum_sessions": 126,
        "target_sessions": 252,
        "current_action": "KEEP_COLLECTING_PROSPECTIVE_OBSERVATIONS",
        "research_readiness": "COLLECTING_EARLY",
    }
    action, should, key = decide_notification(state, now_kst=datetime(2026, 8, 6, 15, 20, tzinfo=KST))
    assert action == "NO_STRICT_PREDICTION_WARNING"
    assert should is True
    assert key and "NO_STRICT_PREDICTION_AVAILABLE" in key
    msg = build_message({**state, "notification_action": action})
    assert "NOT_AVAILABLE" in msg
    assert "NO_STRICT_PREDICTION_GENERATED" in msg
    assert "n/a" not in msg


def test_new_strict_prediction_sends_once():
    state = {
        "new_prediction": True,
        "prediction_available": True,
        "prediction_session": "2026-08-04",
        "locked_signal": "FLAT",
        "probability": EXPECTED_PROB,
        "dual_signal": "LONG",
        "prediction_provenance": "STRICT_PROSPECTIVE",
        "current_refresh_status": "COMPLETE",
        "feature_completeness": "COMPLETE",
        "operational_status": "NORMAL",
        "model_hash": MODEL_HASH,
        "threshold": 0.45,
        "strict_sessions": 3,
        "matured_20d": 0,
        "minimum_sessions": 126,
        "target_sessions": 252,
        "current_action": "KEEP_COLLECTING_PROSPECTIVE_OBSERVATIONS",
        "research_readiness": "COLLECTING_EARLY",
        "data_quality": "PASS",
        "alignment": "PASS",
    }
    action, should, key = decide_notification(state)
    assert action == "NEW_STRICT_PREDICTION"
    assert should is True
    assert key == prediction_dedupe_key(state)
    res = maybe_notify_from_state(state, dry_run=True)
    assert res["status"] in ("DRY_RUN", "ALREADY_SENT_NO_ACTION")
    assert "FLAT" in (res.get("content_preview") or build_message(state))


def test_same_prediction_duplicate_blocked():
    state = {
        "new_prediction": True,
        "prediction_available": True,
        "prediction_session": "2026-08-04",
        "locked_signal": "FLAT",
        "probability": EXPECTED_PROB,
        "dual_signal": "LONG",
        "prediction_provenance": "STRICT_PROSPECTIVE",
        "model_hash": MODEL_HASH,
        "threshold": 0.45,
        "strict_sessions": 3,
        "matured_20d": 0,
        "minimum_sessions": 126,
        "target_sessions": 252,
        "current_action": "KEEP_COLLECTING_PROSPECTIVE_OBSERVATIONS",
        "research_readiness": "COLLECTING_EARLY",
        "data_quality": "PASS",
        "alignment": "PASS",
        "feature_completeness": "COMPLETE",
        "current_refresh_status": "COMPLETE",
    }
    key = prediction_dedupe_key(state)
    # Pre-seed dedupe
    path = DIAG_ROOT / "discord/dedupe_keys.json"
    keys = json.loads(path.read_text()) if path.exists() else {}
    keys[key] = {"sent_at_utc": "2026-08-05T00:20:00+00:00"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keys, indent=2))
    res = maybe_notify_from_state(state, dry_run=False)
    assert res["status"] == "ALREADY_SENT_NO_ACTION"


def test_signal_changed_new_fingerprint():
    a = prediction_dedupe_key(
        {
            "prediction_session": "2026-08-04",
            "locked_signal": "FLAT",
            "probability": EXPECTED_PROB,
            "dual_signal": "LONG",
            "prediction_provenance": "STRICT_PROSPECTIVE",
            "model_hash": MODEL_HASH,
        }
    )
    b = prediction_dedupe_key(
        {
            "prediction_session": "2026-08-04",
            "locked_signal": "LONG",
            "probability": EXPECTED_PROB,
            "dual_signal": "FLAT",
            "prediction_provenance": "STRICT_PROSPECTIVE",
            "model_hash": MODEL_HASH,
        }
    )
    assert a != b


def test_matured_20d_triggers():
    state = {
        "new_prediction": False,
        "prediction_available": True,
        "locked_signal": "FLAT",
        "probability": EXPECTED_PROB,
        "prediction_provenance": "STRICT_PROSPECTIVE",
        "prediction_session": "2026-08-04",
        "current_refresh_status": "IDEMPOTENT_REFRESH_NO_NEW_PREDICTION",
        "operational_status": "NORMAL",
        "matured_20d": 1,
        "matured_20d_new": 1,
        "latest_completed_qqq_session": "2026-08-04",
        "model_hash": MODEL_HASH,
    }
    action, should, key = decide_notification(state)
    assert action == "MATURED_20D"
    assert should is True
    assert key.startswith("QQQ_MATURED_20D_V1:")


def test_pending_does_not_overwrite_ledger_row():
    before = hashlib.sha256((PROSP_ROOT / "predictions/strict_predictions.parquet").read_bytes()).hexdigest()
    pred = get_latest_valid_prediction_for_session(_ledger(), "2026-08-04")
    assert pred["signal"] == "FLAT"
    after = hashlib.sha256((PROSP_ROOT / "predictions/strict_predictions.parquet").read_bytes()).hexdigest()
    assert before == after


def test_pending_does_not_decrement_strict_sessions():
    n = int(len(_ledger()))
    status = build_status_payload(compact=True)
    assert int(status["strict_sessions"]) == n == 3


def test_2026_08_04_prediction_immutable_fields():
    pred = get_latest_valid_prediction_for_session(_ledger(), "2026-08-04")
    assert pred["signal_session"] == "2026-08-04"
    assert pred["signal"] == "FLAT"
    assert abs(pred["probability"] - EXPECTED_PROB) < 1e-12
    assert pred["dual"] == "LONG"
    assert pred["provenance"] == "STRICT_PROSPECTIVE"
    assert int(len(_ledger())) == 3


def test_hashes_and_flags_unchanged():
    assert hashlib.sha256(ORIG_LOCK.read_bytes()).hexdigest() == LOCK_HASH
    snap = json.loads((PROSP_ROOT / "manifests/locked_candidate_snapshot.json").read_text())
    assert snap["fixed_prospective_threshold"] == 0.45
    assert snap["feature_order_hash"] == FEATURE_HASH
    model = PROSP_ROOT / "models/qqq_regime_locked_model.joblib"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == MODEL_HASH
    status = build_status_payload(compact=True)
    assert status["production_ready"] is False
    assert status["promotion_ready"] is False
    assert status["execution_enabled"] is False
    assert status["locked_signal"] == "FLAT"
    assert abs(float(status["probability"]) - EXPECTED_PROB) < 1e-12


def test_dedupe_key_deterministic():
    state = {
        "prediction_session": "2026-08-04",
        "locked_signal": "FLAT",
        "probability": EXPECTED_PROB,
        "dual_signal": "LONG",
        "prediction_provenance": "STRICT_PROSPECTIVE",
        "model_hash": MODEL_HASH,
    }
    assert prediction_dedupe_key(state) == prediction_dedupe_key(dict(state))


def test_launchd_schedule_three_slots():
    plist = Path.home() / "Library/LaunchAgents/com.canbit.qqq-regime-prospective-daily.plist"
    text = plist.read_text()
    assert "<integer>9</integer>" in text or ">9<" in text
    assert "9" in text and "12" in text and "15" in text
    assert "20" in text
    # parse via plutil-equivalent simple check
    assert text.count("<key>Hour</key>") == 3


def test_latest_valid_helper():
    latest = get_latest_valid_strict_prediction(_ledger())
    assert latest is not None
    assert latest["signal_session"] == "2026-08-04"
