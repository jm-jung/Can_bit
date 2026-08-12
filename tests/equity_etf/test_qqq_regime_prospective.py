"""Prospective observer tests."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.regime.prospective.config import DIAG_ROOT, ORIG_LOCK, PROSP_ROOT
from canbit_equity.regime.prospective.health import current_action, readiness_from_counts
from canbit_equity.regime.prospective.lock import audit_lock, resolve_fixed_threshold
from canbit_equity.regime.prospective.model import build_or_load_frozen_model
from canbit_equity.regime.prospective.prediction import classify_provenance
from canbit_equity.backtest import run_backtest, signals_to_positions
from canbit_equity.strategies import signal_buy_and_hold
from datetime import datetime, timezone


def test_original_lock_immutable_hash_stable():
    h1 = hashlib.sha256(ORIG_LOCK.read_bytes()).hexdigest()
    audit_lock()
    h2 = hashlib.sha256(ORIG_LOCK.read_bytes()).hexdigest()
    assert h1 == h2
    assert len(h1) == 64


def test_lock_audit_pass_and_threshold_mode():
    out = audit_lock()
    assert out["verdict"] == "QQQ_PROSPECTIVE_LOCK_PASS"
    assert out["fixed_threshold"] == 0.45
    assert out["threshold_policy"] == "mode_of_locked_development_fold_distribution"
    lock = json.loads(ORIG_LOCK.read_text())
    thr, policy, meta = resolve_fixed_threshold(lock)
    assert thr == 0.45
    assert meta["distribution"]["0.45"] == 8


def test_locked_candidate_exact():
    snap = json.loads((PROSP_ROOT / "manifests/locked_candidate_snapshot.json").read_text())
    assert snap["candidate_id"] == "LOGISTIC_PRICE_PLUS_ALL_REGIME"
    assert snap["feature_group"] == "PRICE_PLUS_ALL_REGIME"
    assert len(snap["exact_feature_order"]) == 28
    assert snap["first_unseen_session"] == "2026-07-31"
    assert snap["production_ready"] is False
    assert snap["promotion_ready"] is False


def test_model_fit_uses_matured_labels_only():
    ma = build_or_load_frozen_model()
    assert ma["verdict"] in ("QQQ_PROSPECTIVE_MODEL_PASS", "QQQ_PROSPECTIVE_MODEL_REBUILT_FROM_LOCK")
    assert ma["unlabeled_rows_used_for_fit"] == 0
    assert ma["model_retrained_daily"] is False
    assert ma["actual_model_fit_end"] <= "2026-07-01"
    assert ma["label_matured_fit_end"] == ma["actual_model_fit_end"]
    # second call must not retrain
    ma2 = build_or_load_frozen_model()
    assert ma2["verdict"] == "QQQ_PROSPECTIVE_MODEL_PASS"
    assert ma2["model_hash"] == ma["model_hash"]


def test_no_pre_lock_prediction_in_strict_ledger():
    path = PROSP_ROOT / "predictions/strict_predictions.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if len(df):
            assert pd.to_datetime(df["signal_session"]).min() >= pd.Timestamp("2026-07-31")


def test_production_flags_false():
    state = PROSP_ROOT / "state/prospective_state.json"
    if state.exists():
        s = json.loads(state.read_text())
        assert s["production_ready"] is False
        assert s["promotion_ready"] is False
        assert s.get("execution_enabled") is False
        assert s.get("private_calls", 0) == 0
        assert s.get("order_calls", 0) == 0


def test_readiness_waiting_and_no_auto_promotion():
    assert readiness_from_counts(0, 0, 1.0, False) == "WAITING_FOR_FIRST_UNSEEN_SESSION"
    assert current_action("MINIMUM_RESEARCH_REVIEW_READY", "NORMAL") == "MANUAL_FORENSIC_REVIEW_REQUIRED"
    r = readiness_from_counts(126, 60, 0.95, True)
    assert r == "MINIMUM_RESEARCH_REVIEW_READY"


def test_strict_vs_late_deadline():
    next_open = datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc)
    assert classify_provenance(datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc), next_open) == "STRICT_PROSPECTIVE"
    assert classify_provenance(datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc), next_open) == "LATE_PROVIDER_REPLAY"


def test_backtest_no_short_no_leverage_tplus1():
    df = pd.DataFrame(
        {
            "session_date": pd.date_range("2026-01-01", periods=5, freq="B"),
            "open_adj": [100, 101, 102, 103, 104],
            "close_adj": [101, 102, 103, 104, 105],
        }
    )
    sig = pd.Series([1, 1, 0, 1, 1], dtype=float)
    pos = signals_to_positions(sig)
    assert pos.iloc[0] == 0.0
    assert pos.iloc[1] == 1.0
    bt = run_backtest(df, sig, cost_bps_per_side=5.0)
    assert (bt["position"] <= 1.0).all()
    assert (bt["position"] >= 0.0).all()
    assert bt["equity"].notna().all()


def test_buy_hold_comparator_exists():
    df = pd.DataFrame(
        {
            "session_date": pd.date_range("2026-01-01", periods=3, freq="B"),
            "open_adj": [100.0, 101.0, 102.0],
            "close_adj": [100.5, 101.5, 102.5],
        }
    )
    bt = run_backtest(df, signal_buy_and_hold(df), cost_bps_per_side=5.0)
    assert bt["long_exposure"] > 0


def test_launchagent_plist_three_schedules_no_keepalive():
    plist = DIAG_ROOT / "launchd/com.canbit.qqq-regime-prospective-daily.plist"
    text = plist.read_text()
    assert "<integer>9</integer>" in text
    assert "<integer>12</integer>" in text
    assert "<integer>15</integer>" in text
    assert text.count("<integer>20</integer>") >= 3
    assert "KeepAlive" in text and "RunAtLoad" in text
    assert text.count("<false/>") >= 2


def test_wrapper_uses_caffeinate():
    wrapper = REPO / "scripts/runtime/run_qqq_regime_prospective_daily.sh"
    text = wrapper.read_text()
    assert "caffeinate -i" in text
    assert ".venv/bin/python" in text


def test_research_helpers_still_importable():
    from canbit_equity.regime.prospective import audit_external_data, research_verdict, write_prospective_lock

    assert callable(audit_external_data)
    assert callable(research_verdict)
    assert callable(write_prospective_lock)


def test_discord_secret_not_in_reports():
    for path in DIAG_ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".txt", ".log", ".jsonl"}:
            text = path.read_text(errors="ignore")
            assert "discord.com/api/webhooks/" not in text


def test_unresolved_threshold_policy_raises():
    import pytest
    from canbit_equity.regime.prospective.lock import resolve_fixed_threshold

    with pytest.raises(RuntimeError, match="THRESHOLD_UNRESOLVED"):
        resolve_fixed_threshold({"selected_development_threshold_distribution": {}})


def test_status_freshness_fields_without_update():
    from canbit_equity.regime.prospective.status import build_status_payload

    before_state = (PROSP_ROOT / "state/prospective_state.json").read_text()
    payload = build_status_payload(compact=True)
    after_state = (PROSP_ROOT / "state/prospective_state.json").read_text()
    assert before_state == after_state
    assert payload.get("status_does_not_run_update") is True
    assert "status_checked_at_utc" in payload
    assert "state_generated_at_utc" in payload or "underlying_state_generated_at_utc" in payload
    assert "state_age_hours" in payload
    assert "state_fresh" in payload


def test_forward_sessions_clamped_to_calendar():
    from canbit_equity.calendar import get_calendar
    from canbit_equity.regime.prospective.outcomes import _forward_sessions_after

    cal = get_calendar("XNYS")
    # Near calendar end: must not raise DateOutOfBounds
    last = pd.Timestamp(pd.Timestamp(cal.last_session).date())
    near = last - pd.Timedelta(days=30)
    fwd = _forward_sessions_after(cal, near, min_sessions=20)
    assert len(fwd) >= 1
    assert pd.Timestamp(pd.Timestamp(fwd[-1]).date()) <= last


def test_july31_strict_ledger_not_reclassified_if_preexisted():
    path = PROSP_ROOT / "predictions/strict_predictions.parquet"
    df = pd.read_parquet(path)
    if len(df) == 0:
        return
    row = df[pd.to_datetime(df["signal_session"]).dt.normalize() == pd.Timestamp("2026-07-31")]
    if len(row) == 0:
        return
    r = row.iloc[0]
    assert r["provenance_tier"] == "STRICT_PROSPECTIVE"
    gen = pd.Timestamp(r["generated_at_utc"])
    nxt_open = pd.Timestamp(r["next_session_open_utc"])
    assert bool(r["generated_before_next_open"]) is True
    assert gen < nxt_open


def test_wrapper_absolute_paths_and_executable():
    wrapper = REPO / "scripts/runtime/run_qqq_regime_prospective_daily.sh"
    assert wrapper.exists()
    assert wrapper.stat().st_mode & 0o111
    text = wrapper.read_text()
    assert "/Users/jeongminjun/Projects/Can_bit" in text
    assert "PYTHONPATH" in text
    assert text.startswith("#!/bin/bash")


def test_threshold_still_fixed():
    man = json.loads((PROSP_ROOT / "models/qqq_regime_locked_model_manifest.json").read_text())
    assert man["fixed_threshold"] == 0.45
    assert man["model_retrained_daily"] is False
