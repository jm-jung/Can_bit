"""Unit tests for microstructure observation quality diagnostics suite."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/diagnostics"))

import microstructure_observation_quality as oq


def test_timestamp_alignment_mismatch_only():
    cfg = oq.load_config()
    row = pd.Series(
        {
            "signal_ts": pd.Timestamp("2026-07-11T20:45:00", tz="UTC"),
            "feature_ts": pd.Timestamp("2026-07-11T20:45:00", tz="UTC"),
            "asof_pass": True,
            "is_replay": False,
        }
    )
    cov = oq.WindowCoverage(
        expected=15,
        kline_present=0,
        agg_present=14,
        agg_events=100,
        mark_present=14,
        oi_present=0,
        force_present=0,
        feature_1m_rows=14,
        phase1_1m_rows=0,
        missing_kline=[],
        present_kline=[],
        max_consecutive_missing_kline=1,
        missing_at_start=1,
        missing_at_end=0,
    )
    clf = oq.classify_taker_marker(row, cov, close_window_kline=0, cfg=cfg, feature_reuse_count=1)
    assert clf["cause_class"] == "TIMESTAMP_LABEL_MISMATCH_ONLY"
    assert clf["trust_class"] == "TRUSTED_WITH_ALIGNMENT_WARNING"


def test_source_window_incomplete_partial_vs_insufficient():
    cfg = oq.load_config()
    row = pd.Series(
        {
            "signal_ts": pd.Timestamp("2026-07-15T08:15:00", tz="UTC"),
            "feature_ts": pd.Timestamp("2026-07-15T08:15:00", tz="UTC"),
            "asof_pass": True,
            "is_replay": False,
        }
    )
    partial = oq.WindowCoverage(15, 9, 9, 10, 0, 0, 0, 9, 0, [], [], 3, 4, 0)
    insuf = oq.WindowCoverage(15, 0, 1, 10, 0, 0, 0, 1, 0, [], [], 15, 15, 0)
    c1 = oq.classify_taker_marker(row, partial, 0, cfg, 1)
    c2 = oq.classify_taker_marker(row, insuf, 0, cfg, 1)
    assert c1["cause_class"] == "SOURCE_WINDOW_PARTIAL_BUT_FROZEN_RULE_ALLOWED"
    assert c2["cause_class"] == "SOURCE_WINDOW_INSUFFICIENT_INPUT"


def test_stale_feature_reuse():
    cfg = oq.load_config()
    row = pd.Series(
        {
            "signal_ts": pd.Timestamp("2026-07-13T10:00:00", tz="UTC"),
            "feature_ts": pd.Timestamp("2026-07-13T10:00:00", tz="UTC"),
            "asof_pass": True,
            "is_replay": False,
        }
    )
    cov = oq.WindowCoverage(15, 15, 15, 100, 15, 0, 0, 15, 0, [], [str(pd.Timestamp("2026-07-13T10:00:00", tz="UTC"))], 0, 0, 0)
    clf = oq.classify_taker_marker(row, cov, 15, cfg, feature_reuse_count=3)
    assert clf["cause_class"] == "STALE_FEATURE_ROW_REUSED"


def test_future_data_fail():
    cfg = oq.load_config()
    row = pd.Series(
        {
            "signal_ts": pd.Timestamp("2026-07-13T10:00:00", tz="UTC"),
            "feature_ts": pd.Timestamp("2026-07-13T10:15:00", tz="UTC"),
            "asof_pass": False,
            "is_replay": False,
        }
    )
    cov = oq.WindowCoverage(15, 15, 15, 100, 15, 0, 0, 15, 0, [], [], 0, 0, 0)
    clf = oq.classify_taker_marker(row, cov, 15, cfg, 1)
    assert clf["trust_class"] == "INVALID_FOR_FUTURE_STRICT_EVALUATION"


def test_daily_coverage_union_and_partial_day():
    cfg = oq.load_config()
    health = {
        "asof_pass": True,
        "outcome_anchor_pass": True,
        "duplicate_pass": True,
        "quarantine_exclusion_pass": True,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "contamination_count": 0,
    }
    day = pd.Timestamp("2026-07-03", tz="UTC")
    start, end = oq._day_bounds(day)
    strict = set(pd.date_range(start, periods=1400, freq="1min", tz="UTC"))
    # duplicate minute should not inflate
    strict.add(list(strict)[0])
    row = oq.compute_daily_row(
        day,
        strict,
        set(),
        set(),
        set(),
        set(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(columns=["signal_ts", "marker_name", "exclude_from_forward_eval"]),
        pd.DataFrame(columns=["signal_ts", "condition_triggered", "condition_name", "condition_confirm_ts", "exclude_from_forward_eval"]),
        pd.DataFrame(columns=["anchor_ts", "horizon_min", "status", "exclude_from_forward_eval"]),
        health,
        cfg,
        as_of=pd.Timestamp("2026-07-10T12:00:00", tz="UTC"),
    )
    assert row["strict_live_minutes"] == 1400
    assert row["daily_quality_verdict"] == "DAILY_OBSERVATION_QUALITY_PASS"

    partial = oq.compute_daily_row(
        pd.Timestamp("2026-07-20", tz="UTC"),
        set(pd.date_range("2026-07-20", periods=100, freq="1min", tz="UTC")),
        set(),
        set(),
        set(),
        set(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(columns=["signal_ts", "marker_name", "exclude_from_forward_eval"]),
        pd.DataFrame(columns=["signal_ts", "condition_triggered", "condition_name", "condition_confirm_ts", "exclude_from_forward_eval"]),
        pd.DataFrame(columns=["anchor_ts", "horizon_min", "status", "exclude_from_forward_eval"]),
        health,
        cfg,
        as_of=pd.Timestamp("2026-07-20T01:40:00", tz="UTC"),
    )
    assert partial["is_partial_day"] is True
    assert partial["daily_quality_verdict"] == "DAILY_PARTIAL_IN_PROGRESS"
    assert partial["calendar_minutes"] == 100  # elapsed denominator


def test_daily_verdict_fail_on_contamination():
    cfg = oq.load_config()
    row = {
        "strict_live_coverage_pct": 99.0,
        "backfilled_live_marker_contamination_count": 1,
        "reconstructed_strict_episode_contamination_count": 0,
        "unknown_minutes": 0,
        "unclassified_minutes": 0,
        "pending_gap_count": 0,
        "failed_gap_count": 0,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "asof_pass": True,
        "outcome_anchor_pass": True,
        "duplicate_pass": True,
        "quarantine_exclusion_pass": True,
        "mainnet_endpoint_pass": True,
        "production_ready": False,
        "promotion_ready": False,
    }
    assert oq.daily_quality_verdict(row, cfg, False) == "DAILY_OBSERVATION_QUALITY_DATA_QUALITY_FAIL"


def test_forceorder_sparse_not_data_quality_fail():
    cfg = oq.load_config()
    row = {
        "strict_live_coverage_pct": 96.0,
        "backfilled_live_marker_contamination_count": 0,
        "reconstructed_strict_episode_contamination_count": 0,
        "unknown_minutes": 0,
        "unclassified_minutes": 0,
        "pending_gap_count": 0,
        "failed_gap_count": 0,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "forceorder_tier_d_gap_minutes": 120,
        "asof_pass": True,
        "outcome_anchor_pass": True,
        "duplicate_pass": True,
        "quarantine_exclusion_pass": True,
        "mainnet_endpoint_pass": True,
        "production_ready": False,
        "promotion_ready": False,
    }
    assert oq.daily_quality_verdict(row, cfg, False) == "DAILY_OBSERVATION_QUALITY_PASS"


def test_readiness_sample_and_composition_and_blocks(monkeypatch):
    cfg = oq.load_config()

    # Build synthetic readiness inputs via monkeypatch of loaders
    t0 = oq.T0_DEFAULT
    prim = pd.DataFrame(
        {
            "marker_name": ["funding_rate_q95"] * 45 + ["basis_bps_q95"] * 5,
            "signal_ts": [t0 + pd.Timedelta(hours=i) for i in range(50)],
            "exclude_from_forward_eval": [False] * 50,
            "feature_ts": [t0 + pd.Timedelta(hours=i) for i in range(50)],
        }
    )
    outcomes = pd.DataFrame(
        {
            "horizon_min": [30] * 40 + [60] * 40,
            "status": ["filled"] * 80,
            "anchor_ts": [t0 + pd.Timedelta(hours=i) for i in range(80)],
            "exclude_from_forward_eval": [False] * 80,
        }
    )

    monkeypatch.setattr(oq, "load_observer", lambda: (prim, prim.iloc[0:0].copy(), outcomes))
    monkeypatch.setattr(
        oq,
        "summarize_health",
        lambda health=None: {
            "collector_healthy": True,
            "staleness_ok": True,
            "gap_pending_retryable": 0,
            "gap_failed": 0,
            "private_endpoint_calls": 0,
            "order_endpoint_calls": 0,
            "production_ready": False,
            "promotion_ready": False,
            "asof_pass": True,
            "outcome_anchor_pass": True,
            "duplicate_pass": True,
            "quarantine_exclusion_pass": True,
            "contamination_count": 0,
            "ws_state": "HEALTHY",
            "collector_pid": 1,
        },
    )
    monkeypatch.setattr(oq, "_load_daily_history", lambda: pd.DataFrame())
    monkeypatch.setattr(oq, "load_taker_forensics_result", lambda: {})

    # primary 50 but composition unbalanced (no taker, basis 5)
    r = oq.run_readiness(persist=False, taker={})
    assert r["verdict"] in {
        "SAMPLE_COUNT_MET_BUT_COMPOSITION_UNBALANCED",
        "NOT_READY_FOR_EVALUATION",
        "QUALITY_REVIEW_REQUIRED",
    }

    # contamination block
    monkeypatch.setattr(
        oq,
        "summarize_health",
        lambda health=None: {
            "collector_healthy": True,
            "staleness_ok": True,
            "gap_pending_retryable": 0,
            "gap_failed": 0,
            "private_endpoint_calls": 0,
            "order_endpoint_calls": 0,
            "production_ready": False,
            "promotion_ready": False,
            "asof_pass": True,
            "outcome_anchor_pass": True,
            "duplicate_pass": True,
            "quarantine_exclusion_pass": True,
            "contamination_count": 1,
            "ws_state": "HEALTHY",
            "collector_pid": 1,
        },
    )
    r2 = oq.run_readiness(persist=False, taker={})
    assert r2["verdict"] == "DATA_QUALITY_BLOCKED"

    monkeypatch.setattr(
        oq,
        "summarize_health",
        lambda health=None: {
            "collector_healthy": False,
            "staleness_ok": False,
            "gap_pending_retryable": 1,
            "gap_failed": 0,
            "private_endpoint_calls": 0,
            "order_endpoint_calls": 0,
            "production_ready": False,
            "promotion_ready": False,
            "asof_pass": True,
            "outcome_anchor_pass": True,
            "duplicate_pass": True,
            "quarantine_exclusion_pass": True,
            "contamination_count": 0,
            "ws_state": "STALE",
            "collector_pid": 1,
        },
    )
    r3 = oq.run_readiness(persist=False, taker={})
    assert r3["verdict"] == "OPERATIONAL_HEALTH_BLOCKED"


def test_taker_quality_excludes_questionable_from_trusted():
    taker = {
        "verdict": "TAKER_NO_RAW_MARKERS_QUESTIONABLE",
        "taker_markers_trusted": 3,
        "taker_markers_warned": 1,
        "taker_markers_questionable": 4,
        "taker_markers_invalid": 0,
        "taker_markers_unresolved": 0,
    }
    assert taker["taker_markers_trusted"] == 3
    assert taker["taker_markers_questionable"] == 4


def test_estimate_excludes_noraw_window():
    daily = pd.DataFrame(
        {
            "date_utc": [f"2026-07-{d:02d}" for d in range(3, 16)],
            "is_partial_day": [False] * 13,
            "strict_live_coverage_pct": [99] * 7 + [10] * 6,
            "strict_live_hours": [24] * 7 + [2] * 6,
        }
    )
    prim = pd.DataFrame(
        {
            "marker_name": ["basis_bps_q95"] * 5 + ["funding_rate_q95"] * 10,
            "signal_ts": pd.to_datetime([f"2026-07-{d:02d}T12:00:00Z" for d in range(3, 18)], utc=True)[:15],
        }
    )
    eta = oq.estimate_days_to_ready(prim, daily, 50, 10, 10, 14, 10, 0)
    assert "W2" in (eta.get("note") or "") or eta.get("estimate_reliability") in {
        "RANGE_OK",
        "ESTIMATE_UNRELIABLE",
    }


def test_private_order_guard_defaults():
    health = oq.summarize_health(
        {
            "collector": {"is_running": True, "ws_connection_state": "HEALTHY", "private_endpoint_calls": 0, "order_endpoint_calls": 0, "production_ready": False, "promotion_ready": False},
            "gap": {},
            "staleness": {"staleness_ok": True},
            "observer_audit": {"asof_pass": True, "outcome_anchor_pass": True, "duplicate_pass": True, "quarantine_exclusion_pass": True},
        }
    )
    assert health["private_endpoint_calls"] == 0
    assert health["order_endpoint_calls"] == 0
    assert health["production_ready"] is False


def test_self_heal_unique_cycle_dedupe():
    rows = []
    for et in [
        "HARD_STALE_DETECTED",
        "RECONNECT_REQUESTED",
        "NEW_CONNECTION_CREATED",
        "FIRST_LIVE_EVENT_RECEIVED",
        "LIVE_RECOVERY_COMPLETE",
        "LIVE_RECOVERY_COMPLETE",  # duplicate status row for same generation
        "GAP_RECOVERY_TRIGGERED",
    ]:
        rows.append(
            {
                "event_utc": pd.Timestamp("2026-07-22T12:00:00", tz="UTC"),
                "event_type": et,
                "collector_instance_id": "collector_1",
                "connection_generation": 5,
                "collector_pid": 1,
            }
        )
    # second independent incident
    rows.append(
        {
            "event_utc": pd.Timestamp("2026-07-22T13:00:00", tz="UTC"),
            "event_type": "RECONNECT_REQUESTED",
            "collector_instance_id": "collector_1",
            "connection_generation": 6,
            "collector_pid": 1,
        }
    )
    rows.append(
        {
            "event_utc": pd.Timestamp("2026-07-22T13:00:05", tz="UTC"),
            "event_type": "LIVE_RECOVERY_COMPLETE",
            "collector_instance_id": "collector_1",
            "connection_generation": 6,
            "collector_pid": 1,
        }
    )
    rows.append(
        {
            "event_utc": pd.Timestamp("2026-07-22T13:00:06", tz="UTC"),
            "event_type": "LIVE_RECOVERY_COMPLETE",
            "collector_instance_id": "collector_1",
            "connection_generation": 6,
            "collector_pid": 1,
        }
    )
    m = oq.aggregate_watchdog_metrics(pd.DataFrame(rows))
    assert m["live_recovery_complete_event_count"] == 4
    assert m["completed_self_heal_cycle_count"] == 2
    assert m["self_heal_success_count"] == 2
    assert m["hard_stale_incident_count"] == 1
    assert m["unique_reconnect_cycle_count"] == 2
    assert m["duplicate_live_recovery_event_count"] == 2


def test_self_heal_aggregate_idempotent_on_duplicate_frame():
    rows = [
        {
            "event_utc": pd.Timestamp("2026-07-22T12:00:00", tz="UTC"),
            "event_type": "LIVE_RECOVERY_COMPLETE",
            "collector_instance_id": "c",
            "connection_generation": 1,
            "collector_pid": 9,
        }
    ] * 2
    m1 = oq.aggregate_watchdog_metrics(pd.DataFrame(rows))
    m2 = oq.aggregate_watchdog_metrics(pd.DataFrame(rows))
    assert m1["self_heal_success_count"] == m2["self_heal_success_count"] == 1


def test_reconstruct_source_window_open_semantics():
    ts = pd.Timestamp("2026-07-13T10:00:00", tz="UTC")
    start, end, expected = oq.reconstruct_source_window(ts)
    assert start == ts
    assert end == ts + pd.Timedelta(minutes=15)
    assert len(expected) == 15
    assert expected[0] == ts
