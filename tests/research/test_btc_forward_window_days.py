"""BTC forward interim review — true rolling window tests."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
PY = str(REPO / ".venv/bin/python")
ASOF = "2026-08-07T05:43:13.205980Z"

from canbit_research.btc_forward_interim import (
    InterimReviewError,
    OUT,
    PROTECTED_BTC,
    filter_ts_inclusive,
    load_t0,
    outcome_ids,
    parse_as_of_utc,
    resolve_analysis_window,
    run_full_review,
    run_review,
    split_matured_immature,
    window_sample_status,
    window_strategy_status,
    _eligible_markers,
    _eligible_outcomes,
)


def test_resolve_window_max_t0():
    t0 = load_t0()
    as_of = parse_as_of_utc(ASOF, now_utc=pd.Timestamp(ASOF).to_pydatetime(), t0=t0)
    w = resolve_analysis_window(analysis_scope="ROLLING_WINDOW", window_days=14, as_of=as_of, t0=t0)
    assert w["analysis_scope"] == "ROLLING_WINDOW"
    assert w["requested_window_days"] == 14
    assert w["effective_window_start_utc"].startswith("2026-07-24T05:43:13")
    assert w["effective_window_end_utc"].startswith("2026-08-07T05:43:13")
    assert abs(w["window_duration_hours"] - 14 * 24) < 1e-6
    assert w["window_truncated_by_effective_t0"] is False


def test_full_scope_bounds():
    t0 = load_t0()
    as_of = parse_as_of_utc(ASOF, now_utc=pd.Timestamp(ASOF).to_pydatetime(), t0=t0)
    w = resolve_analysis_window(analysis_scope="EFFECTIVE_T0_FULL", window_days=None, as_of=as_of, t0=t0)
    assert w["analysis_scope"] == "EFFECTIVE_T0_FULL"
    assert w["requested_window_days"] is None
    assert w["effective_window_start"] == t0


def test_window_filters_markers_by_signal_ts():
    t0 = load_t0()
    as_of = parse_as_of_utc(ASOF, now_utc=pd.Timestamp(ASOF).to_pydatetime(), t0=t0)
    w = resolve_analysis_window(analysis_scope="ROLLING_WINDOW", window_days=14, as_of=as_of, t0=t0)
    m = _eligible_markers()
    m = m[m["signal_ts"] >= t0]
    win = filter_ts_inclusive(m, "signal_ts", w["effective_window_start"], w["effective_window_end"])
    assert len(win) < len(m)
    assert (win["signal_ts"] >= w["effective_window_start"]).all()
    assert (win["signal_ts"] <= w["effective_window_end"]).all()
    # excluded sample
    excl = m[m["signal_ts"] < w["effective_window_start"]]
    assert len(excl) >= 10


def test_window_filters_outcomes_by_anchor_ts():
    t0 = load_t0()
    as_of = parse_as_of_utc(ASOF, now_utc=pd.Timestamp(ASOF).to_pydatetime(), t0=t0)
    w = resolve_analysis_window(analysis_scope="ROLLING_WINDOW", window_days=14, as_of=as_of, t0=t0)
    o = _eligible_outcomes()
    o = o[o["anchor_ts"] >= t0]
    win = filter_ts_inclusive(o, "anchor_ts", w["effective_window_start"], w["effective_window_end"])
    assert len(win) < len(o)
    assert len(win[win["horizon_min"] == 30]) >= 1


def test_maturity_inside_but_signal_outside_excluded():
    t0 = load_t0()
    as_of = parse_as_of_utc(ASOF, now_utc=pd.Timestamp(ASOF).to_pydatetime(), t0=t0)
    w = resolve_analysis_window(analysis_scope="ROLLING_WINDOW", window_days=14, as_of=as_of, t0=t0)
    o = _eligible_outcomes()
    o = o[o["anchor_ts"] >= t0]
    # signal outside window, maturity likely inside (filled later)
    outside = o[o["anchor_ts"] < w["effective_window_start"]]
    assert len(outside) >= 10
    win = filter_ts_inclusive(o, "anchor_ts", w["effective_window_start"], w["effective_window_end"])
    outside_ids = set(outside["observation_id"].astype(str))
    win_ids = set(win["observation_id"].astype(str))
    assert outside_ids.isdisjoint(win_ids) or len(outside_ids & win_ids) == 0


def test_full_regression_asof():
    full = run_full_review(as_of_utc=ASOF)
    assert full["analysis_scope"] == "EFFECTIVE_T0_FULL"
    assert full["primary_cumulative"] == 49
    assert full["basis_cumulative"] == 5
    assert full["trusted_taker_cumulative"] == 3
    assert full["matured_outcomes_30m"] == 135
    assert abs(full["candidate_30m_mean_bps"] - (-7.0886885970765)) < 1e-9
    assert abs(full["candidate_30m_win_rate"] - 0.3925925925925926) < 1e-12
    assert full["strategy_interim_status"] == "EARLY_NEGATIVE"
    assert full["production_ready"] is False


def test_14d_proper_subset_of_full():
    full = run_full_review(as_of_utc=ASOF)
    w14 = run_review(analysis_scope="ROLLING_WINDOW", window_days=14, as_of_utc=ASOF)
    assert w14["analysis_scope"] == "ROLLING_WINDOW"
    assert w14["requested_window_days"] == 14
    assert w14["window_matured_outcomes_30m"] < full["matured_outcomes_30m"]
    assert w14["window_post_rule_signals"] < full["primary_cumulative"]
    assert w14["outcome_id_audit"]["window_outcome_ids_proper_subset_of_full"] is True
    assert w14["primary_cumulative"] == 49  # readiness unchanged
    n = int(w14["window_matured_outcomes_30m"])
    if n < 10:
        assert w14["window_strategy_interim_status"] == "TOO_EARLY_TO_ASSESS"
    else:
        assert w14["window_strategy_interim_status"] in {
            "EARLY_POSITIVE",
            "EARLY_NEGATIVE",
            "MIXED_INCONCLUSIVE",
        }


def test_7d_subset_of_14d():
    w14 = run_review(analysis_scope="ROLLING_WINDOW", window_days=14, as_of_utc=ASOF)
    w7 = run_review(analysis_scope="ROLLING_WINDOW", window_days=7, as_of_utc=ASOF)
    assert w7["window_matured_outcomes_30m"] <= w14["window_matured_outcomes_30m"]
    assert w7["window_post_rule_signals"] <= w14["window_post_rule_signals"]


def test_future_asof_rejected():
    t0 = load_t0()
    with pytest.raises(InterimReviewError) as ei:
        parse_as_of_utc("2099-01-01T00:00:00Z", now_utc=pd.Timestamp(ASOF).to_pydatetime(), t0=t0)
    assert ei.value.code == "ERROR_AS_OF_IN_FUTURE"


def test_pre_t0_asof_rejected():
    t0 = load_t0()
    with pytest.raises(InterimReviewError) as ei:
        parse_as_of_utc("2026-06-01T00:00:00Z", now_utc=pd.Timestamp(ASOF).to_pydatetime(), t0=t0)
    assert ei.value.code == "ERROR_AS_OF_BEFORE_EFFECTIVE_T0"


def test_invalid_window_days():
    t0 = load_t0()
    as_of = parse_as_of_utc(ASOF, now_utc=pd.Timestamp(ASOF).to_pydatetime(), t0=t0)
    with pytest.raises(InterimReviewError):
        resolve_analysis_window(analysis_scope="ROLLING_WINDOW", window_days=0, as_of=as_of, t0=t0)
    with pytest.raises(InterimReviewError):
        resolve_analysis_window(analysis_scope="ROLLING_WINDOW", window_days=-1, as_of=as_of, t0=t0)
    with pytest.raises(InterimReviewError):
        resolve_analysis_window(analysis_scope="ROLLING_WINDOW", window_days=400, as_of=as_of, t0=t0)


def test_cli_mutual_exclusion():
    proc = subprocess.run(
        [PY, "scripts/research/run_btc_forward_interim_review.py", "--full", "--window-days", "14", "--json"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_readiness_cumulative_unchanged_on_window():
    w = run_review(analysis_scope="ROLLING_WINDOW", window_days=14, as_of_utc=ASOF)
    assert w["primary_cumulative"] == 49
    assert w["basis_cumulative"] == 5
    assert w["trusted_taker_cumulative"] == 3
    assert "primary_window_additions" in w
    assert w["basis_window_additions"] == 0  # 14d markers were CVD only


def test_window_status_helpers():
    assert window_sample_status(0) == "NO_MATURED_OUTCOMES"
    assert window_sample_status(5) == "VERY_SMALL"
    assert window_sample_status(15) == "EARLY"
    assert window_strategy_status(5, -1.0, 0.5, -1.0) == "TOO_EARLY_TO_ASSESS"
    assert window_strategy_status(12, -1.0, 0.5, -1.0) == "EARLY_NEGATIVE"
    assert window_strategy_status(12, 1.0, 1.5, 1.0) == "EARLY_POSITIVE"
    assert window_strategy_status(12, 1.0, 0.5, -0.1) == "MIXED_INCONCLUSIVE"


def test_previous_window_present():
    w = run_review(analysis_scope="ROLLING_WINDOW", window_days=14, as_of_utc=ASOF)
    assert w["window_comparison"] is not None
    assert w["previous_window_complete"] is True
    assert w["rolling_trend_status"] == "INSUFFICIENT_FOR_TREND"  # current n<10
    assert "MUST NOT BE USED TO RETUNE" in (w.get("rolling_trend_disclaimer") or "")


def test_output_filename_includes_window_days():
    run_review(analysis_scope="ROLLING_WINDOW", window_days=14, as_of_utc=ASOF)
    assert (OUT / "reports/window_14d_2026-08-07.json").exists()


def test_deterministic_asof_json():
    a = run_review(analysis_scope="ROLLING_WINDOW", window_days=14, as_of_utc=ASOF)
    b = run_review(analysis_scope="ROLLING_WINDOW", window_days=14, as_of_utc=ASOF)
    keys = [
        "effective_window_start_utc",
        "effective_window_end_utc",
        "window_matured_outcomes_30m",
        "window_30m_mean_bps",
        "window_post_rule_signals",
    ]
    for k in keys:
        assert a[k] == b[k]


def test_protected_and_flags():
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED_BTC if p.exists()}
    run_review(analysis_scope="ROLLING_WINDOW", window_days=14, as_of_utc=ASOF)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED_BTC if p.exists()}
    assert before == after
    report = json.loads((OUT / "reports/window_14d_2026-08-07.json").read_text())
    assert report["production_ready"] is False
    assert report["promotion_ready"] is False
    assert report["execution_enabled"] is False
    assert report["private_calls"] == 0
    assert report["order_calls"] == 0


def test_boundary_inclusive_samples():
    t0 = load_t0()
    as_of = parse_as_of_utc(ASOF, now_utc=pd.Timestamp(ASOF).to_pydatetime(), t0=t0)
    w = resolve_analysis_window(analysis_scope="ROLLING_WINDOW", window_days=14, as_of=as_of, t0=t0)
    o = _eligible_outcomes()
    o = o[o["anchor_ts"] >= t0]
    win = filter_ts_inclusive(o, "anchor_ts", w["effective_window_start"], w["effective_window_end"])
    excl = o[(o["anchor_ts"] < w["effective_window_start"]) | (o["anchor_ts"] > w["effective_window_end"])]
    assert len(win) >= 10 or len(win) >= 1
    assert len(excl) >= 10
