"""Read-only BTC forward interim review tests."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_research.btc_forward_interim import (
    OUT,
    PROTECTED_BTC,
    derive_effective_t0,
    load_t0,
    outcome_metrics,
    run_full_review,
)

ASOF = "2026-08-07T05:43:13.205980Z"


def test_t0_resolved():
    t0 = load_t0()
    assert str(t0.date()) == "2026-07-02"


def test_effective_t0_equals_original_no_clock_reset():
    info = derive_effective_t0()
    assert info["forward_clock_reset_required"] is False
    assert info["original_strict_live_t0"] == info["current_configuration_effective_t0"]
    assert info["hash_freeze_verdict"] == "HASH_FREEZE_OK"


def test_protected_hashes_unchanged_by_review():
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED_BTC if p.exists()}
    run_full_review(as_of_utc=ASOF)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED_BTC if p.exists()}
    assert before == after


def test_report_flags_false():
    report = run_full_review(as_of_utc=ASOF)
    assert report["production_ready"] is False
    assert report["promotion_ready"] is False
    assert report["execution_enabled"] is False
    assert report["private_calls"] == 0
    assert report["order_calls"] == 0


def test_pre_t0_excluded_from_eligible_markers():
    report = run_full_review(as_of_utc=ASOF)
    assert report["primary_cumulative"] == 49
    assert report["pre_t0_rows_excluded"] == 58


def test_readiness_counts_reproduced():
    report = run_full_review(as_of_utc=ASOF)
    assert report["basis_cumulative"] == 5
    assert report["trusted_taker_cumulative"] == 3
    assert report["readiness_bottleneck"] in {"basis_markers", "taker_trusted"}


def test_no_promotion_verdict():
    report = run_full_review(as_of_utc=ASOF)
    assert report["strategy_interim_status"] in {
        "TOO_EARLY_TO_ASSESS",
        "EARLY_POSITIVE",
        "EARLY_NEGATIVE",
        "MIXED_INCONCLUSIVE",
        "OPERATIONAL_ONLY_NO_TRADES",
        "INTEGRITY_BLOCKED",
    }
    assert "PROMISING" not in report["strategy_interim_status"]
    assert report["production_ready"] is False


def test_outcome_metrics_n_warning():
    df = pd.DataFrame(
        {
            "horizon_min": [30] * 5,
            "net_current_bps": [1.0, -2.0, 0.5, -1.0, 0.2],
        }
    )
    m = outcome_metrics(df, 30)
    assert m["n"] == 5
    assert m["descriptive_only"] is True


def test_q2_comparison_not_forced():
    report = run_full_review(as_of_utc=ASOF)
    assert report["baseline_candidate_common_trade_comparison"] == "NOT_AVAILABLE_DIFFERENT_LEDGERS"
