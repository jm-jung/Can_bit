"""Tests for corrected same-period QQQ selection v2."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.config import QQQConfig, paths
from canbit_equity.features import logistic_feature_columns
from canbit_equity.selection.same_period import (
    CANDIDATES,
    EXPECTED_CONFIG,
    EXPECTED_FEATURE,
    EXPECTED_LABELED,
    DIAG,
    run_corrected_selection,
    verify_fixed_input,
)
from canbit_equity.walkforward import _slice_folds


def test_fixed_labeled_hash():
    p = paths()["features"] / "qqq_daily_features_labeled.parquet"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == EXPECTED_LABELED


def test_stale_intermediate_not_used():
    out = verify_fixed_input()
    assert out["stale_intermediate_used"] is False
    assert out["stale_features_parquet_rows"] == 68
    assert out["ok"] is True


def test_corrected_selection_common_period():
    # Use existing artifact if present from --full; else run (may be slow once)
    art = paths()["models"] / "qqq_corrected_same_period_selection_v2.json"
    if not art.exists():
        run_corrected_selection(QQQConfig())
    a = json.loads(art.read_text())
    assert a["selection_version"] == "SAME_PERIOD_V2"
    assert a["old_holdout_used"] is False
    assert a["cost_scenario"] == "BASE_5bps"
    table = a["candidate_table"]
    assert {r["candidate"] for r in table} == set(CANDIDATES)
    starts = {r["common_start"] for r in table}
    ends = {r["common_end"] for r in table}
    rows = {r["common_rows"] for r in table}
    assert len(starts) == 1 and len(ends) == 1 and len(rows) == 1
    assert a["common_oos_rows"] == rows.pop()


def test_old_holdout_rows_absent_from_common_index():
    idx = pd.read_parquet(DIAG / "folds/common_oos_row_index.parquet")
    labeled = pd.read_parquet(paths()["features"] / "qqq_daily_features_labeled.parquet").reset_index(drop=True)
    _, ho = _slice_folds(len(labeled), QQQConfig())
    assert int((idx["global_index"] >= ho["start"]).sum()) == 0
    assert pd.to_datetime(idx["session_date"]).max() < pd.Timestamp("2024-06-27")


def test_threshold_set_fixed():
    cfg = QQQConfig()
    assert list(cfg.lr_thresholds) == [0.45, 0.50, 0.55, 0.60]


def test_legacy_lock_untouched():
    lock = paths()["holdout_lock"]
    assert lock.exists()
    data = json.loads(lock.read_text())
    assert data.get("candidate_id") == "DUAL_TREND_FILTER"


def test_legacy_walkforward_not_overwritten_hash_stable():
    # file still exists and still contains legacy unequal ranking metadata
    wf = json.loads((paths()["reports"] / "qqq_walkforward_report.json").read_text())
    assert wf["selected_candidate"]["name"] == "DUAL_TREND_FILTER"


def test_score_formula_and_sort():
    art = json.loads((paths()["models"] / "qqq_corrected_same_period_selection_v2.json").read_text())
    table = art["candidate_table"]
    for r in table:
        expected = (r["Sharpe"] or 0) + 0.01 * (r["MDD_improvement_percent"] or 0) + 0.001 * (
            r["CAGR_preservation_percent"] or 0
        )
        assert abs(r["score"] - expected) < 1e-9
    scores = [r["score"] for r in sorted(table, key=lambda x: x["rank"])]
    assert scores == sorted(scores, reverse=True)


def test_config_feature_hashes():
    assert hashlib.sha256((REPO / "src/canbit_equity/config.py").read_bytes()).hexdigest() == EXPECTED_CONFIG
    assert hashlib.sha256(",".join(logistic_feature_columns()).encode()).hexdigest() == EXPECTED_FEATURE


def test_production_promotion_false():
    c = json.loads((DIAG / "reports/qqq_same_period_selection_compact.json").read_text())
    assert c["production_ready"] is False
    assert c["promotion_ready"] is False


def test_regime_gate_prefers_v2():
    sys.path.insert(0, str(REPO / "scripts/equity_etf"))
    from run_qqq_regime_research_suite import same_period_audit

    out = same_period_audit()
    assert out["verdict"] == "SAME_PERIOD_COMPARISON_PASS"
    assert out.get("legacy_fallback_used") is False or out["checks"]["legacy_fallback_used"] is False
    assert out["selection_version"] == "SAME_PERIOD_V2"
