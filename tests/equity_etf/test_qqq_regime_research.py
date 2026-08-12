"""Core regime research tests (implementation + gate wiring)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.config import paths as qqq_paths
from canbit_equity.regime.config import (
    ALL_REGIME_FEATURES,
    CANDIDATE_FEATURE_GROUPS,
    EXPECTED_LABELED_HASH,
    FRED_SERIES,
    LR_THRESHOLDS,
    V2_PATH,
    YF_SERIES,
)


def test_fixed_labeled_hash():
    p = qqq_paths()["features"] / "qqq_daily_features_labeled.parquet"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == EXPECTED_LABELED_HASH


def test_stale_features_not_used_as_input():
    feat = qqq_paths()["features_file"]
    assert feat.exists()
    import pandas as pd

    assert len(pd.read_parquet(feat)) < 100


def test_v2_artifact_required():
    assert V2_PATH.exists()
    art = json.loads(V2_PATH.read_text())
    assert art["selection_version"] == "SAME_PERIOD_V2"


def test_candidate_universe_fixed():
    assert set(CANDIDATE_FEATURE_GROUPS) == {
        "LOGISTIC_PRICE_ONLY",
        "LOGISTIC_PRICE_PLUS_VOL",
        "LOGISTIC_PRICE_PLUS_RATES",
        "LOGISTIC_PRICE_PLUS_BREADTH",
        "LOGISTIC_PRICE_PLUS_CREDIT",
        "LOGISTIC_PRICE_PLUS_ALL_REGIME",
    }
    assert len(ALL_REGIME_FEATURES) == 10
    assert list(LR_THRESHOLDS) == [0.45, 0.50, 0.55, 0.60]


def test_series_universe_fixed():
    assert set(YF_SERIES) == {"^VIX", "SPY", "RSP", "QEW", "SMH", "HYG", "IEF"}
    assert set(FRED_SERIES) == {"DGS10", "DGS2", "T10Y2Y"}


def test_full_suite_calls_research_not_gate_only():
    src = (REPO / "scripts/equity_etf/run_qqq_regime_research_suite.py").read_text()
    assert "run_full_regime_research" in src
    assert "External regime download not executed by this command" not in src


def test_production_promotion_false_if_compact_exists():
    path = REPO / "data/diagnostics/equity_etf_qqq_regime/reports/qqq_regime_compact_status.json"
    if path.exists():
        c = json.loads(path.read_text())
        assert c.get("production_ready") is False
        assert c.get("promotion_ready") is False
