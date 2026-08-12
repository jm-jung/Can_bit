"""Same-period gate tests — expects SAME_PERIOD_V2 after correction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/equity_etf"))

from canbit_equity.config import QQQConfig, paths
from canbit_equity.walkforward import _slice_folds


def test_frozen_labeled_hash():
    import hashlib

    p = paths()["features"] / "qqq_daily_features_labeled.parquet"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == "5070477672a866749a2e2b7674a2029c1fb5b461e349bd3dc52d9e4e337d0bf5"


def test_stale_68_row_features_not_research_input():
    p = paths()
    labeled = pd.read_parquet(p["features"] / "qqq_daily_features_labeled.parquet")
    feat = pd.read_parquet(p["features_file"])
    assert len(labeled) > 6000
    assert len(feat) < 100


def test_baseline_legacy_ranking_windows_were_unequal():
    """Documents the historical bug (legacy walkforward still on disk)."""
    labeled = pd.read_parquet(paths()["features"] / "qqq_daily_features_labeled.parquet").reset_index(drop=True)
    _, holdout_idx = _slice_folds(len(labeled), QQQConfig())
    dev = labeled.iloc[: holdout_idx["start"]]
    n = len(dev)
    te = dev.iloc[int(n * 0.85) :]
    assert len(dev) != len(te)


def test_same_period_audit_cli_passes_with_v2():
    v2 = paths()["models"] / "qqq_corrected_same_period_selection_v2.json"
    assert v2.exists(), "run corrected selection first"
    from run_qqq_regime_research_suite import same_period_audit

    out = same_period_audit()
    assert out["verdict"] == "SAME_PERIOD_COMPARISON_PASS"
    assert out["continue_regime_research"] is True
    assert out["selection_version"] == "SAME_PERIOD_V2"


def test_production_promotion_false_in_compact():
    path = REPO / "data/diagnostics/equity_etf_qqq_regime/reports/qqq_regime_compact_status.json"
    assert path.exists()
    c = json.loads(path.read_text())
    assert c["production_ready"] is False
    assert c["promotion_ready"] is False
