"""Forensic integrity tests for QQQ research (no performance tuning)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.backtest import run_backtest, signals_to_positions, summarize_equity
from canbit_equity.config import QQQConfig, paths
from canbit_equity.metrics import compare_to_buyhold
from canbit_equity.strategies import signal_dual_trend, signal_buy_and_hold
from canbit_equity.walkforward import _slice_folds, research_verdict


def test_candidate_universe_includes_logistic():
    wf = json.loads((paths()["reports"] / "qqq_walkforward_report.json").read_text())
    ranking_names = [r["name"] for r in wf.get("development_ranking") or []]
    assert "LOGISTIC_REGRESSION" in ranking_names
    assert "DUAL_TREND_FILTER" in ranking_names
    assert "LOGISTIC_REGRESSION" in (wf.get("fold_aggregate") or {})


def test_thresholds_documented():
    cfg = QQQConfig()
    assert list(cfg.lr_thresholds) == [0.45, 0.50, 0.55, 0.60]


def test_selection_deterministic_and_not_holdout():
    src = (REPO / "src/canbit_equity/walkforward.py").read_text()
    # selection happens in run_walkforward before holdout
    assert "ranking_sorted = sorted(ranking" in src
    holdout_fn = src.split("def run_final_holdout")[1].split("def research_verdict")[0]
    assert "ranking_sorted" not in holdout_fn
    assert "rank_key" not in holdout_fn


def test_position_t1_open():
    sig = pd.Series([0.0, 1.0, 1.0, 0.0])
    pos = signals_to_positions(sig)
    assert list(pos) == [0.0, 0.0, 1.0, 1.0]


def test_mdd_improvement_sign():
    # strategy less severe drawdown => positive improvement
    bh, st = -0.226, -0.149
    imp = (abs(bh) - abs(st)) / abs(bh) * 100
    assert imp > 0


def test_cagr_preservation_unit_percent():
    compact = json.loads(paths()["compact_status"].read_text())
    # reported as ~18.95 not 0.1895
    assert compact["cagr_preservation_pct"] > 1.0


def test_holdout_tail_is_forward_horizon():
    p = paths()
    labeled = pd.read_parquet(p["features"] / "qqq_daily_features_labeled.parquet")
    norm = pd.read_parquet(p["normalized_file"])
    labeled_end = pd.to_datetime(labeled["session_date"]).max()
    tail = norm[pd.to_datetime(norm["session_date"]) > labeled_end]
    assert 15 <= len(tail) <= 25  # ~20 trading sessions


def test_verdict_truth_table_inconclusive():
    p = paths()
    holdout = json.loads((p["reports"] / "qqq_final_holdout_report.json").read_text())
    wf = json.loads((p["reports"] / "qqq_walkforward_report.json").read_text())
    v = research_verdict(wf, holdout, "QQQ_DATA_PIPELINE_PASS", "LOOKAHEAD_AUDIT_PASS")
    assert v == "QQQ_BASELINE_RESEARCH_INCONCLUSIVE"


def test_trade_count_is_transitions():
    p = paths()
    labeled = pd.read_parquet(p["features"] / "qqq_daily_features_labeled.parquet").reset_index(drop=True)
    _, h = _slice_folds(len(labeled), QQQConfig())
    holdout = labeled.iloc[h["start"] : h["end"]]
    bt = run_backtest(holdout, signal_dual_trend(holdout), 5.0)
    compact = json.loads(p["compact_status"].read_text())
    assert bt["trade_count"] == compact["trade_count"]


def test_cagr_sharpe_mdd_reconcile():
    p = paths()
    labeled = pd.read_parquet(p["features"] / "qqq_daily_features_labeled.parquet").reset_index(drop=True)
    _, h = _slice_folds(len(labeled), QQQConfig())
    holdout = labeled.iloc[h["start"] : h["end"]]
    compact = json.loads(p["compact_status"].read_text())
    bt = run_backtest(holdout, signal_dual_trend(holdout), 5.0)
    s = summarize_equity(bt)
    assert abs(s["CAGR"] - compact["strategy_cagr"]) < 1e-8
    assert abs(s["Sharpe"] - compact["strategy_sharpe"]) < 1e-8
    assert abs(s["max_drawdown"] - compact["strategy_mdd"]) < 1e-8


def test_production_promotion_false():
    compact = json.loads(paths()["compact_status"].read_text())
    assert compact["production_ready"] is False
    assert compact["promotion_ready"] is False


def test_btc_package_not_imported_by_equity():
    for p in (REPO / "src/canbit_equity").glob("*.py"):
        txt = p.read_text()
        assert "fstream.binance" not in txt
        assert "microstructure_public_live_collector" not in txt


def test_stale_features_intermediate_does_not_invalidate_labeled():
    p = paths()
    labeled = pd.read_parquet(p["features"] / "qqq_daily_features_labeled.parquet")
    feat = pd.read_parquet(p["features_file"])
    assert len(labeled) > 6000
    # Intermediate may be stale; research must use labeled.
    assert len(labeled) > len(feat)
    compact = json.loads(p["compact_status"].read_text())
    assert compact["selected_candidate"] == "DUAL_TREND_FILTER"


def test_additive_return_residual_is_product_term():
    p = paths()
    labeled = pd.read_parquet(p["features"] / "qqq_daily_features_labeled.parquet").reset_index(drop=True)
    _, h = _slice_folds(len(labeled), QQQConfig())
    holdout = labeled.iloc[h["start"] : h["end"]].copy().reset_index(drop=True)
    sig = signal_dual_trend(holdout)
    pos = signals_to_positions(sig)
    prev = pos.shift(1).fillna(0.0)
    open_ = holdout["open_adj"].astype(float)
    close = holdout["close_adj"].astype(float)
    overnight = open_ / close.shift(1) - 1.0
    intrad = close / open_ - 1.0
    day_ret = pos * intrad + np.minimum(pos, prev) * overnight.fillna(0.0)
    c2c = close / close.shift(1) - 1.0
    cont = (pos == 1) & (prev == 1)
    residual = day_ret[cont] - c2c[cont]
    product = (overnight.fillna(0.0) * intrad)[cont]
    assert float((residual + product).abs().max()) < 1e-12
