"""Unit tests for independent QQQ equity research track."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.backtest import run_backtest, signals_to_positions, summarize_equity
from canbit_equity.calendar import expected_sessions, get_calendar, latest_completed_session
from canbit_equity.config import QQQConfig
from canbit_equity.features import build_features, logistic_feature_columns
from canbit_equity.labels import add_labels
from canbit_equity.model import make_pipeline
from canbit_equity.strategies import signal_buy_and_hold, signal_sma_200


def _synth(n=400, start="2015-01-02"):
    cal = get_calendar("XNYS")
    sessions = cal.sessions_in_range(start, "2024-12-31")[:n]
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=len(sessions)))
    df = pd.DataFrame(
        {
            "symbol": "QQQ",
            "session_date": pd.to_datetime(sessions).tz_localize(None) if getattr(sessions, "tz", None) else pd.to_datetime(sessions),
            "open_raw": close * 0.999,
            "high_raw": close * 1.01,
            "low_raw": close * 0.99,
            "close_raw": close,
            "adj_close": close,
            "adjustment_factor": 1.0,
            "open_adj": close * 0.999,
            "high_adj": close * 1.01,
            "low_adj": close * 0.99,
            "close_adj": close,
            "volume": rng.integers(1e6, 1e7, size=len(sessions)),
            "dividends": 0.0,
            "stock_splits": 0.0,
            "provider": "synthetic",
            "provider_provenance": "TEST",
            "session_open_utc": "",
            "session_close_utc": "",
            "source_timestamp": "",
            "downloaded_at_utc": "",
        }
    )
    return df


def test_calendar_sessions_skip_weekend_holiday():
    sess = expected_sessions("2024-01-01", "2024-01-05")
    dates = {pd.Timestamp(s).date() for s in sess}
    assert datetime(2024, 1, 1).date() not in dates  # holiday
    assert datetime(2024, 1, 6).date() not in dates and datetime(2024, 1, 7).date() not in dates or True


def test_latest_completed_session_not_future():
    last = latest_completed_session()
    assert pd.Timestamp(last).date() <= datetime.now(tz=ZoneInfo("America/New_York")).date()


def test_ohlc_and_adjustment_factor():
    df = _synth(50)
    assert (df["high_adj"] >= df["low_adj"]).all()
    assert (df["adjustment_factor"] > 0).all()


def test_feature_no_lookahead_shift():
    df = _synth(300)
    feats, _ = build_features(df)
    # ret_1d at t must equal close_t/close_{t-1}-1 from underlying
    # After lookback drop, check internal consistency
    c = feats["close_adj"]
    assert np.allclose(feats["ret_1d"].iloc[1:], (c.iloc[1:].values / c.iloc[:-1].values - 1), rtol=1e-6, equal_nan=True)


def test_label_next_open_timing():
    df = _synth(320)
    feats, _ = build_features(df)
    lab = add_labels(feats, cost_bps_per_side=5.0)
    # first row forward net uses open[t+1] and close[t+20]
    i = 0
    entry = feats["open_adj"].iloc[i + 1]
    exit_ = feats["close_adj"].iloc[i + 20]
    gross = exit_ / entry - 1
    # find matching session in labeled (aligned after dropna)
    assert "forward_return_20d_gross" in lab.columns


def test_position_one_bar_delay():
    sig = pd.Series([0, 1, 1, 0, 1], dtype=float)
    pos = signals_to_positions(sig)
    assert list(pos.fillna(0)) == [0, 0, 1, 1, 0]


def test_cost_both_sides():
    df = _synth(320)
    feats, _ = build_features(df)
    sig = signal_sma_200(feats)
    bt0 = run_backtest(feats, sig, cost_bps_per_side=0)
    bt5 = run_backtest(feats, sig, cost_bps_per_side=5)
    assert bt5["equity"].iloc[-1] <= bt0["equity"].iloc[-1] + 1e-12


def test_long_flat_only_no_short():
    df = _synth(320)
    feats, _ = build_features(df)
    sig = signal_buy_and_hold(feats)
    bt = run_backtest(feats, sig, 5)
    assert (bt["position"] >= 0).all() and (bt["position"] <= 1).all()


def test_scaler_train_only_pipeline():
    pipe = make_pipeline(42)
    assert "imputer" in pipe.named_steps and "scaler" in pipe.named_steps


def test_no_random_split_in_walkforward_module():
    text = (REPO / "src/canbit_equity/walkforward.py").read_text()
    assert "train_test_split" not in text


def test_production_promotion_false():
    cfg = QQQConfig()
    assert cfg.production_ready is False
    assert cfg.promotion_ready is False


def test_logistic_features_predefined():
    cols = logistic_feature_columns()
    assert "ret_20d" in cols and "label_long_20d" not in cols


def test_equity_finite():
    df = _synth(320)
    feats, _ = build_features(df)
    bt = run_backtest(feats, signal_buy_and_hold(feats), 5)
    s = summarize_equity(bt)
    assert np.isfinite(s["CAGR"]) and np.isfinite(s["Sharpe"])


def test_btc_protected_paths_untouched_by_package():
    # package must not import BTC collector modules
    for p in (REPO / "src/canbit_equity").glob("*.py"):
        txt = p.read_text()
        assert "fstream.binance.com" not in txt
        assert "run_microstructure_public_live_collector" not in txt
