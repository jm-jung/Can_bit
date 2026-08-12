"""QQQ regime research config — fixed series/features/candidates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[3]
REGIME_ROOT = REPO / "data/equity_etf/qqq/regime"
DIAG_ROOT = REPO / "data/diagnostics/equity_etf_qqq_regime"
EXPECTED_LABELED_HASH = "5070477672a866749a2e2b7674a2029c1fb5b461e349bd3dc52d9e4e337d0bf5"
V2_PATH = REPO / "data/equity_etf/qqq/models/qqq_corrected_same_period_selection_v2.json"

YF_SERIES = ("^VIX", "SPY", "RSP", "QEW", "SMH", "HYG", "IEF")
# yfinance symbol QEW is a 2026-inception fund; research intent is Nasdaq-100 equal-weight breadth.
# Resolve to QQEW (long history) while preserving feature names / universe label "QEW".
YF_SYMBOL_RESOLVE = {
    "QEW": "QQEW",
}
FRED_SERIES = ("DGS10", "DGS2", "T10Y2Y")

REGIME_FEATURES: Dict[str, List[str]] = {
    "VOLATILITY": ["vix_level", "vix_change_5d", "vix_sma20_ratio"],
    "RATES": ["dgs10_change_20d", "dgs2_change_20d", "t10y2y_level"],
    "BREADTH_LEADERSHIP": [
        "rsp_spy_relative_return_20d",
        "qew_qqq_relative_return_20d",
        "smh_qqq_relative_return_20d",
    ],
    "CREDIT": ["hyg_ief_relative_return_20d"],
}

ALL_REGIME_FEATURES: List[str] = (
    REGIME_FEATURES["VOLATILITY"]
    + REGIME_FEATURES["RATES"]
    + REGIME_FEATURES["BREADTH_LEADERSHIP"]
    + REGIME_FEATURES["CREDIT"]
)

CANDIDATE_FEATURE_GROUPS: Dict[str, List[str]] = {
    "LOGISTIC_PRICE_ONLY": [],
    "LOGISTIC_PRICE_PLUS_VOL": REGIME_FEATURES["VOLATILITY"],
    "LOGISTIC_PRICE_PLUS_RATES": REGIME_FEATURES["RATES"],
    "LOGISTIC_PRICE_PLUS_BREADTH": REGIME_FEATURES["BREADTH_LEADERSHIP"],
    "LOGISTIC_PRICE_PLUS_CREDIT": REGIME_FEATURES["CREDIT"],
    "LOGISTIC_PRICE_PLUS_ALL_REGIME": ALL_REGIME_FEATURES,
}

RULE_CANDIDATES = ("BUY_AND_HOLD", "DUAL_TREND_FILTER")
LR_THRESHOLDS: Tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)
COST_BASE = 5.0
COST_LOW = 2.0
COST_HIGH = 10.0
MIN_TRAIN = 1260
VAL_SESSIONS = 252
TEST_SESSIONS = 252
STEP_SESSIONS = 252
SELECTION_DEV_END_MAX = "2024-06-26"
FRED_LAG_SESSIONS = 1
MAX_FFILL_AGE = 5
MIN_PROSPECTIVE = 126
TARGET_PROSPECTIVE = 252
PIPELINE_VERSION = "qqq_regime_v1"


@dataclass(frozen=True)
class RegimePaths:
    root: Path = REGIME_ROOT
    diag: Path = DIAG_ROOT

    def ensure(self) -> "RegimePaths":
        for sub in (
            "raw",
            "normalized",
            "features",
            "models",
            "reports",
            "reports/charts",
            "manifests",
            "locks",
            "cache",
        ):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        for sub in (
            "preflight",
            "data_quality",
            "alignment",
            "common_period",
            "walkforward",
            "ablation",
            "selection",
            "lookahead",
            "reports",
            "charts",
        ):
            (self.diag / sub).mkdir(parents=True, exist_ok=True)
        return self


def paths() -> RegimePaths:
    return RegimePaths().ensure()
