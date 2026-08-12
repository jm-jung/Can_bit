"""Immutable research config for QQQ equity-ETF track."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data/equity_etf/qqq"
DIAG_ROOT = REPO / "data/diagnostics/equity_etf_qqq"


@dataclass(frozen=True)
class QQQConfig:
    symbol: str = "QQQ"
    asset_type: str = "ETF"
    exchange: str = "XNYS"
    exchange_timezone: str = "America/New_York"
    currency: str = "USD"
    provider: str = "yfinance"
    provider_provenance: str = "YAHOO_FINANCE_UNOFFICIAL"
    interval: str = "1d"
    start_date: str = "1999-03-10"  # QQQ inception approx; actual start from provider
    end_date: str | None = None  # resolved at runtime from completed sessions
    regular_session_only: bool = True
    price_adjustment_mode: str = "ADJ_CLOSE_FACTOR_TOTAL_RETURN_PROXY"
    cost_bps_per_side_low: float = 2.0
    cost_bps_per_side_base: float = 5.0
    cost_bps_per_side_high: float = 10.0
    target_horizons: Tuple[int, ...] = (5, 10, 20)
    primary_horizon: int = 20
    final_holdout_sessions: int = 504
    minimum_training_sessions: int = 1260
    validation_sessions: int = 252
    test_sessions: int = 252
    step_sessions: int = 252
    lr_thresholds: Tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)
    min_long_exposure: float = 0.25
    min_trades_per_validation_year: int = 5
    random_seed: int = 42
    schema_version: str = "qqq_equity_v1"
    pipeline_version: str = "0.1.0"
    directions: Tuple[str, ...] = ("LONG", "FLAT")
    short_enabled: bool = False
    leverage: float = 1.0
    production_ready: bool = False
    promotion_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["target_horizons"] = list(self.target_horizons)
        d["lr_thresholds"] = list(self.lr_thresholds)
        d["directions"] = list(self.directions)
        return d


def paths(cfg: QQQConfig = QQQConfig()) -> Dict[str, Path]:
    root = DATA_ROOT
    return {
        "repo": REPO,
        "data_root": root,
        "raw": root / "raw",
        "normalized": root / "normalized",
        "features": root / "features",
        "models": root / "models",
        "backtests": root / "backtests",
        "reports": root / "reports",
        "charts": root / "reports" / "charts",
        "manifests": root / "manifests",
        "cache": root / "cache",
        "diag": DIAG_ROOT,
        "diag_reports": DIAG_ROOT / "reports",
        "normalized_file": root / "normalized" / "qqq_1d_normalized.parquet",
        "features_file": root / "features" / "qqq_daily_features.parquet",
        "manifest_latest": root / "manifests" / "qqq_1d_latest_manifest.json",
        "feature_manifest": root / "manifests" / "qqq_daily_feature_manifest.json",
        "holdout_lock": root / "models" / "final_holdout_evaluation_lock.json",
        "compact_status": root / "reports" / "qqq_compact_status.json",
    }


def ensure_dirs(cfg: QQQConfig = QQQConfig()) -> None:
    for key, p in paths(cfg).items():
        if key.endswith("_file") or key.endswith("_lock") or key in {"repo", "manifest_latest", "feature_manifest", "compact_status"}:
            continue
        Path(p).mkdir(parents=True, exist_ok=True)
    for sub in ("preflight", "data_quality", "walkforward", "reports"):
        (DIAG_ROOT / sub).mkdir(parents=True, exist_ok=True)
