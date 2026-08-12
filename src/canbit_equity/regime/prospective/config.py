"""Prospective observation config — frozen lock, no daily retrain."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ORIG_LOCK = REPO / "data/equity_etf/qqq/regime/locks/qqq_regime_prospective_holdout_lock.json"
PROSP_ROOT = REPO / "data/equity_etf/qqq/regime/prospective"
DIAG_ROOT = REPO / "data/diagnostics/equity_etf_qqq_regime_prospective"
EXPECTED_LABELED = "5070477672a866749a2e2b7674a2029c1fb5b461e349bd3dc52d9e4e337d0bf5"
WEBHOOK_DEFAULT = Path.home() / ".config/can_bit/discord_observation_quality_webhook"
LAUNCHD_LABEL = "com.canbit.qqq-regime-prospective-daily"
COST_BASE = 5.0
COST_LOW = 2.0
COST_HIGH = 10.0
HORIZONS = (5, 10, 20)
PRIMARY_HORIZON = 20


def ensure_dirs() -> None:
    for sub in (
        "models",
        "state",
        "predictions",
        "executions",
        "outcomes",
        "shadow",
        "metrics",
        "daily_snapshots",
        "manifests",
        "outbox",
        "locks",
        "reports",
        "cache",
    ):
        (PROSP_ROOT / sub).mkdir(parents=True, exist_ok=True)
    for sub in (
        "preflight",
        "lock_audit",
        "model_audit",
        "daily",
        "data_quality",
        "predictions",
        "outcomes",
        "shadow",
        "discord",
        "launchd",
        "reports",
        "backups",
    ):
        (DIAG_ROOT / sub).mkdir(parents=True, exist_ok=True)
