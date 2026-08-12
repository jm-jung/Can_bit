from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FR2MetaConfig:
    # Package identity
    strategy_name: str = "B_with_meta"

    # Position multipliers
    full_multiplier: float = 1.0
    reduced_multiplier: float = 0.4
    off_multiplier: float = 0.0

    # Startup safety
    initial_state: str = "REDUCED"  # MUST start reduced-first

    # Score weights
    w_60d: float = 0.7
    w_90d: float = 0.3

    # State thresholds
    full_score_min: float = 0.0
    full_alpha_min: float = 1.0
    reduced_score_min: float = -0.2

    # Base OFF rule
    off_score_cutoff: float = -0.2  # config-driven; can test -0.15

    # Trade protection
    trades_60d_min: int = 50

    # Hysteresis
    hysteresis_needed: int = 2

    # Optional alpha OFF rule
    alpha_off_enabled: bool = True
    alpha_off_cutoff: float = 0.5
    alpha_off_consecutive_needed: int = 2

    # Optional prolonged REDUCED rule
    reduced_streak_rule_enabled: bool = False
    reduced_streak_off_threshold: int = 4
    reduced_streak_downshift_multiplier: float = 0.2

    # Cadence
    # In production, this should align with bar-close or a scheduler.
    # For safety, default is not every minute.
    eval_min_interval_seconds: int = 60 * 60  # 1h

    # Persistence paths
    snapshot_path: str = "data/diagnostics/fr2/meta_state_snapshot.json"
    state_log_path: str = "data/diagnostics/fr2/state_log.csv"

    # Rolling metrics supply snapshot (operational health check용)
    metrics_snapshot_path: str = "data/diagnostics/fr2/meta_metrics_snapshot.json"

    # Metrics staleness thresholds (minutes)
    # - warning: stale but can still operate
    # - critical: force OFF for safety
    metrics_stale_warning_minutes: float = 120.0
    metrics_stale_critical_minutes: float = 1440.0

