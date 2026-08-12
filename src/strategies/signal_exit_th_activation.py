from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

RT_A_MIN_UNIQUE_ROUND_TRIPS = 225
# Shadow threshold for telemetry (582). When USE_ADAPTIVE_RT_A_FOR_LOSS_AUX is True,
# loss_aux gating follows shadow_rt_a_on; rt_a_on (225) stays the operational signal.
SHADOW_RT_A_MIN_UNIQUE_ROUND_TRIPS = 582
PRIMARY_THRESHOLD = 0.6

# Partial deployment: loss_aux params follow shadow_rt_a_on instead of rt_a_on.
USE_ADAPTIVE_RT_A_FOR_LOSS_AUX = False

LOSS_AUX_DELTA_P = 0.05
LOSS_AUX_MAE_CUT = -0.007
LOSS_AUX_SKIP_IF_DELTA_UNREAL_POSITIVE = True
STATE_LOG_REQUIRED_COLUMNS = [
    "timestamp",
    "rt_a_on",
    "vol_shadow_on",
    "unique_round_trips",
    "vol_bucket",
    "loss_aux_applied",
    "effective_loss_aux_gate",
    "loss_aux_gate_source",
    "shadow_rt_a_on",
    "shadow_loss_aux_applied",
]


def resolve_loss_aux_gate(rt_a_on: bool, shadow_rt_a_on: bool) -> tuple[bool, str]:
    """Effective gate for loss_aux params and state_log loss_aux_applied."""
    if USE_ADAPTIVE_RT_A_FOR_LOSS_AUX:
        return bool(shadow_rt_a_on), "shadow_rt_a_on"
    return bool(rt_a_on), "rt_a_on"


def should_activate_rt_a(window_stats: dict) -> bool:
    """
    Activation condition:
    - unique_round_trips >= 225

    Return True -> enable loss_aux
    Return False -> disable loss_aux
    """
    try:
        if not isinstance(window_stats, dict):
            logger.warning("[RT-A] window_stats is not dict; fallback OFF")
            return False
        urt_raw = window_stats.get("unique_round_trips")
        if urt_raw is None:
            logger.warning("[RT-A] unique_round_trips missing; fallback OFF")
            return False
        return int(urt_raw) >= RT_A_MIN_UNIQUE_ROUND_TRIPS
    except Exception as exc:
        logger.warning("[RT-A] activation check failed; fallback OFF (%s)", exc)
        return False


def compute_shadow_rt_a_on(unique_round_trips: Any) -> bool:
    """
    Shadow telemetry only: ON iff unique_round_trips >= SHADOW_RT_A_MIN_UNIQUE_ROUND_TRIPS.
    Must not be used for loss_aux or trading decisions.
    """
    try:
        if unique_round_trips is None:
            return False
        return int(unique_round_trips) >= SHADOW_RT_A_MIN_UNIQUE_ROUND_TRIPS
    except Exception:
        return False


def compute_vol_bucket(df: pd.DataFrame) -> str:
    """Tercile bucket based on rolling-return volatility."""
    try:
        if df.empty or "close" not in df.columns:
            return "unknown"
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change().dropna()
        if ret.empty:
            return "unknown"
        vol = float(ret.std())
        abs_ret = ret.abs()
        q1 = float(abs_ret.quantile(1.0 / 3.0))
        q2 = float(abs_ret.quantile(2.0 / 3.0))
        if vol <= q1:
            return "low"
        if vol <= q2:
            return "mid"
        return "high"
    except Exception as exc:
        logger.warning("[RT-A] vol bucket compute failed (%s)", exc)
        return "unknown"


def compute_runtime_trades_60d(df: pd.DataFrame, lookback_days: int = 60) -> int | None:
    """
    Runtime rolling trade-count proxy from already-generated signals.
    - Uses latest `lookback_days` window by timestamp
    - Counts completed round-trips (entry->exit), including flip closes
    """
    try:
        if df.empty or "timestamp" not in df.columns or "signal" not in df.columns:
            return None
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if ts.isna().all():
            return None
        t_max = ts.max()
        if pd.isna(t_max):
            return None
        cutoff = t_max - pd.Timedelta(days=lookback_days)
        work = df.loc[ts >= cutoff, ["signal"]].copy()
        if work.empty:
            return None

        position: str | None = None
        round_trips = 0

        for sig_raw in work["signal"].tolist():
            sig = str(sig_raw).upper() if sig_raw is not None else "HOLD"
            if position is None:
                if sig in {"LONG", "SHORT"}:
                    position = sig
                continue

            if position == "LONG":
                if sig in {"HOLD", "SHORT"}:
                    round_trips += 1
                    position = "SHORT" if sig == "SHORT" else None
            else:  # position == "SHORT"
                if sig in {"HOLD", "LONG"}:
                    round_trips += 1
                    position = "LONG" if sig == "LONG" else None

        return int(round_trips)
    except Exception as exc:
        logger.warning("[RT-A] runtime trades_60d compute failed (%s)", exc)
        return None


def compute_runtime_trades_60d_asof(
    df: pd.DataFrame, end_row_index: int, lookback_days: int = 60
) -> int | None:
    """
    Same as compute_runtime_trades_60d but anchored at df.iloc[end_row_index].timestamp
    (signals only up to and including that row). Used for per-bar ENTRY gating.
    """
    try:
        if df.empty or end_row_index < 0 or end_row_index >= len(df):
            return None
        sub = df.iloc[: end_row_index + 1]
        return compute_runtime_trades_60d(sub, lookback_days=lookback_days)
    except Exception as exc:
        logger.warning("[RT-A] runtime trades_60d asof compute failed (%s)", exc)
        return None


def load_latest_window_stats(state_log_path: Path) -> dict[str, Any]:
    """
    Reuse rolling metrics from state_log schema if available.
    Falls back to empty dict on any issue.
    """
    try:
        if not state_log_path.exists():
            return {}
        with state_log_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return {}
        out: dict[str, Any] = {}
        for row in reversed(rows):
            # Primary source: runtime-like rolling metric from meta path.
            v = row.get("trades_60d")
            if v not in (None, ""):
                trades = int(float(v))
                out["trades_60d"] = trades
                # Keep compatibility with should_activate_rt_a(window_stats) contract.
                out["unique_round_trips"] = trades
                break
            # Fallback only: legacy/activation probe value.
            v = row.get("unique_round_trips")
            if v not in (None, ""):
                out["unique_round_trips"] = int(float(v))
                break
        return out
    except Exception as exc:
        logger.warning("[RT-A] failed to load latest state_log window stats (%s)", exc)
        return {}


def load_latest_entry_urt_snapshot(
    state_log_path: Path,
    *,
    strategy_name_filter: str = "B_with_meta",
) -> dict[str, Any]:
    """
    Entry-gate snapshot: latest row (reverse scan) with strategy_name == filter.
    Uses trades_60d first, then unique_round_trips. Excludes runtime_activation_probe rows.
    """
    try:
        if not state_log_path.exists():
            return {}
        with state_log_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}

        for row in reversed(rows):
            if str(row.get("strategy_name") or "") != strategy_name_filter:
                continue
            ts = row.get("timestamp")
            v_trades = row.get("trades_60d")
            if v_trades not in (None, ""):
                urt = int(float(v_trades))
                return {
                    "unique_round_trips": urt,
                    "trades_60d": urt,
                    "selected_timestamp": ts,
                    "selected_strategy_name": row.get("strategy_name"),
                    "selected_trades_60d_raw": v_trades,
                    "selected_unique_round_trips_raw": row.get("unique_round_trips"),
                    "selected_source": "trades_60d",
                }
            v_urt = row.get("unique_round_trips")
            if v_urt not in (None, ""):
                urt = int(float(v_urt))
                return {
                    "unique_round_trips": urt,
                    "selected_timestamp": ts,
                    "selected_strategy_name": row.get("strategy_name"),
                    "selected_trades_60d_raw": v_trades,
                    "selected_unique_round_trips_raw": v_urt,
                    "selected_source": "unique_round_trips",
                }
        return {}
    except Exception as exc:
        logger.warning("[RT-A] failed to load latest entry URT snapshot (%s)", exc)
        return {}


def append_state_log_activation(
    *,
    state_log_path: Path,
    rt_a_on: bool,
    vol_shadow_on: bool,
    unique_round_trips: Any,
    vol_bucket: str,
) -> None:
    """
    Append activation telemetry row safely.
    - loss_aux_applied / effective_loss_aux_gate follow resolve_loss_aux_gate(rt_a_on, shadow_rt_a_on)
    - Always append mode
    - Never break pipeline on failure
    """
    try:
        state_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = state_log_path.exists() and state_log_path.stat().st_size > 0

        if file_exists:
            try:
                current_df = pd.read_csv(state_log_path)
            except Exception:
                current_df = pd.DataFrame(columns=STATE_LOG_REQUIRED_COLUMNS)
            changed = False
            for col in STATE_LOG_REQUIRED_COLUMNS:
                if col not in current_df.columns:
                    current_df[col] = None
                    changed = True
            if changed:
                current_df.to_csv(state_log_path, index=False)
        else:
            pd.DataFrame(columns=STATE_LOG_REQUIRED_COLUMNS).to_csv(state_log_path, index=False)
            file_exists = True

        shadow_rt_a_on = compute_shadow_rt_a_on(unique_round_trips)
        shadow_loss_aux_applied = bool(shadow_rt_a_on)
        effective_gate, gate_source = resolve_loss_aux_gate(rt_a_on, shadow_rt_a_on)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rt_a_on": bool(rt_a_on),
            "vol_shadow_on": bool(vol_shadow_on),
            "unique_round_trips": None if unique_round_trips is None else int(unique_round_trips),
            "vol_bucket": vol_bucket if vol_bucket else None,
            "loss_aux_applied": effective_gate,
            "effective_loss_aux_gate": effective_gate,
            "loss_aux_gate_source": gate_source,
            "shadow_rt_a_on": shadow_rt_a_on,
            "shadow_loss_aux_applied": shadow_loss_aux_applied,
        }
        # Align row to existing header to avoid column shift in mixed schema files.
        target_columns = list(pd.read_csv(state_log_path, nrows=0).columns) if file_exists else list(STATE_LOG_REQUIRED_COLUMNS)
        for col in STATE_LOG_REQUIRED_COLUMNS:
            if col not in target_columns:
                target_columns.append(col)
        full_row = {k: None for k in target_columns}
        full_row.update(row)
        if "strategy_name" in target_columns:
            full_row["strategy_name"] = "signal_exit_th_activation"
        if "transition_reason" in target_columns:
            full_row["transition_reason"] = "runtime_activation_probe"
        row_df = pd.DataFrame([full_row], columns=target_columns)
        row_df.to_csv(
            state_log_path,
            mode="a",
            header=False,
            index=False,
            columns=target_columns,
        )
    except Exception as exc:
        logger.warning("[RT-A] state_log append failed (non-fatal): %s", exc)
