"""Pure helpers for the diagnostics-only microstructure WS watchdog."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

SELF_HEAL_ROOT = Path("data/diagnostics/microstructure_ws_self_healing")
WATCHDOG_LOG = SELF_HEAL_ROOT / "logs/ws_watchdog_events.jsonl"
PUBLIC_TIME_ENDPOINT = "https://fapi.binance.com/fapi/v1/time"
CORE_STREAMS = ("aggTrade", "markPrice", "kline_1m")
STATUS_TIMESTAMP_KEYS = {
    "aggTrade": "last_aggtrade_time_utc",
    "markPrice": "last_markprice_time_utc",
    "kline_1m": "last_kline_time_utc",
    "open_interest_poll": "last_oi_poll_time_utc",
}


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes, bool)):
            return None
    except Exception:
        pass
    return value.item() if hasattr(value, "item") else value


@dataclass(frozen=True)
class WatchdogConfig:
    check_interval_seconds: float = 15.0
    soft_stale_seconds: float = 60.0
    hard_stale_seconds: float = 120.0
    reconnect_timeout_seconds: float = 30.0
    max_consecutive_reconnect_failures: int = 5
    process_exit_after_failure_seconds: float = 900.0
    backoff_initial_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    backoff_jitter_ratio: float = 0.20
    startup_grace_seconds: float = 45.0
    minimum_reconnect_interval_seconds: float = 10.0
    sleep_wake_elapsed_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "WatchdogConfig":
        def number(name: str, default: float) -> float:
            return float(os.environ.get(name, default))

        return cls(
            check_interval_seconds=number("WS_WATCHDOG_CHECK_INTERVAL_SECONDS", 15),
            soft_stale_seconds=number("WS_SOFT_STALE_SECONDS", 60),
            hard_stale_seconds=number("WS_HARD_STALE_SECONDS", 120),
            reconnect_timeout_seconds=number("WS_RECONNECT_TIMEOUT_SECONDS", 30),
            max_consecutive_reconnect_failures=int(
                number("WS_MAX_CONSECUTIVE_RECONNECT_FAILURES", 5)
            ),
            process_exit_after_failure_seconds=number(
                "WS_PROCESS_EXIT_AFTER_FAILURE_SECONDS", 900
            ),
            backoff_initial_seconds=number("WS_BACKOFF_INITIAL_SECONDS", 2),
            backoff_max_seconds=number("WS_BACKOFF_MAX_SECONDS", 60),
            backoff_jitter_ratio=number("WS_BACKOFF_JITTER_RATIO", 0.20),
            startup_grace_seconds=number("WS_WATCHDOG_STARTUP_GRACE_SECONDS", 45),
            minimum_reconnect_interval_seconds=number(
                "WS_MINIMUM_RECONNECT_INTERVAL_SECONDS", 10
            ),
            sleep_wake_elapsed_seconds=number(
                "WS_SLEEP_WAKE_ELAPSED_SECONDS", 60
            ),
        )


def timestamp_age_seconds(value: Any, now: pd.Timestamp | None = None) -> float:
    if value in (None, ""):
        return math.inf
    parsed = pd.to_datetime(value, utc=True, format="mixed", errors="coerce")
    if pd.isna(parsed):
        return math.inf
    return max(0.0, ((now or now_utc()) - parsed).total_seconds())


def compute_backoff_seconds(
    failure_count: int,
    config: WatchdogConfig,
    random_unit: float | None = None,
) -> float:
    base = min(
        config.backoff_max_seconds,
        config.backoff_initial_seconds * (2 ** max(0, failure_count - 1)),
    )
    unit = random.random() if random_unit is None else random_unit
    jitter = (unit * 2.0 - 1.0) * config.backoff_jitter_ratio
    return max(0.0, base * (1.0 + jitter))


def evaluate_ws_health(
    status: Mapping[str, Any],
    task_alive: Mapping[str, bool],
    connection_open: Mapping[str, bool],
    config: WatchdogConfig,
    now: pd.Timestamp | None = None,
    startup_age_seconds: float | None = None,
) -> Dict[str, Any]:
    current = now or now_utc()
    ages = {
        stream: timestamp_age_seconds(status.get(key), current)
        for stream, key in STATUS_TIMESTAMP_KEYS.items()
    }
    core_alive = {
        stream: bool(task_alive.get(stream)) and bool(connection_open.get(stream))
        for stream in CORE_STREAMS
    }
    receiving = {
        stream: core_alive[stream] and ages[stream] <= config.hard_stale_seconds
        for stream in CORE_STREAMS
    }
    hard_stale = [
        stream for stream in CORE_STREAMS
        if ages[stream] > config.hard_stale_seconds
    ]
    task_dead = [stream for stream in CORE_STREAMS if not core_alive[stream]]
    healthy_count = sum(receiving.values())
    in_startup_grace = (
        startup_age_seconds is not None
        and startup_age_seconds < config.startup_grace_seconds
    )
    trigger_reason = None
    trigger_streams: list[str] = []
    if not in_startup_grace and task_dead:
        trigger_reason = "WS_TASK_DIED_RECOVERY"
        trigger_streams = task_dead
    elif not in_startup_grace and len(hard_stale) >= 2:
        trigger_reason = "WATCHDOG_STALE_RECOVERY"
        trigger_streams = hard_stale
    elif not in_startup_grace:
        extreme = [
            stream for stream in hard_stale
            if ages[stream] > config.hard_stale_seconds * 2
        ]
        if extreme:
            trigger_reason = "PARTIAL_STREAM_RECOVERY"
            trigger_streams = extreme
    market_age = max((ages[stream] for stream in CORE_STREAMS), default=math.inf)
    return {
        "stream_ages": ages,
        "task_alive_states": dict(task_alive),
        "connection_open_states": dict(connection_open),
        "receiving": receiving,
        "aggtrade_age_seconds": ages["aggTrade"],
        "markprice_age_seconds": ages["markPrice"],
        "kline_age_seconds": ages["kline_1m"],
        "market_ws_age_seconds": market_age,
        "oi_age_seconds": ages["open_interest_poll"],
        "health_quorum_pass": healthy_count >= 2,
        "hard_stale_streams": hard_stale,
        "task_dead_streams": task_dead,
        "trigger_reason": trigger_reason,
        "trigger_streams": trigger_streams,
        "in_startup_grace": in_startup_grace,
    }


def public_network_probe(
    timeout_seconds: float = 5.0,
    forced_offline: bool = False,
) -> Dict[str, Any]:
    if forced_offline:
        return {
            "reachable": False,
            "endpoint": PUBLIC_TIME_ENDPOINT,
            "error": "FAULT_INJECTED_NETWORK_UNREACHABLE",
        }
    try:
        with urllib.request.urlopen(PUBLIC_TIME_ENDPOINT, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return {
            "reachable": response.status == 200 and "serverTime" in data,
            "endpoint": PUBLIC_TIME_ENDPOINT,
            "status": response.status,
            "error": None,
        }
    except Exception as exc:
        return {
            "reachable": False,
            "endpoint": PUBLIC_TIME_ENDPOINT,
            "error": f"{type(exc).__name__}: {exc}",
        }


def append_watchdog_event(row: Dict[str, Any], path: Path = WATCHDOG_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"event_utc": now_utc(), **row}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(clean(payload), ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(clean(payload), fh, ensure_ascii=False, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
