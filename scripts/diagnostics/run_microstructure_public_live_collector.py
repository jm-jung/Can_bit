"""Diagnostics-only public Binance microstructure collector.

Collects public websocket market data and public REST OI snapshots only.
No keys, no private endpoints, no trading actions, no production integration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import websockets

try:
    import microstructure_gap_backfill as gap_backfill
    import microstructure_ws_self_healing as ws_healing
except ModuleNotFoundError:  # package import in tests
    from scripts.diagnostics import microstructure_gap_backfill as gap_backfill
    from scripts.diagnostics import microstructure_ws_self_healing as ws_healing

ROOT = Path("data/diagnostics/new_market_microstructure_data_pipeline")
LIVE_ROOT = ROOT / "live"
STATUS_DIR = LIVE_ROOT / "status"
RAW_DIR = LIVE_ROOT / "raw"
NORMALIZED_DIR = LIVE_ROOT / "normalized"
LOG_DIR = LIVE_ROOT / "logs"
DIAG_ROOT = ROOT / "ws_receive_diagnostics"
SYMBOL = "BTCUSDT"
FAULT_REQUEST_PATH = STATUS_DIR / "ws_fault_injection_request.json"
FAPI = "https://fapi.binance.com"
MAINNET_MARKET_WS_PREFIX = "wss://fstream.binance.com/market/ws/btcusdt@"
STREAMS = {
    "forceOrder": f"{MAINNET_MARKET_WS_PREFIX}forceOrder",
    "aggTrade": f"{MAINNET_MARKET_WS_PREFIX}aggTrade",
    "markPrice": f"{MAINNET_MARKET_WS_PREFIX}markPrice",
    "kline_1m": f"{MAINNET_MARKET_WS_PREFIX}kline_1m",
}
COMBINED_URL = "wss://fstream.binance.com/market/stream?streams=btcusdt@forceOrder/btcusdt@aggTrade/btcusdt@markPrice/btcusdt@kline_1m"
FORBIDDEN = [
    "account",
    "balance",
    "position",
    "openOrders",
    "allOrders",
    "myTrades",
    "listenKey",
    "userDataStream",
    "leverage",
    "marginType",
    "positionRisk",
    "apiTradingStatus",
    "income",
    "transfer",
    "withdraw",
    "deposit",
    "newOrder",
    "cancelOrder",
    "createOrder",
    "fetchBalance",
    "fetchPositions",
    "privateGet",
    "privatePost",
    "privateDelete",
]


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def jdump(obj: Any) -> str:
    return json.dumps(clean(obj), ensure_ascii=False, indent=2, default=str)


def ensure_dirs() -> None:
    for d in [
        STATUS_DIR,
        RAW_DIR,
        NORMALIZED_DIR,
        LOG_DIR,
        DIAG_ROOT / "raw_probe",
        DIAG_ROOT / "probe",
        DIAG_ROOT / "reports",
        DIAG_ROOT / "logs",
        DIAG_ROOT / "status",
        DIAG_ROOT / "audit",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now("UTC")


def status_path() -> Path:
    return STATUS_DIR / "microstructure_public_collector_status.json"


def pid_path() -> Path:
    return STATUS_DIR / "microstructure_public_collector.pid"


def append_jsonl(source: str, row: Dict[str, Any]) -> Path:
    day = now_utc().strftime("%Y-%m-%d")
    out_dir = RAW_DIR / source / f"symbol={SYMBOL}" / f"date={day}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean(row), ensure_ascii=False, default=str) + "\n")
    return path


def append_live_jsonl(root: Path, source: str, row: Dict[str, Any]) -> Path:
    day = now_utc().strftime("%Y-%m-%d")
    out_dir = root / source / f"symbol={SYMBOL}" / f"date={day}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean(row), ensure_ascii=False, default=str) + "\n")
    return path


def unwrap_payload(msg: str) -> tuple[str | None, Dict[str, Any]]:
    data = json.loads(msg)
    if isinstance(data, dict) and "stream" in data and "data" in data:
        return str(data.get("stream")), data.get("data") or {}
    return None, data


def normalize_event(source: str, payload: Dict[str, Any], recv_ts: pd.Timestamp) -> Dict[str, Any]:
    event_type = payload.get("e") or source
    event_ts_ms = payload.get("T") or payload.get("E")
    row: Dict[str, Any] = {
        "source": source,
        "event_type": event_type,
        "symbol": payload.get("s") or SYMBOL,
        "event_ts": pd.to_datetime(event_ts_ms, unit="ms", utc=True) if event_ts_ms else recv_ts,
        "event_time_ms": event_ts_ms,
        "local_received_ts": recv_ts,
        "production_ready": False,
        "promotion_ready": False,
    }
    row.update(
        gap_backfill.provenance(
            "LIVE_WS",
            STREAMS.get(source),
            "fstream.binance.com",
            row["event_ts"],
        )
    )
    row["forceorder_complete"] = True if source == "forceOrder" else None
    if source == "aggTrade":
        price = float(payload.get("p", 0) or 0)
        qty = float(payload.get("q", 0) or 0)
        is_buyer_maker = bool(payload.get("m"))
        row.update(
            {
                "agg_trade_id": payload.get("a"),
                "price": price,
                "qty": qty,
                "notional": price * qty,
                "is_buyer_maker": is_buyer_maker,
                "taker_side": "SELL" if is_buyer_maker else "BUY",
                "taker_buy_qty": 0.0 if is_buyer_maker else qty,
                "taker_sell_qty": qty if is_buyer_maker else 0.0,
                "taker_delta_qty": -qty if is_buyer_maker else qty,
            }
        )
    elif source == "markPrice":
        mark = float(payload.get("p", 0) or 0)
        index = float(payload.get("i", 0) or 0)
        row.update(
            {
                "mark_price": mark,
                "index_price": index,
                "funding_rate": float(payload.get("r", 0) or 0),
                "next_funding_time": pd.to_datetime(payload.get("T"), unit="ms", utc=True) if payload.get("T") else None,
                "mark_index_basis_bps": (mark / index - 1) * 10000 if index else None,
            }
        )
    elif source == "kline_1m":
        k = payload.get("k") or {}
        row.update(
            {
                "open_time": pd.to_datetime(k.get("t"), unit="ms", utc=True) if k.get("t") else None,
                "close_time": pd.to_datetime(k.get("T"), unit="ms", utc=True) if k.get("T") else None,
                "interval": k.get("i"),
                "open": float(k.get("o", 0) or 0),
                "high": float(k.get("h", 0) or 0),
                "low": float(k.get("l", 0) or 0),
                "close": float(k.get("c", 0) or 0),
                "volume": float(k.get("v", 0) or 0),
                "is_closed": bool(k.get("x")),
            }
        )
    elif source == "forceOrder":
        o = payload.get("o") or {}
        price = float(o.get("p", 0) or 0)
        avg_price = float(o.get("ap", 0) or 0)
        filled = float(o.get("z", 0) or 0)
        row.update(
            {
                "side": o.get("S"),
                "order_type": o.get("o"),
                "time_in_force": o.get("f"),
                "price": price,
                "avg_price": avg_price,
                "orig_qty": float(o.get("q", 0) or 0),
                "last_filled_qty": float(o.get("l", 0) or 0),
                "filled_qty": filled,
                "notional": (avg_price or price) * filled,
                "status": o.get("X"),
                "trade_time": pd.to_datetime(o.get("T"), unit="ms", utc=True) if o.get("T") else None,
            }
        )
    return row


def public_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if any(x.lower() in path.lower() for x in FORBIDDEN):
        raise RuntimeError(f"forbidden public guard blocked path: {path}")
    query = urllib.parse.urlencode(params or {})
    url = FAPI + path + (("?" + query) if query else "")
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def base_status() -> Dict[str, Any]:
    return {
        "collector_name": "microstructure_public_collector",
        "is_running": True,
        "pid": os.getpid(),
        "start_time_utc": now_utc(),
        "last_event_time_utc": None,
        "last_force_order_time_utc": None,
        "last_aggtrade_time_utc": None,
        "last_markprice_time_utc": None,
        "last_kline_time_utc": None,
        "last_oi_poll_time_utc": None,
        "events_received_total": 0,
        "force_order_events_total": 0,
        "aggtrade_events_total": 0,
        "markprice_events_total": 0,
        "kline_events_total": 0,
        "oi_snapshots_total": 0,
        "errors_total": 0,
        "reconnects_total": 0,
        "last_reconnect_ts": None,
        "automatic_gap_recovery_enabled": True,
        "automatic_gap_recovery_running": False,
        "last_gap_recovery_verdict": None,
        "last_gap_recovery_error": None,
        "collector_instance_id": None,
        "recovery_run_id": None,
        "snapshot_created": False,
        "snapshot_checksum": None,
        "gap_candidates_count": 0,
        "ledger_pre_registered_count": 0,
        "recovery_coordinator_status": "NOT_STARTED",
        "ws_self_healing_enabled": True,
        "watchdog_running": False,
        "watchdog_last_check_utc": None,
        "ws_connection_state": "INITIALIZING",
        "ws_connection_generation": 0,
        "ws_task_generation": 0,
        "last_state_change_utc": now_utc(),
        "last_successful_connect_utc": None,
        "last_successful_message_utc": None,
        "last_disconnect_detected_utc": None,
        "last_stale_detected_utc": None,
        "last_reconnect_attempt_utc": None,
        "last_reconnect_success_utc": None,
        "last_reconnect_error": None,
        "consecutive_reconnect_failures": 0,
        "total_watchdog_triggers": 0,
        "total_silent_stall_recoveries": 0,
        "total_network_recoveries": 0,
        "total_partial_task_recoveries": 0,
        "total_successful_self_heals": 0,
        "total_reconnect_attempts": 0,
        "total_reconnect_successes": 0,
        "current_backoff_seconds": 0.0,
        "reconnect_in_progress": False,
        "reconnect_reason": None,
        "reconnect_trigger_streams": [],
        "last_watchdog_trigger_utc": None,
        "last_watchdog_reason": None,
        "last_self_heal_duration_seconds": None,
        "last_network_offline_utc": None,
        "last_network_restored_utc": None,
        "fatal_restart_requested": False,
        "aggtrade_age_seconds": None,
        "markprice_age_seconds": None,
        "kline_age_seconds": None,
        "market_ws_age_seconds": None,
        "oi_age_seconds": None,
        "aggtrade_task_alive": False,
        "markprice_task_alive": False,
        "kline_task_alive": False,
        "forceorder_task_alive": False,
        "ws_connection_open": False,
        "ws_ping_ok": False,
        "health_quorum_pass": False,
        "market_ws_receiving": False,
        "watchdog_config": {},
        "public_only_guard_pass": True,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "write_paths_ok": True,
        "ws_receive_repair_status": "MAINNET_BINANCE_COM_MARKET_ROUTE",
        "aggtrade_receiving": False,
        "markprice_receiving": False,
        "kline_1m_receiving": False,
        "forceorder_receiving": False,
        "forceorder_waiting_for_events": True,
        "production_ready": False,
        "promotion_ready": False,
    }


def compute_staleness_guard(status: Dict[str, Any]) -> Dict[str, Any]:
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for name, key, thr_min in [
        ("market_ws", "last_event_time_utc", 15),
        ("aggTrade", "last_aggtrade_time_utc", 15),
        ("markPrice", "last_markprice_time_utc", 15),
        ("kline_1m", "last_kline_time_utc", 15),
        ("open_interest_poll", "last_oi_poll_time_utc", 10),
    ]:
        ts = status.get(key)
        if not ts:
            rows.append({"stream": name, "verdict": "MISSING_TIMESTAMP"})
            continue
        age_sec = (now - pd.to_datetime(ts, utc=True)).total_seconds()
        verdict = "OK" if age_sec <= thr_min * 60 else f"{name.upper()}_STALE_ALERT"
        rows.append({"stream": name, "age_seconds": age_sec, "threshold_seconds": thr_min * 60, "verdict": verdict})
    fo = "OK_SPARSE_STREAM" if status.get("forceorder_receiving") else "FORCEORDER_STREAM_WARNING"
    rows.append({"stream": "forceOrder", "verdict": fo})
    overall = "STALENESS_OK" if all(r.get("verdict") in {"OK", "OK_SPARSE_STREAM"} for r in rows) else "STALENESS_ALERT"
    return {"staleness_verdict": overall, "streams": rows}


def write_status(status: Dict[str, Any]) -> None:
    ensure_dirs()
    status["staleness_guard"] = compute_staleness_guard(status)
    gap_backfill.atomic_json(status_path(), status)


def read_status() -> Dict[str, Any]:
    if not status_path().exists():
        return {"is_running": False, "reason": "status_file_missing", "production_ready": False, "promotion_ready": False}
    try:
        return json.loads(status_path().read_text(encoding="utf-8"))
    except Exception as exc:
        return {"is_running": False, "error": str(exc), "production_ready": False, "promotion_ready": False}


def public_only_guard() -> Dict[str, Any]:
    text = Path(__file__).read_text(encoding="utf-8")
    rows = []
    for term in FORBIDDEN:
        rows.append({"term": term, "present": term in text, "allowed_context": "guard literal only"})
    ok = all(url.startswith(MAINNET_MARKET_WS_PREFIX) for url in STREAMS.values())
    return {"verdict": "PUBLIC_WS_ONLY_PASS" if ok else "PUBLIC_WS_ONLY_FAIL", "private_endpoint_calls": 0, "order_endpoint_calls": 0, "scan_rows": rows}


class RecoveryCoordinator:
    """Debounces API work, never immutable watermark capture."""

    def __init__(self, status: Dict[str, Any], collector_instance_id: str) -> None:
        self.status = status
        self.collector_instance_id = collector_instance_id
        self.task: asyncio.Task[Any] | None = None
        self.pending_snapshots: List[Dict[str, Any]] = []

    def schedule_snapshot(self, snapshot: Dict[str, Any]) -> None:
        run_id = str(snapshot["recovery_run_id"])
        if any(str(item["recovery_run_id"]) == run_id for item in self.pending_snapshots):
            return
        self.pending_snapshots.append(snapshot)
        self.status["snapshot_created"] = True
        self.status["recovery_run_id"] = run_id
        self.status["snapshot_checksum"] = snapshot["snapshot_checksum"]
        self.status["gap_candidates_count"] = len(snapshot.get("gap_candidates", []))
        self.status["ledger_pre_registered_count"] = len(snapshot.get("gap_candidates", []))
        self.status["recovery_coordinator_status"] = "LEDGER_PRE_REGISTERED"
        write_status(self.status)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        await asyncio.sleep(2)
        while self.pending_snapshots:
            snapshot = self.pending_snapshots.pop(0)
            self.status["automatic_gap_recovery_running"] = True
            self.status["recovery_run_id"] = snapshot["recovery_run_id"]
            self.status["recovery_coordinator_status"] = "BACKFILL_RUNNING"
            write_status(self.status)
            try:
                result = await asyncio.to_thread(
                    gap_backfill.automatic_recovery,
                    snapshot,
                    None,
                    self.collector_instance_id,
                )
                self.status["last_gap_recovery_verdict"] = result.get("verdict")
                self.status["last_gap_recovery_error"] = None
                self.status["recovery_coordinator_status"] = "COMPLETED"
            except Exception as exc:
                self.status["last_gap_recovery_verdict"] = "BACKFILL_FAILED_RETRYABLE"
                self.status["last_gap_recovery_error"] = str(exc)[:1000]
                self.status["recovery_coordinator_status"] = "FAILED_RETRYABLE"
                append_jsonl(
                    "collector_errors",
                    {
                        "source": "gap_recovery",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "local_received_ts": now_utc(),
                    },
                )
            finally:
                self.status["automatic_gap_recovery_running"] = False
                write_status(self.status)


async def persist_ws_message(
    name: str,
    msg: str,
    status: Dict[str, Any],
    generation: int,
    current_generation: int,
) -> bool:
    if generation != current_generation:
        ws_healing.append_watchdog_event(
            {
                "collector_pid": os.getpid(),
                "collector_instance_id": status.get("collector_instance_id"),
                "connection_generation": generation,
                "event_type": "STALE_CONNECTION_GENERATION",
                "trigger_reason": "OLD_GENERATION_EVENT_DISCARDED",
                "new_state": status.get("ws_connection_state"),
            }
        )
        return False
    recv_ts = now_utc()
    stream_name, data = unwrap_payload(msg)
    raw = {
        "source": name,
        "stream": stream_name,
        "local_received_ts": recv_ts,
        "connection_generation": generation,
        "payload": data,
    }
    append_live_jsonl(RAW_DIR, f"ws_{name}", raw)
    try:
        normalized = normalize_event(name, data, recv_ts)
        normalized["connection_generation"] = generation
        append_live_jsonl(NORMALIZED_DIR, f"ws_{name}", normalized)
    except Exception as exc:
        append_jsonl(
            "collector_errors",
            {
                "source": name,
                "stage": "normalize",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "local_received_ts": recv_ts,
            },
        )
    status["events_received_total"] += 1
    status["last_event_time_utc"] = recv_ts
    status["last_successful_message_utc"] = recv_ts
    if name == "forceOrder":
        status["force_order_events_total"] += 1
        status["last_force_order_time_utc"] = recv_ts
    elif name == "aggTrade":
        status["aggtrade_events_total"] += 1
        status["last_aggtrade_time_utc"] = recv_ts
    elif name == "markPrice":
        status["markprice_events_total"] += 1
        status["last_markprice_time_utc"] = recv_ts
    elif name == "kline_1m":
        status["kline_events_total"] += 1
        status["last_kline_time_utc"] = recv_ts
    write_status(status)
    return True


class WSSelfHealingManager:
    """Supervises independent public WS connections as one recovery domain."""

    def __init__(
        self,
        status: Dict[str, Any],
        recovery: RecoveryCoordinator,
        config: ws_healing.WatchdogConfig | None = None,
        network_probe=ws_healing.public_network_probe,
    ) -> None:
        self.status = status
        self.recovery = recovery
        self.config = config or ws_healing.WatchdogConfig.from_env()
        self.network_probe = network_probe
        self.tasks: Dict[str, asyncio.Task[Any]] = {}
        self.connections: Dict[str, Any] = {}
        self.connection_open: Dict[str, bool] = {name: False for name in STREAMS}
        self.connection_generation = 0
        self.generation_messages: set[str] = set()
        self.live_resumed_event = asyncio.Event()
        self.reconnect_lock = asyncio.Lock()
        self.reconnect_task: asyncio.Task[Any] | None = None
        self.watchdog_task: asyncio.Task[Any] | None = None
        self.fatal_event = asyncio.Event()
        self.stop_requested = False
        self.pending_reasons: set[str] = set()
        self.pending_streams: set[str] = set()
        self.pending_snapshot: Dict[str, Any] | None = None
        self.started_monotonic = time.monotonic()
        self.last_watchdog_monotonic = self.started_monotonic
        self.last_reconnect_monotonic = 0.0
        self.reconnect_failure_started_monotonic: float | None = None
        self.silent_stall_until_monotonic = 0.0
        self.forced_offline_until_monotonic = 0.0
        self.last_fault_nonce: str | None = None
        self._apply_config_status()

    def _apply_config_status(self) -> None:
        self.status["watchdog_config"] = {
            "check_interval_seconds": self.config.check_interval_seconds,
            "soft_stale_seconds": self.config.soft_stale_seconds,
            "hard_stale_seconds": self.config.hard_stale_seconds,
            "reconnect_timeout_seconds": self.config.reconnect_timeout_seconds,
            "max_consecutive_reconnect_failures": self.config.max_consecutive_reconnect_failures,
            "process_exit_after_failure_seconds": self.config.process_exit_after_failure_seconds,
            "backoff_initial_seconds": self.config.backoff_initial_seconds,
            "backoff_max_seconds": self.config.backoff_max_seconds,
            "backoff_jitter_ratio": self.config.backoff_jitter_ratio,
        }

    def _task_alive(self) -> Dict[str, bool]:
        return {
            name: bool(task and not task.done())
            for name, task in ((name, self.tasks.get(name)) for name in STREAMS)
        }

    def transition(
        self,
        new_state: str,
        event_type: str,
        reason: str | None = None,
        streams: List[str] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        previous = self.status.get("ws_connection_state")
        self.status["ws_connection_state"] = new_state
        self.status["last_state_change_utc"] = now_utc()
        ws_healing.append_watchdog_event(
            {
                "collector_pid": os.getpid(),
                "collector_instance_id": self.status.get("collector_instance_id"),
                "connection_generation": self.connection_generation,
                "event_type": event_type,
                "previous_state": previous,
                "new_state": new_state,
                "trigger_reason": reason,
                "trigger_streams": streams or [],
                "stream_ages": {
                    "aggTrade": self.status.get("aggtrade_age_seconds"),
                    "markPrice": self.status.get("markprice_age_seconds"),
                    "kline_1m": self.status.get("kline_age_seconds"),
                },
                "task_alive_states": self._task_alive(),
                **(extra or {}),
            }
        )
        write_status(self.status)

    async def start(self) -> None:
        self.status["watchdog_running"] = True
        self.transition("CONNECTING", "WATCHDOG_STARTED", "PROCESS_STARTUP")
        await self._start_generation()
        self.watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def _start_generation(self) -> None:
        self.connection_generation += 1
        self.status["ws_connection_generation"] = self.connection_generation
        self.status["ws_task_generation"] = self.connection_generation
        self.generation_messages = set()
        self.live_resumed_event = asyncio.Event()
        generation = self.connection_generation
        self.transition(
            "CONNECTING",
            "NEW_CONNECTION_CREATED",
            self.status.get("reconnect_reason") or "PROCESS_STARTUP",
        )
        self.tasks = {
            name: asyncio.create_task(self._stream_worker(name, url, generation))
            for name, url in STREAMS.items()
        }

    async def _stream_worker(self, name: str, url: str, generation: int) -> None:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                open_timeout=20,
                close_timeout=5,
                compression=None,
            ) as ws:
                if generation != self.connection_generation:
                    return
                self.connections[name] = ws
                self.connection_open[name] = True
                connected = now_utc()
                self.status["reconnects_total"] += 1
                self.status["last_successful_connect_utc"] = connected
                gap_backfill.append_transport_event(
                    {
                        "event": "CONNECTED",
                        "stream": name,
                        "reconnect_ts": connected,
                        "source_host": "fstream.binance.com",
                        "connection_generation": generation,
                    }
                )
                while not self.stop_requested and generation == self.connection_generation:
                    msg = await ws.recv()
                    if (
                        name in ws_healing.CORE_STREAMS
                        and time.monotonic() < self.silent_stall_until_monotonic
                    ):
                        continue
                    persisted = await persist_ws_message(
                        name, msg, self.status, generation, self.connection_generation
                    )
                    if not persisted:
                        continue
                    self.generation_messages.add(name)
                    if set(ws_healing.CORE_STREAMS).issubset(self.generation_messages):
                        if not self.live_resumed_event.is_set():
                            self.live_resumed_event.set()
                            self.status["last_reconnect_success_utc"] = now_utc()
                            self.transition(
                                "HEALTHY",
                                "FIRST_LIVE_EVENT_RECEIVED",
                                self.status.get("reconnect_reason"),
                                list(ws_healing.CORE_STREAMS),
                                {"live_resumed": True},
                            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if generation == self.connection_generation and not self.stop_requested:
                self.status["errors_total"] += 1
                self.status["last_disconnect_detected_utc"] = now_utc()
                self.status["last_reconnect_error"] = f"{type(exc).__name__}: {exc}"
                append_jsonl(
                    "collector_errors",
                    {
                        "source": name,
                        "stage": "ws_receive",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "connection_generation": generation,
                        "local_received_ts": now_utc(),
                    },
                )
                self.request_reconnect("WS_TASK_DIED_RECOVERY", [name])
        finally:
            if generation == self.connection_generation:
                self.connection_open[name] = False
            self.connections.pop(name, None)

    def request_reconnect(self, reason: str, streams: List[str]) -> bool:
        self.pending_reasons.add(reason)
        self.pending_streams.update(streams)
        if self.reconnect_task is not None and not self.reconnect_task.done():
            ws_healing.append_watchdog_event(
                {
                    "collector_pid": os.getpid(),
                    "collector_instance_id": self.status.get("collector_instance_id"),
                    "connection_generation": self.connection_generation,
                    "event_type": "RECONNECT_SKIPPED_ALREADY_RUNNING",
                    "previous_state": self.status.get("ws_connection_state"),
                    "new_state": self.status.get("ws_connection_state"),
                    "trigger_reason": reason,
                    "trigger_streams": streams,
                }
            )
            return False
        boundary = now_utc()
        try:
            self.pending_snapshot = gap_backfill.create_recovery_snapshot(
                reason,
                boundary,
                collector_pid=os.getpid(),
                collector_instance_id=self.recovery.collector_instance_id,
                streams=gap_backfill.STREAM_ORDER,
            )
        except Exception as exc:
            self.status["last_reconnect_error"] = (
                f"SNAPSHOT_CREATE_FAILED {type(exc).__name__}: {exc}"
            )
            write_status(self.status)
            return False
        self.reconnect_task = asyncio.create_task(self._reconnect_cycle())
        return True

    async def _cancel_generation(self, log_transition: bool = True) -> None:
        if log_transition:
            self.transition(
                "CLOSING_OLD_CONNECTION",
                "OLD_CONNECTION_CLOSED",
                self.status.get("reconnect_reason"),
                list(self.pending_streams),
            )
        tasks = list(self.tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for ws in list(self.connections.values()):
            try:
                await asyncio.wait_for(ws.close(), timeout=5)
            except Exception:
                pass
        self.tasks = {}
        self.connections = {}
        self.connection_open = {name: False for name in STREAMS}

    async def _reconnect_cycle(self) -> None:
        async with self.reconnect_lock:
            reasons = sorted(self.pending_reasons) or ["WATCHDOG_STALE_RECOVERY"]
            streams = sorted(self.pending_streams) or list(ws_healing.CORE_STREAMS)
            self.pending_reasons.clear()
            self.pending_streams.clear()
            reason = "+".join(reasons)
            started = time.monotonic()
            elapsed_since_last = started - self.last_reconnect_monotonic
            if elapsed_since_last < self.config.minimum_reconnect_interval_seconds:
                await asyncio.sleep(
                    self.config.minimum_reconnect_interval_seconds - elapsed_since_last
                )
            self.last_reconnect_monotonic = time.monotonic()
            self.status["reconnect_in_progress"] = True
            self.status["reconnect_reason"] = reason
            self.status["reconnect_trigger_streams"] = streams
            self.status["last_reconnect_attempt_utc"] = now_utc()
            self.status["last_watchdog_trigger_utc"] = now_utc()
            self.status["last_watchdog_reason"] = reason
            self.status["total_watchdog_triggers"] += 1
            self.transition("RECONNECT_REQUESTED", "RECONNECT_REQUESTED", reason, streams)
            snapshot = self.pending_snapshot
            self.pending_snapshot = None
            if snapshot is None:
                self.status["reconnect_in_progress"] = False
                self.status["last_reconnect_error"] = "MISSING_PRECREATED_SNAPSHOT"
                write_status(self.status)
                return
            boundary = pd.Timestamp(snapshot["reconnect_boundary_utc"])
            self.status["snapshot_created"] = True
            self.status["recovery_run_id"] = snapshot["recovery_run_id"]
            self.status["snapshot_checksum"] = snapshot["snapshot_checksum"]
            self.status["gap_candidates_count"] = len(snapshot.get("gap_candidates", []))
            self.status["ledger_pre_registered_count"] = len(snapshot.get("gap_candidates", []))
            self.transition(
                "GAP_PRE_REGISTERED",
                "GAP_RECOVERY_TRIGGERED",
                reason,
                streams,
                {
                    "snapshot_created": True,
                    "recovery_run_id": snapshot["recovery_run_id"],
                    "gap_candidates_count": len(snapshot.get("gap_candidates", [])),
                    "ledger_pre_registered_count": len(snapshot.get("gap_candidates", [])),
                },
            )
            self.silent_stall_until_monotonic = 0.0
            await self._cancel_generation()
            failures = 0
            self.reconnect_failure_started_monotonic = time.monotonic()
            was_offline = False
            while not self.stop_requested:
                forced_offline = (
                    time.monotonic() < self.forced_offline_until_monotonic
                )
                probe = await asyncio.to_thread(
                    self.network_probe, 5.0, forced_offline
                )
                if not probe.get("reachable"):
                    was_offline = True
                    if self.status.get("last_network_offline_utc") is None:
                        self.status["last_network_offline_utc"] = now_utc()
                    failures += 1
                    backoff = ws_healing.compute_backoff_seconds(
                        failures, self.config
                    )
                    self.status["current_backoff_seconds"] = backoff
                    self.transition(
                        "OFFLINE_WAIT",
                        "NETWORK_OFFLINE_DETECTED",
                        reason,
                        streams,
                        {
                            "network_probe_result": probe,
                            "reconnect_attempt": failures,
                            "backoff_seconds": backoff,
                        },
                    )
                    if self._fatal_elapsed():
                        await self._request_fatal("NETWORK_OFFLINE_TIMEOUT", snapshot)
                        return
                    await asyncio.sleep(backoff)
                    continue
                if was_offline:
                    self.status["last_network_restored_utc"] = now_utc()
                    self.status["total_network_recoveries"] += 1
                    self.transition(
                        "RECONNECTING",
                        "NETWORK_RESTORED",
                        reason,
                        streams,
                        {"network_probe_result": probe},
                    )
                failures += 1
                self.status["total_reconnect_attempts"] += 1
                self.status["consecutive_reconnect_failures"] = failures - 1
                self.transition(
                    "RECONNECTING",
                    "NEW_CONNECTION_CREATED",
                    reason,
                    streams,
                    {"reconnect_attempt": failures},
                )
                await self._start_generation()
                try:
                    await asyncio.wait_for(
                        self.live_resumed_event.wait(),
                        timeout=self.config.reconnect_timeout_seconds,
                    )
                    self.status["consecutive_reconnect_failures"] = 0
                    self.status["current_backoff_seconds"] = 0.0
                    self.status["reconnect_in_progress"] = False
                    self.status["last_reconnect_ts"] = boundary
                    self.status["last_reconnect_success_utc"] = now_utc()
                    self.status["total_successful_self_heals"] += 1
                    self.status["total_reconnect_successes"] += 1
                    if "WATCHDOG_STALE_RECOVERY" in reasons:
                        self.status["total_silent_stall_recoveries"] += 1
                    if "PARTIAL_STREAM_RECOVERY" in reasons or "WS_TASK_DIED_RECOVERY" in reasons:
                        self.status["total_partial_task_recoveries"] += 1
                    self.status["last_self_heal_duration_seconds"] = (
                        time.monotonic() - started
                    )
                    self.transition(
                        "LIVE_RESUMED",
                        "LIVE_RECOVERY_COMPLETE",
                        reason,
                        streams,
                        {"live_resumed": True},
                    )
                    self.recovery.schedule_snapshot(snapshot)
                    self.transition("RECOVERED", "LIVE_RECOVERY_COMPLETE", reason, streams)
                    return
                except asyncio.TimeoutError:
                    self.status["consecutive_reconnect_failures"] = failures
                    self.status["last_reconnect_error"] = "LIVE_RESUME_TIMEOUT"
                    await self._cancel_generation()
                    if (
                        failures >= self.config.max_consecutive_reconnect_failures
                        or self._fatal_elapsed()
                    ):
                        await self._request_fatal(
                            "MAX_RECONNECT_FAILURES", snapshot
                        )
                        return
                    backoff = ws_healing.compute_backoff_seconds(
                        failures, self.config
                    )
                    self.status["current_backoff_seconds"] = backoff
                    self.transition(
                        "BACKOFF",
                        "RECONNECT_REQUESTED",
                        reason,
                        streams,
                        {"reconnect_attempt": failures, "backoff_seconds": backoff},
                    )
                    await asyncio.sleep(backoff)

    def _fatal_elapsed(self) -> bool:
        if self.reconnect_failure_started_monotonic is None:
            return False
        return (
            time.monotonic() - self.reconnect_failure_started_monotonic
            >= self.config.process_exit_after_failure_seconds
        )

    async def _request_fatal(
        self, reason: str, snapshot: Dict[str, Any]
    ) -> None:
        self.status["fatal_restart_requested"] = True
        self.status["reconnect_in_progress"] = False
        self.status["reconnect_reason"] = reason
        self.transition(
            "FATAL_RESTART_REQUESTED",
            "FATAL_PROCESS_RESTART_REQUESTED",
            reason,
            list(self.pending_streams),
            {
                "snapshot_created": True,
                "recovery_run_id": snapshot.get("recovery_run_id"),
            },
        )
        self.fatal_event.set()

    def _consume_fault_request(self) -> None:
        if not FAULT_REQUEST_PATH.exists():
            return
        try:
            request = json.loads(FAULT_REQUEST_PATH.read_text(encoding="utf-8"))
            FAULT_REQUEST_PATH.unlink(missing_ok=True)
        except Exception:
            return
        nonce = str(request.get("nonce"))
        if nonce == self.last_fault_nonce:
            return
        self.last_fault_nonce = nonce
        fault_type = request.get("fault_type")
        seconds = float(request.get("seconds") or 0)
        if fault_type == "silent_stall":
            self.silent_stall_until_monotonic = time.monotonic() + seconds
        elif fault_type == "network_unreachable":
            self.forced_offline_until_monotonic = time.monotonic() + seconds
            self.request_reconnect("NETWORK_RESTORED_RECOVERY", list(ws_healing.CORE_STREAMS))
        elif fault_type == "stop_aggtrade_task":
            task = self.tasks.get("aggTrade")
            if task and not task.done():
                task.cancel()
        elif fault_type == "sleep_wake":
            self.request_reconnect("SLEEP_WAKE_RECOVERY", list(ws_healing.CORE_STREAMS))
        elif fault_type == "close_ws":
            self.request_reconnect("WS_CONNECTION_CLOSED", list(ws_healing.CORE_STREAMS))

    def _apply_health(self, health: Dict[str, Any]) -> None:
        self.status["aggtrade_age_seconds"] = health["aggtrade_age_seconds"]
        self.status["markprice_age_seconds"] = health["markprice_age_seconds"]
        self.status["kline_age_seconds"] = health["kline_age_seconds"]
        self.status["market_ws_age_seconds"] = health["market_ws_age_seconds"]
        self.status["oi_age_seconds"] = health["oi_age_seconds"]
        self.status["aggtrade_task_alive"] = bool(
            health["task_alive_states"].get("aggTrade")
        )
        self.status["markprice_task_alive"] = bool(
            health["task_alive_states"].get("markPrice")
        )
        self.status["kline_task_alive"] = bool(
            health["task_alive_states"].get("kline_1m")
        )
        self.status["forceorder_task_alive"] = bool(
            health["task_alive_states"].get("forceOrder")
        )
        self.status["aggtrade_receiving"] = health["receiving"]["aggTrade"]
        self.status["markprice_receiving"] = health["receiving"]["markPrice"]
        self.status["kline_1m_receiving"] = health["receiving"]["kline_1m"]
        self.status["market_ws_receiving"] = health["health_quorum_pass"]
        self.status["forceorder_receiving"] = (
            self.status["forceorder_task_alive"]
            and self.connection_open.get("forceOrder", False)
            and health["health_quorum_pass"]
        )
        self.status["ws_connection_open"] = all(
            self.connection_open.get(name, False)
            for name in ws_healing.CORE_STREAMS
        )
        self.status["ws_ping_ok"] = self.status["ws_connection_open"]
        self.status["health_quorum_pass"] = health["health_quorum_pass"]

    async def _watchdog_loop(self) -> None:
        self.transition("CONNECTED", "WATCHDOG_STARTED", "INITIAL_CONNECTION")
        while not self.stop_requested:
            loop_started = time.monotonic()
            elapsed = loop_started - self.last_watchdog_monotonic
            self.last_watchdog_monotonic = loop_started
            self._consume_fault_request()
            task_alive = self._task_alive()
            health = ws_healing.evaluate_ws_health(
                self.status,
                task_alive,
                self.connection_open,
                self.config,
                startup_age_seconds=loop_started - self.started_monotonic,
            )
            self._apply_health(health)
            self.status["watchdog_last_check_utc"] = now_utc()
            if (
                elapsed > self.config.sleep_wake_elapsed_seconds
                and loop_started - self.started_monotonic
                > self.config.startup_grace_seconds
            ):
                self.request_reconnect(
                    "SLEEP_WAKE_RECOVERY", list(ws_healing.CORE_STREAMS)
                )
            elif health["trigger_reason"]:
                self.status["last_stale_detected_utc"] = now_utc()
                if not self.status.get("reconnect_in_progress"):
                    self.transition(
                        "STALE_DETECTED",
                        "HARD_STALE_DETECTED"
                        if health["hard_stale_streams"]
                        else "WS_TASK_DONE_DETECTED",
                        health["trigger_reason"],
                        health["trigger_streams"],
                    )
                self.request_reconnect(
                    health["trigger_reason"], health["trigger_streams"]
                )
            elif (
                health["health_quorum_pass"]
                and not self.status.get("reconnect_in_progress")
            ):
                self.status["ws_connection_state"] = "HEALTHY"
            write_status(self.status)
            if self.fatal_event.is_set():
                return
            await asyncio.sleep(self.config.check_interval_seconds)

    async def stop(self) -> None:
        self.stop_requested = True
        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
        await self._cancel_generation(
            log_transition=not self.status.get("fatal_restart_requested", False)
        )
        self.status["watchdog_running"] = False
        write_status(self.status)


async def ws_probe(stream: str, seconds: int) -> Dict[str, Any]:
    ensure_dirs()
    stream_map = {**STREAMS, "combined": COMBINED_URL}
    if stream not in stream_map:
        return {"verdict": "STANDALONE_WS_FAIL", "stream": stream, "error": "unknown_stream"}
    url = stream_map[stream]
    started = now_utc()
    rows = []
    errors = []
    deadline = time.time() + seconds
    connected = False
    try:
        async with websockets.connect(url, ping_interval=None, open_timeout=20, close_timeout=5, compression=None) as ws:
            connected = True
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=min(5, max(1, deadline - time.time())))
                    recv_ts = now_utc()
                    stream_name, payload = unwrap_payload(msg)
                    rows.append({"local_received_ts": recv_ts, "stream": stream_name or stream, "payload": payload})
                    if len(rows) <= 20:
                        out = DIAG_ROOT / "raw_probe" / f"{stream}_first_messages.jsonl"
                        with out.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(clean(rows[-1]), ensure_ascii=False, default=str) + "\n")
                except asyncio.TimeoutError:
                    continue
    except Exception as exc:
        errors.append({"stream": stream, "error": str(exc), "traceback": traceback.format_exc()})
    count = len(rows)
    verdict = "STANDALONE_WS_FAIL"
    if stream == "aggTrade" and count > 0:
        verdict = "STANDALONE_WS_AGGTRADE_OK"
    elif stream == "markPrice" and count > 0:
        verdict = "STANDALONE_WS_MARKPRICE_OK"
    elif stream == "kline_1m" and count > 0:
        verdict = "STANDALONE_WS_KLINE_OK"
    elif stream == "forceOrder":
        verdict = "STANDALONE_WS_FORCEORDER_EVENTS_OK" if count > 0 else "STANDALONE_WS_FORCEORDER_CONNECTED_NO_EVENTS" if connected else "STANDALONE_WS_FAIL"
    elif stream == "combined" and count > 0:
        verdict = "STANDALONE_WS_COMBINED_OK"
    score = {"stream": stream, "url": url, "connected": connected, "seconds": seconds, "messages": count, "started": started, "ended": now_utc(), "verdict": verdict}
    pd.DataFrame([score]).to_csv(DIAG_ROOT / "probe" / f"{stream}_standalone_ws_probe_scorecard.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(DIAG_ROOT / "probe" / f"{stream}_standalone_ws_probe_errors.csv", index=False)
    return score


async def oi_loop(status: Dict[str, Any], stop_at: float | None, interval: int = 60) -> None:
    while stop_at is None or time.time() < stop_at:
        try:
            data = public_get("/fapi/v1/openInterest", {"symbol": SYMBOL})
            collected = now_utc()
            event_time = pd.to_datetime(data.get("time"), unit="ms", utc=True) if data.get("time") else collected
            row = {"source": "open_interest", "local_received_ts": collected, "payload": data}
            row.update(gap_backfill.provenance("LIVE_WS", "/fapi/v1/openInterest", "fapi.binance.com", event_time))
            row.update(
                {
                    "collection_mode": "LIVE_REST_POLL",
                    "source_type": "PUBLIC_REST",
                    "data_quality_tier": "A",
                    "live_observed": True,
                    "backfilled": False,
                    "reconstructed": False,
                    "strict_forward_eval_eligible": True,
                    "historical_research_eligible": True,
                    "exclude_from_forward_eval": False,
                    "exclude_reason": None,
                    "forceorder_complete": None,
                    "oi_resolution": f"{interval}s",
                }
            )
            append_jsonl("open_interest", row)
            status["oi_snapshots_total"] += 1
            status["last_oi_poll_time_utc"] = now_utc()
            write_status(status)
        except Exception as exc:
            status["errors_total"] += 1
            append_jsonl("collector_errors", {"source": "open_interest", "error": str(exc), "local_received_ts": now_utc()})
            write_status(status)
        await asyncio.sleep(interval)


async def run_collector(minutes: int | None) -> Dict[str, Any]:
    ensure_dirs()
    guard = public_only_guard()
    if guard["verdict"] != "PUBLIC_WS_ONLY_PASS":
        return guard
    try:
        collector_lock = gap_backfill.file_lock("collector_instance", blocking=False)
        collector_lock.__enter__()
    except BlockingIOError:
        return {"verdict": "COLLECTOR_INSTANCE_ALREADY_RUNNING", "is_running": False, "production_ready": False, "promotion_ready": False}
    try:
        pid_path().write_text(str(os.getpid()), encoding="utf-8")
        startup_cycle_started_at = now_utc()
        collector_instance_id = (
            f"collector_{os.getpid()}_"
            f"{startup_cycle_started_at.strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        startup_snapshot = gap_backfill.create_recovery_snapshot(
            "PROCESS_STARTUP",
            startup_cycle_started_at,
            collector_pid=os.getpid(),
            collector_instance_id=collector_instance_id,
            startup_cycle_started_at=startup_cycle_started_at,
        )
        status = base_status()
        status["collector_instance_id"] = collector_instance_id
        status["last_reconnect_ts"] = startup_cycle_started_at
        write_status(status)
        stop_at = time.time() + minutes * 60 if minutes else None
        recovery = RecoveryCoordinator(status, collector_instance_id)
        recovery.schedule_snapshot(startup_snapshot)
        manager = WSSelfHealingManager(status, recovery)
        await manager.start()
        oi_task = asyncio.create_task(oi_loop(status, stop_at))
        try:
            while stop_at is None or time.time() < stop_at:
                if manager.fatal_event.is_set():
                    raise RuntimeError(
                        "FATAL_PROCESS_RESTART_REQUESTED: "
                        f"{status.get('reconnect_reason')}"
                    )
                await asyncio.sleep(1)
        finally:
            await manager.stop()
            if not oi_task.done():
                oi_task.cancel()
            await asyncio.gather(oi_task, return_exceptions=True)
            status["is_running"] = False
            status["last_event_time_utc"] = status.get("last_event_time_utc") or now_utc()
            write_status(status)
        return status
    finally:
        collector_lock.__exit__(None, None, None)


def stop_safe() -> Dict[str, Any]:
    if not pid_path().exists():
        return {"stopped": False, "reason": "pidfile_missing"}
    pid = int(pid_path().read_text(encoding="utf-8").strip())
    if pid == os.getpid() or pid <= 1:
        return {"stopped": False, "reason": "invalid_pid"}
    try:
        os.kill(pid, signal.SIGTERM)
        return {"stopped": True, "pid": pid}
    except Exception as exc:
        return {"stopped": False, "pid": pid, "error": str(exc)}


def write_fault_injection(fault_type: str, seconds: float = 0) -> Dict[str, Any]:
    request = {
        "nonce": f"fault_{now_utc().strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}",
        "fault_type": fault_type,
        "seconds": max(0.0, float(seconds)),
        "requested_at_utc": now_utc(),
        "debug_only": True,
        "changes_system_network": False,
        "production_ready": False,
        "promotion_ready": False,
    }
    ws_healing.atomic_json(FAULT_REQUEST_PATH, request)
    return {"verdict": "DEBUG_FAULT_INJECTION_REQUESTED", **request}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--minutes", type=int, default=0)
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop-safe", action="store_true")
    p.add_argument("--ws-probe", action="store_true")
    p.add_argument("--stream", default="aggTrade")
    p.add_argument("--seconds", type=int, default=30)
    p.add_argument("--fault-inject-ws-silent-stall", action="store_true")
    p.add_argument("--fault-inject-close-ws", action="store_true")
    p.add_argument("--fault-inject-stop-aggtrade-task", action="store_true")
    p.add_argument("--fault-inject-network-unreachable-seconds", type=float, default=0)
    p.add_argument("--fault-inject-sleep-wake-gap-seconds", type=float, default=0)
    p.add_argument("--fault-seconds", type=float, default=135)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"verdict": "PUBLIC_LIVE_COLLECTOR_IMPLEMENTED", "guard": public_only_guard(), "streams": STREAMS, "production_ready": False, "promotion_ready": False}
    elif args.status:
        res = read_status()
    elif args.stop_safe:
        res = stop_safe()
    elif args.ws_probe:
        res = asyncio.run(ws_probe(args.stream, args.seconds))
    elif args.fault_inject_ws_silent_stall:
        res = write_fault_injection("silent_stall", args.fault_seconds)
    elif args.fault_inject_close_ws:
        res = write_fault_injection("close_ws")
    elif args.fault_inject_stop_aggtrade_task:
        res = write_fault_injection("stop_aggtrade_task")
    elif args.fault_inject_network_unreachable_seconds > 0:
        res = write_fault_injection(
            "network_unreachable",
            args.fault_inject_network_unreachable_seconds,
        )
    elif args.fault_inject_sleep_wake_gap_seconds > 0:
        res = write_fault_injection(
            "sleep_wake", args.fault_inject_sleep_wake_gap_seconds
        )
    elif args.run:
        res = asyncio.run(run_collector(args.minutes or None))
    else:
        res = read_status()
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
