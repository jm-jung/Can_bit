from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from scripts.diagnostics import microstructure_ws_self_healing as healing
from scripts.diagnostics import run_microstructure_public_live_collector as collector


def fresh_status(now: pd.Timestamp) -> Dict[str, Any]:
    return {
        "last_aggtrade_time_utc": now,
        "last_markprice_time_utc": now,
        "last_kline_time_utc": now,
        "last_oi_poll_time_utc": now,
    }


def alive(value: bool = True) -> Dict[str, bool]:
    return {
        "aggTrade": value,
        "markPrice": value,
        "kline_1m": value,
        "forceOrder": value,
    }


def test_silent_socket_stall_triggers_quorum_reconnect() -> None:
    now = pd.Timestamp("2026-01-01T01:00:00Z")
    status = fresh_status(now)
    status["last_aggtrade_time_utc"] = now - pd.Timedelta(seconds=121)
    status["last_markprice_time_utc"] = now - pd.Timedelta(seconds=121)
    health = healing.evaluate_ws_health(
        status, alive(), alive(), healing.WatchdogConfig(), now, 300
    )
    assert health["trigger_reason"] == "WATCHDOG_STALE_RECOVERY"
    assert set(health["trigger_streams"]) == {"aggTrade", "markPrice"}
    assert not health["receiving"]["aggTrade"]
    assert not health["receiving"]["markPrice"]


def test_partial_task_death_not_hidden_by_fresh_oi() -> None:
    now = pd.Timestamp("2026-01-01T01:00:00Z")
    tasks = alive()
    tasks["aggTrade"] = False
    health = healing.evaluate_ws_health(
        fresh_status(now), tasks, alive(), healing.WatchdogConfig(), now, 300
    )
    assert health["trigger_reason"] == "WS_TASK_DIED_RECOVERY"
    assert health["trigger_streams"] == ["aggTrade"]


def test_markprice_kline_stall_fails_core_quorum() -> None:
    now = pd.Timestamp("2026-01-01T01:00:00Z")
    status = fresh_status(now)
    status["last_markprice_time_utc"] = now - pd.Timedelta(minutes=5)
    status["last_kline_time_utc"] = now - pd.Timedelta(minutes=5)
    health = healing.evaluate_ws_health(
        status, alive(), alive(), healing.WatchdogConfig(), now, 300
    )
    assert health["trigger_reason"] == "WATCHDOG_STALE_RECOVERY"
    assert not health["health_quorum_pass"]


def test_forceorder_sparse_does_not_trigger() -> None:
    now = pd.Timestamp("2026-01-01T01:00:00Z")
    health = healing.evaluate_ws_health(
        fresh_status(now), alive(), alive(), healing.WatchdogConfig(), now, 300
    )
    assert health["trigger_reason"] is None
    assert health["health_quorum_pass"]


def test_exponential_backoff_is_bounded_and_jittered() -> None:
    config = healing.WatchdogConfig(
        backoff_initial_seconds=2,
        backoff_max_seconds=60,
        backoff_jitter_ratio=0.2,
    )
    values = [
        healing.compute_backoff_seconds(i, config, random_unit=0.5)
        for i in range(1, 8)
    ]
    assert values == [2, 4, 8, 16, 32, 60, 60]
    assert healing.compute_backoff_seconds(1, config, 0.0) == pytest.approx(1.6)
    assert healing.compute_backoff_seconds(1, config, 1.0) == pytest.approx(2.4)


class FakeRecovery:
    collector_instance_id = "collector-test"

    def __init__(self) -> None:
        self.snapshots: list[Dict[str, Any]] = []

    def schedule_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.snapshots.append(snapshot)


def manager_status() -> Dict[str, Any]:
    status = collector.base_status()
    status["collector_instance_id"] = "collector-test"
    return status


def fake_snapshot() -> Dict[str, Any]:
    return {
        "recovery_run_id": "run-test",
        "collector_instance_id": "collector-test",
        "snapshot_checksum": "checksum",
        "reconnect_boundary_utc": "2026-01-01T00:00:00Z",
        "gap_candidates": [{"gap_id": "gap-test"}],
    }


def test_reconnect_single_flight_creates_one_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        recovery = FakeRecovery()
        manager = collector.WSSelfHealingManager(manager_status(), recovery)
        count = 0
        blocker = asyncio.Event()

        def create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            nonlocal count
            count += 1
            return fake_snapshot()

        async def blocked_cycle() -> None:
            await blocker.wait()

        monkeypatch.setattr(collector.gap_backfill, "create_recovery_snapshot", create)
        monkeypatch.setattr(manager, "_reconnect_cycle", blocked_cycle)
        monkeypatch.setattr(collector, "write_status", lambda status: None)
        monkeypatch.setattr(healing, "append_watchdog_event", lambda row: None)
        assert manager.request_reconnect("WATCHDOG_STALE_RECOVERY", ["aggTrade"])
        assert not manager.request_reconnect(
            "PARTIAL_STREAM_RECOVERY", ["markPrice"]
        )
        assert count == 1
        blocker.set()
        await manager.reconnect_task

    asyncio.run(scenario())


def test_offline_then_online_reuses_one_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        probes = [
            {"reachable": False, "error": "offline"},
            {"reachable": True, "error": None},
        ]

        def probe(timeout: float, forced: bool) -> Dict[str, Any]:
            return probes.pop(0)

        recovery = FakeRecovery()
        config = healing.WatchdogConfig(
            reconnect_timeout_seconds=1,
            backoff_initial_seconds=0,
            minimum_reconnect_interval_seconds=0,
        )
        manager = collector.WSSelfHealingManager(
            manager_status(), recovery, config=config, network_probe=probe
        )
        manager.pending_reasons = {"NETWORK_RESTORED_RECOVERY"}
        manager.pending_streams = {"aggTrade", "markPrice", "kline_1m"}
        manager.pending_snapshot = fake_snapshot()

        async def no_cancel() -> None:
            return None

        async def instant_generation() -> None:
            manager.connection_generation += 1
            manager.live_resumed_event = asyncio.Event()
            manager.live_resumed_event.set()

        monkeypatch.setattr(manager, "_cancel_generation", no_cancel)
        monkeypatch.setattr(manager, "_start_generation", instant_generation)
        monkeypatch.setattr(collector, "write_status", lambda status: None)
        monkeypatch.setattr(healing, "append_watchdog_event", lambda row: None)
        await manager._reconnect_cycle()
        assert not probes
        assert manager.status["total_network_recoveries"] == 1
        assert manager.status["consecutive_reconnect_failures"] == 0
        assert len(recovery.snapshots) == 1

    asyncio.run(scenario())


def test_stale_generation_event_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(collector, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(collector, "NORMALIZED_DIR", tmp_path / "normalized")
        monkeypatch.setattr(healing, "append_watchdog_event", lambda row: None)
        persisted = await collector.persist_ws_message(
            "aggTrade", "{}", manager_status(), generation=1, current_generation=2
        )
        assert not persisted
        assert not (tmp_path / "raw").exists()

    asyncio.run(scenario())


def test_fatal_fallback_requests_nonzero_process_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        manager = collector.WSSelfHealingManager(manager_status(), FakeRecovery())
        monkeypatch.setattr(collector, "write_status", lambda status: None)
        monkeypatch.setattr(healing, "append_watchdog_event", lambda row: None)
        await manager._request_fatal("MAX_RECONNECT_FAILURES", fake_snapshot())
        assert manager.fatal_event.is_set()
        assert manager.status["fatal_restart_requested"]
        assert manager.status["ws_connection_state"] == "FATAL_RESTART_REQUESTED"

    asyncio.run(scenario())


def test_receiving_flags_follow_timestamp_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp.now(tz="UTC")
    status = manager_status()
    status.update(fresh_status(now))
    status["last_aggtrade_time_utc"] = now - pd.Timedelta(minutes=10)
    manager = collector.WSSelfHealingManager(status, FakeRecovery())
    manager.tasks = {name: object() for name in collector.STREAMS}  # type: ignore[assignment]
    manager.connection_open = {name: True for name in collector.STREAMS}
    health = healing.evaluate_ws_health(
        status, alive(), alive(), manager.config, now, 300
    )
    manager._apply_health(health)
    assert not status["aggtrade_receiving"]
    assert status["markprice_receiving"]
    assert status["kline_1m_receiving"]


def test_fault_injection_never_changes_system_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "FAULT_REQUEST_PATH", tmp_path / "fault.json")
    result = collector.write_fault_injection("network_unreachable", 5)
    assert result["debug_only"]
    assert not result["changes_system_network"]
    payload = healing.clean(
        __import__("json").loads((tmp_path / "fault.json").read_text())
    )
    assert payload["fault_type"] == "network_unreachable"


def test_three_reconnect_failures_then_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        recovery = FakeRecovery()
        config = healing.WatchdogConfig(
            reconnect_timeout_seconds=0.01,
            max_consecutive_reconnect_failures=5,
            backoff_initial_seconds=0,
            minimum_reconnect_interval_seconds=0,
        )
        manager = collector.WSSelfHealingManager(
            manager_status(),
            recovery,
            config=config,
            network_probe=lambda timeout, forced: {"reachable": True},
        )
        manager.pending_reasons = {"WATCHDOG_STALE_RECOVERY"}
        manager.pending_streams = set(healing.CORE_STREAMS)
        manager.pending_snapshot = fake_snapshot()
        attempts = 0

        async def no_cancel() -> None:
            return None

        async def generation() -> None:
            nonlocal attempts
            attempts += 1
            manager.live_resumed_event = asyncio.Event()
            if attempts == 4:
                manager.live_resumed_event.set()

        monkeypatch.setattr(manager, "_cancel_generation", no_cancel)
        monkeypatch.setattr(manager, "_start_generation", generation)
        monkeypatch.setattr(collector, "write_status", lambda status: None)
        monkeypatch.setattr(healing, "append_watchdog_event", lambda row: None)
        await manager._reconnect_cycle()
        assert attempts == 4
        assert manager.status["consecutive_reconnect_failures"] == 0
        assert len(recovery.snapshots) == 1

    asyncio.run(scenario())


def test_max_reconnect_failures_requests_fatal_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        config = healing.WatchdogConfig(
            reconnect_timeout_seconds=0.01,
            max_consecutive_reconnect_failures=2,
            backoff_initial_seconds=0,
            minimum_reconnect_interval_seconds=0,
        )
        manager = collector.WSSelfHealingManager(
            manager_status(),
            FakeRecovery(),
            config=config,
            network_probe=lambda timeout, forced: {"reachable": True},
        )
        manager.pending_reasons = {"WS_TASK_DIED_RECOVERY"}
        manager.pending_streams = {"aggTrade"}
        manager.pending_snapshot = fake_snapshot()

        async def no_cancel() -> None:
            return None

        async def never_ready() -> None:
            manager.live_resumed_event = asyncio.Event()

        monkeypatch.setattr(manager, "_cancel_generation", no_cancel)
        monkeypatch.setattr(manager, "_start_generation", never_ready)
        monkeypatch.setattr(collector, "write_status", lambda status: None)
        monkeypatch.setattr(healing, "append_watchdog_event", lambda row: None)
        await manager._reconnect_cycle()
        assert manager.fatal_event.is_set()
        assert manager.status["fatal_restart_requested"]

    asyncio.run(scenario())


def test_sleep_wake_fault_requests_snapshot_before_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery = FakeRecovery()
    manager = collector.WSSelfHealingManager(manager_status(), recovery)
    monkeypatch.setattr(collector, "FAULT_REQUEST_PATH", tmp_path / "fault.json")
    monkeypatch.setattr(collector, "write_status", lambda status: None)
    monkeypatch.setattr(healing, "append_watchdog_event", lambda row: None)
    monkeypatch.setattr(
        collector.gap_backfill,
        "create_recovery_snapshot",
        lambda *args, **kwargs: fake_snapshot(),
    )
    healing.atomic_json(
        collector.FAULT_REQUEST_PATH,
        {"nonce": "sleep-1", "fault_type": "sleep_wake", "seconds": 300},
    )

    async def scenario() -> None:
        blocker = asyncio.Event()

        async def blocked_cycle() -> None:
            await blocker.wait()

        monkeypatch.setattr(manager, "_reconnect_cycle", blocked_cycle)
        manager._consume_fault_request()
        assert manager.pending_snapshot is not None
        assert "SLEEP_WAKE_RECOVERY" in manager.pending_reasons
        blocker.set()
        await manager.reconnect_task

    asyncio.run(scenario())
