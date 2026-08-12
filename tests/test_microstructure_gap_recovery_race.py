from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from scripts.diagnostics import microstructure_gap_backfill as gap


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(gap, "GAP_ROOT", tmp_path / "gap")
    monkeypatch.setattr(gap, "LIVE_ROOT", tmp_path / "live")
    monkeypatch.setattr(gap.time, "sleep", lambda _: None)
    return tmp_path


def append_live(stream: str, row: Dict[str, Any]) -> Path:
    path = gap.LIVE_ROOT / "normalized" / stream / "symbol=BTCUSDT/date=2026-01-01/events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return path


def seed_watermarks(at: str = "2026-01-01T00:00:00Z") -> None:
    append_live("ws_aggTrade", {"event_ts": at, "agg_trade_id": 10, "symbol": "BTCUSDT"})
    append_live("ws_markPrice", {"event_ts": at, "mark_price": 100, "symbol": "BTCUSDT"})


def fake_public_get(base: str, path: str, params: Dict[str, Any]) -> Any:
    start = int(params.get("startTime", 0))
    end = int(params.get("endTime", start))
    if "fromId" in params:
        return []
    if path == "/fapi/v1/aggTrades":
        return [
            {"a": value, "p": "100", "q": "1", "T": value, "m": False}
            for value in range(start, end + 1, 60_000)
        ]
    if path == "/fapi/v1/markPriceKlines":
        return [
            [value, "100", "101", "99", "100.5", "0", value + 59_999]
            for value in range(start, end + 1, 60_000)
        ]
    raise AssertionError(path)


@pytest.mark.parametrize("hours", [0.5, 4.5, 6, 10, 25])
def test_snapshot_gap_survives_mutable_live_watermark(
    isolated: Path, hours: float
) -> None:
    seed_watermarks()
    boundary = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=hours)
    snapshot = gap.create_recovery_snapshot(
        "PROCESS_STARTUP",
        boundary,
        collector_pid=123,
        collector_instance_id="collector-old",
        streams=["futures_aggTrade", "mark_price"],
    )
    append_live(
        "ws_aggTrade",
        {"event_ts": str(boundary), "agg_trade_id": 999, "symbol": "BTCUSDT"},
    )
    append_live(
        "ws_markPrice",
        {"event_ts": str(boundary), "mark_price": 101, "symbol": "BTCUSDT"},
    )
    candidates = {row["stream"]: row for row in snapshot["gap_candidates"]}
    assert set(candidates) == {"futures_aggTrade", "mark_price"}
    assert candidates["futures_aggTrade"]["gap_start_utc"] == "2026-01-01T00:00:00.001000+00:00"
    assert pd.Timestamp(candidates["futures_aggTrade"]["gap_end_utc"]) < boundary
    ledger = gap.read_ledger()
    assert set(ledger["stream"]) == {"futures_aggTrade", "mark_price"}
    assert set(ledger["backfill_status"]) == {"GAP_DETECTED"}


def test_reconnect_debounce_uses_snapshot_not_mutable_watermark(isolated: Path) -> None:
    seed_watermarks()
    snapshot = gap.create_recovery_snapshot(
        "WS_RECONNECT",
        "2026-01-01T06:00:00Z",
        collector_instance_id="collector-a",
        streams=["futures_aggTrade", "mark_price"],
    )
    frozen = {
        row["stream"]: (row["gap_start_utc"], row["gap_end_utc"])
        for row in snapshot["gap_candidates"]
    }
    append_live(
        "ws_aggTrade",
        {"event_ts": "2026-01-01T06:00:01Z", "agg_trade_id": 999, "symbol": "BTCUSDT"},
    )
    append_live(
        "ws_markPrice",
        {"event_ts": "2026-01-01T06:00:01Z", "mark_price": 101, "symbol": "BTCUSDT"},
    )
    reread = gap.read_recovery_snapshot(gap.snapshot_path(snapshot["recovery_run_id"]))
    assert {
        row["stream"]: (row["gap_start_utc"], row["gap_end_utc"])
        for row in reread["gap_candidates"]
    } == frozen


def test_crash_after_preregistration_resumes_same_gap(isolated: Path) -> None:
    seed_watermarks()
    snapshot = gap.create_recovery_snapshot(
        "PROCESS_STARTUP",
        "2026-01-01T00:30:00Z",
        collector_instance_id="collector-a",
        streams=["futures_aggTrade", "mark_price"],
    )
    before = gap.read_ledger()
    assert len(before) == 2
    assert set(before["backfill_status"]) == {"GAP_DETECTED"}
    result = gap.run_recovery_snapshot(
        gap.snapshot_path(snapshot["recovery_run_id"]),
        current_collector_instance_id="collector-a",
        fetcher=fake_public_get,
    )
    assert result["verdict"] == "AUTO_GAP_BACKFILL_IMPLEMENTED"
    after = gap.read_ledger()
    assert len(after) == 2
    assert after["gap_id"].nunique() == 2
    assert set(after["backfill_status"]) == {"BACKFILL_COMPLETE"}


def test_same_snapshot_is_idempotent(isolated: Path) -> None:
    seed_watermarks()
    snapshot = gap.create_recovery_snapshot(
        "PROCESS_STARTUP",
        "2026-01-01T00:30:00Z",
        collector_instance_id="collector-a",
        streams=["futures_aggTrade", "mark_price"],
    )
    first = gap.run_recovery_snapshot(
        snapshot, "collector-a", fake_public_get, include_pending=False
    )
    second = gap.run_recovery_snapshot(
        snapshot, "collector-a", fake_public_get, include_pending=False
    )
    assert sum(row["rows_inserted"] for row in first["results"]) > 0
    assert sum(row["rows_inserted"] for row in second["results"]) == 0
    ledger = gap.read_ledger()
    assert len(ledger) == ledger["gap_id"].nunique()
    assert ledger["ledger_registered_at_utc"].notna().all()


def test_snapshot_is_immutable_and_checksum_stable(isolated: Path) -> None:
    seed_watermarks()
    snapshot = gap.create_recovery_snapshot(
        "PROCESS_STARTUP",
        "2026-01-01T00:30:00Z",
        collector_instance_id="collector-a",
        streams=["futures_aggTrade"],
    )
    path = gap.snapshot_path(snapshot["recovery_run_id"])
    before = path.read_bytes()
    append_live(
        "ws_aggTrade",
        {"event_ts": "2026-01-01T00:30:01Z", "agg_trade_id": 999, "symbol": "BTCUSDT"},
    )
    after = path.read_bytes()
    assert before == after
    assert gap.read_recovery_snapshot(path)["snapshot_checksum"] == snapshot["snapshot_checksum"]


def test_mark_price_reconstruction_never_impersonates_ws(isolated: Path) -> None:
    seed_watermarks()
    snapshot = gap.create_recovery_snapshot(
        "PROCESS_STARTUP",
        "2026-01-01T00:30:00Z",
        collector_instance_id="collector-a",
        streams=["mark_price"],
    )
    gap.run_recovery_snapshot(snapshot, "collector-a", fake_public_get)
    stored = pd.read_parquet(gap.stream_data_path("mark_price"))
    assert set(stored["collection_mode"]) == {"RECONSTRUCTED_LOWER_RESOLUTION"}
    assert set(stored["data_quality_tier"]) == {"C"}
    assert stored["reconstructed"].all()
    assert not stored["live_observed"].any()
    assert not stored["strict_forward_eval_eligible"].any()
    assert set(stored["data_type"]) == {"mark_price_kline_reconstruction"}


def test_forceorder_preregister_then_tier_d_without_fake_rows(isolated: Path) -> None:
    seed_watermarks()
    snapshot = gap.create_recovery_snapshot(
        "PROCESS_STARTUP",
        "2026-01-01T00:30:00Z",
        collector_instance_id="collector-a",
        streams=["futures_aggTrade", "forceOrder"],
    )
    gap.run_recovery_snapshot(snapshot, "collector-a", fake_public_get)
    row = gap.read_ledger().query("stream == 'forceOrder'").iloc[0]
    assert row["backfill_status"] == "UNRECOVERABLE_STREAM_GAP"
    assert row["provenance_tier"] == "D"
    assert bool(row["forceorder_complete"]) is False
    assert not gap.stream_data_path("forceOrder").exists()


def test_stale_delayed_task_skips_but_preserves_preregistered_gap(
    isolated: Path,
) -> None:
    seed_watermarks()
    snapshot = gap.create_recovery_snapshot(
        "WS_RECONNECT",
        "2026-01-01T00:30:00Z",
        collector_instance_id="collector-old",
        streams=["futures_aggTrade"],
    )
    result = gap.run_recovery_snapshot(
        snapshot,
        current_collector_instance_id="collector-new",
        fetcher=fake_public_get,
    )
    assert result["verdict"] == "STALE_RECOVERY_TASK_SKIPPED"
    ledger = gap.read_ledger()
    assert ledger.iloc[0]["backfill_status"] == "GAP_DETECTED"
