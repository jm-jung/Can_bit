from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from scripts.diagnostics import microstructure_gap_backfill as gap
from scripts.diagnostics import run_microstructure_public_live_collector as collector


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    gap_root = tmp_path / "gap"
    live_root = tmp_path / "live"
    monkeypatch.setattr(gap, "GAP_ROOT", gap_root)
    monkeypatch.setattr(gap, "LIVE_ROOT", live_root)
    monkeypatch.setattr(gap.time, "sleep", lambda _: None)
    return tmp_path


def fake_public_get(base: str, path: str, params: Dict[str, Any]) -> Any:
    start = int(params.get("startTime", 0))
    end = int(params.get("endTime", start))
    if "fromId" in params:
        return []
    if path == "/fapi/v1/aggTrades":
        rows = []
        for value in range(start, end + 1, 60_000):
            rows.append({"a": value, "p": "100.0", "q": "2.0", "T": value, "m": False})
        return rows
    if path in {"/fapi/v1/klines", "/api/v3/klines", "/fapi/v1/markPriceKlines", "/fapi/v1/premiumIndexKlines"}:
        return [
            [value, "100", "101", "99", "100.5", "3", value + 59_999, "0", 1, "0", "0", "0"]
            for value in range(start, end + 1, 60_000)
        ]
    if path == "/fapi/v1/fundingRate":
        return []
    if path == "/futures/data/openInterestHist":
        return [
            {"symbol": "BTCUSDT", "sumOpenInterest": "10", "sumOpenInterestValue": "1000", "timestamp": value}
            for value in range(start, end + 1, 300_000)
        ]
    raise AssertionError(path)


@pytest.mark.parametrize("hours", [0.5, 4.5, 6, 10, 25])
def test_synthetic_gap_duration_and_chunking(isolated: Path, hours: float) -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    end = start + pd.Timedelta(hours=hours) - pd.Timedelta(milliseconds=1)
    gaps = gap.detect_gaps(
        streams=["futures_aggTrade"],
        start_override=start,
        end_override=end,
        reconnect_ts=end + pd.Timedelta(seconds=1),
    )
    assert len(gaps) == 1
    assert gaps[0]["gap_duration_seconds"] == pytest.approx(hours * 3600 - 0.001)
    assert len(gap.time_chunks(start, end)) == math.ceil(hours * 60 / 59)


def test_closed_candle_cutoff_and_persisted_watermark(isolated: Path) -> None:
    path = gap.LIVE_ROOT / "normalized/ws_kline_1m/symbol=BTCUSDT/date=2026-01-01/events.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"open_time": "2026-01-01T11:56:00Z", "close_time": "2026-01-01T11:56:59.999Z", "is_closed": True},
        {"open_time": "2026-01-01T11:57:00Z", "close_time": "2026-01-01T11:57:59.999Z", "is_closed": False},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    detected = gap.detect_gaps(
        reconnect_ts="2026-01-01T12:01:30Z",
        streams=["futures_1m_kline"],
    )
    assert detected[0]["gap_start_utc"] == pd.Timestamp("2026-01-01T11:57:00Z")
    assert detected[0]["gap_end_utc"] == pd.Timestamp("2026-01-01T12:00:00Z")


def test_provenance_tiers_and_live_normalization(isolated: Path) -> None:
    b = gap.provenance("REST_GAP_BACKFILL", "/fapi/v1/aggTrades", gap.FAPI, "2026-01-01T00:00:00Z", "g", "r")
    c = gap.provenance("RECONSTRUCTED_LOWER_RESOLUTION", "/futures/data/openInterestHist", gap.FAPI, "2026-01-01T00:00:00Z", "g", "r")
    live = collector.normalize_event(
        "aggTrade",
        {"e": "aggTrade", "E": 1767225600000, "T": 1767225600000, "s": "BTCUSDT", "a": 1, "p": "100", "q": "1", "m": False},
        pd.Timestamp("2026-01-01T00:00:01Z"),
    )
    assert gap.validate_provenance(b)
    assert gap.validate_provenance(c)
    assert b["data_quality_tier"] == "B" and not b["live_observed"]
    assert c["data_quality_tier"] == "C" and c["oi_resolution"] == "5m"
    assert live["collection_mode"] == "LIVE_WS"
    assert live["strict_forward_eval_eligible"] is True


def test_idempotent_backfill_second_insert_zero(isolated: Path) -> None:
    detected = gap.detect_gaps(
        streams=["futures_1m_kline"],
        start_override="2026-01-01T00:00:00Z",
        end_override="2026-01-01T00:29:59.999Z",
    )
    first = gap.backfill_one_gap(detected[0], fetcher=fake_public_get, run_id="r1")
    second = gap.backfill_one_gap(detected[0], fetcher=fake_public_get, run_id="r2")
    assert first["rows_inserted"] == 30
    assert first["backfill_status"] == "BACKFILL_COMPLETE"
    assert second["rows_inserted"] == 0
    assert second["backfill_status"] == "BACKFILL_COMPLETE"
    stored = pd.read_parquet(gap.stream_data_path("futures_1m_kline"))
    assert len(stored) == 30
    assert set(stored["collection_mode"]) == {"REST_GAP_BACKFILL"}


def test_duplicate_overlap_is_removed(isolated: Path) -> None:
    detected = gap.detect_gaps(
        streams=["futures_aggTrade"],
        start_override="2026-01-01T00:00:00Z",
        end_override="2026-01-01T00:02:59.999Z",
    )

    def duplicate_fetch(base: str, path: str, params: Dict[str, Any]) -> Any:
        if "fromId" in params:
            return []
        row = {"a": 7, "p": "100", "q": "1", "T": params["startTime"], "m": False}
        return [row, dict(row)]

    result = gap.backfill_one_gap(detected[0], fetcher=duplicate_fetch)
    assert result["rows_inserted"] == 1
    assert result["duplicates_removed"] == 1


def test_live_ws_overlap_wins_over_rest(isolated: Path) -> None:
    live = gap.LIVE_ROOT / "normalized/ws_aggTrade/symbol=BTCUSDT/date=2026-01-01/events.jsonl"
    live.parent.mkdir(parents=True)
    live.write_text(
        json.dumps({"symbol": "BTCUSDT", "agg_trade_id": 7, "event_ts": "2026-01-01T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    detected = gap.detect_gaps(
        streams=["futures_aggTrade"],
        start_override="2026-01-01T00:00:00Z",
        end_override="2026-01-01T00:02:59.999Z",
    )

    def overlap_fetch(base: str, path: str, params: Dict[str, Any]) -> Any:
        if "fromId" in params:
            return []
        return [{"a": 7, "p": "100", "q": "1", "T": params["startTime"], "m": False}]

    result = gap.backfill_one_gap(detected[0], fetcher=overlap_fetch)
    assert result["rows_inserted"] == 0
    assert result["duplicates_removed"] == 1


def test_partial_failure_persists_and_resumes(isolated: Path) -> None:
    detected = gap.detect_gaps(
        streams=["futures_1m_kline"],
        start_override="2026-01-01T00:00:00Z",
        end_override="2026-01-01T01:29:59.999Z",
    )
    calls = 0

    def fail_second(base: str, path: str, params: Dict[str, Any]) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return fake_public_get(base, path, params)

    failed = gap.backfill_one_gap(detected[0], fetcher=fail_second)
    assert failed["backfill_status"] == "BACKFILL_FAILED_RETRYABLE"
    partial = pd.read_parquet(gap.stream_data_path("futures_1m_kline"))
    assert len(partial) == 59
    resumed = gap.backfill_one_gap(detected[0], fetcher=fake_public_get)
    assert resumed["backfill_status"] == "BACKFILL_COMPLETE"
    assert len(pd.read_parquet(gap.stream_data_path("futures_1m_kline"))) == 90


def test_empty_response_is_unavailable(isolated: Path) -> None:
    detected = gap.detect_gaps(
        streams=["futures_aggTrade"],
        start_override="2026-01-01T00:00:00Z",
        end_override="2026-01-01T00:29:59.999Z",
    )
    result = gap.backfill_one_gap(detected[0], fetcher=lambda *_args, **_kwargs: [])
    assert result["backfill_status"] == "BACKFILL_UNAVAILABLE"
    assert result["missing_intervals"] == result["expected_intervals"]


def test_forceorder_creates_no_rows_and_ledger_tier_d(isolated: Path) -> None:
    detected = gap.detect_gaps(
        streams=["forceOrder"],
        start_override="2026-01-01T00:00:00Z",
        end_override="2026-01-01T04:29:59.999Z",
    )
    result = gap.run_backfill(detected, fetcher=fake_public_get)
    row = result["results"][0]
    assert row["backfill_status"] == "UNRECOVERABLE_STREAM_GAP"
    assert row["forceorder_complete"] is False
    assert not gap.stream_data_path("forceOrder").exists()


def test_oi_lower_resolution_and_strict_exclusion(isolated: Path) -> None:
    detected = gap.detect_gaps(
        streams=["open_interest"],
        start_override="2026-01-01T00:00:00Z",
        end_override="2026-01-01T00:29:59.999Z",
    )
    result = gap.backfill_one_gap(detected[0], fetcher=fake_public_get)
    stored = pd.read_parquet(gap.stream_data_path("open_interest"))
    assert result["backfill_status"] == "BACKFILL_COMPLETE"
    assert set(stored["oi_resolution"]) == {"5m"}
    assert not stored["strict_forward_eval_eligible"].any()
    assert stored["exclude_from_forward_eval"].all()
    assert set(stored["data_quality_tier"]) == {"C"}


def test_gap_dataset_is_physically_separate_from_live(isolated: Path) -> None:
    detected = gap.detect_gaps(
        streams=["futures_1m_kline"],
        start_override="2026-01-01T00:00:00Z",
        end_override="2026-01-01T00:04:59.999Z",
    )
    gap.backfill_one_gap(detected[0], fetcher=fake_public_get)
    assert gap.stream_data_path("futures_1m_kline").is_relative_to(gap.GAP_ROOT)
    assert not gap.stream_data_path("futures_1m_kline").is_relative_to(gap.LIVE_ROOT)
    assert not (gap.LIVE_ROOT / "normalized").exists()
