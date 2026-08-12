"""Round-trip trade dedupe (engine.dedupe_trades_round_trips)."""
from __future__ import annotations

import logging

import pytest

from src.backtest.engine import Trade, dedupe_trades_round_trips


def _t(
    entry: str = "e",
    exit_: str = "x",
    direction: str = "LONG",
    profit: float = 0.01,
) -> Trade:
    return Trade(
        entry_time=entry,
        exit_time=exit_,
        entry_price=100.0,
        exit_price=101.0,
        direction=direction,  # type: ignore[arg-type]
        profit=profit,
    )


def test_dedupe_two_rows_same_key_keeps_first() -> None:
    a = _t(profit=0.02)
    b = _t(profit=0.01)
    out = dedupe_trades_round_trips([a, b])
    assert len(out) == 1
    assert out[0]["profit"] == 0.02


def test_dedupe_three_rows_same_key_warns_and_keeps_first(caplog: pytest.LogCaptureFixture) -> None:
    rows = [_t(profit=1.0), _t(profit=2.0), _t(profit=3.0)]
    with caplog.at_level(logging.WARNING):
        out = dedupe_trades_round_trips(rows)
    assert len(out) == 1
    assert out[0]["profit"] == 1.0
    assert any("3 rows" in rec.message for rec in caplog.records)


def test_dedupe_distinct_keys_untouched() -> None:
    r1 = _t(entry="a", exit_="b", profit=0.1)
    r2 = _t(entry="c", exit_="d", profit=-0.05)
    out = dedupe_trades_round_trips([r1, r2])
    assert len(out) == 2
