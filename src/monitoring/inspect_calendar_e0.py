"""Inspect E0 calendar features: event count, 7d stats, const detection."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect E0 calendar features (recent N days)")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()

    # Load OHLCV 5m
    from src.services.ohlcv_service import load_ohlcv_df
    from src.events.calendar.storage import load_calendar_events
    from src.events.calendar.features import build_calendar_e0_features

    df = load_ohlcv_df(timeframe=args.timeframe, symbol=args.symbol)
    if df.empty:
        logger.error("No OHLCV data")
        return 1
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    cutoff = datetime.utcnow() - timedelta(days=args.days)
    df = df.loc[df["timestamp"] >= cutoff].copy()
    if df.empty:
        logger.error("No OHLCV in last %d days", args.days)
        return 1
    index_ts = df.set_index("timestamp").index
    if index_ts.tz is not None:
        index_ts = index_ts.tz_localize(None)
    start_utc = index_ts.min().to_pydatetime()
    end_utc = index_ts.max().to_pydatetime()
    # Buffer ±1 day for loading events
    start_load = start_utc - timedelta(days=1)
    end_load = end_utc + timedelta(days=1)
    events = load_calendar_events(start_utc=start_load, end_utc=end_load)
    events_in_range = [e for e in events if start_utc <= e.ts_utc <= end_utc]

    e0 = build_calendar_e0_features(index_ts, events, include_optional=True)

    # Report
    lines = [
        "=" * 60,
        "E0 Calendar Feature Inspection",
        "  days=%s, timeframe=%s, symbol=%s" % (args.days, args.timeframe, args.symbol),
        "=" * 60,
        "",
        "Loaded events (in range): %d" % len(events_in_range),
        "OHLCV index length (5m bars): %d" % len(e0),
        "",
        "event_in_next_60m: sum=%d, ratio=%.4f" % (e0["event_in_next_60m"].sum(), e0["event_in_next_60m"].mean()),
        "event_in_last_60m: sum=%d, ratio=%.4f" % (e0["event_in_last_60m"].sum(), e0["event_in_last_60m"].mean()),
        "",
        "minutes_to_next_event: mean=%.2f, median=%.2f, p10=%.2f, p90=%.2f" % (
            e0["minutes_to_next_event"].mean(),
            e0["minutes_to_next_event"].median(),
            e0["minutes_to_next_event"].quantile(0.1),
            e0["minutes_to_next_event"].quantile(0.9),
        ),
        "",
        "Optional:",
        "  event_count_last_6h: mean=%.4f" % e0["event_count_last_6h"].mean(),
        "  event_score_decay_6h: mean=%.4f" % e0["event_score_decay_6h"].mean(),
        "",
        "Const detection (unique count per column):",
    ]
    for col in e0.columns:
        n = e0[col].nunique()
        lines.append("  %s: unique=%d" % (col, n))
    lines.extend(["", "=" * 60])
    report = "\n".join(lines)
    logger.info(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
