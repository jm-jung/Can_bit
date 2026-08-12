"""Parse local calendar files (JSONL/JSON/CSV) into CalendarEvent list."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .types import CalendarEvent

logger = logging.getLogger(__name__)


def _parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _norm_category(c: str) -> str:
    u = (c or "").strip().upper()
    if u in ("CPI", "FOMC", "RATE", "OTHER"):
        return u
    return "OTHER"


def ingest_jsonl(path: Path) -> List[CalendarEvent]:
    events: List[CalendarEvent] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                ts = raw.get("ts_utc") or raw.get("timestamp")
                if not ts:
                    continue
                if isinstance(ts, str):
                    ts_dt = _parse_ts(ts)
                else:
                    continue
                events.append(CalendarEvent(
                    id=raw.get("id"),
                    title=raw.get("title", ""),
                    category=_norm_category(raw.get("category", "OTHER")),
                    ts_utc=ts_dt,
                    impact=raw.get("impact"),
                    source=raw.get("source"),
                    meta=raw.get("meta"),
                ))
            except Exception as e:
                logger.warning("Skip line in %s: %s", path, e)
    return events


def ingest_json(path: Path) -> List[CalendarEvent]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    events: List[CalendarEvent] = []
    for raw in data:
        try:
            ts = raw.get("ts_utc") or raw.get("timestamp")
            if not ts:
                continue
            ts_dt = _parse_ts(ts) if isinstance(ts, str) else None
            if ts_dt is None:
                continue
            events.append(CalendarEvent(
                id=raw.get("id"),
                title=raw.get("title", ""),
                category=_norm_category(raw.get("category", "OTHER")),
                ts_utc=ts_dt,
                impact=raw.get("impact"),
                source=raw.get("source"),
                meta=raw.get("meta"),
            ))
        except Exception as e:
            logger.warning("Skip item in %s: %s", path, e)
    return events


def ingest_csv(path: Path) -> List[CalendarEvent]:
    events: List[CalendarEvent] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = row.get("ts_utc") or row.get("timestamp")
                if not ts:
                    continue
                ts_dt = _parse_ts(ts.strip())
                events.append(CalendarEvent(
                    id=row.get("id") or None,
                    title=row.get("title", ""),
                    category=_norm_category(row.get("category", "OTHER")),
                    ts_utc=ts_dt,
                    impact=float(row["impact"]) if row.get("impact") else None,
                    source=row.get("source"),
                    meta=None,
                ))
            except Exception as e:
                logger.warning("Skip row in %s: %s", path, e)
    return events


def ingest_directory(input_dir: Path) -> List[CalendarEvent]:
    all_events: List[CalendarEvent] = []
    input_dir = Path(input_dir)
    if not input_dir.exists():
        logger.warning("Input dir does not exist: %s", input_dir)
        return all_events
    for p in sorted(input_dir.iterdir()):
        if p.suffix == ".jsonl":
            all_events.extend(ingest_jsonl(p))
        elif p.suffix == ".json":
            all_events.extend(ingest_json(p))
        elif p.suffix == ".csv":
            all_events.extend(ingest_csv(p))
    all_events.sort(key=lambda e: e.ts_utc)
    return all_events


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ingest local calendar files into E0 storage")
    parser.add_argument("--input", type=str, default="data/events/calendar_raw", help="Input dir with .jsonl/.json/.csv")
    parser.add_argument("--out", type=str, default="data/events/calendar", help="Output dir for calendar_events.jsonl")
    parser.add_argument("--tz", type=str, default="UTC", help="Display timezone (ingest always UTC)")
    args = parser.parse_args()
    in_path = Path(args.input)
    out_path = Path(args.out)
    events = ingest_directory(in_path)
    if not events:
        logger.warning("No events loaded from %s", in_path)
        return
    from .storage import upsert_calendar_events
    upsert_calendar_events(events, base_dir=out_path)
    logger.info("Ingested %d events -> %s", len(events), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
