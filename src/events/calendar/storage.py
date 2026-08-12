"""E0 calendar event storage: load/save from data/events/calendar/ (JSONL)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .types import CalendarEvent

logger = logging.getLogger(__name__)

DEFAULT_CALENDAR_DIR = Path("data/events/calendar")
STORAGE_FILENAME = "calendar_events.jsonl"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _calendar_dir(base_dir: Path | None) -> Path:
    d = base_dir or DEFAULT_CALENDAR_DIR
    if not d.is_absolute():
        d = _project_root() / d
    return d


def _storage_path(base_dir: Path | None) -> Path:
    return _calendar_dir(base_dir) / STORAGE_FILENAME


def _parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def load_calendar_events(
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    base_dir: Path | None = None,
) -> List[CalendarEvent]:
    """
    Load calendar events from JSONL. Optional filter by [start_utc, end_utc].
    All timestamps normalized to tz-naive UTC.
    """
    path = _storage_path(base_dir)
    events: List[CalendarEvent] = []
    if not path.exists():
        logger.debug("Calendar storage not found: %s", path)
        return events

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                ts = raw.get("ts_utc")
                if isinstance(ts, str):
                    ts_dt = _parse_ts(ts)
                else:
                    continue
                if start_utc is not None and ts_dt < start_utc:
                    continue
                if end_utc is not None and ts_dt > end_utc:
                    continue
                ev = CalendarEvent(
                    id=raw.get("id"),
                    title=raw.get("title", ""),
                    category=raw.get("category", "OTHER"),
                    ts_utc=ts_dt,
                    impact=raw.get("impact"),
                    source=raw.get("source"),
                    meta=raw.get("meta"),
                )
                events.append(ev)
            except Exception as e:
                logger.warning("Skip invalid line in %s: %s", path, e)
                continue

    events.sort(key=lambda e: e.ts_utc)
    # De-duplicate by id when present
    seen_ids: set[str] = set()
    out: List[CalendarEvent] = []
    for e in events:
        if e.id is not None and e.id in seen_ids:
            continue
        if e.id is not None:
            seen_ids.add(e.id)
        out.append(e)
    return out


def upsert_calendar_events(
    events: List[CalendarEvent],
    base_dir: Path | None = None,
) -> None:
    """
    Merge events into storage (id-based de-dupe), then normalize and write.
    """
    path = _storage_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_calendar_events(base_dir=base_dir)
    by_id: dict[str | None, CalendarEvent] = {}
    for e in existing:
        by_id[e.id] = e
    for e in events:
        by_id[e.id] = e
    merged = list(by_id.values())
    merged.sort(key=lambda x: x.ts_utc)

    with open(path, "w", encoding="utf-8") as f:
        for e in merged:
            row = {
                "id": e.id,
                "title": e.title,
                "category": e.category,
                "ts_utc": e.ts_utc.isoformat() + "Z",
                "impact": e.impact,
                "source": e.source,
                "meta": e.meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Upserted %d calendar events to %s", len(merged), path)
