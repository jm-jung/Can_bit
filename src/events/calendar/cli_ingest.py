"""CLI: ingest local calendar files -> data/events/calendar/."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .ingest_local import ingest_directory
from .storage import upsert_calendar_events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest calendar events from local files")
    parser.add_argument("--input", type=str, default="data/events/calendar_raw", help="Input dir")
    parser.add_argument("--out", type=str, default="data/events/calendar", help="Output dir")
    parser.add_argument("--tz", type=str, default="UTC", help="Timezone for display (ingest is UTC)")
    args = parser.parse_args()
    in_path = Path(args.input)
    out_path = Path(args.out)
    events = ingest_directory(in_path)
    if not events:
        logger.warning("No events loaded from %s", in_path)
        return 0
    upsert_calendar_events(events, base_dir=out_path)
    logger.info("Ingested %d events -> %s", len(events), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
