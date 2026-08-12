"""E0 calendar event types."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


class CalendarEventCategory:
    """E0 calendar categories. OTHER can be ignored for E0."""
    CPI = "CPI"
    FOMC = "FOMC"
    RATE = "RATE"
    OTHER = "OTHER"


@dataclass
class CalendarEvent:
    """Single calendar event (CPI, FOMC, rate decision, etc.)."""
    id: Optional[str]
    title: str
    category: str  # CPI | FOMC | RATE | OTHER
    ts_utc: datetime
    impact: Optional[float] = None
    source: Optional[str] = None
    meta: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.ts_utc.tzinfo is not None:
            self.ts_utc = self.ts_utc.astimezone(timezone.utc).replace(tzinfo=None)
