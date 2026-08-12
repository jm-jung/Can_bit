"""US equity session calendar helpers (XNYS / America/New_York)."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def get_calendar(name: str = "XNYS"):
    return xcals.get_calendar(name)


def now_ny() -> datetime:
    return datetime.now(tz=NY)


def latest_completed_session(cal=None, asof: Optional[datetime] = None) -> pd.Timestamp:
    """Return the latest fully completed XNYS regular session date (naive date as Timestamp)."""
    cal = cal or get_calendar("XNYS")
    ts = asof.astimezone(NY) if asof else now_ny()
    today = pd.Timestamp(ts.date())
    start = pd.Timestamp(cal.first_session)
    # Clamp end to calendar last session
    end = min(today, pd.Timestamp(cal.last_session))
    sessions = cal.sessions_in_range(start, end)
    if len(sessions) == 0:
        raise RuntimeError("No sessions found")
    last = sessions[-1]
    if pd.Timestamp(last).date() == ts.date():
        close_local = cal.session_close(last).tz_convert(NY)
        if ts < close_local.to_pydatetime():
            if len(sessions) < 2:
                raise RuntimeError("No completed session available")
            last = sessions[-2]
    return pd.Timestamp(pd.Timestamp(last).date())


def expected_sessions(start: str | pd.Timestamp, end: str | pd.Timestamp, cal=None) -> pd.DatetimeIndex:
    cal = cal or get_calendar("XNYS")
    return cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))


def is_early_close(session: pd.Timestamp, cal=None) -> bool:
    cal = cal or get_calendar("XNYS")
    close = cal.session_close(session).tz_convert(NY)
    return close.time() < time(16, 0)


def session_bounds_utc(session: pd.Timestamp, cal=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    cal = cal or get_calendar("XNYS")
    open_ = cal.session_open(session).tz_convert(UTC)
    close_ = cal.session_close(session).tz_convert(UTC)
    return open_, close_
