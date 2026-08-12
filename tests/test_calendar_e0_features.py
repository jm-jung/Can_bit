"""E0 calendar feature builder 단위 테스트."""
import unittest
from datetime import datetime

import pandas as pd

from src.events.calendar.features import (
    MINUTES_CLIP_MAX,
    WINDOW_LAST_60M,
    WINDOW_NEXT_60M,
    build_calendar_e0_features,
)
from src.events.calendar.types import CalendarEvent


def _ts(y, m, d, h, minu=0):
    return datetime(y, m, d, h, minu)


class TestCalendarE0Features(unittest.TestCase):
    def test_event_in_next_60m_and_minutes_to_next(self):
        """이벤트 1개가 t+30분에 있을 때 event_in_next_60m=1, minutes_to_next_event≈30"""
        base = _ts(2026, 2, 15, 12, 0)
        index_ts = pd.DatetimeIndex([base, base + pd.Timedelta(minutes=5), base + pd.Timedelta(minutes=10)])
        event_at_1230 = CalendarEvent(
            id="e1", title="CPI", category="CPI", ts_utc=base + pd.Timedelta(minutes=30)
        )
        out = build_calendar_e0_features(index_ts, [event_at_1230], include_optional=False)
        self.assertEqual(out.loc[base, "event_in_next_60m"], 1.0)
        self.assertAlmostEqual(out.loc[base, "minutes_to_next_event"], 30.0, delta=0.1)
        self.assertEqual(out.loc[base, "event_in_last_60m"], 0.0)

    def test_event_in_last_60m(self):
        """이벤트가 t-30분에 있을 때 event_in_last_60m=1"""
        base = _ts(2026, 2, 15, 12, 0)
        bar_ts = base + pd.Timedelta(minutes=30)  # 12:30 bar
        index_ts = pd.DatetimeIndex([base, base + pd.Timedelta(minutes=30)])
        event_at_12 = CalendarEvent(id="e1", title="FOMC", category="FOMC", ts_utc=base)
        out = build_calendar_e0_features(index_ts, [event_at_12], include_optional=False)
        self.assertEqual(out.loc[bar_ts, "event_in_last_60m"], 1.0)
        self.assertEqual(out.loc[bar_ts, "event_in_next_60m"], 0.0)

    def test_no_events_defaults(self):
        """이벤트 없을 때 minutes_to_next_event=1440(clip), next/last flags=0"""
        base = _ts(2026, 2, 15, 12, 0)
        index_ts = pd.DatetimeIndex([base, base + pd.Timedelta(minutes=5)])
        out = build_calendar_e0_features(index_ts, [], include_optional=False)
        self.assertEqual(out["event_in_next_60m"].iloc[0], 0.0)
        self.assertEqual(out["event_in_last_60m"].iloc[0], 0.0)
        self.assertEqual(out["minutes_to_next_event"].iloc[0], float(MINUTES_CLIP_MAX))
        self.assertTrue((out["minutes_to_next_event"] == MINUTES_CLIP_MAX).all())

    def test_nearest_next_event_used(self):
        """여러 이벤트 있을 때 가장 가까운 다음 이벤트가 선택되는지"""
        base = _ts(2026, 2, 15, 12, 0)
        index_ts = pd.DatetimeIndex([base])
        near = CalendarEvent(id="near", title="A", category="CPI", ts_utc=base + pd.Timedelta(minutes=20))
        far = CalendarEvent(id="far", title="B", category="FOMC", ts_utc=base + pd.Timedelta(minutes=120))
        out = build_calendar_e0_features(index_ts, [far, near], include_optional=False)
        self.assertAlmostEqual(out.loc[base, "minutes_to_next_event"], 20.0, delta=0.1)
        self.assertEqual(out.loc[base, "event_in_next_60m"], 1.0)


if __name__ == "__main__":
    unittest.main()
