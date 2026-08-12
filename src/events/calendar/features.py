"""E0 calendar features aligned to OHLCV 5m index (UTC)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .types import CalendarEvent

logger = logging.getLogger(__name__)

MINUTES_CLIP_MAX = 1440
WINDOW_NEXT_60M = 60
WINDOW_LAST_60M = 60
WINDOW_6H = 6 * 60


def build_calendar_e0_features(
    index_ts_utc: pd.DatetimeIndex,
    events: list[CalendarEvent],
    include_optional: bool = True,
) -> pd.DataFrame:
    """
    Build E0 calendar features for each bar in index_ts_utc (tz-naive UTC).

    Required (3):
      - event_in_next_60m: 0/1
      - event_in_last_60m: 0/1
      - minutes_to_next_event: [0, 1440], no event -> 1440

    Optional if include_optional:
      - event_count_last_6h
      - event_score_decay_6h (exp decay)
    """
    n = len(index_ts_utc)
    out = pd.DataFrame(
        index=index_ts_utc,
        columns=[
            "event_in_next_60m",
            "event_in_last_60m",
            "minutes_to_next_event",
        ],
        dtype=np.float32,
    )
    out["event_in_next_60m"] = 0
    out["event_in_last_60m"] = 0
    out["minutes_to_next_event"] = float(MINUTES_CLIP_MAX)

    if include_optional:
        out["event_count_last_6h"] = 0.0
        out["event_score_decay_6h"] = 0.0

    if not events:
        logger.debug("No calendar events; E0 features set to default (next=1440, flags=0)")
        return out

    # Sort events by ts
    events_sorted = sorted(events, key=lambda e: e.ts_utc)
    event_times = np.array([e.ts_utc for e in events_sorted], dtype="datetime64[s]")
    index_n64 = index_ts_utc.astype("datetime64[s]").values

    for i in range(n):
        t = index_ts_utc[i]
        t_n64 = index_n64[i]

        # Next event: first event strictly after t
        next_mask = event_times > t_n64
        if next_mask.any():
            next_ev_ts = event_times[next_mask][0]
            delta = (pd.Timestamp(next_ev_ts) - pd.Timestamp(t_n64)).total_seconds() / 60.0
            minutes_next = float(np.clip(delta, 0, MINUTES_CLIP_MAX))
            out.loc[t, "minutes_to_next_event"] = minutes_next
            out.loc[t, "event_in_next_60m"] = 1.0 if 0 <= minutes_next <= WINDOW_NEXT_60M else 0.0
        else:
            out.loc[t, "minutes_to_next_event"] = float(MINUTES_CLIP_MAX)
            out.loc[t, "event_in_next_60m"] = 0.0

        # Last event: last event <= t
        last_mask = event_times <= t_n64
        if last_mask.any():
            last_ev_ts = event_times[last_mask][-1]
            delta = (pd.Timestamp(t_n64) - pd.Timestamp(last_ev_ts)).total_seconds() / 60.0
            out.loc[t, "event_in_last_60m"] = 1.0 if 0 <= delta <= WINDOW_LAST_60M else 0.0
        else:
            out.loc[t, "event_in_last_60m"] = 0.0

        # Optional: last 6h count and decay score
        if include_optional:
            window_start = t - timedelta(minutes=WINDOW_6H)
            count = 0
            decay_sum = 0.0
            for e in events_sorted:
                if window_start <= e.ts_utc <= t:
                    count += 1
                    mins_ago = (t - e.ts_utc).total_seconds() / 60.0
                    decay_sum += np.exp(-mins_ago / 60.0)  # half-life ~60 min
            out.loc[t, "event_count_last_6h"] = float(count)
            out.loc[t, "event_score_decay_6h"] = float(decay_sum)

    out = out.astype(np.float32)
    return out
