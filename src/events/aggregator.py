"""이벤트 → 타임프레임 피처 집계 로직."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

import numpy as np
import pandas as pd

from src.events.schemas import Event, EventCategory

logger = logging.getLogger(__name__)


def aggregate_events(
    events: List[Event],
    timeline: Iterable[datetime],
    lookback_minutes: int = 60,
) -> pd.DataFrame:
    """주어진 타임라인에 맞춰 이벤트 피처를 생성."""
    timestamps = pd.to_datetime(list(timeline))
    if timestamps.empty:
        logger.warning("타임라인이 비어있어 빈 DataFrame을 반환합니다.")
        return pd.DataFrame()

    # 🔥 DatetimeIndex는 .dt가 없다 → 바로 tz, tz_convert 사용
    if isinstance(timestamps, pd.DatetimeIndex) and timestamps.tz is not None:
        # UTC로 변환 후 tz 정보를 제거하여 tz-naive로 통일
        timestamps = timestamps.tz_convert("UTC").tz_localize(None)

    categories = list(EventCategory)
    if not events:
        logger.warning("집계 대상 이벤트가 없어 0으로 채운 피처를 반환합니다.")
        return _build_empty_features(timestamps, categories, lookback_minutes)

    # --------- 이벤트 timestamp를 tz-naive UTC로 통일 ----------
    event_rows: list[dict] = []
    for ev in events:
        ev_ts = ev.timestamp
        
        # timezone-aware로 강제 변환 (내부 기준은 UTC)
        if ev_ts.tzinfo is None:
            ev_ts = ev_ts.replace(tzinfo=timezone.utc)
        else:
            ev_ts = ev_ts.astimezone(timezone.utc)
        
        # tz 정보 제거 → tz-naive UTC
        ev_ts = ev_ts.replace(tzinfo=None)
        
        event_rows.append({
            "timestamp": pd.to_datetime(ev_ts),
            "category": ev.category,
            "sentiment": ev.sentiment_score,
            "intensity": ev.intensity,
        })
    
    event_df = pd.DataFrame(event_rows)
    if event_df.empty:
        logger.warning("이벤트 DataFrame이 비어있어 0으로 채운 피처를 반환합니다.")
        return _build_empty_features(timestamps, categories, lookback_minutes)
    
    event_df.sort_values("timestamp", inplace=True)

    # --------- 집계 ---------
    feature_rows: list[dict] = []
    last_event_ts: datetime | None = None
    window_delta = timedelta(minutes=lookback_minutes)

    for ts in timestamps:
        # ts도 tz-naive 상태이므로 window도 tz-naive
        window_start = ts - window_delta
        
        # tz-naive끼리 비교 → 에러 없음
        mask = (event_df["timestamp"] > window_start) & (event_df["timestamp"] <= ts)
        window_events = event_df.loc[mask]

        if not window_events.empty:
            # max()가 반환하는 Timestamp도 tz-naive이므로 그대로 사용
            last_event_ts = window_events["timestamp"].max().to_pydatetime()

        feature_rows.append(
            _compute_window_features(
                window_events,
                categories,
                ts,
                last_event_ts,
                lookback_minutes,
            )
        )

    feature_df = pd.DataFrame(feature_rows)
    if feature_df.empty:
        logger.warning("피처 DataFrame이 비어있어 0으로 채운 피처를 반환합니다.")
        return _build_empty_features(timestamps, categories, lookback_minutes)
    
    feature_df.set_index("timestamp", inplace=True)
    feature_df = feature_df.fillna(0.0)
    logger.debug(
        "이벤트 피처 집계 완료: shape=%s, 컬럼=%s",
        feature_df.shape,
        list(feature_df.columns),
    )
    return feature_df


def _compute_window_features(
    window_events: pd.DataFrame,
    categories: List[EventCategory],
    current_ts: datetime,
    last_event_ts: datetime | None,
    lookback_minutes: int,
) -> dict:
    feature: dict[str, float | datetime] = {"timestamp": current_ts}
    total = int(len(window_events))
    feature["event_count_total"] = float(total)

    for category in categories:
        col = f"event_count_{category.value.lower()}"
        if total == 0:
            feature[col] = 0.0
        else:
            feature[col] = float((window_events["category"] == category).sum())

    if total == 0:
        feature["event_sentiment_mean"] = 0.0
        feature["event_sentiment_positive_mean"] = 0.0
        feature["event_sentiment_negative_mean"] = 0.0
        feature["event_max_intensity"] = 0.0
    else:
        sentiments = window_events["sentiment"]
        feature["event_sentiment_mean"] = float(np.clip(sentiments.mean(), -1.0, 1.0))
        positive = sentiments[sentiments > 0]
        negative = sentiments[sentiments < 0]
        feature["event_sentiment_positive_mean"] = float(
            np.clip(positive.mean() if not positive.empty else 0.0, 0.0, 1.0)
        )
        feature["event_sentiment_negative_mean"] = float(
            np.clip(negative.mean() if not negative.empty else 0.0, -1.0, 0.0)
        )
        feature["event_max_intensity"] = float(
            window_events["intensity"].max(skipna=True)
        )

    # Time since last event
    if last_event_ts is None:
        feature["event_time_since_last_min"] = float(lookback_minutes)
    else:
        delta_min = (current_ts - last_event_ts).total_seconds() / 60
        feature["event_time_since_last_min"] = float(max(delta_min, 0.0))

    # One-hot vectors (비율)
    total_float = feature["event_count_total"] or 1.0
    for category in categories:
        count = feature[f"event_count_{category.value.lower()}"]
        feature[f"event_share_{category.value.lower()}"] = (
            float(count) / float(total_float) if total > 0 else 0.0
        )

    return feature


def _build_empty_features(
    timestamps: pd.DatetimeIndex,
    categories: List[EventCategory],
    lookback_minutes: int,
) -> pd.DataFrame:
    rows = []
    for ts in timestamps:
        base = {
            "timestamp": ts,
            "event_count_total": 0.0,
            "event_sentiment_mean": 0.0,
            "event_sentiment_positive_mean": 0.0,
            "event_sentiment_negative_mean": 0.0,
            "event_max_intensity": 0.0,
            "event_time_since_last_min": float(lookback_minutes),
        }
        for category in categories:
            base[f"event_count_{category.value.lower()}"] = 0.0
            base[f"event_share_{category.value.lower()}"] = 0.0
        rows.append(base)
    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    return df


