"""이벤트 → 타임프레임 피처 집계 로직."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

import numpy as np
import pandas as pd

from src.core.config import settings
from src.events.schemas import Event, EventCategory, EventSourceType
from src.events.store import load_raw_events

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


def build_event_feature_frame(
    ohlcv_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    lookback_minutes: int | None = None,
) -> pd.DataFrame:
    """
    Build event feature frame aligned to OHLCV DataFrame index.
    
    This is the v1 entry point for event feature generation. It:
    1. Loads raw events from storage (append-only file)
    2. Filters events within OHLCV time range (with lookback window)
    3. Aggregates events into 18 features aligned to OHLCV timestamps
    4. Returns DataFrame with same index as ohlcv_df
    
    Args:
        ohlcv_df: OHLCV DataFrame with timestamp index
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "1m")
        lookback_minutes: Lookback window for event aggregation (default: from settings)
    
    Returns:
        DataFrame with event features:
        - Index: same as ohlcv_df.index
        - Columns: 18 event features:
            - event_count_total
            - event_count_influencer
            - event_count_institution
            - event_count_macro_policy
            - event_count_regulation
            - event_count_geopolitical
            - event_count_market_structure
            - event_sentiment_mean
            - event_sentiment_positive_mean
            - event_sentiment_negative_mean
            - event_max_intensity
            - event_time_since_last_min
            - event_share_influencer
            - event_share_institution
            - event_share_macro_policy
            - event_share_regulation
            - event_share_geopolitical
            - event_share_market_structure
    
    If no raw events exist or OHLCV is empty, returns DataFrame with all zeros.
    """
    lookback_minutes = lookback_minutes or settings.EVENTS_LOOKBACK_MINUTES
    
    # Ensure ohlcv_df has timestamp index
    if "timestamp" in ohlcv_df.columns:
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"])
        ohlcv_df = ohlcv_df.set_index("timestamp").sort_index()
    elif not isinstance(ohlcv_df.index, pd.DatetimeIndex):
        raise ValueError("ohlcv_df must have timestamp index or 'timestamp' column")
    else:
        ohlcv_df = ohlcv_df.sort_index()
    
    if ohlcv_df.empty:
        logger.warning("OHLCV DataFrame is empty. Returning empty event features.")
        return pd.DataFrame(index=ohlcv_df.index)
    
    # Get time range from OHLCV
    time_start = ohlcv_df.index.min()
    time_end = ohlcv_df.index.max()
    
    # Load raw events
    raw_events_df = load_raw_events(symbol, timeframe)
    
    if raw_events_df.empty:
        logger.debug(
            f"[EventFeatures] No raw events found for symbol={symbol}, timeframe={timeframe}. "
            f"Returning zero-filled features for {len(ohlcv_df)} OHLCV rows."
        )
        categories = list(EventCategory)
        return _build_empty_features(ohlcv_df.index, categories, lookback_minutes)
    
    # Filter events within time range (with lookback window)
    # Include events from (time_start - lookback_minutes) to time_end
    window_start = time_start - timedelta(minutes=lookback_minutes)
    
    # Ensure raw_events_df has timestamp column
    if "timestamp" not in raw_events_df.columns:
        logger.warning("Raw events DataFrame missing 'timestamp' column. Returning zero-filled features.")
        categories = list(EventCategory)
        return _build_empty_features(ohlcv_df.index, categories, lookback_minutes)
    
    # Filter events in time range
    raw_events_df = raw_events_df.copy()
    raw_events_df["timestamp"] = pd.to_datetime(raw_events_df["timestamp"])
    if raw_events_df["timestamp"].dt.tz is not None:
        raw_events_df["timestamp"] = raw_events_df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    
    time_mask = (raw_events_df["timestamp"] >= window_start) & (raw_events_df["timestamp"] <= time_end)
    filtered_events_df = raw_events_df[time_mask].copy()
    
    if filtered_events_df.empty:
        logger.debug(
            f"[EventFeatures] No events in time range [{window_start}, {time_end}]. "
            f"Returning zero-filled features for {len(ohlcv_df)} OHLCV rows."
        )
        categories = list(EventCategory)
        return _build_empty_features(ohlcv_df.index, categories, lookback_minutes)
    
    # Convert raw events DataFrame to Event objects for aggregation
    # Note: We need category, sentiment_score, intensity from raw_events_df
    events_list: List[Event] = []
    for _, row in filtered_events_df.iterrows():
        try:
            # Map raw event row to Event object
            category_str = str(row.get("category", "")).upper()
            try:
                category = EventCategory(category_str)
            except ValueError:
                logger.warning(f"Unknown category '{category_str}'. Skipping event.")
                continue
            
            # Map source_type to EventSourceType
            source_type_str = str(row.get("source_type", "OTHER")).upper()
            try:
                source_type = EventSourceType(source_type_str)
            except ValueError:
                # Fallback to OTHER if unknown
                source_type = EventSourceType.OTHER
            
            event = Event(
                id=row.get("id"),
                timestamp=row["timestamp"],
                source=source_type,
                raw_text="",  # Not stored in raw events
                title=None,
                url=None,
                meta=None,
                category=category,
                sentiment_score=float(row.get("sentiment_score", 0.0)),
                intensity=float(row.get("intensity", 0.0)),
                related_symbols=[],
                keywords=[],
            )
            events_list.append(event)
        except Exception as e:
            logger.warning(f"Failed to convert raw event row to Event object: {e}")
            continue
    
    logger.info(
        f"[EventFeatures] symbol={symbol}, timeframe={timeframe}, "
        f"ohlcv_rows={len(ohlcv_df)}, raw_events={len(filtered_events_df)}, "
        f"processed_events={len(events_list)}"
    )
    
    # Aggregate events using existing aggregate_events function
    feature_df = aggregate_events(
        events=events_list,
        timeline=ohlcv_df.index,
        lookback_minutes=lookback_minutes,
    )
    
    # Ensure index matches ohlcv_df exactly
    if feature_df.empty:
        categories = list(EventCategory)
        feature_df = _build_empty_features(ohlcv_df.index, categories, lookback_minutes)
    else:
        # Reindex to match ohlcv_df.index exactly
        feature_df = feature_df.reindex(ohlcv_df.index, fill_value=0.0)
    
    # Ensure all 18 columns exist
    categories = list(EventCategory)
    expected_columns = [
        "event_count_total",
        "event_sentiment_mean",
        "event_sentiment_positive_mean",
        "event_sentiment_negative_mean",
        "event_max_intensity",
        "event_time_since_last_min",
    ]
    for category in categories:
        expected_columns.append(f"event_count_{category.value.lower()}")
        expected_columns.append(f"event_share_{category.value.lower()}")
    
    # Add missing columns with zeros
    for col in expected_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
    
    # Ensure column order matches expected order
    feature_df = feature_df[expected_columns]
    
    # Fill NaN with 0
    feature_df = feature_df.fillna(0.0)
    
    logger.debug(
        f"[EventFeatures] features_shape={feature_df.shape}, "
        f"columns={list(feature_df.columns)}"
    )
    
    return feature_df


