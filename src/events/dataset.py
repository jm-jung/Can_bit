"""이벤트 피처 데이터셋 관리."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from src.core.config import PROJECT_ROOT, settings
from src.events.aggregator import aggregate_events
from src.events.classifier import EventClassifier
from src.events.schemas import Event, RawEvent
from src.events.sources import BaseEventSource, load_default_sources

logger = logging.getLogger(__name__)


def _event_data_path(path: str | Path | None = None) -> Path:
    base = Path(path or settings.EVENTS_DATA_PATH)
    if not base.is_absolute():
        base = Path(PROJECT_ROOT) / base
    base.parent.mkdir(parents=True, exist_ok=True)
    return base


def _event_raw_dir(path: str | Path | None = None) -> Path:
    base = Path(path or settings.EVENTS_RAW_PATH)
    if not base.is_absolute():
        base = Path(PROJECT_ROOT) / base
    base.mkdir(parents=True, exist_ok=True)
    return base


def fetch_and_process_events(
    start: datetime,
    end: datetime,
    sources: Sequence[BaseEventSource] | None = None,
    classifier: EventClassifier | None = None,
) -> List[Event]:
    """소스에서 이벤트를 수집하고 분류."""
    # start, end를 timezone-aware로 보장
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)
    
    sources = list(sources or load_default_sources())
    classifier = classifier or EventClassifier()

    processed: List[Event] = []
    for source in sources:
        try:
            raw_events = source.fetch_events(start, end)
        except Exception as exc:
            logger.error("이벤트 소스 %s fetch 실패: %s", source.name, exc)
            continue

        for raw_event in raw_events:
            try:
                # raw_event.timestamp도 timezone-aware로 보장
                if raw_event.timestamp.tzinfo is None:
                    raw_event.timestamp = raw_event.timestamp.replace(tzinfo=timezone.utc)
                else:
                    raw_event.timestamp = raw_event.timestamp.astimezone(timezone.utc)
                
                processed_event = classifier.classify(raw_event)
            except Exception as exc:
                logger.warning("이벤트 분류 실패 (%s): %s", raw_event.id, exc)
                continue
            if processed_event is None:
                continue
            processed.append(processed_event)

    processed.sort(key=lambda ev: ev.timestamp)
    logger.info("총 %d개의 이벤트가 분류되었습니다.", len(processed))
    return processed


def build_event_feature_df(
    ohlcv_df: pd.DataFrame,
    *,
    lookback_minutes: int | None = None,
    sources: Sequence[BaseEventSource] | None = None,
    classifier: EventClassifier | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """OHLCV 타임라인에 맞춘 이벤트 피처 생성."""
    if "timestamp" not in ohlcv_df.columns:
        raise ValueError("OHLCV DataFrame에는 timestamp 컬럼이 필요합니다.")

    lookback_minutes = lookback_minutes or settings.EVENTS_LOOKBACK_MINUTES
    timestamps = pd.to_datetime(ohlcv_df["timestamp"])
    if timestamps.empty:
        logger.warning("OHLCV DataFrame이 비어있어 빈 피처를 반환합니다.")
        return pd.DataFrame()

    # timezone-aware datetime으로 변환
    start = timestamps.min().to_pydatetime()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)
    start = start - timedelta(minutes=lookback_minutes)
    
    end = timestamps.max().to_pydatetime()
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)

    events = fetch_and_process_events(
        start=start,
        end=end,
        sources=sources,
        classifier=classifier,
    )

    feature_df = aggregate_events(
        events=events,
        timeline=timestamps,
        lookback_minutes=lookback_minutes,
    )

    if save:
        if not feature_df.empty:
            save_event_features(feature_df)
        save_processed_events(events)

    return feature_df


def load_event_features(path: str | Path | None = None) -> pd.DataFrame:
    """저장된 이벤트 피처를 읽어온다."""
    data_path = _event_data_path(path)
    if not data_path.exists():
        logger.warning("이벤트 피처 파일이 없어 빈 DataFrame을 반환합니다: %s", data_path)
        return pd.DataFrame()
    try:
        df = pd.read_parquet(data_path, engine="pyarrow")
        df.index = pd.to_datetime(df.index)
        return df
    except ImportError as e:
        logger.error("pyarrow가 설치되지 않았습니다. pip install pyarrow를 실행하세요: %s", e)
        return pd.DataFrame()
    except Exception as e:
        logger.error("이벤트 피처 로드 실패 (%s): %s", data_path, e)
        return pd.DataFrame()


def save_event_features(df: pd.DataFrame, path: str | Path | None = None) -> None:
    """이벤트 피처를 parquet 파일로 저장."""
    if df.empty:
        logger.warning("빈 DataFrame이므로 저장하지 않습니다.")
        return
    
    data_path = _event_data_path(path)
    # 디렉토리가 없으면 생성 (이미 _event_data_path에서 처리하지만 안전을 위해)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_parquet(data_path, engine="pyarrow")
        logger.info("data/events/events.parquet 저장 완료 (shape=%s)", df.shape)
    except ImportError as e:
        logger.error("pyarrow가 설치되지 않았습니다. pip install pyarrow를 실행하세요: %s", e)
        raise
    except Exception as e:
        logger.error("이벤트 피처 저장 실패 (%s): %s", data_path, e)
        raise


def save_processed_events(
    events: Sequence[Event],
    path: str | Path | None = None,
    filename: str = "processed_events.jsonl",
) -> None:
    raw_dir = _event_raw_dir(path or settings.EVENTS_RAW_PATH)
    file_path = raw_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event.model_dump(), default=str, ensure_ascii=False))
            f.write("\n")
    logger.info("Processed 이벤트 %d건을 저장했습니다: %s", len(events), file_path)


def load_processed_events(
    path: str | Path | None = None,
    filename: str = "processed_events.jsonl",
    limit: int = 100,
) -> List[Event]:
    raw_dir = _event_raw_dir(path or settings.EVENTS_RAW_PATH)
    file_path = raw_dir / filename
    if not file_path.exists():
        logger.warning("저장된 이벤트 파일이 없어 빈 리스트를 반환합니다: %s", file_path)
        return []
    events: List[Event] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
                payload["timestamp"] = datetime.fromisoformat(
                    str(payload["timestamp"]).replace("Z", "+00:00")
                )
                events.append(Event(**payload))
            except Exception as exc:
                logger.error("저장 이벤트 파싱 실패: %s", exc)
    return events[-limit:]


def merge_price_and_event_features(
    feature_df: pd.DataFrame,
    event_feature_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """가격/인디케이터 피처와 이벤트 피처를 merge."""
    if event_feature_df is None or event_feature_df.empty:
        return feature_df

    merged = feature_df.copy()
    merged.index = pd.to_datetime(merged.index)
    aligned = event_feature_df.copy()
    aligned.index = pd.to_datetime(aligned.index)
    merged = merged.join(aligned, how="left")
    merged = merged.fillna(0.0)
    return merged


