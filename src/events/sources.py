"""이벤트 소스 추상화 및 기본 구현."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

from src.core.config import PROJECT_ROOT, settings
from src.events.schemas import EventSourceType, RawEvent

logger = logging.getLogger(__name__)


class BaseEventSource(ABC):
    """이벤트 소스 공통 인터페이스."""

    source_type: EventSourceType
    name: str

    def __init__(self, *, timezone_offset: int = 0) -> None:
        self._tz = timezone(timedelta(hours=timezone_offset))

    @abstractmethod
    def fetch_events(self, start: datetime, end: datetime) -> List[RawEvent]:
        """주어진 기간의 이벤트를 반환."""

    def _coerce_timestamp(self, value: datetime | str) -> datetime:
        """timestamp를 timezone-aware (UTC)로 강제 변환."""
        if isinstance(value, datetime):
            ts = value
        else:
            # 문자열인 경우 ISO 형식으로 파싱
            ts_str = value.replace("Z", "+00:00")
            ts = datetime.fromisoformat(ts_str)
        
        # timezone-aware로 변환 (naive면 UTC로 간주)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            # 이미 timezone-aware면 UTC로 변환
            ts = ts.astimezone(timezone.utc)
        
        return ts


class DummyNewsSource(BaseEventSource):
    """로컬 샘플 데이터를 사용하는 더미 뉴스 소스."""

    name = "dummy_news"
    source_type = EventSourceType.NEWS

    def __init__(self, data_path: Path | None = None) -> None:
        super().__init__()
        if data_path is None:
            data_path = Path(PROJECT_ROOT) / "data" / "events" / "raw" / "dummy_news.json"
        self.data_path = data_path

    def fetch_events(self, start: datetime, end: datetime) -> List[RawEvent]:
        # start, end를 timezone-aware로 변환
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        else:
            start = start.astimezone(timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        else:
            end = end.astimezone(timezone.utc)
        
        events = self._load_fallback_events()
        filtered: List[RawEvent] = []
        for event in events:
            # event.timestamp도 timezone-aware로 보장
            ev_ts = event.timestamp
            if ev_ts.tzinfo is None:
                ev_ts = ev_ts.replace(tzinfo=timezone.utc)
            else:
                ev_ts = ev_ts.astimezone(timezone.utc)
            
            if start <= ev_ts <= end:
                filtered.append(event)
        logger.info(
            "[DummyNewsSource] 반환 이벤트 수=%d (기간: %s ~ %s)",
            len(filtered),
            start.isoformat(),
            end.isoformat(),
        )
        return filtered

    def _load_fallback_events(self) -> List[RawEvent]:
        if self.data_path.exists():
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return [
                    RawEvent(
                        id=item.get("id"),
                        timestamp=self._coerce_timestamp(item["timestamp"]),
                        source=self.source_type,
                        raw_text=item["raw_text"],
                        title=item.get("title"),
                        url=item.get("url"),
                        meta=item.get("meta"),
                    )
                    for item in payload
                ]
            except Exception as exc:
                logger.warning("샘플 이벤트 파일 로드 실패 (%s): %s", self.data_path, exc)

        # Fallback 하드코딩 이벤트 (timezone-aware로 보장)
        now = datetime.now(timezone.utc)
        fallback = [
            RawEvent(
                id="sample-news-1",
                timestamp=(now - timedelta(hours=36)).replace(tzinfo=timezone.utc),
                source=self.source_type,
                raw_text="Fed signals possible rate cut while Tesla considers fresh BTC purchase.",
                title="Fed rate outlook & Tesla BTC move",
                url=None,
            ),
            RawEvent(
                id="sample-news-2",
                timestamp=(now - timedelta(hours=20)).replace(tzinfo=timezone.utc),
                source=self.source_type,
                raw_text="SEC hints at stricter crypto regulation, sparking market volatility.",
                title="SEC regulation update",
            ),
            RawEvent(
                id="sample-news-3",
                timestamp=(now - timedelta(hours=5)).replace(tzinfo=timezone.utc),
                source=self.source_type,
                raw_text="Elon Musk tweets optimism about Bitcoin adoption in X payments.",
                title="Elon Musk on BTC",
            ),
        ]
        return fallback


class DummyXSource(BaseEventSource):
    """X(Twitter) 더미 데이터."""

    name = "dummy_x"
    source_type = EventSourceType.X

    def fetch_events(self, start: datetime, end: datetime) -> List[RawEvent]:
        # start, end를 timezone-aware로 변환
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        else:
            start = start.astimezone(timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        else:
            end = end.astimezone(timezone.utc)
        
        now = datetime.now(timezone.utc)
        samples = [
            RawEvent(
                id="dummy-x-1",
                timestamp=(now - timedelta(hours=12)).replace(tzinfo=timezone.utc),
                source=self.source_type,
                raw_text="Jack Dorsey: Lightning network upgrades push BTC mainstream adoption.",
            ),
            RawEvent(
                id="dummy-x-2",
                timestamp=(now - timedelta(hours=2)).replace(tzinfo=timezone.utc),
                source=self.source_type,
                raw_text="Rumors about G20 coordination on Bitcoin reserve requirements surge.",
            ),
        ]
        
        filtered: List[RawEvent] = []
        for ev in samples:
            ev_ts = ev.timestamp
            if ev_ts.tzinfo is None:
                ev_ts = ev_ts.replace(tzinfo=timezone.utc)
            else:
                ev_ts = ev_ts.astimezone(timezone.utc)
            if start <= ev_ts <= end:
                filtered.append(ev)
        
        logger.info(
            "[DummyXSource] 반환 이벤트 수=%d (기간: %s ~ %s)",
            len(filtered),
            start.isoformat(),
            end.isoformat(),
        )
        return filtered


class NewsApiSource(BaseEventSource):
    """실제 뉴스 API 연동용 골격."""

    name = "news_api"
    source_type = EventSourceType.NEWS

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self.api_key = api_key or settings.NEWS_API_KEY

    def fetch_events(self, start: datetime, end: datetime) -> List[RawEvent]:
        if not self.api_key:
            logger.warning("NEWS_API_KEY가 설정되지 않아 NewsApiSource는 더미 동작합니다.")
            return []
        # TODO: 실제 API 연동 구현
        logger.info(
            "NewsApiSource.fetch_events 호출 (start=%s, end=%s) - TODO 구현 필요",
            start,
            end,
        )
        return []


class XApiSource(BaseEventSource):
    """실제 X API 연동 골격."""

    name = "x_api"
    source_type = EventSourceType.X

    def __init__(self, bearer_token: str | None = None) -> None:
        super().__init__()
        self.bearer_token = bearer_token or settings.X_BEARER_TOKEN

    def fetch_events(self, start: datetime, end: datetime) -> List[RawEvent]:
        if not self.bearer_token:
            logger.warning("X_BEARER_TOKEN이 없어 XApiSource는 더미 동작합니다.")
            return []
        # TODO: 실제 API 연동 구현
        logger.info(
            "XApiSource.fetch_events 호출 (start=%s, end=%s) - TODO 구현 필요",
            start,
            end,
        )
        return []


def load_default_sources() -> List[BaseEventSource]:
    """기본 이벤트 소스 목록."""
    sources: List[BaseEventSource] = [DummyNewsSource(), DummyXSource()]
    if settings.NEWS_API_KEY:
        sources.append(NewsApiSource(settings.NEWS_API_KEY))
    if settings.X_BEARER_TOKEN:
        sources.append(XApiSource(settings.X_BEARER_TOKEN))
    return sources


