"""이벤트 데이터 모델과 스키마 정의."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventCategory(str, Enum):
    """정규화된 이벤트 카테고리."""

    INFLUENCER = "INFLUENCER"
    INSTITUTION = "INSTITUTION"
    MACRO_POLICY = "MACRO_POLICY"
    REGULATION = "REGULATION"
    GEOPOLITICAL = "GEOPOLITICAL"
    MARKET_STRUCTURE = "MARKET_STRUCTURE"


class EventSourceType(str, Enum):
    """이벤트 소스 타입."""

    NEWS = "NEWS"
    X = "X"
    OTHER = "OTHER"


class RawEvent(BaseModel):
    """수집 직후의 원시 이벤트."""

    id: Optional[str] = None
    timestamp: datetime
    source: EventSourceType
    raw_text: str
    title: Optional[str] = None
    url: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class Event(RawEvent):
    """분류 및 감정 추정이 끝난 이벤트."""

    category: EventCategory
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    intensity: float = Field(..., ge=0.0, le=1.0)
    related_symbols: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class EventFeatureVector(BaseModel):
    """타임프레임별 집계된 이벤트 피처."""

    timestamp: datetime
    count_total: int = 0
    count_by_category: Dict[EventCategory, int]
    mean_sentiment: float = 0.0
    mean_positive_sentiment: float = 0.0
    mean_negative_sentiment: float = 0.0
    max_intensity: float = 0.0
    time_since_last_event_min: float = 0.0
    onehot_category_vector: Dict[EventCategory, float]


