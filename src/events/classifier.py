"""규칙 기반 이벤트 분류기."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from src.events import parser
from src.events.schemas import Event, EventCategory, RawEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ClassificationConfig:
    """분류기 설정값."""

    influencer_weight: float = 0.6
    institution_weight: float = 0.5
    macro_weight: float = 0.4
    regulation_weight: float = 0.7
    geopolitical_weight: float = 0.5
    market_structure_weight: float = 0.3


class EventClassifier:
    """간단한 규칙 기반 이벤트 분류 및 감정 추정기."""

    def __init__(self, config: ClassificationConfig | None = None) -> None:
        self.config = config or ClassificationConfig()

    def classify(self, raw_event: RawEvent) -> Optional[Event]:
        """원시 이벤트를 Event 모델로 변환."""
        text = raw_event.raw_text or ""
        if not parser.is_about_bitcoin(text):
            logger.debug("비트코인 비관련 이벤트 필터링: %s", raw_event.id or raw_event.raw_text[:30])
            return None

        keywords = parser.detect_keywords(text)
        influencer = parser.detect_influencer(text)

        category = self._infer_category(keywords, influencer)
        sentiment_score = self._estimate_sentiment(text)
        intensity = self._estimate_intensity(category, sentiment_score, keywords)

        related_symbols = self._detect_related_symbols(keywords)

        return Event(
            **raw_event.model_dump(),
            category=category,
            sentiment_score=sentiment_score,
            intensity=intensity,
            related_symbols=related_symbols,
            keywords=sorted(keywords),
        )

    # ---- 내부 헬퍼 ---------------------------------------------------------
    def _infer_category(
        self,
        keywords: Iterable[str],
        influencer: Optional[str],
    ) -> EventCategory:
        lowered = {kw.lower() for kw in keywords}
        if influencer:
            return EventCategory.INFLUENCER

        if lowered & parser.INSTITUTION_KEYWORDS:
            return EventCategory.INSTITUTION
        if lowered & parser.MACRO_KEYWORDS:
            return EventCategory.MACRO_POLICY
        if lowered & parser.REGULATION_KEYWORDS:
            return EventCategory.REGULATION
        if lowered & parser.GEOPOLITICAL_KEYWORDS:
            return EventCategory.GEOPOLITICAL
        if lowered & parser.MARKET_STRUCTURE_KEYWORDS:
            return EventCategory.MARKET_STRUCTURE
        return EventCategory.MARKET_STRUCTURE

    def _estimate_sentiment(self, text: str) -> float:
        lowered = parser.normalize_text(text)
        score = 0.0
        for word in parser.POSITIVE_WORDS:
            if word in lowered:
                score += 0.2
        for word in parser.NEGATIVE_WORDS:
            if word in lowered:
                score -= 0.2
        # clamp
        if score > 1.0:
            score = 1.0
        if score < -1.0:
            score = -1.0
        return score

    def _estimate_intensity(
        self,
        category: EventCategory,
        sentiment_score: float,
        keywords: Iterable[str],
    ) -> float:
        base = 0.2 + min(abs(sentiment_score), 0.8) * 0.5
        kw_boost = min(len(list(keywords)) * 0.05, 0.3)
        category_weight = {
            EventCategory.INFLUENCER: self.config.influencer_weight,
            EventCategory.INSTITUTION: self.config.institution_weight,
            EventCategory.MACRO_POLICY: self.config.macro_weight,
            EventCategory.REGULATION: self.config.regulation_weight,
            EventCategory.GEOPOLITICAL: self.config.geopolitical_weight,
            EventCategory.MARKET_STRUCTURE: self.config.market_structure_weight,
        }[category]
        intensity = base + kw_boost + category_weight * 0.3
        return float(min(max(intensity, 0.05), 1.0))

    def _detect_related_symbols(self, keywords: Iterable[str]) -> List[str]:
        lowered = {kw.lower() for kw in keywords}
        symbols = {"BTC", "BTC-USD"}
        if "tesla" in lowered:
            symbols.add("TSLA")
        if "fed" in lowered or "fomc" in lowered:
            symbols.add("DXY")
        return sorted(symbols)


