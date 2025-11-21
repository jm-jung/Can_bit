"""이벤트 텍스트 전처리 및 키워드 탐지."""
from __future__ import annotations

import re
from typing import Optional, Set

INFLUENCER_KEYWORDS = {
    "elon musk",
    "musk",
    "warren buffett",
    "buffett",
    "jack dorsey",
    "dorsey",
    "janet yellen",
    "yellen",
    "christine lagarde",
    "lagarde",
    "donald trump",
    "trump",
}

INSTITUTION_KEYWORDS = {
    "tesla",
    "microstrategy",
    "blackrock",
    "fidelity",
    "sec",
    "cme",
    "g20",
}

MACRO_KEYWORDS = {
    "fed",
    "fomc",
    "interest rate",
    "rate hike",
    "rate cut",
    "cpi",
    "inflation",
    "employment",
    "unemployment",
}

REGULATION_KEYWORDS = {
    "regulation",
    "regulatory",
    "ban",
    "clampdown",
    "legal",
    "law",
    "bill",
    "sec",
    "cftc",
}

GEOPOLITICAL_KEYWORDS = {
    "trade war",
    "sanction",
    "tariff",
    "geopolitical",
    "conflict",
    "g20",
    "china",
    "us",
    "russia",
    "ukraine",
}

MARKET_STRUCTURE_KEYWORDS = {
    "etf",
    "derivative",
    "exchange",
    "liquidity",
    "market depth",
    "halving",
    "supply",
}

BITCOIN_TERMS = {
    "bitcoin",
    "btc",
    "btc/usd",
    "btc-usd",
    "satoshi",
    "crypto",
    "cryptocurrency",
}

POSITIVE_WORDS = {
    "buy",
    "accumulate",
    "bullish",
    "support",
    "adopt",
    "approval",
    "greenlight",
    "innovation",
}

NEGATIVE_WORDS = {
    "sell",
    "dump",
    "ban",
    "restrict",
    "bearish",
    "downturn",
    "crackdown",
    "fraud",
    "risk",
}


def normalize_text(text: str) -> str:
    """간단한 소문자화 및 공백 정리."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_influencer(text: str) -> Optional[str]:
    """인물 키워드 탐지."""
    lowered = normalize_text(text)
    for keyword in INFLUENCER_KEYWORDS:
        if keyword in lowered:
            return keyword
    return None


def detect_keywords(text: str) -> Set[str]:
    """주요 키워드 집합 추출."""
    lowered = normalize_text(text)
    keywords: Set[str] = set()
    for group in (
        INFLUENCER_KEYWORDS,
        INSTITUTION_KEYWORDS,
        MACRO_KEYWORDS,
        REGULATION_KEYWORDS,
        GEOPOLITICAL_KEYWORDS,
        MARKET_STRUCTURE_KEYWORDS,
    ):
        for kw in group:
            if kw in lowered:
                keywords.add(kw)
    for term in BITCOIN_TERMS:
        if term in lowered:
            keywords.add(term)
    return keywords


def is_about_bitcoin(text: str) -> bool:
    """비트코인 관련 여부 판별."""
    lowered = normalize_text(text)
    return any(term in lowered for term in BITCOIN_TERMS)


