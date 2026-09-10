"""Fast filtering, deduplication, ranking, and optional live news search."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterable

import requests

RISK_TERMS = {
    "strike",
    "shutdown",
    "closed",
    "closure",
    "flood",
    "storm",
    "cyclone",
    "earthquake",
    "war",
    "sanction",
    "shortage",
    "disruption",
    "delay",
    "congestion",
    "fire",
    "bankruptcy",
    "recall",
}

NEGATED_RISK_PATTERNS = (
    r"\bno (?:material )?shortage\b",
    r"\bno disruption\b",
    r"\bno delays?\b",
    r"\bnot affected\b",
    r"\bwithout disruption\b",
)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (value or "").lower()))


def article_text(article: dict) -> str:
    return " ".join(str(article.get(key, "")) for key in ("title", "snippet", "full_text")).strip()


def has_risk_signal(text: str) -> bool:
    """Detect risk language after removing common negated statements."""
    lowered = text.lower()
    for pattern in NEGATED_RISK_PATTERNS:
        lowered = re.sub(pattern, " ", lowered)
    return bool(_tokens(lowered) & RISK_TERMS)


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def filter_recent_risk_news(
    articles: Iterable[dict],
    *,
    days: int = 60,
    now: datetime | None = None,
) -> list[dict]:
    """Keep recent articles whose title/snippet contains a disruption term."""
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(days=days)
    selected: list[dict] = []
    for article in articles:
        published = parse_date(article.get("published_at"))
        if published and published < cutoff:
            continue
        if not has_risk_signal(article_text(article)):
            continue
        selected.append(dict(article))
    return selected


def deduplicate_news(articles: Iterable[dict], threshold: float = 0.72) -> list[dict]:
    """Remove near-duplicate headlines with inexpensive token similarity."""
    unique: list[dict] = []
    title_tokens: list[set[str]] = []
    for article in articles:
        candidate = _tokens(str(article.get("title", "")))
        duplicate = False
        for existing in title_tokens:
            union = candidate | existing
            similarity = len(candidate & existing) / len(union) if union else 1.0
            if similarity >= threshold:
                duplicate = True
                break
        if not duplicate:
            unique.append(dict(article))
            title_tokens.append(candidate)
    return unique


def relevance_score(article: dict, supplier_name: str, location: str) -> float:
    """Cheap first-pass relevance score before transformer inference."""
    text_tokens = _tokens(article_text(article))
    supplier_tokens = _tokens(supplier_name)
    location_tokens = _tokens(location)
    supplier_hit = len(text_tokens & supplier_tokens) / max(len(supplier_tokens), 1)
    location_hit = len(text_tokens & location_tokens) / max(len(location_tokens), 1)
    risk_hit = min(len(text_tokens & RISK_TERMS) / 2, 1.0)
    return round(0.50 * supplier_hit + 0.30 * location_hit + 0.20 * risk_hit, 4)


def rank_for_supplier(
    articles: Iterable[dict], supplier_name: str, location: str, limit: int = 3
) -> list[dict]:
    scored: list[dict] = []
    for article in articles:
        score = relevance_score(article, supplier_name, location)
        if score <= 0.2:
            continue
        item = dict(article)
        item["relevance_score"] = score
        scored.append(item)
    return sorted(scored, key=lambda item: item["relevance_score"], reverse=True)[:limit]


def search_serper(query: str, api_key: str, limit: int = 5) -> list[dict]:
    """Search recent news through Serper. No request is made without a key."""
    if not api_key:
        return []
    response = requests.post(
        "https://google.serper.dev/news",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": limit},
        timeout=15,
    )
    response.raise_for_status()
    records = []
    for result in response.json().get("news", [])[:limit]:
        records.append(
            {
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "url": result.get("link", ""),
                "source": result.get("source", "Web"),
                "published_at": result.get("date", ""),
                "full_text": result.get("snippet", ""),
            }
        )
    return records
