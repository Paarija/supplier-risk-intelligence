from datetime import datetime, timezone

from supplyshield.news_pipeline import (
    deduplicate_news,
    filter_recent_risk_news,
    has_risk_signal,
    relevance_score,
)

NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)


def test_filter_removes_old_and_irrelevant_articles():
    articles = [
        {"title": "Port strike causes delay", "published_at": "2026-09-01T00:00:00+00:00"},
        {"title": "Football results", "published_at": "2026-09-01T00:00:00+00:00"},
        {"title": "Old factory shutdown", "published_at": "2020-01-01T00:00:00+00:00"},
    ]
    result = filter_recent_risk_news(articles, days=60, now=NOW)
    assert [item["title"] for item in result] == ["Port strike causes delay"]


def test_duplicate_headlines_are_removed():
    articles = [
        {"title": "Major port strike delays cargo in Mumbai"},
        {"title": "Major Mumbai port strike delays cargo"},
        {"title": "Flood closes factory in Houston"},
    ]
    result = deduplicate_news(articles, threshold=0.70)
    assert len(result) == 2


def test_supplier_and_location_raise_relevance():
    related = {"title": "Acme factory shutdown in Mumbai", "snippet": "Major delay"}
    unrelated = {"title": "Port delay in Hamburg", "snippet": "Cargo disruption"}
    assert relevance_score(related, "Acme", "Mumbai") > relevance_score(unrelated, "Acme", "Mumbai")


def test_negated_shortage_is_not_a_risk_signal():
    assert not has_risk_signal("Factories report normal operations and no material shortage.")
    assert has_risk_signal("Factories report a severe material shortage and delays.")
