"""End-to-end supplier risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_validation import validate_supplier_data
from .news_pipeline import (
    article_text,
    deduplicate_news,
    filter_recent_risk_news,
    rank_for_supplier,
)
from .risk_engine import calculate_supplier_risk
from .supplier_matching import article_matches_supplier
from .transformer_classifier import EventClassifier


@dataclass(frozen=True)
class PipelineResult:
    suppliers: pd.DataFrame
    evidence: pd.DataFrame
    warnings: tuple[str, ...]
    articles_received: int
    articles_processed: int


def run_risk_pipeline(
    supplier_data: pd.DataFrame,
    news_articles: list[dict],
    *,
    classifier: EventClassifier | None = None,
    news_days: int = 60,
    max_articles_per_supplier: int = 3,
) -> PipelineResult:
    validation = validate_supplier_data(supplier_data)
    suppliers = validation.data
    classifier = classifier or EventClassifier()

    filtered = filter_recent_risk_news(news_articles, days=news_days)
    unique_news = deduplicate_news(filtered)
    evidence_rows: list[dict] = []
    result_rows: list[dict] = []

    for _, supplier in suppliers.iterrows():
        ranked = rank_for_supplier(
            unique_news,
            str(supplier["supplier_name"]),
            str(supplier["location"]),
            limit=max_articles_per_supplier,
        )
        matched = [
            article
            for article in ranked
            if article_matches_supplier(
                article_text(article),
                str(supplier["supplier_name"]),
                str(supplier["location"]),
                str(supplier.get("aliases", "")),
            )
        ]

        classifications = []
        for article in matched:
            classification = classifier.classify(article_text(article))
            classifications.append((article, classification))
            evidence_rows.append(
                {
                    "supplier_id": supplier["supplier_id"],
                    "supplier_name": supplier["supplier_name"],
                    "title": article.get("title", ""),
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("published_at", ""),
                    "event_type": classification.event_type,
                    "event_confidence": classification.confidence,
                    "event_severity": classification.severity,
                    "relevance_score": article.get("relevance_score", 0.0),
                    "classifier_backend": classification.backend,
                }
            )

        if classifications:
            top_article, top_classification = max(
                classifications,
                key=lambda pair: pair[1].severity * pair[1].confidence,
            )
            event_type = top_classification.event_type
            event_title = top_article.get("title", "")
            event_url = top_article.get("url", "")
            confidence = top_classification.confidence
            severity = top_classification.severity
        else:
            event_type = "No relevant external event"
            event_title = ""
            event_url = ""
            confidence = 0.0
            severity = 0.0

        risk = calculate_supplier_risk(
            supplier,
            event_severity=severity,
            event_confidence=confidence,
        )
        result_rows.append(
            {
                **supplier.to_dict(),
                **risk,
                "top_event_type": event_type,
                "top_event_title": event_title,
                "top_event_url": event_url,
                "event_confidence": round(confidence, 3),
                "matched_articles": len(classifications),
            }
        )

    results = pd.DataFrame(result_rows)
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    results["_priority_order"] = results["priority"].map(priority_order)
    results = results.sort_values(
        ["_priority_order", "expected_loss"], ascending=[True, False]
    ).drop(columns="_priority_order")

    return PipelineResult(
        suppliers=results.reset_index(drop=True),
        evidence=pd.DataFrame(evidence_rows),
        warnings=validation.warnings,
        articles_received=len(news_articles),
        articles_processed=len(unique_news),
    )
