"""Held-out TF-IDF baseline versus zero-shot transformer benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .error_analysis import classification_errors
from .transformer_classifier import EventClassifier

LABELS = [
    "Port or transport disruption",
    "Extreme weather",
    "Factory shutdown",
    "Material shortage",
    "Geopolitical disruption",
    "Financial distress",
    "Quality or recall",
    "No disruption",
]


@dataclass
class NewsBenchmarkResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    errors: pd.DataFrame
    baseline_model: Pipeline
    selected_c: float


def _text(data: pd.DataFrame) -> pd.Series:
    return data["title"].fillna("") + ". " + data["text"].fillna("")


def _baseline(c_value: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=5_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _metric_row(name: str, actual: pd.Series, predicted: list[str], seconds: float) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(actual, predicted, average="weighted", zero_division=0),
        "test_articles": len(actual),
        "latency_ms_per_article": seconds * 1_000 / max(len(actual), 1),
    }


def run_news_benchmark(
    dataset: pd.DataFrame,
    *,
    include_transformer: bool = False,
    transformer: EventClassifier | None = None,
) -> NewsBenchmarkResult:
    """Tune on validation, refit on train+validation, and evaluate once on test."""
    required = {"title", "text", "event_type", "split"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Missing news benchmark columns: {', '.join(sorted(missing))}")

    train = dataset[dataset["split"] == "train"]
    validation = dataset[dataset["split"] == "validation"]
    test = dataset[dataset["split"] == "test"].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Dataset must include non-empty train, validation, and test splits.")

    validation_scores: list[tuple[float, float]] = []
    for c_value in (0.25, 1.0, 4.0):
        candidate = _baseline(c_value)
        candidate.fit(_text(train), train["event_type"])
        predicted = candidate.predict(_text(validation))
        score = f1_score(validation["event_type"], predicted, average="macro", zero_division=0)
        validation_scores.append((score, c_value))
    selected_c = max(validation_scores, key=lambda item: (item[0], -item[1]))[1]

    development = pd.concat([train, validation], ignore_index=True)
    baseline = _baseline(selected_c)
    baseline.fit(_text(development), development["event_type"])
    started = perf_counter()
    baseline_prediction = baseline.predict(_text(test)).tolist()
    baseline_probability = baseline.predict_proba(_text(test))
    baseline_seconds = perf_counter() - started
    baseline_confidence = baseline_probability.max(axis=1)

    prediction_rows: list[dict] = []
    for (_, row), predicted, confidence in zip(
        test.iterrows(), baseline_prediction, baseline_confidence, strict=True
    ):
        prediction_rows.append(
            {
                "article_id": row.get("article_id", ""),
                "title": row["title"],
                "text": row["text"],
                "actual": row["event_type"],
                "predicted": predicted,
                "confidence": float(confidence),
                "model": "TF-IDF + logistic regression",
            }
        )

    metric_rows = [
        _metric_row(
            "TF-IDF + logistic regression",
            test["event_type"],
            baseline_prediction,
            baseline_seconds,
        )
    ]

    if include_transformer:
        transformer = transformer or EventClassifier(use_transformer=True)
        transformer_texts = [f"{row['title']}. {row['text']}" for _, row in test.iterrows()]
        started = perf_counter()
        transformer_outputs = transformer.classify_many(transformer_texts, batch_size=8)
        transformer_seconds = perf_counter() - started
        transformer_predictions = [output.event_type for output in transformer_outputs]
        transformer_confidence = [output.confidence for output in transformer_outputs]
        metric_rows.append(
            _metric_row(
                "Zero-shot transformer",
                test["event_type"],
                transformer_predictions,
                transformer_seconds,
            )
        )
        for (_, row), predicted, confidence in zip(
            test.iterrows(), transformer_predictions, transformer_confidence, strict=True
        ):
            prediction_rows.append(
                {
                    "article_id": row.get("article_id", ""),
                    "title": row["title"],
                    "text": row["text"],
                    "actual": row["event_type"],
                    "predicted": predicted,
                    "confidence": float(confidence),
                    "model": "Zero-shot transformer",
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    errors = classification_errors(predictions)
    return NewsBenchmarkResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=predictions,
        errors=errors,
        baseline_model=baseline,
        selected_c=selected_c,
    )
