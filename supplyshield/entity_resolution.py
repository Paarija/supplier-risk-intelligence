"""Supplier entity resolution and benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .supplier_matching import name_similarity, normalize_name


@dataclass(frozen=True)
class EntityMatch:
    matched: bool
    score: float
    best_name: str


def entity_match_score(
    mention: str,
    candidate_name: str,
    aliases: str = "",
    candidate_location: str = "",
    context_location: str = "",
) -> EntityMatch:
    names = [candidate_name, *[item.strip() for item in aliases.split("|") if item.strip()]]
    scored_names = [(name_similarity(mention, name), name) for name in names]
    name_score, best_name = max(scored_names)

    candidate_location_tokens = set(normalize_name(candidate_location).split())
    context_location_tokens = set(normalize_name(context_location).split())
    location_score = 0.0
    if context_location_tokens:
        location_score = len(candidate_location_tokens & context_location_tokens) / len(
            context_location_tokens
        )

    score = 0.85 * name_score + 0.15 * location_score
    return EntityMatch(matched=False, score=round(score, 4), best_name=best_name)


def resolve_entity(row: pd.Series, threshold: float = 0.78) -> EntityMatch:
    result = entity_match_score(
        str(row["mention"]),
        str(row["candidate_name"]),
        str(row.get("aliases", "")),
        str(row.get("candidate_location", "")),
        str(row.get("context_location", "")),
    )
    return EntityMatch(result.score >= threshold, result.score, result.best_name)


def exact_match_baseline(mention: str, candidate_name: str) -> bool:
    return normalize_name(mention) == normalize_name(candidate_name)


def evaluate_entity_resolution(
    benchmark: pd.DataFrame, threshold: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune the resolver threshold on validation and report metrics on held-out test rows."""
    if threshold is None:
        validation = benchmark[benchmark["split"] == "validation"]
        if validation.empty:
            raise ValueError("Entity benchmark requires validation rows for threshold tuning.")
        candidates: list[tuple[float, float, float]] = []
        for candidate_threshold in (0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92):
            actual = validation["expected_match"].astype(int)
            predicted = validation.apply(
                lambda row, selected=candidate_threshold: int(
                    resolve_entity(row, threshold=selected).matched
                ),
                axis=1,
            )
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predicted, average="binary", zero_division=0
            )
            candidates.append((f1, precision, candidate_threshold))
        threshold = max(candidates, key=lambda item: (item[0], item[1], item[2]))[2]

    rows: list[dict] = []
    for _, row in benchmark.iterrows():
        resolved = resolve_entity(row, threshold=threshold)
        rows.append(
            {
                **row.to_dict(),
                "baseline_prediction": int(
                    exact_match_baseline(row["mention"], row["candidate_name"])
                ),
                "resolver_prediction": int(resolved.matched),
                "resolver_score": resolved.score,
                "best_alias": resolved.best_name,
            }
        )
    predictions = pd.DataFrame(rows)
    evaluation = predictions[predictions["split"] == "test"]
    if evaluation.empty:
        raise ValueError("Entity benchmark requires held-out test rows.")
    metrics: list[dict] = []
    for name, column in (
        ("Exact-name baseline", "baseline_prediction"),
        ("Alias + fuzzy + location", "resolver_prediction"),
    ):
        precision, recall, f1, _ = precision_recall_fscore_support(
            evaluation["expected_match"],
            evaluation[column],
            average="binary",
            zero_division=0,
        )
        metrics.append(
            {
                "method": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "errors": int((evaluation[column] != evaluation["expected_match"]).sum()),
                "test_rows": len(evaluation),
                "selected_threshold": threshold if column == "resolver_prediction" else None,
            }
        )
    return pd.DataFrame(metrics), predictions
