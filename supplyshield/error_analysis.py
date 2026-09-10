"""Reusable detailed error analysis for classification and risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classification_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"actual", "predicted", "confidence"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")
    errors = predictions[predictions["actual"] != predictions["predicted"]].copy()
    errors["error_type"] = errors["actual"].astype(str) + " → " + errors["predicted"].astype(str)
    errors["confidence_band"] = pd.cut(
        errors["confidence"],
        bins=[-0.01, 0.5, 0.75, 1.0],
        labels=["low", "medium", "high"],
    )
    return errors.sort_values("confidence", ascending=False).reset_index(drop=True)


def risk_prediction_errors(
    predictions: pd.DataFrame, threshold: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"actual", "probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing risk columns: {', '.join(sorted(missing))}")

    detailed = predictions.copy()
    detailed["predicted"] = (detailed["probability"] >= threshold).astype(int)
    detailed["error_type"] = np.select(
        [
            (detailed["actual"] == 0) & (detailed["predicted"] == 1),
            (detailed["actual"] == 1) & (detailed["predicted"] == 0),
        ],
        ["False positive", "False negative"],
        default="Correct",
    )
    detailed["probability_band"] = pd.cut(
        detailed["probability"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
    )
    summary = (
        detailed.groupby(["probability_band", "error_type"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    errors = detailed[detailed["error_type"] != "Correct"].copy()
    errors["distance_from_threshold"] = (errors["probability"] - threshold).abs()
    errors = errors.sort_values("distance_from_threshold", ascending=False)
    return summary, errors.reset_index(drop=True)


def threshold_sweep(
    predictions: pd.DataFrame,
    *,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> pd.DataFrame:
    """Compare alert thresholds using an explicit missed-event cost assumption."""
    rows: list[dict] = []
    for threshold in np.arange(0.15, 0.86, 0.05):
        predicted = (predictions["probability"] >= threshold).astype(int)
        actual = predictions["actual"].astype(int)
        false_positives = int(((actual == 0) & (predicted == 1)).sum())
        false_negatives = int(((actual == 1) & (predicted == 0)).sum())
        true_positives = int(((actual == 1) & (predicted == 1)).sum())
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(int((actual == 1).sum()), 1)
        cost = false_negatives * false_negative_cost + false_positives * false_positive_cost
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision,
                "recall": recall,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "relative_decision_cost": cost,
            }
        )
    return pd.DataFrame(rows)
