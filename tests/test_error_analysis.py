import pandas as pd

from supplyshield.error_analysis import (
    classification_errors,
    risk_prediction_errors,
    threshold_sweep,
)


def test_classification_errors_include_confidence_bands():
    data = pd.DataFrame(
        {
            "actual": ["weather", "strike"],
            "predicted": ["weather", "weather"],
            "confidence": [0.9, 0.8],
        }
    )
    errors = classification_errors(data)
    assert len(errors) == 1
    assert errors.iloc[0]["error_type"] == "strike → weather"
    assert errors.iloc[0]["confidence_band"] == "high"


def test_risk_errors_separate_false_positive_and_negative():
    data = pd.DataFrame({"actual": [0, 1, 1], "probability": [0.8, 0.2, 0.9]})
    _, errors = risk_prediction_errors(data)
    assert set(errors["error_type"]) == {"False positive", "False negative"}


def test_threshold_sweep_makes_cost_assumption_explicit():
    data = pd.DataFrame({"actual": [0, 0, 1, 1], "probability": [0.1, 0.7, 0.3, 0.9]})
    analysis = threshold_sweep(data, false_negative_cost=10, false_positive_cost=1)
    assert analysis["threshold"].is_monotonic_increasing
    assert {"precision", "recall", "relative_decision_cost"}.issubset(analysis.columns)
