from pathlib import Path

import pandas as pd

from supplyshield.entity_resolution import evaluate_entity_resolution

ROOT = Path(__file__).resolve().parents[1]


def test_entity_threshold_is_tuned_and_metrics_are_held_out():
    data = pd.read_csv(ROOT / "data" / "entity_resolution_benchmark.csv")
    metrics, predictions = evaluate_entity_resolution(data)
    resolver = metrics[metrics["method"] == "Alias + fuzzy + location"].iloc[0]
    assert resolver["test_rows"] == int((data["split"] == "test").sum())
    assert 0.68 <= resolver["selected_threshold"] <= 0.92
    assert 0 <= resolver["f1"] <= 1
    assert set(predictions["case_id"]) == set(data["case_id"])
