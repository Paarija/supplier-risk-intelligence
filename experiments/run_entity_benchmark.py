"""Run validation-tuned supplier entity-resolution evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from supplyshield.entity_resolution import evaluate_entity_resolution

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    benchmark = pd.read_csv(ROOT / "data" / "entity_resolution_benchmark.csv")
    metrics, predictions = evaluate_entity_resolution(benchmark)
    print(metrics.to_string(index=False))
    errors = predictions[
        (predictions["split"] == "test")
        & (predictions["resolver_prediction"] != predictions["expected_match"])
    ]
    print("\nHeld-out resolver errors:")
    print(
        errors[
            [
                "case_id",
                "mention",
                "candidate_name",
                "context_location",
                "resolver_score",
                "expected_match",
                "challenge",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
