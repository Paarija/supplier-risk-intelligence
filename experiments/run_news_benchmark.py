"""Run the labeled news benchmark and print held-out results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from supplyshield.news_benchmark import run_news_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Also run the optional zero-shot transformer on the held-out test set.",
    )
    args = parser.parse_args()
    dataset = pd.read_csv(ROOT / "data" / "labeled_disruption_news.csv")
    result = run_news_benchmark(dataset, include_transformer=args.transformer)
    print(result.metrics.to_string(index=False))
    if result.errors.empty:
        print("\nNo held-out errors.")
    else:
        print("\nHighest-confidence errors:")
        print(
            result.errors[["model", "title", "actual", "predicted", "confidence"]]
            .head(15)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
