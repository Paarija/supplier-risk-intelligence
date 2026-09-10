"""Run temporal risk-model comparison, calibration, errors, and optional SHAP."""

from __future__ import annotations

import argparse

from supplyshield.demo_data import generate_supplier_history
from supplyshield.risk_modeling import calculate_shap_importance, run_risk_model_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap", action="store_true", help="Calculate held-out SHAP values.")
    args = parser.parse_args()
    history = generate_supplier_history()
    result = run_risk_model_benchmark(history)
    print("Temporal split:")
    print(result.split_summary.to_string(index=False))
    print("\nHeld-out model metrics:")
    print(result.metrics.to_string(index=False))
    print("\nHighest-impact model errors:")
    print(result.errors.head(15).to_string(index=False))
    if args.shap:
        print("\nHeld-out SHAP importance:")
        print(calculate_shap_importance(result, history).to_string(index=False))


if __name__ == "__main__":
    main()
