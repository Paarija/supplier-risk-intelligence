from supplyshield.demo_data import generate_supplier_history
from supplyshield.risk_modeling import run_risk_model_benchmark


def test_risk_models_use_chronological_splits_and_calibration():
    history = generate_supplier_history()
    result = run_risk_model_benchmark(history)
    assert list(result.split_summary["split"]) == ["Train", "Calibration", "Test"]
    assert result.split_summary.loc[0, "end"] < result.split_summary.loc[1, "start"]
    assert result.split_summary.loc[1, "end"] < result.split_summary.loc[2, "start"]
    assert set(result.metrics["model"]) == {
        "Logistic regression",
        "Logistic regression + calibration",
        "XGBoost",
        "XGBoost + calibration",
    }
    assert result.metrics["brier_score"].between(0, 1).all()
    assert not result.calibration.empty
    assert not result.feature_importance.empty
