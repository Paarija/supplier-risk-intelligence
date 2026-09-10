"""Temporal logistic-regression versus XGBoost evaluation and calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .error_analysis import risk_prediction_errors, threshold_sweep
from .time_features import TIME_FEATURES, add_time_aware_features

TARGET = "disruption_next_30d"


@dataclass
class RiskBenchmarkResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    calibration: pd.DataFrame
    errors: pd.DataFrame
    error_summary: pd.DataFrame
    threshold_analysis: pd.DataFrame
    feature_importance: pd.DataFrame
    split_summary: pd.DataFrame
    error_model: str
    models: dict[str, object] = field(repr=False)


def temporal_split(
    data: pd.DataFrame, train_fraction: float = 0.60, calibration_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(data["observation_date"]).unique())
    if len(dates) < 10:
        raise ValueError("At least 10 observation dates are required for temporal evaluation.")
    train_end = max(1, int(len(dates) * train_fraction))
    calibration_end = max(train_end + 1, int(len(dates) * (train_fraction + calibration_fraction)))
    train_dates = set(dates[:train_end])
    calibration_dates = set(dates[train_end:calibration_end])
    test_dates = set(dates[calibration_end:])
    return (
        data[data["observation_date"].isin(train_dates)].copy(),
        data[data["observation_date"].isin(calibration_dates)].copy(),
        data[data["observation_date"].isin(test_dates)].copy(),
    )


def _logistic_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                TIME_FEATURES,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _xgboost_model():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost comparison requires: pip install xgboost") from exc
    return XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )


def _metrics(name: str, actual: pd.Series, probability: np.ndarray) -> dict:
    predicted = (probability >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(actual, probability),
        "pr_auc": average_precision_score(actual, probability),
        "brier_score": brier_score_loss(actual, probability),
        "log_loss": log_loss(actual, probability, labels=[0, 1]),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "test_rows": len(actual),
    }


def _curve(name: str, actual: pd.Series, probability: np.ndarray) -> list[dict]:
    observed, predicted = calibration_curve(actual, probability, n_bins=5, strategy="quantile")
    return [
        {"model": name, "mean_predicted_probability": x, "observed_rate": y}
        for x, y in zip(predicted, observed, strict=True)
    ]


def run_risk_model_benchmark(history: pd.DataFrame) -> RiskBenchmarkResult:
    """Run a chronological train/calibration/test comparison."""
    featured = add_time_aware_features(history)
    if TARGET not in featured:
        raise ValueError(f"History must include target column {TARGET}.")
    train, calibration_set, test = temporal_split(featured)
    for name, split in (("train", train), ("calibration", calibration_set), ("test", test)):
        if split[TARGET].nunique() < 2:
            raise ValueError(f"The {name} split must contain both target classes.")

    x_train, y_train = train[TIME_FEATURES], train[TARGET]
    x_cal, y_cal = calibration_set[TIME_FEATURES], calibration_set[TARGET]
    x_test, y_test = test[TIME_FEATURES], test[TARGET]

    logistic = _logistic_model().fit(x_train, y_train)
    xgboost = _xgboost_model().fit(x_train, y_train)
    logistic_calibrated = CalibratedClassifierCV(FrozenEstimator(logistic), method="sigmoid").fit(
        x_cal, y_cal
    )
    xgboost_calibrated = CalibratedClassifierCV(FrozenEstimator(xgboost), method="sigmoid").fit(
        x_cal, y_cal
    )

    models = {
        "Logistic regression": logistic,
        "Logistic regression + calibration": logistic_calibrated,
        "XGBoost": xgboost,
        "XGBoost + calibration": xgboost_calibrated,
    }
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    curve_rows: list[dict] = []
    for name, model in models.items():
        probability = model.predict_proba(x_test)[:, 1]
        metric_rows.append(_metrics(name, y_test, probability))
        curve_rows.extend(_curve(name, y_test, probability))
        for (_, row), actual, predicted_probability in zip(
            test.iterrows(), y_test, probability, strict=True
        ):
            prediction_rows.append(
                {
                    "model": name,
                    "supplier_id": row["supplier_id"],
                    "supplier_name": row.get("supplier_name", row["supplier_id"]),
                    "observation_date": row["observation_date"],
                    "actual": int(actual),
                    "probability": float(predicted_probability),
                }
            )

    feature_importance = pd.DataFrame(
        {
            "feature": TIME_FEATURES,
            "importance": xgboost.feature_importances_,
            "method": "XGBoost gain importance",
        }
    ).sort_values("importance", ascending=False)

    metrics = pd.DataFrame(metric_rows)
    calibrated_metrics = metrics[metrics["model"].str.contains("calibration")]
    error_model = str(calibrated_metrics.sort_values("brier_score").iloc[0]["model"])
    predictions = pd.DataFrame(prediction_rows)
    chosen = predictions[predictions["model"] == error_model]
    error_summary, errors = risk_prediction_errors(chosen)
    threshold_analysis = threshold_sweep(chosen)
    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "start": split["observation_date"].min(),
                "end": split["observation_date"].max(),
                "rows": len(split),
                "disruption_rate": split[TARGET].mean(),
            }
            for name, split in (("Train", train), ("Calibration", calibration_set), ("Test", test))
        ]
    )
    return RiskBenchmarkResult(
        metrics=metrics,
        predictions=predictions,
        calibration=pd.DataFrame(curve_rows),
        errors=errors,
        error_summary=error_summary,
        threshold_analysis=threshold_analysis,
        feature_importance=feature_importance.reset_index(drop=True),
        split_summary=split_summary,
        error_model=error_model,
        models=models,
    )


def calculate_shap_importance(
    benchmark: RiskBenchmarkResult, history: pd.DataFrame
) -> pd.DataFrame:
    """Calculate held-out SHAP importance for the uncalibrated XGBoost base model."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP explanations require: pip install shap") from exc
    featured = add_time_aware_features(history)
    _, _, test = temporal_split(featured)
    x_test = test[TIME_FEATURES]
    explainer = shap.TreeExplainer(benchmark.models["XGBoost"])
    values = explainer.shap_values(x_test)
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    return pd.DataFrame(
        {
            "feature": TIME_FEATURES,
            "mean_abs_shap": np.abs(array).mean(axis=0),
            "directional_mean_shap": array.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
