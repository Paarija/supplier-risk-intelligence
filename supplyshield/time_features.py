"""Leakage-safe, time-aware supplier feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME_FEATURES = [
    "avg_delay_days",
    "risk_score",
    "on_time_delivery_rate",
    "quality_defect_rate",
    "external_event_severity",
    "delay_lag_1m",
    "delay_rolling_mean_3m",
    "delay_rolling_std_3m",
    "delay_trend_3m",
    "on_time_rolling_mean_3m",
    "defect_rolling_mean_3m",
    "external_event_max_3m",
    "risk_rolling_mean_3m",
    "month_sin",
    "month_cos",
]


def add_time_aware_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build features available at each observation time.

    Rolling features use ``shift(1)`` so the current row and future rows cannot enter historical
    aggregates. Current operational columns remain available because they are assumed known at
    scoring time; the target is a disruption in the following 30 days.
    """
    required = {
        "supplier_id",
        "observation_date",
        "avg_delay_days",
        "risk_score",
        "on_time_delivery_rate",
        "quality_defect_rate",
        "external_event_severity",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Missing history columns: {', '.join(sorted(missing))}")

    data = history.copy()
    data["observation_date"] = pd.to_datetime(data["observation_date"], errors="raise")
    data = data.sort_values(["supplier_id", "observation_date"]).reset_index(drop=True)
    grouped = data.groupby("supplier_id", group_keys=False)

    data["delay_lag_1m"] = grouped["avg_delay_days"].shift(1)
    data["delay_lag_3m"] = grouped["avg_delay_days"].shift(3)
    data["delay_rolling_mean_3m"] = grouped["avg_delay_days"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).mean()
    )
    data["delay_rolling_std_3m"] = grouped["avg_delay_days"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).std()
    )
    data["delay_trend_3m"] = data["delay_lag_1m"] - data["delay_lag_3m"]
    data["on_time_rolling_mean_3m"] = grouped["on_time_delivery_rate"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).mean()
    )
    data["defect_rolling_mean_3m"] = grouped["quality_defect_rate"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).mean()
    )
    data["external_event_max_3m"] = grouped["external_event_severity"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).max()
    )
    data["risk_rolling_mean_3m"] = grouped["risk_score"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).mean()
    )

    month = data["observation_date"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)
    return data.drop(columns="delay_lag_3m")
