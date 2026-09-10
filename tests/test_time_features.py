import pandas as pd

from supplyshield.time_features import add_time_aware_features


def history_frame():
    return pd.DataFrame(
        {
            "supplier_id": ["S1"] * 4,
            "observation_date": pd.date_range("2025-01-01", periods=4, freq="MS"),
            "avg_delay_days": [1.0, 2.0, 3.0, 100.0],
            "risk_score": [20.0, 30.0, 40.0, 99.0],
            "on_time_delivery_rate": [0.95, 0.90, 0.85, 0.10],
            "quality_defect_rate": [0.01, 0.02, 0.03, 0.50],
            "external_event_severity": [0.0, 0.0, 0.3, 1.0],
        }
    )


def test_rolling_features_exclude_current_row():
    featured = add_time_aware_features(history_frame())
    april = featured.iloc[-1]
    assert april["delay_lag_1m"] == 3.0
    assert april["delay_rolling_mean_3m"] == 2.0
    assert april["risk_rolling_mean_3m"] == 30.0


def test_changing_current_value_does_not_change_historical_aggregate():
    original = add_time_aware_features(history_frame())
    changed = history_frame()
    changed.loc[3, "avg_delay_days"] = 9999
    changed_featured = add_time_aware_features(changed)
    assert (
        original.loc[3, "delay_rolling_mean_3m"] == changed_featured.loc[3, "delay_rolling_mean_3m"]
    )
