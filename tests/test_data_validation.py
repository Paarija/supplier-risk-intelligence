import pandas as pd
import pytest

from supplyshield.data_validation import validate_supplier_data


def valid_frame():
    return pd.DataFrame(
        [
            {
                "supplier_id": "S1",
                "supplier_name": "Acme",
                "location": "Mumbai, India",
                "avg_delay_days": 4,
                "risk_score": 50,
                "daily_revenue_exposure": 1000,
            }
        ]
    )


def test_missing_required_column_is_rejected():
    frame = valid_frame().drop(columns="location")
    with pytest.raises(ValueError, match="location"):
        validate_supplier_data(frame)


def test_optional_defaults_are_added_and_bounded():
    result = validate_supplier_data(valid_frame())
    assert result.data.loc[0, "on_time_delivery_rate"] == 0.85
    assert result.data.loc[0, "expected_delay_days"] == 7.0
    assert result.warnings


def test_duplicate_supplier_ids_keep_latest_row():
    frame = pd.concat([valid_frame(), valid_frame().assign(risk_score=80)], ignore_index=True)
    result = validate_supplier_data(frame)
    assert len(result.data) == 1
    assert result.data.loc[0, "risk_score"] == 80
