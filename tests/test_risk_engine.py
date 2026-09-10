import pandas as pd

from supplyshield.risk_engine import calculate_supplier_risk


def supplier_row():
    return pd.Series(
        {
            "avg_delay_days": 10,
            "risk_score": 70,
            "on_time_delivery_rate": 0.75,
            "quality_defect_rate": 0.04,
            "daily_revenue_exposure": 40_000,
            "expected_delay_days": 10,
            "backup_capacity_pct": 20,
        }
    )


def test_external_event_increases_probability_and_loss():
    baseline = calculate_supplier_risk(supplier_row())
    event = calculate_supplier_risk(supplier_row(), event_severity=0.9, event_confidence=0.9)
    assert event["disruption_probability"] > baseline["disruption_probability"]
    assert event["expected_loss"] > baseline["expected_loss"]


def test_backup_capacity_reduces_expected_loss():
    no_backup = calculate_supplier_risk(supplier_row())
    with_backup_row = supplier_row().copy()
    with_backup_row["backup_capacity_pct"] = 100
    with_backup = calculate_supplier_risk(with_backup_row)
    assert with_backup["expected_loss"] < no_backup["expected_loss"]
