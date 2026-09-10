"""Explainable supplier risk and financial exposure calculations."""

from __future__ import annotations

import math

import pandas as pd


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def calculate_internal_risk(row: pd.Series) -> float:
    delay = _clip(row["avg_delay_days"] / 20)
    provided_risk = _clip(row["risk_score"] / 100)
    delivery_failure = 1 - _clip(row.get("on_time_delivery_rate", 0.85))
    quality = _clip(row.get("quality_defect_rate", 0.02) / 0.10)
    return _clip(0.35 * delay + 0.30 * provided_risk + 0.20 * delivery_failure + 0.15 * quality)


def calculate_supplier_risk(
    row: pd.Series,
    *,
    event_severity: float = 0.0,
    event_confidence: float = 0.0,
) -> dict:
    internal = calculate_internal_risk(row)
    external = _clip(event_severity) * _clip(event_confidence)
    combined = _clip(0.62 * internal + 0.38 * external)
    probability = 1 / (1 + math.exp(-5 * (combined - 0.48)))

    expected_delay = max(float(row.get("expected_delay_days", 7)), 1)
    daily_exposure = max(float(row.get("daily_revenue_exposure", 0)), 0)
    backup_reduction = _clip(float(row.get("backup_capacity_pct", 0)) / 100) * 0.55
    expected_loss = probability * daily_exposure * expected_delay * (1 - backup_reduction)

    if expected_loss >= 250_000 or probability >= 0.78:
        action = "Escalate and activate backup"
        priority = "Critical"
    elif expected_loss >= 100_000 or probability >= 0.62:
        action = "Contact supplier and prepare backup"
        priority = "High"
    elif expected_loss >= 25_000 or probability >= 0.45:
        action = "Monitor closely"
        priority = "Medium"
    else:
        action = "Continue routine monitoring"
        priority = "Low"

    return {
        "internal_risk": round(internal * 100, 1),
        "external_risk": round(external * 100, 1),
        "risk_score_calculated": round(combined * 100, 1),
        "disruption_probability": round(probability, 4),
        "expected_loss": round(expected_loss, 2),
        "priority": priority,
        "recommended_action": action,
    }
