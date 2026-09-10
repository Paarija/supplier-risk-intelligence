"""Deterministic synthetic history for reproducible model demonstrations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def generate_supplier_history(
    *, seed: int = 42, supplier_count: int = 18, periods: int = 36
) -> pd.DataFrame:
    """Create supplier-month observations with a future-disruption label.

    The data is synthetic. Its purpose is to exercise temporal evaluation code, not to claim
    real-world predictive performance.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=periods, freq="MS")
    rows: list[dict] = []

    for supplier_index in range(supplier_count):
        supplier_id = f"HIST-{supplier_index + 1:03d}"
        base_delay = rng.uniform(1.5, 8.0)
        base_quality = rng.uniform(0.006, 0.045)
        base_risk = rng.uniform(22, 68)
        daily_exposure = rng.uniform(12_000, 65_000)
        backup_capacity = rng.uniform(5, 80)
        previous_delay = base_delay

        for month_index, observation_date in enumerate(dates):
            seasonal = 1.2 * math.sin(2 * math.pi * month_index / 12)
            external_event = float(rng.choice([0.0, 0.3, 0.6, 0.9], p=[0.70, 0.14, 0.10, 0.06]))
            shock = rng.normal(0, 1.1) + external_event * rng.uniform(3, 8)
            delay = max(0, 0.58 * previous_delay + 0.42 * base_delay + seasonal + shock)
            defect_rate = float(
                np.clip(base_quality + 0.0025 * delay + rng.normal(0, 0.006), 0, 0.18)
            )
            on_time_rate = float(np.clip(0.98 - delay / 32 + rng.normal(0, 0.025), 0.35, 0.99))
            risk_score = float(
                np.clip(base_risk + delay * 2.0 + external_event * 18 + rng.normal(0, 5), 0, 100)
            )
            latent = (
                -4.0
                + 0.16 * delay
                + 2.5 * defect_rate
                + 1.6 * (1 - on_time_rate)
                + 1.7 * external_event
                + 0.012 * risk_score
            )
            event_probability = 1 / (1 + math.exp(-latent))
            disruption = int(rng.random() < event_probability)
            rows.append(
                {
                    "supplier_id": supplier_id,
                    "supplier_name": f"Historical Supplier {supplier_index + 1:02d}",
                    "observation_date": observation_date,
                    "avg_delay_days": round(delay, 3),
                    "risk_score": round(risk_score, 3),
                    "on_time_delivery_rate": round(on_time_rate, 4),
                    "quality_defect_rate": round(defect_rate, 4),
                    "external_event_severity": external_event,
                    "daily_revenue_exposure": round(daily_exposure, 2),
                    "expected_delay_days": round(max(delay, 2), 2),
                    "backup_capacity_pct": round(backup_capacity, 2),
                    "disruption_next_30d": disruption,
                }
            )
            previous_delay = delay

    return (
        pd.DataFrame(rows).sort_values(["observation_date", "supplier_id"]).reset_index(drop=True)
    )
