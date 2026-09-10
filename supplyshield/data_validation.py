"""Input validation and normalization for supplier data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = {
    "supplier_id",
    "supplier_name",
    "location",
    "avg_delay_days",
    "risk_score",
    "daily_revenue_exposure",
}

NUMERIC_COLUMNS = {
    "avg_delay_days",
    "risk_score",
    "on_time_delivery_rate",
    "quality_defect_rate",
    "daily_revenue_exposure",
    "expected_delay_days",
    "backup_capacity_pct",
}


@dataclass(frozen=True)
class ValidationResult:
    data: pd.DataFrame
    warnings: tuple[str, ...]


def validate_supplier_data(data: pd.DataFrame) -> ValidationResult:
    """Validate, coerce, and safely fill a supplier dataframe."""
    if data.empty:
        raise ValueError("Supplier data is empty.")

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    cleaned = data.copy()
    warnings: list[str] = []

    duplicate_count = int(cleaned.duplicated(subset=["supplier_id"]).sum())
    if duplicate_count:
        cleaned = cleaned.drop_duplicates(subset=["supplier_id"], keep="last")
        warnings.append(f"Removed {duplicate_count} duplicate supplier IDs.")

    for column in NUMERIC_COLUMNS.intersection(cleaned.columns):
        before = int(cleaned[column].isna().sum())
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        after = int(cleaned[column].isna().sum())
        if after > before:
            warnings.append(f"Converted {after - before} invalid values in {column} to missing.")

    defaults = {
        "on_time_delivery_rate": 0.85,
        "quality_defect_rate": 0.02,
        "expected_delay_days": 7.0,
        "backup_capacity_pct": 0.0,
    }
    for column, default in defaults.items():
        if column not in cleaned:
            cleaned[column] = default
            warnings.append(f"Added {column} using default value {default}.")

    for column in NUMERIC_COLUMNS.intersection(cleaned.columns):
        missing_count = int(cleaned[column].isna().sum())
        if missing_count:
            median = cleaned[column].median()
            fill = float(median) if pd.notna(median) else float(defaults.get(column, 0.0))
            cleaned[column] = cleaned[column].fillna(fill)
            warnings.append(f"Filled {missing_count} missing values in {column} with {fill:.2f}.")

    cleaned["risk_score"] = cleaned["risk_score"].clip(0, 100)
    cleaned["on_time_delivery_rate"] = cleaned["on_time_delivery_rate"].clip(0, 1)
    cleaned["quality_defect_rate"] = cleaned["quality_defect_rate"].clip(0, 1)
    cleaned["backup_capacity_pct"] = cleaned["backup_capacity_pct"].clip(0, 100)
    cleaned["daily_revenue_exposure"] = cleaned["daily_revenue_exposure"].clip(lower=0)
    cleaned["expected_delay_days"] = cleaned["expected_delay_days"].clip(lower=1)

    return ValidationResult(cleaned.reset_index(drop=True), tuple(warnings))
