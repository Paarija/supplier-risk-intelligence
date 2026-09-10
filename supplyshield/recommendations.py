"""Evidence-backed deterministic management summaries."""

from __future__ import annotations

import pandas as pd


def build_executive_summary(results: pd.DataFrame) -> str:
    if results.empty:
        return "No suppliers were analyzed."
    urgent = results[results["priority"].isin(["Critical", "High"])]
    total_loss = float(results["expected_loss"].sum())
    if urgent.empty:
        return (
            f"No high-priority supplier risks were detected. Total modeled exposure is "
            f"${total_loss:,.0f}; continue routine monitoring."
        )
    top = urgent.sort_values("expected_loss", ascending=False).iloc[0]
    return (
        f"{len(urgent)} supplier(s) need action. Modeled exposure is ${total_loss:,.0f}. "
        f"The highest priority is {top['supplier_name']} because of "
        f"{top['top_event_type'].lower()}, with an estimated loss of "
        f"${top['expected_loss']:,.0f}. Recommended next step: "
        f"{top['recommended_action'].lower()}."
    )
