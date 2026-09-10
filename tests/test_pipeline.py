import json
from pathlib import Path

import pandas as pd

from supplyshield.pipeline import run_risk_pipeline
from supplyshield.recommendations import build_executive_summary

ROOT = Path(__file__).resolve().parents[1]


def test_demo_pipeline_returns_ranked_results_and_evidence():
    suppliers = pd.read_csv(ROOT / "data" / "sample_suppliers.csv")
    news = json.loads((ROOT / "data" / "sample_news.json").read_text(encoding="utf-8"))
    result = run_risk_pipeline(suppliers, news)

    assert len(result.suppliers) == len(suppliers)
    assert not result.evidence.empty
    assert result.articles_processed < result.articles_received
    assert result.suppliers.iloc[0]["priority"] in {"Critical", "High"}
    assert result.suppliers["expected_loss"].ge(0).all()


def test_summary_names_highest_priority_supplier():
    frame = pd.DataFrame(
        [
            {
                "supplier_name": "Acme",
                "priority": "Critical",
                "expected_loss": 300_000,
                "top_event_type": "Port disruption",
                "recommended_action": "Activate backup",
            },
            {
                "supplier_name": "Beta",
                "priority": "Low",
                "expected_loss": 1_000,
                "top_event_type": "No event",
                "recommended_action": "Monitor",
            },
        ]
    )
    summary = build_executive_summary(frame)
    assert "Acme" in summary
    assert "$301,000" in summary
