from pathlib import Path

import pandas as pd

from supplyshield.news_benchmark import LABELS, run_news_benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_labeled_news_dataset_is_balanced_and_has_fixed_splits():
    data = pd.read_csv(ROOT / "data" / "labeled_disruption_news.csv")
    assert len(data) == 64
    assert set(data["event_type"]) == set(LABELS)
    counts = data.groupby(["event_type", "split"]).size().unstack(fill_value=0)
    assert counts["train"].eq(5).all()
    assert counts["validation"].eq(1).all()
    assert counts["test"].eq(2).all()


def test_tfidf_baseline_uses_only_held_out_test_rows():
    data = pd.read_csv(ROOT / "data" / "labeled_disruption_news.csv")
    result = run_news_benchmark(data)
    assert result.metrics.loc[0, "test_articles"] == 16
    assert set(result.predictions["article_id"]) == set(
        data.loc[data["split"] == "test", "article_id"]
    )
    assert 0 <= result.metrics.loc[0, "macro_f1"] <= 1
