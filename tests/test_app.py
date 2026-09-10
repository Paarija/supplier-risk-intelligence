from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_demo_dashboard_runs_end_to_end():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    assert not app.exception

    app.button[0].click().run()

    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.dataframe) >= 2
    assert "Suppliers analyzed" in [metric.label for metric in app.metric]
