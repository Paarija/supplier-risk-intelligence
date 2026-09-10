"""Supplier Risk Intelligence Streamlit application."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from supplyshield.demo_data import generate_supplier_history
from supplyshield.entity_resolution import evaluate_entity_resolution
from supplyshield.news_benchmark import NewsBenchmarkResult, run_news_benchmark
from supplyshield.news_pipeline import search_serper
from supplyshield.pipeline import PipelineResult, run_risk_pipeline
from supplyshield.recommendations import build_executive_summary
from supplyshield.risk_modeling import (
    RiskBenchmarkResult,
    calculate_shap_importance,
    run_risk_model_benchmark,
)
from supplyshield.transformer_classifier import EventClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="Supplier Risk Intelligence", page_icon="🛡️", layout="wide")


@st.cache_data
def load_demo_suppliers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_suppliers.csv")


@st.cache_data
def load_demo_news() -> list[dict]:
    return json.loads((DATA_DIR / "sample_news.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading event classifier...")
def get_classifier(use_transformer: bool) -> EventClassifier:
    return EventClassifier(use_transformer=use_transformer)


@st.cache_resource(show_spinner="Running chronological model comparison...")
def run_demo_risk_benchmark() -> tuple[pd.DataFrame, RiskBenchmarkResult]:
    history = generate_supplier_history()
    return history, run_risk_model_benchmark(history)


@st.cache_resource(show_spinner="Running held-out news benchmark...")
def run_demo_news_benchmark(use_transformer: bool) -> NewsBenchmarkResult:
    dataset = pd.read_csv(DATA_DIR / "labeled_disruption_news.csv")
    return run_news_benchmark(dataset, include_transformer=use_transformer)


@st.cache_data
def run_entity_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = pd.read_csv(DATA_DIR / "entity_resolution_benchmark.csv")
    return evaluate_entity_resolution(benchmark)


def collect_live_news(suppliers: pd.DataFrame, api_key: str) -> list[dict]:
    articles: list[dict] = []
    for _, row in suppliers.head(10).iterrows():
        query = (
            f'"{row["supplier_name"]}" OR "{row["location"]}" '
            "supply chain disruption strike shutdown flood shortage"
        )
        articles.extend(search_serper(query, api_key, limit=5))
    return articles


def render_decision_dashboard(result: PipelineResult) -> None:
    suppliers = result.suppliers
    critical = int(suppliers["priority"].isin(["Critical", "High"]).sum())
    total_loss = float(suppliers["expected_loss"].sum())

    st.subheader("Executive risk snapshot")
    st.info(build_executive_summary(suppliers))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suppliers analyzed", len(suppliers))
    c2.metric("Need action", critical)
    c3.metric("Modeled exposure", f"${total_loss:,.0f}")
    c4.metric("News retained", f"{result.articles_processed}/{result.articles_received}")

    st.subheader("Procurement action center")
    display_columns = [
        "priority",
        "supplier_name",
        "location",
        "risk_score_calculated",
        "disruption_probability",
        "expected_loss",
        "top_event_type",
        "recommended_action",
    ]
    st.dataframe(
        suppliers[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "disruption_probability": st.column_config.ProgressColumn(
                "Disruption probability", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
            "expected_loss": st.column_config.NumberColumn("Expected loss", format="$%.0f"),
            "risk_score_calculated": st.column_config.ProgressColumn(
                "Risk score", min_value=0.0, max_value=100.0, format="%.1f"
            ),
        },
    )

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            suppliers,
            x="risk_score_calculated",
            y="expected_loss",
            color="priority",
            size="daily_revenue_exposure",
            hover_name="supplier_name",
            category_orders={"priority": ["Critical", "High", "Medium", "Low"]},
            title="Risk versus financial exposure",
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        top = suppliers.nlargest(min(8, len(suppliers)), "expected_loss")
        fig = px.bar(
            top,
            x="expected_loss",
            y="supplier_name",
            color="priority",
            orientation="h",
            title="Highest expected losses",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.subheader("Supplier evidence and review")
    selected_name = st.selectbox("Select supplier", suppliers["supplier_name"].tolist())
    selected = suppliers[suppliers["supplier_name"] == selected_name].iloc[0]
    st.markdown(
        f"**Recommended action:** {selected['recommended_action']}  \n"
        f"**Reason:** {selected['top_event_type']} combined with an internal risk score of "
        f"{selected['internal_risk']:.1f}/100."
    )
    supplier_evidence = result.evidence[result.evidence["supplier_name"] == selected_name]
    if supplier_evidence.empty:
        st.caption("No relevant external evidence was matched. The score uses internal data only.")
    else:
        st.dataframe(
            supplier_evidence[
                ["title", "source", "published_at", "event_type", "event_confidence", "url"]
            ],
            width="stretch",
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn("Evidence")},
        )

    decision_key = f"decision_{selected['supplier_id']}"
    d1, d2, d3 = st.columns(3)
    if d1.button("Approve recommendation", width="stretch"):
        st.session_state[decision_key] = "Approved"
    if d2.button("Needs review", width="stretch"):
        st.session_state[decision_key] = "Needs review"
    if d3.button("Reject recommendation", width="stretch"):
        st.session_state[decision_key] = "Rejected"
    if decision_key in st.session_state:
        st.success(f"Decision recorded for this session: {st.session_state[decision_key]}")

    st.download_button(
        "Download risk report",
        data=suppliers.to_csv(index=False).encode("utf-8"),
        file_name="supplier_risk_intelligence_report.csv",
        mime="text/csv",
    )


def render_model_lab(history: pd.DataFrame, benchmark: RiskBenchmarkResult) -> None:
    st.subheader("Temporal risk-model evaluation")
    st.caption(
        "Models train on earlier months, calibrate on later months, and are evaluated once "
        "on the newest held-out months. All records in this lab are synthetic."
    )
    st.dataframe(
        benchmark.split_summary,
        width="stretch",
        hide_index=True,
        column_config={
            "disruption_rate": st.column_config.NumberColumn("Disruption rate", format="percent")
        },
    )
    metric_columns = [
        "model",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "precision",
        "recall",
        "f1",
    ]
    st.dataframe(
        benchmark.metrics[metric_columns],
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(column.replace("_", " ").title(), format="%.3f")
            for column in metric_columns
            if column != "model"
        },
    )

    left, right = st.columns(2)
    with left:
        calibration = benchmark.calibration
        fig = px.line(
            calibration,
            x="mean_predicted_probability",
            y="observed_rate",
            color="model",
            markers=True,
            title="Probability calibration on held-out months",
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash"})
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(
            benchmark.feature_importance.head(10).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="XGBoost feature importance",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### SHAP explanations")
    st.caption(
        "SHAP values are calculated only on held-out test months and explain the uncalibrated "
        "XGBoost ranking model. Calibration changes probabilities, not the underlying ranking."
    )
    if st.button("Calculate held-out SHAP importance"):
        with st.spinner("Calculating SHAP values..."):
            st.session_state["shap_importance"] = calculate_shap_importance(benchmark, history)
    if "shap_importance" in st.session_state:
        shap_data = st.session_state["shap_importance"].head(12)
        fig = px.bar(
            shap_data.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Mean absolute SHAP value",
        )
        st.plotly_chart(fig, width="stretch")


def render_nlp_lab(benchmark: NewsBenchmarkResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.subheader("Disruption-news benchmark")
    st.caption(
        "The 64 labeled articles include eight balanced classes and hard negatives containing "
        "risk words such as ‘strike’, ‘shortage’, and ‘recall’. Results use only the test split."
    )
    st.dataframe(
        benchmark.metrics,
        width="stretch",
        hide_index=True,
        column_config={
            "accuracy": st.column_config.NumberColumn(format="%.3f"),
            "macro_f1": st.column_config.NumberColumn(format="%.3f"),
            "weighted_f1": st.column_config.NumberColumn(format="%.3f"),
            "latency_ms_per_article": st.column_config.NumberColumn(format="%.1f ms"),
        },
    )
    if len(benchmark.metrics) == 1:
        st.info(
            "Enable ‘Run transformer benchmark’ in the sidebar to compare the TF-IDF baseline "
            "with the optional zero-shot transformer."
        )

    st.markdown("#### Supplier entity-resolution evaluation")
    entity_metrics, entity_predictions = run_entity_benchmark()
    st.caption(
        "The matching threshold is selected on validation cases and reported on held-out test "
        "cases containing aliases, typos, shared tokens, and wrong-location traps."
    )
    st.dataframe(entity_metrics, width="stretch", hide_index=True)
    return entity_metrics, entity_predictions


def render_error_lab(
    risk_benchmark: RiskBenchmarkResult,
    news_benchmark: NewsBenchmarkResult,
    entity_predictions: pd.DataFrame,
) -> None:
    st.subheader("Detailed error analysis")
    st.markdown("#### Risk-model false positives and false negatives")
    st.caption(
        f"Errors below use {risk_benchmark.error_model}, selected because it had the lowest "
        "held-out Brier score among the calibrated models."
    )
    best_threshold = risk_benchmark.threshold_analysis.sort_values(
        ["relative_decision_cost", "threshold"]
    ).iloc[0]
    st.info(
        f"If a missed disruption costs five times more than a false alert, the lowest-cost "
        f"demo threshold is {best_threshold['threshold']:.0%}. This assumption must be replaced "
        "with the organization's actual investigation and disruption costs."
    )
    threshold_fig = px.line(
        risk_benchmark.threshold_analysis,
        x="threshold",
        y=["precision", "recall"],
        markers=True,
        title="Precision–recall tradeoff by decision threshold",
    )
    st.plotly_chart(threshold_fig, width="stretch")
    st.dataframe(
        risk_benchmark.error_summary,
        width="stretch",
        hide_index=True,
    )
    error_counts = risk_benchmark.errors["error_type"].value_counts().rename_axis("error_type")
    st.bar_chart(error_counts)
    st.dataframe(
        risk_benchmark.errors[
            [
                "supplier_name",
                "observation_date",
                "actual",
                "probability",
                "error_type",
                "probability_band",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "probability": st.column_config.NumberColumn(format="percent"),
        },
    )

    st.markdown("#### News-classification errors")
    if news_benchmark.errors.empty:
        st.success("No news-classification errors were found on the held-out set.")
    else:
        news_model = st.selectbox(
            "News model for confusion matrix",
            news_benchmark.predictions["model"].unique().tolist(),
        )
        selected_news_predictions = news_benchmark.predictions[
            news_benchmark.predictions["model"] == news_model
        ]
        confusion = pd.crosstab(
            selected_news_predictions["actual"],
            selected_news_predictions["predicted"],
        )
        confusion_fig = px.imshow(
            confusion,
            text_auto=True,
            aspect="auto",
            title="Held-out news confusion matrix",
            labels={"x": "Predicted", "y": "Actual", "color": "Articles"},
        )
        st.plotly_chart(confusion_fig, width="stretch")
        st.dataframe(
            news_benchmark.errors[
                ["model", "title", "actual", "predicted", "confidence", "confidence_band"]
            ],
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Entity-resolution errors")
    entity_errors = entity_predictions[
        (entity_predictions["split"] == "test")
        & (entity_predictions["resolver_prediction"] != entity_predictions["expected_match"])
    ]
    st.dataframe(
        entity_errors[
            [
                "mention",
                "candidate_name",
                "context_location",
                "expected_match",
                "resolver_prediction",
                "resolver_score",
                "challenge",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


st.title("🛡️ Supplier Risk Intelligence")
st.caption("Explainable supplier disruption intelligence for procurement teams")

with st.sidebar:
    st.header("Analysis settings")
    source_mode = st.radio("News source", ["Demo evidence", "Live Serper search"])
    serper_key = ""
    if source_mode == "Live Serper search":
        serper_key = st.text_input("Serper API key", type="password")
    use_transformer = st.toggle(
        "Use local transformer",
        value=False,
        help=(
            "Optional. Install requirements-ai.txt first. Demo mode uses a fast offline classifier."
        ),
    )
    run_transformer_benchmark = st.toggle(
        "Run transformer benchmark",
        value=False,
        help="Runs the zero-shot transformer over the 16 held-out benchmark articles.",
    )
    st.caption("Only the top three relevant articles per supplier reach the classifier.")

uploaded = st.file_uploader("Upload supplier CSV", type=["csv"])
supplier_df = pd.read_csv(uploaded) if uploaded else load_demo_suppliers()
st.caption(
    f"Using {'uploaded data' if uploaded else 'included demo data'}: {len(supplier_df)} suppliers"
)

auto_demo = st.query_params.get("demo") == "1" and "pipeline_result" not in st.session_state
if st.button("Run risk analysis", type="primary") or auto_demo:
    try:
        with st.spinner("Filtering evidence and calculating supplier risk..."):
            if source_mode == "Live Serper search":
                if not serper_key:
                    st.error("Enter a Serper API key or choose Demo evidence.")
                    st.stop()
                news = collect_live_news(supplier_df, serper_key)
            else:
                news = load_demo_news()
            result = run_risk_pipeline(
                supplier_df,
                news,
                classifier=get_classifier(use_transformer),
            )
            st.session_state["pipeline_result"] = result
    except Exception as exc:
        st.exception(exc)

if "pipeline_result" in st.session_state:
    current_result = st.session_state["pipeline_result"]
    for warning in current_result.warnings:
        st.warning(warning)
    history, risk_benchmark = run_demo_risk_benchmark()
    try:
        news_benchmark = run_demo_news_benchmark(run_transformer_benchmark)
    except RuntimeError as exc:
        st.warning(f"Transformer benchmark unavailable: {exc}")
        news_benchmark = run_demo_news_benchmark(False)
    decision_tab, model_tab, nlp_tab, error_tab = st.tabs(
        ["Decision dashboard", "Model evaluation", "NLP evaluation", "Error analysis"]
    )
    with decision_tab:
        render_decision_dashboard(current_result)
    with model_tab:
        render_model_lab(history, risk_benchmark)
    with nlp_tab:
        _, entity_predictions = render_nlp_lab(news_benchmark)
    with error_tab:
        render_error_lab(risk_benchmark, news_benchmark, entity_predictions)
else:
    st.info("Run the included demo to see prioritized supplier alerts without an API key.")
