# Evaluation report

This report records the reproducible demo results generated with Python 3.13,
scikit-learn 1.7.2, XGBoost 3.1.2, and SHAP 0.52.0. Exact numbers can move slightly across
library versions. Both datasets are synthetic, so these values demonstrate evaluation
discipline rather than real-world model quality.

## Disruption-news classification

The labeled dataset contains 64 articles across eight balanced classes. Each class has five
training, one validation, and two held-out test articles. The validation split selects the
logistic-regression regularization value; test labels are not used for tuning.

| Model | Test articles | Accuracy | Macro F1 | Approx. latency/article |
|---|---:|---:|---:|---:|
| TF-IDF word/character n-grams + logistic regression | 16 | 0.500 | 0.467 | 1–2 ms |
| DeBERTa-v3-xsmall zero-shot transformer | 16 | **0.750** | **0.729** | ~6.8 s |

Run the measured transformer comparison locally with:

```bash
pip install -r requirements-ai.txt
python -m experiments.run_news_benchmark --transformer
```

The experiment uses the same held-out rows for both models and records accuracy, macro F1,
weighted F1, confidence, and latency per article. The measured transformer is
`MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, run on CPU with batches of eight. Batching
reduced observed latency from roughly 13.5 seconds to 6.8 seconds per article on the development
machine. Latency is hardware- and version-dependent.

### News error analysis

The TF-IDF model struggles with precisely the examples included to make the benchmark useful:

- Negation and scope: “denies bankruptcy” and “consumer division only.”
- Semantically related classes: material shortage versus quality/recall.
- Unseen event wording: barge collision and airport cargo stoppage.
- Cause versus consequence: landslide versus transport interruption.

The zero-shot transformer handles more unseen wording but still confuses events whose consequence
is a transport delay with their underlying cause. Its four held-out errors include geopolitical,
quality, and financial events predicted as transport disruption. This suggests that adding a
second-stage cause-versus-impact classifier may be more valuable than simply increasing model size.

The model's maximum probabilities are low, which is another sign that 48 development examples
are insufficient for a production eight-class classifier. The next data milestone should be at
least several hundred independently annotated, event-clustered articles per major class.

## Supplier entity resolution

The threshold is chosen on validation cases and metrics are reported on 15 held-out test cases.

| Method | Precision | Recall | F1 | Test errors |
|---|---:|---:|---:|---:|
| Exact normalized name | 0.500 | 0.250 | 0.333 | 4 |
| Alias + fuzzy name + location | 1.000 | 0.500 | 0.667 | 2 |

The selected threshold is `0.92`. The resolver misses “Nile Industrial Group” and “Meridian
Plastics Asia.” This exposes the precision/recall tradeoff: reducing the threshold recovers
regional and group suffixes but increases false matches for exact names appearing with the wrong
location. In production, company identifiers and a reviewed alias registry should supplement
string similarity.

## Future supplier-disruption risk

The generated history contains 18 suppliers over 36 months. The target is whether a disruption
occurs in the next 30 days. Time-aware rolling features call `shift(1)` before aggregation, so
the current or future row cannot enter its own historical windows.

Chronological evaluation:

| Split | Period | Rows | Disruption rate |
|---|---|---:|---:|
| Train | Jan 2023–Sep 2024 | 378 | 27.2% |
| Calibration | Oct 2024–Apr 2025 | 126 | 27.8% |
| Test | May 2025–Dec 2025 | 144 | 20.8% |

Held-out results:

| Model | ROC-AUC | PR-AUC | Brier | Log loss | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.730 | 0.483 | 0.175 | 0.533 | 0.421 | 0.533 | 0.471 |
| Logistic + sigmoid calibration | 0.730 | 0.483 | **0.144** | **0.462** | 0.600 | 0.300 | 0.400 |
| XGBoost | 0.724 | 0.443 | 0.154 | 0.484 | 0.435 | 0.333 | 0.377 |
| XGBoost + sigmoid calibration | 0.724 | 0.443 | 0.151 | 0.471 | 0.409 | 0.300 | 0.346 |

The simpler logistic model ranks risk slightly better than XGBoost on this synthetic dataset.
Calibration improves logistic Brier score and log loss, but at a fixed `0.5` threshold recall
drops from `0.533` to `0.300`. A real procurement team should choose the alert threshold using
the relative cost of missed disruptions and unnecessary investigations—not default to 0.5.

## SHAP explanations

SHAP values are calculated on held-out rows for the uncalibrated XGBoost base model. The first
run ranked on-time delivery rate, average delay, one-month delay lag, internal risk score, and
external-event severity among the most influential features.

SHAP explains the ranking model. The separate sigmoid calibrator changes probability mapping,
so its effect should be shown through the reliability plot rather than attributed to the SHAP
values.

## What is still required before production

- Replace synthetic data with legally sourced and independently annotated records.
- Group articles by real-world event before splitting to prevent story-level leakage.
- Validate supplier matches against stable company identifiers and supply-route data.
- Backtest over multiple temporal windows, regions, and supplier categories.
- Select decision thresholds using financial cost and operational capacity.
- Monitor data drift, calibration drift, subgroup errors, and analyst overrides.
