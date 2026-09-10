"""Disruption event classification with an optional transformer backend."""

from __future__ import annotations

import re
from dataclasses import dataclass

EVENT_KEYWORDS = {
    "Port or transport disruption": {
        "port",
        "strike",
        "shipping",
        "congestion",
        "transport",
        "rail",
    },
    "Extreme weather": {"flood", "storm", "cyclone", "hurricane", "earthquake", "weather"},
    "Factory shutdown": {"factory", "plant", "shutdown", "closure", "fire"},
    "Material shortage": {"shortage", "scarcity", "raw material", "semiconductor", "component"},
    "Geopolitical disruption": {"war", "sanction", "border", "political", "conflict", "tariff"},
    "Financial distress": {"bankruptcy", "insolvency", "default", "financial distress"},
    "Quality or recall": {"recall", "defect", "contamination", "quality failure"},
}

TRANSFORMER_LABEL_DESCRIPTIONS = {
    "Port or transport disruption": "a port, shipping, road, rail, or air-freight disruption",
    "Extreme weather": "an extreme-weather or natural-hazard supply disruption",
    "Factory shutdown": "a factory closure or production shutdown",
    "Material shortage": "a shortage of raw materials or components",
    "Geopolitical disruption": "a geopolitical, sanctions, conflict, tariff, or border disruption",
    "Financial distress": "supplier bankruptcy, insolvency, default, or financial distress",
    "Quality or recall": "a product defect, contamination, quality failure, or recall",
    "No disruption": "no active supply-chain disruption or a cancelled or denied threat",
}

SEVERITY_WORDS = {
    "critical": 1.0,
    "major": 0.85,
    "severe": 0.85,
    "emergency": 0.9,
    "closed": 0.8,
    "shutdown": 0.8,
    "delay": 0.55,
    "warning": 0.45,
}


@dataclass(frozen=True)
class Classification:
    event_type: str
    confidence: float
    severity: float
    backend: str


class EventClassifier:
    """Lazy classifier that works offline and can optionally use Hugging Face."""

    def __init__(self, use_transformer: bool = False, model_name: str | None = None):
        self.use_transformer = use_transformer
        self.model_name = model_name or "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
        self._pipeline = None

    def _load_transformer(self):
        if self._pipeline is None:
            try:
                import truststore
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Transformer mode requires: pip install -r requirements-ai.txt"
                ) from exc
            # Use the operating-system certificate store. This is especially helpful on
            # managed Windows machines whose trusted proxy/root certificates are not present
            # in Python's bundled certifi file.
            truststore.inject_into_ssl()
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,
            )
        return self._pipeline

    def classify(self, text: str) -> Classification:
        text = text or ""
        if self.use_transformer:
            return self.classify_many([text], batch_size=1)[0]
        return self._keyword_classify(text)

    def classify_many(self, texts: list[str], batch_size: int = 8) -> list[Classification]:
        """Classify a batch, amortizing transformer inference overhead when enabled."""
        if not self.use_transformer:
            return [self._keyword_classify(text or "") for text in texts]
        model = self._load_transformer()
        description_to_label = {
            description: label for label, description in TRANSFORMER_LABEL_DESCRIPTIONS.items()
        }
        results = model(
            [(text or "")[:4000] for text in texts],
            batch_size=batch_size,
            candidate_labels=list(description_to_label),
            hypothesis_template="This news describes {}.",
            multi_label=False,
        )
        return [
            Classification(
                description_to_label[str(result["labels"][0])],
                float(result["scores"][0]),
                self._severity(text),
                "transformer",
            )
            for text, result in zip(texts, results, strict=True)
        ]

    @staticmethod
    def _severity(text: str) -> float:
        lowered = text.lower()
        scores = [score for word, score in SEVERITY_WORDS.items() if word in lowered]
        duration = re.search(r"(\d+)\s*(day|week|month)s?", lowered)
        duration_score = 0.0
        if duration:
            value = int(duration.group(1))
            multiplier = {"day": 1, "week": 7, "month": 30}[duration.group(2)]
            duration_score = min(value * multiplier / 30, 1.0)
        return round(max(scores + [duration_score, 0.35]), 3)

    def _keyword_classify(self, text: str) -> Classification:
        lowered = text.lower()
        matches: list[tuple[int, str]] = []
        for label, keywords in EVENT_KEYWORDS.items():
            count = sum(
                1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", lowered)
            )
            matches.append((count, label))
        count, label = max(matches)
        if count == 0:
            return Classification("Other disruption", 0.35, self._severity(text), "keyword")
        confidence = min(0.50 + count * 0.12, 0.92)
        return Classification(label, confidence, self._severity(text), "keyword")
