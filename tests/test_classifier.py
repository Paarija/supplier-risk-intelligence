from supplyshield.transformer_classifier import EventClassifier


def test_keyword_classifier_detects_weather_event():
    result = EventClassifier().classify(
        "A severe flood warning may close factories for seven days."
    )
    assert result.event_type == "Extreme weather"
    assert result.confidence >= 0.5
    assert result.severity >= 0.5
    assert result.backend == "keyword"


def test_keyword_classifier_has_safe_fallback():
    result = EventClassifier().classify("An unusual event was reported.")
    assert result.event_type == "Other disruption"
    assert 0 <= result.confidence <= 1


def test_keyword_classifier_supports_batches():
    results = EventClassifier().classify_many(
        ["A flood closed the port.", "A supplier filed for bankruptcy."]
    )
    assert [result.event_type for result in results] == [
        "Port or transport disruption",
        "Financial distress",
    ]
