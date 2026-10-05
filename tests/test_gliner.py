"""GlinerDetector tests via an injectable fake model — no gliner install needed."""

from typing import Any

import pytest

from llm_redact.detection.engine import (
    Allowlist,
    DetectionConfig,
    NerConfig,
    build_detectors,
    detect_all,
)
from llm_redact.detection.gliner_ner import GlinerDetector
from llm_redact.detection.ner import NER_PRIORITY

NO_ALLOW = Allowlist(exact=frozenset(), patterns=())


class FakeModel:
    """Recognizes 'Jane Doe' when prompted "person" (score 0.9) and 'Acme'
    when prompted "ORG" (0.3). A PERSON type request sends the
    natural-language prompt "person"; the raw request ORG is sent verbatim."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], float]] = []

    def predict_entities(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, Any]]:
        self.calls.append((text, labels, threshold))
        entities = []
        for name, label, score in (("Jane Doe", "person", 0.9), ("Acme", "ORG", 0.3)):
            index = text.find(name)
            if index != -1 and label in labels and score >= threshold:
                entities.append(
                    {
                        "start": index,
                        "end": index + len(name),
                        "label": label,
                        "text": name,
                        "score": score,
                    }
                )
        return entities


def test_entity_mapping_and_priority() -> None:
    detector = GlinerDetector(FakeModel(), frozenset({"PERSON"}), max_chars=1000, threshold=0.5)
    detections = list(detector.detect("ask Jane Doe about Acme"))
    assert len(detections) == 1
    d = detections[0]
    assert (d.detector_type, d.value, d.priority) == ("PERSON", "Jane Doe", NER_PRIORITY)


def test_threshold_filters_low_scores() -> None:
    detector = GlinerDetector(
        FakeModel(), frozenset({"PERSON", "ORG"}), max_chars=1000, threshold=0.5
    )
    types = [d.detector_type for d in detector.detect("Jane Doe works at Acme")]
    assert types == ["PERSON"]  # ORG scored 0.3 < 0.5

    permissive = GlinerDetector(
        FakeModel(), frozenset({"PERSON", "ORG"}), max_chars=1000, threshold=0.2
    )
    types = [d.detector_type for d in permissive.detect("Jane Doe works at Acme")]
    assert types == ["PERSON", "ORG"]


def test_type_request_sends_its_natural_language_prompt() -> None:
    model = FakeModel()
    detector = GlinerDetector(model, frozenset({"PERSON", "ORG"}), max_chars=1000, threshold=0.5)
    list(detector.detect("Jane Doe"))
    ((_text, labels, threshold),) = model.calls
    assert labels == ["person", "ORG"]  # type request's prompt, then the raw one
    assert threshold == 0.5


def test_max_chars_gate() -> None:
    model = FakeModel()
    detector = GlinerDetector(model, frozenset({"PERSON"}), max_chars=10, threshold=0.5)
    assert list(detector.detect("Jane Doe " + "x" * 100)) == []
    assert model.calls == []


def test_label_case_normalized() -> None:
    class WeirdLabelModel:
        def predict_entities(
            self, text: str, labels: list[str], threshold: float
        ) -> list[dict[str, Any]]:
            return [{"start": 0, "end": 4, "label": "job title", "text": "misc", "score": 0.9}]

    detector = GlinerDetector(
        WeirdLabelModel(), frozenset({"job title"}), max_chars=100, threshold=0.5
    )
    assert next(iter(detector.detect("misc"))).detector_type == "JOB_TITLE"


def test_allowlist_applies() -> None:
    detectors: list[Any] = [
        GlinerDetector(FakeModel(), frozenset({"PERSON"}), max_chars=1000, threshold=0.5)
    ]
    assert [d.value for d in detect_all(detectors, "ask Jane Doe", NO_ALLOW)] == ["Jane Doe"]
    allow = Allowlist(exact=frozenset({"Jane Doe"}), patterns=())
    assert detect_all(detectors, "ask Jane Doe", allow) == []


def test_enabled_without_gliner_config_error() -> None:
    try:
        import gliner  # noqa: F401

        pytest.skip("gliner installed; missing-dependency path not testable")
    except ImportError:
        pass
    from llm_redact.config import ConfigError

    with pytest.raises(ConfigError, match="uv sync --extra gliner"):
        build_detectors(DetectionConfig(ner=NerConfig(enabled=True, backend="gliner")))


def test_config_rejects_threshold_for_spacy(tmp_path: Any) -> None:
    from llm_redact.config import ConfigError, load_config

    config_file = tmp_path / "c.toml"
    config_file.write_text("[detection.ner]\nenabled = false\nscore_threshold = 0.7\n")
    with pytest.raises(ConfigError, match="score_threshold"):
        load_config(config_file)


def test_config_accepts_threshold_for_gliner(tmp_path: Any) -> None:
    from llm_redact.config import load_config

    config_file = tmp_path / "c.toml"
    config_file.write_text(
        '[detection.ner]\nenabled = false\nbackend = "gliner"\nscore_threshold = 0.7\n'
    )
    config = load_config(config_file)
    assert config.detection.ner.backend == "gliner"
    assert config.detection.ner.score_threshold == 0.7
