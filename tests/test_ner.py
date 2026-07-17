"""NerDetector tests via an injectable fake nlp — no spaCy install needed."""

from dataclasses import dataclass
from typing import Any

import pytest

from llm_redact.detection.engine import (
    Allowlist,
    DetectionConfig,
    NerConfig,
    build_detectors,
    detect_all,
)
from llm_redact.detection.ner import NER_PRIORITY, NerDetector
from llm_redact.redactor import _resolve_overlaps

NO_ALLOW = Allowlist(exact=frozenset(), patterns=())


@dataclass
class FakeEnt:
    start_char: int
    end_char: int
    label_: str
    text: str


class FakeDoc:
    def __init__(self, ents: list[FakeEnt]) -> None:
        self.ents = ents


class FakeNlp:
    """Recognizes the literal names 'Jane Doe' (PERSON) and 'Acme' (ORG)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> Any:
        self.calls.append(text)
        ents = []
        for name, label in (("Jane Doe", "PERSON"), ("Acme", "ORG")):
            index = text.find(name)
            if index != -1:
                ents.append(FakeEnt(index, index + len(name), label, name))
        return FakeDoc(ents)


def test_entity_mapping_and_label_filter() -> None:
    detector = NerDetector(FakeNlp(), frozenset({"PERSON"}), max_chars=1000)
    detections = list(detector.detect("send it to Jane Doe at Acme"))
    assert len(detections) == 1
    d = detections[0]
    assert (d.detector_type, d.value, d.priority) == ("PERSON", "Jane Doe", NER_PRIORITY)
    assert "send it to Jane Doe at Acme"[d.start : d.end] == "Jane Doe"


def test_org_entities_when_configured() -> None:
    detector = NerDetector(FakeNlp(), frozenset({"PERSON", "ORG"}), max_chars=1000)
    types = [d.detector_type for d in detector.detect("Jane Doe works at Acme")]
    assert types == ["PERSON", "ORG"]


def test_max_chars_gate_skips_giant_strings() -> None:
    nlp = FakeNlp()
    detector = NerDetector(nlp, frozenset({"PERSON"}), max_chars=10)
    assert list(detector.detect("Jane Doe " + "x" * 100)) == []
    assert nlp.calls == []  # model never invoked


def test_rides_the_standard_pipeline() -> None:
    detectors = build_detectors(DetectionConfig())
    detectors.append(NerDetector(FakeNlp(), frozenset({"PERSON"}), max_chars=1000))
    text = "Jane Doe <jane@corp.example>"
    chosen = _resolve_overlaps(detect_all(detectors, text, NO_ALLOW))
    assert [d.detector_type for d in chosen] == ["PERSON", "EMAIL"]


def test_allowlist_suppresses_ner_detection() -> None:
    detectors: list[Any] = [NerDetector(FakeNlp(), frozenset({"PERSON"}), max_chars=1000)]
    allow = Allowlist(exact=frozenset({"Jane Doe"}), patterns=())
    assert detect_all(detectors, "ask Jane Doe", allow) == []


def test_enabled_without_spacy_config_error() -> None:
    try:
        import spacy  # noqa: F401

        pytest.skip("spaCy installed; missing-dependency path not testable")
    except ImportError:
        pass
    from llm_redact.config import ConfigError

    with pytest.raises(ConfigError, match="uv sync --extra ner"):
        build_detectors(DetectionConfig(ner=NerConfig(enabled=True)))


def test_real_spacy_model_if_available() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not downloaded")
    from llm_redact.detection.ner import build_ner_detector

    detector = build_ner_detector(NerConfig(enabled=True))
    values = [d.value for d in detector.detect("Please email Jane Doe about the invoice.")]
    assert any("Jane" in v for v in values)
