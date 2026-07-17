"""Stanza + HF token-classification NER backends via injectable fakes.

Neither extra is installed in the default test env, so the detectors are
exercised with hand-rolled stand-ins matching the sliver of each library's
interface the backend uses; the not-installed paths are pinned separately.
"""

from dataclasses import dataclass

import pytest

from llm_redact.detection.base import Detection
from llm_redact.detection.engine import NerConfig
from llm_redact.detection.hf_ner import HfDetector
from llm_redact.detection.ner import NER_PRIORITY
from llm_redact.detection.stanza_ner import StanzaDetector

# --- Stanza (no confidences) ------------------------------------------------


@dataclass
class _StanzaEnt:
    start_char: int
    end_char: int
    type: str
    text: str


class _StanzaDoc:
    def __init__(self, ents: list[_StanzaEnt]) -> None:
        self.ents = ents


class _FakeStanza:
    def __call__(self, text: str) -> _StanzaDoc:
        ents = []
        for name, kind in (("Jane Doe", "PERSON"), ("Acme", "ORG")):
            i = text.find(name)
            if i != -1:
                ents.append(_StanzaEnt(i, i + len(name), kind, name))
        return _StanzaDoc(ents)


def test_stanza_entity_mapping_and_filter() -> None:
    det = StanzaDetector(_FakeStanza(), frozenset({"PERSON"}), max_chars=1000)
    found = list(det.detect("hi Jane Doe at Acme"))
    assert [(d.detector_type, d.value, d.priority) for d in found] == [
        ("PERSON", "Jane Doe", NER_PRIORITY)
    ]
    both = StanzaDetector(_FakeStanza(), frozenset({"PERSON", "ORG"}), max_chars=1000)
    assert {d.detector_type for d in both.detect("hi Jane Doe at Acme")} == {"PERSON", "ORG"}


def test_stanza_max_chars_gate() -> None:
    det = StanzaDetector(_FakeStanza(), frozenset({"PERSON"}), max_chars=5)
    assert list(det.detect("hi Jane Doe at Acme")) == []


def test_stanza_not_installed_is_config_error() -> None:
    try:
        import stanza  # noqa: F401

        pytest.skip("stanza installed; missing-dependency path not testable")
    except ImportError:
        pass
    from llm_redact.config import ConfigError
    from llm_redact.detection.stanza_ner import build_stanza_detector

    with pytest.raises(ConfigError, match="stanza extra"):
        build_stanza_detector(NerConfig(enabled=True, backend="stanza"))


# --- HF token-classification (confidences → score_threshold) ----------------


class _FakeHfPipe:
    """Aggregation-strategy 'simple' output: entity_group + score + span."""

    def __init__(self, score: float = 0.99) -> None:
        self._score = score

    def __call__(self, text: str) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for word, group in (("Jane Doe", "PER"), ("Acme", "ORG")):
            i = text.find(word)
            if i != -1:
                out.append(
                    {
                        "entity_group": group,
                        "score": self._score,
                        "word": word,
                        "start": i,
                        "end": i + len(word),
                    }
                )
        return out


def test_hf_entity_mapping_and_filter() -> None:
    det = HfDetector(_FakeHfPipe(), frozenset({"PER"}), max_chars=1000, threshold=0.5)
    found = list(det.detect("hi Jane Doe at Acme"))
    assert found == [
        Detection(start=3, end=11, detector_type="PER", value="Jane Doe", priority=NER_PRIORITY)
    ]


def test_hf_score_threshold_filters() -> None:
    low = HfDetector(_FakeHfPipe(score=0.30), frozenset({"PER"}), max_chars=1000, threshold=0.5)
    assert list(low.detect("hi Jane Doe")) == []  # below threshold → dropped
    ok = HfDetector(_FakeHfPipe(score=0.80), frozenset({"PER"}), max_chars=1000, threshold=0.5)
    assert [d.value for d in ok.detect("hi Jane Doe")] == ["Jane Doe"]


def test_hf_max_chars_gate() -> None:
    det = HfDetector(_FakeHfPipe(), frozenset({"PER"}), max_chars=5, threshold=0.5)
    assert list(det.detect("hi Jane Doe")) == []


def test_hf_not_installed_is_config_error() -> None:
    try:
        import transformers  # noqa: F401

        pytest.skip("transformers installed; missing-dependency path not testable")
    except ImportError:
        pass
    from llm_redact.config import ConfigError
    from llm_redact.detection.hf_ner import build_hf_detector

    with pytest.raises(ConfigError, match="hf extra"):
        build_hf_detector(NerConfig(enabled=True, backend="hf"))


# --- config parsing ---------------------------------------------------------


def test_config_accepts_new_backends() -> None:
    from llm_redact.config import parse_config

    for backend in ("stanza", "hf"):
        cfg = parse_config({"detection": {"ner": {"backend": backend}}}, "t")
        assert cfg.detection.ner.backend == backend
    multi = parse_config({"detection": {"ner": {"backends": ["spacy", "stanza", "hf"]}}}, "t")
    assert multi.detection.ner.backends == ("spacy", "stanza", "hf")


def test_score_threshold_requires_a_confidence_backend() -> None:
    from llm_redact.config import ConfigError, parse_config

    # hf emits confidences -> allowed.
    ok = parse_config({"detection": {"ner": {"backend": "hf", "score_threshold": 0.7}}}, "t")
    assert ok.detection.ner.score_threshold == 0.7
    # stanza does not -> rejected, like spacy.
    with pytest.raises(ConfigError, match="score_threshold"):
        parse_config({"detection": {"ner": {"backend": "stanza", "score_threshold": 0.7}}}, "t")


def test_per_backend_model_override_for_new_backends() -> None:
    from llm_redact.config import parse_config

    cfg = parse_config(
        {
            "detection": {
                "ner": {
                    "backends": ["stanza", "hf"],
                    "models": {"hf": "org/multilingual-ner"},
                }
            }
        },
        "t",
    )
    assert cfg.detection.ner.model_for("hf") == "org/multilingual-ner"
    assert cfg.detection.ner.model_for("stanza") is None
