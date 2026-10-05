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
from ner_fakes import install_hub

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


def test_hf_entity_mapping_and_filter(fold_raw: bool) -> None:
    # The raw entity PER keeps its own type until raw entities fold (2.0.0).
    det = HfDetector(_FakeHfPipe(), frozenset({"PER"}), max_chars=1000, threshold=0.5)
    found = list(det.detect("hi Jane Doe at Acme"))
    expected = "PERSON" if fold_raw else "PER"
    assert found == [
        Detection(start=3, end=11, detector_type=expected, value="Jane Doe", priority=NER_PRIORITY)
    ]


def test_hf_type_request_folds_the_model_label(fold_raw: bool) -> None:
    # The default `entities = ["PERSON"]` requests the PERSON type, which a
    # PER-emitting model (dslim/bert-base-NER) now serves in both modes.
    det = HfDetector(_FakeHfPipe(), frozenset({"PERSON"}), max_chars=1000, threshold=0.5)
    assert [(d.detector_type, d.value) for d in det.detect("hi Jane Doe at Acme")] == [
        ("PERSON", "Jane Doe")
    ]


class _ScriptedHfPipe:
    """Returns exactly the entities it was given (offsets as the pipeline
    reported them), whatever the text."""

    def __init__(self, ents: list[dict[str, object]]) -> None:
        self._ents = ents

    def __call__(self, text: str) -> list[dict[str, object]]:
        return list(self._ents)


def test_hf_value_is_the_source_slice_not_the_decoded_word() -> None:
    # transformers builds `word` with convert_tokens_to_string, which can
    # differ from what the user sent (lowercased, re-spaced, [UNK]); the
    # vault must map the exact sent text or rehydration restores text the
    # user never wrote.
    text = "hi JANE  Doe!"
    pipe = _ScriptedHfPipe(
        [{"entity_group": "PER", "score": 0.99, "word": "jane doe", "start": 3, "end": 12}]
    )
    det = HfDetector(pipe, frozenset({"PER"}), max_chars=1000, threshold=0.5)
    (found,) = det.detect(text)
    assert found.value == text[3:12] == "JANE  Doe"
    assert (found.start, found.end) == (3, 12)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 4), (3, 3), (5, 3), (10, 14), (14, 15), (0, 14)],
    ids=["negative", "empty", "reversed", "past-end", "beyond", "one-past-len"],
)
def test_hf_out_of_range_offsets_are_skipped(start: int, end: int) -> None:
    text = "hi Jane Doe!!"  # 13 characters
    good = {"entity_group": "PER", "score": 0.9, "word": "Jane", "start": 3, "end": 7}
    bad = {"entity_group": "PER", "score": 0.9, "word": "x", "start": start, "end": end}
    det = HfDetector(_ScriptedHfPipe([bad, good]), frozenset({"PER"}), 1000, 0.5)
    assert [(d.start, d.end, d.value) for d in det.detect(text)] == [(3, 7, "Jane")]


def test_hf_span_ending_exactly_at_the_text_end_is_kept() -> None:
    text = "Jane"
    pipe = _ScriptedHfPipe([{"entity_group": "PER", "score": 0.9, "start": 0, "end": 4}])
    det = HfDetector(pipe, frozenset({"PER"}), 1000, 0.5)
    assert [d.value for d in det.detect(text)] == ["Jane"]


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


def _failing_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    def pipeline(*args: object, **kwargs: object) -> object:
        raise OSError("no such model")

    module = types.ModuleType("transformers")
    module.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)
    install_hub(monkeypatch)


def test_hf_load_failure_without_torch_names_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from llm_redact.config import ConfigError
    from llm_redact.detection.hf_ner import build_hf_detector

    _failing_transformers(monkeypatch)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "torch" else real_find_spec(name, *a),
    )
    with pytest.raises(ConfigError, match="torch is not installed; install the hf extra"):
        build_hf_detector(NerConfig(enabled=True, backend="hf"))


def test_hf_load_failure_with_torch_names_the_model_and_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    from llm_redact.config import ConfigError
    from llm_redact.detection.hf_ner import build_hf_detector

    _failing_transformers(monkeypatch)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    with pytest.raises(ConfigError) as caught:
        build_hf_detector(NerConfig(enabled=True, backend="hf", model="org/ner-model"))
    assert str(caught.value) == (
        "failed to load Hugging Face token-classification model 'org/ner-model': OSError"
    )
