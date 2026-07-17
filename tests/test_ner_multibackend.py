"""Multi-backend NER config + per-type suppression (single source of truth).

Backends are toggled individually via ``[detection.ner] backends``; a
placeholder type disabled at the rule level must not come back through an
NER fold. Fake builders are injected so no extra is needed.
"""

import tomllib
from dataclasses import dataclass

import pytest

from llm_redact.config import ConfigError, parse_config
from llm_redact.config_write import emit_config_toml
from llm_redact.detection.base import Detection
from llm_redact.detection.engine import (
    DetectionConfig,
    NerConfig,
    TypeFilteredDetector,
    build_detectors,
)


@dataclass
class _StubDetector:
    emit: tuple[Detection, ...]

    def detect(self, text: str) -> tuple[Detection, ...]:
        return self.emit


def _detection(detector_type: str) -> Detection:
    return Detection(start=0, end=4, value="xxxx", detector_type=detector_type, priority=120)


# --- NerConfig helpers -------------------------------------------------------


def test_active_backends_legacy_and_list() -> None:
    assert NerConfig().active_backends() == ("spacy",)
    assert NerConfig(backend="gliner").active_backends() == ("gliner",)
    multi = NerConfig(backends=("spacy", "presidio"))
    assert multi.active_backends() == ("spacy", "presidio")


def test_model_for_scoping() -> None:
    # Legacy single `model` applies only when exactly one backend is active
    # — a spaCy pipeline name handed to gliner would be nonsense.
    single = NerConfig(model="en_core_web_md")
    assert single.model_for("spacy") == "en_core_web_md"
    multi = NerConfig(backends=("spacy", "gliner"), model="en_core_web_md")
    assert multi.model_for("spacy") is None
    assert multi.model_for("gliner") is None
    mapped = NerConfig(
        backends=("spacy", "gliner"),
        models=(("gliner", "urchade/gliner_medium-v2.1"), ("spacy", "en_core_web_md")),
    )
    assert mapped.model_for("spacy") == "en_core_web_md"
    assert mapped.model_for("gliner") == "urchade/gliner_medium-v2.1"


# --- config parsing ----------------------------------------------------------


def _parse_ner(**ner: object) -> NerConfig:
    return parse_config({"detection": {"ner": ner}}, "<test>").detection.ner


def test_parse_backends_and_models() -> None:
    ner = _parse_ner(
        enabled=True,
        backends=["spacy", "presidio"],
        models={"presidio": "en_core_web_md"},
    )
    assert ner.backends == ("spacy", "presidio")
    assert ner.models == (("presidio", "en_core_web_md"),)


def test_parse_rejects_bad_backends() -> None:
    with pytest.raises(ConfigError, match="unknown backend"):
        _parse_ner(backends=["spacy", "flair"])  # flair is not a shipped backend
    with pytest.raises(ConfigError, match="must not repeat"):
        _parse_ner(backends=["spacy", "spacy"])
    with pytest.raises(ConfigError, match="must not be empty"):
        _parse_ner(backends=[])
    with pytest.raises(ConfigError, match="models"):
        _parse_ner(models={"flair": "x"})  # unknown backend key in [models]


def test_threshold_requires_confidence_backend() -> None:
    # Legacy single-spacy form keeps rejecting it...
    with pytest.raises(ConfigError, match="score_threshold"):
        _parse_ner(backend="spacy", score_threshold=0.4)
    # ...multi-backend without a confidence backend rejects it too...
    with pytest.raises(ConfigError, match="score_threshold"):
        _parse_ner(backends=["spacy"], score_threshold=0.4)
    # ...and any confidence backend in the list allows it.
    assert _parse_ner(backends=["spacy", "presidio"], score_threshold=0.4).score_threshold == 0.4


def test_emitter_round_trips_backends() -> None:
    config = parse_config(
        {
            "detection": {
                "ner": {
                    "enabled": True,
                    "backends": ["spacy", "gliner"],
                    "models": {"gliner": "urchade/gliner_medium-v2.1"},
                    "score_threshold": 0.35,
                }
            }
        },
        "<test>",
    )
    reparsed = parse_config(tomllib.loads(emit_config_toml(config)), "<reparse>")
    assert reparsed.detection.ner == config.detection.ner


# --- engine: multi-backend build + suppression -------------------------------


def test_type_filtered_detector_drops_suppressed() -> None:
    stub = _StubDetector((_detection("EMAIL"), _detection("PERSON")))
    filtered = TypeFilteredDetector(stub, frozenset({"EMAIL"}))
    assert [d.detector_type for d in filtered.detect("x")] == ["PERSON"]


def test_build_detectors_runs_each_enabled_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_redact.detection.ner as ner_mod
    import llm_redact.detection.presidio_ner as presidio_mod

    built: list[str] = []

    def fake_spacy(config: NerConfig) -> _StubDetector:
        built.append(f"spacy:{config.model}")
        return _StubDetector((_detection("PERSON"),))

    def fake_presidio(config: NerConfig) -> _StubDetector:
        built.append(f"presidio:{config.model}")
        return _StubDetector((_detection("EMAIL"), _detection("PERSON")))

    monkeypatch.setattr(ner_mod, "build_ner_detector", fake_spacy)
    monkeypatch.setattr(presidio_mod, "build_presidio_detector", fake_presidio)

    # email rule disabled at the rule level: presidio's EMAIL fold must be
    # suppressed while PERSON (no built-in rule) flows from both backends.
    config = DetectionConfig(
        enabled=("ipv4",),
        ner=NerConfig(
            enabled=True,
            backends=("spacy", "presidio"),
            models=(("presidio", "en_core_web_md"),),
        ),
    )
    detectors = build_detectors(config)
    assert built == ["spacy:None", "presidio:en_core_web_md"]
    ner_detectors = [d for d in detectors if isinstance(d, TypeFilteredDetector)]
    assert len(ner_detectors) == 2
    emitted = [d.detector_type for det in ner_detectors for d in det.detect("x")]
    assert emitted == ["PERSON", "PERSON"]  # EMAIL suppressed, PERSON kept

    # With the email rule enabled the fold flows through again.
    config_on = DetectionConfig(
        enabled=("email",),
        ner=NerConfig(enabled=True, backends=("presidio",)),
    )
    built.clear()
    detectors_on = build_detectors(config_on)
    ner_on = [d for d in detectors_on if isinstance(d, TypeFilteredDetector)]
    assert len(ner_on) == 1
    assert "EMAIL" in [d.detector_type for d in ner_on[0].detect("x")]
